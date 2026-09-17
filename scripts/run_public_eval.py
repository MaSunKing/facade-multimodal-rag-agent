"""CPU retrieval/context regression, optional sequential real local-model HTTP run."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def compact(text): return ''.join(str(text).split()).casefold()
def matches(item,gold):
    basic = (item.get('document_id')==gold['document_id']
            and item.get('original_chunk_id',item.get('chunk_id'))==gold['canonical_chunk_id']
            and all(compact(a) in compact(item.get('text','')) for a in gold['anchors']))
    if not basic: return False
    if Path(gold['file']).suffix.lower() in {'.csv','.xlsx','.xls'}:
        # A number, entity and year somewhere in one big table are NOT enough.
        # A sheet locator belongs to TABLE provenance, not inside a data row.
        # Business anchors must still occur together in ONE native data row.
        business = [a for a in gold['anchors'] if not str(a).startswith('sheet=')]
        for anchor in gold['anchors']:
            if str(anchor).startswith('sheet='):
                sheets = re.findall(r'\[TABLE[^\n]*? sheet=(.*?)(?= state=| range=| parser=|\])', str(item.get('text','')))
                if anchor[6:] not in sheets:
                    return False
        return bool(business) and any('[ROW' in line and all(value_present(a,line) for a in business)
                   for line in item.get('text','').splitlines())
    return True

def value_present(value,text):
    if re.fullmatch(r'\d+(?:\.\d+)?',str(value)):
        return bool(re.search(r'(?<![\d.])'+re.escape(str(value))+r'(?![\d.])',text))
    return compact(value) in compact(text)

def check_freeze(folder,dataset):
    manifest=json.loads((folder/'freeze_manifest.json').read_text(encoding='utf-8'))
    if sha(folder/'public_eval.json')!=manifest['dataset_sha256']: raise ValueError('Frozen dataset changed')
    if sha(folder/'canonical_snapshot.json')!=manifest['canonical_snapshot_sha256']: raise ValueError('Frozen canonical snapshot changed')
    for name,info in dataset['files'].items():
        if sha(folder/info['path'])!=info['sha256']: raise ValueError('Frozen file changed: '+name)
    cases=dataset['offline_cases']+dataset['generation_cases']
    if len({c['id'] for c in cases})!=len(cases): raise ValueError('Duplicate sample IDs')
    generation_only = dataset.get('evaluation_scope') == 'independent_native_generation_smoke_not_replacement_score'
    if generation_only and (dataset['offline_cases'] or len(dataset['generation_cases']) != 6):
        raise ValueError('Independent native smoke requires zero offline and six generation cases')
    if not generation_only and Counter(c['category'] for c in dataset['offline_cases'])!=Counter(retrieval=15,numeric=5,conflict=5,context=5):
        raise ValueError('Unexpected offline quotas')
    corpus=json.loads((folder/'canonical_snapshot.json').read_text(encoding='utf-8'))
    for case in dataset['offline_cases']:
        for gold in case.get('gold',[]):
            doc=corpus[gold['file']]
            chunks=[c for c in doc['chunks'] if c['chunk_id']==gold['canonical_chunk_id']]
            if doc['document_id']!=gold['document_id'] or not chunks:
                raise ValueError('Gold ID missing in canonical snapshot: '+case['id'])
            if not all(compact(a) in compact(chunks[0]['text']) for a in gold['anchors']):
                raise ValueError('Gold anchors missing in canonical snapshot: '+case['id'])
    return manifest

def load_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path,local_files_only=True)

def offline(folder,dataset,output,tokenizer_path, *, prompt_budget=None):
    # Explicit CPU track: no embeddings, OCR, model weights, planner or network.
    os.environ['CUSTOMER_DOCUMENT_OCR_ENABLED']='0'
    os.environ['CUSTOMER_ATTACHMENT_SEMANTIC_RERANK']='0'
    os.environ['RAG_HYBRID_ENABLED']=os.getenv('PUBLIC_EVAL_HYBRID_ENABLED','0')
    from backend.documents.customer_sessions import add_files,bind_session_owner,delete_session,get_session,retrieve
    from backend.sales.context_engine import optimise_evidence_context,validate_packed_evidence
    from backend.app import compact_grounded_payload_for_generation
    from backend.documents.customer_sessions import _estimate_text_tokens
    tokenizer=load_tokenizer(tokenizer_path) if tokenizer_path else None
    sessions={}; results=[]
    for sample in dataset['offline_cases']:
        started=time.perf_counter(); key=tuple(sample['files'])
        row={'id':sample['id'],'category':sample['category'],'question':sample['question']}
        try:
            if sample.get('evidence') is not None:
                evidence=sample['evidence']; input_audit={'track':'authored_context_candidate_fixture'}
            else:
                if key not in sessions:
                    files=[(name,(folder/dataset['files'][name]['path']).read_bytes()) for name in key]
                    sessions[key]=add_files(files,owner_id='public-eval-cpu')['session_id']
                with bind_session_owner('public-eval-cpu'):
                    recalled=retrieve(sessions[key],sample['question'],max_chunks=24,max_text_tokens=5000)
                evidence=recalled['evidence']; input_audit=recalled['input_snapshot']
            ranked,context_audit=optimise_evidence_context(sample['question'],evidence,target_terms=sample.get('target_terms',[]))
            top5=ranked[:5]
            gold=sample.get('gold',[])
            # Multi-source questions need ALL gold support, not one matching
            # window. MRR is the first rank at which support becomes complete.
            hits=[i+1 for i in range(len(ranked)) if gold and all(
                any(matches(item,g) for item in ranked[:i+1]) for g in gold)]
            row.update({'top5':top5,'retrieval_audit':input_audit,'context_audit':context_audit,
                        'gold_evidence_ids':[g['stable_id'] for g in gold],
                        'evidence_hit_at_5':any(i<=5 for i in hits),'reciprocal_rank':1/min(hits) if hits else 0})
            budget=prompt_budget if prompt_budget is not None else sample.get('packed_token_budget',1800)
            if tokenizer:
                payload,audit=compact_grounded_payload_for_generation({'customer_question':sample['question'],'evidence':ranked},
                    tokenizer,max_prompt_tokens=budget,system_prompt='Answer only from evidence.')
                packed=json.loads(payload)['evidence']; row['packing_audit']=audit
            else:
                # Never call this an exact tokenizer/processor or final model-input test.
                packed=[]; total=0
                for item in ranked:
                    cost=_estimate_text_tokens(json.dumps(item,ensure_ascii=False))
                    if total+cost<=budget: packed.append(item); total+=cost
                row['packing_audit']={'mode':'estimated_CPU_only_not_exact_prompt','estimated_tokens':total}
            text='\n'.join(item.get('text','') for item in packed)
            row['packed_evidence']=packed
            # Packing strips some navigation fields, never restore removed text.
            back={item['evidence_id']:item for item in ranked}
            scored=[dict(back.get(item['evidence_id'],{}),**item) for item in packed]
            row['packed_gold_covered']=bool(gold) and all(any(matches(item,g) for item in scored) for g in gold)
            row['input_kind']='authored_evidence_control' if sample.get('evidence') is not None else 'native_file'
            row['expected_values_retained']=all(compact(v) in compact(text) for v in sample.get('expected_values',[]))
            row['protected_text_retained']=all(compact(v) in compact(text) for v in sample.get('protected_text',[]))
            expected=sample.get('expected_relation')
            row['protected_relation_marked']=bool(not expected or any(expected in item.get('protected_relation_types',[]) and any(matches(item,g) for g in gold) for item in packed))
            required=set(sample.get('required_ids',[])); packed_ids={item['evidence_id'] for item in packed}
            row['conflict_all_members_retained']=required.issubset(packed_ids)
            row['conflict_group_detected']=bool(required) and any(required.issubset(set(group.get('evidence_ids',[]))) for group in context_audit.get('conflict_groups',[]))
            row['integrity']=validate_packed_evidence(ranked,packed)
            if sample['category']=='retrieval': row['passed']=row['evidence_hit_at_5']
            elif sample['category']=='numeric': row['passed']=row['packed_gold_covered'] and row['expected_values_retained']
            elif sample['category']=='context': row['passed']=row['protected_text_retained'] and row['protected_relation_marked']
            else: row['passed']=row['conflict_all_members_retained'] and row['conflict_group_detected'] and row['integrity']['valid']
        except Exception as exc:
            row.update({'passed':False,'error':{'type':type(exc).__name__,'message':str(exc)}})
        row['elapsed_seconds']=round(time.perf_counter()-started,3); results.append(row)
        print(json.dumps({'id':row['id'],'passed':row['passed'],'error':row.get('error')},ensure_ascii=False),flush=True)
    for sid in sessions.values(): delete_session(sid,owner_id='public-eval-cpu')
    retrieval=[r for r in results if r['category']=='retrieval']
    metric=lambda category,field: {'numerator':sum(bool(r.get(field)) for r in results if r['category']==category),
                                  'denominator':sum(r['category']==category for r in results)}
    report={'dataset_version':dataset['version'],'dataset_sha256':sha(folder/'public_eval.json'),
        'track':'CPU_attachment_retrieval_and_context_regression_not_full_hybrid_RAG',
        'scoring_version':'native_locator_and_same_row_v2',
        'model_generation_tested':False,'tokenizer_path_used':bool(tokenizer_path),'case_count':len(results),
        'error_count':sum('error' in r for r in results),'passed':sum(r['passed'] for r in results),
        'metrics':{'evidence_recall_at_5':metric('retrieval','evidence_hit_at_5'),
                   'mrr_all_recalled_windows':sum(r.get('reciprocal_rank',0) for r in retrieval)/len(retrieval),
                   'numeric_row_coverage':metric('numeric','packed_gold_covered'),
                   'protected_relation_retention':metric('context','protected_relation_marked'),
                   'protected_text_retention':metric('context','protected_text_retained'),
                   'conflict_group_detection':metric('conflict','conflict_group_detected'),
                   'conflict_member_retention':metric('conflict','conflict_all_members_retained')},
        'details':results,'human_semantic_review':'pending','created_at':datetime.now(timezone.utc).isoformat()}
    (output/'offline_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='details'},ensure_ascii=False))

def generation_contract_checks(case, data):
    """Protocol checks only; HTTP success and semantic correctness differ."""
    answer = str(data.get('customer_reply', ''))
    meta = data.get('meta') or {}
    checks = {
        'runtime_error_absent': not bool(meta.get('error')),
        'expected_value_groups_present': all(any(value_present(v, answer) for v in group) for group in case.get('expected_value_groups', [])),
        'refusal_answerable_flag_correct': not data.get('answerable', True) if case.get('should_refuse') else bool(data.get('answerable', False)),
        'citations_present': bool(data.get('citations')) if not case.get('should_refuse') else True,
    }
    coverage = meta.get('answer_aspect_coverage')
    if coverage and coverage.get('requested_aspects'):
        checks['requested_aspect_contract_complete'] = bool(coverage.get('complete'))
    return checks


def generation(folder,dataset,output,base_url,timeout):
    from urllib.parse import urlparse
    if urlparse(base_url).hostname not in {'localhost','127.0.0.1','::1'}: raise ValueError('Only local inference allowed')
    import httpx
    path=output/'generation_predictions.jsonl'
    previous=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    if any(r['dataset_sha256']!=sha(folder/'public_eval.json') for r in previous):raise ValueError('Resume dataset mismatch')
    done={r['id'] for r in previous}; owner={'x-facade-client-id':'public_eval_local_generation_v1'}
    with httpx.Client(base_url=base_url,timeout=timeout,headers=owner) as client:
        client.get('/health/live').raise_for_status()
        for case in dataset['generation_cases']:
            if case['id'] in done: continue
            started=time.perf_counter(); sid=None
            row={'id':case['id'],'dataset_sha256':sha(folder/'public_eval.json'),'question':case['question'],
                 'category':case['category'],'manual_assertions':case.get('manual_assertions',[]),'human_review_status':'pending'}
            try:
                uploaded=client.post('/api/copilot/documents',files=[('files',(name,(folder/dataset['files'][name]['path']).read_bytes())) for name in case['files']])
                uploaded.raise_for_status(); sid=uploaded.json()['session_id']
                response=client.post('/api/copilot/answer',json={'customer_question':case['question'],'document_session_id':sid,
                    'conversation_context':[],'project_context':{},'use_online_search':False,'memory_enabled':False})
                response.raise_for_status(); data=response.json(); row['response']=data
                row['checks'] = generation_contract_checks(case, data)
                runtime_error = (data.get('meta') or {}).get('error')
                row['status']='failed' if runtime_error else 'completed'
                row['transport_completed']=True
                if runtime_error:
                    row['error']={'type':'StructuredRuntimeError','api_error':runtime_error}
                # Preserve the complete runtime audits. No made-up "citation valid" or visual pass score.
                row['automatic_checks_not_semantic_accuracy']=True
            except Exception as exc:
                error={'type':type(exc).__name__,'message':str(exc)}
                if isinstance(exc,httpx.HTTPStatusError):
                    error['http_status']=exc.response.status_code
                    # Local structured stage/code diagnostics are essential
                    # for distinguishing timeout, parse and model failures.
                    try: error['api_error']=exc.response.json()
                    except ValueError: error['api_error_text']=exc.response.text[:2000]
                row.update(status='failed',error=error)
            finally:
                if sid:
                    try: client.delete('/api/copilot/documents/'+sid).raise_for_status()
                    except Exception: row['session_cleanup_failed']=True
            row['elapsed_seconds']=round(time.perf_counter()-started,3)
            with path.open('a',encoding='utf-8') as handle:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(json.dumps({'id':row['id'],'status':row['status'],'elapsed_seconds':row['elapsed_seconds']},ensure_ascii=False),flush=True)
            previous.append(row)
    summary={'cases':len(previous),'completed':sum(r['status']=='completed' for r in previous),
        'failed':sum(r['status']=='failed' for r in previous),
        'automatic_contract_checks_passed':sum(r['status']=='completed' and bool(r.get('checks')) and all(r['checks'].values()) for r in previous),
        'structured_runtime_errors':sum(bool((r.get('response', {}).get('meta') or {}).get('error')) for r in previous),
        'human_semantic_review':'pending',
        'model_calls':'Use response.meta node/runtime audit; ten cases may invoke planner and recovery as well as generation.',
        'llm_judge_used':False,'details_file':path.name}
    (output/'generation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',type=Path,default=ROOT/'evaluation/public_v1')
    parser.add_argument('--output',type=Path);parser.add_argument('--generation',action='store_true')
    parser.add_argument('--base-url',default='http://127.0.0.1:8000');parser.add_argument('--timeout',type=float,default=150)
    parser.add_argument('--tokenizer-path',type=Path,default=ROOT/'models/Qwen3-VL-8B-Instruct')
    args=parser.parse_args();folder=args.dataset.resolve()
    dataset=json.loads((folder/'public_eval.json').read_text(encoding='utf-8'));check_freeze(folder,dataset)
    output=args.output or ROOT/'outputs/public_eval'/datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True,exist_ok=True)
    if args.generation:generation(folder,dataset,output,args.base_url,args.timeout)
    else:offline(folder,dataset,output,args.tokenizer_path if args.tokenizer_path.exists() else None)

if __name__=='__main__':main()
