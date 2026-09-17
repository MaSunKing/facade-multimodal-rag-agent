"""Frozen, common-condition rerun. Historical reports and gold stay unchanged.

Attachment regression, enterprise hybrid execution and OCR are separate tracks.
This does not evaluate Planner, generated answers, or production accuracy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def rate(rows,field):
    hits=sum(bool(r.get(field)) for r in rows)
    return dict(hits=hits,total=len(rows),rate=hits/len(rows) if rows else None)

def offline(out,config,tokenizer_path):
    env=os.environ.copy()
    env.update(config['environment'])
    env['PUBLIC_EVAL_TOKENIZER_PATH']=str(tokenizer_path)
    # Official originals and whole canonical snapshots are intentionally not
    # distributed. Rebuild locally, verify fixed native gold, never rewrite
    # the historical freeze manifest to make a new parser look identical.
    subprocess.run([sys.executable,str(ROOT/'scripts/prepare_public_eval.py')],cwd=ROOT,env=env,check=True)
    jobs=[('regression_30',
        "import json; from pathlib import Path; import run_public_eval as m; "
        "f=m.ROOT/'evaluation/public_v1'; d=json.loads((f/'public_eval.json').read_text(encoding='utf-8')); "
        "a=json.loads((m.ROOT/'runtime/public_eval/preparation_audit.json').read_text(encoding='utf-8')); "
        "manifest=json.loads((f/'freeze_manifest.json').read_text(encoding='utf-8')); "
        "assert a['native_gold_bindings_valid']; assert m.sha(f/'public_eval.json')==manifest['dataset_sha256']; "
        "assert m.sha(m.ROOT/'runtime/public_eval/canonical_snapshot.json')==a['reconstructed_snapshot_sha256']; "
        "assert all(m.sha(f/i['path'])==i['sha256'] for i in d['files'].values()); "
        "o=Path(__import__('sys').argv[1]); o.mkdir(parents=True,exist_ok=True); "
        "m.offline(f,d,o,Path(__import__('os').environ['PUBLIC_EVAL_TOKENIZER_PATH']),prompt_budget=5000)",[]),
        ('fresh_english_30',"import run_fresh_english_native_eval as m; from pathlib import Path; import sys; m.OUTPUT=Path(sys.argv[1]); m.main()",[]),
        ('extra_15',"import run_basic_capabilities_extra_eval as m; import sys; sys.argv=['eval','--prompt-budget','5000','--output-dir',sys.argv[1]]; m.main()",[])]
    for name,code,extra in jobs:
        destination=out/name
        destination.mkdir(parents=True,exist_ok=True)
        # Same interpreter and config, isolated model/global state in every run.
        with (destination/'run.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-c',"import sys; sys.path.insert(0,'scripts'); "+code,str(destination),*extra],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'{name} failed: see {destination}/run.log')
        print(f'{name}: completed',flush=True)
    old=read(out/'regression_30/offline_results.json')['details']
    fresh=read(out/'fresh_english_30/results.json')['details']
    extra=read(out/'extra_15/results.json')['details']
    retrieval=[dict(r,hit_at_5=r.get('evidence_hit_at_5')) for r in old if r['category']=='retrieval']+fresh
    numeric=[dict(r,row_value_unit_retained=r.get('packed_gold_covered')) for r in old if r['category']=='numeric']+[r for r in extra if r['category']=='numeric']
    condition=[dict(r,condition_relation_retained=r.get('protected_text_retained'),condition_protection_tagged=r.get('protected_relation_marked')) for r in old if r['category']=='context']+[r for r in extra if r['category']=='condition']
    native_conflict=[r for r in extra if r['category']=='conflict']
    controls=[dict(r,conflict_detected=r.get('conflict_group_detected'),both_sources_retained=r.get('conflict_all_members_retained')) for r in old if r['category']=='conflict']
    rows=old+fresh+extra
    ids=[r['id'] for r in rows]
    if len(ids)!=75 or len(set(ids))!=75: raise ValueError('Expected exactly 75 unchanged unique questions')
    if len(retrieval)!=45 or len(numeric)!=10 or len(condition)!=10: raise ValueError('Unexpected metric denominators')
    summary=dict(version='common_conditions_75_v1',unique_questions=75,native_file_cases=70,authored_evidence_controls=5,
        config_sha256=sha(out/'protocol.json'),errors=[r['id'] for r in rows if r.get('error')],
        metrics=dict(evidence_recall_at_5=rate(retrieval,'hit_at_5'),
            mrr=sum(r.get('reciprocal_rank',0) for r in retrieval)/45,
            numeric_row_value_unit_retention=rate(numeric,'row_value_unit_retained'),
            condition_text_retention=rate(condition,'condition_relation_retained'),
            condition_protection_tagging=rate(condition,'condition_protection_tagged'),
            native_conflict_detection=rate(native_conflict,'conflict_detected'),
            native_conflict_both_sources=rate(native_conflict,'both_sources_retained'),
            authored_conflict_detection=rate(controls,'conflict_detected'),
            authored_conflict_both_sources=rate(controls,'both_sources_retained')),
        definitions=dict(retrieval='Complete source-grounded support set in Top5, not merely a matching number.',
            numeric='Entity, target value and explicit native unit are bound to the source row/header, not global substring matches.',
            condition='All labelled limiting/negative text retained; protection annotation is reported separately.',
            authored_controls='Original five authored Evidence cases bypass parsing/retrieval and are NOT native-file success.'),
        model_generation_tested=False,attachment_semantic_reranker_tested=False,
        hybrid_metrics_included=False,ocr_metrics_included=False,
        details=[{k:v for k,v in r.items() if k not in ['ranked_evidence','packed_evidence','top5','retrieval_audit','context_audit','packing_audit','integrity','matching_conflict_groups']} for r in rows])
    save(out/'results.json',summary)
    print(json.dumps(summary['metrics']),flush=True)

def hybrid(out):
    # Gold is frozen BEFORE any model prediction. Cases are regression probes,
    # not held-out labels and not included in the attachment denominator.
    cases=[
        ('真岩石的饰面有哪些效果和颜色？','txt_a8001516b8eea7ea',['荔枝面','黄金麻']),
        ('真岩石饰面原材料和厚度是多少？','txt_9fb7ef064b6f43f7',['3毫米','无机胶凝材料']),
        ('真岩石板损坏之后能如何修补？','txt_5209083b6f19c572',['修补料','更换装饰板']),
        ('真岩石可以采用哪些安装体系和保温配置？','txt_b5b9d82f8c3e71cb',['粘锚','岩棉']),
        ('真岩石保温装饰一体板与饰面装饰板是什么关系？','txt_05800c4b62a30f80',['保温装饰一体板','交付']),
    ]
    from backend.sales.retriever import LocalRagRetriever, tokenize
    from backend.sales.dense_retrieval import retrieval_runtime_status
    import time
    r=LocalRagRetriever()
    docs={d['id']:d for d in r.documents}
    for question,eid,anchors in cases:
        if eid not in docs or not all(a in docs[eid].get('text','') for a in anchors): raise ValueError('Hybrid gold absent or changed')
    save(out/'hybrid_cases.json',dict(cases=cases,index_validation=r._hybrid_validation))
    if not r._hybrid_validation['ready']: raise ValueError('Hybrid index not ready')
    results=[]
    for question,eid,anchors in cases:
        start=time.perf_counter()
        ranked=r._hybrid_scored(question,tokenize(question),node_atlas_request=False,
            standard_request=False,procedure_request=False,product_overview_request=False)
        modes={d.get('_retrieval_score_mode','lexical') for _,d in ranked}
        top=[d for _,d in ranked[:5]]
        results.append(dict(question=question,gold_id=eid,hit_at_5=any(d['id']==eid for d in top),
            actual_hybrid_executed='hybrid' in modes,top_ids=[d['id'] for d in top],
            channels=[d.get('_retrieval_channels',[]) for d in top],
            skip_reasons=sorted({d['_hybrid_skip_reason'] for _,d in ranked if d.get('_hybrid_skip_reason')}),
            elapsed_seconds=round(time.perf_counter()-start,3)))
        print(json.dumps(results[-1],ensure_ascii=True),flush=True)
    report=dict(track='enterprise_hybrid_execution_smoke_not_held_out_accuracy',
        index_validation=r._hybrid_validation,runtime=retrieval_runtime_status(),
        execution=rate(results,'actual_hybrid_executed'),gold_recall_at_5=rate(results,'hit_at_5'),details=results)
    save(out/'hybrid_results.json',report)
    if not all(r['actual_hybrid_executed'] for r in results): raise RuntimeError('Hybrid silently fell back; inspect hybrid_results.json')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--track',choices=['offline','hybrid','all'],default='all')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/public_eval/unified_20260917_v1')
    parser.add_argument('--tokenizer-path',type=Path,default=Path(os.getenv('PUBLIC_EVAL_TOKENIZER_PATH',str(ROOT/'models/Qwen3-VL-8B-Instruct'))))
    args=parser.parse_args(); out=args.output.resolve()
    tokenizer_path=args.tokenizer_path.resolve()
    if args.track in ['offline','all'] and not tokenizer_path.is_dir():
        raise ValueError('Local tokenizer required; pass --tokenizer-path. No model weights are loaded.')
    if (out/'protocol.json').exists(): raise ValueError('Immutable output already exists; use a NEW --output')
    config=dict(version='unified_protocol_v1',created_at=datetime.now(timezone.utc).isoformat(),
        prompt_token_budget=5000,retrieval_token_budget=5000,max_chunks=24,
        tokenizer='Qwen3-VL-8B-Instruct',system_prompt='Answer only from evidence.',
        environment=dict(PUBLIC_EVAL_HYBRID_ENABLED='1',PUBLIC_EVAL_PROMPT_BUDGET='5000',RAG_HYBRID_ENABLED='1',RAG_RETRIEVAL_DEVICE='cuda',
            CUSTOMER_ATTACHMENT_TEXT_WINDOW_TOKENS='768',CUSTOMER_ATTACHMENT_TEXT_WINDOW_OVERLAP_TOKENS='96'),
        question_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'evaluation/public_v1/public_eval.json',ROOT/'evaluation/fresh_english_native_20260917/english_questions.json',ROOT/'evaluation/basic_capabilities_extra_20260917/questions.json']},
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/run_public_eval.py',ROOT/'scripts/run_fresh_english_native_eval.py',ROOT/'scripts/run_basic_capabilities_extra_eval.py',ROOT/'backend/documents/customer_sessions.py',ROOT/'backend/sales/context_engine.py',ROOT/'backend/sales/retriever.py',ROOT/'backend/app.py']},
        exclusions=['No 8B generation or Planner','No variant questions','No OCR claim from native-text inputs','Hybrid GPU protection retained'])
    save(out/'protocol.json',config)
    os.environ.update(config['environment'])
    if args.track in ['offline','all']: offline(out,config,tokenizer_path)
    if args.track in ['hybrid','all']: hybrid(out)

if __name__=='__main__': main()
