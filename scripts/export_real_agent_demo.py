"""Bound reviewed demo records for publication; never export raw responses."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
def pick(value, names):
    return {k:value[k] for k in names if k in value} if isinstance(value,dict) else {}
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fixtures-dir',type=Path,default=ROOT/'runtime/public_smoke/documents')
    p.add_argument('--reviewed-public-content',action='store_true',required=True)
    args=p.parse_args()
    raw=json.loads((args.input/'raw_report.json').read_text(encoding='utf-8'))
    source=json.loads((args.input/'protocol.json').read_text(encoding='utf-8'))
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    records=[]
    for case in raw['cases']:
        answer=case.get('response',{}); meta=answer.get('meta',{})
        graph=meta.get('orchestration',{}); gen=meta.get('generation_input_audit',{})
        record=pick(case,['id','question','files','upload_status','answer_status','latency_seconds','exception_type','session_deleted'])
        record.update(pick(answer,['answerable','customer_reply','key_points','missing_information','risk_warnings','next_action','image_observations']))
        record['model_used']=meta.get('model_used',False)
        record['selected_tools']=graph.get('tools',[])
        record['execution']=pick(graph,['engine','workflow','node_trace','plan_reason','planner_latency_ms','planner_model_load_ms','planner_generation_ms','planning_rounds','tool_rounds','recovery_actions','gpu_execution_policy'])
        record['generation_input_audit']=pick(gen,['max_prompt_tokens','actual_prompt_tokens','kept_evidence_ids','removed_evidence_ids','budget_satisfied','canonical_evidence_preserved','generated_tokens','generation_seconds','max_new_tokens','hit_output_limit','oom_retry_used','analysis_refusal_retry_used','aspect_coverage_repair_used','visual_runtime_timings','visual_input_selection'])
        record['visual_input_manifest']=gen.get('visual_input_manifest',[])
        record['citations']=[pick(c,['evidence_id','document_name','source_page','section_heading','source_type','sheet_name','source_range','row_index','bounding_box','parser']) for c in answer.get('citations',[])]
        record['retrieval_evidence']=[{**pick(r,['result_id','document_name','source_page','section_heading']),'excerpt':str(r.get('excerpt',''))[:160]} for r in answer.get('retrieval',{}).get('supporting_results',[])]
        record['excerpt_note']='Publication excerpts are capped at 160 characters; they are NOT the complete model input.'
        record['audits']=pick(meta,['answer_document_coverage','answer_aspect_coverage','evidence_support_audit','visual_input_coverage_audit','normative_claim_audit','error'])
        if case.get('answer_status')!=200:
            record['http_error']=pick(answer,['detail'])
        records.append(record)
    assets=out/'assets'; assets.mkdir(exist_ok=True)
    for name,digest in raw['source_sha256'].items():
        original=args.fixtures_dir/name
        assert hashlib.sha256(original.read_bytes()).hexdigest()==digest,name
        shutil.copyfile(original,assets/name)
    health=raw.get('health_after',{})
    report=dict(provenance='actual_local_http_run_with_project_authored_synthetic_attachments',
        model_generation_tested=any(r['model_used'] for r in records),
        evaluation_claim='Three qualitative demos, not an accuracy benchmark or unseen test.',
        web_search_used=False,llm_judge_used=False,
        service_runtime=pick(health,['generation_mode','model_loaded','planner_max_new_tokens']),
        source_sha256=raw['source_sha256'],execution_provenance=source.get('execution_provenance',{}),
        code_fingerprints=source.get('code_fingerprints',{}),
        publication_excludes=['Owner/session/request identifiers','Signed asset URLs and tickets','Local paths','Full business index and raw responses'],cases=records)
    (out/'execution_records.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'published_case_count':len(records),'model_used':[r['model_used'] for r in records]}))
if __name__=='__main__': main()
