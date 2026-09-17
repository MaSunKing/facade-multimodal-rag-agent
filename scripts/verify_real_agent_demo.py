"""Verify fixed demo records and asset hashes; no inference or semantic judge."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEMO=ROOT/'examples/real_agent_demo'

def load(stage):
    folder=DEMO/stage
    report=json.loads((folder/'execution_records.json').read_text(encoding='utf-8'))
    assert len(report['cases'])==len({c['id'] for c in report['cases']})==3
    assert report['model_generation_tested'] and not report['web_search_used'] and not report['llm_judge_used']
    for name,digest in report['source_sha256'].items():
        assert hashlib.sha256((folder/'assets'/name).read_bytes()).hexdigest()==digest,name
    def no_private_fields(value):
        if isinstance(value,dict):
            assert not set(value).intersection({'session_id','request_id','ticket','authorization','storage_ref','model_path','owner_id','url'})
            for v in value.values(): no_private_fields(v)
        elif isinstance(value,list):
            for v in value: no_private_fields(v)
    no_private_fields(report)
    return report

def main():
    before=load('before_contract_fix'); after=load('after_contract_fix')
    assert before['source_sha256']==after['source_sha256']
    old={c['id']:c for c in before['cases']}; new={c['id']:c for c in after['cases']}
    assert set(old)==set(new)
    for ident,c in new.items():
        assert old[ident]['question']==c['question'] and old[ident]['files']==c['files']
        assert old[ident]['audits']['error']['code']=='GROUNDING_FAILED'
        assert not c['audits'].get('error') and c['model_used'] and c['answer_status']==200
        assert c['generation_input_audit']['budget_satisfied']
        allowed=set(c['generation_input_audit']['kept_evidence_ids'])
        allowed.update(s['evidence_id'] for i in c['visual_input_manifest'] for s in i['source_candidates'])
        assert all(citation['evidence_id'] in allowed for citation in c['citations'])
    visual=new['visual_and_text']
    assert len(visual['visual_input_manifest'])==1
    assert {s['document_name'] for s in visual['visual_input_manifest'][0]['source_candidates']}=={'sample_checklist.docx','sample_drawing.png'}
    # Preserve the limitation; do not turn missing visual citations into a pass.
    assert visual['audits']['visual_input_coverage_audit']['complete'] is False
    assert len(new['joint_sources']['selected_tools'])==2
    print(json.dumps(dict(valid=True,unique_demo_questions=3,paired_source_bytes_identical=True,
        citation_ids_in_input=True,visual_manifest_verified=True,known_visual_citation_gap_preserved=True,
        semantic_accuracy_scored=False,model_invoked_by_this_verifier=False)))

if __name__=='__main__': main()
