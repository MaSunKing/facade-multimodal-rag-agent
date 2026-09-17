"""Native fictional sources: false-conflict controls, fixed before retrieval."""
from pathlib import Path
import hashlib
import json
import os
import sys
import argparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'outputs/public_eval/unified_20260917_v4/negative_conflict')
    args=p.parse_args(); out=args.output
    if (out/'cases.json').exists(): raise ValueError('Use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    cases=[
        dict(id='negative_conflict_01',question='Are the two Sable-X thickness records consistent?',
             texts=['Sable-X thickness: 20 mm','Sable-X thickness: 20 mm'],reason='equal_values'),
        dict(id='negative_conflict_02',question='Are the two Cobalt-Y thickness records consistent after unit conversion?',
             texts=['Cobalt-Y thickness: 20 mm','Cobalt-Y thickness: 2 cm'],reason='equivalent_units'),
        dict(id='negative_conflict_03',question='Do the thickness measurements describe the same product?',
             texts=['Juniper-A thickness: 18 mm','Juniper-B thickness: 20 mm'],reason='different_entities'),
        dict(id='negative_conflict_04',question='Are these Marigold-Z thickness and length values a contradiction?',
             texts=['Marigold-Z thickness: 18 mm','Marigold-Z length: 20 mm'],reason='different_metrics'),
        dict(id='negative_conflict_05',question='Can these two thickness figures be reliably attributed to the same named product?',
             texts=['thickness: 18 mm','thickness: 20 mm'],reason='unbound_entity_not_eligible'),
    ]
    (out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n',encoding='utf-8')
    os.environ.update(CUSTOMER_DOCUMENT_OCR_ENABLED='0',CUSTOMER_ATTACHMENT_SEMANTIC_RERANK='0',RAG_HYBRID_ENABLED='1',
        CUSTOMER_ATTACHMENT_TEXT_WINDOW_TOKENS='768',CUSTOMER_ATTACHMENT_TEXT_WINDOW_OVERLAP_TOKENS='96')
    from backend.documents.customer_sessions import add_files,bind_session_owner,retrieve,delete_session
    from backend.sales.context_engine import optimise_evidence_context
    from backend.app import compact_grounded_payload_for_generation
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(os.getenv('PUBLIC_EVAL_TOKENIZER_PATH',str(ROOT/'models/Qwen3-VL-8B-Instruct')),local_files_only=True)
    rows=[]; owner='negative-conflict-smoke'
    for c in cases:
        # Source identity differs even for equal-valued records.
        files=[(f'{c["id"]}_{j}.txt',('FICTIONAL TEST ONLY.\n'+text+'\n').encode()) for j,text in enumerate(c['texts'])]
        sid=add_files(files,owner_id=owner)['session_id']
        try:
            with bind_session_owner(owner): evidence=retrieve(sid,c['question'],max_chunks=24,max_text_tokens=5000)['evidence']
            ranked,audit=optimise_evidence_context(c['question'],evidence)
            payload,packing=compact_grounded_payload_for_generation(dict(customer_question=c['question'],evidence=ranked),tokenizer,
                max_prompt_tokens=5000,system_prompt='Answer only from evidence.')
            packed=json.loads(payload)['evidence']
            # A clean flag with missing input is NOT a valid negative test.
            text='\n'.join(e['text'] for e in packed)
            both=all(t in text for t in c['texts'])
            rows.append(dict(id=c['id'],reason=c['reason'],both_source_records_retained=both,
                false_conflict_detected=bool(audit.get('conflict_groups')),
                passed=both and not audit.get('conflict_groups'),packing_audit=packing))
        finally: delete_session(sid,owner_id=owner)
    report=dict(track='native_synthetic_false_conflict_controls_not_generation',cases=5,
        model_generation_tested=False,gold_provenance='project_authored',
        prompt_budget=5000,retrieval_budget=5000,cases_sha256=hashlib.sha256((out/'cases.json').read_bytes()).hexdigest(),
        passed=sum(r['passed'] for r in rows),false_positives=sum(r['false_conflict_detected'] for r in rows),details=rows)
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='details'}),flush=True)

if __name__=='__main__': main()
