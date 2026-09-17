"""Real cached Paddle CPU OCR on an existing, visibly inspected scan fixture.

Not a visual-answer or OCR accuracy benchmark: one easy page, three checks.
Never silently download weights or promote OCR candidates to official facts.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import argparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--file',type=Path,default=ROOT/'evaluation/ocr_smoke/synthetic_scanned.pdf')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/public_eval/unified_20260917_v1/ocr_results.json')
    args=p.parse_args()
    if args.output.exists(): raise ValueError('Do not overwrite a completed OCR report')
    content=args.file.read_bytes()
    import fitz
    with fitz.open(stream=content,filetype='pdf') as pdf:
        text_layer=pdf[0].get_text().strip()
    if text_layer: raise ValueError('Fixture is not image-only')
    gold=dict(entity='CARD-DELTA',thickness_value='17',unit='mm',
        provenance='project_authored_synthetic_fixture_visually_verified_before_prediction',
        color_not_scored='Colour requires visual input and is not an OCR-text label')
    start=time.perf_counter()
    os.environ['CUSTOMER_DOCUMENT_OCR_ENABLED']='1'
    os.environ['CUSTOMER_ATTACHMENT_SEMANTIC_RERANK']='0'
    from backend.documents.customer_sessions import add_files,get_session,bind_session_owner,retrieve,delete_session
    report=dict(track='one_page_native_scan_CPU_OCR_execution_smoke',gold=gold,file_sha256=hashlib.sha256(content).hexdigest(),
        text_layer_empty=True,model_generation_tested=False)
    sid=None
    try:
        owner='public-ocr-smoke'
        sid=add_files([(args.file.name,content)],owner_id=owner)['session_id']
        doc=get_session(sid,owner_id=owner).documents[0]
        with bind_session_owner(owner): recalled=retrieve(sid,'What is the thickness of CARD-DELTA in mm?',max_chunks=24,max_text_tokens=5000)
        text='\n'.join(c['text'] for c in doc.chunks)
        # OCR candidates are stored on visual metadata, not canonical facts.
        ocr_candidates=[v.metadata['ocr_literal_text'] for v in doc.visuals if v.metadata.get('ocr_literal_text')]
        ocr_text='\n'.join(ocr_candidates)
        report.update(ocr_candidate_count=len(ocr_candidates),
            checks=dict(entity=gold['entity'].casefold() in ocr_text.casefold(),
                thickness_and_unit='17 mm' in ocr_text.casefold(),
                original_visual_retained=any(v.image_bytes for v in doc.visuals)),
            recalled_ocr=any('[OCR_CANDIDATE' in e['text'] for e in recalled['evidence']),
            ocr_candidate_text=ocr_text,coverage=doc.visual_coverage,stage_timings_ms=doc.stage_timings_ms)
    except Exception as exc:
        report['error']=dict(type=type(exc).__name__,message=str(exc))
    finally:
        if sid: delete_session(sid,owner_id='public-ocr-smoke')
    report['elapsed_seconds']=round(time.perf_counter()-start,3)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True),flush=True)
    if report.get('error') or not all(report.get('checks',{}).values()): raise SystemExit(1)

if __name__=='__main__': main()
