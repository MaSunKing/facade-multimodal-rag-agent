"""Prepare frozen public source files locally; do not redistribute originals.

Network is opt-in. A changed file never silently replaces a frozen source.
The original corpus snapshot is reconstructed locally, not shipped as text.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--download',action='store_true',help='Fetch official originals from recorded source URLs')
    args=parser.parse_args()
    old=read(ROOT/'evaluation/public_v1/public_eval.json')
    official=[dict(info,file=name,folder='public_v1') for name,info in old['files'].items() if info.get('source_url')]
    official += [dict(info,folder='fresh_english_native_20260917') for info in read(ROOT/'evaluation/fresh_english_native_20260917/download_manifest.json')]
    for record in official:
        path=ROOT/'evaluation'/record['folder']/record['path']
        if not path.exists():
            if not args.download: raise SystemExit(f'Missing {path.name}. Re-run with --download or place the SHA-matching original in its assets folder.')
            request=urllib.request.Request(record['source_url'],headers={'User-Agent':'PublicEvidenceEvaluation/1.0'})
            with urllib.request.urlopen(request,timeout=60) as response: content=response.read(30*1024*1024+1)
            if len(content)>30*1024*1024: raise ValueError('Unexpectedly large source')
            if hashlib.sha256(content).hexdigest()!=record['sha256']:
                raise ValueError(f'Frozen source SHA mismatch: {record["file"]}; no gold or existing file was changed')
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
        if digest(path)!=record['sha256']: raise ValueError(f'Frozen source SHA mismatch: {path.name}')
    for key in ['CUSTOMER_DOCUMENT_OCR_ENABLED','CUSTOMER_ATTACHMENT_SEMANTIC_RERANK','RAG_HYBRID_ENABLED']:os.environ[key]='0'
    from backend.documents.customer_sessions import add_files,get_session,delete_session
    corpus={}
    for name,info in old['files'].items():
        path=ROOT/'evaluation/public_v1'/info['path']
        if digest(path)!=info['sha256']: raise ValueError(f'Fixture SHA mismatch: {name}')
        sid=add_files([(name,path.read_bytes())],owner_id='public-eval-prepare')['session_id']
        try:
            doc=get_session(sid,owner_id='public-eval-prepare').documents[0]
            corpus[name]=dict(document_id=doc.document_id,parser=doc.parser,chunks=doc.chunks,visual_ids=[v.visual_id for v in doc.visuals])
        finally:delete_session(sid,owner_id='public-eval-prepare')
    # Validate canonical identities/anchors against the published fixed gold.
    # Parser-version differences in snapshots are exposed, not written over
    # the original freeze manifest. The collection runner uses this audit.
    compact=lambda s: ''.join(str(s).split()).casefold()
    for sample in old['offline_cases']:
        for gold in sample.get('gold',[]):
            doc=corpus[gold['file']]
            candidates=[c for c in doc['chunks'] if c['chunk_id']==gold['canonical_chunk_id']]
            if doc['document_id']!=gold['document_id'] or not any(all(compact(a) in compact(c['text']) for a in gold['anchors']) for c in candidates):
                raise ValueError(f'Canonical gold binding changed: {sample["id"]}; review parser version before rerunning')
    folder=ROOT/'runtime/public_eval';folder.mkdir(parents=True,exist_ok=True)
    (folder/'canonical_snapshot.json').write_text(json.dumps(corpus,ensure_ascii=False,indent=2),encoding='utf-8')
    audit=dict(official_sources_verified=len(official),native_gold_bindings_valid=True,
               reconstructed_snapshot_sha256=digest(folder/'canonical_snapshot.json'),
               historical_snapshot_sha256=read(ROOT/'evaluation/public_v1/freeze_manifest.json')['canonical_snapshot_sha256'])
    (folder/'preparation_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit))


if __name__=='__main__':main()
