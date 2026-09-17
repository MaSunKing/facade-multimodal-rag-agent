"""Independent arithmetic, complete support, freeze and budget checks."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'outputs/public_eval/unified_20260917_v4')
    args=p.parse_args(); out=args.output
    protocol=read(out/'protocol.json'); result=read(out/'results.json')
    for path,digest in {**protocol['code_sha256'],**protocol['question_sha256']}.items():
        if sha(ROOT/path)!=digest: raise ValueError('Frozen source changed: '+path)
    assert result['config_sha256']==sha(out/'protocol.json')
    old=read(out/'regression_30/offline_results.json')['details']
    fresh=read(out/'fresh_english_30/results.json')['details']
    extra=read(out/'extra_15/results.json')['details']
    rows=old+fresh+extra
    assert len(rows)==len({r['id'] for r in rows})==75
    assert not any(r.get('error') for r in rows)
    assert all(r['packing_audit']['max_prompt_tokens']==5000 and r['packing_audit']['budget_satisfied'] for r in rows)
    assert all(r['packing_audit']['actual_prompt_tokens']<=5000 for r in rows)
    retrieval=[r for r in old if r['category']=='retrieval']
    hits=sum(r['evidence_hit_at_5'] for r in retrieval)+sum(r['hit_at_5'] for r in fresh)
    rr=sum(r['reciprocal_rank'] for r in retrieval+fresh)/45
    assert result['metrics']['evidence_recall_at_5']['hits']==hits
    assert result['metrics']['evidence_recall_at_5']['total']==45
    assert abs(result['metrics']['mrr']-rr)<1e-10
    # The old set has multi-gold questions. Recompute the full Top5 support
    # directly from retained source locators, independent of the stored flag.
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_public_eval import matches
    labels={c['id']:c for c in read(ROOT/'evaluation/public_v1/public_eval.json')['offline_cases']}
    for r in retrieval:
        complete=all(any(matches(item,g) for item in r['top5']) for g in labels[r['id']]['gold'])
        assert complete==r['evidence_hit_at_5'],r['id']
    report=dict(valid=True,unique_questions=75,budget_compliance='75/75',
        errors=0,complete_support_scoring_verified=True,question_and_code_hashes_verified=True,
        recall_at_5=dict(hits=hits,total=45),mrr=rr,
        failed_retrieval_ids=[r['id'] for r in retrieval if not r['evidence_hit_at_5']]+[r['id'] for r in fresh if not r['hit_at_5']])
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__': main()
