"""Recompute paired repair flags; this does not rerun retrieval or generation."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    report=json.loads((ROOT/'evaluation/results/native_relations_repair_20260917.json').read_text(encoding='utf-8'))
    labels=json.loads((ROOT/'evaluation/public_eval_75.json').read_text(encoding='utf-8'))
    questions={row['id']:row['question'] for rows in labels['groups'].values() for row in rows}
    assert report['model_generation_tested'] is False
    for name,run in report['runs'].items():
        rows=run['details']
        assert len({r['id'] for r in rows})==len(rows)
        assert all(questions[r['id']]==r['question'] and r['error'] is None for r in rows)
        for side in ('before','after'):
            metrics=run[side+'_metrics']
            if name=='fresh_english_30':
                actual=metrics['all']; hits=sum(r[side]['hit_at_5'] for r in rows)
                assert actual['cases']==len(rows)==30 and actual['hits']==hits
                assert abs(actual['recall_at_5']-hits/len(rows))<1e-9
                assert abs(actual['mrr']-sum(r[side]['reciprocal_rank'] for r in rows)/len(rows))<1e-9
                assert actual['packed_support']==sum(r[side]['packed_support'] for r in rows)
            else:
                assert len(rows)==15
                for kind,stats in metrics.items():
                    subset=[r for r in rows if r['category']==kind]
                    assert stats['cases']==len(subset)==5
                    assert stats['passed']==sum(r[side]['passed'] for r in subset)
                    for field,value in stats.items():
                        if isinstance(value,dict):
                            assert value==dict(hits=sum(r[side][field] for r in subset),total=5)
    assert all(hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value
               for p,value in report['published_code_sha256'].items())
    print(json.dumps(dict(paired_runs=3,unique_questions=45,flags_recomputed=True,published_code_sha_matches=True)))


if __name__=='__main__':
    main()
