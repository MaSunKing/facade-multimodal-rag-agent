"""Independently recompute published metrics from all 75 bounded case rows."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    report=json.loads((ROOT/'evaluation/results/results.json').read_text(encoding='utf-8'))
    labels=json.loads((ROOT/'evaluation/public_eval_75.json').read_text(encoding='utf-8'))
    rows=report['details']; questions=[q for group in labels['groups'].values() for q in group]
    assert len(rows)==len(questions)==75
    assert len({q['id'] for q in questions})==75
    assert {r['id'] for r in rows}=={q['id'] for q in questions}
    q_by_id={q['id']:q for q in questions}
    assert all(r['question']==q_by_id[r['id']]['question'] for r in rows)
    assert Counter(r['category'] for r in rows)==Counter(retrieval=45,numeric=10,condition=10,conflict=10)
    selected=[r for r in rows if r['category']=='retrieval']
    assert sum(r.get('hit_at_5',r.get('evidence_hit_at_5',False)) for r in selected)==38
    assert abs(sum(r['reciprocal_rank'] for r in selected)/45-report['metrics']['mrr_all_recalled_windows'])<1e-12
    field_rules={
        'target_row_value_retention':('numeric','packed_gold_covered','target_row_value_retained'),
        'condition_text_relation_retention':('condition','protected_text_retained','condition_relation_retained'),
        'conflict_group_detection':('conflict','conflict_group_detected','conflict_detected'),
        'conflict_member_retention':('conflict','conflict_all_members_retained','both_sources_retained')}
    for metric,(category,old,new) in field_rules.items():
        subset=[r for r in rows if r['category']==category]
        numerator=sum(r.get(new,r.get(old,False)) for r in subset)
        published=report['metrics'][metric]
        assert numerator==published['numerator'] and len(subset)==published['denominator']
        assert abs(numerator/len(subset)-published['rate'])<1e-12
    assert report['metrics']['strict_native_numeric_unit_retention']['numerator']==sum(r.get('row_value_unit_retained',False) for r in rows if r['category']=='numeric' and r['run_id']=='basic_extra_15')
    assert not report['model_generation_tested'] and report['overall_answer_accuracy'] is None
    print(json.dumps(dict(cases=75,ids_and_questions_valid=True,metrics_recomputed=True,model_accuracy_claim=False)))


if __name__=='__main__':main()
