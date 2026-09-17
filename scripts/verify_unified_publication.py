"""Verify published arithmetic, fixed question membership and release hashes.

No originals, GPU or network required. This is not a new retrieval run and
does not independently establish correctness of source annotations.
"""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rate(rows,field,fallback=None):
    hits=sum(bool(r.get(field,r.get(fallback))) for r in rows)
    return dict(hits=hits,total=len(rows),rate=hits/len(rows))

def main():
    report=read(ROOT/'evaluation/results/unified_75_v1.json')
    protocol=read(ROOT/'evaluation/results/unified_75_protocol_v1.json')
    labels=read(ROOT/'evaluation/public_eval_75.json')
    questions={r['id']:r['question'] for rows in labels['groups'].values() for r in rows}
    rows=report['details']
    assert len(rows)==len({r['id'] for r in rows})==75
    assert set(questions)=={r['id'] for r in rows}
    assert all(questions[r['id']]==r['question'] and not r.get('error') for r in rows)
    retrieval=[r for r in rows if r.get('category')=='retrieval' or r['id'].startswith('fresh_en_')]
    numeric=[r for r in rows if r.get('category')=='numeric']
    condition=[r for r in rows if r.get('category') in ['context','condition']]
    conflicts=[r for r in rows if r.get('category')=='conflict']
    native=[r for r in conflicts if r['id'].startswith('extra_')]
    authored=[r for r in conflicts if not r['id'].startswith('extra_')]
    metrics=dict(evidence_recall_at_5=rate(retrieval,'hit_at_5','evidence_hit_at_5'),
        mrr=sum(r['reciprocal_rank'] for r in retrieval)/len(retrieval),
        numeric_row_value_unit_retention=rate(numeric,'row_value_unit_retained','packed_gold_covered'),
        condition_text_retention=rate(condition,'condition_relation_retained','protected_text_retained'),
        condition_protection_tagging=rate(condition,'condition_protection_tagged','protected_relation_marked'),
        native_conflict_detection=rate(native,'conflict_detected'),
        native_conflict_both_sources=rate(native,'both_sources_retained'),
        authored_conflict_detection=rate(authored,'conflict_group_detected'),
        authored_conflict_both_sources=rate(authored,'conflict_all_members_retained'))
    assert metrics==report['metrics']
    assert (len(retrieval),len(numeric),len(condition),len(native),len(authored))==(45,10,10,5,5)
    assert protocol['public_code_hash_normalization']=='CRLF_to_LF'
    for relative,digest in protocol['public_code_sha256'].items():
        content=(ROOT/relative.replace('\\','/')).read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha256(content).hexdigest()==digest,relative
    for relative,digest in protocol['question_sha256'].items():
        assert hashlib.sha256((ROOT/relative.replace('\\','/')).read_bytes()).hexdigest()==digest,relative
    hybrid=report['supplements']['hybrid']
    assert rate(hybrid['details'],'actual_hybrid_executed')==hybrid['execution']
    assert rate(hybrid['details'],'hit_at_5')==hybrid['gold_recall_at_5']
    ocr=report['supplements']['ocr']
    assert hashlib.sha256((ROOT/'evaluation/ocr_smoke/synthetic_scanned.pdf').read_bytes()).hexdigest()==ocr['file_sha256']
    assert all(ocr['checks'].values()) and ocr['recalled_ocr']
    negative=report['supplements']['negative_conflict']
    assert sum(r['passed'] for r in negative['details'])==negative['passed']==5
    assert sum(r['false_conflict_detected'] for r in negative['details'])==negative['false_positives']==0
    print(json.dumps(dict(valid=True,unique_questions=75,metrics_recomputed=True,
        frozen_question_bytes_valid=True,published_code_lf_hashes_valid=True,
        supplements_verified=True,model_generation_tested=False)))

if __name__=='__main__': main()
