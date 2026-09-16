import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('public_eval_runner',ROOT/'scripts/run_public_eval.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

class PublicEvalRunnerTests(unittest.TestCase):
    def test_http_success_cannot_hide_budget_error_or_missing_aspects(self):
        error = dict(answerable=False, customer_reply='budget exceeded', citations=[],
            meta=dict(error=dict(code='CONTEXT_BUDGET_EXCEEDED')))
        self.assertFalse(runner.generation_contract_checks(dict(should_refuse=True), error)['runtime_error_absent'])
        partial = dict(answerable=True, customer_reply='capacity discussion', citations=[dict(evidence_id='E')],
            meta=dict(answer_aspect_coverage=dict(requested_aspects=['capacity','approval'],complete=False)))
        self.assertFalse(runner.generation_contract_checks({},partial)['requested_aspect_contract_complete'])

    def test_table_entity_value_period_must_bind_in_one_row(self):
        gold={'file':'a.xlsx','document_id':'doc1','canonical_chunk_id':'table1','anchors':['BeamY','2025','12']}
        split={'document_id':'doc1','original_chunk_id':'table1','text':'[ROW] BeamY 2025 9\n[ROW] Other 2024 12'}
        self.assertFalse(runner.matches(split,gold))
        split['text']='[ROW] BeamY 2025 12'
        self.assertTrue(runner.matches(split,gold))

    def test_other_file_cannot_match_even_if_number_same(self):
        gold={'file':'a.pdf','document_id':'doc1','canonical_chunk_id':'e1','anchors':['78 percent']}
        self.assertFalse(runner.matches({'document_id':'doc2','original_chunk_id':'e1','text':'78 percent'},gold))

    def test_numeric_boundary(self):
        self.assertFalse(runner.value_present('78','178 or 78.5'))
        self.assertTrue(runner.value_present('78','78%'))
        self.assertFalse(runner.value_present('23.9','23.93'))

    def test_sheet_locator_is_provenance_not_a_row_value(self):
        gold=dict(file='a.xlsx', document_id='doc1', canonical_chunk_id='table1',
            anchors=['sheet=Specs', 'BeamY', '12'])
        evidence=dict(document_id='doc1', original_chunk_id='table1',
            text="[TABLE id=T sheet=Specs state=visible]\n[ROW] A2='BeamY' | B2='12'")
        self.assertTrue(runner.matches(evidence,gold))
        evidence['text']="[TABLE id=T sheet=SpecsExtra state=visible]\n[ROW] A2='BeamY' | B2='12'"
        self.assertFalse(runner.matches(evidence,gold))
        evidence['text']="[TABLE id=T sheet=Specs state=visible]\n[ROW] A2='BeamY' | B2='9'\n[ROW] A3='Other' | B3='12'"
        self.assertFalse(runner.matches(evidence,gold))

if __name__=='__main__':unittest.main()
