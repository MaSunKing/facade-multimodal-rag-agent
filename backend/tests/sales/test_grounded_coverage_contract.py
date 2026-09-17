from copy import deepcopy
import unittest

from backend.app import is_safe_grounded_answer


class GroundedCoverageContractTests(unittest.TestCase):
    def value(self):
        return dict(intent='document_qa',normalized_terms=[],answerable=True,
            customer_reply='Thickness is 20 mm.',key_points=[],
            citations=[dict(evidence_id='U1')],missing_information=[],
            risk_warnings=[],next_action='Check versions.',image_observations=[])

    def coverage_value(self):
        value=self.value()
        value['answer_aspect_coverage']=[dict(aspect_index=0,status='answered',
            evidence_ids=['U1'],answer_quote='Thickness is 20 mm.')]
        return value

    def test_legacy_contract_remains_valid(self):
        self.assertTrue(is_safe_grounded_answer(self.value(),{'U1'}))

    def test_prompt_coverage_extension_is_valid(self):
        self.assertTrue(is_safe_grounded_answer(self.coverage_value(),{'U1'}))

    def test_unknown_top_level_extension_is_rejected(self):
        value=self.coverage_value(); value['made_up_field']=True
        self.assertFalse(is_safe_grounded_answer(value,{'U1'}))

    def test_unknown_citation_remains_rejected(self):
        value=self.coverage_value(); value['citations']=[dict(evidence_id='U9')]
        self.assertFalse(is_safe_grounded_answer(value,{'U1'}))

    def test_malformed_coverage_is_rejected(self):
        for changes in [dict(aspect_index=True),dict(aspect_index=-1),dict(status='verified'),
                        dict(status=['answered']),
                        dict(evidence_ids=['U9']),dict(evidence_ids='U1'),dict(answer_quote=20)]:
            value=deepcopy(self.coverage_value()); value['answer_aspect_coverage'][0].update(changes)
            with self.subTest(changes=changes):
                self.assertFalse(is_safe_grounded_answer(value,{'U1'}))

    def test_empty_optional_checklist_is_valid_structure(self):
        value=self.value(); value['answer_aspect_coverage']=[]
        self.assertTrue(is_safe_grounded_answer(value,{'U1'}))


if __name__=='__main__': unittest.main()
