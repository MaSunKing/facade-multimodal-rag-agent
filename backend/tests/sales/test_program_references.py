"""CPU protocol tests. References are Context provenance, not fact approval."""
import json
import unittest

from backend.sales.output_contract import repair_and_validate, OutputContractError
from backend.sales.answer_review import revision_schema, REVISION_PROMPT
from backend.sales.draft_prompt import COMMON


class ProgramReferencesTests(unittest.TestCase):
    def validate(self, value, ids=('T1',)):
        return repair_and_validate(value, set(ids), {'unknown'}, program_references=True)

    def test_missing_or_empty_model_citations_do_not_retry(self):
        for extra in ({}, {'citations': []}):
            result = self.validate(dict(answerable=True, customer_reply='已支持部分。', **extra))
            self.assertEqual(result['citations'], [{'evidence_id': 'T1'}])

    def test_model_cannot_add_source_or_choose_private_candidate(self):
        result = self.validate({'answerable': True, 'customer_reply': '部分回答。',
            'citations': [{'evidence_id': 'PRIVATE', 'url': 'secret'}]}, ('U2', 'W4'))
        self.assertEqual(result['citations'], [{'evidence_id': 'U2'}, {'evidence_id': 'W4'}])

    def test_no_answer_does_not_attach_background(self):
        result = self.validate({'answerable': False, 'customer_reply': '本轮不足。'})
        self.assertEqual(result['citations'], [])

    def test_no_sources_is_not_missing_citation(self):
        self.assertEqual(self.validate({'answerable': True, 'customer_reply': '可展示图片。'}, ())['citations'], [])

    def test_content_contract_still_checked(self):
        with self.assertRaises(OutputContractError):
            self.validate({'answerable': 'true', 'customer_reply': '内容'})

    def test_final_and_revision_use_text_not_json_protocol(self):
        from backend.sales.model_contract import answer_contract
        contract = answer_contract(None, {'T1'})
        self.assertIsNone(contract.json_schema)
        self.assertFalse(contract.json_output)
        for prompt in (COMMON, REVISION_PROMPT):
            self.assertIn('自然语言正文', prompt)
            self.assertNotIn('至少一个真实引用', prompt)
        result = contract.validate('介绍。')
        self.assertEqual(result['customer_reply'], '介绍。')
        self.assertEqual(result['citations'], [{'evidence_id': 'T1'}])

    def test_materialized_source_identity_cannot_override_backend(self):
        from backend.app import materialize_citations
        rows = {'T1': {'citations': [{'evidence_id': 'FORGED', 'document_name': '来源', 'source_page': 2}]},
                'U2': {'citations': []}}
        refs = materialize_citations(self.validate({'answerable': True, 'customer_reply': '内容'}, ('T1', 'U2'))['citations'], rows)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]['evidence_id'], 'T1')
        self.assertEqual(refs[0]['reference_label'], '参考资料')
        self.assertFalse(refs[0]['fact_support_verified'])

    def test_web_compatibility_cannot_expand_actual_context(self):
        from backend.sales.model_contract import chat_contract
        contract = chat_contract(None, ['W2'], program_references=True)
        value = contract.validate('公开线索。')
        self.assertEqual(value['used_source_ids'], ['W2'])
        self.assertFalse(contract.json_output)

    def test_missing_citation_does_not_repeat_model_generation(self):
        import torch
        from backend.tests.sales.test_model_contract import Model, Tokenizer
        from backend.sales.model_contract import answer_contract, checked_generate
        tokenizer = Tokenizer()
        model = Model(['介绍。'])
        checked_generate(model, contract=answer_contract(tokenizer, {'T1'}),
                         input_ids=torch.tensor([[1, 2]]), max_new_tokens=100)
        self.assertEqual(len(model.calls), 1)


if __name__ == '__main__':
    unittest.main()
