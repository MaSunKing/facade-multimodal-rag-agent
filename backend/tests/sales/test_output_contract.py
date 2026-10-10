import json
import unittest
from backend.sales.output_contract import repair_and_validate, OutputContractError, parse_or_repair
from backend.sales.runtime_status import classify


def answer():
    return dict(intent='document_qa', answerable=True, customer_reply='合计123.45，仅为示例。', citations=['U1'])


class ContractTests(unittest.TestCase):
    def test_minimal_answer_defaults_nonfactual_fields(self):
        result = repair_and_validate(dict(answerable=True, customer_reply='有限介绍', citations=['U1']),
                                     {'U1'}, {'unknown'})
        self.assertEqual(result['intent'], 'unknown')
        self.assertEqual(result['key_points'], [])

    def test_absent_checklist_is_not_missing_answer(self):
        from backend.sales.answer_patch import checklist_diagnostics
        result = checklist_diagnostics(answer(), ['概括附件'], [], {'U1': {'text':'内容'}})
        self.assertEqual(result['status'], 'not_provided')
        self.assertNotIn('missing_documents', result)

    def check_error(self, value, code, field=None):
        with self.assertRaises(OutputContractError) as caught:
            repair_and_validate(value, {'U1'}, {'document_qa'})
        info = classify(caught.exception, 'validate_answer')
        self.assertEqual(info['code'], code)
        if field:
            self.assertEqual(info['field'], field)

    def test_wrap_known_id_and_keep_numbers(self):
        result = repair_and_validate(answer(), {'U1'}, {'document_qa'})
        self.assertEqual(result['customer_reply'], answer()['customer_reply'])
        self.assertEqual(result['citations'], [{'evidence_id':'U1'}])

    def test_unknown_id_not_guessed(self):
        self.check_error({**answer(), 'citations':['SECRET_ID']}, 'OUTPUT_CITATION_UNKNOWN','citations')

    def test_missing_core(self):
        value=answer(); value.pop('customer_reply')
        self.check_error(value, 'OUTPUT_FIELD_MISSING','customer_reply')

    def test_wrong_type(self):
        self.check_error({**answer(), 'answerable':'true'}, 'OUTPUT_FIELD_TYPE_INVALID','answerable')

    def test_bad_citation_format(self):
        self.check_error({**answer(), 'citations':[{'url':'PRIVATE'}]}, 'OUTPUT_CITATION_FORMAT_INVALID')

    def test_missing_citation(self):
        self.check_error({**answer(), 'citations':[]}, 'OUTPUT_CITATION_MISSING')

    def test_empty_reply(self):
        self.check_error({**answer(), 'customer_reply':' '}, 'OUTPUT_REPLY_EMPTY')

    def test_bad_checklist_is_only_diagnostic(self):
        result = repair_and_validate({**answer(), 'answer_aspect_coverage':'bad'}, {'U1'}, {'document_qa'})
        self.assertNotIn('answer_aspect_coverage',result)

    def test_token_limit_specific_error(self):
        def fail(_): raise ValueError('private source text')
        with self.assertRaises(OutputContractError) as caught:
            parse_or_repair('{', fail, fail, hit_output_limit=True)
        self.assertEqual(classify(caught.exception)['code'], 'OUTPUT_TOKEN_LIMIT')

    def test_invalid_json_specific_error(self):
        def fail(_): raise ValueError('private source text')
        with self.assertRaises(OutputContractError) as caught:
            parse_or_repair('{', fail, fail)
        self.assertEqual(classify(caught.exception)['code'], 'OUTPUT_INVALID_JSON')

    def test_array_not_object(self):
        def fail(_): raise ValueError('not object')
        with self.assertRaises(OutputContractError) as caught:
            parse_or_repair('[]', fail, fail)
        self.assertEqual(caught.exception.code,'OUTPUT_NOT_OBJECT')

    def test_answer_workers_have_no_numeric_fact_check_calls(self):
        import ast
        from pathlib import Path
        tree = ast.parse((Path(__file__).parents[2] / 'app.py').read_text(encoding='utf-8'))
        # Definitions remain available to legacy tests, but production workers
        # may not invoke these post-generation heuristics.
        prohibited = {'evidence_support_audit','guard_normative_claims',
                      'remove_unsupported_numeric_sentences','repair_supported_negative_answer'}
        calls = [n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertFalse(prohibited.intersection(calls))
