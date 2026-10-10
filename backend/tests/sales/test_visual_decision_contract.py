"""Source-availability decisions are strict protocol, not semantic routing."""
import json
import unittest
import torch

from backend.sales.planning_spec import planning_output_schema
from backend.sales.model_contract import validate_plan, validate_replan, checked_generate, json_contract
from backend.sales.output_contract import OutputContractError
from backend.sales.tool_planner import ToolPlan
from backend.sales.attachment_visuals import wants_inspection
from backend.tests.sales.test_model_contract import Model, Tokenizer


def wire(source='customer_documents', **changes):
    return dict(sources=[source], operation='comparison', categories=[],
                targets=[], aspects=['可见内容'], wants_images=False, **changes)


class VisualDecisionContractTests(unittest.TestCase):
    def test_available_attachment_requires_decision_even_unselected(self):
        for source in ('general_chat', 'company_rag', 'customer_documents', 'public_web_search'):
            with self.subTest(source=source), self.assertRaises(OutputContractError) as error:
                validate_plan(wire(source), require_compact=True,
                              allowed_tools=[source, 'customer_documents'])
            self.assertEqual(error.exception.field, 'document_visual_required')
            self.assertEqual(error.exception.code, 'OUTPUT_FIELD_MISSING')

    def test_unavailable_attachment_does_not_add_requirement(self):
        for source in ('company_rag', 'general_chat', 'public_web_search'):
            validate_plan(wire(source), require_compact=True, allowed_tools=[source])
            schema = planning_output_schema([source])
            self.assertNotIn('document_visual_required', schema['properties'])
            self.assertNotIn('document_visual_required', schema['required'])

    def test_null_and_coercible_values_never_become_false(self):
        for value in (None, 0, 1, 'true', 'false', [], {}):
            with self.subTest(value=value), self.assertRaises(OutputContractError) as error:
                validate_plan(wire(document_visual_required=value), require_compact=True)
            self.assertEqual(error.exception.field, 'document_visual_required')
            self.assertEqual(error.exception.code, 'OUTPUT_FIELD_VALUE_INVALID')

    def test_bool_decisions_survive_lowering_and_inspection(self):
        for inspect in (True, False):
            for display in (True, False):
                value = {**wire(document_visual_required=inspect), 'wants_images':display}
                plan = ToolPlan.model_validate(validate_plan(value, require_compact=True))
                self.assertIs(plan.document_visual_required, inspect)
                self.assertIs(wants_inspection(plan), inspect)
                self.assertIs(plan.wants_visuals, display)

    def test_mixed_source_uses_explicit_visual_decision(self):
        value = {**wire(document_visual_required=True), 'sources':['customer_documents','company_rag']}
        plan = ToolPlan.model_validate(validate_plan(value, require_compact=True))
        self.assertTrue(wants_inspection(plan))
        self.assertFalse(plan.wants_visuals)

    def test_existing_one_round_recovery_can_override_false(self):
        plan = ToolPlan.model_validate(wire(document_visual_required=False))
        self.assertFalse(wants_inspection(plan))
        plan.recovery_directive = {'customer_visual_recovery':True}
        self.assertTrue(wants_inspection(plan))

    def test_schema_requires_nonnullable_bool_when_available(self):
        for sources in (['customer_documents'], ['company_rag','customer_documents']):
            schema = planning_output_schema(sources)
            self.assertIn('document_visual_required', schema['required'])
            self.assertEqual(schema['properties']['document_visual_required'], {'type':'boolean'})

    def test_replan_obeys_same_requirement_without_changing_selection(self):
        value = {'action':'execute','gap_ids':['G'], 'revised_plan':wire()}
        options = dict(allowed_tools=['customer_documents'], gap_ids=['G'], require_compact=True)
        with self.assertRaises(OutputContractError):
            validate_replan(value, **options)
        value['revised_plan']['document_visual_required'] = True
        revised = validate_replan(value, **options)['revised_plan']
        self.assertTrue(revised['document_visual_required'])

    def test_missing_field_uses_format_retry_not_tool_recovery(self):
        bad = json.dumps(wire())
        good = json.dumps(wire(document_visual_required=True))
        model, tokenizer = Model([bad, good]), Tokenizer()
        contract = json_contract('planner', tokenizer,
            lambda value:validate_plan(value, require_compact=True))
        checked_generate(model, contract=contract,
                         input_ids=torch.tensor([[1,2]]), max_new_tokens=512)
        self.assertEqual(len(model.calls), 2)

    def test_legacy_internal_plan_keeps_compatibility(self):
        plan = ToolPlan(tools=['customer_documents'])
        self.assertIsNone(plan.document_visual_required)
        self.assertFalse(wants_inspection(plan))

    def test_malformed_source_type_still_reports_contract_error(self):
        for sources in (None, 1, 'customer_documents'):
            with self.subTest(sources=sources), self.assertRaises(OutputContractError):
                validate_plan({**wire(document_visual_required=False), 'sources':sources},
                              require_compact=True)


if __name__ == '__main__':
    unittest.main()
