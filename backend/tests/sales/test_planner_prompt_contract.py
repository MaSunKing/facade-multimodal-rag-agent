"""Validate compact prompt examples against the real schema and policy guard."""
import json
import unittest

from backend.sales.planner_prompt import TOOL_PLANNER_SYSTEM_PROMPT, PLANNER_EXAMPLES
from backend.sales.tool_planner import ToolPlan, guard_plan
from backend.sales.model_contract import validate_plan


class CompactPlannerPromptTests(unittest.TestCase):
    def test_identity_example_does_not_expand_into_product_task(self):
        line=next(line for line in TOOL_PLANNER_SYSTEM_PROMPT.splitlines()
                  if line.startswith('{"sources"'))
        plan=ToolPlan.model_validate(json.loads(line))
        self.assertFalse(plan.product_overview)
        self.assertEqual(plan.answer_aspects,[])
        self.assertNotIn('产品',plan.retrieval_query)

    def test_examples_are_executable_contracts_not_invalid_shapes(self):
        plans = []
        for data in PLANNER_EXAMPLES:
            # Attachments are available in this test even for unrelated routes.
            data = {**data, 'document_visual_required':data.get('document_visual_required',False)}
            validate_plan(data, require_company_scope=True, allowed_tools=[
                'general_chat','customer_documents','company_rag','visual_inspection','public_web_search'])
            proposed = ToolPlan.model_validate(data)
            safe = guard_plan(proposed, has_documents=True, has_image=True,
                              facade_related=False, web_allowed=True)
            self.assertEqual(safe.tools, proposed.tools)
            plans.append(safe)
        self.assertGreaterEqual(len(plans), 5)
        self.assertTrue(any(p.document_operation == 'summary' for p in plans))
        self.assertTrue(any(p.tools == ['customer_documents', 'company_rag'] for p in plans))
        self.assertTrue(any(p.wants_visuals and p.visual_scope == 'product' for p in plans))

    def test_web_example_still_requires_permission(self):
        plan = ToolPlan(tools=['public_web_search'], requires_public_web=True)
        safe = guard_plan(plan, has_documents=False, has_image=False,
                          facade_related=False, web_allowed=False)
        self.assertNotIn('public_web_search', safe.tools)

    def test_compact_contract_retains_nonrouting_responsibilities(self):
        for contract in ('当前用户要求优先', '助手说法当事实',
                         'memory_updates', 'document_visual_required', 'document_names',
                         '否定', '私有正文', '库外公开信息',
                         'operation', 'document_names'):
            self.assertIn(contract, TOOL_PLANNER_SYSTEM_PROMPT)

    def test_generic_tool_selection_and_multisource_contract(self):
        self.assertNotIn('general AI knowledge', TOOL_PLANNER_SYSTEM_PROMPT)
        self.assertNotIn('minimum useful tools', TOOL_PLANNER_SYSTEM_PROMPT)
        self.assertIn('只选相关且获授权的来源', TOOL_PLANNER_SYSTEM_PROMPT)
        self.assertIn('可多选', TOOL_PLANNER_SYSTEM_PROMPT)
        self.assertIn('source_operations', TOOL_PLANNER_SYSTEM_PROMPT)

    def test_removed_duplicate_fields_not_visible_but_legacy_is_compatible(self):
        from backend.sales.planning_spec import planning_output_schema, expand_planning_spec
        from backend.sales.output_contract import OutputContractError
        self.assertEqual(len(planning_output_schema()['properties']), 11)
        self.assertFalse({'image_scope','image_target','image_match'} & set(planning_output_schema()['properties']))
        self.assertEqual(len(planning_output_schema(['company_rag'])['properties']), 9)
        for field in ('targets','filters','aspects'):
            self.assertNotIn(field,planning_output_schema()['properties'])
            with self.assertRaises(OutputContractError):
                validate_plan({**PLANNER_EXAMPLES[0],field:[]},require_compact=True)
        self.assertNotIn('targets',planning_output_schema()['$defs']['ScopedQuery']['properties'])
        for field in ('query', 'reason', 'document_selection'):
            with self.assertRaises(OutputContractError):
                validate_plan({**PLANNER_EXAMPLES[0],field:'all'}, require_compact=True)
        legacy = expand_planning_spec({**PLANNER_EXAMPLES[0], 'query':'企业身份', 'reason':'兼容'})
        self.assertEqual(legacy['retrieval_query'], '企业身份')

    def test_all_examples_use_current_model_contract(self):
        for example in PLANNER_EXAMPLES:
            validate_plan(example, require_compact=True)

    def test_field_and_enum_contract_is_generated_from_schema(self):
        from backend.sales.planning_spec import planning_output_schema
        schema=json.loads(next(line for line in TOOL_PLANNER_SYSTEM_PROMPT.splitlines() if line.startswith('{"$defs"')))
        self.assertEqual(schema['properties']['operation']['enum'], ['lookup','summary','comparison'])
        self.assertEqual(set(schema['properties']), set(planning_output_schema()['properties']))
        self.assertNotIn('不负责猜具体实体、产品形态编码', TOOL_PLANNER_SYSTEM_PROMPT)
        self.assertNotIn('条件必填覆盖前面示例', TOOL_PLANNER_SYSTEM_PROMPT)
        self.assertLess(len(TOOL_PLANNER_SYSTEM_PROMPT), 8100)


if __name__ == '__main__':
    unittest.main()
