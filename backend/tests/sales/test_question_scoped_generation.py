"""CPU contract checks for question-scoped generation and bounded replanning."""
import ast
import inspect
import unittest
from unittest.mock import patch

from backend import app
from backend.sales.tool_planner import ToolPlan
from backend.sales.planner_prompt import TOOL_PLANNER_SYSTEM_PROMPT
from backend.sales.task_support import REPLAN_PROMPT


class QuestionScopedGenerationTests(unittest.TestCase):
    def test_enterprise_and_attachment_routes_have_no_literal_answer_shortcuts(self):
        forbidden = {'product_overview_answer', '_single_evidence_answer', 'catalogue_case_answer'}
        for route in (app._run_facade_rag_answer, app._run_customer_document_answer):
            tree = ast.parse(inspect.getsource(route.steps))
            calls = {node.func.id for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
            self.assertFalse(calls & forbidden)
            self.assertIn('load_model', calls)

    def test_generation_prompts_scope_answer_without_dossier_copy(self):
        for prompt in (app.COMPANY_RAG_SYSTEM_PROMPT, app.COMPANY_RAG_VISUAL_SYSTEM_PROMPT,
                       app.CUSTOMER_DOCUMENT_SYSTEM_PROMPT):
            self.assertIn('不复制整份规格', prompt)
            self.assertIn('未进入Context不等于原资料没有信息', prompt)
            self.assertIn('内部回答草稿', prompt)

    def test_generation_failure_does_not_export_source_as_answer(self):
        request = app.DraftRequest(customer_question='只说产品有哪些')
        request._evidence_gate_audit = {'status': 'supported'}
        with patch('backend.app._source_derived_answer', side_effect=AssertionError('no literal fallback')):
            result = app.fallback_grounded_answer(request, {'text_evidence': [], 'visual_assets': []}, 'generation_failed')
        self.assertFalse(result['answerable'])
        self.assertEqual(result['citations'], [])
        self.assertEqual(result['meta']['mode'], 'grounded_answer_incomplete')

    def test_missing_context_goal_is_feedback_not_fixed_query_or_tool(self):
        request = app.DraftRequest(customer_question='比较两种材料的维护方法')
        plan = ToolPlan(tools=['company_rag'], retrieval_query=request.customer_question)
        response = {'answerable': False, 'meta': {'generation_input_audit': {
            'context_engine': {'missing_answer_goals': ['天然石材的维护方法']}}}}
        with patch.object(app._CustomerAnswerWorkflowCallbacks, 'replan_from_feedback', return_value=None) as replanner:
            app._CustomerAnswerWorkflowCallbacks.plan_retry(request, plan, response)
        # Candidate coverage alone is not proof of a semantic retrieval gap.
        replanner.assert_not_called()
        response['meta']['answer_review']={'decision':'recover','status':'partial',
            'missing_aspects':['天然石材的维护方法'],'recovery_need':'lookup',
            'proof':{'requirement':'维护方法','basis':'evidence_gap','reason':'当前实际Context只含一方',
                'evidence_ids':['T1'],'next_query':'天然石材维护方法'}}
        with patch.object(app._CustomerAnswerWorkflowCallbacks,'replan_from_feedback',return_value=None) as replanner:
            app._CustomerAnswerWorkflowCallbacks.plan_retry(request,plan,response)
        feedback=replanner.call_args.args[2]
        self.assertEqual(feedback['proof']['requirement'],'维护方法')
        self.assertNotIn('tools',feedback)

    def test_both_planners_allow_nontemporal_web_supplementation(self):
        self.assertIn('实时或库外公开信息', TOOL_PLANNER_SYSTEM_PROMPT)
        self.assertIn('不是固定兜底', REPLAN_PROMPT)
        self.assertIn('不能替代私有项目事实', REPLAN_PROMPT)


if __name__ == '__main__':
    unittest.main()
