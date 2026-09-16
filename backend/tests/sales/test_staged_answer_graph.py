"""Node execution tests: no GPU, network, production data, or fixed answers."""
import unittest
from unittest.mock import patch
from contextvars import ContextVar
from backend.sales.answer_graph import build_customer_answer_graph
from backend.sales.staged_execution import WorkflowStep, staged_answer
from backend.sales.tool_planner import ToolPlan
import backend.app as app


class StagedGraphTests(unittest.TestCase):
    def test_tools_are_real_nodes_and_results_reach_generation_once(self):
        calls = []
        context = ContextVar('test_owner', default='missing')
        class Callbacks:
            def plan_tools(self, request): return ToolPlan(tools=['customer_documents','company_rag'])
            def open_steps(self, request, plan):
                a = yield WorkflowStep('customer_documents', lambda: calls.append(('documents',context.get())) or 7)
                b = yield WorkflowStep('company_rag', lambda: calls.append(('rag',context.get())) or 9)
                yield WorkflowStep('compose_evidence')
                combined = a+b
                yield WorkflowStep('generate_answer')
                calls.append(('generate',context.get()))
                yield WorkflowStep('validate_answer')
                return {'answer': combined, 'meta': {}}
        token = context.set('owner-A')
        try: result = build_customer_answer_graph(Callbacks()).invoke({'request': {}})['response']
        finally: context.reset(token)
        self.assertEqual(result['answer'],16)
        self.assertEqual(calls,[('documents','owner-A'),('rag','owner-A'),('generate','owner-A')])
        self.assertEqual([v['node'] for v in result['meta']['orchestration']['node_trace']],
                         ['customer_documents','company_rag','compose_evidence','generate_answer','validate_answer'])

    def test_retry_reuses_web_result_and_is_bounded(self):
        calls=[]
        class Callbacks:
            def plan_tools(self, request): return ToolPlan(tools=['public_web_search'],retrieval_query='a')
            def open_steps(self, request, plan):
                value=yield WorkflowStep('public_web_search',lambda: calls.append('web') or 'source')
                yield WorkflowStep('generate_answer')
                return {'answer':value,'meta':{}}
            def plan_retry(self,request,plan,response):
                return plan.model_copy(update={'retrieval_query':'b'})
        result=build_customer_answer_graph(Callbacks()).invoke({'request':{}})['response']
        self.assertEqual(calls,['web'])
        self.assertEqual(result['meta']['orchestration']['tool_rounds'],2)
        self.assertTrue(result['meta']['orchestration']['node_trace'][2]['reused'])

    def test_rejected_tool_closes_temporary_resources(self):
        closed=[]; calls=[]
        class Callbacks:
            def plan_tools(self,request): return ToolPlan(tools=[])
            def check_step(self,request,plan,node): raise PermissionError('denied')
            def open_steps(self,request,plan):
                try: yield WorkflowStep('public_web_search',lambda:calls.append('web'))
                finally: closed.append(True)
        with self.assertRaises(PermissionError):
            build_customer_answer_graph(Callbacks()).invoke({'request':{}})
        self.assertEqual(calls,[]); self.assertEqual(closed,[True])

    def test_tool_error_reaches_existing_handler_not_full_replay(self):
        calls=[]
        def fail(): calls.append(1); raise OSError('offline')
        @staged_answer
        def flow():
            try: yield WorkflowStep('public_web_search',fail)
            except OSError: return {'answer':'unavailable'}
        self.assertEqual(flow(),{'answer':'unavailable'})
        self.assertEqual(calls,[1])

    def test_actual_general_path_defers_web_and_preserves_current_question(self):
        request=app.DraftRequest(customer_question='查一条公开资料',use_online_search=True)
        with patch.object(app,'maybe_search_online') as search:
            steps=app._run_general_local_answer.steps(request,allow_public_web=True,web_source_profile='general_public')
            step=next(steps)
            search.assert_not_called()
            self.assertEqual(step.node,'public_web_search')
            self.assertEqual(step.args[1],request.customer_question)
            steps.close()

    def test_policy_removes_unapproved_web_without_loading_model(self):
        request=app.DraftRequest(customer_question='查询公开信息',use_online_search=False)
        proposed=ToolPlan(tools=['public_web_search'],requires_public_web=True)
        with patch.object(app,'load_model') as model:
            plan=app._CustomerAnswerWorkflowCallbacks.validate_plan(request,proposed)
        self.assertEqual(plan.tools,[]); model.assert_not_called()

    def test_direct_image_joint_answer_uses_visual_node_without_extra_pass(self):
        request=app.DraftRequest(customer_question='描述图中内容', image_data_url='data:image/png;base64,AAAA')
        with patch.object(app,'general_local_chat_answer') as generate:
            steps=app._run_general_local_answer.steps(request,allow_public_web=False)
            step=next(steps)
            self.assertEqual(step.node,'visual_inspection')
            generate.assert_not_called()
            steps.close()

    def test_graph_failure_does_not_replay_answer(self):
        class Broken:
            def invoke(self,*args,**kwargs): raise RuntimeError('after tool')
        with patch.object(app,'customer_answer_graph',return_value=Broken()), patch.object(
            app._CustomerAnswerWorkflowCallbacks,'answer_planned') as answer:
            with self.assertRaises(RuntimeError):
                app.grounded_answer(app.DraftRequest(customer_question='普通问题'))
        answer.assert_not_called()


if __name__=='__main__': unittest.main()
