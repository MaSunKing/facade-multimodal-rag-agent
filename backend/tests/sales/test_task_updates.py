import unittest
from unittest.mock import patch
from pydantic import ValidationError
from backend.app import DraftRequest, planner_context_input, task_update_context, _CustomerAnswerWorkflowCallbacks
from backend.sales.task_updates import bind_task_instructions
from backend.sales.tool_planner import ToolPlan
from backend.tests.sales.test_audit_error_contract import text_audit_model


class TaskUpdateTests(unittest.TestCase):
    def test_multiple_updates_keep_order_and_normalize_once(self):
        request = DraftRequest(customer_question='取消厚度，增加价格',
            task_instructions=['比较附件A和B的厚度', '加入C，并比较安装方式'],
            interrupted_question='legacy flattened chain', memory_enabled=True)
        bind_task_instructions(request)
        text = request.customer_question
        bind_task_instructions(request)
        self.assertEqual(text, request.customer_question)
        self.assertNotIn('legacy flattened chain', text)
        self.assertFalse(request.memory_enabled)
        context = planner_context_input(request)
        self.assertEqual(context['current_question'], text)
        self.assertEqual(context['task_update']['user_instructions'],
            ['比较附件A和B的厚度', '加入C，并比较安装方式', '取消厚度，增加价格'])
        self.assertEqual(context['task_update']['latest_requirement'], '取消厚度，增加价格')

    def test_legacy_and_independent_queue_contracts(self):
        request = DraftRequest(customer_question='再加一个对象', interrupted_question='原对象')
        bind_task_instructions(request)
        self.assertEqual(task_update_context(request)['user_instructions'], ['原对象','再加一个对象'])
        independent = DraftRequest(customer_question='独立问题')
        bind_task_instructions(independent)
        self.assertEqual(independent.customer_question, '独立问题')
        self.assertEqual(task_update_context(independent), {})

    def test_independent_question_keeps_labelled_history_out_of_primary_query(self):
        from backend.app import ConversationTurn
        request = DraftRequest(customer_question='新的主问题', conversation_context=[
            ConversationTurn(role='user', content='旧问题中的对象'),
            ConversationTurn(role='assistant', content='未经验证的旧回答')])
        bind_task_instructions(request)
        context = planner_context_input(request)
        self.assertEqual(context['current_question'], '新的主问题')
        self.assertEqual(context['task_update'], {})
        self.assertEqual(context['previous_exchange'][0]['role'], 'user')
        self.assertNotIn('旧问题中的对象', request.customer_question)

    def test_independent_question_keeps_labelled_history_out_of_primary_query(self):
        from backend.app import ConversationTurn
        request = DraftRequest(customer_question='新的主问题', conversation_context=[
            ConversationTurn(role='user', content='旧问题中的对象'),
            ConversationTurn(role='assistant', content='未经验证的旧回答')])
        bind_task_instructions(request)
        context = planner_context_input(request)
        self.assertEqual(context['current_question'], '新的主问题')
        self.assertEqual(context['task_update'], {})
        self.assertEqual(context['previous_exchange'][0]['role'], 'user')
        self.assertNotIn('旧问题中的对象', request.customer_question)

    def test_input_limit_rejects_without_silent_truncation(self):
        for prior in (['x']*20, ['x'*12000], ['']):
            with self.assertRaises(ValidationError):
                DraftRequest(customer_question='new', task_instructions=prior)

    def test_task_envelope_does_not_change_answer_language(self):
        from backend.sales.response_language import request_answer_language
        request=DraftRequest(customer_question='Also describe target B; please answer in English', task_instructions=['Describe target A'])
        bind_task_instructions(request)
        self.assertEqual(request_answer_language(request), 'English')

    def test_task_text_contains_only_user_text_not_protocol(self):
        from backend.sales.task_updates import current_task_text
        for chain in (['比较甲乙安装方法', '再增加丙'], ['Explain A', 'Also explain B']):
            self.assertEqual(current_task_text(chain), '\n'.join(chain))

    def test_actual_search_outbound_and_plan_guard_exclude_protocol(self):
        from backend.app import maybe_search_online
        from backend.sales.tool_planner import retrieval_query_for_attempt
        for prior, latest, query in [
            ('新加坡今天天气如何', '再给我杭州今天天气', '新加坡和杭州今天天气'),
            ('介绍甲的维护要求', '增加乙的维护要求', '甲与乙的维护要求'),
            ('Compare public standards A and B', 'Include standard C', 'Compare public standards A B C')]:
            with self.subTest(query=query):
                request=DraftRequest(customer_question=latest, task_instructions=[prior], use_online_search=True)
                bind_task_instructions(request)
                plan=ToolPlan(tools=['public_web_search'],requires_public_web=True,web_retrieval_query=query)
                guarded=_CustomerAnswerWorkflowCallbacks.validate_plan(request,plan)
                self.assertEqual(guarded.web_retrieval_query,query)
                actual_query=retrieval_query_for_attempt(request.customer_question,guarded,public=True)
                with patch('backend.app.is_baidu_search_configured',return_value=True), patch(
                    'backend.sales.search_mcp.search_web_via_mcp',return_value={
                        'query':actual_query,'sources':[{'source_id':'W1','url':'https://example.org','title':'Reference'}]}) as search:
                    maybe_search_online(request,actual_query)
                outbound=search.call_args.args[0]
                from backend.app import resolve_current_public_query
                self.assertEqual(outbound,resolve_current_public_query(query))
                self.assertNotIn('Current interrupted task',outbound)
                self.assertNotIn('chronological',outbound)
                self.assertNotIn('policy',outbound)

    def test_language_fallback_uses_raw_requests_not_task_instructions(self):
        from backend.sales.response_language import compatible_web_query
        request=DraftRequest(customer_question='再增加乙',task_instructions=['查询甲'])
        bind_task_instructions(request)
        self.assertEqual(compatible_web_query(request.customer_question,'English only query'), '查询甲\n再增加乙')

    def test_audit_receives_entire_task_not_last_target(self):
        for tool, prefix in [('company_rag','T'),('customer_documents','U'),('public_web_search','W')]:
            with self.subTest(tool=tool):
                request = DraftRequest(customer_question='再增加目标B', task_instructions=['介绍目标A','都要图片'], use_online_search=True)
                bind_task_instructions(request)
                request._prepared_evidence = [{'evidence_id':prefix+'1','text':'目标B的片段', 'source_type':tool}]
                sources={tool:([],{}) if tool=='public_web_search' else {}}
                request._source_results=sources
                with text_audit_model([]), patch('backend.app.evaluate_task_support',return_value={
                    'decision':'pass','status':'supported','supported_evidence_ids':[prefix+'1']}) as evaluate:
                    _CustomerAnswerWorkflowCallbacks.review_answer(request, ToolPlan(tools=[tool]),
                        {'customer_reply':'目标A和B资料','meta':{'model_used':True}},1)
                payload=evaluate.call_args.args[1]
                self.assertIn('介绍目标A',payload['question'])
                self.assertIn('都要图片',payload['question'])
                self.assertIn('再增加目标B',payload['question'])
                self.assertEqual(payload['task_update']['user_instructions'],request._task_instruction_chain)


if __name__ == '__main__':
    unittest.main()
