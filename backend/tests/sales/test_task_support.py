import unittest
import json
from pathlib import Path
from unittest.mock import patch
from backend.sales.task_support import candidate_risk, AUDIT_PROMPT, CHAT_SELF_CHECK

class TaskSupportTests(unittest.TestCase):
    def test_audit_and_replan_use_global_deadline_not_twenty_seconds(self):
        import ast
        import inspect
        from backend.app import DraftRequest,evaluate_task_support
        tree=ast.parse(inspect.getsource(evaluate_task_support))
        timers=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
                and isinstance(node.func,ast.Name) and node.func.id=='local_generation_stopping_criteria']
        self.assertEqual(len(timers),1)
        self.assertIsNone(timers[0].args[0].value)
        for purpose,value in [('audit',{'status':'supported','supported_evidence_ids':['image1']}),
                              ('replan',{'action':'stop'})]:
            with patch('backend.app.generate_multi_visual_response') as model:
                with self.assertRaisesRegex(ValueError,'audit_pixels_forbidden'):
                    evaluate_task_support(DraftRequest(customer_question='Describe image'),{},
                        image_paths=[Path('fixture.png')],purpose=purpose)
            model.assert_not_called()

    def test_normal_chat_has_no_extra_call(self):
        self.assertEqual(candidate_risk({'customer_reply':'Hello!',
            'task_support':{'status':'supported','needs_verification':False,'ambiguous':False}}),[])

    def test_missing_self_check_is_not_implicitly_trusted(self):
        self.assertTrue(candidate_risk({'customer_reply':'Unknown entity is a game character'}))

    def test_ambiguity_triggers_independent_review(self):
        self.assertTrue(candidate_risk({'task_support':{'status':'supported','ambiguous':True}}))

    def test_search_claim_cannot_pass_on_self_confidence(self):
        value={'customer_reply':'当前未找到相关公开资料。','task_support':{'status':'supported'}}
        self.assertIn('SEARCH_CLAIM_WITHOUT_SEARCH',candidate_risk(value))
        self.assertEqual(candidate_risk(value,searched=True),[])

    def test_prompts_do_not_contain_case_specific_routes(self):
        self.assertNotIn('无尽冬日',AUDIT_PROMPT+CHAT_SELF_CHECK)
        self.assertIn('当前用户问题',AUDIT_PROMPT)

    def test_planners_allow_bounded_public_supplement_without_keyword_queries(self):
        from backend.app import TOOL_PLANNER_SYSTEM_PROMPT
        from backend.sales.task_support import REPLAN_PROMPT
        self.assertIn('不仅限实时信息', TOOL_PLANNER_SYSTEM_PROMPT)
        self.assertIn('联网权限不是自动搜索指令', TOOL_PLANNER_SYSTEM_PROMPT)
        self.assertIn('完整独立', TOOL_PLANNER_SYSTEM_PROMPT)
        self.assertIn('不是固定兜底', REPLAN_PROMPT)
        self.assertIn('私有项目', REPLAN_PROMPT)

    def test_visual_support_ids_must_be_real_manifest_ids(self):
        from backend.app import DraftRequest,evaluate_task_support
        value={'status':'supported','supported_evidence_ids':['blue panel description']}
        with patch('backend.app.generate_multi_visual_response',return_value=json.dumps(value)) as model:
            with self.assertRaisesRegex(ValueError,'audit_pixels_forbidden'):
                evaluate_task_support(DraftRequest(customer_question='Which panel?'),{},image_paths=[Path('fixture.png')])
            model.assert_not_called()

    def test_real_visual_id_is_accepted_as_opinion_not_gold(self):
        from backend.app import DraftRequest,evaluate_task_support
        value={'status':'supported','supported_evidence_ids':['image1']}
        with patch('backend.app.generate_multi_visual_response',return_value=json.dumps(value)) as model:
            with self.assertRaisesRegex(ValueError,'audit_pixels_forbidden'):
                evaluate_task_support(DraftRequest(customer_question='Which panel?'),{},image_paths=[Path('fixture.png')])
            model.assert_not_called()

    def test_chat_feedback_is_preserved_for_replanning(self):
        from backend.app import DraftRequest,_CustomerAnswerWorkflowCallbacks
        from backend.sales.tool_planner import ToolPlan
        with patch('backend.app.evaluate_task_support',return_value={'action':'execute',
            'revised_plan':{'tools':['public_web_search'],'requires_public_web':True,'retrieval_query':'Identify entity'}}) as model:
            result=_CustomerAnswerWorkflowCallbacks.plan_retry(DraftRequest(customer_question='Identify entity',use_online_search=True),
                ToolPlan(tools=['general_chat']),{'meta':{'mode':'task_support_blocked','task_support_audit':{
                    'status':'uncertain','error_code':'ENTITY_UNVERIFIED','problem':'identity not established'}}})
        self.assertEqual(model.call_args.kwargs['purpose'],'replan')
        self.assertEqual(result.recovery_directive['problem'],'identity not established')
        self.assertEqual(result.tools,['public_web_search'])

    def test_no_web_permission_removes_model_web_replan(self):
        from backend.app import DraftRequest,_CustomerAnswerWorkflowCallbacks
        from backend.sales.tool_planner import ToolPlan
        with patch('backend.app.evaluate_task_support',return_value={'action':'execute',
            'revised_plan':{'tools':['public_web_search'],'requires_public_web':True}}):
            result=_CustomerAnswerWorkflowCallbacks.plan_retry(DraftRequest(customer_question='Current public facts'),
                ToolPlan(tools=['general_chat']),{'meta':{'mode':'task_support_blocked','task_support_audit':{'status':'unsupported'}}})
        self.assertIsNone(result)

    def test_clear_missing_facts_replan_even_if_auditor_says_clarify(self):
        from backend.app import DraftRequest,_CustomerAnswerWorkflowCallbacks
        from backend.sales.tool_planner import ToolPlan
        with patch('backend.app.evaluate_task_support',return_value={'action':'execute',
            'revised_plan':{'tools':['public_web_search'],'requires_public_web':True}}) as model:
            result=_CustomerAnswerWorkflowCallbacks.plan_retry(DraftRequest(customer_question='Current public facts',use_online_search=True),
                ToolPlan(tools=['general_chat']),{'meta':{'mode':'task_support_blocked','task_support_audit':{
                    'status':'unsupported','recommended_action':'clarify','gap_type':'current_public_fact'}}})
        self.assertEqual(result.tools,['public_web_search'])
        model.assert_called_once()

    def test_replanner_can_stop_without_guessing_or_tool_execution(self):
        from backend.app import DraftRequest,_CustomerAnswerWorkflowCallbacks
        from backend.sales.tool_planner import ToolPlan
        with patch('backend.app.evaluate_task_support',return_value={'action':'stop','revised_plan':None}):
            result=_CustomerAnswerWorkflowCallbacks.plan_retry(DraftRequest(customer_question='ambiguous phrase',use_online_search=True),
                ToolPlan(tools=['general_chat']),{'meta':{'mode':'task_support_blocked','task_support_audit':{'status':'uncertain'}}})
        self.assertIsNone(result)

    def test_replanner_contract_has_no_clarify_action(self):
        from backend.sales.task_support import REPLAN_PROMPT
        self.assertIn('"action":"execute|stop"', REPLAN_PROMPT)
        self.assertNotIn('execute|clarify', REPLAN_PROMPT)
        self.assertIn('即使审核错误给出 web_can_resolve=false', REPLAN_PROMPT)

    def test_local_query_preserves_original_and_late_constraints(self):
        from backend.sales.tool_planner import local_retrieval_query
        question = '主体的方案是什么？' * 70 + '不得用于浸水环境，需说明原因。'
        combined = local_retrieval_query(question, '主体 方案')
        self.assertTrue(combined.startswith(question))
        self.assertIn('不得用于浸水环境', combined)
        self.assertEqual(local_retrieval_query('What is the thickness?', 'What is the thickness?'), 'What is the thickness?')

    def test_initial_planner_no_longer_requests_keyword_query(self):
        from backend.app import TOOL_PLANNER_SYSTEM_PROMPT
        self.assertNotIn('short equivalent search terms', TOOL_PLANNER_SYSTEM_PROMPT)
        self.assertNotIn('"retrieval_query":"payment date"', TOOL_PLANNER_SYSTEM_PROMPT)
        self.assertIn('完整独立', TOOL_PLANNER_SYSTEM_PROMPT)

    def test_replan_false_web_hint_does_not_veto_authorised_model_choice(self):
        from backend.app import DraftRequest, _CustomerAnswerWorkflowCallbacks
        from backend.sales.tool_planner import ToolPlan
        request = DraftRequest(customer_question='Which models does this public company currently publish?', use_online_search=True)
        feedback = {'gap_type':'current_public_fact','web_can_resolve':False,
                    'missing_aspects':['Current model names published by the requested company']}
        with patch('backend.app.evaluate_task_support', return_value={'action':'execute',
                'revised_plan':{'tools':['public_web_search'],'requires_public_web':True,
                    'retrieval_query':request.customer_question}}):
            result = _CustomerAnswerWorkflowCallbacks.replan_from_feedback(
                request, ToolPlan(tools=['company_rag']), feedback, {})
        self.assertEqual(result.tools, ['public_web_search'])
        self.assertEqual(result.retrieval_query, request.customer_question)

if __name__=='__main__':unittest.main()
