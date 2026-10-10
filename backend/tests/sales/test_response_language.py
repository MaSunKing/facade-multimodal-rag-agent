import json
import unittest
from unittest.mock import patch
import backend.app as app
from backend.sales.response_language import request_answer_language, resolve_answer_language, compatible_web_query, web_caution
from backend.sales.tool_planner import ToolPlan, retrieval_query_for_attempt


class ResponseLanguageTests(unittest.TestCase):
    def test_request_language_ignores_sources_and_old_topic(self):
        request = app.DraftRequest(customer_question="how's the weather like in beijing today")
        self.assertEqual(request_answer_language(request), 'English')
        request._source_results = {'company_rag': {'text': '中文企业资料'}}
        self.assertEqual(request_answer_language(request), 'English')

    def test_explicit_preference_wins(self):
        self.assertEqual(resolve_answer_language('请用英语回答：今天北京天气如何？'), 'English')
        self.assertEqual(resolve_answer_language('What is this? Please answer in Chinese.'), 'Chinese')
        self.assertEqual(resolve_answer_language('What is this? Reply in Spanish.'), 'Spanish')

    def test_missing_replan_web_query_never_uses_local_chinese_query(self):
        question = 'How is the weather in Beijing today?'
        plan = ToolPlan(retrieval_query='今天北京天气如何？', recovery_directive={'gap':'weather'})
        self.assertEqual(retrieval_query_for_attempt(question, plan, public=True), question)

    def test_wrong_script_web_query_falls_back_without_extra_model(self):
        self.assertEqual(compatible_web_query('How is the weather today?', '今天的天气如何？'), 'How is the weather today?')
        self.assertEqual(compatible_web_query('今天的天气如何？', 'How is the weather today?'), '今天的天气如何？')
        self.assertEqual(compatible_web_query('How is Beijing weather?', 'Beijing weather forecast'), 'Beijing weather forecast')

    def test_plan_guard_inherits_request_language_and_corrects_web(self):
        request = app.DraftRequest(customer_question='How is Beijing weather?', use_online_search=True)
        with patch.object(app, 'check_budget'):
            plan = app._CustomerAnswerWorkflowCallbacks.validate_plan(request,
                ToolPlan(tools=['public_web_search'], requires_public_web=True,
                         answer_language='Chinese', web_retrieval_query='北京天气如何？'))
        self.assertEqual(plan.answer_language, 'English')
        self.assertEqual(plan.web_retrieval_query, request.customer_question)

    def test_english_caution_is_not_chinese(self):
        self.assertIn('public web', web_caution('English'))
        self.assertNotIn('以下', web_caution('English'))

    def test_replanner_missing_fields_inherits_language_and_original_web_query(self):
        request = app.DraftRequest(customer_question='How is the weather in Beijing today?', use_online_search=True)
        plan = ToolPlan(tools=['public_web_search'], requires_public_web=True)
        response = {'action':'execute', 'revised_plan':{'tools':['public_web_search'],
                    'requires_public_web':True, 'retrieval_query':'今天北京的天气如何？'}}
        with patch.object(app, 'evaluate_task_support', return_value=response) as evaluate, \
             patch.object(app, 'check_budget'):
            revised = app._CustomerAnswerWorkflowCallbacks.replan_from_feedback(
                request, plan, {'gap_type':'current_public_fact', 'missing_aspects':['Beijing weather today']},
                {'answerable':False})
        self.assertIsNotNone(revised)
        self.assertEqual(revised.answer_language, 'English')
        self.assertEqual(revised.web_retrieval_query, request.customer_question)
        self.assertEqual(evaluate.call_args.args[1]['answer_language'], 'English')
        self.assertEqual(retrieval_query_for_attempt(request.customer_question, revised, public=True), request.customer_question)

    def test_general_generation_receives_request_language_without_loading_model(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        import torch
        captured = []
        class Batch(dict):
            @property
            def input_ids(self): return self['input_ids']
            def to(self, device): return self
        class Tokenizer:
            eos_token_id = 0
            def apply_chat_template(self, messages, **kwargs):
                captured.append(messages)
                return 'prompt'
            def __call__(self, *args, **kwargs):
                return Batch(input_ids=torch.zeros((1,2), dtype=torch.long))
            def decode(self, *args, **kwargs):
                return 'Hello.'  # Final uses natural language, not the old JSON envelope.
        model = SimpleNamespace(device='cpu', generate=lambda **kw: torch.zeros((1,3), dtype=torch.long))
        with patch.object(app, 'load_model', return_value=(Tokenizer(),model)), \
             patch.object(app, 'generation_session', return_value=nullcontext()), \
             patch.object(app, 'save_question_router_debug'):
            result = app.general_local_chat_answer(app.DraftRequest(customer_question='Hello, how are you?'))
        payload = json.loads(captured[0][1]['content'])
        self.assertEqual(payload['answer_language'], 'English')
        self.assertEqual(result['customer_reply'], 'Hello.')


if __name__ == '__main__':
    unittest.main()
