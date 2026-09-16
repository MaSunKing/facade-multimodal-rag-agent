"""Search-field length is not the limit of the user's semantic request."""
from contextlib import nullcontext
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pydantic import ValidationError

import backend.app as app
from backend.sales.tool_planner import (
    MAX_RETRIEVAL_QUERY_CHARACTERS, ToolPlan,
    bounded_fallback_retrieval_query, fallback_plan,
)


class RetrievalQueryBoundaryTests(unittest.TestCase):
    def setUp(self):
        with app._planner_cache_lock:
            app._planner_cache.clear()

    def test_query_field_stays_strict_at_300_characters(self):
        ToolPlan(retrieval_query='a'*300)
        with self.assertRaises(ValidationError):
            ToolPlan(retrieval_query='a'*301)

    def test_short_fallback_query_keeps_complete_terms(self):
        self.assertEqual(bounded_fallback_retrieval_query('设备  不得\n露天存放'), '设备 不得 露天存放')

    def test_long_fallback_does_not_use_a_question_prefix(self):
        question='Background information. '*60+'Only consider 2027; must not assume certification.'
        request=app.DraftRequest(customer_question=question, memory_enabled=False)
        plan=fallback_plan(has_documents=True, has_image=False, facade_related=False,
                           web_requested=False, retrieval_query=question,
                           document_scope='local_lookup')
        self.assertEqual(plan.retrieval_query, '')
        self.assertEqual(request.customer_question, question)
        self.assertEqual(app.question_plan_from_tool_plan(plan, question)['retrieval_query'], question)
        self.assertEqual(plan.tools, ['customer_documents'])

    def test_fallback_prefers_complete_existing_anchors(self):
        plan=fallback_plan(has_documents=True,has_image=False,facade_related=False,
            web_requested=False,retrieval_query='Background '*100,
            target_terms=['Anchor-A', '2027', 'must not assume certification', 'Anchor-A'],
            document_scope='local_lookup')
        self.assertEqual(plan.retrieval_query, 'Anchor-A 2027 must not assume certification')

    def test_overlong_anchors_are_not_cut_mid_term(self):
        query=bounded_fallback_retrieval_query('背景'*500,['X'*301,'not permitted','2027'])
        self.assertEqual(query,'not permitted 2027')
        self.assertLessEqual(len(query),MAX_RETRIEVAL_QUERY_CHARACTERS)

    def test_long_questions_have_safe_fallbacks_in_multiple_languages(self):
        for question in ['说明'*1490+'不得忽略最后条件', 'review '*420+'do not omit the last condition']:
            with self.subTest(question_length=len(question)):
                request=app.DraftRequest(customer_question=question,memory_enabled=False)
                plan=app.fallback_customer_tool_plan(request,has_documents=False,has_image=False)
                self.assertLessEqual(len(plan.retrieval_query),300)
                self.assertEqual(request.customer_question,question)
                self.assertFalse(plan.requires_public_web)

    def test_long_attachment_request_reaches_semantic_planner_before_fallback(self):
        question='请结合附件分析：'+'需要综合核对相关约束和实施条件。'*40+'不得忽略最后的2027年度限制。'
        session=SimpleNamespace(documents=[SimpleNamespace(file_name='notes.docx', visuals=[],
            source_type='docx', chunks=[])])
        request=app.DraftRequest(customer_question=question,document_session_id='test',memory_enabled=False)
        with patch.object(app,'get_session',return_value=session), \
             patch.object(app,'load_model',side_effect=RuntimeError('offline planner')) as model:
            plan=app.plan_customer_tools(request)
        model.assert_called_once()
        self.assertIn('customer_documents',plan.tools)
        self.assertEqual(request.customer_question,question)
        self.assertLessEqual(len(plan.retrieval_query),300)

    def test_semantic_planner_sees_complete_request_and_returns_independent_short_query(self):
        import torch
        question='Please review the uploaded plan. '+'Background context about a scheduling decision. '*18+'Do not omit the 2027 storage restriction.'
        captured=[]
        class Batch(dict):
            @property
            def input_ids(self): return self['input_ids']
            def to(self,device): return self
        class Tokenizer:
            eos_token_id=0
            def apply_chat_template(self,messages,**kwargs):
                captured.append(messages)
                return json.dumps(messages,ensure_ascii=False)
            def __call__(self,prompt,**kwargs):
                return Batch(input_ids=torch.zeros((1,2),dtype=torch.long))
            def decode(self,ids,**kwargs):
                return json.dumps({'tools':['customer_documents'],'task_type':'procedure',
                    'document_scope':'whole_document','document_visual_required':False,
                    'retrieval_query':'scheduling capacity 2027 storage restriction',
                    'search_targets':['scheduling','capacity','2027','storage restriction'],
                    'answer_goals':['What capacity and storage restrictions apply in 2027?']})
        model=SimpleNamespace(device='cpu',generate=lambda **kwargs:torch.zeros((1,3),dtype=torch.long))
        document=SimpleNamespace(file_name='notes.docx',visuals=[],source_type='docx',
            chunks=[{'kind':'paragraph','text':'The uploaded scheduling plan explains capacity, staffing, storage restrictions and annual limits.'}])
        request=app.DraftRequest(customer_question=question,document_session_id='test',memory_enabled=False)
        with patch.object(app,'get_session',return_value=SimpleNamespace(documents=[document])), \
             patch.object(app,'load_model',return_value=(Tokenizer(),model)), \
             patch.object(app,'generation_session',return_value=nullcontext()), \
             patch.object(app,'save_question_router_debug'):
            plan=app.plan_customer_tools(request)
        planner_input=json.loads(captured[0][1]['content'])
        self.assertEqual(planner_input['question'],question)
        self.assertEqual(planner_input['retrieval_query_contract']['max_characters'],300)
        self.assertEqual(plan.retrieval_query,'scheduling capacity 2027 storage restriction')
        self.assertEqual(plan.tools,['customer_documents'])
        self.assertEqual(request.customer_question,question)


if __name__=='__main__':
    unittest.main()
