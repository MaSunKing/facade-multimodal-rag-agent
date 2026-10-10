import unittest
from unittest.mock import patch
from backend.sales.answer_review import support_pack, validate_review, draft_context
from backend.sales.output_contract import OutputContractError
from backend.sales.answer_graph import build_customer_answer_graph


class ReviewTests(unittest.TestCase):
    def test_unsupported_context_cannot_become_context_only_revision(self):
        verdict=validate_review({'decision':'revise','status':'unsupported',
            'supported_evidence_ids':['T1'],'revision':{'requirement':'比较甲乙',
                'reason':'当前只包含背景，缺少所问做法','evidence_ids':['T1']}},
            {'question':'比较甲乙','evidence':[{'evidence_id':'T1','text':'背景'}]})
        self.assertEqual(verdict['decision'],'limited')
        self.assertEqual(verdict['missing_aspects'],['比较甲乙'])
    def test_quote_keys_must_be_supported_not_merely_in_context(self):
        with self.assertRaises(OutputContractError):
            validate_review({'decision':'pass','status':'supported',
                'supported_evidence_ids':['T1'],'support_spans':{'T2':['乙']}},
                {'evidence':[{'evidence_id':'T1','text':'甲'},{'evidence_id':'T2','text':'乙'}]})

    def test_actual_post_draft_callback_preserves_task_and_source_ids(self):
        from backend.app import DraftRequest,_CustomerAnswerWorkflowCallbacks as C
        from backend.sales.tool_planner import ToolPlan
        from backend.sales.task_updates import bind_task_instructions
        from backend.tests.sales.test_audit_error_contract import text_audit_model
        request=DraftRequest(customer_question='再增加乙',task_instructions=['介绍甲'],use_online_search=True)
        bind_task_instructions(request)
        request._prepared_evidence=[{'evidence_id':'T1','text':'甲资料'},{'evidence_id':'T2','text':'乙资料'}]
        request._source_results={'company_rag':{}}
        with text_audit_model([]),patch('backend.app.evaluate_task_support',return_value={
                'decision':'pass','status':'supported','supported_evidence_ids':['T1','T2']}) as model:
            result=C.review_answer(request,ToolPlan(tools=['company_rag']),
                {'customer_reply':'甲与乙','citations':[{'evidence_id':'T1',
                    'content':'不得传给审核的原文','text':'另一个隐含正文'}],
                 'meta':{'model_used':True}},1)
        payload=model.call_args.args[1]
        self.assertIn('介绍甲',payload['question'])
        self.assertIn('再增加乙',payload['question'])
        self.assertNotIn('evidence',payload)
        self.assertEqual(payload['review_policy'],'task_coverage_only')
        self.assertEqual([r['evidence_id'] for r in request._prepared_evidence],['T1','T2'])
        self.assertNotIn('image_paths',payload)
        self.assertEqual(payload['final_draft']['citations'],[{'evidence_id':'T1','source_reference_count':1}])
        self.assertNotIn('不得传给审核的原文',str(payload))
        self.assertEqual(model.call_args.kwargs['purpose'],'answer_audit')
        self.assertEqual(payload['wire_policy'],'compact_navigation_not_fact_rewrite')
        self.assertIn('company_capabilities',payload)
        self.assertEqual(result['meta']['answer_review']['decision'],'pass')

    def test_revision_executes_once_without_tools_and_is_rechecked(self):
        import json
        from backend.app import DraftRequest,_CustomerAnswerWorkflowCallbacks as C
        from backend.sales.tool_planner import ToolPlan
        from backend.tests.sales.test_audit_error_contract import text_audit_model
        request=DraftRequest(customer_question='厚度？')
        request._prepared_evidence=[{'evidence_id':'T1','text':'厚度20毫米。',
            'citations':[{'document_name':'规格','source_page':1}]}]
        response={'customer_reply':'遗漏','key_points':['不受支持的旧主张'],
            'missing_information':['旧缺口'],'meta':{'model_used':True,'answer_review':{
            'decision':'revise','status':'supported','supported_evidence_ids':['T1'],
            'revision':{'requirement':'厚度','reason':'漏写','evidence_ids':['T1']}}}}
        revised='厚度20毫米。'
        approved={**response,'answerable':True,'customer_reply':revised,'key_points':[],'missing_information':[],
            'meta':{'model_used':True,'answer_review':{'decision':'pass','status':'supported'}}}
        with (text_audit_model([revised]),
              patch.object(C,'review_answer',return_value=approved) as review,
              patch.object(C,'replan_from_feedback',side_effect=AssertionError('no tools'))):
            result=C.finish_review(request,ToolPlan(tools=['company_rag']),response)
        self.assertEqual(request._answer_revision_count,1)
        review.assert_called_once()
        self.assertEqual(review.call_args.args[2]['key_points'],[])
        self.assertEqual(review.call_args.args[2]['missing_information'],[])
        self.assertEqual(result['customer_reply'],'厚度20毫米。')
        self.assertEqual(result['key_points'],[])
        self.assertEqual(result['missing_information'],[])

    def test_failed_review_never_publishes_draft(self):
        from backend.app import DraftRequest,_CustomerAnswerWorkflowCallbacks as C
        from backend.sales.tool_planner import ToolPlan
        request=DraftRequest(customer_question='Thickness?')
        response={'customer_reply':'Unchecked invented claim','meta':{'model_used':True,
            'answer_review':{'decision':'limited','error_code':'AUDIT_FAILED','status':'uncertain'}}}
        with patch('backend.app.load_model',side_effect=AssertionError('no revision on audit error')):
            result=C.finish_review(request,ToolPlan(tools=['company_rag']),response)
        self.assertNotIn('invented',result['customer_reply'])
        self.assertFalse(result['answerable'])

    def test_revision_preserves_cancellation(self):
        from backend.app import DraftRequest,_CustomerAnswerWorkflowCallbacks as C
        from backend.sales.tool_planner import ToolPlan
        from backend.request_budget import RequestCancelled
        request=DraftRequest(customer_question='厚度')
        request._prepared_evidence=[{'evidence_id':'T1','text':'厚度20毫米。'}]
        response={'customer_reply':'草稿','meta':{'answer_review':{
            'decision':'revise','status':'partial','supported_evidence_ids':['T1']}}}
        with patch('backend.app.load_model',side_effect=RequestCancelled('request_cancelled')):
            with self.assertRaises(RequestCancelled):
                C.finish_review(request,ToolPlan(tools=['company_rag']),response)
        self.assertEqual(request._answer_revision_count,1)

    def test_spent_revision_cannot_generate_again(self):
        from backend.app import DraftRequest,_CustomerAnswerWorkflowCallbacks as C
        from backend.sales.tool_planner import ToolPlan
        request=DraftRequest(customer_question='厚度')
        request._answer_revision_count=1
        response={'customer_reply':'未通过的修改','meta':{'answer_review':{
            'decision':'revise','status':'partial','supported_evidence_ids':['T1']}}}
        with patch('backend.app.load_model',side_effect=AssertionError('no second revision')):
            result=C.finish_review(request,ToolPlan(tools=['company_rag']),response)
        self.assertFalse(result['answerable'])
        self.assertEqual(request._answer_revision_count,1)

    def test_background_ids_do_not_spend_revision_when_context_unsupported(self):
        from backend.app import DraftRequest,_CustomerAnswerWorkflowCallbacks as C
        from backend.sales.tool_planner import ToolPlan
        request=DraftRequest(customer_question='比较甲乙做法')
        response={'customer_reply':'未支持的草稿','meta':{'answer_review':{
            'decision':'limited','status':'unsupported','supported_evidence_ids':['T1'],
            'missing_aspects':['比较甲乙做法']}}}
        with patch('backend.app.load_model',side_effect=AssertionError('no unsupported-context revision')):
            result=C.finish_review(request,ToolPlan(tools=['company_rag']),response)
        self.assertFalse(result['answerable'])
        self.assertEqual(result['missing_information'],['比较甲乙做法'])
        self.assertEqual(request._answer_revision_count,0)

    def test_draft_map_identity_is_preserved_without_mutating_rows(self):
        values={'T4':{'text':'真实片段'}}
        rows=draft_context(values)
        self.assertEqual(rows[0]['evidence_id'],'T4')
        self.assertNotIn('evidence_id',values['T4'])

    def payload(self, full=True):
        return {'question':'比较甲和乙的安装方式','evidence':[{'evidence_id':'T1','text':'甲使用机械连接。'}],
            'review_scope':{'full_context':full},'actual_tools':['company_rag']}

    def recovery(self):
        return {'decision':'recover','status':'partial','supported_evidence_ids':['T1'],
            'missing_aspects':['乙的安装方式'],'recovery_need':'lookup',
            'proof':{'requirement':'安装方式','basis':'evidence_gap','reason':'仅有甲连接记录',
                     'next_query':'乙的安装方式','evidence_ids':['T1']}}

    def test_complete_review_can_recover(self):
        self.assertEqual(validate_review(self.recovery(),self.payload())['decision'],'recover')

    def test_excerpt_absence_cannot_recover(self):
        result=validate_review(self.recovery(),self.payload(False))
        self.assertEqual(result['decision'],'limited')
        # Canonical result may be validated twice by the shared model contract.
        self.assertEqual(validate_review(result,self.payload(False))['decision'],'limited')

    def test_added_requirement_cannot_recover(self):
        value=self.recovery();value['proof']['requirement']='防火寿命'
        self.assertEqual(validate_review(value,self.payload())['decision'],'limited')

    def test_recovery_requires_proof(self):
        value=self.recovery();value.pop('proof')
        with self.assertRaises(OutputContractError):validate_review(value,self.payload())

    def test_revision_requires_supported_witness(self):
        value={'decision':'revise','status':'partial','supported_evidence_ids':['T1'],
               'revision':{'requirement':'安装方式','reason':'甲连接记录漏写','evidence_ids':['T1']}}
        self.assertEqual(validate_review(value,self.payload())['decision'],'revise')
        value['revision']['evidence_ids']=[]
        with self.assertRaises(OutputContractError):validate_review(value,self.payload())

    def test_unknown_id_not_repaired(self):
        value=self.recovery();value['supported_evidence_ids']=['W99']
        with self.assertRaises(OutputContractError):validate_review(value,self.payload())

    def test_comparison_pass_requires_verifiable_quote(self):
        payload=self.payload();payload['planned_task']={'operation':'comparison'}
        value={'decision':'pass','status':'supported','supported_evidence_ids':['T1']}
        with self.assertRaises(OutputContractError):validate_review(value,payload)
        value['support_spans']={'T1':['甲使用机械连接。']}
        self.assertEqual(validate_review(value,payload)['decision'],'pass')

    def test_company_visual_recovery_is_not_executable(self):
        value=self.recovery();value['proof']['basis']='visual_input';value['recovery_need']='visual'
        payload=self.payload();payload['company_visual_policy']={'company_only':True}
        result=validate_review(value,payload)
        self.assertEqual(result['decision'],'limited')

    def test_full_pack_preserves_original(self):
        rows=[{'evidence_id':'X','text':'不适用于乙。甲厚度20mm，条件为干燥基层。'}]
        pack,scope=support_pack(rows,'甲厚度','')
        self.assertEqual(pack[0]['text'],rows[0]['text'])
        self.assertTrue(scope['full_context'])

    def test_long_atomic_unit_not_cut(self):
        rows=[{'evidence_id':'X','text':'a'*200}]
        pack,scope=support_pack(rows,'a','',per_row=50)
        self.assertFalse(scope['full_context'])
        self.assertEqual(pack[0]['text'],'')
        self.assertEqual(rows[0]['text'],'a'*200)

    def test_graph_reviews_after_generation_and_recovers_once(self):
        events=[]
        class Callbacks:
            def plan_tools(self,request):return {'tools':['company_rag']}
            def answer_planned(self,request,plan):
                events.append('draft');return {'customer_reply':'draft','meta':{}}
            def review_answer(self,request,plan,response,round_number):
                events.append('audit');return response
            def plan_retry(self,request,plan,response):
                events.append('replan');return {'tools':['company_rag'],'retrieval_query':'new'}
            def finish_review(self,request,plan,response):
                events.append('finish');return response
        state=build_customer_answer_graph(Callbacks()).invoke({'request':object()})
        self.assertEqual(events,['draft','audit','replan','draft','audit','finish'])
        self.assertEqual(state['response']['meta']['orchestration']['tool_rounds'],2)
        self.assertEqual(state['response']['meta']['orchestration']['max_answer_revisions'],1)


if __name__=='__main__':unittest.main()
