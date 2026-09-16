import time
import json
import unittest
from unittest.mock import Mock, patch
from backend.request_budget import RequestBudget, budget_scope, reserve_recovery
from backend.sales.staged_execution import WorkflowStep, WorkflowCursor
from backend.sales.answer_graph import build_customer_answer_graph
from backend.sales.runtime_status import ExecutionStatus, current_execution
from backend.sales.tool_planner import ToolPlan
import backend.app as app


class NodeRecoveryTests(unittest.TestCase):
    def test_visual_geometry_does_not_pollute_search_text(self):
        from backend.documents.visual_layout import compact_visual_metadata
        metadata={'title':'Sample', 'raster_region_boxes':[[0,0,1,1]],
                  'raster_layout_groups':{'groups_left_to_right':[], 'excluded_object_count':1}}
        self.assertEqual(compact_visual_metadata(metadata), {'title':'Sample'})
        self.assertIn('native_layout_hint', compact_visual_metadata(metadata, include_layout=True))
        self.assertIn('raster_region_boxes', metadata)
        self.assertEqual(compact_visual_metadata(None, include_layout=True), {})

    def test_incomplete_json_is_not_misreported_as_bad_evidence(self):
        from backend.sales.runtime_status import classify
        info=classify(ValueError('截断 JSON 尚未完整输出回答与引用'),'validate_answer')
        self.assertEqual(info['code'],'OUTPUT_INVALID_JSON')
    def test_layout_is_geometric_not_a_fixed_picture_count(self):
        from backend.documents.visual_layout import raster_layout_groups
        boxes=[[.05,.1,.4,.8],[.5,.1,.7,.3],[.75,.1,.95,.3],
               [.5,.4,.95,.6],[.1,.9,.9,.91]]
        result=raster_layout_groups(boxes)
        self.assertEqual([g['row_raster_counts'] for g in result['groups_left_to_right']],[[1],[2,1]])
        self.assertEqual(result['excluded_object_count'],1)
        self.assertFalse(result['coverage_complete'])
        self.assertEqual(raster_layout_groups([[0,0,1,1]])['groups_left_to_right'],[])

    def test_prose_packing_preserves_rank_not_table_endpoints(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs): return messages[-1]['content']
            def __call__(self,text,**kwargs): return {'input_ids':text.split()}
        payload={'attachment_context':{'global_document_question':True},'evidence':[
            {'evidence_id':'I','document_name':'manual.pdf','evidence_scope':'document_index','text':'index'},
            {'evidence_id':'P','document_name':'manual.pdf','evidence_scope':'content','text':'Relevant native paragraph'},
            {'evidence_id':'T','document_name':'manual.pdf','evidence_scope':'content','text':'[TABLE parser=camelot-stream] duplicate prose'},
        ]}
        text,audit=app.compact_grounded_payload_for_generation(payload,Tokenizer(),max_prompt_tokens=1000)
        self.assertEqual(audit['kept_evidence_ids'],['P','T','I'])
        self.assertTrue(audit['canonical_evidence_preserved'])

    def test_percent_spelling_is_not_ratio_derivation(self):
        result={'customer_reply':'比例为17.5%。','citations':[{'evidence_id':'U1'}]}
        for source in ['Share is 17.5 percent.','Share is 17.5 per cent.','百分之17.5']:
            audit=app.evidence_support_audit(result,{'U1':{'text':source}})
            self.assertEqual(audit['unsupported_numeric_claims'],[],source)
        for source in ['Count is 17.5.','Ratio is 0.175.','Share is 15 percent.']:
            audit=app.evidence_support_audit(result,{'U1':{'text':source}})
            self.assertIn('17.5%',audit['unsupported_numeric_claims'],source)

    def test_shared_cap_deadline_and_per_stage_cap(self):
        with budget_scope(RequestBudget(time.monotonic()+90)):
            self.assertTrue(reserve_recovery('public_web_search','retry'))
            self.assertFalse(reserve_recovery('public_web_search','retry'))
            self.assertFalse(reserve_recovery('customer_documents','broaden'))
            self.assertFalse(reserve_recovery('generate_answer','repack'))
        with budget_scope(RequestBudget(time.monotonic()+5)):
            self.assertFalse(reserve_recovery('public_web_search','retry'))

    def test_transient_read_retries_but_not_gpu_or_auth(self):
        for node, exc, count in [('company_rag',TimeoutError(),2),
                                 ('generate_answer',TimeoutError(),1),
                                 ('customer_documents',PermissionError(),1)]:
            operation = Mock(side_effect=[exc, 42])
            def flow():
                try:
                    return (yield WorkflowStep(node,operation))
                except Exception:
                    return 'failed'
            with budget_scope(RequestBudget(time.monotonic()+90)):
                cursor=WorkflowCursor(flow());cursor.advance();cursor.execute()
                self.assertEqual(operation.call_count,count)
                self.assertEqual(cursor.result,42 if count==2 else 'failed')

    def test_actual_retry_callback_accepts_graph_dictionary(self):
        request=app.DraftRequest(customer_question='介绍附件',document_session_id='test')
        plan=ToolPlan(tools=['customer_documents'],document_scope='local_lookup')
        result={'answerable':False,'meta':{'model_used':False,'mode':'customer_document_index_only'}}
        with patch.object(app,'get_session',return_value=Mock(documents=[Mock()])):
            revised=app._CustomerAnswerWorkflowCallbacks.plan_retry(request,plan,result)
        self.assertEqual(revised.document_scope,'whole_document')
        result['answerable']=True
        self.assertIsNone(app._CustomerAnswerWorkflowCallbacks.plan_retry(request,plan,result))

    def test_real_callback_in_staged_graph_retrieves_once_more(self):
        class Callbacks(app._CustomerAnswerWorkflowCallbacks):
            def plan_tools(self,request):
                return ToolPlan(tools=['customer_documents'],document_scope='local_lookup')
            def validate_plan(self,request,plan): return plan
            def check_step(self,*args): pass
            def open_steps(self,request,plan):
                yield WorkflowStep('customer_documents')
                return {'answerable':plan.document_scope=='whole_document',
                        'meta':{'model_used':False,'mode':'customer_document_index_only'}}
        request=app.DraftRequest(customer_question='介绍附件',document_session_id='test')
        budget=RequestBudget(time.monotonic()+90)
        with budget_scope(budget), patch.object(app,'get_session',return_value=Mock(documents=[Mock()])):
            result=build_customer_answer_graph(Callbacks()).invoke({'request':request})['response']
        self.assertTrue(result['answerable'])
        self.assertEqual(result['meta']['orchestration']['tool_rounds'],2)
        self.assertEqual(len(budget.recoveries),1)
        self.assertNotEqual(result['meta']['orchestration']['retry_reason'],'retry_policy_failed_closed')

    def test_retry_failure_is_observable_without_losing_first_answer(self):
        class Callbacks:
            def plan_tools(self,request): return ToolPlan(tools=['general_chat'])
            def answer_planned(self,request,plan): return {'answerable':True,'customer_reply':'retained','meta':{}}
            def plan_retry(self,*args): raise AttributeError('private detail')
        status=ExecutionStatus(); token=current_execution.set(status)
        try:
            result=build_customer_answer_graph(Callbacks()).invoke({'request':{}})['response']
        finally:
            current_execution.reset(token)
        self.assertEqual(result['customer_reply'],'retained')
        self.assertEqual(status.errors[0]['code'],'RETRY_POLICY_FAILED')
        self.assertNotIn('private',str(status.errors))


if __name__=='__main__': unittest.main()
