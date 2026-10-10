import unittest
from unittest.mock import patch
from backend.sales.evidence_gate import evidence_preview, relevance_risk, validate_evaluation,question_excerpt
from backend.sales.answer_graph import build_customer_answer_graph
from backend.sales.staged_execution import WorkflowStep
from backend.sales.tool_planner import ToolPlan
from backend.app import DraftRequest, _CustomerAnswerWorkflowCallbacks


class EvidenceGateTests(unittest.TestCase):
    def review(self,request,plan,sources,round_number=1):
        from backend.tests.sales.test_audit_error_contract import text_audit_model
        request._source_results=sources
        draft={'answerable':True,'customer_reply':'Internal draft','citations':[],
               'meta':{'model_used':True}}
        with text_audit_model([]):
            reviewed=_CustomerAnswerWorkflowCallbacks.review_answer(request,plan,draft,round_number)
            return _CustomerAnswerWorkflowCallbacks.finish_review(request,plan,reviewed)

    def test_technical_failure_is_not_insufficient_evidence_or_semantic_replan(self):
        request=DraftRequest(customer_question='Who manufactures this panel?')
        plan=ToolPlan(tools=['company_rag'])
        with patch('backend.app.evaluate_task_support',return_value={
                'status':'uncertain','error_code':'AUDIT_FAILED','reason':'support_audit_failed'}):
            result=self.review(request,plan,
                {'company_rag':{'text':'Manufacturer: Example Ltd.'}},1)
        self.assertEqual(result['meta']['mode'],'evidence_audit_failed')
        self.assertIn('could not be completed',result['customer_reply'])
        self.assertNotIn('insufficient',result['customer_reply'])
        with patch('backend.app.evaluate_task_support',side_effect=AssertionError('no semantic replan')):
            self.assertIsNone(_CustomerAnswerWorkflowCallbacks.plan_retry(request,plan,result))

    def test_audit_payload_does_not_repeat_full_plan_or_retrieval_metadata(self):
        request=DraftRequest(customer_question='Panel thickness?')
        with patch('backend.app.evaluate_task_support',return_value={
                'decision':'pass','status':'supported','supported_evidence_ids':['E1']}) as model:
            self.review(request,
                ToolPlan(tools=['company_rag'],answer_aspects=['Panel thickness']),
                {'company_rag':{'text':'Panel thickness 20 mm.','citations':[
                    {'document_name':'Specification','page_number':2,'content':'repeated full text'}]}},1)
        payload=model.call_args.args[1]
        for field in ('previous_plan','available','risk_signals','tried_tools'):
            self.assertNotIn(field,payload)
        self.assertNotIn('evidence',payload)
        self.assertEqual(payload['review_policy'],'task_coverage_only')
        self.assertNotIn('Panel thickness 20 mm.',str(payload))
        self.assertIn('Panel thickness 20 mm.',str(request._prepared_evidence))

    def test_audit_preview_can_keep_late_answer_without_mutating_source(self):
        text='General background. '*80+'TargetPanel thickness 13 mm.'
        excerpt=question_excerpt(text,'TargetPanel thickness?')
        self.assertIn('13 mm',excerpt)
        self.assertEqual(text.count('13 mm'),1)
    def test_unrelated_sources_are_suspicious_not_gold(self):
        rows = evidence_preview([{'text_evidence':[{'text':'真岩石是一种外墙板材'}]}])
        risk = relevance_risk('无尽冬日是什么', ToolPlan(tools=['company_rag']), rows)
        self.assertTrue(risk['suspicious'])
        self.assertFalse(risk['signal_is_semantic_gold'])

    def test_related_partial_evidence_is_not_automatically_rejected(self):
        rows = evidence_preview([{'text':'Panel X installation uses compatible anchors.'}])
        self.assertFalse(relevance_risk('Panel X installation instructions', ToolPlan(), rows)['suspicious'])
        self.assertEqual(validate_evaluation({'status':'partial', 'supported_evidence_ids':['E1'],
            'missing_aspects':['sealant']}, rows)['status'], 'partial')

    def test_unknown_or_missing_support_ids_rejected(self):
        rows = [{'evidence_id':'E1','text':'A'}]
        for ids in [[], ['fake']]:
            with self.assertRaises(ValueError):
                validate_evaluation({'status':'supported','supported_evidence_ids':ids}, rows)

    def test_missing_source_constraint_is_uncertain_not_permission(self):
        result=validate_evaluation({'status':'unsupported'}, [])
        self.assertEqual(result['source_constraint'], 'uncertain')

    def test_unapproved_web_replan_is_removed(self):
        plan = ToolPlan(tools=['public_web_search'], requires_public_web=True)
        safe = _CustomerAnswerWorkflowCallbacks.validate_plan(DraftRequest(customer_question='当前报价'), plan)
        self.assertEqual(safe.tools, [])

    def test_gate_stops_first_generation_then_replans_once(self):
        calls=[]
        class Callbacks:
            def plan_tools(self, request): return ToolPlan(tools=['company_rag'],retrieval_query='old')
            def open_steps(self, request, plan):
                yield WorkflowStep('company_rag', lambda: {'text':'source'})
                yield WorkflowStep('compose_evidence')
                yield WorkflowStep('generate_answer', lambda: calls.append('generate'))
                return {'answerable':True,'meta':{}}
            def evaluate_sources(self, request, plan, sources, round_number):
                calls.append(('evaluate',round_number))
                self.sources=sources
                if round_number == 1:
                    return {'answerable':False,'meta':{'mode':'blocked'}}
            def plan_retry(self, request, plan, response):
                return plan.model_copy(update={'retrieval_query':'new'})
        callbacks=Callbacks()
        result=build_customer_answer_graph(callbacks).invoke({'request':{}})['response']
        self.assertEqual(calls,[('evaluate',1),('evaluate',2),'generate'])
        self.assertEqual(result['meta']['orchestration']['tool_rounds'],2)
        self.assertEqual(callbacks.sources['company_rag'], {'text':'source'})

    def test_second_unsupported_round_never_generates_or_loops(self):
        calls=[]
        class Callbacks:
            def plan_tools(self, request): return ToolPlan(tools=['company_rag'])
            def open_steps(self, request, plan):
                yield WorkflowStep('company_rag',lambda:{'text':'unrelated'})
                yield WorkflowStep('compose_evidence')
                yield WorkflowStep('generate_answer',lambda:calls.append('generate'))
                return {'meta':{}}
            def evaluate_sources(self,request,plan,sources,round_number):
                calls.append(round_number)
                return {'answerable':False,'meta':{}}
            def plan_retry(self,request,plan,response):
                return plan.model_copy(update={'retrieval_query':plan.retrieval_query+'x'})
        result=build_customer_answer_graph(Callbacks()).invoke({'request':{}})['response']
        self.assertEqual(calls,[1,2])
        self.assertFalse(result['answerable'])

    def test_no_budget_fails_closed_without_loading_model(self):
        request=DraftRequest(customer_question='陌生实体是什么')
        with patch('backend.sales.task_support.audit_time_available',return_value=False), patch('backend.app.load_model',side_effect=AssertionError):
            result=_CustomerAnswerWorkflowCallbacks.evaluate_sources(request, ToolPlan(tools=['company_rag']),
                {'company_rag':{'text':'真岩石外墙材料'}},1)
        self.assertFalse(result['answerable'])

    def test_one_keyword_cannot_bypass_semantic_audit(self):
        request=DraftRequest(customer_question='无相神器 不吃压力')
        with patch('backend.app.evaluate_task_support',return_value={'decision':'limited','status':'unsupported',
            'source_constraint':'uncertain','reason':'entity mismatch','supported_evidence_ids':[]}) as audit:
            result=self.review(request,ToolPlan(tools=['company_rag']),
                {'company_rag':{'text':'检查均匀静态压力抗弯检测试验报告。'}},1)
        audit.assert_called_once()
        self.assertFalse(result['answerable'])

    def test_successful_audit_does_not_consume_recovery(self):
        request=DraftRequest(customer_question='Panel thickness?')
        with patch('backend.app.evaluate_task_support',return_value={'status':'supported',
            'supported_evidence_ids':['E1']}), patch('backend.app.reserve_recovery') as reserve:
            result=_CustomerAnswerWorkflowCallbacks.evaluate_sources(request,ToolPlan(tools=['company_rag']),
                {'company_rag':{'text':'Panel thickness 20 mm.'}},1)
        self.assertIsNone(result)
        reserve.assert_not_called()

    def test_recovered_evidence_is_reaudited_without_third_execution(self):
        request=DraftRequest(customer_question='Panel thickness?')
        with patch('backend.app.evaluate_task_support',return_value={'decision':'limited','status':'unsupported',
            'supported_evidence_ids':[],'revised_plan':{'tools':['general_chat']}}), patch('backend.app.reserve_recovery') as reserve:
            result=self.review(request,ToolPlan(tools=['company_rag']),
                {'company_rag':{'text':'Panel colour is blue.'}},2)
        self.assertFalse(result['answerable'])
        reserve.assert_not_called()


if __name__ == '__main__': unittest.main()
