"""CPU/protocol regressions. No external requests or Qwen loading."""
import unittest
from unittest.mock import patch
from backend.sales.model_contract import validate_audit, validate_plan, repair_plan_query_fields
from backend.sales.output_contract import OutputContractError
from backend.sales.task_support import supported_evidence_rows, supported_web_sources, unavailable_web_response
from backend.sales.visual_policy import generation_evaluation
from backend.sales.web_requirements import retrieve_web_requirements
from backend.sales.runtime_status import ExecutionStatus, current_execution
from backend.sales.stage_details import begin, end
from scripts.run_agent_eval_100 import Client, response_status, TransportFailure, run_case


class GenericRepairs(unittest.TestCase):
    def test_legacy_query_objects_repaired_without_changing_scope(self):
        value={'tools':['customer_documents'],'document_retrieval_queries':[
            {'query':'甲的条件','targets':['甲']},{'query':'乙的条件','targets':['乙']}]}
        repaired=repair_plan_query_fields(value)
        self.assertEqual(repaired['document_retrieval_queries'],['甲的条件','乙的条件'])
        self.assertEqual([r['targets'] for r in repaired['retrieval_requests']],[['甲'],['乙']])
        self.assertIsInstance(value['document_retrieval_queries'][0],dict)
        validate_plan(repaired,require_scoped_requests=True)

    def test_model_already_split_queries_not_program_semantic_split(self):
        value={'tools':['public_web_search'],'retrieval_queries':['对象甲的条件是什么','对象乙的条件是什么']}
        self.assertEqual([r['query'] for r in repair_plan_query_fields(value)['retrieval_requests']],value['retrieval_queries'])
        multi={**value,'tools':['company_rag','public_web_search']}
        self.assertNotIn('retrieval_requests',repair_plan_query_fields(multi))
        bad={'tools':['customer_documents'],'document_retrieval_queries':[{'query':'甲','owner':'someone'}]}
        self.assertEqual(repair_plan_query_fields(bad),bad)

    def test_enterprise_assembly_reaches_rerank_with_real_ids(self):
        import backend.app as app
        from backend.sales.tool_planner import ToolPlan
        plan=ToolPlan(tools=['company_rag'],knowledge_categories=['02_product_information'],
                      company_operation='summary',retrieval_query='概括资料')
        request=app.DraftRequest(customer_question='概括资料')
        retrieval={'text_evidence':[{'evidence_id':'T7','text':'登记产品介绍'}],
                   'visual_assets':[], 'meta':{}}
        with patch.object(app,'retrieve_catalog_evidence',return_value=[]), patch.object(app,'load_retriever'):
            generator=app._run_facade_rag_answer.steps(request,tool_plan=plan)
            step=next(generator)
            for _ in range(12):
                if step.node=='rerank_evidence':
                    self.assertEqual(step.args[1][0]['evidence_id'],'T7')
                    break
                step=generator.send(retrieval if step.node=='company_rag' else None)
            else: self.fail('assembly did not reach evidence reranking')
            generator.close()

    def test_scope_spans_are_exact_and_preserve_provenance(self):
        rows=[{'evidence_id':'D1','text':'旧版：厚度18mm。新版：厚度20mm。','document_id':'doc'}]
        audit={'status':'supported','supported_evidence_ids':['D1'],
               'support_spans':{'D1':['新版：厚度20mm。']}}
        validate_audit(audit,['D1'],evidence_rows=rows)
        selected=supported_evidence_rows(rows,audit)
        self.assertEqual(selected[0]['text'],'新版：厚度20mm。')
        self.assertEqual(selected[0]['document_id'],'doc')
        self.assertEqual(rows[0]['text'],'旧版：厚度18mm。新版：厚度20mm。')
        from backend.sales.context_engine import validate_packed_evidence
        self.assertTrue(validate_packed_evidence(selected,selected)['valid'])
        self.assertFalse(validate_packed_evidence(selected,[{**selected[0],'text':'新版：厚度18mm。'}])['valid'])
        self.assertIn('support_spans',generation_evaluation(audit,0))
        audit['support_spans']['D1']=['新版：厚度30mm。']
        with self.assertRaises(OutputContractError): validate_audit(audit,['D1'],evidence_rows=rows)

    def test_time_row_span_same_rule_for_web(self):
        row={'source_id':'W1','excerpt':'7日(今天) 产量25。8日 产量30。','url':'https://example.org'}
        audit={'status':'partial','supported_evidence_ids':['W1'], 'missing_aspects':['第二厂产量'],
               'reason':'只支持目标日第一厂记录','support_spans':{'W1':['8日 产量30。']}}
        self.assertEqual(supported_web_sources([row],audit)[0]['excerpt'],'8日 产量30。')
        self.assertEqual(generation_evaluation(audit,0)['reason'],audit['reason'])

    def test_unknown_span_id_not_allowed(self):
        with self.assertRaises(OutputContractError):
            validate_audit({'status':'supported','supported_evidence_ids':['D1'],
                            'support_spans':{'D2':['x']}},['D1'])

    def test_summary_contract_feedback_does_not_guess_task(self):
        with self.assertRaises(OutputContractError) as caught:
            validate_plan({'tools':['company_rag'],'knowledge_categories':[], 'company_operation':'summary'},require_company_scope=True)
        self.assertIn('required_shape',caught.exception.validation_details)

    def test_quota_aggregate_and_no_invented_negative_finding(self):
        requests=[{'source':'public_web_search','query':'目标甲资料','targets':['甲']},
                  {'source':'public_web_search','query':'目标乙资料','targets':['乙']}]
        def search(q): return [], {'status':'quota_exhausted','error_code':'WEB_QUOTA_EXHAUSTED'}
        rows,meta=retrieve_web_requirements(requests,search)
        self.assertEqual(meta['status'],'quota_exhausted')
        self.assertEqual(len(meta['requirement_coverage']['unexecuted_due_to_call_budget']),1)
        response=unavailable_web_response(meta,'Chinese')
        self.assertFalse(response['answerable']); self.assertFalse(response['meta']['model_used'])
        self.assertIn('不表示所问信息不存在',response['customer_reply'])
        self.assertIn('quota',unavailable_web_response(meta,'English')['customer_reply'])
        self.assertIsNone(unavailable_web_response({'status':'not_requested'},'Chinese'))

    def test_general_failure_shortcuts_do_not_load_model(self):
        import backend.app as app
        with patch.object(app,'load_model',side_effect=AssertionError('must not load')):
            result=app.general_local_chat_answer(app.DraftRequest(customer_question='查询外部资料'),[],
                                                 {'status':'quota_exhausted','error_code':'WEB_QUOTA_EXHAUSTED'})
        self.assertFalse(result['answerable'])

    def test_unavailable_web_skips_semantic_audit_not_local_sources(self):
        import backend.app as app
        from backend.sales.tool_planner import ToolPlan
        request=app.DraftRequest(customer_question='核验公开资料',use_online_search=True)
        with patch.object(app,'evaluate_task_support',side_effect=AssertionError('nothing to audit')):
            result=app._CustomerAnswerWorkflowCallbacks.evaluate_sources(request,
                ToolPlan(tools=['public_web_search'],requires_public_web=True),
                {'public_web_search':([],{'status':'quota_exhausted','error_code':'WEB_QUOTA_EXHAUSTED'})},1)
        self.assertEqual(result['meta']['mode'],'web_tool_unavailable')

    def test_http_200_error_is_not_completion(self):
        for code,expected in [('INTERNAL_ERROR','technical_failure'),('WEB_QUOTA_EXHAUSTED','resource_limited')]:
            self.assertEqual(response_status({'answerable':True,'meta':{'execution':{'errors':[{'code':code}]}}}),expected)
        self.assertEqual(response_status({'answerable':False}),'insufficient_evidence')

    def test_admission_pacing_does_not_rotate_identity(self):
        client=Client('http://localhost'); identity=dict(client.headers)
        clock=[0.]
        def sleep(seconds): clock[0]+=seconds
        with patch('scripts.run_agent_eval_100.time.monotonic',side_effect=lambda:clock[0]), patch('scripts.run_agent_eval_100.time.sleep',side_effect=sleep):
            for _ in range(5): client.admit('/api/copilot/answer',200)
        self.assertGreaterEqual(clock[0],61); self.assertEqual(client.headers,identity)

    def test_http_rejection_preserves_status_not_server_body(self):
        class Fake:
            def request(self,*a,**kw): raise TransportFailure(429,'/api/copilot/answer','RATE_LIMITED',60)
            def cancel_and_wait(self,ids): return True
        row=run_case(Fake(),{'id':'x','group':'company','question':'q','rubric':[]})
        self.assertEqual(row['status'],'resource_limited')
        self.assertEqual(row['transport_error']['http_status'],429)

    def test_swallowed_stage_error_not_completed(self):
        journal=ExecutionStatus(); token=current_execution.set(journal)
        try:
            handle=begin('assess_coverage')
            journal.errors.append({'code':'INTERNAL_ERROR','stage':'assess_coverage'})
            end(handle)
            self.assertEqual(journal.stage_details[0]['status'],'failed')
        finally: current_execution.reset(token)


if __name__=='__main__': unittest.main()
