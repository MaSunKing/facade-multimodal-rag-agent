import unittest
from backend.sales.gap_recovery import gap_state,new_operations,merge_sources
from backend.sales.tool_planner import ToolPlan


class GapRecoveryTests(unittest.TestCase):
    def test_partial_recovery_precedes_single_final_generation(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from backend.sales.answer_graph import build_customer_answer_graph
        from backend.sales.staged_execution import WorkflowStep
        from backend.request_budget import RequestBudget,budget_scope,reserve_recovery
        calls=[]
        request=SimpleNamespace(_prior_source_results={},_prepared_evidence=None)
        class Callbacks:
            def plan_tools(self,request): return ToolPlan(tools=['customer_documents'],retrieval_query='original')
            def open_steps(self,request,plan):
                result=yield WorkflowStep('customer_documents',lambda: {'evidence':[{'document_id':plan.retrieval_query,'text':plan.retrieval_query} ]})
                yield WorkflowStep('compose_evidence')
                calls.append(('generate',len(result['evidence'])))
                yield WorkflowStep('generate_answer')
                return {'answerable':True,'meta':{'model_used':True}}
            def evaluate_sources(self,request,plan,sources,round_number):
                calls.append(('audit',round_number))
                if round_number==1:
                    request._prior_source_results=sources
                    self.pending=plan.model_copy(update={'retrieval_query':'gap','recovery_directive':{'stage':'task_support_audit'}})
                    self.assert_reserved=reserve_recovery('task_support_audit','semantic_replan')
                    return {'answerable':False,'meta':{'model_used':False,'evidence_gate':{'recovery_reserved':True}}}
            def plan_retry(self,request,plan,response): return self.pending
        budget=RequestBudget.for_answer(has_attachments=True)
        with budget_scope(budget):
            result=build_customer_answer_graph(Callbacks()).invoke({'request':request})['response']
        self.assertEqual(calls,[('audit',1),('audit',2),('generate',2)])
        self.assertEqual(len(budget.recoveries),1);self.assertTrue(result['answerable'])

    def test_no_semantic_recovery_after_final_answer(self):
        from unittest.mock import patch
        from backend.app import DraftRequest,_CustomerAnswerWorkflowCallbacks
        request=DraftRequest(customer_question='general summary')
        with patch('backend.app.evaluate_task_support') as audit:
            result=_CustomerAnswerWorkflowCallbacks.plan_retry(request,ToolPlan(tools=['customer_documents']),
                {'answerable':True,'meta':{'model_used':True,'file_task_coverage':{'missing_document_ids':['b']}}})
        self.assertIsNone(result);audit.assert_not_called()

    def test_backend_gaps_stable_and_original_requirements_retained(self):
        documents=[{'document_id':'a'},{'document_id':'b'}]
        a=gap_state('task',{'status':'partial','missing_aspects':['condition','condition']},documents)
        self.assertEqual(a['required_document_ids'],['a','b']);self.assertEqual(len(a['gaps']),1)
        self.assertEqual(a,gap_state('task',{'status':'partial','missing_aspects':['condition']},documents))
        self.assertEqual(gap_state('original',{'status':'uncertain'})['gaps'][0]['requirement'],'original')

    def test_cosmetic_plan_changes_not_new_operations(self):
        old=ToolPlan(tools=['customer_documents'],document_retrieval_queries=['Find condition'],reason='a')
        new=old.model_copy(update={'reason':'b','document_retrieval_queries':['  find   condition  ']})
        self.assertEqual(new_operations(old,new),[])
        new=new.model_copy(update={'document_retrieval_queries':['Find excluded conditions']})
        self.assertEqual(new_operations(old,new),['customer_documents'])

    def test_company_filter_change_is_real_execution_change(self):
        old=ToolPlan(tools=['company_rag'],retrieval_query='find')
        new=old.model_copy(update={'case_reference':True})
        self.assertEqual(new_operations(old,new),['company_rag'])

    def test_changed_retrieval_unions_sources_and_unique_ids(self):
        old={'evidence':[{'evidence_id':'U1','document_id':'a','text':'old conditional statement'}],'documents':[{'document_id':'a'}]}
        new={'evidence':[{'evidence_id':'U1','document_id':'b','text':'new fact'}],'documents':[{'document_id':'b'}]}
        merged=merge_sources(old,new,'customer_documents')
        self.assertEqual({r['document_id'] for r in merged['evidence']},{'a','b'})
        self.assertEqual([r['evidence_id'] for r in merged['evidence']],['U2','U1'])
        self.assertIn('old conditional statement',[r['text'] for r in merged['evidence']])
        self.assertEqual(old['evidence'][0]['evidence_id'],'U1')

    def test_web_union_deduplicates_by_url_and_preserves_provenance(self):
        old=([{'source_id':'W1','url':'https://a','excerpt':'old','date':'2020'}],{'cache_hit':True})
        new=([{'source_id':'W1','url':'https://b','excerpt':'new','date':'2026'}],{'cache_hit':False})
        rows,meta=merge_sources(old,new,'public_web_search')
        self.assertEqual([r['date'] for r in rows],['2026','2020']);self.assertTrue(meta['merged_previous_sources'])
        self.assertEqual([r['source_id'] for r in rows], ['W2', 'W1'])

    def test_stable_ids_all_sources_despite_reordering(self):
        for tool, field, prefix in [('company_rag','text_evidence','T'), ('customer_documents','evidence','U')]:
            old={field:[{'evidence_id':prefix+'7','document_id':'a','text':'condition'},
                        {'evidence_id':prefix+'2','document_id':'b','text':'other'}]}
            new={field:[{'evidence_id':prefix+'1','document_id':'b','text':'other'},
                        {'evidence_id':prefix+'2','document_id':'c','text':'new'}]}
            rows=merge_sources(old,new,tool)[field]
            self.assertEqual({r['document_id']:r['evidence_id'] for r in rows}, {'a':prefix+'7','b':prefix+'2','c':prefix+'1'})
            self.assertEqual(rows[-1]['retrieval_rounds'], [1])

    def test_updated_web_content_is_version_not_silent_overwrite(self):
        old=([{'source_id':'W1','url':'https://a','excerpt':'old','date':'2020'}],{})
        new=([{'source_id':'W1','url':'https://a','excerpt':'new','date':'2026'}],{})
        rows,_=merge_sources(old,new,'public_web_search')
        self.assertEqual([r['source_id'] for r in rows], ['W2','W1'])
        self.assertEqual(rows[0]['previous_version_ids'], ['W1'])
        self.assertEqual(old[0][0]['excerpt'], 'old')

    def test_recovery_review_does_not_promote_prior_diagnostic(self):
        from backend.sales.gap_recovery import recovery_review_context
        state=gap_state('current requirement', {'status':'partial','missing_aspects':['W1 only old version']})
        review=recovery_review_context('current requirement', state)
        self.assertNotIn('W1',str(review))
        self.assertEqual(review['original_requirement'], 'current requirement')
        self.assertIn('W1',state['gaps'][0]['requirement'])  # Trace retains diagnosis.

    def test_support_selection_and_conflict_contract(self):
        from backend.sales.task_support import supported_evidence_rows
        from backend.sales.model_contract import validate_audit
        from backend.sales.output_contract import OutputContractError
        rows=[{'evidence_id':i} for i in ['T1','U1','W1']]
        audit={'status':'partial','supported_evidence_ids':['T1','U1'],
               'conflict_groups':[['T1','U1']], 'missing_aspects':['remaining condition'], 'recovery_need':'lookup'}
        validate_audit(audit,['T1','U1','W1'])
        self.assertEqual(supported_evidence_rows(rows,audit),rows[:2])
        for groups in ([['T1','foreign']], [['T1','W1']], [['T1','T1']], 'invalid'):
            with self.assertRaises(OutputContractError):
                validate_audit({**audit,'conflict_groups':groups},['T1','U1','W1'])


if __name__=='__main__': unittest.main()
