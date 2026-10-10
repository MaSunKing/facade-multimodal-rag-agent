import unittest
from unittest.mock import patch, Mock
from contextlib import contextmanager
import json

from backend.sales.query_constraints import protect_query_time
from backend.sales.web_content import extract_page_text, select_page_windows
from backend.sales.web_network import resolve_public_host


class WebRecoveryTests(unittest.TestCase):
    def test_fetching_marketing_page_does_not_certify_its_claims(self):
        from backend.sales.baidu_search import _source_quality_summary
        audit=_source_quality_summary([{'evidence_level':'verified_page_content','authority_score':0.18}])
        self.assertEqual(audit['verified_page_count'],1)
        self.assertEqual(audit['reliable_source_count'],0)
        self.assertTrue(audit['requires_cautious_wording'])

    def test_joint_preview_does_not_let_first_source_fill_every_slot(self):
        from backend.sales.evidence_gate import evidence_preview
        rows=evidence_preview([{'text_evidence':[{'text':f'company {i}'} for i in range(10)]},
                               ([{'excerpt':'web fact','evidence_level':'unverified_search_excerpt'}],{})],limit=4)
        self.assertEqual(rows[1]['text'],'web fact')
        self.assertEqual(rows[1]['evidence_level'],'unverified_search_excerpt')
        self.assertEqual([r['evidence_id'] for r in rows],['E1','E2','E3','E4'])

    def test_failed_partial_answer_never_pastes_unrelated_true_evidence(self):
        from backend.app import DraftRequest, fallback_grounded_answer
        request=DraftRequest(customer_question='Which material has the largest current market share?')
        request._evidence_gate_audit={'status':'partial','missing_aspects':['market share evidence']}
        result=fallback_grounded_answer(request, {'text_evidence':[{'text':'Anchor strength 20 MPa'}]},'validation_failed')
        self.assertFalse(result['answerable'])
        self.assertNotIn('20 MPa', result['customer_reply'])
        self.assertIn('market share evidence', result['customer_reply'])

    def test_current_question_rejects_invented_historical_year(self):
        value, audit = protect_query_time('现在有哪些方案？','2025方案','2026-10-04')
        self.assertEqual(value,'现在有哪些方案？')
        self.assertTrue(audit['applied'])

    def test_explicit_historical_year_is_not_changed_to_current(self):
        value, audit = protect_query_time('2024年有哪些方案？','2024年方案','2026-10-04')
        self.assertEqual(value,'2024年方案'); self.assertFalse(audit['applied'])

    def test_private_dns_is_never_replaced_by_public_dns(self):
        with patch('backend.sales.web_network.socket.getaddrinfo',return_value=[(None,None,None,None,('127.0.0.1',443))]), patch('backend.sales.web_network.build_opener') as opener:
            with self.assertRaisesRegex(ValueError,'non_public_address'):
                resolve_public_host('private.example',443)
            opener.assert_not_called()

    def test_normal_dns_does_not_use_stale_vpn_proxy(self):
        with patch('backend.sales.web_network.socket.getaddrinfo',return_value=[(None,None,None,None,('8.8.8.8',443))]), patch('backend.sales.web_network.configured_local_proxy') as proxy:
            ips, selected = resolve_public_host('public.example',443)
            self.assertEqual(ips,['8.8.8.8']); self.assertIsNone(selected)
            proxy.assert_not_called()

    def test_fake_dns_public_verification_via_local_proxy(self):
        response=Mock()
        response.geturl.return_value='https://dns.alidns.com/resolve'
        response.read.return_value=json.dumps({'Answer':[{'type':1,'data':'8.8.4.4'}]}).encode()
        opener=Mock(); opener.open.return_value.__enter__=Mock(return_value=response)
        opener.open.return_value.__exit__=Mock(return_value=False)
        with patch('backend.sales.web_network.socket.getaddrinfo',return_value=[(None,None,None,None,('198.18.0.1',443))]), patch('backend.sales.web_network.configured_local_proxy',return_value='http://127.0.0.1:7890'), patch('backend.sales.web_network.build_opener',return_value=opener):
            ips, proxy=resolve_public_host('vpn-test.example',443)
            self.assertEqual(ips,['8.8.4.4']); self.assertEqual(proxy,'http://127.0.0.1:7890')

    def test_short_labels_and_tables_survive_structural_fallback(self):
        with patch.dict('sys.modules',{'trafilatura':None}):
            text, method=extract_page_text('<html><nav>Advertisement</nav><main><h1>Panel</h1><table><tr><th>厚度</th><td>20 mm</td></tr></table><p>不得浸水</p></main></html>')
        self.assertIn('厚度 | 20 mm',text); self.assertIn('不得浸水',text)
        self.assertNotIn('Advertisement',text)

    def test_late_answer_is_eligible_not_prefix_only(self):
        text='Unrelated background. '*200+' TargetPanel thickness 20 mm. Do not immerse.'
        selected, audit=select_page_windows(text,'TargetPanel thickness',limit=1000)
        self.assertIn('20 mm',selected); self.assertGreater(audit['selected_window_indices'][-1],0)

    def test_replan_retains_selected_local_evidence_without_forcing_it(self):
        from backend.app import DraftRequest, _CustomerAnswerWorkflowCallbacks
        from backend.sales.tool_planner import ToolPlan
        request=DraftRequest(customer_question='现在方案有哪些？',use_online_search=True)
        original={'text_evidence':[{'text':'Product facts'}]}
        request._source_results={'company_rag':original}
        result={'action':'execute','retain_source_tools':['company_rag'], 'revised_plan':{
            'tools':['public_web_search'],'requires_public_web':True,'retrieval_query':'现在方案有哪些？'}}
        with patch('backend.app.evaluate_task_support',return_value=result):
            revised=_CustomerAnswerWorkflowCallbacks.replan_from_feedback(request,ToolPlan(tools=['company_rag'],retrieval_query='现在方案有哪些？'),
                {'status':'partial','suggested_query':'2025方案'},{})
        self.assertEqual(set(revised.tools),{'company_rag','public_web_search'})
        self.assertEqual(revised.retrieval_query,request.customer_question)
        self.assertIs(_CustomerAnswerWorkflowCallbacks.retained_source(request,'company_rag'),original)

    def test_graph_reuses_retained_source_with_changed_query(self):
        from backend.sales.answer_graph import build_customer_answer_graph
        from backend.sales.staged_execution import WorkflowStep
        from backend.sales.tool_planner import ToolPlan
        calls=[]
        class Callbacks:
            def plan_tools(self,request): return ToolPlan(tools=['company_rag'],retrieval_query='first')
            def open_steps(self,request,plan):
                value=yield WorkflowStep('company_rag',lambda:calls.append('retrieve') or 'evidence')
                yield WorkflowStep('generate_answer')
                return {'answer':value,'meta':{}}
            def plan_retry(self,request,plan,response): return plan.model_copy(update={'retrieval_query':'changed'})
            def retained_source(self,request,tool): return 'retained evidence' if calls else None
        result=build_customer_answer_graph(Callbacks()).invoke({'request':{}})['response']
        self.assertEqual(calls,['retrieve']); self.assertEqual(result['answer'],'retained evidence')


if __name__=='__main__': unittest.main()
