import unittest
from types import SimpleNamespace
from unittest.mock import patch
from backend.sales.requirements import build_requirements, annotate, source_requests
from backend.sales.model_contract import validate_plan, validate_replan, support_contract
from backend.sales.output_contract import OutputContractError
from backend.sales.tool_planner import ToolPlan, guard_plan
from backend.sales.web_requirements import retrieve_web_requirements
from backend.sales.web_context import web_evidence_row
from backend.sales.prepared_support import fit_support_budget


class ScopedRetrievalTests(unittest.TestCase):
    targets = [{'id':'a','name':'Alpha','kind':'entity','aliases':['A-code']},
               {'id':'b','name':'Beta','kind':'entity','aliases':['B-code']}]

    def test_lookup_shared_targets_and_alias_binding(self):
        self.assertEqual(len(build_requirements(self.targets,['dimensions'],operation='lookup')),2)
        rows=build_requirements(self.targets,[],requests=[{'query':'dimensions','targets':['A-code']},
                                                       {'query':'installation','targets':['Beta']}])
        self.assertEqual([(r['target_id'],r['aspect']) for r in rows],[('a','dimensions'),('b','installation')])

    def test_unresolved_never_widens_and_summary_unchanged(self):
        request={'query':'conditions','targets':['missing']}
        rows=build_requirements(self.targets,[],requests=[request])
        self.assertEqual(rows[0]['target_kind'],'unresolved')
        self.assertEqual(build_requirements(self.targets,[],requests=[{'query':'general scope','targets':[]}])[0]['target_kind'],'request')
        self.assertEqual(build_requirements(self.targets,['overview'],operation='summary',requests=[request]),[])

    def test_strict_fields_sources_and_guard(self):
        valid={'tools':['public_web_search'],'retrieval_requests':[
            {'source':'public_web_search','query':'independent public fact','targets':[]}]}
        validate_plan(valid,allowed_tools=['public_web_search'])
        self.assertEqual(len(source_requests(ToolPlan(**valid),'public_web_search')),1)
        for row in ({'source':'company_rag','query':'x'},
                    {'source':'public_web_search','query':' '},
                    {'source':'public_web_search','query':'x','acl':'public'}):
            with self.assertRaises(OutputContractError):
                validate_plan({'tools':['public_web_search'],'retrieval_requests':[row]})

    def test_web_separate_calls_dedup_and_coverage(self):
        calls=[]
        def search(query):
            calls.append(query)
            return [{'source_id':'W1','url':'https://example.com/shared','excerpt':'shared evidence'},
                    {'source_id':'W2','url':'https://example.com/'+query,'excerpt':query}],{'status':'ok','api_calls_for_query':1}
        requests=[{'query':'A public fact','targets':[]},{'query':'B public fact','targets':[]}]
        rows,meta=retrieve_web_requirements(requests,search)
        self.assertEqual(calls,['A public fact','B public fact'])
        self.assertEqual(len(rows),3)
        self.assertEqual(len(rows[0]['requirement_ids']),2)
        self.assertEqual([r['source_id'] for r in rows],['W1','W2','W3'])
        wire=[web_evidence_row(r) for r in rows]
        packed,audit=fit_support_budget(wire,audit_tokens=lambda r:len(r)*10,
            generation_tokens=lambda r:len(r)*10,ceiling=10,
            retrieval_requirements=meta['requirement_coverage']['requirements'])
        self.assertEqual(len(packed),1)
        self.assertTrue(all(r['candidate_state']=='represented' for r in audit['requirement_coverage']['requirements']))

    def test_web_budget_stops_without_silently_losing_requirement(self):
        calls=[]
        def search(query):
            calls.append(query)
            return [],{'status':'quota_exhausted'}
        _,meta=retrieve_web_requirements([{'query':q,'targets':[]} for q in ('one','two')],search)
        self.assertEqual(calls,['one'])
        self.assertEqual(len(meta['requirement_coverage']['unexecuted_due_to_call_budget']),1)
        self.assertEqual(len(meta['requirement_coverage']['requirements']),2)

    def test_actual_web_entry_fanout_and_permission(self):
        from backend.app import DraftRequest, maybe_search_online
        plan=ToolPlan(tools=['public_web_search'],retrieval_requests=[
            {'source':'public_web_search','query':q} for q in ('public topic A','public topic B')])
        calls=[]
        def search(q,**kwargs):
            calls.append((q,kwargs.get('bypass_cache')))
            return {'sources':[{'source_id':'W1','url':'https://example.com/'+str(len(calls)),
                               'excerpt':'result'}],'status':'ok'}
        request=DraftRequest(customer_question='two public topics',use_online_search=True)
        request._refresh_public_web_once=True
        with patch('backend.app.is_baidu_search_configured',return_value=True), \
             patch('backend.sales.search_mcp.search_web_via_mcp',side_effect=search):
            rows,meta=maybe_search_online(request,'combined',plan=plan)
            self.assertEqual(calls,[('public topic A',True),('public topic B',True)])
            self.assertEqual(len(rows),2)
            self.assertFalse(request._refresh_public_web_once)
            request.use_online_search=False
            maybe_search_online(request,'combined',plan=plan)
            self.assertEqual(len(calls),2)

    def test_attachment_lookup_explicit_scopes(self):
        from backend.documents.customer_sessions import retrieve_goal_candidates
        session=SimpleNamespace(documents=[SimpleNamespace(document_id='a',file_name='Alpha.pdf'),
                                          SimpleNamespace(document_id='b',file_name='Beta.pdf')])
        calls=[]
        def retrieve(sid,q,**kw):
            calls.append((q,kw.get('target_document_ids')))
            ids=kw.get('target_document_ids') or ['a','b']
            return {'status':'ready','evidence':[{'document_id':i,'original_chunk_id':q+i,
                'text':q,'evidence_scope':'content'} for i in ids],'input_snapshot':{},'selected_visuals':[]}
        requests=[{'query':'terms','targets':['a']},{'query':'conditions','targets':['Beta.pdf']},
                  {'query':'missing','targets':['unknown']}]
        with patch('backend.documents.customer_sessions.get_session',return_value=session), \
             patch('backend.documents.customer_sessions.retrieve',side_effect=retrieve):
            result=retrieve_goal_candidates('s','task',search_targets=[],document_scope='local_lookup',
                document_operation='lookup',max_text_tokens=5000,visual_required=False,
                max_chunks=24,max_characters=None,retrieval_requests=requests)
        self.assertEqual(calls[1:],[('terms',['a']),('conditions',['b'])])
        self.assertEqual(result['input_snapshot']['requirement_coverage']['requirements'][-1]['candidate_state'],'missing')
        self.assertEqual(result['input_snapshot']['requirement_coverage']['unexecuted_due_to_call_budget'],[])
        self.assertEqual(len(result['input_snapshot']['requirement_coverage']['unexecuted_due_to_scope']),1)

    def test_recovery_shape_and_signature(self):
        from backend.sales.gap_recovery import operation_signature
        original=ToolPlan(tools=['public_web_search'])
        updated=original.model_copy(update={'retrieval_requests':[
            {'source':'public_web_search','query':'new target','targets':[]}]})
        self.assertNotEqual(operation_signature(original,'public_web_search'),operation_signature(updated,'public_web_search'))
        value={'action':'execute','gap_ids':['g1'],'revised_plan':{'tools':['public_web_search'],
            'retrieval_requests':[{'source':'public_web_search','query':'missing fact'}]}}
        self.assertEqual(validate_replan(value,allowed_tools=['public_web_search'],gap_ids=['g1']),value)
        with self.assertRaises(OutputContractError) as raised:
            validate_replan({**value,'missing_aspects':['x']},allowed_tools=['public_web_search'],gap_ids=['g1'])
        self.assertEqual(raised.exception.validation_details['unexpected_fields'],['missing_aspects'])

    def test_initial_declared_multi_lookup_requires_execution_requests(self):
        value={'tools':['public_web_search'],'answer_aspects':['target A fact','target B fact']}
        with self.assertRaises(OutputContractError) as raised:
            validate_plan(value,require_scoped_requests=True)
        self.assertEqual(raised.exception.field,'retrieval_requests')
        validate_plan({**value,'retrieval_requests':[{'source':'public_web_search','query':'target A fact'},
            {'source':'public_web_search','query':'target B fact'}]},require_scoped_requests=True)
        validate_plan({'tools':['company_rag'],'company_operation':'summary','answer_aspects':['A','B']},
                      require_scoped_requests=True)

    def test_identical_web_query_reused_and_round_coverage_preserved(self):
        from backend.sales.gap_recovery import merge_sources
        calls=[]
        def search(q):
            calls.append(q)
            return [{'source_id':'W1','url':'https://example.com','excerpt':'candidate'}],{'status':'ok','api_calls_for_query':1}
        old=retrieve_web_requirements([{'query':'same query','targets':['A','B']}],search)
        self.assertEqual(calls,['same query'])
        self.assertEqual(len(old[0][0]['requirement_ids']),2)
        new=retrieve_web_requirements([{'query':'new query','targets':[]}],search)
        rows,meta=merge_sources(old,new,'public_web_search')
        self.assertEqual(len(rows[0]['requirement_ids']),3)
        self.assertEqual(len(meta['requirement_coverage']['requirements']),3)
        self.assertTrue(all(r['evidence_ids']==['W1'] for r in meta['requirement_coverage']['requirements']))


if __name__=='__main__':unittest.main()
