import unittest
from types import SimpleNamespace
from unittest.mock import patch
from backend.sales.requirements import build_requirements, annotate, representatives, coverage
from backend.sales.prepared_support import fit_support_budget


class RequirementTests(unittest.TestCase):
    def test_query_binds_backend_identity_without_model_targets(self):
        targets=[{'id':'a','name':'Alpha','aliases':['A-code'],'kind':'entity'},
                 {'id':'b','name':'Beta','kind':'entity'}]
        rows=build_requirements(targets,[],requests=[{'query':'A-code dimensions'}, {'query':'Beta installation'}])
        self.assertEqual([r['target_id'] for r in rows],['a','b'])
        rows=build_requirements(targets,['compare dimensions'],operation='comparison')
        self.assertEqual({r['target_id'] for r in rows},{'a','b'})

    def test_comparison_grid_one_to_four_targets(self):
        for count in (1,2,4):
            targets=[{'id':str(i),'name':'file'+str(i),'kind':'document'} for i in range(count)]
            requirements=build_requirements(targets,['dimensions','installation','applications'],operation='comparison')
            self.assertEqual(len(requirements),count*3)
            self.assertEqual(len({r['requirement_id'] for r in requirements}),count*3)

    def test_named_aspect_does_not_cross_all_targets(self):
        targets=[{'id':'a','name':'Alpha.pdf','kind':'document'},{'id':'b','name':'Beta.pdf','kind':'document'}]
        req=build_requirements(targets,['Alpha.pdf costs','Beta.pdf method'],operation='comparison')
        self.assertEqual([r['target_id'] for r in req],['a','b'])
        self.assertEqual(build_requirements(targets,['summarize'],operation='summary'),[])

    def test_shared_evidence_covers_multiple_cells_not_semantic_proof(self):
        req=build_requirements([],['dimensions','method'])
        rows=annotate(annotate([{'evidence_id':'X','text':'candidate'}],req[0]),req[1])
        self.assertEqual(len(representatives(rows)),1)
        self.assertFalse(coverage(req,rows)['semantic_support_verified'])

    def test_comparison_recalls_are_document_scoped(self):
        from backend.documents.customer_sessions import retrieve_goal_candidates
        session=SimpleNamespace(documents=[SimpleNamespace(document_id='a',file_name='Alpha.pdf'),
                                          SimpleNamespace(document_id='b',file_name='Beta.pdf')])
        calls=[]
        def retrieve(sid,q,**kw):
            calls.append((q,kw.get('target_document_ids')))
            ids=kw.get('target_document_ids') or ['a','b']
            return {'status':'ready','evidence':[{'document_id':key,'document_name':key,
                'original_chunk_id':key+q,'text':key+' '+q,'evidence_scope':'content'} for key in ids],
                'input_snapshot':{'max_text_tokens':5000},'selected_visuals':[]}
        with patch('backend.documents.customer_sessions.get_session',return_value=session), \
             patch('backend.documents.customer_sessions.retrieve',side_effect=retrieve):
            result=retrieve_goal_candidates('session','compare',search_targets=['dimensions','method'],
                document_scope='cross_document',document_operation='comparison',max_text_tokens=5000,
                visual_required=False,max_chunks=24,max_characters=None)
        self.assertEqual(calls[1:],[('dimensions',['a']),('dimensions',['b']),('method',['a']),('method',['b'])])
        self.assertTrue(all(r['candidate_state']=='represented' for r in result['input_snapshot']['requirement_coverage']['requirements']))

    def test_missing_cell_survives_packing(self):
        req=build_requirements([],['size','method'])
        rows=annotate([{'evidence_id':'X','text':'size','source_type':'customer_document'}],req[0])
        packed,audit=fit_support_budget(rows,audit_tokens=lambda r:len(r)*10,
            generation_tokens=lambda r:len(r)*10,ceiling=100,retrieval_requirements=req)
        states=[r['candidate_state'] for r in audit['requirement_coverage']['requirements']]
        self.assertEqual(states,['represented','missing'])


if __name__=='__main__':unittest.main()
