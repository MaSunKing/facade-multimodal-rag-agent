import copy
import unittest
from unittest.mock import patch
from backend.sales.context_engine import pack_task_context
from backend.sales.prepared_support import stage_support_candidates


class UnifiedContextPackTests(unittest.TestCase):
    def test_handoff_does_not_select(self):
        rows=[{'evidence_id':'a','text':'unrelated '*100},{'evidence_id':'b','text':'requested capacity 20 L'}]
        staged,audit=stage_support_candidates('capacity',rows)
        self.assertEqual([r['evidence_id'] for r in staged],['a','b'])
        self.assertEqual(audit['policy'],'candidate_handoff_no_secondary_selection')

    def test_product_text_survives_gallery_eviction_and_both_images_survive(self):
        rows=[{'evidence_id':'T','source_type':'company_rag','text':'Product facts ' * 5},
              *[{'evidence_id':'G'+str(i),'source_type':'visual','document_name':'gallery'+str(i),'text':'metadata '*15} for i in range(4)],
              *[{'evidence_id':'V'+str(i),'source_type':'customer_document','document_id':'d'+str(i),
                 'document_name':'image','visual_description':True,'text':'visible color, unknown material'} for i in range(2)]]
        original=copy.deepcopy(rows)
        count=lambda selected:sum(len(r['text']) for r in selected)
        kept,audit=pack_task_context(rows,question='Compare product and images',scope='cross_document',operation='comparison',
            required_document_ids=['d0','d1'],audit_tokens=count,generation_tokens=count,ceiling=200,target_tokens=200)
        self.assertEqual({r['evidence_id'] for r in kept},{'T','V0','V1'})
        self.assertEqual(rows,original)
        self.assertFalse(audit['missing_target_document_ids'])
        self.assertEqual(audit['policy'],'unified_task_context_pack_v1')

    def test_one_to_four_summary_documents_cannot_disappear(self):
        for n in range(1,5):
            rows=[{'evidence_id':str(i),'document_id':str(i),'document_name':str(i),'text':'summary '*40,
                   'source_type':'customer_document','derived_summary':True,'required_document_coverage':True} for i in range(n)]
            kept,audit=pack_task_context(rows,question='Summarize',scope='cross_document',operation='summary',
                required_document_ids=[str(i) for i in range(n)],audit_tokens=lambda _:6000,
                generation_tokens=lambda _:6100,ceiling=8000,target_tokens=4500)
            self.assertEqual(len(kept),n)
            self.assertFalse(audit['missing_target_document_ids'])

    def test_customer_visual_requirement_does_not_trigger_company_pixel_calls(self):
        from backend.app import DraftRequest,prepare_visual_text_evidence
        from backend.sales.tool_planner import ToolPlan
        with patch('backend.app.generate_multi_visual_response',side_effect=AssertionError('no unnecessary image read')):
            rows=prepare_visual_text_evidence(DraftRequest(customer_question='compare'),
                ToolPlan(tools=['customer_documents','company_rag'],document_visual_required=True),
                {'company_rag':{'visual_assets':[{'asset_id':'x','image_labels':{'display_ready':True}}]}})
        self.assertEqual(rows,[])

    def test_cross_document_operation_controls_goal_expansion(self):
        from backend.documents.customer_sessions import retrieve_goal_candidates
        for operation, expected_calls in [('summary', 1), ('comparison', 3), ('lookup', 3)]:
            with patch('backend.documents.customer_sessions.retrieve', return_value={
                'status':'ok', 'evidence':[], 'input_snapshot':{}}) as retrieval, \
                 patch('backend.documents.customer_sessions.get_session', return_value=None):
                retrieve_goal_candidates('session', 'task', search_targets=['capacity', 'installation'],
                    document_scope='cross_document', document_operation=operation,
                    max_text_tokens=2000, visual_required=False, max_chunks=24, max_characters=None)
                self.assertEqual(retrieval.call_count, expected_calls)
                self.assertEqual(retrieval.call_args_list[0].kwargs['document_operation'], operation)


if __name__=='__main__':unittest.main()
