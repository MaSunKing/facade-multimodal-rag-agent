"""CPU-only gallery contract tests: never bypass external/technical review."""
import unittest
from unittest.mock import patch
from backend.app import DraftRequest, AnswerResponse, _CustomerAnswerWorkflowCallbacks
from backend.sales.tool_planner import ToolPlan, guard_plan
from backend.sales.visual_policy import labelled_company_gallery, select_visual_support


def sample(key='a', variant='Finish A'):
    return {'asset_id':key,'customer_title':variant,'citation':{'document_name':'Catalogue','source_page':2},
            'origin':'company_rag','image_labels':{'asset_id':key,'display_ready':True,'label_issues':[],
                'label_basis':'existing_source_metadata_and_explicit_catalogue_joins',
                'citation':{'document_name':'Catalogue','source_page':2},'sha256':key,
                'category':'product_sample','variant_or_code':variant,'product_name':'DemoPanel'}}


class CompanyGalleryTests(unittest.TestCase):
    def setUp(self):
        self.plan=ToolPlan(tools=['company_rag'],wants_visuals=True,visual_scope='product',
                           visual_task='gallery',gallery_scope='category',gallery_product_family='all',task_type='factual_lookup')
        self.sources={'company_rag':{'visual_assets':[sample()]}}

    def test_gallery_skips_pixel_audit_but_continues_to_language_generation(self):
        request=DraftRequest(customer_question='Show panel sample images')
        with patch('backend.app.evaluate_task_support',side_effect=AssertionError('no model audit')):
            result=_CustomerAnswerWorkflowCallbacks.evaluate_sources(
                request,self.plan,self.sources,1)
        self.assertIsNone(result)
        self.assertEqual(request._evidence_gate_audit['model_calls'],0)
        self.assertFalse(request._evidence_gate_audit['pixel_contents_verified'])

    def test_gallery_task_is_not_overridden_by_broad_parameter_intent(self):
        request = DraftRequest(customer_question='Show the product pictures')
        plan = self.plan.model_copy(update={'intent':'product_parameter'})
        with patch('backend.app.evaluate_task_support', side_effect=AssertionError('no model audit')):
            self.assertIsNone(_CustomerAnswerWorkflowCallbacks.evaluate_sources(request, plan, self.sources, 1))
        self.assertEqual(request._evidence_gate_audit['model_calls'], 0)

    def test_mixed_inspection_unknown_and_technical_tasks_never_bypass(self):
        for update in [{'visual_task':'mixed'},{'visual_task':'inspection'},{'visual_task':'unknown'},
                       {'product_overview':True},{'intent':'technical_performance'},
                       {'task_type':'project_fit'},{'wants_visuals':False}]:
            self.assertIsNone(labelled_company_gallery(self.plan.model_copy(update=update),self.sources))

    def test_external_and_direct_uploads_never_bypass(self):
        for name in ['customer_documents','public_web_search']:
            self.assertIsNone(labelled_company_gallery(self.plan,{name:self.sources['company_rag']}))
            self.assertIsNone(labelled_company_gallery(self.plan,{**self.sources,name:{}}))
        self.assertIsNone(labelled_company_gallery(self.plan,self.sources,has_direct_image=True))
        asset=sample();asset['origin']='customer_documents'
        self.assertIsNone(labelled_company_gallery(self.plan,{'company_rag':{'visual_assets':[asset]}}))

    def test_missing_disputed_or_unproven_labels_use_normal_review(self):
        for changes in [{'label_issues':['ambiguous_product']},{'display_ready':False},
                        {'label_basis':'uploaded_claim'},{'asset_id':'wrong'},{'citation':{}}]:
            asset=sample();asset['image_labels'].update(changes)
            self.assertIsNone(labelled_company_gallery(self.plan,{'company_rag':{'visual_assets':[asset]}}))

    def test_variant_diversity_and_duplicate_hashes(self):
        assets=[sample('a','Finish A'),sample('b','Finish A'),sample('c','Finish B'),sample('d','Finish C')]
        assets[-1]['image_labels']['sha256']='a'
        result=labelled_company_gallery(self.plan,{'company_rag':{'visual_assets':assets}},limit=2)
        self.assertEqual([a['asset_id'] for a in result['visual_assets']],['a','c'])
        self.assertIn('不代表完整产品目录',result['customer_reply'])

    def test_default_display_returns_at_most_ten_without_pixel_inference(self):
        assets=[sample(str(i), 'Finish '+str(i)) for i in range(12)]
        result=labelled_company_gallery(self.plan,{'company_rag':{'visual_assets':assets}})
        self.assertEqual(len(result['visual_assets']),10)
        self.assertEqual(result['meta']['evidence_gate']['visual_need'],'not_required')
        self.assertEqual(result['meta']['evidence_gate']['model_calls'],0)

    def test_external_visuals_still_need_actual_pixels(self):
        external={'document_id':'upload','visual_id':'page1',
                  'image_labels':sample()['image_labels']}
        selected,audit=select_visual_support([external],{'visual_need':'required'})
        self.assertEqual(selected,[external])
        self.assertEqual(audit['metadata_only_display_asset_ids'],[])

    def test_guard_preserves_semantic_gallery_contract(self):
        safe=guard_plan(self.plan,has_documents=False,has_image=False,facade_related=False,web_allowed=False)
        self.assertEqual(safe.visual_task,'gallery')
        self.assertEqual(safe.gallery_product_family,'all')

    def test_graph_gallery_reaches_generation_without_extra_audit_or_replan(self):
        from backend.sales.answer_graph import build_customer_answer_graph
        from backend.sales.staged_execution import WorkflowStep
        plan,sources=self.plan,self.sources
        class Callbacks(_CustomerAnswerWorkflowCallbacks):
            @staticmethod
            def plan_tools(request): return plan
            @staticmethod
            def open_steps(request,plan):
                yield WorkflowStep('company_rag',lambda:sources['company_rag'])
                yield WorkflowStep('compose_evidence')
                yield WorkflowStep('generate_answer')
                return {'answerable':True,'customer_reply':'Organised gallery captions', 'meta':{}}
        with patch('backend.app.evaluate_task_support',side_effect=AssertionError('no model')):
            result=build_customer_answer_graph(Callbacks()).invoke(
                {'request':DraftRequest(customer_question='Show sample images')})['response']
        self.assertEqual(result['customer_reply'],'Organised gallery captions')
        self.assertTrue(any(row['node']=='generate_answer' for row in result['meta']['orchestration']['node_trace']))
        self.assertEqual(result['meta']['orchestration']['tool_rounds'],1)
        self.assertEqual(result['meta']['orchestration']['visual_task'],'gallery')


    def test_family_filter_never_substitutes_insulated_for_single_panel(self):
        single,insulated=sample('single'),sample('insulated')
        single['image_labels']['product_family_key']='decorative_panel'
        insulated['image_labels']['product_family_key']='insulated_panel'
        result=labelled_company_gallery(self.plan.model_copy(update={'gallery_product_family':'decorative_panel'}),
            {'company_rag':{'visual_assets':[insulated,single]}})
        self.assertEqual([a['asset_id'] for a in result['visual_assets']],['single'])
        self.assertIsNone(labelled_company_gallery(self.plan.model_copy(update={'gallery_product_family':'unknown'}),self.sources))


if __name__=='__main__':unittest.main()
