import unittest
from unittest.mock import patch
from backend.sales.visual_evidence import validate_descriptions, description_row
from backend.sales.visual_policy import enforce_requested_visual_support
from backend.sales.output_contract import OutputContractError


class TextualVisualEvidenceTests(unittest.TestCase):
    def test_unrequested_company_gallery_does_not_expand_context(self):
        from backend.app import DraftRequest, prepare_visual_text_evidence
        from backend.sales.tool_planner import ToolPlan
        rows=prepare_visual_text_evidence(DraftRequest(customer_question='介绍类别'),
            ToolPlan(tools=['company_rag']), {'company_rag':{'visual_assets':[
                {'asset_id':'a','image_labels':{'display_ready':True,'product_name':'目录名称'}}]}})
        self.assertEqual(rows,[])

    def row(self):
        return dict(image_id='image1', labels=['纹理'], description='浅色粗糙表面',
                    visible_text=[], relations=[], uncertainties=['无法判断材料配方'])

    def test_complete_unique_ids_required(self):
        validate_descriptions({'images':[self.row()]}, ['image1'])
        for rows in ([], [self.row(), self.row()], [dict(self.row(), image_id='other')]):
            with self.assertRaises(OutputContractError):
                validate_descriptions({'images':rows}, ['image1'])

    def test_model_cannot_set_provenance(self):
        with self.assertRaises(OutputContractError):
            validate_descriptions({'images':[dict(self.row(), owner='admin')]}, ['image1'])
        row=description_row({'document_id':'trusted','visual_id':'a','document_name':'客户文件'}, self.row(), 'VL1')
        self.assertEqual(row['document_id'], 'trusted')
        self.assertEqual(row['source_refs'], ['a'])
        self.assertFalse(row['candidate_requires_pixel_verification'])

    def test_descriptions_do_not_force_partial_or_remove_genuine_gap(self):
        for status in ('supported','partial','unsupported'):
            result=enforce_requested_visual_support({'status':status,'missing_aspects':['未描述尺寸']}, [],
                                                    required=True, descriptions_ready=True)
            self.assertEqual(result['status'],status)
            self.assertEqual(result['missing_aspects'],['未描述尺寸'])
            self.assertFalse(result['pixel_support_pending'])

    def test_pixels_rejected_before_model_load(self):
        from backend.app import DraftRequest, evaluate_task_support
        with patch('backend.app.load_model',side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(ValueError,'audit_pixels_forbidden'):
                evaluate_task_support(DraftRequest(customer_question='比较图片'),{},image_paths=['a'])

    def test_summary_without_visual_requirement_skips_embedded_images(self):
        from backend.app import DraftRequest, prepare_visual_text_evidence
        from backend.sales.tool_planner import ToolPlan
        request=DraftRequest(customer_question='介绍文件内容')
        with patch('backend.app.generate_multi_visual_response',side_effect=AssertionError('no pixel call')):
            rows=prepare_visual_text_evidence(request,ToolPlan(tools=['customer_documents'],document_operation='summary'),
                                             {'customer_documents':{'selected_visuals':[{'image_bytes':b'QR'}]}})
        self.assertEqual(rows,[])

    def test_multiple_images_source_binding_cache_and_packing(self):
        import json
        from backend.app import DraftRequest, prepare_visual_text_evidence
        from backend.sales.tool_planner import ToolPlan
        from backend.sales.prepared_support import prepare_support
        from backend.sales.evidence_gate import compact_audit_evidence
        request=DraftRequest(customer_question='比较两张图片')
        sources={'customer_documents':{'selected_visuals':[
            {'document_id':'A','visual_id':'skip','image_bytes':None},
            {'document_id':'A','visual_id':'one','image_bytes':b'first'},
            {'document_id':'B','visual_id':'two','image_bytes':b'second'}]}}
        plan=ToolPlan(tools=['customer_documents'],document_visual_required=True)
        with patch('backend.app.generate_multi_visual_response',return_value=json.dumps({'images':[self.row()]})) as model:
            rows=prepare_visual_text_evidence(request,plan,sources)
            repeat=prepare_visual_text_evidence(request,plan,sources)
        self.assertEqual(model.call_count,2)
        self.assertEqual([row['visual_id'] for row in rows],['one','two'])
        self.assertEqual(rows,repeat)
        packed,_=prepare_support('比较图片',rows)
        self.assertEqual(len(packed),2)
        self.assertTrue(all(row['description_basis']=='pixel_tool_observation' for row in compact_audit_evidence(packed)))

    def test_healthy_company_gallery_uses_labels_without_pixels(self):
        from backend.app import DraftRequest, prepare_visual_text_evidence
        from backend.sales.tool_planner import ToolPlan
        with patch('backend.app.generate_multi_visual_response',side_effect=AssertionError('no pixel call')):
            rows=prepare_visual_text_evidence(DraftRequest(customer_question='展示图库'),
                ToolPlan(tools=['company_rag'],visual_task='gallery'),
                {'company_rag':{'visual_assets':[{'asset_id':'a','image_labels':{'display_ready':True,'product_name':'目录名称'}}]}})
        self.assertEqual(rows[0]['description_basis'],'source_metadata_labels')

    def test_company_inspection_uses_detailed_labels_not_model_or_materialization(self):
        from backend.app import DraftRequest, prepare_visual_text_evidence
        from backend.sales.tool_planner import ToolPlan
        for task in ['inspection','mixed']:
            with patch('backend.app.generate_multi_visual_response',side_effect=AssertionError('company pixels forbidden')):
                rows=prepare_visual_text_evidence(DraftRequest(customer_question='detail'),
                    ToolPlan(tools=['company_rag'],visual_task=task),
                    {'company_rag':{'visual_assets':[{'asset_id':'a','image_labels':{
                        'display_ready':True,'product_name':'registered product','surface_finish':'rough',
                        'color_name':'registered color','caption':'source caption'}}]}})
            self.assertEqual(rows[0]['description_basis'],'source_metadata_labels')
            self.assertIn('registered color',rows[0]['text'])
            self.assertIn('rough',rows[0]['text'])
            self.assertNotIn('pixel_tool_observation',rows[0]['text'])


if __name__=='__main__': unittest.main()
