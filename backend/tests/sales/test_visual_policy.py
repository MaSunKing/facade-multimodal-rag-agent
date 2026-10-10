import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from backend.sales.visual_policy import (
    asset_key, visual_candidates, validate_visual_audit, select_visual_support,
    materialize_company_assets, input_manifest, unavailable_visual_response,
)


class VisualPolicyTests(unittest.TestCase):
    def test_explicit_attachment_visual_requirement_survives_text_only_verdict(self):
        from backend.sales.visual_policy import enforce_requested_visual_support
        from backend.sales.evidence_gate import validate_evaluation
        candidates = [{'asset_id':'customer:d1:p'}, {'asset_id':'customer:d2:p'}]
        result = enforce_requested_visual_support(
            {'status':'supported','supported_evidence_ids':['T1'], 'visual_need':'not_required'},
            candidates, required=True)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['visual_need'], 'required')
        self.assertEqual(result['candidate_asset_ids'], ['customer:d1:p','customer:d2:p'])
        selected, _ = select_visual_support(
            [{'document_id':'d1','visual_id':'p'}, {'document_id':'d2','visual_id':'p'}], result)
        self.assertEqual(len(selected), 2)
        self.assertEqual(validate_evaluation(result, [{'evidence_id':'T1'}])['status'], 'partial')

    def test_visual_candidate_alone_is_not_support(self):
        from backend.sales.visual_policy import enforce_requested_visual_support
        from backend.sales.evidence_gate import validate_evaluation
        result = enforce_requested_visual_support(
            {'status':'partial','supported_evidence_ids':[], 'visual_need':'required'},
            [{'asset_id':'customer:d:p'}], required=True)
        self.assertEqual(validate_evaluation(result, [])['status'], 'unsupported')
        self.assertFalse(result['pixel_support_pending'])
        self.assertEqual(result['recovery_need'],'visual')

    def test_missing_pixels_cannot_be_completed_by_company_text(self):
        from backend.sales.visual_policy import enforce_requested_visual_support
        result = enforce_requested_visual_support(
            {'status':'supported','supported_evidence_ids':['T1']}, [], required=True)
        self.assertEqual(result['status'], 'unsupported')
        self.assertEqual(result['gap_type'], 'visual_input')
        company_only = enforce_requested_visual_support(
            {'status':'supported','supported_evidence_ids':['T1'],
             'candidate_asset_ids':['company:a']}, [{'asset_id':'company:a'}], required=True)
        self.assertEqual(company_only['status'], 'unsupported')
        self.assertEqual(company_only['candidate_asset_ids'], [])

    def test_unrelated_attachment_does_not_force_pixels(self):
        from backend.sales.visual_policy import enforce_requested_visual_support
        raw = {'status':'supported','supported_evidence_ids':['T1'], 'visual_need':'not_required'}
        self.assertEqual(enforce_requested_visual_support(raw,
            [{'asset_id':'customer:d:p'}], required=False), raw)

    def test_visual_generation_has_no_independent_default_timer(self):
        import inspect
        from backend.app import generate_visual_response, generate_multi_visual_response
        for function in (generate_visual_response, generate_multi_visual_response):
            self.assertIsNone(inspect.signature(function).parameters['max_seconds'].default)

    def test_company_labels_have_lightweight_checks_not_semantic_certification(self):
        from backend.sales.visual_policy import company_metadata_audit
        asset = {'asset_id':'a', 'customer_title':'Node detail', 'citation':{'document_name':'Manual'}}
        audit = company_metadata_audit([asset, asset, {'asset_id':'b'}])
        self.assertEqual(audit['model_calls'], 0)
        self.assertEqual(audit['duplicate_asset_ids'], ['a'])
        self.assertEqual(audit['incomplete_label_asset_ids'], ['b'])
        self.assertFalse(audit['semantic_verified'])
        self.assertFalse(audit['pixel_contents_verified'])
    def test_false_planner_flag_cannot_veto_pixels(self):
        asset = {'document_id':'doc', 'visual_id':'page9'}
        selected, audit = select_visual_support([asset], {'document_visual_required':False, 'visual_need':'required'})
        self.assertEqual(selected, [asset])
        self.assertFalse(audit['planner_hint_is_veto'])

    def test_gallery_metadata_does_not_require_vlm(self):
        selected, audit = select_visual_support([{'asset_id':'product'}], {'visual_need':'not_required'})
        self.assertEqual(selected, [])
        self.assertFalse(audit['required_but_missing'])

    def test_ocr_pixel_verification_is_protected(self):
        asset = {'document_id':'doc', 'visual_id':'page9'}
        selected, _ = select_visual_support([asset], {'visual_need':'not_required'}, mandatory_keys=[asset_key(asset)])
        self.assertEqual(selected, [asset])

    def test_sources_share_remaining_budget(self):
        assets = [{'document_id':'d', 'visual_id':str(i)} for i in range(4)] + [{'asset_id':'company'}]
        selected, audit = select_visual_support(assets, {}, limit=2)
        self.assertEqual([asset_key(a) for a in selected], ['customer:d:0', 'customer:d:1'])
        self.assertEqual(len(audit['budget_excluded_asset_ids']), 2)

    def test_company_inspection_and_dispute_never_schedule_pixels(self):
        for purpose in ['content_inspection','label_dispute','unreadable']:
            selected,audit=select_visual_support([{'asset_id':'company','image_labels':{'label_issues':['disputed']}}],
                {'visual_need':'required','visual_purpose':purpose},mandatory_keys=['company:company'])
            self.assertEqual(selected,[])
            self.assertFalse(audit['required_but_missing'])

    def test_company_gap_does_not_block_supported_text_or_affect_customer_gap(self):
        from backend.sales.visual_policy import enforce_company_label_policy
        raw={'status':'partial','supported_evidence_ids':['T1'],'missing_aspects':['image detail'],
             'visual_need':'required','visual_purpose':'content_inspection','recovery_need':'visual','gap_type':'visual_input'}
        value=enforce_company_label_policy(raw,company_only=True)
        self.assertEqual(value['status'],'partial')
        self.assertEqual(value['missing_aspects'],['image detail'])
        self.assertEqual(value['visual_need'],'not_required')
        self.assertEqual(value['recovery_need'],'none')
        self.assertEqual(enforce_company_label_policy(raw,company_only=False),raw)

    def test_auditor_cannot_invent_assets_or_paths(self):
        with self.assertRaisesRegex(ValueError, 'invalid_visual_audit_reference'):
            validate_visual_audit({'visual_need':'required', 'candidate_asset_ids':['company:a', '/tmp/secret', {}]}, [{'asset_id':'company:a'}])

    def test_invalid_need_remains_uncertain(self):
        with self.assertRaisesRegex(ValueError, 'invalid_visual_audit_need'):
            validate_visual_audit({'visual_need':'yes'}, [])

    def test_candidates_are_metadata_not_pixel_evidence(self):
        rows = visual_candidates({'customer_documents':{'selected_visuals':[{'document_id':'d', 'visual_id':'p'}]},
                                  'company_rag':{'visual_assets':[{'asset_id':'a'}]}})
        self.assertEqual([a['asset_id'] for a in rows], ['customer:d:p', 'company:a'])
        self.assertTrue(all(not a['pixels_supplied'] for a in rows))

    def test_actual_pixels_and_all_duplicate_source_ids_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'
            Image.new('RGB', (12, 12), 'blue').save(path)
            assets, errors = materialize_company_assets([{'asset_id':'a', 'image_path':'untrusted'}],
                                                       lambda key: {'image_path':str(path)})
            self.assertEqual(errors, [])
            self.assertEqual(assets[0]['image_bytes'], path.read_bytes())
            assets[0]['evidence_id'] = 'V1'
            rows = input_manifest([path], assets + [{**assets[0], 'evidence_id':'V2'}])
            self.assertEqual(rows[0]['image_number'], 1)
            self.assertEqual([c['evidence_id'] for c in rows[0]['source_candidates']], ['V1','V2'])
            self.assertEqual(rows[0]['source_candidates'][0]['origin'], 'company_rag')
            self.assertNotIn(str(path), json.dumps(rows))

    def test_not_authorized_does_not_use_supplied_path(self):
        assets, errors = materialize_company_assets([{'asset_id':'private', 'image_path':'C:/secret.png'}], lambda _: None)
        self.assertEqual(assets, [])
        self.assertEqual(errors[0]['error'], 'not_indexed_or_not_authorised')

    def test_bad_image_never_becomes_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.png'
            path.write_bytes(b'not an image')
            assets, errors = materialize_company_assets([{'asset_id':'a'}], lambda _: {'image_path':str(path)})
            self.assertEqual(assets, [])
            self.assertEqual(len(errors), 1)

    def test_missing_pixels_cannot_fallback_to_metadata_claims(self):
        from backend.app import AnswerResponse
        value = AnswerResponse(**unavailable_visual_response({'asset_failures':[{'asset_id':'a'}]}))
        self.assertFalse(value.answerable)
        self.assertEqual(value.citations, [])
        self.assertEqual(value.image_observations, [])
        self.assertFalse(value.meta['model_used'])

    def test_existing_evidence_audit_defers_visual_support_not_verifies_it(self):
        from backend.app import DraftRequest, _CustomerAnswerWorkflowCallbacks
        from backend.sales.tool_planner import ToolPlan
        request = DraftRequest(customer_question='Explain the structural relationships in the diagram')
        with patch('backend.app.evaluate_task_support', return_value={
                'status':'unsupported', 'supported_evidence_ids':[], 'visual_need':'required',
                'candidate_asset_ids':['company:a'], 'visual_reason':'layout requires pixels'}) as model:
            response = _CustomerAnswerWorkflowCallbacks.evaluate_sources(request,
                ToolPlan(tools=['company_rag']), {'company_rag':{'visual_assets':[{'asset_id':'a'}]}}, 0)
        self.assertIsNone(response)  # Context preparation no longer calls Audit.
        model.assert_not_called()
        self.assertEqual(request._evidence_gate_audit['phase'],'draft_context_only')
        self.assertFalse(request._evidence_gate_audit.get('pixel_support_pending',False))
        self.assertFalse(request._evidence_gate_audit['semantic_verified'])

    def test_pixel_candidates_are_not_misreported_as_empty_text(self):
        from backend.sales.recovery_policy import source_diagnostics
        states = source_diagnostics({'customer_documents':{'evidence':[], 'selected_visuals':[{'visual_id':'p'}]},
                                    'company_rag':{'visual_assets':[{'asset_id':'a'}]}})
        self.assertEqual(states['customer_documents']['state'], 'ready')
        self.assertEqual(states['company_rag']['state'], 'ready')

    def test_visual_failure_never_returns_unrelated_source_snippets(self):
        from backend.app import DraftRequest, fallback_grounded_answer
        request = DraftRequest(customer_question='Explain the drawing')
        request._evidence_gate_audit = {'status':'supported', 'visual_need':'required'}
        result = fallback_grounded_answer(request, {'text_evidence':[{'text':'Unrelated standard summary'}]}, 'generation_failed')
        self.assertFalse(result['answerable'])
        self.assertEqual(result['citations'], [])

    def test_server_metadata_does_not_invalidate_an_otherwise_grounded_answer(self):
        from backend.app import normalise_nonfactual_output_fields, is_safe_grounded_answer
        from backend.sales.answer_coverage import canonicalise_known_citation_ids
        value = {'intent':'document_qa', 'answerable':True, 'customer_reply':'A is blue.',
                 'citations':['V1'], 'key_points':['A is blue.'], 'normalized_terms':[],
                 'missing_information':[], 'risk_warnings':[], 'next_action':'Done',
                 'image_observations':['[image1|V1] A is blue.'],
                 'task_support':{'status':'supported'}, 'memory_updates':[]}
        result = canonicalise_known_citation_ids(normalise_nonfactual_output_fields(value), {'V1':{}})
        self.assertTrue(is_safe_grounded_answer(result, {'V1'}, allow_image_only=True))
        self.assertIn('task_support', value)
        with self.assertRaises(ValueError):
            canonicalise_known_citation_ids({**result, 'citations':['V9']}, {'V1':{}})
        self.assertFalse(is_safe_grounded_answer({**result, 'invented_field':True}, {'V1'}, allow_image_only=True))

    def test_pre_pixel_missing_reason_is_not_a_current_fact(self):
        from backend.sales.visual_policy import generation_evaluation
        before = {'status':'unsupported', 'pixel_support_pending':True, 'visual_need':'required',
                  'problem':'No pixels supplied'}
        after = generation_evaluation(before, 2)
        self.assertEqual(before['problem'], 'No pixels supplied')
        self.assertEqual(after['actual_pixel_count'], 2)
        self.assertFalse(after['semantic_verified'])
        self.assertEqual(after['status'], 'uncertain')
        self.assertEqual(generation_evaluation(before, 0), before)


if __name__ == '__main__':
    unittest.main()
