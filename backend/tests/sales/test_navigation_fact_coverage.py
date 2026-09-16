import unittest
from backend.sales.context_engine import optimise_evidence_context, validate_packed_evidence
from backend.sales.answer_coverage import audit_answer_coverage, guard_normative_claims
from backend.sales.tool_planner import guard_plan
from backend.sales.fact_normalization import disagreement_groups


class NavigationFactCoverageTests(unittest.TestCase):
    def test_only_complete_factual_prefix_recovers_and_ids_remain_exact(self):
        from backend.app import recover_truncated_grounded_json
        from backend.sales.answer_coverage import canonicalise_known_citation_ids
        raw = '{"intent":"document_qa","answerable":true,"citations":["E"],"customer_reply":"capacity 21", "key_points'
        recovered = recover_truncated_grounded_json(raw)
        self.assertEqual(recovered['key_points'], [])
        known = canonicalise_known_citation_ids(recovered, {'E':dict(text='capacity 21')})
        self.assertEqual(known['citations'], [dict(evidence_id='E')])
        self.assertEqual(known['customer_reply'], 'capacity 21')
        with self.assertRaises(ValueError):
            canonicalise_known_citation_ids(recovered, {'OTHER':{}})
        with self.assertRaises(ValueError):
            recover_truncated_grounded_json(raw.replace('capacity 21",', 'capacity'))

    def test_navigation_never_wins_answer_ranking(self):
        items, audit = optimise_evidence_context('thickness', [
            dict(evidence_id='N', evidence_scope='document_index', text='thickness '*100),
            dict(evidence_id='E', text='Panel-Q thickness: 21mm')])
        self.assertEqual([i['evidence_id'] for i in items], ['E'])
        self.assertEqual(audit['excluded_navigation_evidence_ids'], ['N'])

    def test_entity_alias_unit_and_scope(self):
        def item(i, text, **kw):
            return dict(evidence_id=i, document_id=i, text=text, **kw)
        self.assertEqual(len(disagreement_groups([item('A','Panel-Q thickness: 21mm'), item('B','Panel-Q 板厚：19毫米')])), 1)
        self.assertEqual(disagreement_groups([item('A','Panel-Q thickness: 20mm'), item('B','Panel-Q 厚度：2厘米')]), [])
        self.assertEqual(disagreement_groups([item('A','Panel-Q thickness: 20mm'), item('B','Panel-R thickness: 18mm')]), [])
        self.assertEqual(disagreement_groups([item('A','Panel-Q thickness: 20mm', version='v1'), item('B','Panel-Q thickness: 18mm', version='v2')]), [])
        self.assertEqual(disagreement_groups([item('A','thickness: 20mm'), item('B','thickness: 18mm')]), [])

    def test_native_cells_and_same_source_are_not_guessed(self):
        a = dict(evidence_id='A', document_id='doc-a', text="[ROW] A8[product]='Panel-Q' | B8[板厚]='19毫米'")
        b = dict(evidence_id='B', document_id='doc-b', text="[ROW] A2[product]='Panel-Q' | B2[thickness]='21mm'")
        self.assertEqual(len(disagreement_groups([a,b])), 1)
        b['document_id'] = 'doc-a'
        self.assertEqual(disagreement_groups([a,b]), [])

    def test_header_bound_numeric_cell_and_separate_unit(self):
        a = dict(evidence_id='A', document_id='a', text="[COLUMNS] A=Product | B=Thickness | C=Unit\n[ROW] A2='BeamY' | B2='20' | C2='mm'")
        b = dict(evidence_id='B', document_id='b', text="[COLUMNS] A=Product | B=板厚 | C=Unit\n[ROW] A7='BeamY' | B7='18' | C7='毫米'")
        self.assertEqual(len(disagreement_groups([a,b])),1)

    def test_overlap_conflicts_pack_atomically(self):
        items, _ = optimise_evidence_context('dimensions', [
            dict(evidence_id='A',document_id='a',text='Panel-Q thickness: 20mm'),
            dict(evidence_id='B',document_id='b',text='Panel-Q thickness: 18mm\nPanel-Q width: 50mm'),
            dict(evidence_id='C',document_id='c',text='Panel-Q width: 60mm')])
        self.assertEqual(len({i['packing_group_id'] for i in items}), 1)
        self.assertFalse(validate_packed_evidence(items, items[:2])['valid'])

    def test_checklist_requires_actual_answer_span_and_cited_evidence(self):
        reply = dict(customer_reply='Capacity needs measurement. Approval evidence is absent.',
            citations=[dict(evidence_id='A')], answer_aspect_coverage=[
                dict(aspect_index=0,status='answered',evidence_ids=['A'],answer_quote='Capacity needs measurement.'),
                dict(aspect_index=1,status='evidence_missing',evidence_ids=[],answer_quote='Approval evidence is absent.')])
        audit = audit_answer_coverage(reply, ['capacity','approval'], {'A':dict(text='capacity')})
        self.assertTrue(audit['complete'])
        self.assertEqual(audit['covered_aspect_indices'], [0])
        reply['answer_aspect_coverage'][0]['answer_quote'] = 'invented answer'
        self.assertFalse(audit_answer_coverage(reply,['capacity','approval'],{'A':{}})['complete'])

    def test_unrelated_standard_citation_does_not_allow_compliance(self):
        result = dict(customer_reply='The logo is compliant. Verify the requirements.', citations=[dict(evidence_id='S')])
        cleaned, audit = guard_normative_claims(result, {'S':dict(text='ISO 9001 management systems')})
        self.assertNotIn('logo is compliant', cleaned['customer_reply'])
        self.assertTrue(audit['removed_claims'])
        self.assertIn('Verify', cleaned['customer_reply'])

    def test_planner_keeps_bilingual_queries_and_six_aspects(self):
        plan = guard_plan(dict(tools=['customer_documents'],retrieval_queries=['原问题','English query'],
            answer_aspects=['one','two','three','four','five','six']), has_documents=True,
            has_image=False, facade_related=False, web_allowed=False)
        self.assertEqual(len(plan.answer_aspects),6)
        self.assertEqual(plan.retrieval_queries,['原问题','English query'])

    def test_model_wire_omits_repeated_ranking_goals_but_not_fact_or_role(self):
        from backend.sales.evidence_packing import model_evidence_view
        original = dict(evidence_id='E', text='BeamY thickness 21 mm', facts_eligible=False,
            candidate_requires_pixel_verification=True, retrieval_aspects=['capacity','approval'],
            protected_goal_ids=['capacity'], protected_goal_terms=['capacity'])
        wire = model_evidence_view(original)
        self.assertEqual(wire['text'], original['text'])
        self.assertFalse(wire['facts_eligible'])
        self.assertNotIn('retrieval_aspects', wire)
        self.assertIn('retrieval_aspects', original)

    def test_ocr_literal_is_not_duplicated_inside_visual_metadata(self):
        from backend.documents.visual_layout import compact_visual_metadata
        source = dict(ocr_literal_text='count 52', ocr_verified=False, original_page_number=9)
        self.assertNotIn('ocr_literal_text', compact_visual_metadata(source))
        self.assertEqual(source['ocr_literal_text'], 'count 52')

    def test_english_numeric_filter_keeps_supported_sections_and_drops_invented_threshold(self):
        from backend.app import remove_unsupported_numeric_sentences
        result = dict(customer_reply='1. Measure generation and throughput. 2. Activate overflow at 80% occupancy for 2 hours. 3. Define approved alternate routes. 4. Keep treatment logs.', key_points=[])
        filtered = remove_unsupported_numeric_sentences(result, ['80%', '2'])
        self.assertNotIn('80%', filtered['customer_reply'])
        self.assertNotIn('2 hours', filtered['customer_reply'])
        self.assertIn('Measure generation', filtered['customer_reply'])
        self.assertIn('approved alternate routes', filtered['customer_reply'])
        self.assertIn('treatment logs', filtered['customer_reply'])

    def test_inline_english_list_and_known_filename_are_not_numeric_facts(self):
        from backend.app import evidence_support_audit
        evidence = {'E': dict(document_name='SAND2022-1046-O.docx',text='capacity generation records')}
        out = dict(customer_reply='1. Measure capacity. 2. Record generation. 3. Keep records from SAND2022-1046-O.docx.',citations=[dict(evidence_id='E')])
        self.assertEqual(evidence_support_audit(out,evidence)['unsupported_numeric_claims'], [])
        out['customer_reply'] += ' Limit storage to 24 hours.'
        self.assertEqual(evidence_support_audit(out,evidence)['unsupported_numeric_claims'], ['24'])


if __name__ == '__main__':
    unittest.main()
