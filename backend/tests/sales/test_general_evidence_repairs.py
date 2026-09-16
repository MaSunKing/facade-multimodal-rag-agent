import time
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from backend.sales.evidence_packing import compact_source_text, model_payload_view
from backend.sales.answer_integrity import reject_stripped_numeric_answer
from backend.sales.goal_reranking import rerank_goal_evidence
from backend.document_parsing.document_ocr import pack_ocr_index_lines
from backend.request_budget import RequestBudget, budget_scope, reserve_recovery


class GeneralEvidenceRepairTests(unittest.TestCase):
    def test_native_metadata_survives_provenance_compaction(self):
        raw = '[BLOCK id=R1 kind=derived_text source=file.xlsx;sheet=Sheet1 metadata={"formula":"A1>0"}] rule exists'
        compact = compact_source_text(raw)
        self.assertIn('A1>0', compact)
        self.assertIn('kind=derived_text', compact)
        self.assertNotIn('source=file.xlsx', compact)
        self.assertTrue(compact.endswith('rule exists'))

    def test_only_absent_headers_are_removed(self):
        raw = '[COLUMNS] A=name | B=currency | C=value | D=extra\n[ROW] A7[name]=\'Panel\' | B7[currency]=\'EUR\' | C7[value]=3.25'
        compact = compact_source_text(raw)
        self.assertIn('B=currency', compact)
        self.assertIn("B7[currency]='EUR'", compact)
        self.assertIn('3.25', compact)
        self.assertNotIn('D=extra', compact)

    def test_wire_view_keeps_ids_and_fact_groups_not_rank_telemetry(self):
        original = dict(evidence_id='E', text='not for wet walls',
                        conflict_group_ids=['G'], source_location='page 8',
                        score=0.9, goal_support_scores={'goal':0.8})
        wire = model_payload_view({'evidence':[original]}, 'evidence')['evidence'][0]
        self.assertEqual(wire['text'], original['text'])
        self.assertEqual(wire['conflict_group_ids'], ['G'])
        self.assertEqual(wire['source_location'], 'page 8')
        self.assertNotIn('goal_support_scores', wire)
        self.assertIn('goal_support_scores', original)

    def test_raw_question_qualifiers_are_scored_independently(self):
        scorer = Mock(); scorer.score.return_value = [0.8]
        raw = 'Compare graduates over age 65, excluding tablets, in 2013.'
        _, audit = rerank_goal_evidence(raw, [dict(evidence_id='E',text='source')],
                                        goals=['compare devices'], scorer=scorer)
        self.assertEqual(audit['goals'][0], raw)
        self.assertEqual(scorer.score.call_args_list[0].args[0], raw)

    def test_query_normalization_and_execution_each_have_one_slot(self):
        budget = RequestBudget(time.monotonic()+100)
        with budget_scope(budget):
            self.assertTrue(reserve_recovery(stage='plan_request', action='repair_query_language'))
            self.assertFalse(reserve_recovery(stage='plan_request', action='repair_query_language'))
            self.assertTrue(reserve_recovery(stage='assess_retry', action='repair_visual'))
            self.assertFalse(reserve_recovery(stage='tool_retry', action='repair_web'))
        self.assertEqual(len(budget.recoveries), 2)

    def test_numeric_stripping_rejects_bare_intro(self):
        original = dict(customer_reply='In this study, the unsupported answer is 912 people.', answerable=True)
        result, audit = reject_stripped_numeric_answer(original,
                        dict(customer_reply='In this study,',answerable=True),dict(passed=True))
        self.assertFalse(result['answerable'])
        self.assertTrue(audit['numeric_answer_removed'])
        self.assertFalse(audit['passed'])

    def test_unchanged_short_answer_is_not_rejected(self):
        original = dict(customer_reply='EUR',answerable=True)
        result, _ = reject_stripped_numeric_answer(original, original, dict(passed=True))
        self.assertTrue(result['answerable'])

    def test_unrelated_visual_observation_does_not_exempt_numbers(self):
        from backend.app import evidence_support_audit
        result = dict(customer_reply='Store for 72 hours.',
                      image_observations=['A blue logo is visible.'],
                      citations=[dict(evidence_id='V')])
        audit = evidence_support_audit(result, {'V':dict(text='Blue logo',visual_id='v1')},
                                       allow_visual_observation=True)
        self.assertIn('72', audit['unsupported_numeric_claims'])

    def test_row_locator_does_not_masquerade_as_a_measurement(self):
        from backend.app import evidence_support_audit
        source = {'E':dict(text="[ROW] D23='EUR'")}
        result = dict(customer_reply='The currency in row 23 is EUR.',citations=[dict(evidence_id='E')])
        self.assertEqual(evidence_support_audit(result,source)['unsupported_numeric_claims'],[])
        result['customer_reply'] = 'The cost in row 23 is 23 euros.'
        self.assertIn('23',evidence_support_audit(result,source)['unsupported_numeric_claims'])
        result['customer_reply'] = 'The currency in row 24 is EUR.'
        self.assertIn('24',evidence_support_audit(result,source)['unsupported_numeric_claims'])
        result['customer_reply'] = '第23行的币种是EUR。'
        self.assertEqual(evidence_support_audit(result,source)['unsupported_numeric_claims'],[])
        result['customer_reply'] = '第23行的费用是23欧元。'
        self.assertIn('23',evidence_support_audit(result,source)['unsupported_numeric_claims'])

    def test_text_index_groups_all_lines_and_discloses_overflow(self):
        lines = ['line '+str(n) for n in range(100)]
        blocks, overflow = pack_ocr_index_lines(lines)
        self.assertFalse(overflow)
        self.assertEqual('\n'.join(blocks).splitlines(), lines)
        _, overflow = pack_ocr_index_lines(['x'*900 for _ in range(30)])
        self.assertTrue(overflow)

    def test_text_index_candidate_passes_intake_protocol_without_fact_writeback(self):
        from io import BytesIO
        from PIL import Image
        from backend.document_parsing.document_ingestion import ingest_image
        from backend.document_parsing.document_ocr import augment_with_document_ocr_candidates
        stream=BytesIO(); Image.new('RGB',(30,30)).save(stream,format='PNG')
        original=ingest_image('page.png',stream.getvalue())
        indexed=augment_with_document_ocr_candidates(original,lambda asset:dict(
            recognizer='paddle_text_index',text_blocks=['Literal field'],confidence=.92))
        self.assertEqual(indexed.vision_document_candidates[0].recognizer,'paddle_text_index')
        self.assertEqual(indexed.vision_document_candidates[0].status,'candidate_ready')
        self.assertTrue(indexed.vision_document_candidates[0].requires_confirmation)
        self.assertEqual(indexed.standard.facts,[])

    def test_optional_ocr_adapter_errors_are_explicit_not_silent(self):
        from io import BytesIO
        from PIL import Image
        from backend.document_parsing.document_ingestion import ingest_image
        from backend.documents.customer_sessions import _with_optional_document_ocr
        stream=BytesIO(); Image.new('RGB',(30,30)).save(stream,format='PNG')
        original=ingest_image('page.png',stream.getvalue())
        original=original.model_copy(update={'intermediate':original.intermediate.model_copy(update={'source_type':'pdf'})})
        with patch.dict('os.environ',{'CUSTOMER_DOCUMENT_OCR_ENABLED':'1'}), patch(
                'backend.document_parsing.document_ocr.augment_with_document_ocr_candidates',side_effect=ValueError('bad adapter')):
            result=_with_optional_document_ocr(original)
        self.assertIn('document_ocr_adapter_failed',[issue.code for issue in result.validation])
        self.assertEqual(result.intermediate.visual_assets[0].image_bytes,stream.getvalue())

    def test_expired_ocr_budget_keeps_original_page_and_reports_pending(self):
        from io import BytesIO
        from PIL import Image
        from backend.document_parsing.document_ingestion import ingest_image
        from backend.document_parsing.document_ocr import augment_with_document_ocr_candidates
        stream=BytesIO(); Image.new('RGB',(30,30)).save(stream,format='PNG')
        original=ingest_image('page.png',stream.getvalue())
        inference=Mock()
        indexed=augment_with_document_ocr_candidates(original,inference,deadline=time.monotonic()-1)
        inference.assert_not_called()
        self.assertEqual(indexed.vision_document_candidates[0].status,'unavailable')
        self.assertIn('pending',indexed.vision_document_candidates[0].message)
        self.assertEqual(indexed.intermediate.visual_assets[0].image_bytes,stream.getvalue())

    def test_wide_row_cells_never_split_away_from_column_identity(self):
        from backend.documents.customer_sessions import _window_chunk
        cells = [f"{chr(65+i)}8[field{i}]='value{i} " + 'word '*18 + "'" for i in range(15)]
        text = '[TABLE id=T sheet=Data]\n[COLUMNS] ' + ' | '.join(f'{chr(65+i)}=field{i}' for i in range(15)) + '\n[ROW] ' + ' | '.join(cells)
        windows = _window_chunk(text,'T',window_tokens=180,overlap_tokens=32)
        self.assertGreater(len(windows),1)
        for cell in cells:
            self.assertTrue(any(cell in window['text'] for window in windows))
        for window in windows:
            self.assertIn(cells[0], window['text'])
            self.assertIn(cells[1], window['text'])

    def test_neighbor_cell_offsets_cover_real_source_span(self):
        from backend.documents.customer_sessions import _window_chunk
        text = "[TABLE id=T sheet=Data]\n[ROW] A1='Item' | B1='JPY'\n[ROW] A2='Product' | B2='12.5'"
        windows = _window_chunk(text,'T',window_tokens=384,overlap_tokens=64)
        self.assertIn('JPY',windows[-1]['text'])
        self.assertEqual(windows[-1]['start_character'],text.index('[ROW]'))

    def test_native_rule_inventory_includes_empty_ranges_without_evaluation(self):
        from backend.document_parsing.ingestion import SourcePointer, conditional_formatting_evidence
        group = Mock(sqref='C3:F200')
        rule = SimpleNamespace(type='expression',operator=None,formula=['C3<>0'],
                               priority=7,stopIfTrue=True)
        class Rules:
            def __iter__(self): return iter([group])
            def __getitem__(self,key): return [rule]
        sheet = SimpleNamespace(title='条件',conditional_formatting=Rules())
        blocks = conditional_formatting_evidence(sheet,SourcePointer(source_type='excel',file_name='x.xlsx',sheet_name=sheet.title))
        self.assertEqual(len(blocks),2)
        self.assertEqual(blocks[1].source.cell,'C3:F200')
        self.assertEqual(blocks[1].metadata['formula'],['C3<>0'])
        self.assertEqual(blocks[1].metadata['evaluation_status'],'not_evaluated')
        self.assertTrue(blocks[1].metadata['stop_if_true'])

    def test_zero_rules_are_explicit_native_inventory_not_missing_evidence(self):
        from backend.document_parsing.ingestion import SourcePointer, conditional_formatting_evidence
        blocks = conditional_formatting_evidence(SimpleNamespace(title='Plain',conditional_formatting=[]),
                    SourcePointer(source_type='excel',file_name='x.xlsx'))
        self.assertEqual(len(blocks),1)
        self.assertEqual(blocks[0].metadata['conditional_formatting_group_count'],0)

    def test_native_page_visual_discovery_is_shared_with_continuation(self):
        import fitz
        from pathlib import Path
        from backend.document_parsing.pdf_ingestion import visual_recovery_page_numbers
        empty = SimpleNamespace(rect=fitz.Rect(0,0,100,100),
                                get_image_info=lambda: [],get_drawings=lambda: [])
        chart = SimpleNamespace(rect=fitz.Rect(0,0,100,100),
                                get_image_info=lambda: [],
                                get_drawings=lambda: [dict(rect=fitz.Rect(0,0,70,50))])
        pdf = Mock(); pdf.__enter__ = Mock(return_value=[empty,chart]); pdf.__exit__=Mock(return_value=False)
        with patch('fitz.open',return_value=pdf):
            pages = visual_recovery_page_numbers(Path('unused.pdf'),
                       [SimpleNamespace(page_number=1,parse_route='vision'),
                        SimpleNamespace(page_number=2,parse_route='direct_text')])
        self.assertEqual(pages,[1,2])

    def test_lean_wire_retains_protected_relations(self):
        wire = model_payload_view({'evidence':[dict(evidence_id='E',text='not for external use',
                protected_relation_types=['negation'],protected_goal_ids=['G'])]},'evidence')
        self.assertEqual(wire['evidence'][0]['protected_relation_types'],['negation'])
        self.assertNotIn('protected_goal_ids', wire['evidence'][0])
        self.assertEqual(wire['evidence'][0]['text'], 'not for external use')

    def test_metadata_cover_does_not_override_relevant_page(self):
        from backend.documents.customer_sessions import StoredVisualAsset, _select_visuals
        def asset(page,text):
            return StoredVisualAsset(str(page),'D','doc.pdf','document_page',
                dict(page_number=page,source_type='pdf'),'image/png',{},b'image',text)
        doc=SimpleNamespace(document_id='D',source_type='pdf',
                            visuals=[asset(1,'document cover'),asset(7,'crew fieldwork personnel count')])
        selected=[dict(document_id='D',text='[VISUAL id=1 metadata={}]',
                       citations=[dict(source_page=1)])]
        result=_select_visuals([doc],'fieldwork personnel count',selected,max_visuals=1,visual_required=True)
        self.assertEqual(result[0]['source']['page_number'],7)

    def test_numeric_filter_removes_unanchored_observation_numbers(self):
        from backend.app import remove_unsupported_numeric_sentences
        result=remove_unsupported_numeric_sentences(dict(answerable=False,
                customer_reply='Evidence is insufficient.',image_observations=['Cover date: 2099.']),['2099'])
        self.assertFalse(result['answerable'])
        self.assertEqual(result['image_observations'],[])
