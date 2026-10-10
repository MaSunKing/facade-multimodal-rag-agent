import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.documents.summaries import (
    summary_evidence, SummaryBudget, SummaryLimit, build_packets, model_coverage, summary_notice,
)
from backend.request_budget import RequestBudget, RequestCancelled, budget_scope, check_budget


def document(key, text, chunks=None):
    return SimpleNamespace(document_id=key, file_name=key + '.txt', chunks=chunks or [
        {'chunk_id': key + ':1', 'kind': 'evidence', 'text': text, 'source_refs': []}],
        source_locations={}, parser='test', parse_quality={}, summary_cache={}, summary_cache_bytes=0)


class BoundedSummaryTests(unittest.TestCase):
    def test_one_to_four_short_files_bypass_summary_model(self):
        for number in range(1, 5):
            model = Mock(side_effect=AssertionError('unexpected inference'))
            docs = [document(str(i), 'short') for i in range(number)]
            rows, audit = summary_evidence(docs, summarize=model, language='zh', model_revision='test')
            self.assertEqual(len(rows), number)
            self.assertTrue(all(r['coverage']['processing_status'] == 'complete' for r in rows))
            self.assertEqual(audit['model_calls'], 0)

    def test_short_file_total_budget_not_fixed_per_file(self):
        model = Mock(side_effect=AssertionError('unexpected inference'))
        rows, audit = summary_evidence([document('a', 'a' * 900), document('b', 'b')],
            summarize=model, language='zh', model_revision='test', text_budget=1000, count_tokens=len)
        self.assertEqual(audit['model_calls'], 0)
        self.assertEqual(rows[0]['text'], 'a' * 900)

    def test_four_long_files_share_eight_calls_round_robin(self):
        docs = [document(key, key * 40000) for key in 'abcd']
        called = []
        def model(payload):
            called.append(payload['document_name'])
            return 'overview'
        rows, audit = summary_evidence(docs, summarize=model, language='zh', model_revision='test', count_tokens=len)
        self.assertEqual(called, [key + '.txt' for key in 'abcdabcd'])
        self.assertEqual(audit['model_calls'], 8)
        self.assertTrue(audit['budget_exhausted'])
        self.assertTrue(all(r['coverage']['processing_status'] == 'partial' for r in rows))
        self.assertTrue(all(r['coverage']['processed_chunks'] == 0 for r in rows))
        self.assertTrue(all(d.chunks[0]['text'] == d.document_id * 40000 for d in docs))

    def test_many_rows_no_summary_llm_and_no_claimed_content_coverage(self):
        table = '[TABLE id=t sheet=Revenue range=A1:B5001]\n[COLUMNS] A=区域 | B=收入\n' + '\n'.join(
            f"[ROW source=s] A{i}='北京' | B{i}='{i}'" for i in range(1, 5002))
        model = Mock(side_effect=AssertionError('table rows must not call Qwen'))
        rows, audit = summary_evidence([document('sheet', table)], summarize=model, language='zh', model_revision='test')
        self.assertEqual(audit['model_calls'], 0)
        self.assertEqual(rows[0]['coverage']['processing_status'], 'structural_overview')
        self.assertEqual(rows[0]['coverage']['processed_chunks'], 0)
        self.assertIn('完整内容分析', rows[0]['text'])
        self.assertFalse(rows[0]['coverage']['visual_pixels_reviewed'])

    def test_cancel_after_first_packet_retains_cache_and_resume_skips_it(self):
        doc = document('a', 'x' * 20000)
        parent = RequestBudget(float('inf'))
        calls = []
        def model(payload):
            calls.append(payload)
            if len(calls) == 2:
                parent.cancelled.set(); check_budget()
            return 'overview'
        with budget_scope(parent), self.assertRaises(RequestCancelled):
            summary_evidence([doc], summarize=model, language='zh', model_revision='test', count_tokens=len)
        self.assertEqual(len(doc.summary_cache), 1)
        again = Mock(return_value='overview')
        rows, audit = summary_evidence([doc], summarize=again, language='zh', model_revision='test', count_tokens=len)
        self.assertEqual(audit['documents'][0]['cache_hits'], 1)
        self.assertTrue(doc.summary_cache)

    def test_deadline_returns_missing_and_replan_does_not_refund_calls(self):
        budget = SummaryBudget(seconds=.01)
        budget.started -= 1
        model = Mock(return_value='overview')
        doc = document('a', 'x' * 10000)
        rows, audit = summary_evidence([doc], summarize=model, language='zh', model_revision='test', budget=budget, count_tokens=len)
        model.assert_not_called()
        self.assertEqual(rows[0]['coverage']['processing_status'], 'missing')
        self.assertIn('summary_deadline', rows[0]['coverage']['reason'])
        self.assertTrue(audit['budget_exhausted'])
        self.assertIn('未完成', summary_notice(audit))
        request = RequestBudget(float('inf'))
        with budget_scope(request):
            for _ in range(2):
                summary_evidence([doc], summarize=model, language='zh', model_revision='test', count_tokens=len)
        self.assertLessEqual(model.call_count, 8)
        self.assertLessEqual(request.summary_state.calls, 8)

    def test_incomplete_generation_not_cached_and_sections_spread(self):
        doc = document('a', '', chunks=[{'chunk_id': str(i), 'kind': 'evidence', 'text': 'x' * 10000,
            'source_refs': [str(i)]} for i in range(4)])
        doc.source_locations = {str(i): {'section_heading': f'chapter {i}'} for i in range(4)}
        packets, hierarchy = build_packets(doc, doc.chunks, count=len)
        self.assertEqual(len({p['spans'][0]['section_id'] for p in packets[:4]}), 4)
        rows, audit = summary_evidence([doc], summarize=lambda _: {'text': 'truncated', 'complete': False},
            language='zh', model_revision='test', count_tokens=len)
        self.assertEqual(doc.summary_cache, {})
        self.assertEqual(rows[0]['coverage']['processed_packets'], 0)
        self.assertEqual(audit['model_calls'], 1)

    def test_progress_and_compact_coverage_preserve_real_limits(self):
        progress = []
        doc = document('a', 'x' * 20000)
        rows, audit = summary_evidence([doc], summarize=lambda _: 'overview', language='en', model_revision='test',
            count_tokens=len, progress=progress.append)
        self.assertEqual(progress[-1]['status'], 'finished')
        self.assertIn('packet_count', progress[1])
        view = model_coverage(rows[0]['coverage'])
        self.assertNotIn('source_spans', view)
        self.assertEqual(view, model_coverage(view))

    def test_empty_parsed_file_is_missing_not_complete(self):
        rows, audit = summary_evidence([document('empty', '')], summarize=Mock(), language='zh', model_revision='test')
        self.assertEqual(rows[0]['coverage']['processing_status'], 'missing')
        self.assertFalse(rows[0]['facts_eligible'])

    def test_layout_table_prose_is_not_spreadsheet_structure_only(self):
        for extension in ['pdf', 'docx']:
            doc = document('layout', '[TABLE id=layout]\n' + '教育经历。项目经历。必要条件。\n' * 800)
            doc.file_name = 'layout.' + extension
            rows, audit = summary_evidence([doc], summarize=lambda _: '教育与项目经历，保留必要条件。',
                language='zh', model_revision='test', count_tokens=len, budget=SummaryBudget(max_calls=2))
            self.assertGreater(audit['model_calls'], 0)
            self.assertNotEqual(rows[0]['coverage']['summary_status'], 'table_structure_only')

    def test_failed_packet_diagnostics_and_honest_original_excerpt_fallback(self):
        doc = document('a', '可见的教育经历。项目内容和条件。\n' * 1000)
        def fail(payload):
            raise SummaryLimit('summary_generation_incomplete', {'input_tokens':1200,
                'output_tokens':256, 'finish_reason':'output_token_limit'})
        rows, audit = summary_evidence([doc], summarize=fail, language='zh', model_revision='test', count_tokens=len)
        self.assertFalse(audit['budget_exhausted'])
        self.assertTrue(audit['summary_incomplete'])
        coverage = rows[0]['coverage']
        self.assertEqual(coverage['successful_model_summary_packets'], 0)
        self.assertEqual(coverage['summary_status'], 'extractive_fallback_only')
        self.assertEqual(coverage['processed_packets'], 0)
        self.assertIn('模型摘要未完成', summary_notice(audit))
        packet = audit['documents'][0]['packets'][0]
        self.assertEqual(packet['finish_reason'], 'output_token_limit')
        self.assertEqual(packet['output_tokens'], 256)
        self.assertTrue(packet['model_called'])
        self.assertEqual(doc.summary_cache, {})

    def test_spreadsheet_synopsis_is_whole_json_records_not_prefix_cut(self):
        import json
        from backend.documents.summaries import table_synopsis
        chunks = [{'chunk_id':str(i), 'text':f'[TABLE id=t{i} sheet=S{i} range=A1:B900]\n'
            + '[ROW] A1=地域 | B1=收入\n' * 100} for i in range(6)]
        text, audit = table_synopsis(chunks, count=len, budget=1200)
        for line in text.splitlines()[1:]:
            self.assertIn('table', json.loads(line))
        self.assertEqual(audit['visible_table_count'], 6)
        self.assertLessEqual(len(text), 1200)

    def test_packet_builder_never_counts_entire_remaining_huge_file(self):
        seen = []
        def count(text): seen.append(len(text)); return len(text)
        doc = document('large', 'x' * 100000)
        packets, _ = build_packets(doc, doc.chunks, count=count)
        self.assertLessEqual(max(seen), 1400 * 8)
        self.assertEqual(sum(s['end']-s['start'] for p in packets for s in p['spans']), 100000)

    def test_expensive_summary_is_not_replanned_when_budget_exhausted(self):
        from backend import app
        from backend.sales.tool_planner import ToolPlan
        request = app.DraftRequest(customer_question='总结全部文件')
        request._summary_audit = {'budget_exhausted': True}
        result = app._CustomerAnswerWorkflowCallbacks.plan_retry(request, ToolPlan(),
            {'meta': {'mode': 'evidence_gate_blocked'}, 'customer_reply': ''})
        self.assertIsNone(result)
        self.assertEqual(request._replan_audit['reason'], 'summary_budget_exhausted_use_partial')
