import copy
import json
import unittest
from unittest.mock import Mock
from backend.documents.summary_context import allocate_summary_context, final_summary_payload, compact_overview
from backend.sales.prepared_support import fit_support_budget
from backend.sales.runtime_status import ExecutionStatus, classify, current_execution
from backend.sales.task_support import audit_execution_failed


class SummaryContextTests(unittest.TestCase):
    def test_allocation_precedes_inference_and_stricter_envelope_wins(self):
        for count in range(1, 5):
            budget, audit = allocate_summary_context(audit_tokens=lambda _:1000,
                final_tokens=lambda _:2000, audit_limit=8000, final_limit=6000, file_count=count)
            self.assertEqual(budget, min(2800, 6000-2000-280*count-512))
            self.assertEqual(audit['target_file_count'], count)
        with self.assertRaisesRegex(ValueError, 'before_summary'):
            allocate_summary_context(audit_tokens=lambda _:7900, final_tokens=lambda _:7000,
                audit_limit=8000, final_limit=8000, file_count=4)

    def test_both_exact_prechecks_have_distinct_limits(self):
        rows = [{'evidence_id':'U1','text':'source','required_document_coverage':True}]
        kept, audit = fit_support_budget(rows, audit_tokens=lambda _:500, generation_tokens=lambda _:900,
            ceiling=1000, audit_limit=600, final_limit=800)
        self.assertTrue(audit['fits_audit']); self.assertFalse(audit['fits_final'])
        self.assertFalse(audit['budget_satisfied']); self.assertEqual(kept, rows)

    def test_overview_can_compact_before_audit_but_all_files_and_hashes_survive(self):
        rows = [{'evidence_id':f'U{i}', 'document_id':str(i), 'text':'原文事实与条件。\n\n'*80,
            'overview_content':True, 'required_document_coverage':True,
            'coverage':{'processing_status':'complete'}} for i in range(1,5)]
        original = copy.deepcopy(rows)
        count = lambda values:sum(len(r['text']) for r in values)
        kept, audit = fit_support_budget(rows, audit_tokens=count, generation_tokens=count, ceiling=count(rows)*.8,
            compact_overview=lambda row:compact_overview(row,len))
        self.assertTrue(audit['budget_satisfied'])
        self.assertEqual({r['document_id'] for r in kept}, {'1','2','3','4'})
        self.assertTrue(audit['compacted_overview_ids']); self.assertEqual(rows, original)
        self.assertTrue(all(r.get('prepared_content_sha256') for r in kept if r.get('coverage',{}).get('prompt_excerpted')))

    def test_wire_has_limits_not_full_parser_diagnostics_or_citations(self):
        rows = [{'evidence_id':'U1','text':'可见原文', 'document_id':'d1',
            'citations':[{'section_heading':'verbose'*500}], 'original_source_refs':['private'],
            'coverage':{'processing_status':'partial', 'summary_status':'extractive_fallback_only',
                        'parse_quality':{'warnings':['limited'], 'other':'verbose'*1000}}}]
        result = final_summary_payload(question='概括', language='zh', date='2026-10-06',
            evidence=rows, aspects=['逐文件概括'], required=[{'document_id':'d1','document_name':'one.pdf'}],
            evaluation={'status':'partial','file_coverage':{'very_large':'x'*10000}})
        encoded = json.dumps(result)
        self.assertNotIn('verbose', encoded); self.assertNotIn('very_large', encoded)
        self.assertEqual(result['evidence'][0]['text'], '可见原文')
        self.assertEqual(result['evidence'][0]['coverage']['summary_status'],'extractive_fallback_only')

    def test_budget_failure_is_technical_and_invocation_flags_default_false(self):
        failure = classify(ValueError('context_budget_exceeded'),'evidence_evaluation')
        self.assertEqual(failure['code'],'CONTEXT_BUDGET_EXCEEDED')
        self.assertTrue(audit_execution_failed({'error_code':failure['code']}))
        snapshot = ExecutionStatus().snapshot()
        self.assertFalse(snapshot['audit_invoked']); self.assertFalse(snapshot['final_generation_invoked'])
        journal = ExecutionStatus()
        token = current_execution.set(journal)
        try:
            self.assertEqual(classify(ValueError('invalid_input'), 'evidence_evaluation')['code'], 'AUDIT_INPUT_INVALID')
            journal.audit_invoked = True
            self.assertEqual(classify(ValueError('invalid_output'), 'evidence_evaluation')['code'], 'AUDIT_SCHEMA_INVALID')
        finally:
            current_execution.reset(token)
