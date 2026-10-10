import copy
import hashlib
import unittest
from unittest.mock import Mock, patch

from backend.sales.prepared_support import prepare_support, support_signature
from backend.sales.evidence_gate import budget_audit_payload
from backend.sales.runtime_status import ExecutionStatus, current_execution, timed_phase


class PreparedSupportTests(unittest.TestCase):
    def test_requested_gallery_manifest_survives_without_certifying_facts(self):
        from backend.sales.prepared_support import gallery_label_evidence, fit_support_budget
        gallery = gallery_label_evidence([{'image_labels':{'product_name':'Sample A'}}])
        rows = [{'evidence_id':'T1','source_type':'company','text':'real product introduction'},
                gallery, {'evidence_id':'V1','source_type':'visual','text':'redundant image card'}]
        counter = lambda selected: len(selected) * 100
        kept, audit = fit_support_budget(rows, audit_tokens=counter, generation_tokens=counter,
            ceiling=200, target_tokens=200, question='Introduce the product and show images')
        self.assertTrue(audit['budget_satisfied'])
        self.assertEqual({r['evidence_id'] for r in kept}, {'T1','G1'})
        self.assertFalse(gallery['facts_eligible'])
        self.assertEqual(gallery['claim_scope'], 'asset_identity_only_not_performance')

    def test_generation_budget_counts_verdict_not_full_audit_telemetry(self):
        import json
        from backend.app import compact_grounded_payload_for_generation
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):return messages[-1]['content']
            def __call__(self,text,**kwargs):return {'input_ids':list(text)}
        payload={'evidence':[{'evidence_id':'T1','text':'actual fact'}],
                 'audited_evidence_ids':['T1'],
                 'evidence_evaluation':{'status':'supported','supported_evidence_ids':['T1'],
                    'prompt_budget_preflight':{'compression':'x'*10000},'runtime_profile':{'gpu':'y'*10000}}}
        raw,audit=compact_grounded_payload_for_generation(payload,Tokenizer(),max_prompt_tokens=500)
        self.assertTrue(audit['budget_satisfied'])
        wire=json.loads(raw)
        self.assertEqual(wire['evidence_evaluation'],{'status':'supported','supported_evidence_ids':['T1']})
        self.assertEqual(wire['evidence'][0]['text'],'actual fact')
        self.assertIn('runtime_profile',payload['evidence_evaluation'])

    def test_incidental_documents_and_numeric_conditions_are_not_unlimited_locks(self):
        from backend.sales.prepared_support import fit_support_budget
        rows=[{'evidence_id':'a','source_type':'company_rag','document_name':'A',
               'text':'company overview','goal_support_scores':{'identity':1}},
              {'evidence_id':'b','source_type':'company_rag','document_name':'B',
               'text':'numeric qualified statement '*20,'protected_relation_types':['entity_value','condition'],
               'protected_goal_ids':['B:identity']}]
        count=lambda selected:sum(len(r['text']) for r in selected)
        kept,status=fit_support_budget(rows,audit_tokens=count,generation_tokens=count,
            ceiling=100,question='company overview')
        self.assertTrue(status['budget_satisfied'])
        self.assertEqual([r['evidence_id'] for r in kept],['a'])
        self.assertEqual(rows[1]['text'],'numeric qualified statement '*20)

    def test_explicit_target_files_remain_reserved_in_question_aware_packing(self):
        from backend.sales.prepared_support import fit_support_budget
        rows=[{'evidence_id':key,'document_id':key,'document_name':key,
               'source_type':'customer_document','text':'content '+key} for key in ['a','b']]
        count=lambda selected:sum(len(r['text']) for r in selected)
        kept,status=fit_support_budget(rows,audit_tokens=count,generation_tokens=count,
            ceiling=10,question='summarize',required_document_ids=['a','b'])
        self.assertFalse(status['budget_satisfied'])
        self.assertEqual(len(kept),2)

    def test_collection_represents_catalog_bucket_without_protecting_company_baseline(self):
        from backend.sales.prepared_support import fit_support_budget
        from backend.sales.evidence_packing import model_evidence_view
        rows=[{'evidence_id':'identity','source_type':'catalog_sql','text':'x'*60},
              {'evidence_id':'members','source_type':'catalog_sql','text':'A B',
               'catalog_record_type':'product_collection','entity_coverage_ids':['A','B']}]
        count=lambda selected:sum(len(r['text']) for r in selected)
        kept,audit=fit_support_budget(rows,audit_tokens=count,generation_tokens=count,ceiling=10)
        self.assertTrue(audit['budget_satisfied'])
        self.assertEqual([r['evidence_id'] for r in kept],['members'])
        wire=model_evidence_view({'text':'value=12mm','support_audit':{'heuristic':0.2}})
        self.assertEqual(wire,{'text':'value=12mm'})

    def test_budget_is_fitted_before_audit_without_changing_text(self):
        from backend.sales.prepared_support import fit_support_budget
        rows = [{'evidence_id':'U1','document_name':'one.pdf','text':'a'*100,'protected_goal_ids':['g1']},
                {'evidence_id':'U2','document_name':'two.xls','text':'b'*100,'protected_goal_ids':['g2']},
                {'evidence_id':'U3','document_name':'two.xls','text':'c'*100}]
        original = copy.deepcopy(rows)
        count = lambda selected:sum(len(r['text']) for r in selected)
        kept, audit = fit_support_budget(rows, audit_tokens=count,
            generation_tokens=lambda selected:count(selected)+50, ceiling=260)
        self.assertTrue(audit['budget_satisfied'])
        self.assertEqual(audit['removed_evidence_ids'], ['U3'])
        self.assertEqual({r['document_name'] for r in kept}, {'one.pdf','two.xls'})
        self.assertEqual(rows, original)

    def test_shared_budget_does_not_erase_required_document(self):
        from backend.sales.prepared_support import fit_support_budget
        rows = [{'evidence_id':'U1','document_name':'one.pdf','text':'a'*100},
                {'evidence_id':'U2','document_name':'two.xls','text':'b'*100}]
        kept, audit = fit_support_budget(rows, audit_tokens=lambda _:200,
            generation_tokens=lambda _:300, ceiling=250)
        self.assertFalse(audit['budget_satisfied'])
        self.assertEqual(kept, rows)

    def test_attachment_budget_uses_hardware_ceiling_not_scope_cap(self):
        from backend.sales.context_engine import choose_context_budget
        for scope in ['local_lookup','whole_document','cross_document']:
            budget = choose_context_budget({'tools':['customer_documents'],'document_scope':scope},
                                           has_documents=True, has_image=False)
            self.assertEqual(budget.max_prompt_tokens, 8000)
        self.assertEqual(choose_context_budget({'tools':['company_rag']},
            has_documents=False,has_image=False).max_prompt_tokens, 6500)

    def test_cleaned_support_has_separate_hash_and_packs_unchanged(self):
        import json
        from backend.app import compact_grounded_payload_for_generation
        from backend.sales.context_engine import validate_packed_evidence
        text = '[ROW source=xls; sheet=Sheet1] A1=2025 income'
        source_hash = hashlib.sha256(text.encode()).hexdigest()
        original = {'evidence_id':'U1', 'text':text, 'content_sha256':source_hash}
        rows, _ = prepare_support('income', [original])
        self.assertEqual(original['text'], text)
        self.assertEqual(rows[0]['content_sha256'], source_hash)
        self.assertNotEqual(rows[0]['prepared_content_sha256'], source_hash)
        self.assertTrue(validate_packed_evidence(rows, rows)['valid'])
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return messages[-1]['content']
            def __call__(self, text, **kwargs):
                return {'input_ids':list(text)}
        raw, audit = compact_grounded_payload_for_generation(
            {'evidence':rows, 'audited_evidence_ids':['U1']}, Tokenizer(), max_prompt_tokens=4000)
        self.assertTrue(audit['integrity_check']['valid'])
        self.assertEqual(json.loads(raw)['evidence'][0]['text'], rows[0]['text'])
        self.assertNotIn('prepared_content_sha256', json.loads(raw)['evidence'][0])

    def test_prepared_hash_blocks_changed_text_even_if_packed_hash_replaced(self):
        from backend.sales.context_engine import validate_packed_evidence
        rows, _ = prepare_support('income', [{'evidence_id':'U1', 'text':'income 20 not 30'}])
        changed = {**rows[0], 'text':'income 30'}
        changed['prepared_content_sha256'] = hashlib.sha256(changed['text'].encode()).hexdigest()
        result = validate_packed_evidence(rows, [changed])
        self.assertFalse(result['valid'])
        self.assertEqual(result['violations'][0]['type'], 'prepared_content_hash_mismatch')

    def test_prepared_snapshot_keeps_conflict_group_checks(self):
        from backend.sales.context_engine import validate_packed_evidence
        rows, _ = prepare_support('income', [
            {'evidence_id':key, 'text':'income '+value, 'packing_group_id':'conflict_1',
             'protected_relation_types':['conflict_group']}
            for key,value in [('U1','20'),('U2','30')]])
        result = validate_packed_evidence(rows, rows[:1])
        self.assertFalse(result['valid'])
        self.assertEqual(result['violations'][0]['type'], 'partial_conflict_group')

    def test_integrity_failure_details_survive_outer_exception_without_duplicate(self):
        from backend.sales.runtime_status import report_error
        journal = ExecutionStatus()
        token = current_execution.set(journal)
        try:
            details = {'valid':False, 'violations':[{'type':'unknown_evidence_id','evidence_id':'U9'}]}
            report_error(code='CONTEXT_INTEGRITY_FAILED', stage='compose_evidence', details=details)
            report_error(ValueError('context_integrity_failed'), stage='generate_answer')
            self.assertEqual(len(journal.errors), 1)
            self.assertEqual(journal.errors[0]['details'], details)
            self.assertEqual(journal.errors[0]['stage'], 'compose_evidence')
        finally:
            current_execution.reset(token)

    def test_protected_group_and_conditions_are_not_split(self):
        rows = [dict(evidence_id='T1', text='Panel thickness 20 mm, not for immersion.', packing_group_id='conflict_pack_1'),
                dict(evidence_id='T2', text='Panel thickness 18 mm, dry substrate only.', packing_group_id='conflict_pack_1'),
                dict(evidence_id='T3', text='unrelated ' * 900)]
        original = copy.deepcopy(rows)
        selected, audit = prepare_support('Panel thickness', rows)
        self.assertEqual(rows, original)
        self.assertEqual({r['evidence_id'] for r in selected}, {'T1', 'T2'})
        self.assertIn('not for immersion', selected[0]['text'])
        self.assertFalse(audit['score_is_support_verdict'])

    def test_different_requested_aspects_preserve_representatives(self):
        rows = [dict(evidence_id='U1', text='capacity calculation: 20 litres'),
                dict(evidence_id='U2', text='approval must precede storage')]
        selected, _ = prepare_support('Explain capacity and approval', rows,
                                      goals=['capacity calculation', 'approval requirements'])
        self.assertEqual(len(selected), 2)

    def test_cross_language_does_not_discard_everything(self):
        selected, _ = prepare_support('自转时间是多少', [dict(evidence_id='P1', text='Rotation period 11 hours')])
        self.assertEqual(len(selected), 1)

    def test_cache_ignores_timing_not_evidence_content(self):
        a = {'evidence':[{'evidence_id':'T1','text':'20 mm'}], 'elapsed_ms':20}
        b = {**a, 'elapsed_ms':90}
        self.assertEqual(support_signature('width', a, []), support_signature('width', b, []))
        b['evidence'] = [{'evidence_id':'T1','text':'18 mm'}]
        self.assertNotEqual(support_signature('width', a, []), support_signature('width', b, []))

    def test_gallery_summary_does_not_import_engineering_claims(self):
        from backend.sales.prepared_support import gallery_label_evidence
        row = gallery_label_evidence([{'customer_title':'Finish A',
            'image_labels':{'product_name':'Panel', 'variant_or_code':'Finish A'},
            'linked_text_evidence':[{'text':'maximum thickness 8 mm'}]}])
        self.assertIn('Finish A', row['text'])
        self.assertNotIn('8 mm', row['text'])
        self.assertEqual(row['claim_scope'], 'asset_identity_only_not_performance')

    def test_audit_cannot_silently_drop_prepared_rows(self):
        payload = {'prepared_support_contract':True, 'evidence':[{'evidence_id':'T1','text':'large'}]}
        with self.assertRaisesRegex(ValueError, 'audit_input_budget_exceeded'):
            budget_audit_payload(payload, lambda _:100, max_tokens=10)
        self.assertEqual(len(payload['evidence']), 1)

    def test_final_pack_cannot_shorten_or_drop_audited_text(self):
        import json
        from backend.app import compact_grounded_payload_for_generation
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return messages[-1]['content']
            def __call__(self, text, **kwargs):
                return {'input_ids':list(text)}
        text = 'Keep the restriction: not for immersion. ' * 50
        raw, audit = compact_grounded_payload_for_generation(
            {'evidence':[{'evidence_id':'T1','text':text}], 'audited_evidence_ids':['T1']},
            Tokenizer(), max_prompt_tokens=100)
        self.assertFalse(audit['budget_satisfied'])
        self.assertEqual(json.loads(raw)['evidence'][0]['text'], text)
        self.assertEqual(audit['removed_evidence_ids'], [])

    def test_finished_timings_are_frozen(self):
        with patch('backend.sales.runtime_status.time.perf_counter', return_value=10):
            journal = ExecutionStatus()
            journal._stage_started = journal._request_started = 10
        with patch('backend.sales.runtime_status.time.perf_counter', return_value=12):
            journal.enter('generate_answer')
        with patch('backend.sales.runtime_status.time.perf_counter', return_value=15):
            journal.finish(True)
        with patch('backend.sales.runtime_status.time.perf_counter', return_value=30):
            snapshot = journal.snapshot()
        self.assertEqual(snapshot['elapsed_ms'], 5000)
        self.assertFalse(any(t.get('in_progress') for t in snapshot['stage_timings']))
        self.assertEqual(snapshot['stage_timings'][-1]['elapsed_ms'], 3000)

    def test_phase_failure_still_records_duration(self):
        journal = ExecutionStatus()
        token = current_execution.set(journal)
        try:
            with self.assertRaises(ValueError), timed_phase('image_preprocessing'):
                raise ValueError('failed')
        finally:
            current_execution.reset(token)
        self.assertEqual(journal.timings[-1]['phase'], 'image_preprocessing')

    def test_rerank_pair_cache_reuses_scores(self):
        import time
        from backend.request_budget import RequestBudget, budget_scope
        from backend.sales.goal_reranking import rerank_goal_evidence
        scorer = Mock()
        scorer.score.return_value = [0.8]
        rows = [dict(evidence_id='T1', text='Panel thickness 20 mm')]
        with budget_scope(RequestBudget(time.monotonic()+200)):
            rerank_goal_evidence('Panel thickness', rows, goals=[], scorer=scorer)
            _, audit = rerank_goal_evidence('Panel thickness', rows, goals=[], scorer=scorer)
        scorer.score.assert_called_once()
        self.assertEqual(audit['cached_pair_count'], 1)


if __name__ == '__main__':
    unittest.main()
