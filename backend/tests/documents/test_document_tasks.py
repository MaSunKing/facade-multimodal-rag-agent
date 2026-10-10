import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from backend.documents.document_tasks import resolve_documents, file_aspects, audit_file_answers, enforce_file_support
from backend.documents.summaries import summary_evidence
from backend.documents.upload_policy import positive_setting
from backend.sales.prepared_support import prepare_support
from backend.sales.knowledge_hierarchy import rebuild_hierarchy, document_sections


def document(key, text, name='same.txt'):
    return SimpleNamespace(document_id=key, file_name=name, chunks=[{
        'chunk_id': 'c1', 'kind': 'evidence', 'text': text, 'source_refs': []}],
        source_locations={}, parser='test', source_type='text', parse_quality={}, summary_cache={}, summary_cache_bytes=0)


class DocumentTaskTests(unittest.TestCase):
    def test_backend_resolves_only_owned_manifest(self):
        docs = [document('a', 'one'), document('b', 'two')]
        selected, required = resolve_documents(docs, ['b'], scope='whole_document', operation='summary')
        self.assertEqual([d.document_id for d in selected], ['b'])
        self.assertEqual(required[0]['document_id'], 'b')
        with self.assertRaisesRegex(ValueError, 'unknown_target_document_id'):
            resolve_documents(docs, ['foreign'], scope='whole_document', operation='summary')

    def test_local_lookup_does_not_force_all_uploads(self):
        _, required = resolve_documents([document('a', 'one')], [], scope='local_lookup', operation='lookup')
        self.assertEqual(required, [])

    def test_summary_stage_is_registered_in_finite_graph(self):
        from backend.sales.answer_graph import build_customer_answer_graph
        from backend.sales.staged_execution import WorkflowStep
        class Callbacks:
            def plan_tools(self, request):
                return {'tools': ['customer_documents'], 'document_operation': 'summary'}
            def open_steps(self, request, plan):
                value = yield WorkflowStep('document_summary', lambda: 'file summary')
                return {'customer_reply': value, 'answerable': True, 'meta': {}}
        result = build_customer_answer_graph(Callbacks()).invoke({'request': SimpleNamespace(customer_question='summary')})
        self.assertEqual(result['response']['customer_reply'], 'file summary')

    def test_overview_does_not_load_similarity_scorer(self):
        from backend.sales.goal_reranking import rerank_goal_evidence
        scorer = Mock(side_effect=AssertionError('unexpected scorer'))
        rows = [{'evidence_id': 'U1', 'document_id': 'a', 'text': 'entire small document',
                 'required_document_coverage': True, 'overview_content': True}]
        kept, audit = rerank_goal_evidence('summarise', rows, goals=[], scorer=scorer)
        self.assertEqual(kept, rows)
        self.assertFalse(audit['applied'])

    def test_low_similarity_does_not_erase_required_files(self):
        rows = [{'evidence_id': 'U1', 'document_id': 'a', 'text': 'one', 'required_document_coverage': True,
                 'goal_support_scores': {'summarise': .01}},
                {'evidence_id': 'U2', 'document_id': 'b', 'text': 'two', 'required_document_coverage': True,
                 'goal_support_scores': {'summarise': .02}}]
        kept, audit = prepare_support('summarise', rows)
        self.assertEqual({r['document_id'] for r in kept}, {'a', 'b'})
        self.assertFalse(audit['score_is_support_verdict'])

    def test_packing_cannot_remove_or_reassign_required_file(self):
        from backend.sales.context_engine import validate_packed_evidence
        rows = [{'evidence_id': 'U1', 'document_id': 'a', 'text': 'first', 'required_document_coverage': True},
                {'evidence_id': 'U2', 'document_id': 'b', 'text': 'second', 'required_document_coverage': True}]
        self.assertFalse(validate_packed_evidence(rows, rows[:1])['valid'])
        self.assertFalse(validate_packed_evidence(rows, [rows[0], {**rows[1], 'document_id': 'a'}])['valid'])

    def test_file_check_rejects_citations_to_other_file(self):
        aspects, bindings = file_aspects([], [{'document_id': 'a', 'document_name': 'same.txt'},
                                             {'document_id': 'b', 'document_name': 'same.txt'}])
        evidence = {'U1': {'document_id': 'a'}}
        result = {'customer_reply': 'first second', 'citations': [{'evidence_id': 'U1'}],
            'answer_aspect_coverage': [{'aspect_index': i, 'status': 'answered',
                                      'answer_quote': 'first', 'evidence_ids': ['U1']} for i in range(2)]}
        audit = audit_file_answers(result, bindings, evidence)
        self.assertFalse(audit['complete']); self.assertEqual(audit['missing_document_ids'], ['b'])
        result['answer_aspect_coverage'][1] = {'aspect_index': 1, 'status': 'evidence_missing', 'answer_quote': 'second'}
        self.assertTrue(audit_file_answers(result, bindings, evidence)['complete'])

    def test_support_verdict_cannot_claim_missing_file_supported(self):
        verdict = enforce_file_support({'status': 'supported', 'supported_evidence_ids': ['U1']},
            [{'document_id': 'a', 'document_name': 'a'}, {'document_id': 'b', 'document_name': 'b'}],
            [{'evidence_id': 'U1', 'document_id': 'a'}])
        self.assertEqual(verdict['status'], 'partial')
        self.assertFalse(verdict['file_coverage']['complete'])

    def test_short_files_use_complete_content_without_inference(self):
        model = Mock(side_effect=AssertionError('must not load a model'))
        rows, audit = summary_evidence([document('a', 'first'), document('b', 'second')],
            summarize=model, language='en', model_revision='fake')
        self.assertEqual([r['text'] for r in rows], ['first', 'second'])
        self.assertFalse(model.called)
        self.assertTrue(all(d['policy'] == 'direct_complete_content' for d in audit['documents']))

    def test_long_file_maps_every_chunk_and_caches_in_session(self):
        doc = document('a', '内容 ' * 3000)
        doc.chunks.append({'chunk_id': 'c2', 'kind': 'evidence', 'text': '最后部分 ' * 500, 'source_refs': []})
        model = Mock(return_value='内容概括，无新增结论。')
        rows, audit = summary_evidence([doc], summarize=model, language='zh', model_revision='fake', text_budget=300)
        self.assertGreater(model.call_count, 1)
        self.assertEqual(rows[0]['coverage']['processed_chunks'], 2)
        self.assertTrue(rows[0]['derived_summary'])
        count = model.call_count
        summary_evidence([doc], summarize=model, language='zh', model_revision='fake', text_budget=300)
        self.assertEqual(model.call_count, count)
        self.assertTrue(doc.chunks[1]['text'].startswith('最后部分'))

    def test_invalid_settings_are_not_silently_unlimited(self):
        with patch.dict('os.environ', {'TEST_FILE_LIMIT': '0'}):
            with self.assertRaises(ValueError):
                positive_setting('TEST_FILE_LIMIT', 4, 32)

    def test_company_navigation_is_acl_filtered_and_has_no_new_facts(self):
        with tempfile.TemporaryDirectory() as directory:
            index, database = Path(directory) / 'index.json', Path(directory) / 'catalog.sqlite3'
            index.write_text(json.dumps({'documents': [
                {'id': 'E1', 'kind': 'text', 'access_scope': 'internal', 'source_refs': [
                    {'document_id': 'doc1', 'document_name': 'manual', 'source_page': 1}]}]}), encoding='utf-8')
            report = rebuild_hierarchy(index, database)
            self.assertEqual(report['evidence_links'], 1)
            self.assertEqual(document_sections(database, 'doc1'), [])
            rows = document_sections(database, 'doc1', allowed_access_scopes=['internal'])
            self.assertEqual(rows[0]['kind'], 'page')
            self.assertEqual(rows[0]['evidence_ids'], ['E1'])


if __name__ == '__main__':
    unittest.main()
