"""Selection contracts, not a benchmark of model semantic correctness."""
import copy
import unittest
from unittest.mock import Mock

from backend.sales.web_context import web_evidence_row
from backend.sales.prepared_support import prepare_support
from backend.sales.goal_reranking import rerank_goal_evidence
from backend.sales.evidence_gate import compact_audit_evidence
from backend.sales.runtime_status import ExecutionStatus, current_execution


class WebTaskSelectionTests(unittest.TestCase):
    def row(self, i, text, **fields):
        return web_evidence_row(dict(source_id=i, title=fields.pop('title', 'Reference'),
            url=f'https://{i.lower()}.example.org/article', excerpt=text, **fields))

    def test_task_facts_are_not_excluded_when_question_is_repeated_elsewhere(self):
        # Different topics; no topic-specific selection rule or model call.
        for question, fact in [
            ('How long is the train journey from A to B?', 'Route A–B: duration 47 minutes.'),
            ('这个板材有哪些使用限制？', '仅用于干燥基层；不可长期浸水。'),
            ('When does the museum open on Monday?', 'Monday admission begins at 09:30.'),
            ('How much memory does Model Z need?', 'Z requires 12 GB for inference.'),
        ]:
            with self.subTest(question=question):
                rows = [self.row('W1', f'Language exercise: {question} '*4),
                        self.row('W2', fact)]
                before = copy.deepcopy(rows)
                selected, audit = prepare_support(question, rows)
                self.assertIn('W2', [r['evidence_id'] for r in selected])
                self.assertFalse(audit['score_is_support_verdict'])
                self.assertEqual(rows, before)

    def test_semantics_override_question_overlap_without_forced_fetch_quota(self):
        q = 'What are the installation requirements?'
        rows = [self.row('W1', q * 9, page_fetch_status='ok'),
                self.row('W2', 'Use a dry and level substrate. Do not install in rain.')]
        rows[0]['goal_support_scores'] = {q: .01}
        rows[1]['goal_support_scores'] = {q: .94}
        selected, audit = prepare_support(q, rows)
        self.assertEqual([r['evidence_id'] for r in selected], ['W2'])
        self.assertEqual(audit['decisions'][0]['reason'], 'low_semantic_score')

    def test_metadata_reaches_reranker_and_audit_without_inventing_fact_date(self):
        row = self.row('W1', 'Record effective 2026-10-05', date='2026-09-21',
                       title='Entity Z schedule', evidence_level='verified_page_content')
        scorer = Mock(); scorer.score.return_value = [.9]
        rerank_goal_evidence('Entity Z schedule', [row], goals=[], scorer=scorer)
        document = scorer.score.call_args.args[1][0]
        self.assertIn('2026-09-21', document)
        self.assertIn('2026-10-05', document)
        audit_row = compact_audit_evidence([row])[0]
        self.assertEqual(audit_row['date_scope'], 'page_metadata_not_fact_date')
        self.assertEqual(audit_row['source_date'], '2026-09-21')

    def test_budget_bounds_diversity_and_records_all_candidate_decisions(self):
        rows = [self.row(f'W{i}', 'A long paragraph. '*300) for i in range(20)]
        selected, audit = prepare_support('Explain the requested relationship', rows)
        self.assertLess(len(selected), len(rows))
        self.assertLessEqual(audit['selected_chars'], audit['target_chars'])
        self.assertEqual(len(audit['decisions']), len(rows))
        self.assertTrue(all('reason' in d for d in audit['decisions']))

    def test_admin_records_real_selection_without_raw_body(self):
        journal = ExecutionStatus(); token = current_execution.set(journal)
        try:
            prepare_support('What is Z?', [self.row('W1', 'private raw marker')])
        finally:
            current_execution.reset(token)
        detail = journal.stage_details[-1]
        self.assertEqual(detail['stage'], 'support_selection')
        self.assertEqual(detail['output']['kept_evidence_ids'], ['W1'])
        self.assertNotIn('private raw marker', str(detail))

    def test_public_metadata_changes_score_cache_key(self):
        import time
        from backend.request_budget import RequestBudget, budget_scope
        scorer = Mock(); scorer.score.return_value = [.9]
        row = self.row('W1', 'same body', date='2026-09-21')
        with budget_scope(RequestBudget(time.monotonic()+200)):
            rerank_goal_evidence('Q', [row], goals=[], scorer=scorer)
            rerank_goal_evidence('Q', [{**row, 'source_date':'2026-10-05'}], goals=[], scorer=scorer)
        self.assertEqual(scorer.score.call_count, 2)

    def test_one_ranking_call_retains_whole_question_and_dimensions(self):
        scorer = Mock(); scorer.score.return_value = [.8]
        question = 'Compare installation of A and B, excluding indoor use'
        _, audit = rerank_goal_evidence(question, [self.row('W1', 'Fact')],
            goals=['Compare substrate requirements of A and B', 'Compare anchoring of A and B'], scorer=scorer)
        scorer.score.assert_called_once()
        query = scorer.score.call_args.args[0]
        self.assertTrue(query.startswith(question))
        self.assertIn('anchoring of A and B', query)
        self.assertFalse(audit['dimension_scores_are_independent'])

    def test_bad_replan_shape_is_explicit_error_not_silent_stop(self):
        from unittest.mock import patch
        from backend import app
        from backend.sales.tool_planner import ToolPlan
        request = app.DraftRequest(customer_question='What is the admission price?', use_online_search=True)
        journal = ExecutionStatus(); token = current_execution.set(journal)
        try:
            with patch.object(app, 'evaluate_task_support', return_value={
                    'action':'execute', 'revised_plan':[{'tool':'public_web_search'}]}), \
                 patch.object(app, 'check_budget'):
                result = app._CustomerAnswerWorkflowCallbacks.replan_from_feedback(request,
                    ToolPlan(tools=['public_web_search']),
                    {'problem':'Selected excerpts lack admission price', 'missing_aspects':['Admission price']},
                    {'answerable':False})
        finally:
            current_execution.reset(token)
        self.assertIsNone(result)
        self.assertEqual(request._replan_audit['error_code'], 'REPLAN_INVALID_PLAN')
        self.assertEqual(journal.errors[-1]['code'], 'REPLAN_INVALID_PLAN')


if __name__ == '__main__':
    unittest.main()
