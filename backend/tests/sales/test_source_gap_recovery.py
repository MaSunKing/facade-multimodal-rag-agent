import time
import unittest
from unittest.mock import Mock

from backend.request_budget import RequestBudget, budget_scope, reserve_recovery
from backend.sales.recovery_policy import (
    assess_source_gaps, repair_source_bundle, query_language_mismatch,
    source_diagnostics, answer_document_coverage, normalise_anchor,
)
from backend.sales.staged_execution import WorkflowStep, WorkflowCursor
from backend.sales.tool_planner import ToolPlan
from backend.sales.answer_graph import build_customer_answer_graph


class SourceGapRecoveryTests(unittest.TestCase):
    def test_proper_name_is_not_english_query_translation(self):
        self.assertTrue(query_language_mismatch('NASA earth资料 主题 地球自转时间 对照', 'Earth rotation facts '*20))
        self.assertTrue(query_language_mismatch('ESA 行星大气压力比较', 'Planet atmosphere pressure '*20))
        self.assertFalse(query_language_mismatch('Earth rotation period compare documents', 'Earth rotation facts '*20))
        self.assertFalse(query_language_mismatch('企业介绍', '公司产品参数 '*20))

    def test_lexical_target_absence_alone_never_triggers_retry(self):
        plan = ToolPlan(tools=['customer_documents'])
        decision = assess_source_gaps(plan, {'customer_documents': {'state': 'literal_target_missing'}}, web_allowed=False)
        self.assertFalse(decision.actions)
        self.assertTrue(decision.blocked)

    def test_missing_planned_dimensions_are_bundled_not_hard_routed(self):
        plan = ToolPlan(tools=['customer_documents', 'company_rag', 'public_web_search'],
                        requires_public_web=True, document_visual_required=True)
        decision = assess_source_gaps(plan, {
            'customer_documents': {'state': 'index_only'}, 'company_rag': {'state': 'empty'},
            'customer_visual': {'state': 'missing_visual_input'},
            'public_web_search': {'state': 'current_fact_missing'},
        }, web_allowed=True)
        self.assertEqual({tool for tool, _ in decision.actions},
                         {'customer_documents', 'company_rag', 'customer_visual', 'public_web_search'})

    def test_no_unplanned_company_or_web_supplement(self):
        plan = ToolPlan(tools=['customer_documents'])
        decision = assess_source_gaps(plan, {'company_rag': {'state': 'empty'},
            'public_web_search': {'state': 'current_fact_missing'}}, web_allowed=True)
        self.assertFalse(decision.actions)

    def test_web_needs_both_need_and_consent(self):
        for consent, need in [(False, True), (True, False)]:
            plan = ToolPlan(tools=['public_web_search'], requires_public_web=need)
            self.assertFalse(assess_source_gaps(plan, {'public_web_search': {'state': 'current_fact_missing'}}, web_allowed=consent).actions)

    def test_terminal_errors_and_real_no_support_do_not_retry(self):
        for reason in ('quota_exhausted', 'permission_denied', 'invalid_input',
                       'confirmed_no_support', 'authentication_failed', 'unavailable'):
            plan = ToolPlan(tools=['company_rag'])
            self.assertFalse(assess_source_gaps(plan, {'company_rag': {'state': reason}}, web_allowed=True).actions)

    def run_bundle(self, statuses, results, operations, budget):
        plan = ToolPlan(tools=['customer_documents', 'company_rag'])
        with budget_scope(budget):
            cursor = WorkflowCursor(repair_source_bundle(plan, statuses, results, operations, web_allowed=False))
            cursor.advance()
            while not cursor.done:
                cursor.execute()
            return cursor.result

    def test_ready_tools_not_replayed_and_only_one_shared_slot(self):
        documents, company = Mock(return_value={'evidence': ['new']}), Mock()
        budget = RequestBudget(time.monotonic()+90)
        results, audit = self.run_bundle({'customer_documents': {'state': 'empty'}, 'company_rag': {'state': 'ready'}},
            {'customer_documents': {}, 'company_rag': {'text_evidence': ['retained']}},
            {'customer_documents': WorkflowStep('customer_documents', documents), 'company_rag': WorkflowStep('company_rag', company)}, budget)
        documents.assert_called_once(); company.assert_not_called()
        self.assertEqual(results['company_rag']['text_evidence'], ['retained'])
        self.assertTrue(audit['attempted']); self.assertEqual(len(budget.recoveries), 1)
        with budget_scope(budget):
            self.assertFalse(reserve_recovery('generate_answer', 'repack_after_oom'))

    def test_failed_supplement_preserves_initial_evidence(self):
        operation = Mock(side_effect=TimeoutError())
        budget = RequestBudget(time.monotonic()+90)
        results, audit = self.run_bundle({'customer_documents': {'state': 'index_only'}},
            {'customer_documents': {'evidence': ['index']}}, {'customer_documents': WorkflowStep('customer_documents', operation)}, budget)
        operation.assert_called_once()
        self.assertEqual(results['customer_documents']['evidence'], ['index'])
        self.assertTrue(audit['failed_tools'])

    def test_prior_query_normalization_does_not_steal_source_retry(self):
        budget = RequestBudget(time.monotonic()+90)
        with budget_scope(budget):
            self.assertTrue(reserve_recovery('plan_request', 'repair_query_language'))
        operation = Mock(return_value={'text_evidence': ['retrieved']})
        _, audit = self.run_bundle({'company_rag': {'state': 'empty'}}, {'company_rag': {}},
                                  {'company_rag': WorkflowStep('company_rag', operation)}, budget)
        operation.assert_called_once(); self.assertTrue(audit['attempted'])

    def test_time_shortage_does_not_launch_repair(self):
        operation = Mock()
        _, audit = self.run_bundle({'company_rag': {'state': 'empty'}}, {'company_rag': {}},
            {'company_rag': WorkflowStep('company_rag', operation)}, RequestBudget(time.monotonic()+10))
        operation.assert_not_called(); self.assertEqual(audit['reason'], 'remaining_time_insufficient')

    def test_answer_source_acknowledgment_is_not_semantic_accuracy(self):
        visible = {'U1': {'document_name': 'a.pdf', 'evidence_scope': 'content'},
                   'U2': {'document_name': 'b.xlsx', 'evidence_scope': 'content'}}
        result = {'answerable': True, 'customer_reply': 'a.pdf shows 20 mm', 'citations': []}
        audit = answer_document_coverage(ToolPlan(document_scope='cross_document'), result, visible)
        self.assertEqual(audit['verified_missing_documents'], ['b.xlsx'])
        self.assertFalse(audit['semantic_assertion_completeness_verified'])
        self.assertFalse(answer_document_coverage(ToolPlan(document_scope='local_lookup'), result, visible)['verified_missing_documents'])

    def test_diagnostics_do_not_treat_expired_session_as_empty_retrieval(self):
        states = source_diagnostics({'customer_documents': {'status': 'session_not_found'},
                                    'public_web_search': ([], {'status': 'quota_exhausted'})})
        self.assertEqual(states['customer_documents']['state'], 'unavailable')
        self.assertEqual(states['public_web_search']['state'], 'quota_exhausted')
        self.assertEqual(normalise_anchor('２０ mm'), '20mm')

    def test_graph_reuses_unchanged_cpu_results_during_answer_repair(self):
        calls = []
        class Callbacks:
            def plan_tools(self, request): return ToolPlan(tools=['customer_documents'])
            def open_steps(self, request, plan):
                yield WorkflowStep('customer_documents', lambda: calls.append('retrieve') or {'evidence': ['value']})
                yield WorkflowStep('generate_answer')
                return {'answerable': True, 'meta': {'model_used': True}}
            def plan_retry(self, request, plan, response):
                return plan.model_copy(update={'recovery_directive': {'stage': 'validate_answer', 'action': 'repair_answer_coverage'}})
        with budget_scope(RequestBudget(time.monotonic()+90)):
            result = build_customer_answer_graph(Callbacks()).invoke({'request': {}})['response']
        self.assertEqual(calls, ['retrieve'])
        trace = result['meta']['orchestration']['node_trace']
        self.assertTrue(trace[2]['reused'])
        self.assertEqual(result['meta']['orchestration']['recovery_actions'][0]['stage'], 'validate_answer')

    def test_source_language_metric_goal_survives_prose_preference(self):
        from backend.sales.context_engine import optimise_evidence_context
        from backend.app import compact_grounded_payload_for_generation
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs): return messages[-1]['content']
            def __call__(self, text, **kwargs): return {'input_ids': text.split()}
        candidates = [
            {'evidence_id': 'A0', 'document_name': 'manual.pdf', 'evidence_scope': 'content',
             'text': 'Publisher overview ' * 120},
            {'evidence_id': 'A1', 'document_name': 'manual.pdf', 'evidence_scope': 'content',
             'text': '[ROW source=pdf;page=2] depth = 12 mm'},
            {'evidence_id': 'B1', 'document_name': 'guide.html', 'evidence_scope': 'content',
             'text': 'depth = 10 mm'},
        ]
        ranked, _ = optimise_evidence_context('比较这两份资料的depth', candidates, target_terms=['depth'])
        _, audit = compact_grounded_payload_for_generation(
            {'customer_question': 'Compare depth', 'attachment_context': {'global_document_question': True}, 'evidence': ranked},
            Tokenizer(), max_prompt_tokens=130)
        self.assertIn('A1', audit['kept_evidence_ids'])
        self.assertIn('B1', audit['kept_evidence_ids'])
        self.assertEqual(audit['missing_packed_goal_ids'], [])

    def test_search_targets_survive_plan_guard(self):
        from backend.sales.tool_planner import guard_plan
        plan = ToolPlan(tools=['customer_documents'], search_targets=['thermal conductivity'])
        safe = guard_plan(plan, has_documents=True, has_image=False, facade_related=False, web_allowed=False)
        self.assertEqual(safe.search_targets, ['thermal conductivity'])

    def test_goal_candidate_expansion_is_bounded_and_owner_isolated(self):
        from backend.documents.customer_sessions import add_files, retrieve, delete_session, bind_session_owner
        owner = 'test-goal-retrieval'
        with bind_session_owner(owner):
            session = add_files([('guide.txt', ('Overview notes '*450+'\nDepth: 12 mm\nPressure: 7 Pa').encode())])
            try:
                result = retrieve(session['session_id'], 'Summarise this guide', document_scope='whole_document',
                    max_text_tokens=2000, search_targets=['depth in document 1', 'pressure in document 1'])
                snapshot = result['input_snapshot']
                self.assertEqual(snapshot['goal_candidate_expansion']['queries'], ['depth', 'pressure'])
                self.assertLessEqual(snapshot['selected_estimated_text_tokens'], snapshot['max_text_tokens'])
                self.assertTrue(any('Depth: 12' in item['text'] for item in result['evidence']))
                self.assertEqual(len({item['evidence_id'] for item in result['evidence']}), len(result['evidence']))
                with bind_session_owner('another-owner'):
                    denied = retrieve(session['session_id'], 'depth', search_targets=['depth'])
                self.assertEqual(denied['status'], 'session_not_found')
            finally:
                delete_session(session['session_id'])

    def test_failed_source_is_retained_as_typed_status_and_cancel_propagates(self):
        from backend.sales.recovery_policy import read_source_step
        from backend.request_budget import RequestBudgetExceeded
        for error in (PermissionError(), RequestBudgetExceeded()):
            cursor = WorkflowCursor(read_source_step(WorkflowStep('company_rag', Mock(side_effect=error)),
                                                      {'text_evidence': [], 'meta': {}}))
            with budget_scope(RequestBudget(time.monotonic()+90)):
                cursor.advance()
                if isinstance(error, RequestBudgetExceeded):
                    with self.assertRaises(RequestBudgetExceeded): cursor.execute()
                else:
                    cursor.execute()
                    self.assertEqual(source_diagnostics({'company_rag': cursor.result})['company_rag']['state'], 'permission_denied')


if __name__ == '__main__': unittest.main()
