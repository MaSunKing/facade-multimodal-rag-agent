import time
import unittest
from unittest.mock import Mock, patch

from backend.request_budget import RequestBudget, budget_scope
from backend.sales.goal_reranking import rerank_goal_evidence
from backend.sales.response_language import evidence_gate_notice
from backend.sales import baidu_search as search


class RecoveryRefreshTests(unittest.TestCase):
    def test_absent_reranker_is_loaded_once_on_current_safe_device(self):
        scorer = Mock(device='cpu'); scorer.score.return_value = [.8]
        with patch('backend.sales.dense_retrieval.loaded_reranker_model', return_value=None), \
             patch('backend.sales.dense_retrieval.get_reranker_model', return_value=scorer) as load:
            with budget_scope(RequestBudget(time.monotonic()+180)):
                _, audit = rerank_goal_evidence('original question',
                    [{'evidence_id':'W2','text':'relevant fact'}], goals=['complete dimension'])
        load.assert_called_once(); scorer.score.assert_called_once()
        self.assertTrue(audit['applied'])
        self.assertEqual(audit['model_device'], 'cpu')

    def test_refresh_bypasses_cache_with_one_authorised_search_call(self):
        with patch.object(search, '_read_cache') as read, \
             patch.object(search, '_reserve_api_call', return_value={}), \
             patch.object(search, '_call_baidu_with_retry', return_value=({},1)) as call, \
             patch.object(search, '_references_from_response', return_value=[]), \
             patch.object(search, '_write_cache'):
            result = search.search_baidu_web('full question', bypass_cache=True)
        read.assert_not_called(); call.assert_called_once()
        self.assertTrue(result['cache_bypassed'])
        self.assertFalse(result['cache_hit'])
        self.assertEqual(result['api_calls_for_query'], 1)

    def test_request_refresh_is_consumed_once_and_cannot_bypass_permission(self):
        from backend import app
        request = app.DraftRequest(customer_question='What is Z?', use_online_search=True)
        request._refresh_public_web_once = True
        with patch.object(app, 'is_baidu_search_configured', return_value=True), \
             patch('backend.sales.search_mcp.search_web_via_mcp', return_value={'sources':[{'source_id':'W1'}]}) as call:
            app.maybe_search_online(request, request.customer_question)
            app.maybe_search_online(request, request.customer_question)
            self.assertTrue(call.call_args_list[0].kwargs['bypass_cache'])
            self.assertFalse(call.call_args_list[1].kwargs['bypass_cache'])
            request.use_online_search = False
            request._refresh_public_web_once = True
            app.maybe_search_online(request, request.customer_question)
            self.assertEqual(call.call_count, 2)

    def test_english_refusal_is_not_a_request_to_clarify_an_already_clear_question(self):
        reply, action, _ = evidence_gate_notice('English')
        self.assertIn('selected evidence', reply)
        self.assertNotIn('clarify', action)
        self.assertNotIn('当前', reply)
        self.assertIn('could not be completed', evidence_gate_notice('English', audit_failed=True)[0])

    def test_partial_gate_does_not_pass_rejected_sources_into_generation(self):
        from backend.sales.task_support import supported_web_sources
        sources = [{'source_id':'W1','excerpt':'old data'}, {'source_id':'W2','excerpt':'limited supported fact'}]
        selected = supported_web_sources(sources, {'status':'partial','supported_evidence_ids':['W2']})
        self.assertEqual(selected, [sources[1]])
        self.assertEqual(len(sources), 2)
        self.assertEqual(supported_web_sources(sources, {'status':'supported','supported_evidence_ids':[]}), [])


if __name__ == '__main__':
    unittest.main()
