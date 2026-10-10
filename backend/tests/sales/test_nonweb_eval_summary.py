import unittest

from scripts.summarize_nonweb_eval import summarize


class NonwebSummaryTests(unittest.TestCase):
    def test_completion_does_not_assign_semantic_accuracy(self):
        result = summarize([{'id': 'a', 'group': 'company', 'status': 'completed'}])
        self.assertIsNone(result['semantic_accuracy'])
        self.assertFalse(result['cases'][0]['token_usage_available'])

    def test_journal_total_is_not_double_counted_with_stages(self):
        execution = {'token_usage': {'input_tokens': 100, 'output_tokens': 10},
                     'stage_details': [{'token_usage': {'input_tokens': 100, 'output_tokens': 10}}],
                     'model_contracts': [{'attempts': [{'error_code': 'OUTPUT_TOKEN_LIMIT'}, {}]}]}
        row = summarize([{'id': 'a', 'status': 'completed', 'responses': [
            {'meta': {'execution': execution, 'token_usage': execution['token_usage']}}]}])['cases'][0]
        self.assertEqual(row['input_tokens'], 100)
        self.assertEqual(row['output_tokens'], 10)
        self.assertEqual(row['format_invalid_attempts'], 1)
        self.assertEqual(row['model_attempts'], 2)

    def test_transport_failure_uses_trace_without_response(self):
        row = summarize([{'id': 'a', 'status': 'technical_failure', 'traces': [
            {'summary': {'execution': {'token_usage': {'input_tokens': 7, 'output_tokens': 2}}}}]}])['cases'][0]
        self.assertEqual(row['input_tokens'], 7)
        self.assertTrue(row['token_usage_available'])

    def test_each_conversation_request_counts_once(self):
        row = summarize([{'id': 'a', 'status': 'completed', 'responses': [
            {'meta': {'execution': {'token_usage': {'input_tokens': 7, 'output_tokens': 2}},
                      'answer_revision_count': 1, 'online_search': {'status': 'not_requested'}}},
            {'meta': {'execution': {'token_usage': {'input_tokens': 8, 'output_tokens': 3}}}}]}])['cases'][0]
        self.assertEqual(row['input_tokens'], 15)
        self.assertEqual(row['answer_revisions'], 1)
        self.assertFalse(row['web_executed'])


if __name__ == '__main__':
    unittest.main()
