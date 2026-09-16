import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from backend.sales.runtime_status import (
    ExecutionStatus, current_execution, report_error, model_tool_errors,
    classify, install_error_protocol,
)
from backend.sales.web_context import prepare_web_context, clean_excerpt


class RuntimeStatusTests(unittest.TestCase):
    def test_model_load_failure_keeps_sources_and_correct_primary_error(self):
        import backend.app as app
        sources = [{'source_id':'s1','title':'公开资料','url':'https://example.org','excerpt':'资料内容'}]
        with patch.object(app, 'load_model', side_effect=RuntimeError('private file missing')):
            result = app.general_local_chat_answer(app.DraftRequest(customer_question='介绍一下'), sources)
        self.assertFalse(result['answerable'])
        self.assertFalse(result['meta']['model_used'])
        self.assertEqual(result['meta']['error']['code'], 'MODEL_LOAD_FAILED')
        self.assertEqual(result['online_sources'], sources)
        self.assertNotIn('private', result['customer_reply'])

    def test_terminal_states_and_model_contract(self):
        journal = ExecutionStatus()
        token = current_execution.set(journal)
        try:
            journal.enter('public_web_search')
            report_error(code='WEB_QUOTA_EXHAUSTED')
            self.assertEqual(model_tool_errors()[0]['model_action'], 'do_not_retry_web')
            self.assertNotIn('exception_type', model_tool_errors()[0])
            journal.enter('generate_answer')
            journal.finish(True)
            self.assertEqual(journal.state, 'degraded')
            journal.enter('plan_request')
            journal.finish(False)
            self.assertEqual(journal.state, 'degraded')
        finally:
            current_execution.reset(token)

    def test_failure_classes_are_safe(self):
        for exception, stage, expected in [
            (RuntimeError('CUDA out of memory secret=private'), 'generate_answer', 'MODEL_OOM'),
            (TimeoutError('private endpoint'), 'public_web_search', 'WEB_TIMEOUT'),
            (RuntimeError('C:/private/model'), 'model_load', 'MODEL_LOAD_FAILED'),
            (ValueError('context_budget exceeded'), 'generate_answer', 'CONTEXT_BUDGET_EXCEEDED'),
            (json.JSONDecodeError('secret', 'private', 0), 'validate_answer', 'OUTPUT_INVALID_JSON'),
        ]:
            info = classify(exception, stage)
            self.assertEqual(info['code'], expected)
            self.assertNotIn('private', json.dumps(info))
            self.assertNotIn('secret', json.dumps(info))

    def test_web_provider_error_details(self):
        from backend.sales.baidu_search import BaiduSearchError
        for status, expected in [(401,'WEB_AUTH_FAILED'),(403,'WEB_AUTH_FAILED'),
                                 (429,'WEB_RATE_LIMITED'),(503,'WEB_PROVIDER_ERROR')]:
            self.assertEqual(classify(BaiduSearchError('secret', http_status=status),
                                      'public_web_search')['code'], expected)
        self.assertEqual(classify(BaiduSearchError('private', error_code='timeout'),
                                  'public_web_search')['code'], 'WEB_TIMEOUT')

    def test_http_errors_ids_and_safe_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'errors.jsonl'
            app = FastAPI()
            install_error_protocol(app, path)

            @app.get('/failure/{status}')
            def fail(status: int):
                if status == 500:
                    raise RuntimeError('private-key C:/private/file')
                raise HTTPException(status, 'private-key C:/private/file')

            @app.get('/ok')
            def ok():
                return {'ok': True}

            with TestClient(app) as client:
                ids = set()
                for status, code in [(401, 'AUTH_REQUIRED'), (403, 'ACCESS_DENIED'),
                                     (404, 'NOT_FOUND'), (413, 'FILE_TOO_LARGE'),
                                     (429, 'SERVICE_BUSY'), (504, 'REQUEST_TIMEOUT'),
                                     (500, 'INTERNAL_ERROR')]:
                    response = client.get('/failure/' + str(status))
                    self.assertEqual(response.status_code, status)
                    self.assertEqual(response.json()['error']['code'], code)
                    self.assertEqual(response.headers['X-Request-ID'], response.json()['request_id'])
                    ids.add(response.json()['request_id'])
                    self.assertNotIn('private', response.text)
                self.assertEqual(len(ids), 7)
                self.assertEqual(client.get('/failure/not-a-number').status_code, 422)
                self.assertTrue(client.get('/ok').json()['ok'])
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(row['state'] == 'failed' for row in rows))
            self.assertNotIn('private', path.read_text(encoding='utf-8'))

    def test_web_versions_not_false_independent_sources(self):
        sources = [
            {'source_id':'a','url':'https://a.weather.com.cn/x','excerpt':'2026-09-10 今天晴', 'search_excerpt':'duplicate'},
            {'source_id':'b','url':'https://b.weather.com.cn/x','excerpt':'2026-09-10 今天晴'},
            {'source_id':'c','url':'https://c.weather.com.cn/x','excerpt':'2026-09-12 今天雨'},
        ]
        output, audit = prepare_web_context(sources, '2026-09-12')
        self.assertEqual([s['source_id'] for s in output], ['a','c'])
        self.assertEqual(audit['exact_duplicates_removed'], ['b'])
        self.assertEqual(audit['same_publisher_groups']['weather.com.cn'], ['a','c'])
        self.assertEqual(output[0]['mentioned_dates'], ['2026-09-10'])
        self.assertNotIn('search_excerpt', output[0])
        self.assertIn('|', clean_excerpt('<tr><td>日期</td><td>温度</td></tr>'))


if __name__ == '__main__':
    unittest.main()
