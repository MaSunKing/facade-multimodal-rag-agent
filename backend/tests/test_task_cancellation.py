import asyncio
import threading
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from backend import app
from backend.request_budget import RequestBudget, RequestCancelled, budget_scope, check_budget
from backend import task_cancellation as tasks
from backend.sales.runtime_status import classify
from backend.documents.ownership import attachment_owner_id


class CancellationTests(unittest.TestCase):
    def setUp(self):
        tasks._active.clear()
        tasks._pending.clear()

    def test_cancel_is_owner_scoped(self):
        budget = RequestBudget.for_answer(has_attachments=True)
        tasks.register_task('alice', 'a' * 32, budget)
        tasks.cancel_task('bob', 'a' * 32)
        self.assertFalse(budget.cancelled.is_set())
        tasks.cancel_task('alice', 'a' * 32)
        self.assertTrue(budget.cancelled.is_set())
        with budget_scope(RequestBudget(float('inf'))):
            pass
        with self.assertRaises(RequestCancelled), budget_scope(budget):
            check_budget()
        self.assertEqual(classify(RequestCancelled('request_cancelled'), 'evidence_evaluation')['code'], 'REQUEST_CANCELLED')

    def test_cancel_before_registration_and_between_upload_answer(self):
        tasks.cancel_task('alice', 'a' * 32)
        for _ in range(2):
            budget = RequestBudget(float('inf'))
            tasks.register_task('alice', 'a' * 32, budget)
            self.assertTrue(budget.cancelled.is_set())
            tasks.unregister_task('alice', 'a' * 32, budget)

    def test_pending_cancellations_bounded(self):
        for i in range(300):
            tasks.cancel_task('alice', f'{i:032x}')
        self.assertEqual(len(tasks._pending), 256)

    def test_status_does_not_release_or_reveal_another_owners_worker(self):
        budget = RequestBudget(float('inf'))
        tasks.register_task('alice', 'a'*32, budget)
        self.assertEqual(tasks.task_status('bob','a'*32),{'active':False,'status':'idle'})
        self.assertEqual(tasks.task_status('alice','a'*32)['status'],'running')
        tasks.cancel_task('alice','a'*32)
        self.assertEqual(tasks.task_status('alice','a'*32)['status'],'cancelling')
        tasks.unregister_task('alice','a'*32,budget)
        self.assertEqual(tasks.task_status('alice','a'*32)['status'],'idle')

    def test_task_status_endpoint_owner_and_id_validation(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        application=FastAPI()
        application.get('/tasks/{client_task_id}/status')(app.answer_task_status)
        with TestClient(application) as client:
            self.assertEqual(client.get('/tasks/'+'a'*32+'/status').status_code,400)
            headers={'X-Facade-Client-ID':'web_'+'a'*32}
            self.assertEqual(client.get('/tasks/invalid/status',headers=headers).status_code,422)
            self.assertEqual(client.get('/tasks/'+'a'*32+'/status',headers=headers).json()['active'],False)

    def test_interrupted_question_is_separate_from_latest_and_old_history(self):
        req=app.DraftRequest(customer_question='改为比较价格',interrupted_question='比较附件的厚度和安装方式',
            conversation_context=[app.ConversationTurn(role='assistant',content='unverified old answer')])
        req._task_memory={'related_history':[{'content':'unrelated'}],'task_state':[{'key':'old'}]}
        payload=app.planner_context_input(req)
        self.assertEqual(payload['current_question'],'改为比较价格')
        self.assertEqual(payload['task_update']['interrupted_question'],'比较附件的厚度和安装方式')
        self.assertEqual(payload['previous_exchange'],[])
        self.assertEqual(payload['related_history'],[])
        self.assertEqual(app.task_update_context(app.DraftRequest(customer_question='独立问题')), {})

    def test_cancelled_upload_does_not_start_parser(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.documents.router import router
        application = FastAPI()
        application.include_router(router)
        owner = attachment_owner_id(None, 'web_' + 'c' * 32, required=True)
        tasks.cancel_task(owner, 'c' * 32)
        with TestClient(application) as client, patch('backend.documents.router.add_files') as parser:
            response = client.post('/api/copilot/documents',
                headers={'X-Facade-Client-ID': 'web_' + 'c' * 32},
                data={'client_task_id': 'c' * 32}, files={'files': ('note.txt', b'content', 'text/plain')})
        self.assertEqual(response.status_code, 499)
        parser.assert_not_called()

    def test_http_cancel_requires_owner_and_valid_id(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        application = FastAPI()
        application.post('/cancel')(app.cancel_answer_task)
        with TestClient(application) as client:
            self.assertEqual(client.post('/cancel', json={'client_task_id': 'd' * 32}).status_code, 400)
            self.assertEqual(client.post('/cancel', headers={'X-Facade-Client-ID': 'web_' + 'd' * 32},
                                         json={'client_task_id': '../invalid'}).status_code, 422)


class EndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_keeps_slot_until_worker_exits(self):
        started, release = threading.Event(), threading.Event()
        admission = threading.BoundedSemaphore(1)
        task_id = 'b' * 32

        def slow_answer(*args):
            started.set()
            release.wait(2)
            check_budget()

        class Connected:
            async def is_disconnected(self):
                return False

        with patch.object(app, '_answer_admission', admission), patch.object(app, 'grounded_answer', slow_answer):
            request = asyncio.create_task(app.answer_http(
                app.DraftRequest(customer_question='test', client_task_id=task_id), Connected(), None, 'alice'))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 1))
                await app.cancel_answer_task(app.CancelTaskRequest(client_task_id=task_id), 'alice')
                with self.assertRaises(HTTPException) as failure:
                    await request
                self.assertEqual(failure.exception.status_code, 499)
                self.assertFalse(admission.acquire(blocking=False))
            finally:
                release.set()
            for _ in range(100):
                await asyncio.sleep(.01)
                if admission.acquire(blocking=False):
                    admission.release()
                    break
            else:
                self.fail('cancelled worker did not release slot')
            self.assertNotIn(('alice', task_id), tasks._active)
