import asyncio
import threading
import time
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from backend import app
from backend.request_budget import RequestBudget, RequestBudgetExceeded, budget_scope, check_budget


class BudgetTests(unittest.TestCase):
    def test_generation_stops_after_request_cancel(self):
        budget = RequestBudget(time.monotonic() + 30)
        with budget_scope(budget):
            criteria = app.local_generation_stopping_criteria(60)
            budget.cancelled.set()
            self.assertTrue(criteria[0](None, None))
            with self.assertRaises(RequestBudgetExceeded):
                check_budget()


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_queue_rejects_before_work(self):
        semaphore = threading.BoundedSemaphore(1)
        semaphore.acquire()
        with patch.object(app, "_answer_admission", semaphore):
            with self.assertRaises(HTTPException) as failure:
                await app.answer_http(app.DraftRequest(customer_question="test"), None, None, None)
        self.assertEqual(failure.exception.status_code, 429)

    async def test_disconnect_keeps_slot_until_worker_stops(self):
        started, release = threading.Event(), threading.Event()
        semaphore = threading.BoundedSemaphore(1)

        def slow_answer(*args):
            started.set()
            release.wait(2)
            check_budget()

        class Disconnected:
            async def is_disconnected(self):
                await asyncio.to_thread(started.wait, 1)
                return True

        with patch.object(app, "_answer_admission", semaphore), patch.object(app, "grounded_answer", slow_answer):
            with self.assertRaises(HTTPException):
                await app.answer_http(app.DraftRequest(customer_question="test"), Disconnected(), None, None)
            self.assertFalse(semaphore.acquire(blocking=False))
            release.set()
            for _ in range(50):
                await asyncio.sleep(0.01)
                if semaphore.acquire(blocking=False):
                    semaphore.release()
                    break
            else:
                self.fail("worker did not release admission")
