import asyncio
import threading
import time
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from backend import app
from backend.request_budget import RequestBudget, RequestBudgetExceeded, budget_scope, check_budget


class BudgetTests(unittest.TestCase):
    def test_legacy_forced_stage_deadline_is_disabled(self):
        budget = RequestBudget.for_answer(has_attachments=True)
        with patch('backend.app.time.monotonic', return_value=100), budget_scope(budget):
            criteria = app.local_generation_stopping_criteria(30, force_node_deadline=True)
            with patch('backend.app.time.monotonic', return_value=131):
                self.assertFalse(criteria[0](None, None))
            self.assertFalse(budget.has_deadline)
    def test_attachment_disables_global_and_node_deadlines_but_not_cancel(self):
        budget = RequestBudget.for_answer(has_attachments=True)
        self.assertFalse(budget.has_deadline)
        with budget_scope(budget):
            with patch('backend.app.time.monotonic', return_value=10**12):
                criteria = app.local_generation_stopping_criteria(1, reserve_seconds=60)
                self.assertFalse(criteria[0](None, None))
                check_budget()
                budget.cancelled.set()
                self.assertTrue(criteria[0](None, None))

    def test_plain_question_has_no_deadline_even_after_cold_start(self):
        with patch('backend.request_budget.time.monotonic', return_value=100):
            budget = RequestBudget.for_answer(has_attachments=False)
        self.assertFalse(budget.has_deadline)
        with patch('backend.app.time.monotonic', return_value=100), budget_scope(budget):
            criteria = app.local_generation_stopping_criteria(None)
            with patch('backend.app.time.monotonic', return_value=281):
                self.assertFalse(criteria[0](None, None))

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
