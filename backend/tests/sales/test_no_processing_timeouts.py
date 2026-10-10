import json
import os
import unittest
from unittest.mock import patch
from backend.request_budget import RequestBudget, budget_scope, check_budget, RequestCancelled
from backend.documents.summaries import shared_budget
from backend.sales.recovery_budget import budget_view, validate_budget_view
from backend.sales.model_contract import generation_guard


class NoProcessingTimeoutTests(unittest.TestCase):
    def test_summary_time_settings_no_longer_limit_task(self):
        with patch.dict(os.environ,{'CUSTOMER_SUMMARY_SECONDS':'1','CUSTOMER_SUMMARY_CALL_SECONDS':'1'}):
            request=RequestBudget.for_answer(has_attachments=False)
            with budget_scope(request):
                summary=shared_budget()
                summary.started-=10000
                summary.reserve(10,10)
                self.assertGreater(summary.remaining_seconds,10000)
                view=validate_budget_view(budget_view(request,tools=['company_rag']))
                self.assertIsNone(view['summary']['remaining_seconds'])
                self.assertEqual(view['final_reserve_seconds'],0)
                json.dumps(view,allow_nan=False)
                request.cancelled.set()
                with self.assertRaises(RequestCancelled):check_budget()

    def test_no_model_stage_timeout_but_cancel_checked(self):
        request=RequestBudget.for_answer(has_attachments=True)
        with budget_scope(request):
            guard,check=generation_guard(0,None)
            check()
            self.assertFalse(guard(None,None))
            request.cancelled.set()
            with self.assertRaises(RequestCancelled):check()

    def test_resource_limits_still_apply(self):
        from backend.documents.summaries import SummaryLimit
        request=RequestBudget.for_answer(has_attachments=True)
        with budget_scope(request):
            summary=shared_budget()
            summary.calls=summary.max_calls
            with self.assertRaises(SummaryLimit):summary.reserve(1,1)
