"""Contract check: FastAPI answer entrypoint uses the stateless graph."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import backend.app as facade_app
from backend.sales.tool_planner import ToolPlan


class AnswerGraphIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        facade_app._answer_graph = None

    def tearDown(self) -> None:
        facade_app._answer_graph = None

    def test_entrypoint_keeps_general_response_contract(self) -> None:
        request = facade_app.DraftRequest(customer_question="你好")
        original = {
            "intent": "general_chat",
            "normalized_terms": [],
            "answerable": True,
            "customer_reply": "你好，有什么可以帮你？",
            "key_points": [],
            "citations": [],
            "missing_information": [],
            "risk_warnings": [],
            "next_action": "继续提问即可。",
            "visual_assets": [],
            "retrieval": {"result_count": 0, "supporting_results": [], "visual_count": 0},
            "meta": {"mode": "fake_general"},
            "online_sources": [],
        }
        with patch.object(
            facade_app, "plan_customer_tools", return_value=ToolPlan(tools=["general_chat"], reason="test")
        ), patch.object(
            facade_app, "_run_general_local_answer", return_value=facade_app.AnswerResponse(**original)
        ) as general:
            response = facade_app.grounded_answer(request)

        general.assert_called_once_with(request, allow_public_web=False)
        self.assertEqual(response.customer_reply, original["customer_reply"])
        self.assertEqual(response.meta["mode"], "fake_general")
        self.assertEqual(response.meta["orchestration"]["engine"], "langgraph")
        self.assertEqual(response.meta["orchestration"]["persistence"], "disabled")
