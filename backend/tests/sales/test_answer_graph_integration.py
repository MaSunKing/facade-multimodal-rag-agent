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

        general.assert_called_once_with(request, allow_public_web=False, web_source_profile="auto")
        self.assertEqual(response.customer_reply, original["customer_reply"])
        self.assertEqual(response.meta["mode"], "fake_general")
        self.assertEqual(response.meta["orchestration"]["engine"], "langgraph")
        self.assertEqual(response.meta["orchestration"]["persistence"], "disabled")

    def test_dynamic_inventory_value_refuses_before_model_and_retrieval(self) -> None:
        request = facade_app.DraftRequest(
            customer_question="今天仓库里某型号还有多少平方米现货？",
            use_online_search=True,
        )
        with patch.object(facade_app, "load_model") as load_model, patch.object(
            facade_app, "load_retriever"
        ) as load_retriever, patch.object(facade_app, "maybe_search_online") as online_search:
            response = facade_app.grounded_answer(request)

        load_model.assert_not_called()
        load_retriever.assert_not_called()
        online_search.assert_not_called()
        self.assertFalse(response.answerable)
        self.assertEqual(response.citations, [])
        self.assertEqual(response.retrieval["result_count"], 0)
        self.assertEqual(response.retrieval["strategy"], "dynamic_private_business_data_guard")
        self.assertIn("dynamic_private_business_data_unavailable", response.risk_warnings)
        self.assertEqual(response.meta["mode"], "dynamic_private_business_data_refusal")
        self.assertEqual(response.meta["orchestration"]["tools"], [])

    def test_static_warehouse_rule_keeps_normal_company_rag_route(self) -> None:
        request = facade_app.DraftRequest(customer_question="板材在仓库中应如何存放并保持通风？")
        plan = ToolPlan(
            tools=["company_rag"],
            reason="model_static_rule",
            intent="construction",
            task_type="procedure",
            retrieval_query=request.customer_question,
        )
        original = {
            "intent": "construction",
            "normalized_terms": [],
            "answerable": True,
            "customer_reply": "应按企业施工资料中的仓储要求执行。",
            "key_points": [],
            "citations": [],
            "missing_information": [],
            "risk_warnings": [],
            "next_action": "继续提问即可。",
            "image_observations": [],
            "visual_assets": [],
            "retrieval": {"result_count": 0, "supporting_results": [], "visual_count": 0},
            "meta": {"mode": "fake_company_rag"},
            "online_sources": [],
        }
        with patch.object(
            facade_app,
            "plan_customer_tools",
            return_value=plan,
        ), patch.object(
            facade_app,
            "_run_facade_rag_answer",
            return_value=facade_app.AnswerResponse(**original),
        ) as company_rag:
            response = facade_app.grounded_answer(request)

        company_rag.assert_called_once_with(
            request, allow_public_web=False, web_source_profile="auto", tool_plan=plan
        )
        self.assertTrue(response.answerable)
        self.assertEqual(response.meta["mode"], "fake_company_rag")
