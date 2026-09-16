"""Unit tests for bounded, stateless tool orchestration."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from backend.sales.answer_graph import build_customer_answer_graph


@dataclass
class FakeRequest:
    tools: list[str]


class FakeCallbacks:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def plan_tools(self, request: FakeRequest) -> dict:
        self.calls.append("plan")
        return {"tools": request.tools}

    def answer_planned(self, request: FakeRequest, plan: dict) -> dict:
        self.calls.append("execute")
        return {"customer_reply": "planned", "meta": {"mode": "planned", "plan": plan}}


class RetryCallbacks(FakeCallbacks):
    def __init__(self) -> None:
        super().__init__()
        self.retry_checks = 0

    def plan_retry(self, request: FakeRequest, plan: dict, response: dict) -> dict | None:
        self.retry_checks += 1
        if self.retry_checks == 1:
            return {
                "tools": ["customer_documents", "company_rag"],
                "retrieval_query": "expanded evidence query",
                "reason": "insufficient_first_pass_evidence",
            }
        raise AssertionError("bounded graph must not request a second retry")


class EquivalentRetryCallbacks(FakeCallbacks):
    def plan_retry(self, request: FakeRequest, plan: dict, response: dict) -> dict | None:
        return dict(plan)


class FailingRetryCallbacks(FakeCallbacks):
    def plan_retry(self, request: FakeRequest, plan: dict, response: dict) -> dict | None:
        raise RuntimeError("retry audit unavailable")


class CustomerAnswerGraphTests(unittest.TestCase):
    def test_plan_is_executed_once_without_persistence(self) -> None:
        callbacks = FakeCallbacks()
        graph = build_customer_answer_graph(callbacks)
        state = graph.invoke({"request": FakeRequest(tools=["general_chat"])})
        self.assertEqual(callbacks.calls, ["plan", "execute"])
        self.assertEqual(state["response"]["meta"]["orchestration"]["persistence"], "disabled")
        self.assertEqual(state["response"]["meta"]["orchestration"]["planning_rounds"], 1)
        self.assertIn("planner_latency_ms", state["response"]["meta"]["orchestration"])
        self.assertFalse(state["response"]["meta"]["orchestration"]["planner_cache_hit"])
        self.assertEqual(state["response"]["meta"]["orchestration"]["tool_rounds"], 1)
        self.assertEqual(
            state["response"]["meta"]["orchestration"]["gpu_execution_policy"],
            "sequential",
        )

    def test_multiple_tools_are_exposed_in_orchestration_meta(self) -> None:
        callbacks = FakeCallbacks()
        graph = build_customer_answer_graph(callbacks)
        state = graph.invoke({"request": FakeRequest(tools=["customer_documents", "company_rag"])})
        self.assertEqual(callbacks.calls, ["plan", "execute"])
        self.assertEqual(state["response"]["customer_reply"], "planned")
        self.assertEqual(
            state["response"]["meta"]["orchestration"]["tools"],
            ["customer_documents", "company_rag"],
        )

    def test_one_evidence_driven_retry_is_allowed_and_then_stops(self) -> None:
        callbacks = RetryCallbacks()
        graph = build_customer_answer_graph(callbacks)
        state = graph.invoke({"request": FakeRequest(tools=["customer_documents"])})

        self.assertEqual(callbacks.calls, ["plan", "execute", "execute"])
        orchestration = state["response"]["meta"]["orchestration"]
        self.assertEqual(orchestration["planning_rounds"], 2)
        self.assertEqual(orchestration["tool_rounds"], 2)
        self.assertTrue(orchestration["bounded_retry_attempted"])
        self.assertEqual(orchestration["max_tool_retries"], 1)
        self.assertEqual(
            orchestration["tool_plan_history"],
            [["customer_documents"], ["customer_documents", "company_rag"]],
        )

    def test_equivalent_retry_plan_is_not_reexecuted(self) -> None:
        callbacks = EquivalentRetryCallbacks()
        graph = build_customer_answer_graph(callbacks)
        state = graph.invoke({"request": FakeRequest(tools=["company_rag"])})

        self.assertEqual(callbacks.calls, ["plan", "execute"])
        self.assertEqual(state["response"]["meta"]["orchestration"]["tool_rounds"], 1)
        self.assertFalse(state["response"]["meta"]["orchestration"]["bounded_retry_attempted"])

    def test_retry_policy_failure_keeps_first_successful_response(self) -> None:
        callbacks = FailingRetryCallbacks()
        graph = build_customer_answer_graph(callbacks)
        state = graph.invoke({"request": FakeRequest(tools=["company_rag"])})

        self.assertEqual(callbacks.calls, ["plan", "execute"])
        self.assertEqual(state["response"]["customer_reply"], "planned")
        self.assertEqual(
            state["response"]["meta"]["orchestration"]["retry_reason"],
            "retry_policy_failed_closed",
        )
