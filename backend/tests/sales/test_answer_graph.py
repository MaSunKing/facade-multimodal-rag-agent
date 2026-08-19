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


class CustomerAnswerGraphTests(unittest.TestCase):
    def test_plan_is_executed_once_without_persistence(self) -> None:
        callbacks = FakeCallbacks()
        graph = build_customer_answer_graph(callbacks)
        state = graph.invoke({"request": FakeRequest(tools=["general_chat"])})
        self.assertEqual(callbacks.calls, ["plan", "execute"])
        self.assertEqual(state["response"]["meta"]["orchestration"]["persistence"], "disabled")
        self.assertEqual(state["response"]["meta"]["orchestration"]["planning_rounds"], 1)

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
