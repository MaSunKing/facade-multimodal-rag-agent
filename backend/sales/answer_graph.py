"""Finite, stateless LangGraph orchestration for customer questions.

The local model proposes tools once, a policy layer validates the proposal,
and the selected tools execute once.  There is deliberately no free-form loop
or persistent graph memory.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph


class AnswerWorkflowCallbacks(Protocol):
    def plan_tools(self, request: Any) -> Any: ...

    def answer_planned(self, request: Any, plan: Any) -> Any: ...


class CustomerAnswerState(TypedDict, total=False):
    request: Any
    plan: Any
    response: Any


def _as_response_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"Answer route returned unsupported response type: {type(value).__name__}")


def build_customer_answer_graph(callbacks: AnswerWorkflowCallbacks):
    """Build a plan-once/execute-once graph with no checkpointer."""

    def plan_request(state: CustomerAnswerState, config: RunnableConfig) -> dict[str, Any]:
        return {"plan": callbacks.plan_tools(state["request"])}

    def execute_plan(state: CustomerAnswerState, config: RunnableConfig) -> dict[str, Any]:
        return {"response": callbacks.answer_planned(state["request"], state["plan"])}

    def finalize_response(state: CustomerAnswerState, config: RunnableConfig) -> dict[str, dict[str, Any]]:
        response = _as_response_dict(state["response"])
        meta = response.get("meta")
        plan = state.get("plan")
        tools = getattr(plan, "tools", None)
        web_source_profile = getattr(plan, "web_source_profile", None)
        requires_public_web = getattr(plan, "requires_public_web", None)
        plan_reason = getattr(plan, "reason", None)
        if tools is None and isinstance(plan, dict):
            tools = plan.get("tools")
            web_source_profile = plan.get("web_source_profile")
            requires_public_web = plan.get("requires_public_web")
            plan_reason = plan.get("reason")
        response["meta"] = {
            **(meta if isinstance(meta, dict) else {}),
            "orchestration": {
                "engine": "langgraph",
                "workflow": "bounded_tool_agent_v2",
                "tools": list(tools or []),
                "web_source_profile": web_source_profile or "auto",
                "requires_public_web": bool(requires_public_web),
                "plan_reason": str(plan_reason or ""),
                "planning_rounds": 1,
                "tool_rounds": 1,
                "persistence": "disabled",
            },
        }
        return {"response": response}

    graph = StateGraph(CustomerAnswerState)
    graph.add_node("plan_request", plan_request)
    graph.add_node("execute_plan", execute_plan)
    graph.add_node("finalize_response", finalize_response)
    graph.add_edge(START, "plan_request")
    graph.add_edge("plan_request", "execute_plan")
    graph.add_edge("execute_plan", "finalize_response")
    graph.add_edge("finalize_response", END)
    return graph.compile()
