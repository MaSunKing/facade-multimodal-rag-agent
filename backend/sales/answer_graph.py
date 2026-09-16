"""Finite, stateless LangGraph orchestration for customer questions.

The local model proposes tools once and a policy layer validates the proposal.
Tools normally execute once.  An executor may optionally request one bounded,
evidence-driven retry through ``plan_retry``; there is deliberately no
free-form loop, parallel GPU branch or persistent graph memory.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict
import time
import json
from backend.sales.staged_execution import WorkflowCursor
from backend.sales.runtime_status import enter_stage, report_error
from backend.request_budget import reserve_recovery, current_budget

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph


class AnswerWorkflowCallbacks(Protocol):
    def plan_tools(self, request: Any) -> Any: ...

    def answer_planned(self, request: Any, plan: Any) -> Any: ...

    # Optional at runtime.  Implementations should inspect the first tool
    # result (for example, empty/low-coverage retrieval) and return a revised
    # plan, or ``None``.  It should be a CPU-only policy decision; the graph
    # keeps all execution nodes sequential for a single-GPU deployment.
    def plan_retry(self, request: Any, plan: Any, response: Any) -> Any | None: ...


class CustomerAnswerState(TypedDict, total=False):
    request: Any
    plan: Any
    response: Any
    planning_rounds: int
    tool_rounds: int
    retry_plan: Any
    retry_reason: str
    plan_history: list[Any]
    cursor: Any
    node_trace: list[dict[str, Any]]
    web_results: dict[str, Any]
    tool_results: dict[str, Any]


MAX_TOOL_RETRIES = 1


def _as_response_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"Answer route returned unsupported response type: {type(value).__name__}")


def _plan_value(plan: Any, field: str, default: Any = None) -> Any:
    if isinstance(plan, dict):
        return plan.get(field, default)
    return getattr(plan, field, default)


def _plan_signature(plan: Any) -> tuple[Any, ...]:
    """Compare retry plans without serialising runtime-only planner metrics."""

    tools = tuple(_plan_value(plan, "tools", []) or [])
    return (
        tools,
        str(_plan_value(plan, "retrieval_query", "") or ""),
        str(_plan_value(plan, "document_scope", "unknown") or "unknown"),
        bool(_plan_value(plan, "requires_public_web", False)),
        str(_plan_value(plan, "web_source_profile", "auto") or "auto"),
        json.dumps(_plan_value(plan, 'recovery_directive', {}) or {}, sort_keys=True),
        _plan_value(plan, 'document_visual_required', None),
    )


def build_customer_answer_graph(callbacks: AnswerWorkflowCallbacks):
    """Build a sequential graph with at most one evidence-driven retry."""

    staged = callable(getattr(callbacks, "open_steps", None))
    stage_nodes = ("customer_documents", "company_rag", "public_web_search", "visual_inspection",
                   "assess_coverage", "rerank_evidence", "compose_evidence", "generate_answer", "validate_answer")

    def plan_request(state: CustomerAnswerState, config: RunnableConfig) -> dict[str, Any]:
        enter_stage('plan_request')
        plan = callbacks.plan_tools(state["request"])
        return {
            "plan": plan,
            "planning_rounds": 1,
            "tool_rounds": 0,
            "retry_reason": "",
            "plan_history": [plan],
            "node_trace": [],
            "web_results": {},
            "tool_results": {},
        }

    def prepare_workflow(state, config):
        enter_stage('guard_tools')
        validator = getattr(callbacks, "validate_plan", None)
        plan = validator(state["request"], state["plan"]) if callable(validator) else state["plan"]
        cursor = WorkflowCursor(callbacks.open_steps(state["request"], plan))
        try:
            cursor.advance()
        except BaseException:
            cursor.close()
            raise
        return {"cursor": cursor, "plan": plan, "tool_rounds": int(state.get("tool_rounds", 0)) + 1}

    def route_step(state):
        cursor = state["cursor"]
        if cursor.done:
            return "collect_response"
        if cursor.pending.node not in stage_nodes:
            cursor.close()
            raise ValueError("unregistered_workflow_node")
        return cursor.pending.node

    def run_step(state, config):
        cursor = state["cursor"]
        step = cursor.pending
        enter_stage(step.node)
        started = time.perf_counter()
        cached = False
        cursor.operation_error = None
        web_results = dict(state.get("web_results", {}))
        tool_results = dict(state.get('tool_results', {}))
        try:
            check = getattr(callbacks, "check_step", None)
            if callable(check):
                check(state["request"], state["plan"], step.node)
            if step.node == "public_web_search" and step.operation:
                # Same request, same arguments: a retrieval-scope retry must
                # not spend a second paid web call (including empty results).
                cache_key = repr((step.args, step.kwargs))
                if cache_key in web_results:
                    cached = True
                    cursor.advance(web_results[cache_key])
                else:
                    try:
                        value = step.operation(*step.args, **step.kwargs)
                    except Exception as exc:
                        cursor.operation_error = type(exc).__name__
                        report_error(exc,stage=step.node)
                        cursor.advance(error=exc)
                    else:
                        web_results[cache_key] = value
                        cursor.advance(value)
            elif step.node in {'customer_documents', 'company_rag'} and step.operation:
                # Reuse successful unchanged CPU tool results during a repair.
                # GPU tensors, visual calls and writes are never cached here.
                cache_key = repr((step.node, step.args, step.kwargs))
                if cache_key in tool_results:
                    cached = True
                    cursor.advance(tool_results[cache_key])
                else:
                    cursor.execute()
                    if cursor.operation_error is None and getattr(cursor, 'last_value', None) is not None:
                        tool_results[cache_key] = cursor.last_value
            else:
                cursor.execute()
        except BaseException as exc:
            if isinstance(exc,Exception): report_error(exc,stage=step.node)
            cursor.close()
            raise
        trace = {"node": step.node, "round": state.get("tool_rounds", 1),
                 "elapsed_ms": round((time.perf_counter()-started)*1000, 2), "reused": cached,
                 "operation_error": cursor.operation_error}
        return {"cursor": cursor, "node_trace": [*state.get("node_trace", []), trace],
                "web_results": web_results, 'tool_results': tool_results}

    def collect_response(state, config):
        cursor = state["cursor"]
        try:
            return {"response": _as_response_dict(cursor.result)}
        finally:
            cursor.close()

    def execute_plan(state: CustomerAnswerState, config: RunnableConfig) -> dict[str, Any]:
        return {
            "response": callbacks.answer_planned(state["request"], state["plan"]),
            "tool_rounds": int(state.get("tool_rounds", 0)) + 1,
        }

    def assess_retry(state: CustomerAnswerState, config: RunnableConfig) -> dict[str, Any]:
        if int(state.get("tool_rounds", 0)) > MAX_TOOL_RETRIES:
            return {"retry_plan": None}
        retry_callback = getattr(callbacks, "plan_retry", None)
        if not callable(retry_callback):
            return {"retry_plan": None}
        try:
            revised = retry_callback(state["request"], state["plan"], state["response"])
        except Exception as exc:
            # A best-effort retry policy must never discard a valid first-pass
            # response.  The application can observe the absence of a retry;
            # normal endpoint-level error handling remains reserved for the
            # actual plan and answer nodes.
            report_error(code='RETRY_POLICY_FAILED',stage='assess_retry')
            return {"retry_plan": None, "retry_reason": "retry_policy_failed_closed"}
        if revised is None or _plan_signature(revised) == _plan_signature(state["plan"]):
            return {"retry_plan": None}
        directive = _plan_value(revised, 'recovery_directive', {}) or {}
        stage = str(directive.get('stage') or 'assess_retry')
        action = str(directive.get('action') or 'repair_evidence_gaps')
        minimum = 45 if _as_response_dict(state['response']).get('meta', {}).get('model_used') else 30
        if not reserve_recovery(stage, action, minimum_seconds=minimum):
            return {'retry_plan':None,'retry_reason':'shared_recovery_budget_exhausted'}
        return {
            "retry_plan": revised,
            "retry_reason": str(_plan_value(revised, "reason", "bounded_evidence_retry") or "bounded_evidence_retry"),
        }

    def route_after_retry_assessment(state: CustomerAnswerState) -> str:
        return "retry" if state.get("retry_plan") is not None else "finalize"

    def apply_retry_plan(state: CustomerAnswerState, config: RunnableConfig) -> dict[str, Any]:
        revised = state["retry_plan"]
        return {
            "plan": revised,
            "retry_plan": None,
            "planning_rounds": int(state.get("planning_rounds", 1)) + 1,
            "plan_history": [*state.get("plan_history", []), revised],
        }

    def finalize_response(state: CustomerAnswerState, config: RunnableConfig) -> dict[str, dict[str, Any]]:
        response = _as_response_dict(state["response"])
        meta = response.get("meta")
        plan = state.get("plan")
        tools = _plan_value(plan, "tools", [])
        web_source_profile = _plan_value(plan, "web_source_profile", "auto")
        requires_public_web = _plan_value(plan, "requires_public_web", False)
        plan_reason = _plan_value(plan, "reason", "")
        plan_history = state.get("plan_history", [])
        response["meta"] = {
            **(meta if isinstance(meta, dict) else {}),
            "orchestration": {
                "engine": "langgraph",
                "workflow": "bounded_tool_agent_v3" if staged else "bounded_tool_agent_v2",
                "node_trace": state.get("node_trace", []),
                "tool_execution_policy": "sequential" if staged else "executor_managed",
                "tools": list(tools or []),
                "answer_basis": _plan_value(plan, "answer_basis", "tools"),
                "web_source_profile": web_source_profile or "auto",
                "requires_public_web": bool(requires_public_web),
                "plan_reason": str(plan_reason or ""),
                "retrieval_query": str(_plan_value(plan, "retrieval_query", "") or ""),
                "document_visual_required": _plan_value(plan, "document_visual_required", None),
                "intent": str(_plan_value(plan, "intent", "unknown") or "unknown"),
                "task_type": str(_plan_value(plan, "task_type", "unknown") or "unknown"),
                "target_terms": list(_plan_value(plan, "target_terms", []) or []),
                "search_targets": list(_plan_value(plan, 'search_targets', []) or []),
                "answer_goals": list(_plan_value(plan, 'answer_goals', []) or []),
                "answer_aspects": list(_plan_value(plan, 'answer_aspects', []) or []),
                "retrieval_queries": list(_plan_value(plan, 'retrieval_queries', []) or []),
                "planner_latency_ms": round(float(_plan_value(plan, "planner_latency_ms", 0.0) or 0.0), 2),
                "planner_model_load_ms": round(float(_plan_value(plan, "planner_model_load_ms", 0.0) or 0.0), 2),
                "planner_generation_ms": round(float(_plan_value(plan, "planner_generation_ms", 0.0) or 0.0), 2),
                "planner_cache_hit": bool(_plan_value(plan, "planner_cache_hit", False)),
                "planner_input_tokens": int(_plan_value(plan, "planner_input_tokens", 0) or 0),
                "planner_output_tokens": int(_plan_value(plan, "planner_output_tokens", 0) or 0),
                "planning_rounds": int(state.get("planning_rounds", 1)),
                "tool_rounds": int(state.get("tool_rounds", 1)),
                "max_tool_retries": MAX_TOOL_RETRIES,
                "max_request_recoveries": 2,
                "max_execution_recoveries": 1,
                "max_query_normalizations": 1,
                "recovery_actions": list(current_budget.get().recoveries) if current_budget.get() else [],
                "bounded_retry_attempted": int(state.get("tool_rounds", 1)) > 1,
                "retry_reason": str(state.get("retry_reason", "")),
                "tool_plan_history": [list(_plan_value(item, "tools", []) or []) for item in plan_history],
                "gpu_execution_policy": "sequential",
                "persistence": "disabled",
            },
        }
        return {"response": response}

    graph = StateGraph(CustomerAnswerState)
    graph.add_node("plan_request", plan_request)
    if staged:
        graph.add_node("guard_tools", prepare_workflow)
        graph.add_node("collect_response", collect_response)
        routes = {name: name for name in (*stage_nodes, "collect_response")}
        graph.add_conditional_edges("guard_tools", route_step, routes)
        for name in stage_nodes:
            graph.add_node(name, run_step)
            graph.add_conditional_edges(name, route_step, routes)
        graph.add_edge("collect_response", "assess_retry")
    else:
        graph.add_node("execute_plan", execute_plan)
        graph.add_edge("execute_plan", "assess_retry")
    graph.add_node("assess_retry", assess_retry)
    graph.add_node("apply_retry_plan", apply_retry_plan)
    graph.add_node("finalize_response", finalize_response)
    graph.add_edge(START, "plan_request")
    graph.add_edge("plan_request", "guard_tools" if staged else "execute_plan")
    graph.add_conditional_edges(
        "assess_retry",
        route_after_retry_assessment,
        {"retry": "apply_retry_plan", "finalize": "finalize_response"},
    )
    graph.add_edge("apply_retry_plan", "guard_tools" if staged else "execute_plan")
    graph.add_edge("finalize_response", END)
    return graph.compile()
