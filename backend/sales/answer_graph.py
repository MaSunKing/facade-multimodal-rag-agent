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
from backend.sales.stage_details import begin, end, plan_summary, step_input, result_summary
from backend.request_budget import reserve_recovery, current_budget
from backend.sales.gap_recovery import merge_sources, operation_signature

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph


class AnswerWorkflowCallbacks(Protocol):
    def plan_tools(self, request: Any) -> Any: ...

    def answer_planned(self, request: Any, plan: Any) -> Any: ...

    # Optional at runtime.  Implementations should inspect the first tool
    # result (for example, empty/low-coverage retrieval) and return a revised
    # plan, or ``None``. It may invoke the bounded local Replanner; the graph
    # keeps all GPU operations sequential for a single-GPU deployment.
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
    current_sources: dict[str, Any]
    gated_response: Any


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


def _review_draft(callbacks, request, plan, response, round_number):
    review=getattr(callbacks,'review_answer',None)
    if not callable(review):return response
    try:
        return review(request,plan,response,round_number)
    except Exception as exc:
        from backend.request_budget import RequestBudgetExceeded
        if isinstance(exc,RequestBudgetExceeded):raise
        failure=report_error(exc,stage='answer_audit')
        # Never turn a review exception into an implicitly accepted draft.
        return {**response,'meta':{**response.get('meta',{}),'answer_review':{
            'decision':'limited','status':'uncertain','error_code':failure['code'],
            'reason':('answer_review_input_budget_exceeded' if failure['code']=='CONTEXT_BUDGET_EXCEEDED'
                      else 'answer_review_execution_failed'),'supported_evidence_ids':[]}}}


def _plan_signature(plan: Any) -> tuple[Any, ...]:
    """Compare retry plans without serialising runtime-only planner metrics."""

    tools = tuple(_plan_value(plan, "tools", []) or [])
    return (
        tools,
        str(_plan_value(plan, "retrieval_query", "") or ""),
        str(_plan_value(plan, "document_scope", "unknown") or "unknown"),
        str(_plan_value(plan, 'document_operation', 'unknown')),
        str(_plan_value(plan, 'company_operation', 'unknown')),
        tuple(_plan_value(plan, 'knowledge_categories', []) or []),
        str(_plan_value(plan, 'company_collection', '') or ''),
        tuple(_plan_value(plan, 'target_document_ids', []) or []),
        bool(_plan_value(plan, "requires_public_web", False)),
        str(_plan_value(plan, "web_source_profile", "auto") or "auto"),
        json.dumps(_plan_value(plan, 'recovery_directive', {}) or {}, sort_keys=True),
        _plan_value(plan, 'document_visual_required', None),
    )


def build_customer_answer_graph(callbacks: AnswerWorkflowCallbacks):
    """Build a sequential graph with at most one evidence-driven retry."""

    staged = callable(getattr(callbacks, "open_steps", None))
    stage_nodes = ("customer_documents", "document_summary", "company_rag", "public_web_search", "visual_inspection", "visual_evidence",
                   "assess_coverage", "rerank_evidence", "compose_evidence", "generate_answer", "validate_answer")

    def plan_request(state: CustomerAnswerState, config: RunnableConfig) -> dict[str, Any]:
        enter_stage('plan_request')
        from backend.app import planner_context_input
        detail=begin('plan_request',{'question_length':len(getattr(state['request'],'customer_question','')),
                                    'context':planner_context_input(state['request'])})
        try:
            plan = callbacks.plan_tools(state["request"])
        except Exception as exc:
            end(detail,error_code=report_error(exc,stage='plan_request')['code'])
            raise
        end(detail,{'parsed_plan':plan_summary(plan)})
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
        if hasattr(state['request'], '_prepared_evidence'):
            state['request']._prepared_evidence = None
        validator = getattr(callbacks, "validate_plan", None)
        detail=begin('guard_tools',{'proposed_plan':plan_summary(state['plan'])})
        try:
            plan = validator(state["request"], state["plan"]) if callable(validator) else state["plan"]
        except Exception as exc:
            end(detail,error_code=report_error(exc,stage='guard_tools')['code'])
            raise
        end(detail,{'allowed_plan':plan_summary(plan)})
        cursor = WorkflowCursor(callbacks.open_steps(state["request"], plan))
        try:
            cursor.advance()
        except BaseException:
            cursor.close()
            raise
        return {"cursor": cursor, "plan": plan, "current_sources": {}, "gated_response": None,
                "tool_rounds": int(state.get("tool_rounds", 0)) + 1}

    def route_step(state):
        if state.get('gated_response') is not None:
            return 'collect_response'
        cursor = state["cursor"]
        if cursor.done:
            return "collect_response"
        if cursor.pending.node not in stage_nodes:
            # Attribute routing failure to the intended next stage, rather
            # than the last successfully completed tool.
            enter_stage(cursor.pending.node)
            detail = begin(cursor.pending.node, {'routing_status':'unregistered'})
            end(detail, error_code='WORKFLOW_NODE_UNREGISTERED')
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
        current_sources = dict(state.get('current_sources', {}))
        prior_trace = list(state.get('node_trace', []))
        if step.node == 'compose_evidence':
            evaluator = getattr(callbacks, 'evaluate_sources', None)
            if callable(evaluator):
                preparation_node = 'context_preflight' if callable(getattr(callbacks,'review_answer',None)) else 'evidence_evaluation'
                enter_stage(preparation_node)
                detail=begin(preparation_node,{'sources':{key:result_summary(value) for key,value in current_sources.items()},
                                                  'requirements':_plan_value(state['plan'],'answer_aspects',[])})
                try:
                    blocked = evaluator(state['request'], state['plan'], current_sources,
                                        int(state.get('tool_rounds', 1)))
                except Exception as exc:
                    end(detail,error_code=report_error(exc,stage=preparation_node)['code'])
                    raise
                audit=getattr(state['request'],'_evidence_gate_audit',{})
                audit_tokens=end(detail,{'verdict':audit,'blocked':blocked is not None},
                    error_code=audit.get('error_code') if str(audit.get('error_code') or '').startswith('AUDIT_') else None)
                prior_trace.append({'node': preparation_node, 'round': state.get('tool_rounds', 1),
                    'elapsed_ms': round((time.perf_counter()-started)*1000, 2), 'reused': False,
                    'operation_error': None,'token_usage':audit_tokens})
                if blocked is not None:
                    cursor.close()
                    return {'gated_response': blocked, 'node_trace': prior_trace}
                enter_stage('compose_evidence')
                started = time.perf_counter()
        detail=begin(step.node,step_input(step))
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
                    cursor.last_value = web_results[cache_key]
                    cursor.advance(web_results[cache_key])
                else:
                    try:
                        value = step.operation(*step.args, **step.kwargs)
                    except Exception as exc:
                        cursor.operation_error = type(exc).__name__
                        report_error(exc,stage=step.node)
                        cursor.advance(error=exc)
                    else:
                        value = merge_sources(getattr(state['request'],'_prior_source_results',{}).get(step.node),value,step.node)
                        web_results[cache_key] = value
                        cursor.last_value = value
                        cursor.advance(value)
            elif step.node in {'customer_documents', 'company_rag'} and step.operation:
                # Reuse successful unchanged CPU tool results during a repair.
                # GPU tensors, visual calls and writes are never cached here.
                cache_key = repr((step.node, operation_signature(state['plan'],step.node), step.args, step.kwargs))
                retained = getattr(callbacks, 'retained_source', None)
                prior = retained(state['request'], step.node) if callable(retained) else None
                if prior is not None:
                    cached = True
                    cursor.last_value = prior
                    cursor.advance(prior)
                elif cache_key in tool_results:
                    cached = True
                    cursor.last_value = tool_results[cache_key]
                    cursor.advance(tool_results[cache_key])
                else:
                    try:
                        value = step.operation(*step.args, **step.kwargs)
                    except Exception as exc:
                        cursor.operation_error=type(exc).__name__
                        report_error(exc,stage=step.node)
                        cursor.advance(error=exc)
                    else:
                        value=merge_sources(getattr(state['request'],'_prior_source_results',{}).get(step.node),value,step.node)
                        tool_results[cache_key]=value
                        cursor.last_value=value
                        cursor.advance(value)
            else:
                cursor.execute()
        except BaseException as exc:
            failure=report_error(exc,stage=step.node) if isinstance(exc,Exception) else {'code':'REQUEST_CANCELLED'}
            end(detail,error_code=failure['code'])
            cursor.close()
            raise
        node_tokens=end(detail,{'result':result_summary(cursor.result if cursor.done else getattr(cursor,'last_value',None)), 'reused':cached},
            error_code=cursor.operation_error)
        trace = {"node": step.node, "round": state.get("tool_rounds", 1),
                 "elapsed_ms": round((time.perf_counter()-started)*1000, 2), "reused": cached,
                 "operation_error": cursor.operation_error,'token_usage':node_tokens}
        if step.node in {'company_rag', 'customer_documents', 'public_web_search'}:
            current_sources[step.node] = getattr(cursor, 'last_value', None)
        return {"cursor": cursor, "current_sources": current_sources, "node_trace": [*prior_trace, trace],
                "web_results": web_results, 'tool_results': tool_results}

    def collect_response(state, config):
        if state.get('gated_response') is not None:
            return {'response': _as_response_dict(state['gated_response'])}
        cursor = state["cursor"]
        try:
            response = _as_response_dict(cursor.result)
            response = _review_draft(callbacks,state['request'],state['plan'],response,int(state.get('tool_rounds',1)))
            return {"response":response}
        finally:
            cursor.close()

    def execute_plan(state: CustomerAnswerState, config: RunnableConfig) -> dict[str, Any]:
        response = _as_response_dict(callbacks.answer_planned(state['request'],state['plan']))
        response=_review_draft(callbacks,state['request'],state['plan'],response,int(state.get('tool_rounds',0))+1)
        return {
            "response": response,
            "tool_rounds": int(state.get("tool_rounds", 0)) + 1,
        }

    def assess_retry(state: CustomerAnswerState, config: RunnableConfig) -> dict[str, Any]:
        if int(state.get("tool_rounds", 0)) > MAX_TOOL_RETRIES:
            return {"retry_plan": None}
        retry_callback = getattr(callbacks, "plan_retry", None)
        if not callable(retry_callback):
            return {"retry_plan": None}
        detail=begin('assess_retry',{'old_plan':plan_summary(state['plan']),
                    'response':result_summary(_as_response_dict(state['response']))})
        try:
            revised = retry_callback(state["request"], state["plan"], state["response"])
        except Exception as exc:
            # A best-effort retry policy must never discard a valid first-pass
            # response.  The application can observe the absence of a retry;
            # normal endpoint-level error handling remains reserved for the
            # actual plan and answer nodes.
            report_error(code='RETRY_POLICY_FAILED',stage='assess_retry')
            end(detail,error_code='RETRY_POLICY_FAILED')
            return {"retry_plan": None, "retry_reason": "retry_policy_failed_closed"}
        end(detail,{'revised_plan':plan_summary(revised) if revised is not None else None,
                    'feedback':getattr(state['request'],'_replan_audit',{})})
        if revised is None or _plan_signature(revised) == _plan_signature(state["plan"]):
            return {"retry_plan": None}
        directive = _plan_value(revised, 'recovery_directive', {}) or {}
        stage = str(directive.get('stage') or 'assess_retry')
        action = str(directive.get('action') or 'repair_evidence_gaps')
        minimum = 45 if _as_response_dict(state['response']).get('meta', {}).get('model_used') else 30
        pre_reserved = bool(_as_response_dict(state['response']).get('meta', {}).get('evidence_gate', {}).get('recovery_reserved'))
        if not pre_reserved and not reserve_recovery(stage, action, minimum_seconds=minimum):
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
        finish = getattr(callbacks,'finish_review',None)
        if callable(finish):
            response=finish(state['request'],state['plan'],response)
        meta = response.get("meta")
        plan = state.get("plan")
        tools = _plan_value(plan, "tools", [])
        web_source_profile = _plan_value(plan, "web_source_profile", "auto")
        requires_public_web = _plan_value(plan, "requires_public_web", False)
        plan_reason = _plan_value(plan, "reason", "")
        plan_history = state.get("plan_history", [])
        response["meta"] = {
            **(meta if isinstance(meta, dict) else {}),
            'evidence_gate': getattr(state['request'], '_evidence_gate_audit', {}) or (meta or {}).get('evidence_gate', {}),
            'replanning': getattr(state['request'], '_replan_audit', {}),
            "orchestration": {
                "engine": "langgraph",
                "workflow": "bounded_tool_agent_v3" if staged else "bounded_tool_agent_v2",
                "node_trace": state.get("node_trace", []),
                "tool_execution_policy": "sequential_stages_parallel_independent_recall" if staged else "executor_managed",
                "tools": list(tools or []),
                "answer_basis": _plan_value(plan, "answer_basis", "tools"),
                "web_source_profile": web_source_profile or "auto",
                "requires_public_web": bool(requires_public_web),
                "plan_reason": str(plan_reason or ""),
                "guard_diagnostics": list(_plan_value(plan, 'guard_diagnostics', []) or []),
                "query_policy": "original_first_feedback_rewrite_local_context_auxiliary",
                "retrieval_query": str(_plan_value(plan, "retrieval_query", "") or ""),
                "document_visual_required": _plan_value(plan, "document_visual_required", None),
                "visual_task": _plan_value(plan, "visual_task", 'unknown'),
                "gallery_product_family": _plan_value(plan, "gallery_product_family", 'unknown'),
                "gallery_scope": _plan_value(plan, "gallery_scope", 'unknown'),
                "gallery_target": _plan_value(plan, "gallery_target", ''),
                "intent": str(_plan_value(plan, "intent", "unknown") or "unknown"),
                "task_type": str(_plan_value(plan, "task_type", "unknown") or "unknown"),
                "target_terms": list(_plan_value(plan, "target_terms", []) or []),
                "search_targets": list(_plan_value(plan, 'search_targets', []) or []),
                "answer_goals": list(_plan_value(plan, 'answer_goals', []) or []),
                "answer_aspects": list(_plan_value(plan, 'answer_aspects', []) or []),
                "retrieval_queries": list(_plan_value(plan, 'retrieval_queries', []) or []),
                "retrieval_requests": [r.model_dump() if hasattr(r,'model_dump') else r
                    for r in (_plan_value(plan, 'retrieval_requests', []) or [])],
                "planner_latency_ms": round(float(_plan_value(plan, "planner_latency_ms", 0.0) or 0.0), 2),
                "planner_model_load_ms": round(float(_plan_value(plan, "planner_model_load_ms", 0.0) or 0.0), 2),
                "planner_generation_ms": round(float(_plan_value(plan, "planner_generation_ms", 0.0) or 0.0), 2),
                "planner_cache_hit": bool(_plan_value(plan, "planner_cache_hit", False)),
                "planner_input_tokens": int(_plan_value(plan, "planner_input_tokens", 0) or 0),
                "planner_output_tokens": int(_plan_value(plan, "planner_output_tokens", 0) or 0),
                "planning_rounds": int(state.get("planning_rounds", 1)),
                "tool_rounds": int(state.get("tool_rounds", 1)),
                "max_tool_retries": MAX_TOOL_RETRIES,
                "max_answer_revisions": 1,
                "answer_revision_count": getattr(state['request'],'_answer_revision_count',0),
                "answer_audit_policy": "post_draft_text_only",
                "max_request_recoveries": 2,
                "max_execution_recoveries": 1,
                "max_query_normalizations": 1,
                "recovery_actions": list(current_budget.get().recoveries) if current_budget.get() else [],
                "bounded_retry_attempted": int(state.get("tool_rounds", 1)) > 1,
                "retry_reason": str(state.get("retry_reason", "")),
                "tool_plan_history": [list(_plan_value(item, "tools", []) or []) for item in plan_history],
                "tool_plan_details": [item.model_dump(exclude={'memory_updates'}) if hasattr(item, 'model_dump')
                                      else dict(item) for item in plan_history],
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
