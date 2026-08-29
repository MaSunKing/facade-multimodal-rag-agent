"""Bounded semantic tool-plan schema and deterministic policy guard.

The local model owns business intent.  This module only validates the model
contract, removes unavailable/unauthorised tools and supplies a conservative
fallback when planning fails.  In particular, product, case and visual intent
must not be inferred here from a growing keyword list.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ToolName = Literal["general_chat", "customer_documents", "company_rag", "visual_inspection", "public_web_search"]
BusinessIntent = Literal[
    "product_parameter",
    "technical_performance",
    "application_condition",
    "construction",
    "quote_delivery",
    "warranty",
    "case_reference",
    "comparison",
    "complaint_after_sales",
    "unknown",
]
TaskType = Literal[
    "factual_lookup",
    "procedure",
    "node_detail",
    "case_reference",
    "comparison",
    "project_fit",
    "commercial",
    "unknown",
]
VisualScope = Literal["mixed", "product", "case", "node", "process"]
WebSourceProfile = Literal[
    "auto",
    "construction_standard",
    "public_project",
    "manufacturer_product",
    "industry_news",
    "general_public",
]
DocumentScope = Literal["local_lookup", "whole_document", "cross_document", "unknown"]


class CaseFilters(BaseModel):
    locations: list[str] = Field(default_factory=list, max_length=5)
    project_types: list[str] = Field(default_factory=list, max_length=5)
    installation_methods: list[str] = Field(default_factory=list, max_length=5)
    products: list[str] = Field(default_factory=list, max_length=5)


class ToolPlan(BaseModel):
    tools: list[ToolName] = Field(default_factory=list, max_length=5)
    reason: str = Field(default="", max_length=300)
    web_source_profile: WebSourceProfile = "auto"
    # This is a semantic decision made by the planner.  The frontend checkbox
    # only grants permission; it must never imply that every turn needs web.
    requires_public_web: bool = False
    # Business semantics are generated together with the tool choice so the
    # execution layer does not run another keyword router over the same turn.
    intent: BusinessIntent = "unknown"
    task_type: TaskType = "unknown"
    retrieval_query: str = Field(default="", max_length=300)
    # Semantic attachment scope chosen by the model.  The retrieval layer
    # consumes this field directly and never reclassifies it with keywords.
    document_scope: DocumentScope = "unknown"
    target_terms: list[str] = Field(default_factory=list, max_length=8)
    case_reference: bool = False
    case_filters: CaseFilters = Field(default_factory=CaseFilters)
    product_overview: bool = False
    # This means that the user asked the knowledge base to return a relevant
    # product/case/node/process image.  It is distinct from visual_inspection,
    # which reads a newly uploaded image.
    wants_visuals: bool = False
    visual_scope: VisualScope = "mixed"


def fallback_plan(
    *,
    has_documents: bool,
    has_image: bool,
    facade_related: bool,
    web_requested: bool,
    intent: BusinessIntent = "unknown",
    task_type: TaskType = "unknown",
    retrieval_query: str = "",
    document_scope: DocumentScope = "unknown",
    target_terms: list[str] | None = None,
    case_reference: bool = False,
    case_filters: dict[str, list[str]] | CaseFilters | None = None,
    product_overview: bool = False,
    wants_visuals: bool = False,
    visual_scope: VisualScope = "mixed",
) -> ToolPlan:
    tools: list[ToolName] = []
    if has_documents:
        tools.append("customer_documents")
    if has_image:
        tools.append("visual_inspection")
    if facade_related:
        tools.append("company_rag")
    if web_requested:
        tools.append("public_web_search")
    if not tools:
        tools.append("general_chat")
    return ToolPlan(
        tools=tools,
        reason="deterministic_safe_fallback",
        web_source_profile="auto",
        requires_public_web=web_requested,
        intent=intent,
        task_type=task_type,
        retrieval_query=retrieval_query,
        document_scope=document_scope,
        target_terms=list(target_terms or [])[:8],
        case_reference=case_reference,
        case_filters=case_filters or CaseFilters(),
        product_overview=product_overview,
        wants_visuals=wants_visuals,
        visual_scope=visual_scope if wants_visuals else "mixed",
    )


def guard_plan(
    raw: dict[str, Any] | ToolPlan,
    *,
    has_documents: bool,
    has_image: bool,
    facade_related: bool,
    web_allowed: bool,
) -> ToolPlan:
    try:
        proposed = raw if isinstance(raw, ToolPlan) else ToolPlan.model_validate(raw)
    except Exception:
        return fallback_plan(
            has_documents=has_documents,
            has_image=has_image,
            facade_related=facade_related,
            # A malformed/failed model plan must fail closed for a metered
            # external tool.  Permission to search is not evidence that the
            # current question needs search.
            web_requested=False,
        )
    allowed: list[ToolName] = []
    for tool in proposed.tools:
        if tool == "customer_documents" and not has_documents:
            continue
        if tool == "visual_inspection" and not has_image:
            continue
        # Company RAG is an always-local, read-only tool. Whether a phrase
        # such as "your products" refers to company knowledge is a semantic
        # decision for the planner, not a keyword gate for this policy layer.
        if tool == "public_web_search":
            if not web_allowed or not proposed.requires_public_web:
                continue
        if tool not in allowed:
            allowed.append(tool)
    if has_documents and "customer_documents" not in allowed:
        allowed.insert(0, "customer_documents")
    if has_image and "visual_inspection" not in allowed:
        allowed.append("visual_inspection")
    if proposed.product_overview and "company_rag" not in allowed:
        # This is contract consistency, not keyword routing: a plan that says
        # it needs the reviewed company catalogue also needs its local source.
        allowed.insert(0, "company_rag")
    if not allowed:
        allowed = ["company_rag"] if facade_related else ["general_chat"]
    task_type = "case_reference" if proposed.case_reference else proposed.task_type
    case_reference = task_type == "case_reference"
    return ToolPlan(
        tools=allowed[:5],
        reason=proposed.reason or "model_planned_policy_guarded",
        web_source_profile=proposed.web_source_profile if "public_web_search" in allowed else "auto",
        requires_public_web="public_web_search" in allowed and proposed.requires_public_web,
        intent=proposed.intent,
        task_type=task_type,
        retrieval_query=proposed.retrieval_query,
        document_scope=proposed.document_scope,
        target_terms=proposed.target_terms,
        case_reference=case_reference,
        case_filters=proposed.case_filters,
        product_overview=proposed.product_overview,
        wants_visuals=proposed.wants_visuals,
        visual_scope=proposed.visual_scope if proposed.wants_visuals else "mixed",
    )
