"""Bounded tool-plan schema and deterministic policy guard."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ToolName = Literal["general_chat", "customer_documents", "company_rag", "visual_inspection", "public_web_search"]


class ToolPlan(BaseModel):
    tools: list[ToolName] = Field(default_factory=list, max_length=5)
    reason: str = Field(default="", max_length=300)


def fallback_plan(*, has_documents: bool, has_image: bool, facade_related: bool, web_requested: bool) -> ToolPlan:
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
    return ToolPlan(tools=tools, reason="deterministic_safe_fallback")


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
            web_requested=web_allowed,
        )
    allowed: list[ToolName] = []
    for tool in proposed.tools:
        if tool == "customer_documents" and not has_documents:
            continue
        if tool == "visual_inspection" and not has_image:
            continue
        if tool == "company_rag" and not facade_related:
            continue
        if tool == "public_web_search" and not web_allowed:
            continue
        if tool not in allowed:
            allowed.append(tool)
    if has_documents and "customer_documents" not in allowed:
        allowed.insert(0, "customer_documents")
    if has_image and "visual_inspection" not in allowed:
        allowed.append("visual_inspection")
    if not allowed:
        allowed = ["company_rag"] if facade_related else ["general_chat"]
    return ToolPlan(tools=allowed[:5], reason=proposed.reason or "model_planned_policy_guarded")
