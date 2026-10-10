"""Bounded semantic tool-plan schema and deterministic policy guard.

The local model owns business intent.  This module only validates the model
contract, removes unavailable/unauthorised tools and supplies a conservative
fallback when planning fails.  In particular, product, case and visual intent
must not be inferred here from a growing keyword list.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict, model_validator
from backend.sales.task_memory import MemoryUpdate


MAX_RETRIEVAL_QUERY_CHARACTERS = 300


def local_retrieval_query(question: str, supplemental_query: str = "") -> str:
    """Keep the entire current question primary; expansion never replaces it.

    Local-only: never use this helper to export private question text to web.
    No prefix truncation, domain routing or inference from keywords.
    """
    primary = question.strip()
    supplement = supplemental_query.strip()
    return "\n".join(dict.fromkeys(q for q in (primary, supplement) if q))


def retrieval_query_for_attempt(question: str, plan=None, *, public=False, source=None) -> str:
    """Use independent source-language equivalents of the current task.

    Do not export history, evidence or local excerpts to public search.
    Permissions remain an execution-time guard, not a property of this helper.
    The legacy unspecified-source contract retains the original first query.
    """
    source = 'public_web_search' if public else source
    if source == 'company_rag':
        query = getattr(plan, 'company_retrieval_query', '') or ''
        # Never silently submit an English-only query to the Chinese corpus.
        import re
        if query.strip() and re.search(r'[\u4e00-\u9fff]', query):
            return query.strip()
        if re.search(r'[\u4e00-\u9fff]', question):
            rewritten = getattr(plan, 'retrieval_query', '') or ''
            if getattr(plan, 'recovery_directive', {}) and re.search(r'[\u4e00-\u9fff]', rewritten):
                return local_retrieval_query(question, rewritten)
            return question.strip()
        raise ValueError('company_chinese_query_missing')
    if source == 'customer_documents':
        queries = getattr(plan, 'document_retrieval_queries', []) or []
        if queries:
            return '\n'.join(dict.fromkeys(q.strip() for q in queries if q.strip()))
    if source == 'public_web_search':
        from backend.sales.response_language import compatible_web_query
        query = getattr(plan, 'web_retrieval_query', '') or ''
        return compatible_web_query(question, query)
    directive = getattr(plan, 'recovery_directive', {}) or {}
    if not directive:
        return question.strip()
    rewritten = getattr(plan, 'retrieval_query', '') or ''
    return (rewritten.strip() or question.strip()) if public else local_retrieval_query(question, rewritten)

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
    model_config = ConfigDict(extra='forbid')
    locations: list[str] = Field(default_factory=list, max_length=5)
    project_types: list[str] = Field(default_factory=list, max_length=5)
    installation_methods: list[str] = Field(default_factory=list, max_length=5)
    products: list[str] = Field(default_factory=list, max_length=5)


class CompanyCollection(BaseModel):
    """Safe directory operations, not arbitrary model-authored SQL."""
    model_config = ConfigDict(extra='forbid')
    action: Literal['none', 'list', 'count'] = 'none'
    group_by: list[Literal['category', 'province', 'city', 'project_category',
        'family', 'product_form', 'surface_finish', 'document_name', 'section_heading']] = Field(default_factory=list, max_length=2)


class RetrievalRequest(BaseModel):
    """Model-authored query and scope descriptions; never model-authored IDs/ACL."""
    model_config = ConfigDict(extra='forbid')
    source: Literal['company_rag', 'customer_documents', 'public_web_search']
    query: str = Field(min_length=1, max_length=300)
    targets: list[str] = Field(default_factory=list, max_length=8)


class ToolPlan(BaseModel):
    # New model wire schema is lowered once; legacy fields are execution adapters.
    canonical_task: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @model_validator(mode='before')
    @classmethod
    def accept_compact_task(cls, value):
        from backend.sales.planning_spec import expand_planning_spec
        return expand_planning_spec(value)

    memory_updates: list[MemoryUpdate] = Field(default_factory=list, max_length=4)
    answer_basis: Literal["tools", "conversation"] = "tools"
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
    retrieval_query: str = Field(default="", max_length=MAX_RETRIEVAL_QUERY_CHARACTERS)
    retrieval_queries: list[str] = Field(default_factory=list, max_length=3)
    retrieval_requests: list[RetrievalRequest] = Field(default_factory=list, max_length=8)
    company_retrieval_query: str = Field(default='', max_length=MAX_RETRIEVAL_QUERY_CHARACTERS)
    knowledge_categories: list[str] = Field(default_factory=list, max_length=6)
    company_operation: Literal['lookup', 'summary', 'comparison', 'unknown'] = 'unknown'
    company_collection: CompanyCollection = Field(default_factory=CompanyCollection)
    document_retrieval_queries: list[str] = Field(default_factory=list, max_length=8)
    web_retrieval_query: str = Field(default='', max_length=MAX_RETRIEVAL_QUERY_CHARACTERS)
    answer_language: str = Field(default='', max_length=40)
    # Semantic attachment scope chosen by the model.  The retrieval layer
    # consumes this field directly and never reclassifies it with keywords.
    document_scope: DocumentScope = "unknown"
    document_operation: Literal['lookup', 'summary', 'comparison', 'unknown'] = 'unknown'
    # Only an explicit user restriction may narrow the active attachment set.
    document_selection: Literal['all', 'explicit'] = 'all'
    target_document_ids: list[str] = Field(default_factory=list, max_length=32)
    target_terms: list[str] = Field(default_factory=list, max_length=8)
    # Source-language retrieval goals are separate from user-language targets.
    search_targets: list[str] = Field(default_factory=list, max_length=8)
    answer_goals: list[str] = Field(default_factory=list, max_length=8)
    answer_aspects: list[str] = Field(default_factory=list, max_length=8)
    case_reference: bool = False
    case_filters: CaseFilters = Field(default_factory=CaseFilters)
    product_overview: bool = False
    # This means that the user asked the knowledge base to return a relevant
    # product/case/node/process image.  It is distinct from visual_inspection,
    # which reads a newly uploaded image.
    wants_visuals: bool = False
    document_visual_required: bool | None = None
    visual_scope: VisualScope = "mixed"
    # Semantic task contract, never inferred by substring rules. Unknown is conservative.
    visual_task: Literal['unknown', 'gallery', 'inspection', 'mixed'] = 'unknown'
    gallery_product_family: Literal['unknown', 'all', 'decorative_panel', 'insulated_panel'] = 'unknown'
    gallery_scope: Literal['unknown', 'category', 'exact', 'descriptive'] = 'unknown'
    gallery_target: str = Field(default='', max_length=120)
    # Runtime-only observability.  These fields are deliberately excluded
    # from the planner contract and API serialisation, but LangGraph may expose
    # them in response metadata for local latency audits.
    planner_latency_ms: float = Field(default=0.0, exclude=True)
    planner_model_load_ms: float = Field(default=0.0, exclude=True)
    planner_generation_ms: float = Field(default=0.0, exclude=True)
    planner_cache_hit: bool = Field(default=False, exclude=True)
    planner_input_tokens: int = Field(default=0, exclude=True)
    planner_output_tokens: int = Field(default=0, exclude=True)
    # Application-authored repair directive; never accepted as source evidence.
    recovery_directive: dict[str, Any] = Field(default_factory=dict, exclude=True)
    guard_diagnostics: list[dict[str, str]] = Field(default_factory=list, exclude=True)


def bounded_fallback_retrieval_query(query: str, target_terms: list[str] | None = None) -> str:
    """Bound a *fallback* search field, never truncate the user's question.

    Normal plans obtain a concise query from the semantic planner. If that
    planner is unavailable, keep complete existing anchors when they fit.
    With no usable anchors, leave the optional query empty: downstream local
    retrieval uses the original question, which remains on DraftRequest.
    A prefix cut would silently discard late conditions, entities or negations.
    This helper neither selects tools nor adds permissions.
    """
    normalised = ' '.join(query.split())
    if len(normalised) <= MAX_RETRIEVAL_QUERY_CHARACTERS:
        return normalised
    anchors: list[str] = []
    seen: set[str] = set()
    for term in target_terms or []:
        anchor = ' '.join(term.split())
        if not anchor or anchor.casefold() in seen:
            continue
        if len(' '.join([*anchors, anchor])) <= MAX_RETRIEVAL_QUERY_CHARACTERS:
            anchors.append(anchor)
            seen.add(anchor.casefold())
    return ' '.join(anchors)


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
    documents_relevant: bool | None = None,
    image_relevant: bool = True,
) -> ToolPlan:
    """Build a bounded fallback without confusing availability with relevance.

    ``has_documents`` and ``has_image`` describe capabilities available in the
    current request.  They do not, on their own, prove that the current
    question is about those attachments.  Callers that have a reliable
    semantic signal (for example, an explicit ``document_scope`` recovered by
    a lightweight fallback) opt in through ``documents_relevant`` or
    ``image_relevant``.  This keeps planner failures local and conservative
    without forcing every later turn through an old attachment session.
    """

    use_documents = (
        document_scope in {"local_lookup", "whole_document", "cross_document"}
        if documents_relevant is None
        else documents_relevant
    )
    tools: list[ToolName] = []
    if has_documents and use_documents:
        tools.append("customer_documents")
    if has_image and image_relevant:
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
        retrieval_query=bounded_fallback_retrieval_query(retrieval_query, target_terms),
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
        ).model_copy(update={'guard_diagnostics':[{'code':'PLAN_SCHEMA_INVALID','action':'safe_fallback'}]})
    allowed: list[ToolName] = []
    diagnostics = []
    for tool in proposed.tools:
        if tool == "customer_documents" and not has_documents:
            diagnostics.append({'tool':tool,'code':'ATTACHMENT_UNAVAILABLE','action':'removed'})
            continue
        if tool == "visual_inspection" and not has_image:
            diagnostics.append({'tool':tool,'code':'VISUAL_INPUT_UNAVAILABLE','action':'removed'})
            continue
        # Company RAG is an always-local, read-only tool. Whether a phrase
        # such as "your products" refers to company knowledge is a semantic
        # decision for the planner, not a keyword gate for this policy layer.
        if tool == "public_web_search":
            if not web_allowed or not proposed.requires_public_web:
                diagnostics.append({'tool':tool,'code':'WEB_NOT_AUTHORISED' if not web_allowed else 'WEB_NOT_REQUESTED','action':'removed'})
                continue
        if tool not in allowed:
            allowed.append(tool)
    # Attachment availability is not attachment relevance.  The semantic
    # planner may deliberately omit an old session when the user changes topic
    # (for example, from an uploaded workbook to today's weather).  Only a
    # semantic contract that explicitly scopes the question to documents may
    # repair a missing tool selection here.
    if (
        has_documents
        and proposed.document_scope in {"local_lookup", "whole_document", "cross_document"}
        and "customer_documents" not in allowed
    ):
        allowed.insert(0, "customer_documents")
    if proposed.product_overview and "company_rag" not in allowed:
        # This is contract consistency, not keyword routing: a plan that says
        # it needs the reviewed company catalogue also needs its local source.
        allowed.insert(0, "company_rag")
    if not allowed:
        allowed = ["company_rag"] if facade_related else ["general_chat"]
    if proposed.answer_basis == "conversation":
        # The MODEL classified this as recalling user-provided context, not
        # verifying a product/project claim. Do not search a public/company
        # corpus for the customer's private budget or earlier preferences.
        allowed = ["general_chat"]
    task_type = "case_reference" if proposed.case_reference else proposed.task_type
    case_reference = task_type == "case_reference"
    return ToolPlan(
        guard_diagnostics=diagnostics,
        canonical_task=proposed.canonical_task,
        tools=allowed[:5],
        answer_basis=proposed.answer_basis,
        memory_updates=proposed.memory_updates,
        reason=proposed.reason or "model_planned_policy_guarded",
        web_source_profile=proposed.web_source_profile if "public_web_search" in allowed else "auto",
        requires_public_web="public_web_search" in allowed and proposed.requires_public_web,
        intent=proposed.intent,
        task_type=task_type,
        retrieval_query=proposed.retrieval_query,
        retrieval_queries=list(dict.fromkeys(q.strip() for q in proposed.retrieval_queries if q.strip()))[:3],
        retrieval_requests=[r for r in proposed.retrieval_requests if r.source in allowed],
        company_retrieval_query=proposed.company_retrieval_query,
        knowledge_categories=proposed.knowledge_categories,
        company_operation=proposed.company_operation,
        company_collection=proposed.company_collection,
        document_retrieval_queries=proposed.document_retrieval_queries,
        web_retrieval_query=proposed.web_retrieval_query,
        answer_language=proposed.answer_language,
        document_scope=proposed.document_scope,
        document_selection=proposed.document_selection,
        document_operation=('comparison' if proposed.document_operation == 'unknown' and task_type == 'comparison'
                            else proposed.document_operation),
        target_document_ids=proposed.target_document_ids,
        target_terms=proposed.target_terms,
        search_targets=proposed.search_targets,
        answer_goals=proposed.answer_goals or proposed.answer_aspects,
        answer_aspects=proposed.answer_aspects,
        case_reference=case_reference,
        case_filters=proposed.case_filters,
        product_overview=proposed.product_overview,
        wants_visuals=proposed.wants_visuals,
        document_visual_required=proposed.document_visual_required,
        visual_task=proposed.visual_task,
        gallery_product_family=proposed.gallery_product_family,
        gallery_scope=proposed.gallery_scope,
        gallery_target=proposed.gallery_target,
        visual_scope=proposed.visual_scope if proposed.wants_visuals else "mixed",
    )
