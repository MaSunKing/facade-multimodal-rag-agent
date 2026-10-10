"""Compact model-visible task contract; one adapter owns legacy execution fields."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from backend.sales.tool_planner import CaseFilters, CompanyCollection, MemoryUpdate, ToolName

Operation = Literal['lookup', 'summary', 'comparison']
RetrievalSource = Literal['company_rag', 'customer_documents', 'public_web_search']


class ScopedQuery(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source: RetrievalSource
    query: str = Field(min_length=1, max_length=300)
    targets: list[str] = Field(default_factory=list, max_length=8)


class PlanningSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sources: list[ToolName] = Field(min_length=1, max_length=5)
    operation: Operation = 'lookup'
    categories: list[str] = Field(default_factory=list, max_length=6)
    targets: list[str] = Field(default_factory=list, max_length=8)
    filters: CaseFilters = Field(default_factory=CaseFilters)
    aspects: list[str] = Field(default_factory=list, max_length=8)
    wants_images: bool
    # Only mixed-source tasks need overrides. A source has one execution operation.
    source_operations: dict[RetrievalSource, Operation] = Field(default_factory=dict)
    query: str = Field(default='', max_length=300)
    queries: list[ScopedQuery] = Field(default_factory=list, max_length=8)
    collection: CompanyCollection = Field(default_factory=CompanyCollection)
    document_selection: Literal['all', 'explicit'] = 'all'
    document_names: list[str] = Field(default_factory=list, max_length=32)
    target_document_ids: list[str] = Field(default_factory=list, max_length=32)
    document_scope: Literal['local_lookup', 'whole_document', 'cross_document'] | None = None
    document_visual_required: bool | None = None
    image_scope: Literal['mixed', 'product', 'case', 'node', 'process'] = 'mixed'
    image_family: Literal['all', 'decorative_panel', 'insulated_panel'] | None = None
    image_target: str = Field(default='', max_length=120)
    image_match: Literal['category', 'exact', 'descriptive'] | None = None
    answer_basis: Literal['tools', 'conversation'] = 'tools'
    reason: str = Field(default='', max_length=300)
    memory_updates: list[MemoryUpdate] = Field(default_factory=list, max_length=2)


BACKEND_PLAN_FIELDS = frozenset({'answer_basis', 'document_scope', 'target_document_ids',
                               'document_selection', 'query', 'reason',
                               'image_scope', 'image_target', 'image_match',
                               'targets', 'filters', 'aspects'})


def planning_output_schema(allowed_sources=None):
    """Constrain wire fields, not user intent, SQL IDs or factual support."""
    from backend.sales.content_taxonomy import LABELS
    schema = PlanningSpec.model_json_schema()
    for field in BACKEND_PLAN_FIELDS:
        schema['properties'].pop(field, None)
    schema['required'] = ['sources', 'operation', 'categories', 'wants_images']
    schema.get('$defs', {}).get('ScopedQuery', {}).get('properties', {}).pop('targets', None)
    schema.get('$defs', {}).pop('CaseFilters', None)
    # Availability is a backend fact. A decision is required even when the
    # planner chooses another source; omission must not silently mean false.
    schema['properties']['document_visual_required'] = {'type': 'boolean'}
    if allowed_sources is not None:
        schema['properties']['sources']['items'] = {'type': 'string', 'enum': list(allowed_sources)}
        if 'customer_documents' not in allowed_sources:
            for field in ('document_selection', 'document_names', 'document_visual_required'):
                schema['properties'].pop(field, None)
        else:
            schema['required'].append('document_visual_required')
    schema['properties']['categories']['items'] = {'type': 'string', 'enum': list(LABELS)}
    return schema


def validate_visual_decision(value, *, documents_available=False):
    """Validate a wire decision, without guessing whether pixels are needed."""
    from backend.sales.output_contract import OutputContractError
    sources = value.get('sources')
    required = documents_available or isinstance(sources, list) and 'customer_documents' in sources
    if required and 'document_visual_required' not in value:
        raise OutputContractError('OUTPUT_FIELD_MISSING', 'document_visual_required')
    if 'document_visual_required' in value and type(value['document_visual_required']) is not bool:
        raise OutputContractError('OUTPUT_FIELD_VALUE_INVALID', 'document_visual_required')


def bind_document_names(value, directory):
    """Resolve explicit attachment names only within the caller's session.

    Exact names select their same-name set; unknown names fail closed.
    This is identity binding, not semantic task selection.
    """
    from backend.sales.output_contract import OutputContractError
    result = dict(value)
    names = result.get('document_names', [])
    if not isinstance(names, list) or any(not isinstance(n, str) or not n.strip() for n in names):
        raise OutputContractError('OUTPUT_FIELD_VALUE_INVALID', 'document_names')
    if names and 'document_selection' in result and result['document_selection'] != 'explicit':
        raise OutputContractError('OUTPUT_STATE_INCONSISTENT', 'document_names')
    identities = []
    for name in names:
        matches = [r['document_id'] for r in directory
                   if name in (r.get('file'), r.get('document_name'), r.get('display_name')) and r.get('document_id')]
        if not matches:
            raise OutputContractError('OUTPUT_FIELD_VALUE_INVALID', 'document_names')
        identities.extend(matches)
    if names:
        result['document_selection'] = 'explicit'
        result['target_document_ids'] = list(dict.fromkeys(identities))
    return result


def expand_planning_spec(value):
    """Validate the new wire contract then lower once to the existing tool API.

    No prose splitting, entity guessing, permissions or source ID allocation.
    Legacy plans are unchanged; mixing both schemas is an error, not precedence.
    """
    if not isinstance(value, dict) or 'sources' not in value:
        return value
    from backend.sales.output_contract import OutputContractError
    validate_visual_decision(value)
    # Default-empty optional strings accept null as omission. Required fields
    # and substantive targets/queries are never guessed or prefix-truncated.
    value = {k:v for k,v in value.items() if not (v is None and k in PlanningSpec.model_fields
        and PlanningSpec.model_fields[k].default == '')}
    try:
        spec = PlanningSpec.model_validate(value, strict=True)
    except Exception as exc:
        errors = exc.errors(include_input=False)
        field = str(errors[0]['loc'][0]) if errors else 'plan'
        kind = errors[0]['type'] if errors else ''
        failure = OutputContractError('OUTPUT_FIELD_UNEXPECTED' if kind == 'extra_forbidden'
                                      else 'OUTPUT_FIELD_MISSING' if kind == 'missing' else 'OUTPUT_FIELD_VALUE_INVALID', field)
        failure.validation_details = {'violations': [{'location':list(e['loc']), 'type':e['type']} for e in errors[:3]]}
        raise failure from exc
    retrieval = set(spec.sources) & {'company_rag','customer_documents','public_web_search'}
    for name in ('operation',):
        if retrieval and name not in value:
            raise OutputContractError('OUTPUT_FIELD_MISSING', name)
    if 'company_rag' in retrieval and 'categories' not in value:
        raise OutputContractError('OUTPUT_FIELD_MISSING', 'categories')
    if set(spec.source_operations) - retrieval:
        raise OutputContractError('OUTPUT_STATE_INCONSISTENT', 'source_operations')
    if spec.answer_basis == 'conversation' and set(spec.sources) != {'general_chat'}:
        raise OutputContractError('OUTPUT_STATE_INCONSISTENT', 'sources')
    company = spec.source_operations.get('company_rag', spec.operation)
    documents = spec.source_operations.get('customer_documents', spec.operation)
    # Summary already defines coverage; no second mandatory list/overview
    # classification. Explicit count/group operations remain available.
    source_queries = {s: [q.query for q in spec.queries if q.source == s] for s in retrieval}
    # Scope expresses content coverage, independently of selected files.
    scope = spec.document_scope or ('cross_document' if documents in {'summary','comparison'} else 'local_lookup')
    image_scopes = {'02_product_information':'product', '05_project_case':'case',
                    '03_construction_method':'process', '04_node_atlas':'node'}
    scopes = {image_scopes[c] for c in spec.categories if c in image_scopes}
    image_scope = spec.image_scope if 'image_scope' in value else next(iter(scopes)) if len(scopes)==1 else 'mixed'
    image_target = spec.image_target if 'image_target' in value else (
        spec.targets[0] if len(spec.targets)==1 and len(spec.targets[0])<=120 else '')
    image_match = spec.image_match or ('descriptive' if spec.targets else 'category')
    compact = spec.model_dump(exclude_unset=True)
    return dict(tools=spec.sources, company_operation=company if 'company_rag' in retrieval else 'unknown',
        document_operation=documents if 'customer_documents' in retrieval else 'unknown',
        knowledge_categories=spec.categories if 'company_rag' in retrieval else [], target_terms=spec.targets, case_filters=spec.filters.model_dump(),
        answer_aspects=spec.aspects, answer_goals=spec.aspects, wants_visuals=spec.wants_images,
        requires_public_web='public_web_search' in retrieval, retrieval_query=spec.query,
        # One source-language seed for the legacy base recall. All independent
        # queries remain in retrieval_requests; this never replaces their bindings.
        company_retrieval_query=next(iter(source_queries.get('company_rag',[])), ''),
        document_retrieval_queries=source_queries.get('customer_documents',[]),
        web_retrieval_query=source_queries.get('public_web_search',[''])[0] if len(source_queries.get('public_web_search',[])) == 1 else '',
        retrieval_requests=[q.model_dump() for q in spec.queries], company_collection=spec.collection.model_dump(),
        document_scope=scope if 'customer_documents' in retrieval else 'unknown',
        document_selection=spec.document_selection, target_document_ids=spec.target_document_ids,
        document_visual_required=spec.document_visual_required, visual_scope=image_scope,
        gallery_product_family=spec.image_family or 'unknown', gallery_target=image_target,
        gallery_scope=image_match, visual_task='mixed' if spec.wants_images else 'unknown',
        task_type='comparison' if company == 'comparison' and 'company_rag' in retrieval else 'factual_lookup',
        product_overview='company_rag' in retrieval and company == 'summary' and
            '02_product_information' in spec.categories and not spec.targets and spec.collection.action == 'none',
        answer_basis=('conversation' if set(spec.sources) == {'general_chat'} else 'tools'),
        reason=spec.reason, memory_updates=[m.model_dump() for m in spec.memory_updates],
        canonical_task=compact)


def planner_view(plan):
    """Compact recovery/audit view, including legacy plans converted without inference."""
    operations = {s:op for s,op in [('company_rag',plan.company_operation),
        ('customer_documents',plan.document_operation)] if s in plan.tools and op != 'unknown'}
    canonical = getattr(plan, 'canonical_task', {})
    operation = canonical.get('operation', next(iter(operations.values()), 'lookup'))
    if 'public_web_search' in plan.tools:
        operations['public_web_search'] = canonical.get('source_operations',{}).get('public_web_search', operation)
    view = dict(sources=plan.tools, operation=operation,
        source_operations={s:op for s,op in operations.items() if op != operation},
        categories=plan.knowledge_categories, targets=plan.target_terms, filters=plan.case_filters.model_dump(),
        aspects=plan.answer_aspects or plan.answer_goals, wants_images=plan.wants_visuals,
        query=plan.retrieval_query, queries=[q.model_dump() for q in plan.retrieval_requests],
        collection=plan.company_collection.model_dump(), document_selection=plan.document_selection,
        target_document_ids=plan.target_document_ids,
        **({'document_scope':plan.document_scope} if plan.document_scope != 'unknown' else {}),
        image_scope=plan.visual_scope, answer_basis=plan.answer_basis)
    for wire, runtime in [('image_family','gallery_product_family'),('image_match','gallery_scope'),
                          ('image_target','gallery_target'),('document_visual_required','document_visual_required')]:
        item = getattr(plan, runtime)
        if item is not None and item not in ('unknown',''):
            view[wire] = item
    return view


def delivery_view(plan):
    """Root deliverable survives recovery; current plan remains tool execution."""
    root = plan.recovery_directive.get('original_delivery_task')
    return dict(root) if isinstance(root, dict) else planner_view(plan)


def delivery_operation(plan, source):
    task = delivery_view(plan)
    return task.get('source_operations', {}).get(source, task.get('operation', 'lookup'))


def clear_summary_lookup_bindings(rows, requirements, plan):
    """Discard only provisional lookup bindings for a source being summarised.

    No text, provenance, permissions or semantic verdict is changed.
    Scoped lookup/comparison queries on other sources remain intact.
    """
    summary_kinds=set()
    if 'customer_documents' in plan.tools and delivery_operation(plan,'customer_documents')=='summary':
        summary_kinds.add('document')
    if 'company_rag' in plan.tools and delivery_operation(plan,'company_rag')=='summary':
        summary_kinds.add('entity')
    combined=[*requirements, *[r for row in rows for r in row.get('retrieval_requirements',[])]]
    root=delivery_view(plan)
    pure_summary=(len(root.get('sources',[]))==1 and
        delivery_operation(plan,root['sources'][0])=='summary')
    invalid={r['requirement_id'] for r in combined
        if (pure_summary or r.get('target_kind') in summary_kinds) and 'requirement_id' in r}
    if not invalid:
        return rows, requirements
    return [{**row,
        'requirement_ids':[key for key in row.get('requirement_ids',[]) if key not in invalid],
        'retrieval_requirements':[r for r in row.get('retrieval_requirements',[]) if r.get('requirement_id') not in invalid]}
        for row in rows], [r for r in requirements if r.get('requirement_id') not in invalid]
