"""Typed execution witnesses for task review, never evidence prose or pixels.

This module does not select sources, assert factual support or authorize retries.
It projects records already produced by the active task's tools.
"""
from __future__ import annotations

from backend.sales.recovery_policy import source_diagnostics


def _planned_sources(plan):
    if isinstance(plan, dict):
        return list(plan.get('tools') or plan.get('sources') or [])
    return list(getattr(plan, 'tools', []) or [])


def _coverage_bundle(source):
    if not isinstance(source, dict):
        return {}
    for container in (source.get('input_snapshot'), source.get('meta'), source):
        if isinstance(container, dict) and isinstance(container.get('requirement_coverage'), dict):
            return container['requirement_coverage']
    return {}


def execution_witnesses(plan, sources):
    """Missing candidates are not unexecuted work, and errors are not absence.

    The only scope gaps emitted here concern requirements explicitly recorded as
    skipped by the tool. No word matching, relevance thresholds or file routing.
    Permission and quota failures are reported but never recovery authorization.
    """
    statuses = source_diagnostics(sources)
    planned = _planned_sources(plan)
    states = []
    gaps = []
    for name in dict.fromkeys([*planned, *sources]):
        raw = statuses.get(name, {})
        state = raw.get('state') or ('executed' if name in sources else 'not_executed')
        item = {'source': name, 'executed': name in sources, 'state': state}
        if isinstance(raw.get('error_code'), str):
            item['error_code'] = raw['error_code']
        bundle = _coverage_bundle(sources.get(name))
        requirements = {r['requirement_id']: r for r in bundle.get('requirements', [])
                        if isinstance(r, dict) and isinstance(r.get('requirement_id'), str)}
        skipped = {
            'unexecuted_due_to_scope': 'scope_unresolved',
            'unexecuted_due_to_call_budget': 'call_budget_exhausted',
        }
        for field, reason in skipped.items():
            for identity in dict.fromkeys(bundle.get(field, [])):
                requirement = requirements.get(identity)
                if not requirement:
                    continue
                gap = {'source': name, 'requirement_id': identity, 'state': reason,
                       'semantic_support_verified': False}
                for key in ('target_kind', 'target_id', 'target_name', 'aspect'):
                    value = requirement.get(key)
                    if isinstance(value, str):
                        gap[key] = value
                gaps.append(gap)
        item['requirement_count'] = len(requirements)
        item['skipped_requirement_count'] = sum(g['source'] == name for g in gaps)
        states.append(item)
    return {'source_execution': states, 'execution_gaps': gaps,
            'boundary': 'Execution metadata only; candidate presence is not factual support.'}


def display_associations(asset):
    """Only existing structured ownership links; do not infer from a caption."""
    allowed = ('case_id', 'project_name', 'product_id', 'product_name',
               'document_id', 'document_name', 'source_page')
    return {key: asset[key] for key in allowed
            if key in asset and isinstance(asset[key], (str, int)) and not isinstance(asset[key], bool)}


def review_citations(citations):
    """One draft citation per evidence identity, not one row per provenance.

    Full materialized sources remain in the response and trace. Lightweight
    coverage review needs identities and source counts, not repeated locations.
    """
    from collections import Counter
    counts=Counter(c.get('evidence_id') for c in citations
                   if isinstance(c,dict) and isinstance(c.get('evidence_id'),str))
    return [{'evidence_id':identity,'source_reference_count':count} for identity,count in counts.items()]


def review_task_scope(payload):
    """Separate navigation scope from the authoritative user's answer scope.

    Never infer that a directory filter, candidate count or planned aspect is
    a confirmed requirement. This is a review projection only: retrieval,
    generation Context, IDs and server-side diagnostic records are unchanged.
    """
    from copy import deepcopy
    result = deepcopy(payload)
    # The reviewer reads user words, not the Planner's decomposition of them.
    # Keep only tool strategy; actual conditions/objects remain in execution
    # records, not as a competing declaration of what the user must receive.
    tool_fields = {'sources', 'tools', 'operation', 'source_operations',
                   'categories', 'wants_images', 'wants_visuals'}
    for name in ('planned_task', 'execution_task', 'previous_plan'):
        planned = result.get(name)
        if isinstance(planned, dict):
            result[name] = {k: v for k, v in planned.items() if k in tool_fields}
    result.pop('planner_suggestions', None)
    result.pop('proposed_requirements', None)
    # This grid is derived from planning facets and candidate bindings; it is
    # neither user authority nor verified completeness. Concrete skipped work
    # is still conveyed separately through execution_gaps/source_execution.
    result.pop('requirement_coverage', None)
    guide = result.get('field_guide')
    if isinstance(guide, dict):
        guide.pop('requirement_coverage', None)
        guide['planned_task'] = '工具计划：来源、任务类型、企业类别和图片需求；不定义用户必答对象或维度'
    scope = result.get('actual_execution_scope')
    if isinstance(scope, dict):
        # A broad candidate list encourages the model to invent missing answer
        # members even when its count is labelled non-authoritative. Transmit
        # tool scope and explicitly located referents, not the candidate pool.
        matches = scope.get('matches', [])
        targets = scope.get('entity_targets', [])
        matches = matches if isinstance(matches, list) else []
        targets = targets if isinstance(targets, list) else []
        explicit_ids = {m.get('entity_id') for m in matches if isinstance(m, dict)
                        and m.get('basis') in {'exact_name', 'registered_name_or_catalog_variant'}}
        referents = [{k: t[k] for k in ('id', 'name', 'kind') if k in t}
                     for t in targets if isinstance(t, dict) and t.get('id') in explicit_ids]
        methods = list(dict.fromkeys(m.get('basis', 'unknown') for m in matches if isinstance(m, dict)))
        if scope.get('candidate_members_and_counts') == 'omitted_unverified_navigation_retained_server_side':
            referents = scope.get('resolved_referents', [])
            methods = scope.get('matching_methods', [])
        result['actual_execution_scope'] = {k: scope[k] for k in
            ('status', 'categories', 'filters', 'reranker_used', 'reranker_error') if k in scope}
        result['actual_execution_scope'].update(
            resolved_referents=referents,
            matching_methods=methods,
            population_role='retrieval_candidates_not_required_answer_members',
            answer_scope_verified=False,
            candidate_members_and_counts='omitted_unverified_navigation_retained_server_side')
    for name in ('directory_coverage', 'company_directory_scope'):
        result.pop(name, None)
    result['task_scope_contract'] = {
        'authority': 'effective_user_words_and_task_update',
        'planner_decomposition': 'omitted_tool_strategy_only',
        'candidate_population': 'may_be_broader_than_requested_answer_scope',
        'answer_cardinality': 'not_established_by_navigation_or_candidate_counts',
        'missing_member_decision': 'requires_confirmed_membership_in_user_scope_not_count_difference',
    }
    return result


def compact_review_metadata(payload):
    """Factor repeated navigation fields, never the question, draft or labels."""
    from copy import deepcopy
    from collections import Counter
    result=deepcopy(payload)
    # Field definitions already exist in the shared system instructions.
    # This repeated glossary is not user scope, evidence or an image label.
    if 'field_guide' in result:
        result['field_guide']='见系统说明：覆盖、工具、标签；非事实认证。'
    if 'task_scope_contract' in result:
        result['task_scope_contract']={'user_words_authoritative':True,
            'planner_and_candidates_define_answer_scope':False,
            'missing_members_require_confirmed_user_scope':True}
    scope=result.get('actual_execution_scope',{})
    targets=scope.get('entity_targets',[])
    if isinstance(targets,list) and targets:
        scope['entity_targets']={'columns':['name','kind'],
            'rows':[[t.get(k) for k in ('name','kind')] for t in targets],
            'count':len(targets),'aliases_scope':'omitted_navigation_aliases_not_new_targets'}
    matches=scope.get('matches',[])
    if isinstance(matches,list) and matches:
        scope['matches']={'basis_counts':dict(Counter(m.get('basis','unknown') for m in matches)),
            'count':len(matches),'identities_scope':'represented_in_entity_targets_server_trace_retains_match_records'}
    coverage=result.get('requirement_coverage',{})
    requirements=coverage.get('requirements',[])
    if isinstance(requirements,list) and requirements:
        names=list(dict.fromkeys(r.get('target_name') for r in requirements))
        aspects=list(dict.fromkeys(r.get('aspect') for r in requirements))
        coverage['requirements']={'target_names':names,'aspects':aspects,
            'columns':['target_index','aspect_index','candidate_state','candidate_count'],
            'rows':[[names.index(r.get('target_name')),aspects.index(r.get('aspect')),
                     r.get('candidate_state'),r.get('candidate_count')] for r in requirements],
            'count':len(requirements),
            'identity_scope':'requirement_ids_retained_server_side_execution_gaps_unchanged'}
    result['metadata_packing']='factored_navigation_preserving_all_target_names_and_candidate_states'
    return result


def review_image_labels(context, display_assets):
    """Use only source-bound labels already available to the draft.

    Customer descriptions preserve uncertainties. Never fall back to bytes,
    paths, raw OCR pages or a new visual call, and never create missing labels.
    """
    import json
    from backend.sales.visual_labels import company_label_view
    output = []
    known = set()
    for row in context:
        if row.get('evidence_scope') != 'visual_description':
            continue
        try:
            label = json.loads(row.get('text') or '{}')
        except (ValueError, TypeError):
            continue
        if not isinstance(label, dict):
            continue
        basis = row.get('description_basis')
        if basis == 'source_metadata_labels':
            label = company_label_view(label)
        elif basis == 'pixel_tool_observation':
            fields = ('labels', 'description', 'visible_text', 'relations', 'uncertainties')
            label = {k: label[k] for k in fields if k in label}
        else:
            continue
        identity = row.get('visual_id') or row.get('asset_id') or row.get('evidence_id')
        known.add(identity)
        output.append({'asset_id': identity, 'evidence_id': row.get('evidence_id'),
                       'document_id': row.get('document_id'), 'source_type':row.get('source_type'), 'label_basis': basis,
                       'labels': label, 'in_draft_context': True})
    for asset in display_assets:
        identity = asset.get('asset_id') or asset.get('visual_id')
        labels = asset.get('image_labels')
        if identity in known or not isinstance(labels, dict):
            continue
        # These are backend-selected display assets, not the entire gallery.
        # Uninterpreted display cards expose title/association, not the full
        # asset catalogue. Do not introduce unseen metadata as audit facts.
        # Observations actually present in Context keep their complete labels
        # above; uncertain attributes remain constraints even on display cards.
        display_labels={'title':asset.get('customer_title') or asset.get('title') or labels.get('title'),
                        'scope':'display_title_and_explicit_associations_only'}
        for key in ('uncertainties','uncertainty','issues'):
            if labels.get(key): display_labels[key]=labels[key]
        output.append({'asset_id': identity, 'labels': display_labels,
                       'label_basis': 'source_metadata_labels',
                       'display_associations': display_associations(asset),
                       'in_draft_context': False, 'display_delivered': True})
    return output
