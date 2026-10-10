"""One audit wire representation, shared by preflight and invocation.

Only navigation/diagnostic duplication is compacted. Evidence, original task,
negative constraints, recovery resources and visual observations stay intact.
No pixels or another model are used by this module.
"""
import copy
import json


def factor_label_records(labels):
    """Lossless wire factoring: repeated field names only, no value clipping."""
    groups = {}
    for record in labels:
        if not isinstance(record, dict):
            return labels
        nested = record.get('labels')
        keys = tuple(k for k in record if k != 'labels')
        label_keys = tuple(nested) if isinstance(nested, dict) else None
        groups.setdefault((keys,label_keys), []).append(record)
    result = []
    for (keys,label_keys),records in groups.items():
        if label_keys is None:
            result.append({'columns':list(keys)+['labels'],
                'rows':[[r[k] for k in keys]+[r.get('labels')] for r in records]})
        else:
            result.append({'identity_columns':list(keys),'label_columns':list(label_keys),
                'rows':[[[r[k] for k in keys],[r['labels'][k] for k in label_keys]] for r in records]})
    return {'encoding':'lossless_grouped_columns','groups':result}


def audit_wire_text(payload):
    """Drop JSON formatting whitespace only after the overflow decision."""
    if payload.get('review_policy') == 'task_coverage_only':
        # A natural-language draft is not JSON. Escaping it can inflate a
        # 3072-token completion unpredictably after generation. Keep it verbatim
        # in a data-only section; the same wire is counted and invoked.
        value = copy.deepcopy(payload)
        draft = value.get('final_draft', {})
        reply = draft.pop('customer_reply', '')
        labels=value.pop('image_labels', [])
        wire=(json.dumps(value, ensure_ascii=False, separators=(',', ':')) +
              '\n<final_draft_data>\n' + str(reply or '') + '\n</final_draft_data>')
        if labels:
            # Source-bound observations immediately follow the claim to check;
            # do not bury uncertainty in navigation metadata or treat the draft
            # as the last/most authoritative description of the image.
            wire+='\n<image_labels_data>\n'+json.dumps(factor_label_records(labels),ensure_ascii=False,separators=(',',':'))+'\n</image_labels_data>'
        return wire
    return json.dumps(payload, ensure_ascii=False,
        separators=(',', ':') if payload.get('wire_policy') == 'compact_navigation_not_fact_rewrite' else None)


def audit_wire_payload(payload, *, current_date=None, compact=False):
    from backend.documents.document_tasks import attachment_support_envelope
    from backend.sales.content_taxonomy import audit_capability_map
    result = (dict(payload) if payload.get('review_policy') == 'task_coverage_only' or payload.get('wire_policy') == 'compact_navigation_not_fact_rewrite'
        else attachment_support_envelope(dict(payload)))
    if result.get('review_policy') == 'task_coverage_only':
        from backend.sales.review_execution import review_task_scope
        result = review_task_scope(result)
    if current_date is not None:
        result['current_date_china'] = current_date
    result.setdefault('company_capabilities', audit_capability_map())
    if not compact:
        return result
    result = copy.deepcopy(result)
    # Runtime/coverage diagnostics remain on the server trace, not model input.
    for name in ('planned_task', 'execution_task', 'previous_plan'):
        task = result.get(name)
        if isinstance(task, dict):
            result[name] = {k:v for k,v in task.items() if k not in
                {'query','queries','guard_diagnostics','planner_input_tokens','planner_output_tokens'}}
    planned = result.get('planned_task')
    if isinstance(planned,dict):
        result['planned_task'] = {k:v for k,v in planned.items() if v not in (None, '', [], {})}
    if result.get('execution_task') == result.get('planned_task'):
        result.pop('execution_task', None)
    elif isinstance(result.get('execution_task'),dict) and isinstance(result.get('planned_task'),dict):
        planned = result['planned_task']
        result['execution_task'] = {'inherits':'planned_task', **{
            k:v for k,v in result['execution_task'].items()
            if (k in planned and planned[k] != v) or (k not in planned and v not in (None, '', [], {}))}}
    # Factor the binding grid instead of repeating opaque backend IDs and
    # the same aspect in every target row. All target/aspect/evidence links and
    # missing states remain; the model never outputs Requirement IDs.
    requirements = result.get('retrieval_requirements', [])
    if isinstance(requirements, list) and requirements and all('requirement_id' in r for r in requirements):
        groups = {}
        targets = []
        for r in requirements:
            target = r.get('target_name') or r.get('target_id') or 'current_request'
            if target not in targets:
                targets.append(target)
            groups.setdefault(r.get('aspect',''), []).append([
                targets.index(target),r.get('candidate_state','missing'),r.get('evidence_ids',[])])
        result['retrieval_requirements'] = {
            'targets':targets,'cell_columns':['target_index','candidate_state','evidence_ids'],
            'aspects':[{'aspect':a,'cells':ts} for a,ts in groups.items()]}
        for r in result.get('evidence', []):
            r.pop('requirement_ids',None)  # Explicit links live in the grid above.
    policy = result.get('company_visual_policy')
    if isinstance(policy, dict):
        result['company_visual_policy'] = {k:v for k,v in policy.items() if k in
            {'input_policy','company_only','candidate_count','incomplete_label_asset_ids',
             'disputed_label_asset_ids','pixel_contents_verified'}}
    # Display identity is already in image_delivery and label-backed Evidence.
    # Only customer candidates can be requested for visual recovery.
    result['visual_candidates'] = [c for c in result.get('visual_candidates', [])
        if not str(c.get('asset_id','')).startswith('company:')]
    for row in result.get('evidence', []):
        if row.get('gallery_collection') and row.get('evidence_scope') == 'display_labels':
            citations = row.get('citations', [])
            if len(citations) > 2:
                row['citations'] = citations[:2]
                row['citation_reference_count'] = len(citations)
                row['provenance_scope'] = 'representative_refs_full_image_provenance_retained_server_side'
        sources = row.get('sources', [])
        if (len(sources) > 2 and row.get('catalog_record_type') in
                {'directory_collection','directory_aggregate','product_collection'}):
            row['sources'] = sources[:2]
            row['source_reference_count'] = len(sources)
            row['omitted_source_references'] = len(sources)-2
            row['source_reference_scope'] = 'representative_refs_full_provenance_retained_server_side'
    # Missing members are coverage, not additional evidence. Never claim complete.
    scope = result.get('company_directory_scope')
    if isinstance(scope, dict):
        for key in ('target_names','missing_target_names'):
            names = scope.get(key, [])
            if len(names) > 8:
                scope[key] = names[:8]
                scope[key+'_omitted_count'] = len(names)-8
    result['wire_policy'] = 'compact_navigation_not_fact_rewrite'
    return result


def audit_system_prompt(system, payload):
    from backend.documents.document_tasks import attachment_scope_instruction
    return system + attachment_scope_instruction(payload.get('attachment_task', {}))
