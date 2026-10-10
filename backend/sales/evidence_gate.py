"""Cheap risk signals, not semantic gold. No product/topic hard routing."""
import re


def terms(text):
    words = set(re.findall(r"[a-z0-9]{2,}", str(text).casefold()))
    for run in re.findall(r"[\u4e00-\u9fff]+", str(text)):
        words.update(run[i:i+2] for i in range(len(run)-1))
    return words - {"什么", "是什", "日是", "介绍", "一下", "请问", "如何", "相关", "这个", "方案", "详细", "哪些"}


def question_excerpt(text, question='', limit=700):
    if len(text)<=limit or not question:
        return text[:limit]
    # Keep the source intact; preview a question-relevant window instead of
    # always dropping a factual paragraph after character 700.
    query_terms=terms(question)
    size=limit-160
    starts=range(0,len(text),max(1,size//2))
    best=max(starts,key=lambda start:len(query_terms & terms(text[start:start+size])))
    if best<160:return text[:limit]
    return text[:140]+'\n[... source excerpt ...]\n'+text[best:best+size]


def evidence_preview(values, limit=6, question='', plan=None, max_characters=6000):
    """Bounded, goal-aware previews; selection scores are not support verdicts.

    Preserve source diversity and complete approved dossiers when requested by
    the semantic plan. Never interpret a preview omission as corpus absence.
    """
    rows = []
    def visit(value, depth=0):
        if depth > 8 or len(rows) >= 60:
            return
        if hasattr(value, 'model_dump'):
            value = value.model_dump()
        if isinstance(value, dict):
            text = value.get('text') or value.get('excerpt') or value.get('content')
            if isinstance(text, str) and text.strip():
                rows.append({'evidence_id': f'E{len(rows)+1}', 'text': question_excerpt(text,question),
                             'preview_truncated':len(text)>700,
                             'document_name': str(value.get('document_name') or value.get('title') or '')[:120],
                             'source_url':value.get('source_url') or value.get('url'),
                             'evidence_level':value.get('evidence_level'),
                             'sales_playbook_use':value.get('sales_playbook_use'),
                             'citations':value.get('citations') or [],
                             'page_fetch_status':value.get('page_fetch_status'),
                             'source_date':value.get('date')})
                rows[-1]['entity_coverage_ids']=value.get('entity_coverage_ids',[])
            else:
                for key, child in value.items():
                    if key not in {'meta', 'audit', 'input_snapshot', 'question', 'query'}:
                        visit(child, depth+1)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, depth+1)
    buckets = []
    for value in values:
        rows = []
        visit(value)
        if rows:
            buckets.append(rows)
    goals = list(dict.fromkeys([*(getattr(plan, 'answer_aspects', []) or []),
                                *(getattr(plan, 'answer_goals', []) or [])]))
    overview = bool(getattr(plan, 'product_overview', False))
    limit = min(12, max(limit, len(goals) * 2))
    flat = [row for bucket in buckets for row in bucket]
    selected = []
    def add(row):
        if row not in selected:
            selected.append(row)
    for goal in goals:
        if flat:
            ranked = sorted(flat, key=lambda row:len(terms(goal) & terms(row['text'])), reverse=True)
            if terms(goal) & terms(ranked[0]['text']):
                add(ranked[0])
    if overview:
        for row in flat:
            if row.get('sales_playbook_use') == 'approved_product_master_profile':
                add(row)
        limit = min(12, max(limit, len(selected)))
    # Round robin still reserves sources rather than filling from one corpus.
    combined = selected[:limit]
    while len(combined)<limit and any(buckets):
        for bucket in buckets:
            if bucket and len(combined)<limit:
                row = bucket.pop(0)
                if row not in combined:
                    combined.append(row)
    output, used = [], 0
    for row in combined:
        if used + len(row['text']) > max_characters:
            continue  # Do not split a fact just to fit an extra row.
        output.append({**row, 'evidence_id':f'E{len(output)+1}'})
        used += len(row['text'])
    return output


def compact_audit_evidence(rows):
    """Keep support text and provenance, not repeated retrieval/parse metadata."""
    result = []
    for row in rows:
        item = {key: row[key] for key in ('evidence_id', 'text', 'preview_truncated',
                'document_name', 'document_id', 'derived_summary', 'coverage', 'facts_eligible', 'claim_scope', 'evidence_scope',
                'visual_description', 'description_basis', 'visual_id',
                'requirement_ids',
                'directory_category','directory_item_count','fact_support_scope','catalog_record_type',
                'collection_action','collection_group_by','collection_total_count','collection_group_count','collection_missing_fields',
                'source_url', 'evidence_level', 'page_fetch_status', 'source_date',
                'publisher_group', 'date_scope', 'mentioned_dates', 'page_fetched_at_china')
                if row.get(key) not in (None, '', [])}
        if row.get('coverage'):
            from backend.documents.summaries import model_coverage
            item['coverage'] = model_coverage(row['coverage'])
        provenance = []
        for citation in row.get('citations') or []:
            if hasattr(citation, 'model_dump'):
                citation = citation.model_dump()
            if isinstance(citation, dict):
                source = {key: citation[key] for key in ('document_name', 'source_url', 'page',
                    'page_number', 'sheet', 'evidence_level', 'date')
                    if citation.get(key) not in (None, '', [])}
                if source and source not in provenance:
                    provenance.append(source)
        if provenance:
            item['sources'] = provenance
        from backend.sales.evidence_packing import nonempty_metadata
        result.append(nonempty_metadata(item))
    return result


def relevance_risk(question, plan, rows):
    query_terms = terms(question)
    source_terms = terms(' '.join(row['text'] for row in rows))
    overlap = len(query_terms & source_terms) / max(1, len(query_terms))
    targets = list(getattr(plan, 'target_terms', []) or [])
    absent = [target for target in targets if terms(target) and not (terms(target) & source_terms)]
    suspicious = not rows or (bool(query_terms) and overlap < .15) or (bool(targets) and len(absent) == len(targets))
    return {'suspicious': suspicious, 'lexical_coverage': round(overlap, 3),
            'missing_target_terms': absent, 'evidence_preview_count': len(rows),
            'signal_is_semantic_gold': False}


def budget_audit_payload(payload, token_count, max_tokens=5800):
    """Exact-token cap via caller's tokenizer, preserving whole evidence rows.

    Requirements and original question are never prefix-truncated. If even
    their irreducible input exceeds the limit, the caller fails explicitly.
    """
    result = dict(payload)
    removed = []
    for field in ('evidence','existing_source_previews','visual_candidates'):
        if field in result:
            result[field] = list(result[field])
    while token_count(result) > max_tokens:
        if result.get('prepared_support_contract'):
            # Never silently audit only a subset then generate from the rest.
            raise ValueError('audit_input_budget_exceeded')
        field = next((name for name in ('evidence','existing_source_previews','visual_candidates')
                      if result.get(name)), None)
        if not field:
            raise ValueError('audit_input_budget_exceeded')
        item = result[field].pop()
        removed.append(item.get('evidence_id') or item.get('asset_id') or field)
        result['preview_token_budget'] = {'omitted_preview_ids':removed,
                                        'omissions_are_not_corpus_absence':True}
    return result, {'input_tokens':token_count(result),'max_input_tokens':max_tokens,
                    'omitted_preview_ids':removed}


def validate_evaluation(raw, rows):
    if not isinstance(raw, dict):
        raise ValueError('invalid_evaluation_object')
    status = raw.get('status')
    if status not in {'supported', 'partial', 'unsupported', 'uncertain'}:
        raise ValueError('invalid_evaluation_status')
    ids = {row['evidence_id'] for row in rows}
    supports = raw.get('supported_evidence_ids', [])
    if not isinstance(supports, list) or any(not isinstance(value, str) or value not in ids for value in supports):
        raise ValueError('invalid_evaluation_evidence_id')
    missing = raw.get('missing_aspects', [])
    if not isinstance(missing, list) or any(not isinstance(value, str) for value in missing):
        raise ValueError('invalid_evaluation_missing_aspects')
    if 'web_can_resolve' in raw and not isinstance(raw['web_can_resolve'], bool):
        raise ValueError('invalid_evaluation_web_flag')
    # Candidate asset IDs are pointers, not support. Label-backed gallery rows
    # have real evidence IDs; pixel observations are captured by the tool first.
    if status in {'supported', 'partial'} and not supports:
        raise ValueError('support_requires_evidence')
    conflicts = raw.get('conflict_groups', [])
    if not isinstance(conflicts, list) or any(
        not isinstance(group, list) or len(group) < 2
        or any(not isinstance(i, str) or i not in supports for i in group)
        or len(set(group)) != len(group) for group in conflicts):
        raise ValueError('invalid_evaluation_conflict_groups')
    return {'status': status, 'supported_evidence_ids': supports,
            'conflict_groups': conflicts,
            'source_constraint': raw.get('source_constraint') if raw.get('source_constraint') in
                {'evidence_only', 'general_knowledge_allowed', 'uncertain'} else 'uncertain',
            'missing_aspects': [x[:400] for x in missing[:8]],
            'reason': str(raw.get('reason') or '')[:400]}
