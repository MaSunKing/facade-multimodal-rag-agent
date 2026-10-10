"""Backend-owned retrieval requirements. Candidate coverage is not support."""
import hashlib


def source_requests(plan, source):
    rows = plan.get('retrieval_requests', []) if isinstance(plan, dict) else getattr(plan, 'retrieval_requests', [])
    return [r.model_dump() if hasattr(r, 'model_dump') else r for r in rows
            if (r.source if hasattr(r, 'source') else r.get('source')) == source]


def build_requirements(targets, aspects, *, operation='lookup', requests=()):
    """Prefer scoped queries; unnamed lookup/comparison aspects span targets.

    targets contain trusted IDs and names. Models never allocate IDs or ACLs.
    Whole-document summaries use their existing structural coverage instead.
    """
    if operation == 'summary':
        return []
    from backend.sales.relation_scope import normalized
    output = []
    if requests:
        for request in requests:
            descriptions = request.get('targets', [])
            if not descriptions:
                # Bind independent queries to backend-resolved catalog/session
                # identities. No model-authored target IDs or synonym guesses.
                q=normalized(request['query'])
                descriptions=[t['name'] for t in targets if any(
                    normalized(name) and normalized(name) in q
                    for name in [t.get('name',''), *t.get('aliases',[])])]
            bound = [t for t in targets if any(normalized(name) in
                     {normalized(t.get('name', '')), normalized(str(t['id'])),
                      *[normalized(a) for a in t.get('aliases', [])]} for name in descriptions)]
            # Unresolved explicit scope is visible and never widened silently.
            for target in bound or ([{'id':None,'kind':'unresolved','name':' / '.join(descriptions)}]
                                   if descriptions else [{'id':None,'kind':'request','name':''}]):
                identity = (target['kind'], target['id'], target['name'], request['query'])
                output.append({'requirement_id':'R'+hashlib.sha256(repr(identity).encode()).hexdigest()[:16],
                    'target_kind':target['kind'],'target_id':target['id'],'target_name':target.get('name',''),
                    'target_aliases':list(target.get('aliases', [])),
                    'aspect':request['query']})
        return list({r['requirement_id']:r for r in output}.values())
    for aspect in dict.fromkeys(str(a).strip() for a in aspects if str(a).strip()):
        named = [t for t in targets if t.get('name') and t['name'] in aspect]
        bound = named or (targets if operation in {'lookup', 'comparison'} else [])
        for target in bound or [{'id': None, 'kind': 'request', 'name': ''}]:
            identity = (target['kind'], target['id'], aspect)
            output.append({'requirement_id': 'R'+hashlib.sha256(repr(identity).encode()).hexdigest()[:16],
                'target_kind': target['kind'], 'target_id': target['id'],
                'target_name': target.get('name', ''),
                'target_aliases':list(target.get('aliases', [])), 'aspect': aspect})
    return output


def rank_requirement_candidates(rows, requirement, *, scope=None, category_ids=None,
                                tokenize, score, limit=3):
    """Recall facets within relations plus a bounded category complement.

    Entity-name overlap alone must not stop broader recall. This is lexical
    candidate ranking, NOT support verification; the existing joint semantic
    reranker still evaluates the merged pool. No per-cell generation/model call.
    """
    query = ' '.join(v for v in (requirement.get('target_name'), requirement['aspect']) if v)
    query_terms = tokenize(query)
    identity_terms = set(tokenize(' '.join([requirement.get('target_name', ''),
                                          *requirement.get('target_aliases', [])])))
    facet_terms = [t for t in tokenize(requirement['aspect']) if t not in identity_terms]
    permitted = None if scope is None and not category_ids else set(scope or ()) | set(category_ids or ())
    ranked = []
    for row in rows:
        if row.get('kind') != 'text' or (permitted is not None and row['id'] not in permitted):
            continue
        full_score = score(query_terms, row)
        facet_score = score(facet_terms, row) if facet_terms else full_score
        if full_score <= 0 and facet_score <= 0:
            continue
        ranked.append(((facet_score, full_score, int(scope is not None and row['id'] in scope)), row))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [(values[1], row) for values, row in ranked[:limit]]


def annotate(rows, requirement):
    return [{**r, 'requirement_ids': list(dict.fromkeys([*r.get('requirement_ids', []), requirement['requirement_id']])),
             'retrieval_requirements': list({v['requirement_id']:v for v in
                 [*r.get('retrieval_requirements', []), requirement]}.values())} for r in rows]


def merge_requirements(destination, source):
    destination['requirement_ids'] = list(dict.fromkeys([
        *destination.get('requirement_ids', []), *source.get('requirement_ids', [])]))
    destination['retrieval_requirements'] = list({v['requirement_id']:v for v in
        [*destination.get('retrieval_requirements', []), *source.get('retrieval_requirements', [])]}.values())


def coverage(requirements, rows):
    represented = {v for r in rows for v in r.get('requirement_ids', [])
                   if r.get('facts_eligible') is not False and r.get('evidence_scope') not in {'document_index','display_labels'}}
    return {'requirements': [{**r, 'candidate_state': 'represented' if r['requirement_id'] in represented else 'missing',
             'evidence_ids': [v.get('evidence_id') or v.get('source_id') or v.get('chunk_id') for v in rows
                              if r['requirement_id'] in v.get('requirement_ids', [])]} for r in requirements],
            'semantic_support_verified': False}


def representatives(rows):
    """Greedy set cover using existing ranking; no additional model judge."""
    pending = {key for row in rows for key in row.get('requirement_ids', [])}
    selected = []
    while pending:
        candidates = [r for r in rows if r not in selected and r.get('facts_eligible') is not False
                      and r.get('evidence_scope') not in {'document_index','display_labels'}]
        if not candidates:
            break
        winner = max(candidates, key=lambda r: len(pending & set(r.get('requirement_ids', []))))
        hits = pending & set(winner.get('requirement_ids', []))
        if not hits:
            break
        selected.append(winner)
        pending -= hits
    return selected
