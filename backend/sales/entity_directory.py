"""Resolve public directory objects independently of task planning/evidence."""
from backend.sales.content_taxonomy import canonical_domains
from backend.request_budget import RequestBudgetExceeded


def normalize(value):
    from backend.sales.relation_scope import normalized
    return normalized(value)


def rank_directory(query, descriptions):
    from backend.request_budget import check_budget
    from backend.sales.dense_retrieval import RETRIEVAL_INFERENCE_LOCK, get_reranker_model
    check_budget()
    with RETRIEVAL_INFERENCE_LOCK:
        result=get_reranker_model().score(query, descriptions)
    check_budget()
    return result


def resolve_entities(products, cases, query, *, categories=(), targets=(), filters=None,
                     operation='unknown', scorer=None):
    from backend.sales.relation_scope import entity_names
    domains=canonical_domains(categories)
    from backend.sales.content_taxonomy import LABELS, TOPICS, DIRECTORY_CLASS_NAMES
    # Registered navigation classes are not entity names. This is a schema
    # boundary, not a semantic guess about arbitrary user descriptions.
    class_names={normalize(x) for x in [*LABELS, *LABELS.values(), *DIRECTORY_CLASS_NAMES,
        *[topic for values in TOPICS.values() for topic in values]]}
    catalog_names={normalize(row.get('name') or row.get('project_name')) for row in [*products,*cases]}
    targets=[target for target in targets if normalize(target) not in class_names
             or normalize(target) in catalog_names]
    filters=filters or {}
    derived_filters={}
    if not filters.get('locations') and ('05_project_case' in domains or not domains):
        question=normalize(query)
        regions=set()
        for row in cases:
            for field in ('province','city'):
                full=normalize(row.get(field))
                short=full.removesuffix('省').removesuffix('市').removesuffix('自治区')
                for name in (full,short):
                    if len(name)>=2 and name in question:
                        # Negated metadata values are not positive SQL filters.
                        pos=question.find(name)
                        if not any(question[:pos].endswith(prefix) for prefix in ('不含','不包括','排除','除了','非')):
                            regions.add(short)
        if regions:
            filters={**filters,'locations':sorted(regions)}
            derived_filters={'locations':sorted(regions)}
    selected_products=[];selected_cases=[];matches=[]
    # Prefer a complete catalog name over its broader color alias within each
    # explicit target. Separate targets retain their own identities for compare.
    for target in targets or [query]:
        q=normalize(target)
        for rows,key,is_case in ((products,'product_id',False),(cases,'case_id',True)):
            exact=[r for r in rows if normalize(r['project_name'] if is_case else r['name']) in q]
            found=exact or [r for r in rows if any(n in q for n in entity_names(r,is_case))]
            for row in found:
                bucket=selected_cases if is_case else selected_products
                if row not in bucket:bucket.append(row)
                matches.append({'entity_id':row[key],'basis':'exact_name' if exact else 'registered_name_or_catalog_variant'})
    active={k:v for k,v in filters.items() if v}
    # Conditions describe the current target scope. A product/method filter
    # must not turn every associated project into an answer target for a
    # non-project task. Explicitly named projects still support cross-category
    # questions such as their construction/node details.
    project_filter_scope = bool(selected_cases or '05_project_case' in domains
        or not domains or active.get('locations') or active.get('project_types'))
    if active and project_filter_scope:
        # AND between filter fields, OR between alternatives in each field.
        fields={'locations':('province','city'),'project_types':('project_category',),
                'installation_methods':('installation_method',),'products':('product_name',)}
        pool=selected_cases or cases
        selected_cases=[r for r in pool if all(any(normalize(v) in normalize(' '.join(str(r.get(f) or '') for f in fields[k]))
                      for v in values) for k,values in active.items() if k in fields)]
        matches.extend({'entity_id':r['case_id'],'basis':'structured_filter'} for r in selected_cases)
    elif operation=='summary' and not targets and not selected_products and not selected_cases:
        if '02_product_information' in domains:selected_products=list(products)
        if '05_project_case' in domains:selected_cases=list(cases)
        matches.extend({'entity_id':r.get('product_id') or r.get('case_id'),'basis':'category_collection'}
                       for r in [*selected_products,*selected_cases])
    fuzzy=False;error=None
    # A category/topic query is not an unnamed product or project. Only a
    # declared referent needs semantic entity disambiguation; otherwise keep
    # category retrieval available instead of narrowing to arbitrary Top-3.
    if not selected_products and not selected_cases and not active and targets:
        pool=[]
        if not domains or '02_product_information' in domains:pool.extend(('product',r) for r in products)
        if not domains or '05_project_case' in domains:pool.extend(('project',r) for r in cases)
        if pool:
            try:
                descriptions = [
                    ' '.join(str(r.get(field) or '') for field in ('project_name', 'province', 'city'))
                    if kind == 'project' else
                    str(r.get('name') or '') + ' ' + str(r.get('family') or r.get('city') or '')
                    for kind, r in pool
                ]
                scores=(scorer or rank_directory)(query, descriptions)
                if len(scores)!=len(pool):raise ValueError('directory_score_count_invalid')
                ranked=sorted(zip(scores,pool),key=lambda x:float(x[0]),reverse=True)
                # A summary needs collection membership, not a lookup sample.
                # An unresolved referent remains ambiguous: scores may order
                # the available directory but cannot silently define its scope.
                if operation != 'summary':
                    ranked=ranked[:3]
                for score,(kind,row) in ranked:
                    (selected_products if kind=='product' else selected_cases).append(row)
                    matches.append({'entity_id':row.get('product_id') or row['case_id'],'basis':'semantic_candidate',
                                    'score':float(score)})
                fuzzy=True  # No uncalibrated Top-1 is declared a unique identity.
            except RequestBudgetExceeded:
                raise
            except Exception as exc:
                error=type(exc).__name__
    count=len(selected_products)+len(selected_cases)
    status=('ambiguous' if fuzzy else 'collection' if active or operation in {'summary','comparison'} and count>1
            else 'ambiguous' if count>1 else 'matched' if count else 'unresolved')
    return selected_products,selected_cases,{'status':status,'matches':matches,'categories':domains,
        'filters':active,'reranker_used':fuzzy,'reranker_error':error,
        'query_derived_filters':derived_filters,
        'selection_scope':('category_candidates_unresolved_target' if fuzzy and operation=='summary'
                           else 'resolved_or_lookup_candidates'),
        'boundary':'Directory scores locate candidates, not semantic evidence or confidence.'}


def directory_context(retrieval, evidence):
    """Recompute representative coverage from the actual current Context rows."""
    scope=(retrieval.get('meta') or {}).get('relation_scope') or {}
    coverage=scope.get('entity_coverage') or {}
    targets=coverage.get('target_ids') or []
    represented={eid for row in evidence for eid in row.get('entity_coverage_ids',[])}
    listed={eid for row in evidence if row.get('directory_collection') for eid in row.get('directory_member_ids',[])}
    missing=set(targets)-represented
    hierarchical=[r for r in evidence if r.get('directory_level')=='family_with_requested_options']
    if hierarchical:
        families=list(dict.fromkeys(m['attributes']['family'] for r in hierarchical
                                    for m in r.get('directory_members',[])))
        return {'operation':coverage.get('operation','unknown'),'resolution_status':'hierarchical_candidates',
                'target_names':families,'resolved_target_names':families,
                'target_count':len(families),'represented_count':len(families),
                'missing_target_names':[], 'scope':'product_families_and_requested_options_not_complete_sku_review',
                'leaf_records_represented_count':len(represented & set(targets)),
                'semantic_selection':[r.get('directory_semantic_selection',{}) for r in hierarchical]}
    return {'operation':coverage.get('operation','unknown'),
            'resolution_status':(scope.get('directory_resolution') or {}).get('status','unresolved'),
            'family_filter':(scope.get('directory_resolution') or {}).get('family_filter', []),
            'target_names':[t['name'] for t in scope.get('entity_targets', []) if t.get('name') and t['id'] not in listed],
            'resolved_target_names':[t['name'] for t in scope.get('entity_targets', [])
                                     if t.get('name') and t['id'] in represented][:40],
            'target_count':len(targets),'represented_count':len(represented & set(targets)),
            'missing_target_names':[t['name'] for t in scope.get('entity_targets', []) if t.get('name') and t['id'] in missing],
            'scope':'selected_representatives_not_complete_document_review',
            'expansion_reason':scope.get('expansion_reason')}
