"""Backend-owned gap identities and source-operation deduplication.

Semantics remain Qwen's job. These guards do not predict retrieval success.
"""
import hashlib
import json
from copy import deepcopy
from backend.sales.requirements import source_requests


def gap_state(question, verdict, required_documents=()):
    gaps=[]
    requirements=verdict.get('missing_aspects') or []
    if not requirements and verdict.get('status') in {'unsupported','uncertain'}:
        requirements=[question]  # Original task, never a program-invented semantic gap.
    for requirement in dict.fromkeys(requirements):
        if not isinstance(requirement,str) or not requirement.strip(): continue
        identity=hashlib.sha256((question+'\0'+requirement).encode()).hexdigest()[:16]
        gaps.append({'gap_id':'gap_'+identity,'requirement':requirement,
                     'reason':str(verdict.get('problem') or verdict.get('reason') or ''),
                     'status':'pending'})
    return {'gaps':gaps,'required_document_ids':[d['document_id'] for d in required_documents],
            'policy':'original_task_gap_bound_one_execution_recovery'}


def operation_signature(plan,tool):
    def get(k,default=None): return plan.get(k,default) if isinstance(plan,dict) else getattr(plan,k,default)
    query=(get('document_retrieval_queries',[]) if tool=='customer_documents' else
           get('company_retrieval_query','') if tool=='company_rag' else get('web_retrieval_query',''))
    value={'tool':tool,'query':query or get('retrieval_query',''),
           'retrieval_requests': source_requests(plan,tool),
           'retrieval_queries':get('retrieval_queries',[]),
           'operation':get('document_operation','') if tool=='customer_documents' else get('company_operation','') if tool=='company_rag' else get('task_type',''),
           'scope':get('document_scope','') if tool=='customer_documents' else get('visual_scope','') if get('wants_visuals',False) else '',
           'documents':sorted(get('target_document_ids',[]) or []) if tool=='customer_documents' else [],
           'visual_required':get('document_visual_required') if tool=='customer_documents' else get('wants_visuals',False)}
    if tool=='company_rag':
        for key in ('knowledge_categories','company_operation','company_collection','target_terms','case_filters','case_reference','product_overview','search_targets','gallery_scope','gallery_target','gallery_product_family'):
            item=get(key)
            value[key]=item.model_dump() if hasattr(item,'model_dump') else item
        value['company_collection']=value['company_collection'] or {'action':'none','group_by':[]}
    if tool=='public_web_search': value['web_source_profile']=get('web_source_profile','auto')
    def normalize(item):
        if isinstance(item,str): return ' '.join(item.split()).casefold()
        if isinstance(item,list): return sorted((normalize(v) for v in item),key=lambda v:json.dumps(v,sort_keys=True))
        if isinstance(item,dict): return {k:normalize(v) for k,v in item.items()}
        return item
    return hashlib.sha256(json.dumps(normalize(value),ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def new_operations(old,new):
    tools=new.get('tools',[]) if isinstance(new,dict) else new.tools
    prior=old.get('tools',[]) if isinstance(old,dict) else old.tools
    return [t for t in tools if t not in prior or operation_signature(old,t)!=operation_signature(new,t)]


def recovery_review_context(question, state):
    """Old diagnostics remain in trace, not assertions in a fresh audit.

    A diagnostic may embed round-local references or an incorrect interpretation.
    Reassess the actual task instead of turning that diagnostic into a requirement.
    """
    if not state:
        return {}
    return {'original_requirement': question,
            'gap_ids': [g['gap_id'] for g in state.get('gaps', [])],
            'policy': 'reassess_current_evidence_no_prior_verdict_authority'}


def _merge_rows(old_rows, new_rows, id_field, prefix, web=False):
    def identity(row):
        if web:
            return (row.get('url'), row.get('excerpt') or row.get('search_excerpt'), row.get('date'))
        return (row.get('document_id') or row.get('document_name'),
                row.get('original_chunk_id'), row.get('text'))
    old_by_key = {identity(row): row for row in old_rows}
    used = {r[id_field] for r in old_rows if r.get(id_field)}
    next_id = 1
    rows, seen = [], set()
    for source, round_number in [*((r, 2) for r in new_rows), *((r, 1) for r in old_rows)]:
        key = identity(source)
        if key in seen:
            continue
        seen.add(key)
        row = deepcopy(source)
        prior = old_by_key.get(key)
        if prior and prior.get(id_field):
            row[id_field] = prior[id_field]
            from backend.sales.requirements import merge_requirements
            merge_requirements(row, prior)
            row['retrieval_rounds'] = sorted(set(prior.get('retrieval_rounds', [1]) + [round_number]))
        else:
            while prefix + str(next_id) in used:
                next_id += 1
            row[id_field] = prefix + str(next_id)
            used.add(row[id_field])
            row['retrieval_rounds'] = [round_number]
        # Updated source contents are distinct evidence versions, not silently
        # overwritten facts. Both versions remain candidates for semantic review.
        if web and prior is None:
            versions = [r[id_field] for r in old_rows if r.get('url') == row.get('url') and r.get(id_field)]
            if versions:
                row['previous_version_ids'] = versions
        rows.append(row)
    return rows


def merge_sources(old,new,tool):
    """Union canonical source candidates; final packing/audit still runs again."""
    if not old: return new
    if not new: return old
    if tool=='public_web_search':
        old_rows,old_meta=old; new_rows,meta=new
        rows = _merge_rows(old_rows, new_rows, 'source_id', 'W', web=True)
        from backend.sales.requirements import coverage
        requirements = list({r['requirement_id']:r for m in (old_meta,meta)
            for r in (m.get('requirement_coverage') or {}).get('requirements', [])}.values())
        return rows,{**meta,'merged_previous_sources':True,
                     'requirement_coverage':coverage(requirements,rows),
                     'evidence_reconciliation': _reconciliation_trace(rows, 'source_id')}
    field='evidence' if tool=='customer_documents' else 'text_evidence'
    result=deepcopy(new)
    rows = _merge_rows(old.get(field, []), new.get(field, []), 'evidence_id',
                       'U' if tool=='customer_documents' else 'T')
    result[field]=rows
    for name in ('documents','selected_visuals','visual_assets'):
        if name in old or name in new:
            merged=[]; seen=set()
            for row in [*new.get(name,[]),*old.get(name,[])]:
                key=json.dumps(row,sort_keys=True,ensure_ascii=False,default=str)
                if key not in seen: merged.append(deepcopy(row));seen.add(key)
            result[name]=merged
    result.setdefault('input_snapshot' if tool=='customer_documents' else 'meta',{})['merged_previous_sources']=True
    metadata_field='input_snapshot' if tool=='customer_documents' else 'meta'
    result[metadata_field]['evidence_reconciliation'] = _reconciliation_trace(rows, 'evidence_id')
    from backend.sales.requirements import coverage
    requirements=list({r['requirement_id']:r for source in (old,new)
        for r in (source.get(metadata_field,{}).get('requirement_coverage') or {}).get('requirements', [])}.values())
    result[metadata_field]['requirement_coverage']=coverage(requirements,rows)
    return result


def _reconciliation_trace(rows, id_field):
    return {'policy': 'stable_identity_candidate_union_not_semantic_approval',
            'candidates': [{id_field: r[id_field], 'retrieval_rounds': r.get('retrieval_rounds', []),
                            'previous_version_ids': r.get('previous_version_ids', [])} for r in rows],
            'scope_decision': 'semantic_audit_unless_trusted_scope_proves_mismatch'}
