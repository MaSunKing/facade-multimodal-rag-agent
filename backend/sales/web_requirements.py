"""Bounded query execution, one shared workflow and semantic audit."""
from backend.sales.requirements import build_requirements, annotate, merge_requirements, coverage


def retrieve_web_requirements(requests, search):
    scopes = [{'id':None,'kind':'request','name':t}
              for t in dict.fromkeys(t for r in requests for t in r.get('targets', []))]
    requirements = build_requirements(scopes, [], requests=requests)
    sources, seen, executions, query_cache = [], {}, [], {}
    stopped = False
    for requirement in requirements:
        if stopped:
            executions.append({'requirement_id':requirement['requirement_id'],'status':'not_executed_budget'})
            continue
        query = requirement['aspect']
        reused = query in query_cache
        if not reused:
            query_cache[query] = search(query)
        rows, meta = query_cache[query]
        if reused:
            meta = {**meta,'api_calls_for_query':0,'reused_query_result':True}
        executions.append({'requirement_id':requirement['requirement_id'], **meta})
        stopped = meta.get('status') == 'quota_exhausted'
        for row in annotate(rows, requirement):
            identity = (row.get('url'), row.get('excerpt') or row.get('search_excerpt'))
            if identity in seen:
                merge_requirements(seen[identity], row)
                continue
            row = dict(row, provider_source_id=row.get('source_id'), source_id='W'+str(len(sources)+1))
            seen[identity] = row
            sources.append(row)
    snapshot = coverage(requirements, sources)
    snapshot['executed_ids'] = [r['requirement_id'] for r in executions if r.get('status') != 'not_executed_budget']
    snapshot['unexecuted_due_to_call_budget'] = [r['requirement_id'] for r in executions if r.get('status') == 'not_executed_budget']
    failures = [r for r in executions if r.get('status') not in {'ok','not_executed_budget'}]
    failure = next((r for r in failures if r.get('status') == 'quota_exhausted'), failures[0] if failures else {})
    return sources, {'status':'ok' if sources else failure.get('status', 'no_usable_results'),
        'error_code':failure.get('error_code'),
        'partial_tool_failure':bool(sources and failures),
        'query_executions':executions,'requirement_coverage':snapshot,
        'api_calls_for_query':sum(r.get('api_calls_for_query',0) for r in executions),
        'quota':next((r['quota'] for r in reversed(executions) if r.get('quota')),{}),
        'cache_hit':bool(executions) and all(r.get('cache_hit',False) for r in executions),
        'cache_bypassed':any(r.get('cache_bypassed',False) for r in executions),
        'tool_transport':'mcp','mcp_tool':'network_search'}
