"""Bounded admin-only step diagnostics, including interrupted requests.

No raw prompts, attachment bodies, pixels, credentials or chain of thought.
Parsed plans and bounded search queries are intentionally visible to admins.
"""
import re
import time
from backend.sales.runtime_status import current_execution


def safe(value, depth=0):
    if depth>5: return '[depth limit]'
    if isinstance(value,dict):
        return {str(k):safe(v,depth+1) for k,v in list(value.items())[:40]
                if str(k).lower() not in {'text','content','prompt','raw','raw_output','excerpt','source_text','candidate_answer',
                    'customer_reply','key_points','image_observations','image_bytes','image_data_url',
                    'conversation_context','task_memory','memory_updates','support_spans','password','token','api_key',
                    'authorization','image_path','snapshot_path','document_session_id'}}
    if isinstance(value,(list,tuple)): return [safe(v,depth+1) for v in value[:30]]
    if isinstance(value,str):
        value=re.sub(r'(?i)bearer\s+\S+|\bsk-[A-Za-z0-9_-]{12,}', '[credential redacted]', value)
        value=re.sub(r'(?i)(?:api[_-]?key|password|secret|token)\s*[:=]\s*\S+', '[credential redacted]',value)
        value=re.sub(r'[A-Za-z]:[\\/][^\s"<>]+', '[local path redacted]', value)
        return value[:600]
    return value if value is None or isinstance(value,(int,float,bool)) else type(value).__name__


def begin(stage, inputs=None):
    journal=current_execution.get()
    if not journal: return None
    row={'stage':stage,'status':'running','input':safe(inputs or {}),'output':{},
         '_contract_start': len(journal.model_contracts),
         '_error_start': len(journal.errors),
         'token_usage': {'input_tokens':0,'output_tokens':0,'total_tokens':0,'model_calls':0,
                         'complete':False,'scope':'generation_only_inclusive_children'},
         'started_elapsed_ms':round((time.perf_counter()-journal._request_started)*1000,2)}
    journal.stage_details.append(row)
    journal.stage_details=journal.stage_details[-60:]
    return (journal,row,time.perf_counter())


def end(handle, output=None, *, error_code=None):
    if not handle: return
    journal,row,started=handle
    from backend.sales.token_usage import usage
    tokens=usage(journal.model_contracts[row['_contract_start']:])
    if not error_code:
        relevant = [e for e in journal.errors[row.get('_error_start',len(journal.errors)):]
                    if e.get('stage') == row['stage']]
        if relevant:
            error_code = relevant[-1]['code']
    completed={**row,'status':'failed' if error_code else 'completed', 'output':safe(output or {}),
               'token_usage':tokens,
               'elapsed_ms':round((time.perf_counter()-started)*1000,2)}
    completed['output']['token_usage']=tokens
    if error_code: completed['error_code']=error_code
    for index,item in enumerate(journal.stage_details):
        if item is row:
            journal.stage_details[index]=completed
            break
    return tokens


def plan_summary(plan):
    if getattr(plan, 'canonical_task', None):
        from backend.sales.planning_spec import planner_view
        value=planner_view(plan)
    else:
        value=plan.model_dump(exclude={'memory_updates'}) if hasattr(plan,'model_dump') else dict(plan)
    return safe({**value,'guard_diagnostics':getattr(plan,'guard_diagnostics',[]),
                 'planner_output_tokens':getattr(plan,'planner_output_tokens',None),
                 'planner_input_tokens':getattr(plan,'planner_input_tokens',None)})


def step_input(step):
    result={k:v for k,v in step.kwargs.items() if k in {'query','top_k','visual_k','case_k','retrieval_mode',
            'document_scope','target_document_ids','visual_required','retrieval_queries','retrieval_requests','search_targets','source_profile','goals',
            'visual_scope','max_text_tokens','limit','knowledge_categories','company_operation','answer_aspects'}}
    if step.node in {'customer_documents','public_web_search'} and len(step.args)>1:
        result['query']=step.args[1]
    if step.node == 'document_summary' and step.args:
        result.update(document_ids=[d.document_id for d in step.args[0]],
                      text_budget=step.kwargs.get('text_budget'), policy='lazy_per_file_overview')
    return safe(result)


def result_summary(value):
    if isinstance(value,(tuple,list)) and len(value)==2 and isinstance(value[1],dict):
        return {'result_count':len(value[0]) if isinstance(value[0],list) else 0,'search':safe(value[1])}
    if not isinstance(value,dict):
        if hasattr(value,'model_dump'): value=value.model_dump()
        else: return {'result_type':type(value).__name__}
    if not any(key in value for key in ('text_evidence','evidence','visual_assets','selected_visuals','project_cases','customer_reply')):
        return safe(value)  # CPU audit/coverage/packing outputs retain their diagnostic fields.
    summary={k:value[k] for k in ('status','answerable','error','error_code') if k in value}
    for key in ('text_evidence','evidence','visual_assets','selected_visuals','project_cases'):
        rows=value.get(key)
        if isinstance(rows,list):
            summary[key+'_count']=len(rows)
            summary[key]=[{k:r[k] for k in ('evidence_id','asset_id','visual_id','evidence_scope',
                'document_name','customer_title','project_name','city','location','visual_role','requirement_ids') if k in r}
                for r in rows[:20] if isinstance(r,dict)]
    meta=value.get('meta') or {}
    summary['mode']=meta.get('mode')
    summary['audit']={k:meta[k] for k in ('evidence_gate','task_support_audit','generation_input_audit',
        'relation_scope','requirement_coverage','hybrid_validation','query_retrieval_execution_modes','hybrid_query_skip_reasons') if k in meta}
    snapshot = value.get('input_snapshot') or {}
    if snapshot.get('requirement_coverage'):
        summary['requirement_coverage'] = safe(snapshot['requirement_coverage'])
    # Numerical and ID-only context diagnostics, not a copy of model evidence.
    for key in ('kept_evidence_ids','removed_evidence_ids','actual_prompt_tokens','budget_satisfied'):
        if key in value: summary[key]=value[key]
    return safe(summary)
