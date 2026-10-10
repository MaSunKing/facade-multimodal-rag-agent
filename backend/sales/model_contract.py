"""One reusable model-output protocol: safe validation, at most two retries.

Format retries never execute tools, expand ACLs, or reserve semantic recovery.
"""
import contextvars
import json
import time
import os
from contextlib import contextmanager
from dataclasses import dataclass
from backend.sales.output_contract import OutputContractError
from backend.request_budget import check_budget


@dataclass
class Contract:
    role: str
    tokenizer: object
    validate: object
    max_input_tokens: int = 8000
    max_output_tokens: int = 1536
    retry_guard: object = None
    retry_input_builder: object = None
    require_eos: bool = False
    retry_instructions: str = ''
    json_output: bool = False
    json_schema: dict | None = None


def generation_guard(started, seconds):
    """Cooperative protection; cannot interrupt an in-flight CUDA prefill."""
    from backend.request_budget import ModelCallTimeout
    def check():
        check_budget()
        if seconds is not None and time.monotonic()-started >= seconds:
            raise ModelCallTimeout('model_call_timeout')
    class Guard:
        def __call__(self, input_ids, scores, **kwargs):
            check()
            return False
    return Guard(), check


_contract = contextvars.ContextVar('model_output_contract', default=None)


@contextmanager
def output_scope(contract):
    token = _contract.set(contract)
    try:
        yield
    finally:
        _contract.reset(token)


def load_json_delimiters(value):
    """Repair only unambiguous missing/trailing commas; never close truncation.

    Values, keys and their order are unchanged. Embedded quotation mistakes,
    absent fields and incomplete strings/containers remain errors.
    """
    for attempt in range(5):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            if attempt == 4:
                raise
            pos = exc.pos
            before = value[:pos].rstrip()
            after = value[pos:]
            if (exc.msg == "Expecting ',' delimiter" and before and after
                    and after[0] in '"{[' and before[-1] in '"}]0123456789efl'):
                value = value[:pos] + ',' + value[pos:]
            elif (exc.msg in ('Expecting property name enclosed in double quotes', 'Expecting value')
                    and before.endswith(',') and after[:1] in ('}', ']')):
                comma = len(before) - 1
                value = value[:comma] + value[comma + 1:]
            else:
                raise


def object_json(raw):
    value = raw.strip()
    if value.startswith('```') and value.endswith('```'):
        value = value.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    try:
        parsed = load_json_delimiters(value)
    except json.JSONDecodeError as exc:
        failure = OutputContractError('OUTPUT_INVALID_JSON')
        # Locate syntax faults without persisting raw outputs or private text.
        failure.validation_details = {'json_error': exc.msg, 'line': exc.lineno,
                                      'column': exc.colno, 'position': exc.pos}
        raise failure from exc
    if not isinstance(parsed, dict):
        raise OutputContractError('OUTPUT_NOT_OBJECT')
    return parsed


def field_error(field, code='OUTPUT_FIELD_TYPE_INVALID'):
    raise OutputContractError(code, field)


def repair_plan_query_fields(value):
    """Lossless wire repair only: already model-authored queries and scopes.

    Some outputs put request objects in a legacy string-array field. Move the
    explicit scopes to retrieval_requests and retain the identical query text.
    Never split prose, infer a target, change tools or broaden permission.
    """
    if not isinstance(value,dict): return value
    if 'canonical_task' in value:
        field_error('canonical_task', 'OUTPUT_FIELD_UNEXPECTED')
    from backend.sales.planning_spec import expand_planning_spec
    value=expand_planning_spec(value)
    value=dict(value)
    tools=value.get('tools',[])
    existing=value.get('retrieval_requests',[])
    if not isinstance(existing,list): return value
    requests=list(existing)
    for name in ('document_retrieval_queries','retrieval_queries'):
        rows=value.get(name)
        if not isinstance(rows,list) or not rows: continue
        source='customer_documents' if name=='document_retrieval_queries' else (
            tools[0] if isinstance(tools,list) and len(tools)==1 else None)
        if source not in {'customer_documents','company_rag','public_web_search'} or source not in tools: continue
        operation=value.get('document_operation','lookup') if source=='customer_documents' else value.get('company_operation','lookup')
        if not requests and len(rows)>1 and all(isinstance(r,str) and r.strip() for r in rows) and (source=='public_web_search' or operation=='lookup'):
            requests.extend({'source':source,'query':q,'targets':[]} for q in rows)
        if not all(isinstance(r,dict) for r in rows): continue
        if any(set(r)-{'source','query','targets'} or r.get('source',source)!=source or
               not isinstance(r.get('query'),str) or not r['query'].strip() or
               not isinstance(r.get('targets',[]),list) or any(not isinstance(t,str) for t in r.get('targets',[])) for r in rows): continue
        value[name]=[r['query'] for r in rows]
        for r in rows:
            request={'source':source,'query':r['query'],'targets':r.get('targets',[])}
            if request not in requests: requests.append(request)
    if requests: value['retrieval_requests']=requests
    return value


def validate_plan(value, **options):
    compact = isinstance(value, dict) and 'sources' in value
    try:
        return _validate_execution_plan(value, **options)
    except OutputContractError as exc:
        if compact:
            aliases = {'tools':'sources','knowledge_categories':'categories',
                'company_operation':'operation','document_operation':'operation',
                'company_collection':'collection','retrieval_requests':'queries',
                'target_terms':'targets','case_filters':'filters','answer_aspects':'aspects',
                'target_document_ids':'document_names'}
            exc.field = aliases.get(exc.field, exc.field)
        raise


def _validate_execution_plan(value, *, document_ids=(), document_directory=(), allowed_tools=None, require_company_scope=False, require_scoped_requests=False, require_compact=False):
    from backend.sales.tool_planner import ToolPlan
    if require_compact and 'sources' not in value:
        field_error('sources', 'OUTPUT_FIELD_MISSING')
    if require_compact:
        from backend.sales.planning_spec import BACKEND_PLAN_FIELDS, validate_visual_decision
        validate_visual_decision(value, documents_available=bool(document_ids or document_directory
            or allowed_tools is not None and 'customer_documents' in allowed_tools))
        for name in sorted(BACKEND_PLAN_FIELDS & set(value)):
            field_error(name, 'OUTPUT_FIELD_UNEXPECTED')
        if any(isinstance(q,dict) and 'targets' in q for q in value.get('queries',[])):
            field_error('queries', 'OUTPUT_FIELD_UNEXPECTED')
    if 'sources' in value and value.get('document_names'):
        from backend.sales.planning_spec import bind_document_names
        value = bind_document_names(value, document_directory)
    value=repair_plan_query_fields(value)
    if 'tools' not in value:
        field_error('tools', 'OUTPUT_FIELD_MISSING')
    if not isinstance(value['tools'], list) or any(not isinstance(t, str) for t in value['tools']):
        field_error('tools')
    if set(value) - ({k for k,v in ToolPlan.model_fields.items() if not v.exclude} | {'canonical_task'}):
        field_error('plan', 'OUTPUT_FIELD_UNEXPECTED')
    try:
        plan = ToolPlan.model_validate(value, strict=True)
    except Exception as exc:
        errors = exc.errors(include_input=False) if callable(getattr(exc, 'errors', None)) else []
        name = str(errors[0]['loc'][0]) if errors and errors[0]['loc'] else 'plan'
        failure=OutputContractError('OUTPUT_FIELD_VALUE_INVALID',name)
        failure.validation_details={'violations':[{'location':list(e['loc']),'type':e['type']} for e in errors[:3]],
            'field_schema':{k:v for k,v in ToolPlan.model_json_schema().get('properties',{}).get(name,{}).items()
                            if k in {'type','items','enum','maxItems','minItems','maxLength'}}}
        raise failure
    if plan.answer_basis == 'tools' and not plan.tools:
        field_error('tools', 'OUTPUT_FIELD_VALUE_INVALID')
    if require_company_scope and 'company_rag' in plan.tools:
        for name in ('knowledge_categories', 'company_operation'):
            if name not in value:
                field_error(name, 'OUTPUT_FIELD_MISSING')
        if plan.company_operation == 'unknown':
            field_error('company_operation', 'OUTPUT_FIELD_VALUE_INVALID')
        if plan.company_operation == 'summary' and ('company_collection' not in value or 'action' not in value['company_collection']):
            failure = OutputContractError('OUTPUT_FIELD_MISSING', 'company_collection')
            failure.validation_details = {'required_shape': {'action': 'none|list|count', 'group_by': []},
                'policy': 'none for content summary, list for directory enumeration, count for aggregation; preserve user intent'}
            raise failure
    if any(i not in set(document_ids) for i in plan.target_document_ids):
        field_error('target_document_ids', 'OUTPUT_CITATION_UNKNOWN')
    if plan.document_selection == 'explicit' and not plan.target_document_ids:
        field_error('target_document_ids', 'OUTPUT_FIELD_MISSING')
    from backend.sales.content_taxonomy import LABELS, ALIASES
    if any(x not in LABELS and x not in ALIASES for x in plan.knowledge_categories):
        field_error('knowledge_categories', 'OUTPUT_FIELD_VALUE_INVALID')
    from backend.sales.content_taxonomy import canonical_domains
    plan.knowledge_categories = canonical_domains(plan.knowledge_categories)
    if plan.company_collection.action != 'none':
        if 'company_rag' not in plan.tools or plan.company_operation != 'summary' or not plan.knowledge_categories:
            field_error('company_collection', 'OUTPUT_STATE_INCONSISTENT')
        fields_by_category = {
            '02_product_information': {'category','family','product_form','surface_finish'},
            '05_project_case': {'category','province','city','project_category'},
        }
        allowed = set.intersection(*(fields_by_category.get(c, {'category','document_name','section_heading'})
                                    for c in plan.knowledge_categories))
        if not set(plan.company_collection.group_by) <= allowed:
            field_error('company_collection', 'OUTPUT_FIELD_VALUE_INVALID')
        if len(set(plan.company_collection.group_by)) != len(plan.company_collection.group_by):
            field_error('company_collection', 'OUTPUT_FIELD_VALUE_INVALID')
    elif plan.company_collection.group_by:
        field_error('company_collection', 'OUTPUT_STATE_INCONSISTENT')
    if 'knowledge_categories' in value:
        value['knowledge_categories']=plan.knowledge_categories
    if allowed_tools is not None and not set(plan.tools) <= set(allowed_tools):
        # Permissions are execution facts, not an opportunity for model retry.
        raise OutputContractError('ACCESS_DENIED', 'tools')
    if any(r.source not in plan.tools or not r.query.strip() or any(not t.strip() for t in r.targets)
           for r in plan.retrieval_requests):
        field_error('retrieval_requests','OUTPUT_STATE_INCONSISTENT')
    if require_scoped_requests:
        from backend.sales.requirements import source_requests
        # Inspect only model-declared multiplicity, never infer semantics from words.
        multiple = len(set(plan.answer_aspects or plan.answer_goals)) > 1 or len(set(plan.target_terms)) > 1
        sources = [s for s in plan.tools if s == 'public_web_search'
                   or s == 'company_rag' and plan.company_operation == 'lookup'
                   or s == 'customer_documents' and plan.document_operation == 'lookup']
        if (multiple or len(set(plan.target_document_ids)) > 1) and any(not source_requests(plan,s) for s in sources):
            error=OutputContractError('OUTPUT_FIELD_MISSING','retrieval_requests')
            error.validation_details={'required_sources':sources,
                'entry_shape':{'source':sources[0], 'query':'完整独立查询，保留所问对象和条件', 'targets':['当前查询所问目标']},
                'instruction':'For model-declared multiple lookup requirements, output scoped retrieval_requests with source, query, targets; independent web targets require separate queries.'}
            raise error
    return value


def validate_audit(value, evidence_ids, candidate_ids=(), recovery_budget=None, evidence_rows=None):
    from backend.sales.recovery_budget import RECOVERY_NEEDS, validate_budget_view
    if recovery_budget is not None:
        validate_budget_view(recovery_budget)
    if value.get('missing_aspects') and value.get('status') != 'supported' and 'recovery_need' not in value:
        field_error('recovery_need','OUTPUT_FIELD_MISSING')
    if 'recovery_need' in value and (not isinstance(value['recovery_need'],str) or value['recovery_need'] not in RECOVERY_NEEDS):
        field_error('recovery_need','OUTPUT_FIELD_VALUE_INVALID')
    if value.get('status') == 'supported' and value.get('recovery_need', 'none') != 'none':
        field_error('recovery_need','OUTPUT_STATE_INCONSISTENT')
    from backend.sales.evidence_gate import validate_evaluation
    for name in ('status', 'supported_evidence_ids'):
        if name not in value:
            field_error(name, 'OUTPUT_FIELD_MISSING')
    if value['status'] not in ('supported','partial','unsupported','uncertain'):
        field_error('status','OUTPUT_FIELD_VALUE_INVALID')
    ids = value['supported_evidence_ids']
    if not isinstance(ids,list) or any(not isinstance(i,str) for i in ids):
        field_error('supported_evidence_ids')
    if not set(ids) <= set(evidence_ids):
        field_error('supported_evidence_ids','OUTPUT_CITATION_UNKNOWN')
    for name in ('missing_aspects','candidate_asset_ids'):
        if name in value and (not isinstance(value[name],list) or any(not isinstance(i,str) for i in value[name])):
            field_error(name)
    for name in ('problem','reason','visual_reason'):
        if name in value and not isinstance(value[name],str): field_error(name)
    if 'web_can_resolve' in value and not isinstance(value['web_can_resolve'],bool): field_error('web_can_resolve')
    enums={'visual_need':('required','not_required','uncertain'),
        'source_constraint':('evidence_only','general_knowledge_allowed','uncertain'),
        'visual_purpose':('display_only','content_inspection','label_dispute','unreadable','uncertain'),
        'gap_type':('none','current_public_fact','public_fact','local_evidence','ambiguous_request','visual_input','answer_mismatch','scope_mismatch','unknown')}
    conflicts = value.get('conflict_groups', [])
    if not isinstance(conflicts, list) or any(not isinstance(g, list) or len(g) < 2 or any(not isinstance(i, str) for i in g) for g in conflicts):
        field_error('conflict_groups')
    if any(not set(g) <= set(evidence_ids) for g in conflicts):
        field_error('conflict_groups','OUTPUT_CITATION_UNKNOWN')
    if any(not set(g) <= set(ids) or len(set(g)) != len(g) for g in conflicts):
        field_error('conflict_groups','OUTPUT_STATE_INCONSISTENT')
    spans = value.get('support_spans', {})
    if not isinstance(spans,dict) or len(spans)>8:
        field_error('support_spans')
    texts = {r.get('evidence_id') or r.get('source_id'):str(r.get('text') or r.get('excerpt') or '') for r in (evidence_rows or [])}
    for identity, quotes in spans.items():
        if identity not in ids:
            field_error('support_spans','OUTPUT_CITATION_UNKNOWN')
        if not isinstance(quotes,list) or not 1<=len(quotes)<=4 or any(not isinstance(q,str) or not q.strip() or len(q)>1200 for q in quotes):
            field_error('support_spans')
        if evidence_rows is not None and any(q not in texts.get(identity,'') for q in quotes):
            field_error('support_spans','OUTPUT_FIELD_VALUE_INVALID')
    allowed={'status','supported_evidence_ids','document_support','support_spans','conflict_groups','missing_aspects','candidate_asset_ids','problem','reason','visual_reason','web_can_resolve','recovery_need','suggested_categories',*enums}
    if 'suggested_categories' in value:
        from backend.sales.content_taxonomy import LABELS
        choices=value['suggested_categories']
        if not isinstance(choices,list) or len(choices)>6 or any(not isinstance(c,str) or c not in LABELS for c in choices):
            field_error('suggested_categories','OUTPUT_FIELD_VALUE_INVALID')
        if value['status']=='supported' and choices:
            field_error('suggested_categories','OUTPUT_STATE_INCONSISTENT')
    if set(value)-allowed: field_error('audit','OUTPUT_FIELD_UNEXPECTED')
    for name,choices in enums.items():
        if name in value and value[name] not in choices: field_error(name,'OUTPUT_FIELD_VALUE_INVALID')
    if not set(value.get('candidate_asset_ids',[]))<=set(candidate_ids):
        error=OutputContractError('OUTPUT_CITATION_UNKNOWN','candidate_asset_ids')
        error.validation_details={'allowed_candidate_asset_ids':sorted(candidate_ids),
            'rule':'只能原样复制visual_candidates.asset_id；不是证据ID、标题或去掉来源前缀的ID。不需要视觉恢复时省略。'}
        raise error
    try:
        validate_evaluation(value, [{'evidence_id':i} for i in evidence_ids])
    except ValueError as exc:
        raise OutputContractError('OUTPUT_FIELD_VALUE_INVALID','audit') from exc
    return value


def validate_replan(value, *, document_ids=(), document_directory=(), allowed_tools=(), gap_ids=(), recovery_budget=None, previous_plan=None, require_compact=False):
    unexpected=set(value)-{'action','gap_ids','reason','retain_source_tools','revised_plan'}
    if unexpected:
        error=OutputContractError('OUTPUT_FIELD_UNEXPECTED','revised_plan')
        error.validation_details={'unexpected_fields':sorted(unexpected),
            'allowed_fields':['action','gap_ids','reason','retain_source_tools','revised_plan']}
        raise error
    if 'reason' in value and not isinstance(value['reason'],str): field_error('reason')
    retained=value.get('retain_source_tools',[])
    if not isinstance(retained,list) or any(not isinstance(t,str) for t in retained): field_error('retain_source_tools')
    if not set(retained)<=set(allowed_tools): field_error('retain_source_tools','OUTPUT_FIELD_VALUE_INVALID')
    if value.get('action') not in ('execute','stop'): field_error('action','OUTPUT_FIELD_VALUE_INVALID')
    if value['action']=='stop':
        if value.get('revised_plan') is not None: field_error('revised_plan','OUTPUT_STATE_INCONSISTENT')
        return value
    if not isinstance(value.get('revised_plan'),dict): field_error('revised_plan')
    prior=previous_plan or {}
    if not require_compact and 'customer_documents' in prior.get('sources',prior.get('tools',[])):
        candidate=dict(value['revised_plan'])
        ids=candidate.get('target_document_ids',[])
        if not isinstance(ids,list) or any(not isinstance(i,str) or i not in document_ids for i in ids):
            field_error('target_document_ids','OUTPUT_CITATION_UNKNOWN')
        # A recovery query may focus a file; only the user changes selection.
        candidate['document_selection']=prior.get('document_selection','all')
        candidate['target_document_ids']=prior.get('target_document_ids',[])
        value['revised_plan']=candidate
    revised=validate_plan(value['revised_plan'],document_ids=document_ids,document_directory=document_directory,
        allowed_tools=allowed_tools,require_company_scope=True, require_compact=require_compact)
    if require_compact and 'customer_documents' in prior.get('sources',prior.get('tools',[])):
        revised['document_selection'] = prior.get('document_selection', 'all')
        revised['target_document_ids'] = prior.get('target_document_ids', [])
    old_collection=(previous_plan or {}).get('collection', (previous_plan or {}).get('company_collection',{}))
    if 'company_rag' in revised.get('tools',[]) and old_collection.get('action','none')!='none' and 'company_collection' not in revised:
        field_error('company_collection','OUTPUT_FIELD_MISSING')
    if recovery_budget is not None:
        from backend.sales.recovery_budget import validate_budget_view
        validate_budget_view(recovery_budget)
        if not recovery_budget['recovery_available']:
            field_error('action','OUTPUT_STATE_INCONSISTENT')
        if 'customer_documents' in revised.get('tools',[]) and revised.get('document_operation') == 'summary' and not recovery_budget['summary_available']:
            field_error('document_operation','OUTPUT_STATE_INCONSISTENT')
    ids=value.get('gap_ids')
    if not isinstance(ids,list) or not ids:
        field_error('gap_ids','RECOVERY_GAP_IDS_MISSING')
    if any(not isinstance(i,str) or i not in gap_ids for i in ids):
        error=OutputContractError('RECOVERY_GAP_ID_UNKNOWN','gap_ids')
        error.validation_details={'allowed_gap_ids':list(gap_ids),'returned_gap_count':len(ids),
            'invalid_gap_count':sum(not isinstance(i,str) or i not in gap_ids for i in ids)}
        raise error
    if require_compact:
        value = {**value, 'revised_plan': revised}
    return value


def checked_generate(model, *, contract=None, **kwargs):
    """Retry only generation with the same inputs/assets and safe error feedback.

    The caller still sees its original prefix length, including on multimodal
    retries. Image tensors/grids stay identical; positional state is recomputed.
    """
    contract = contract or _contract.get()
    if contract is None:
        from backend.sales.runtime_status import current_execution
        journal = current_execution.get()
        record = {'role':'uncontracted_generation','attempts':[]}
        count = int(kwargs['input_ids'].shape[-1]) if kwargs.get('input_ids') is not None else 0
        row = {'input_tokens':count,'status':'running'}
        record['attempts'].append(row)
        if journal: journal.model_contracts.append(record)
        try:
            output = model.generate(**kwargs)
            row.update(output_tokens=int(output.shape[-1])-count,status='valid')
            return output
        except Exception:
            row['status']='failed'
            raise
    import torch
    from backend.sales.runtime_status import current_execution
    journal=current_execution.get()
    record={'role':contract.role,'max_retries':2,'attempts':[],'status':'running'}
    if not contract.json_output:
        record['json_mode'] = 'disabled'
    if journal:
        journal.model_contracts.append(record)
    original_ids=kwargs['input_ids']
    # No stage timer, including when an older deployment still sets
    # MODEL_CALL_MAX_SECONDS. Request cancellation remains checked per token.
    guard,guard_check=generation_guard(time.monotonic(),None)
    base_kwargs=dict(kwargs)
    from transformers import StoppingCriteriaList
    base_kwargs['stopping_criteria']=StoppingCriteriaList([*(kwargs.get('stopping_criteria') or []),guard])
    feedback=''
    failed_raw=''
    for attempt in range(3):
        guard_check()
        options=dict(base_kwargs)
        if feedback:
            # Close the pending assistant turn before asking for a protocol
            # correction. Do not prefill an instruction as the assistant's answer.
            render=getattr(contract.tokenizer,'apply_chat_template',None)
            # Correct the actual failed candidate, not an empty assistant turn.
            # This remains local model context, never status/log/source evidence.
            retry_text=failed_raw + (getattr(contract.tokenizer,'eos_token','') or '')
            retry_text += render([{'role':'user','content':feedback}],tokenize=False,
                add_generation_prompt=True,enable_thinking=False) if render else feedback
            more=contract.tokenizer(retry_text,return_tensors='pt',add_special_tokens=False)['input_ids'].to(original_ids.device)
            options['input_ids']=torch.cat((original_ids,more),dim=-1)
            if 'attention_mask' in options:
                options['attention_mask']=torch.cat((options['attention_mask'],torch.ones_like(more)),dim=-1)
            # A protocol correction adds text tokens, not new image tokens.
            # Transformers 5 uses this aligned mask for multimodal RoPE.
            for name in ('mm_token_type_ids','token_type_ids'):
                if name in options:
                    extension=torch.zeros(more.shape,dtype=options[name].dtype,device=options[name].device)
                    options[name]=torch.cat((options[name],extension),dim=-1)
            for name in ('position_ids','past_key_values','cache_position'):
                options.pop(name,None)
        input_count=int(options['input_ids'].shape[-1])
        if attempt and input_count > contract.max_input_tokens and contract.retry_input_builder:
            options.update(contract.retry_input_builder(failed_raw, feedback, original_ids.device))
            input_count=int(options['input_ids'].shape[-1])
            record['retry_context_compacted']=True
        if input_count>contract.max_input_tokens:
            code = 'CONTRACT_RETRY_BUDGET_EXCEEDED' if attempt else 'CONTEXT_BUDGET_EXCEEDED'
            record.update(status='failed',error_code=code)
            raise OutputContractError(code)
        if contract.json_output and hasattr(contract.tokenizer, 'get_vocab'):
            from backend.sales.json_decoding import json_prefix
            options['prefix_allowed_tokens_fn'] = json_prefix(contract.tokenizer,contract.json_schema)
            record['json_mode'] = 'schema_constrained_object' if contract.json_schema else 'syntax_constrained_object'
        if attempt and contract.retry_guard:
            try:
                contract.retry_guard(input_count,options['max_new_tokens'])
            except Exception as exc:
                record.update(status='failed',error_code=getattr(exc, 'code', str(exc)),
                              error_type=type(exc).__name__)
                raise
        if attempt and callable(getattr(options.get('streamer'),'start_new_attempt',None)):
            options['streamer'].start_new_attempt()
        started=time.monotonic()
        row={'attempt':attempt+1,'input_tokens':input_count,'output_limit':options['max_new_tokens'],
             'started_unix_seconds':time.time(),'status':'running'}
        record['attempts'].append(row)
        from backend.sales.generation_profile import AttemptTimer
        profile = {'input_tokens': input_count}
        timer = AttemptTimer(profile, options.get('streamer'))
        options['streamer'] = timer
        try:
            guard_check()
            output=model.generate(**options)
        except Exception as exc:
            record.update(status='failed',error_type=type(exc).__name__)
            row.update(status='failed',error_type=type(exc).__name__,elapsed_seconds=round(time.monotonic()-started,3))
            raise
        finally:
            timer.end()
            profile['client_generation_seconds'] = round(time.monotonic()-started,4)
            if getattr(model, 'is_remote_model', False):
                profile.update(getattr(model, 'last_runtime_profile', {}))
                profile['gpu_metrics_origin'] = 'independent_model_service'
            row['runtime_profile'] = dict(profile)
            record['runtime_profile'] = dict(profile)
        suffix=output[:,input_count:]
        raw=contract.tokenizer.decode(suffix[0],skip_special_tokens=True)
        count=int(suffix.shape[-1])
        row.update(output_tokens=count,elapsed_seconds=round(time.monotonic()-started,3))
        if contract.json_output:
            # Structural diagnostics only: never persist raw model/source text.
            row['output_shape']={'last_token_id':int(suffix[0,-1]) if count else None,
                'line_count':raw.count('\n')+1,'ends_with_object':raw.rstrip().endswith('}'),
                'character_count':len(raw)}
            constraint = options.get('prefix_allowed_tokens_fn')
            row['schema_fallback_steps'] = getattr(constraint,'schema_fallback_steps',0)
        guard_check()
        try:
            if contract.require_eos:
                eos=getattr(getattr(model,'generation_config',None),'eos_token_id',getattr(contract.tokenizer,'eos_token_id',None))
                eos=eos if isinstance(eos,list) else [eos]
                if int(output[0,-1]) not in eos:
                    raise OutputContractError('OUTPUT_TOKEN_LIMIT' if count>=options['max_new_tokens'] else 'OUTPUT_INCOMPLETE','text')
            contract.validate(raw)
        except (OutputContractError, ValueError, json.JSONDecodeError) as exc:
            code=getattr(exc,'code','OUTPUT_FIELD_VALUE_INVALID')
            if code=='OUTPUT_INVALID_JSON' and count>=options['max_new_tokens']:
                code='OUTPUT_TOKEN_LIMIT'
            field=getattr(exc,'field',None)
            row.update(status='invalid',error_code=code,field=field)
            if getattr(exc,'validation_details',None):
                row['validation_details']=exc.validation_details
            if code=='ACCESS_DENIED' or attempt==2:
                record.update(status='failed',error_code=code)
                failure=OutputContractError(code,field)
                failure.role=contract.role; failure.attempts=attempt+1
                raise failure from exc
            # No raw outputs or source contents in feedback/status. Original
            # task/evidence remain authoritative, including negative claims.
            feedback=protocol_feedback(code,field,getattr(exc,'validation_details',{}))
            failed_raw=raw
            if contract.retry_instructions:
                feedback += contract.retry_instructions + '\n'
            if code=='OUTPUT_TOKEN_LIMIT':
                base_kwargs['max_new_tokens']=min(contract.max_output_tokens,max(options['max_new_tokens']+128,int(options['max_new_tokens']*1.5)))
            del output,suffix
            continue
        row['status']='valid'; record.update(status='passed',retries_used=attempt)
        return torch.cat((original_ids,suffix),dim=-1) if attempt else output


def json_contract(role, tokenizer, validator=None, **limits):
    def check(raw):
        value=object_json(raw)
        return validator(value) if validator else value
    return Contract(role,tokenizer,check,json_output=True,**limits)


def answer_contract(tokenizer, evidence_ids, *, allow_images=False):
    from backend.sales.output_contract import natural_language_answer
    def check(raw):
        return natural_language_answer(raw, evidence_ids)
    return Contract('final_answer',tokenizer,check,json_output=False,json_schema=None,
        retry_instructions='只输出完整自然语言答复正文；不要程序字段、JSON协议包装或分析过程。')


def chat_contract(tokenizer, source_ids=(), *, program_references=False):
    if program_references:
        from backend.sales.output_contract import natural_language_answer
        def text_check(raw):
            value = natural_language_answer(raw)
            value['used_source_ids'] = list(dict.fromkeys(source_ids))
            return value
        return Contract('chat_answer',tokenizer,text_check,json_output=False,json_schema=None)
    def check(value):
        if 'customer_reply' not in value: field_error('customer_reply','OUTPUT_FIELD_MISSING')
        if not isinstance(value['customer_reply'],str): field_error('customer_reply')
        if not value['customer_reply'].strip(): field_error('customer_reply','OUTPUT_REPLY_EMPTY')
        if program_references:
            value['used_source_ids']=list(dict.fromkeys(source_ids))
        ids=value.get('used_source_ids',[])
        if not isinstance(ids,list) or any(not isinstance(i,str) for i in ids): field_error('used_source_ids')
        if not set(ids)<=set(source_ids): field_error('used_source_ids','OUTPUT_CITATION_UNKNOWN')
        if 'image_observations' in value and (not isinstance(value['image_observations'],list)
            or any(not isinstance(i,str) for i in value['image_observations'])): field_error('image_observations')
        return value
    return json_contract('chat_answer',tokenizer,check)


def text_contract(role,tokenizer,**limits):
    def check(raw):
        if not isinstance(raw,str) or not raw.strip(): field_error('text','OUTPUT_REPLY_EMPTY')
        return raw
    return Contract(role,tokenizer,check,**limits)


def fields_contract(role,tokenizer,required,optional=None,enums=None,validator=None,**limits):
    """Schema-only adapter for secondary model roles, independent of modality."""
    optional=optional or {}; enums=enums or {}
    def check(value):
        if set(value)-set(required)-set(optional): field_error('schema','OUTPUT_FIELD_UNEXPECTED')
        for name,kind in required.items():
            if name not in value: field_error(name,'OUTPUT_FIELD_MISSING')
            if not isinstance(value[name],kind): field_error(name)
        for name,kind in optional.items():
            if name in value and not isinstance(value[name],kind): field_error(name)
        for name,choices in enums.items():
            if name in value and value[name] not in choices: field_error(name,'OUTPUT_FIELD_VALUE_INVALID')
        return validator(value) if validator else value
    return json_contract(role,tokenizer,check,**limits)


def mapping_contract(tokenizer):
    def check(value):
        if not (set(value)&{'tables','sections'}): field_error('tables','OUTPUT_FIELD_MISSING')
        if set(value)-{'tables','sections'}: field_error('schema','OUTPUT_FIELD_UNEXPECTED')
        for name,rows in value.items():
            if not isinstance(rows,list) or any(not isinstance(r,dict) for r in rows): field_error(name)
            for row in rows:
                fields={'table_id':str,'period_year':(str,int,type(None)),'mappings':list} if name=='tables' else {'section_id':str,'scope':str,'confidence':(float,int)}
                if set(row)-set(fields): field_error(name,'OUTPUT_FIELD_UNEXPECTED')
                for key,kind in fields.items():
                    if key in row and not isinstance(row[key],kind): field_error(key)
                identity='table_id' if name=='tables' else 'section_id'
                if identity not in row: field_error(identity,'OUTPUT_FIELD_MISSING')
                for entry in row.get('mappings',[]):
                    if not isinstance(entry,dict): field_error('mappings')
                    for key,kind in {'label':str,'target':str,'confidence':(float,int)}.items():
                        if key not in entry: field_error(key,'OUTPUT_FIELD_MISSING')
                        if not isinstance(entry[key],kind): field_error(key)
        return value
    return json_contract('table_mapping',tokenizer,check)


def case_contract(tokenizer):
    import re
    def check(raw):
        if not re.fullmatch(r'(?:NOT_CASE_REFERENCE\||CASE_REFERENCE\|[^\n]*)',raw.strip()): field_error('case_reference','OUTPUT_FIELD_VALUE_INVALID')
        return raw
    return Contract('case_router',tokenizer,check)


def protocol_feedback(code, field, details):
    feedback = '\n上一次输出未通过协议检查：'+json.dumps(
        {'code':code,'field':field,'details':details},ensure_ascii=False)+\
        '。仅纠正格式、字段或允许的ID；保持原任务、证据、范围与事实不变。不调用工具，不增加事实。重新输出完整合规结果，不输出解释。\n'
    if code == 'OUTPUT_TOKEN_LIMIT':
        feedback += ('先输出核心必填字段，省略可选字段及重复清单/背景，不续写截断文本。'
                     '集合过长时明确给代表性子集及实际覆盖范围，不能声称列全；'
                     '保留所问对象、关键条件、否定与单位，重新生成完整紧凑JSON。\n')
    return feedback


def format_retry_reserve(count_tokens, instructions, *, max_failed_output_tokens):
    """Reserve the latest failed candidate plus protocol correction envelope.

    checked_generate replaces feedback on each attempt; it does not accumulate
    both failed answers. Source evidence and the original task remain intact.
    """
    # Count the same correction template used by checked_generate, including
    # a bounded diagnostic envelope. Chat/EOS delimiters get 64 extra tokens.
    envelope=protocol_feedback('OUTPUT_FIELD_VALUE_INVALID','supported_evidence_ids',
        {'json_error':"Expecting ',' delimiter",'line':999,'column':99999,'position':99999})
    return max_failed_output_tokens + count_tokens(envelope + instructions + '\n') + 64


def support_contract(tokenizer,payload,purpose='audit',*,max_input_tokens=8000):
    if purpose=='answer_audit':
        from backend.sales.answer_review import validate_review,review_schema
        contract=json_contract('answer_audit',tokenizer,lambda value:validate_review(value,payload),
            retry_instructions=('仅输出decision、coverage_status及必要字段；revise须revision，recover须proof。不输出证据支持ID或原文引用。' if payload.get('review_policy')=='task_coverage_only' else '输出decision、status、supported_evidence_ids及必要字段；revise须revision，recover须proof。comparison须support_spans对象。'),
          max_input_tokens=max_input_tokens)
        # Format repair must not silently expand beyond the admitted output
        # reservation. Semantic recovery/revision counters are independent.
        contract.max_output_tokens=384
        contract.json_schema=review_schema(payload)
        if payload.get('review_policy')=='task_coverage_only' and payload.get('image_labels'):
            contract.retry_instructions += ' 含图片标签时必须先输出label_consistency；conflicts的引文原样来自标签uncertainties和草稿，同一asset_id。'
        return contract
    if purpose=='replan':
        documents=[r['document_id'] for r in payload.get('attachment_language_summary',[])]
        tools=[k for k,v in payload.get('available',{}).items() if v]
        gaps=[g['gap_id'] for g in payload.get('gap_state',{}).get('gaps',[])]
        contract=json_contract('replanner',tokenizer,lambda value:validate_replan(value,
            document_ids=documents,document_directory=payload.get('attachment_language_summary',[]),
            allowed_tools=tools,gap_ids=gaps,recovery_budget=payload.get('recovery_budget'),previous_plan=payload.get('previous_plan'),
            require_compact='sources' in payload.get('previous_plan',{})))
        contract.max_input_tokens=max_input_tokens
        from backend.sales.planning_spec import planning_output_schema
        fields=list(planning_output_schema(tools)['properties'])
        contract.retry_instructions='外层只允许action、gap_ids、reason、retain_source_tools、revised_plan。revised_plan允许字段：'+','.join(fields)+'。本轮允许的gap_ids：'+json.dumps(gaps,ensure_ascii=False)+'。execute须从中原样复制，先输出action和gap_ids；无法执行时选择stop。'
        return contract
    ids={str(r.get('evidence_id') or r.get('source_id')) for r in payload.get('evidence',[]) if isinstance(r,dict)}
    ids.update(r['image_id'] for r in payload.get('audit_visual_manifest',[]) if r.get('image_id'))
    candidates={r['asset_id'] for r in payload.get('visual_candidates',[]) if r.get('asset_id')}
    from backend.sales.task_support import AUDIT_RETRY_CONTRACT
    def check(value):
        # Company pixels cannot be requested. Apply the existing backend
        # policy BEFORE validating recovery candidate IDs, not after expensive
        # retries. Preserve status/support/missing facts exactly as returned.
        from backend.sales.visual_policy import enforce_company_label_policy
        policy = enforce_company_label_policy(value,company_only=bool(
            (payload.get('company_visual_policy') or {}).get('company_only')))
        value = {**value, **{k:policy[k] for k in ('visual_need','visual_purpose',
            'candidate_asset_ids','recovery_need','gap_type') if k in policy}}
        validate_audit(value,ids,candidates,payload.get('recovery_budget'),payload.get('evidence',[]))
        from backend.documents.document_tasks import validate_document_support
        validate_document_support(value,payload.get('attachment_task',{}))
        # A scope coverage gap is not malformed JSON. Keep only model-selected
        # support and route the missing file through the existing semantic
        # recovery; never promote it to support or spend format retries on it.
        from backend.documents.document_tasks import enforce_file_support
        required=[{**d,'document_name':d.get('document_name') or d['document_id']}
            for d in payload.get('required_documents',[]) if d.get('document_id')]
        value=enforce_file_support(value,required,payload.get('evidence',[]))
        if value.get('file_coverage',{}).get('complete') is False:
            value.setdefault('recovery_need','lookup')
        files=payload.get('attachment_task',{}).get('files',[])
        if files:
            chosen=set(value.get('supported_evidence_ids',[]))
            value['document_support']={doc['document_id']:[i for i in doc['available_evidence_ids'] if i in chosen]
                for doc in files}
        return value
    return json_contract('support_audit',tokenizer,check,
                         retry_instructions=AUDIT_RETRY_CONTRACT,max_input_tokens=max_input_tokens)
