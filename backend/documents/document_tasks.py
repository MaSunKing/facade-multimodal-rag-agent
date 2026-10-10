"""Backend-owned file scope and coverage, independent of relevance scores."""
from __future__ import annotations


def resolve_documents(documents, requested_ids, *, scope, operation, selection=None):
    available = {d.document_id: d for d in documents}
    requested = list(dict.fromkeys(requested_ids or []))
    unknown = [key for key in requested if key not in available]
    if unknown:
        raise ValueError('unknown_target_document_id')
    # None preserves old callers; runtime plans explicitly pass all/explicit.
    if selection == 'explicit' and not requested:
        raise ValueError('explicit_document_selection_requires_ids')
    selected = ([available[key] for key in requested] if requested else list(documents)) if selection != 'all' else list(documents)
    coverage_required = selection == 'all' or operation == 'summary' or scope in {'whole_document', 'cross_document'}
    required = [{'document_id': d.document_id, 'document_name': d.file_name,
                 'parse_quality': getattr(d, 'parse_quality', {})}
                for d in selected] if coverage_required else []
    return selected, required


def file_aspects(aspects, required, *, operation=None):
    base = list(dict.fromkeys(aspects))
    bindings = []
    for document in required:
        action = '概括主要内容' if operation == 'summary' else '处理本轮要求'
        label = f"文件《{document['document_name']}》（ID:{document['document_id']}）：{action}"
        if label not in base:
            base.append(label)
        bindings.append({**document, 'aspect_index': base.index(label), 'requirement': label})
    return base, bindings


def audit_file_answers(result, bindings, evidence):
    reply = str(result.get('customer_reply') or '')
    checklist = result.get('answer_aspect_coverage') or []
    if not isinstance(checklist, list):
        checklist = []
    output = []
    for document in bindings:
        matched = [r for r in checklist if isinstance(r, dict)
                   and r.get('aspect_index') == document['aspect_index']]
        valid = False
        for row in matched:
            quote = str(row.get('answer_quote') or '').strip()
            if not quote or quote not in reply:
                continue
            # Explicit inability is handled, not a supported file summary.
            if row.get('status') == 'evidence_missing':
                valid = True
                break
            ids = row.get('evidence_ids') or []
            if not isinstance(ids, list):
                continue
            cited = {c.get('evidence_id') for c in result.get('citations', []) if isinstance(c, dict)}
            if row.get('status') == 'answered' and ids and all(
                    isinstance(key, str) and key in evidence and key in cited for key in ids) and any(
                    evidence[key].get('document_id') == document['document_id'] for key in ids):
                valid = True
                break
        output.append({**document, 'accounted': valid})
    return {'complete': all(r['accounted'] for r in output), 'documents': output,
            'missing_document_ids': [r['document_id'] for r in output if not r['accounted']],
            'method': 'file_identity_citation_and_answer_span_not_semantic_gold'}


def enforce_file_support(verdict, required, evidence):
    """A supported verdict must actually reference every required file."""
    if not required or verdict.get('error_code') or verdict.get('status') not in {'supported', 'partial'}:
        return verdict
    ids = set(verdict.get('supported_evidence_ids') or [])
    covered = {r.get('document_id') for r in evidence if r.get('evidence_id') in ids}
    missing = [d for d in required if d['document_id'] not in covered]
    if not missing:
        return {**verdict, 'file_coverage': {'complete': True, 'required': required}}
    return {**verdict, 'status': 'partial' if covered else 'unsupported', 'gap_type': 'local_evidence',
            'problem': '审核支持的证据未覆盖目标文件：' + '、'.join(d['document_name'] for d in missing),
            'missing_aspects': [f"文件《{d['document_name']}》（ID:{d['document_id']}）的本轮内容要求" for d in missing],
            'file_coverage': {'complete': False, 'required': required, 'missing': missing}}


def attachment_task_operation(plan):
    if isinstance(plan.recovery_directive.get('original_delivery_task'),dict):
        from backend.sales.planning_spec import delivery_operation
        return delivery_operation(plan,'customer_documents')
    return plan.recovery_directive.get('original_document_operation',plan.document_operation)


def attachment_scope_instruction(task):
    if task.get('selection')!='all' or task.get('operation')!='summary' or len(task.get('files',[]))<2:
        return ''
    names='、'.join('《'+d['document_name']+'》' for d in task['files'])
    return ('\n本轮用户选择了全部附件范围，当前实际交付任务：分别概括'+names+'。'
        '原话点名某文件不取消这个范围。各文件分别支持自己的概括，不要求另一个文件去证明点名文件的内容。'
        '逐文件检查并回答，保留原话提出的维度与条件；确实不支持的文件说明具体缺口，不能强行引用。')


def attachment_delivery_question(payload, sources):
    """One resolved file task for Audit and Final; raw conditions stay visible."""
    task=payload.get('attachment_task',{})
    if (set(sources)!={'customer_documents'} or task.get('operation')!='summary'
            or task.get('selection')!='all' or len(task.get('files',[]))<2):
        return payload
    result=dict(payload)
    for key in ('question','customer_question'):
        if key in payload:
            result['original_question']=payload.get('original_question',payload[key])
            names='、'.join('《'+d['document_name']+'》' for d in task['files'])
            result[key]=('分别概括下列所选文件各自的内容：'+names+'。各文件只支持自己的概括，不要求其他文件证明点名文件的内容。\n'
                '用户对概括维度、条件、否定及排除项的要求（完整原话）：'+result['original_question']+'\n'
                '以每个所选文件作为其自身概括对象；原话点名不缩小文件范围，但其明确维度和排除项必须遵守，不扩展为其他内容。')
    result['question_basis']='backend_selected_file_scope; original_question_constraints_still_apply'
    return result


def attachment_support_envelope(payload):
    """Summary obligations come from selected files, not synthesized lookup hints.

    Preserve the original question and any explicit constraints in it. Only
    discard model-generated summary facets; these are not user requirements.
    Mixed company/web requirements remain untouched.
    """
    task=payload.get('attachment_task',{})
    if task.get('operation')!='summary' or task.get('selection')!='all':
        return payload
    plan=payload.get('planned_task') or payload.get('previous_plan') or {}
    sources=plan.get('sources',plan.get('tools',[]))
    if any(source!='customer_documents' for source in sources):
        return payload
    result=dict(attachment_delivery_question(payload,sources or ['customer_documents']))
    # The selected-file policy is an explicit user configuration. Make that
    # resolved deliverable the audit task, not a second contradictory hint.
    # Keep the exact utterance independently for conditions and task fidelity.
    result['proposed_requirements']=[]
    result['retrieval_requirements']=[r for r in payload.get('retrieval_requirements',[])
        if r.get('target_kind')!='document']
    result['retrieval_requirements'].extend({
        'target_kind':'document','target_id':d['document_id'],
        'target_name':d['document_name'],'aspect':d['task'],
        'evidence_ids':d.get('available_evidence_ids',[]),
        'origin':'selected_file_summary_obligation_not_semantic_support'
    } for d in task.get('files',[]))
    for field in ('planned_task','previous_plan','execution_task'):
        if isinstance(result.get(field),dict):
            result[field]={k:v for k,v in result[field].items()
                if k not in {'aspects','targets','answer_aspects','answer_goals'}}
    return result


def attachment_task_table(required, evidence, *, operation, selection):
    """Explicit server-owned file obligations, not a guessed semantic rewrite."""
    return {'operation':operation, 'selection':selection,
        'effective_task':('分别概括以下每个所选文件的主要内容：' if operation=='summary'
            else '对以下每个所选文件分别处理客户当前要求，不互换对象或条件：')+
            '、'.join(d['document_name'] for d in required),
        'policy':'each_selected_file_independently; named_file_does_not_exclude_others_unless_explicit',
        'files':[{'document_id':d['document_id'], 'document_name':d['document_name'],
            'task':'概括该文件在原始用户要求范围内的内容，遵守其条件和排除项' if operation == 'summary' else d.get('requirement','处理当前要求'),
            'available_evidence_ids':[r['evidence_id'] for r in evidence if r.get('document_id')==d['document_id']]}
            for d in required]}


def validate_document_support(value, task):
    from backend.sales.output_contract import OutputContractError
    files=task.get('files',[])
    if not files:
        result=value.get('document_support')
        if result is not None and (not isinstance(result,dict) or any(
                not isinstance(ids,list) or any(not isinstance(i,str) or i not in value.get('supported_evidence_ids',[]) for i in ids)
                for ids in result.values())):
            raise OutputContractError('OUTPUT_STATE_INCONSISTENT','document_support')
        return
    result=value.get('document_support')
    if result is None:
        return
    # Equivalent encodings only: no selection, missing citations or guessed IDs.
    if isinstance(result,list) and all(isinstance(row,dict) and set(row) in (
            {'document_id','evidence_ids'}, {'document_id','supported_evidence_ids'}) for row in result):
        if len({row['document_id'] for row in result if isinstance(row['document_id'],str)})==len(result):
            result={row['document_id']:row.get('evidence_ids',row.get('supported_evidence_ids')) for row in result}
    if isinstance(result,dict):
        result={key:([ids] if isinstance(ids,str) else ids) for key,ids in result.items()}
        value['document_support']=result
    required={d['document_id'] for d in files}
    if not isinstance(result,dict) or set(result)!=required:
        code=('OUTPUT_FIELD_MISSING' if result is None or isinstance(result,dict) and required-set(result)
              else 'OUTPUT_FIELD_TYPE_INVALID' if not isinstance(result,dict) else 'OUTPUT_CITATION_UNKNOWN')
        failure=OutputContractError(code,'document_support')
        failure.validation_details={'required_document_ids':[d['document_id'] for d in files],
            'available_evidence_ids_by_document':{d['document_id']:d['available_evidence_ids'] for d in files},
            'shape':'JSON object: document_id -> evidence ID string array; [] only for an actual unsupported file',
            'returned_type':type(result).__name__}
        raise failure
    for doc in files:
        ids=result[doc['document_id']]
        if not isinstance(ids,list) or any(not isinstance(i,str) or i not in doc['available_evidence_ids'] for i in ids):
            raise OutputContractError('OUTPUT_CITATION_UNKNOWN','document_support')
        if not ids and (value.get('status')=='supported' or not value.get('missing_aspects')):
            failure=OutputContractError('OUTPUT_STATE_INCONSISTENT','document_support')
            failure.validation_details={'unsupported_document_id':doc['document_id'],
                'rule':'空支持数组必须是partial/unsupported并说明该文件真实缺口；不强行引用',
                'available_evidence_ids_by_document':{d['document_id']:d['available_evidence_ids'] for d in files}}
            raise failure
    # The model already selected these IDs per file. Canonicalise the redundant
    # flat list without choosing sources or manufacturing semantic support.
    value['supported_evidence_ids']=list(dict.fromkeys([*value.get('supported_evidence_ids',[]),
        *(i for ids in result.values() for i in ids)]))
