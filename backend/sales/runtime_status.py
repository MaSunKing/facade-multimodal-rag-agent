"""Safe, request-scoped execution/error protocol shared by API and graph.

Never put exception messages, prompts, credentials or local paths in this protocol.
"""
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import time
from contextlib import contextmanager
from functools import wraps
import uuid
from pathlib import Path

# code: user message, next action, retry allowed, model recovery action
ERRORS = {
    'RECOVERY_GAP_IDS_MISSING': ('恢复规划未指定要补充的缺口，补取证未执行。','当前回答仅使用已有支持内容；请根据请求编号排查恢复规划字段。',False,'return_supported_partial_without_tool_execution'),
    'RECOVERY_GAP_ID_UNKNOWN': ('恢复规划填写了不属于本轮的缺口ID，补取证未执行。','当前回答仅使用已有支持内容；这不是最终答案的引用错误。',False,'return_supported_partial_without_tool_execution'),
    'ATTACHMENT_SESSION_UNAVAILABLE': ('关联附件会话不可用，未开始模型处理。','会话可能已过期、服务已重启或不属于当前客户端；请重新上传附件。',False,'require_available_owned_attachment_no_model_retry'),
    'WORKFLOW_NODE_UNREGISTERED': ('工作流节点尚未注册，本次操作未完成。','请提供请求编号排查节点配置；重新上传或重复提问不能修复此问题。',False,'fix_workflow_registration_no_model_retry'),
    'MODEL_CALL_TIMEOUT': ('单次模型处理超过阶段安全时限。','请根据请求编号查看卡住的模型阶段；本次未继续格式重试。',False,'stop_model_stage'),
    'OUTPUT_INCOMPLETE': ('模型输出在完成前停止。','请根据请求编号查看对应阶段的停止原因。',False,'report_stage_interruption'),
    'CONTRACT_RETRY_BUDGET_EXCEEDED': ('模型输出修复需要的输入超过本轮预算。','请减少输入范围或重新提交；未继续调用模型。',False,'stop_format_retries'),
    'OUTPUT_NOT_OBJECT': ('回答输出不是JSON对象。','请重新提交；系统未自动调用模型修复。',False,'return_specific_error'),
    'OUTPUT_FIELD_MISSING': ('回答缺少必要字段。','请重新提交；系统不会猜测缺失内容。',False,'return_specific_error'),
    'OUTPUT_FIELD_UNEXPECTED': ('回答包含协议不允许的字段。','请重新提交或提供请求编号排查。',False,'return_specific_error'),
    'OUTPUT_FIELD_TYPE_INVALID': ('回答字段的数据类型不正确。','请重新提交或提供请求编号排查。',False,'return_specific_error'),
    'OUTPUT_FIELD_VALUE_INVALID': ('回答字段的取值不符合协议。','请重新提交或提供请求编号排查。',False,'return_specific_error'),
    'OUTPUT_REPLY_EMPTY': ('模型未生成有效回答正文。','请重新提交。',False,'return_specific_error'),
    'OUTPUT_CITATION_FORMAT_INVALID': ('回答引用格式不正确，无法确定对应来源。','请重新提交；系统不会猜测引用来源。',False,'return_specific_error'),
    'OUTPUT_CITATION_UNKNOWN': ('回答引用了不存在或未进入本轮模型输入的证据ID。','请重新提交；系统未替换为其他来源。',False,'return_specific_error'),
    'OUTPUT_CITATION_MISSING': ('回答缺少必要的来源引用。','请重新提交；系统不会自动添加猜测的来源。',False,'return_specific_error'),
    'OUTPUT_STATE_INCONSISTENT': ('回答状态与引用或要点字段相互矛盾。','请重新提交；系统未改变答案含义。',False,'return_specific_error'),
    'DOCUMENT_SUMMARY_FAILED': ('文件内容概括未完成。','原始附件仍保留，请缩小范围或重试；不把失败的摘要当作文件事实。',False,'report_summary_failure_preserve_original_sources'),
    'PLAN_DOCUMENT_SCOPE_INVALID': ('规划指定了当前会话不存在的文件。','请重试；未读取其他用户或不存在的文件。',False,'select_only_backend_provided_document_ids'),
    'REPLAN_INVALID_PLAN': ('恢复计划格式或权限校验失败，未执行补充工具。','请查看首轮证据与具体缺口；格式错误不是资料不足。',False,'report_plan_contract_failure'),
    'AUDIT_OUTPUT_TRUNCATED': ('证据审核输出达到长度上限，审核未完成。','请重试；这不代表资料不足。',True,'repair_audit_format_once_with_shared_budget'),
    'AUDIT_INVALID_JSON': ('证据审核未返回完整有效的JSON。','请重试；尚不能判断证据是否足够。',True,'repair_audit_format_once_with_shared_budget'),
    'AUDIT_SCHEMA_INVALID': ('证据审核字段或类型不符合约定。','请重试；不采用未验证的审核结论。',True,'repair_audit_format_once_with_shared_budget'),
    'AUDIT_INVALID_REFERENCE': ('证据审核引用了本轮不存在的证据或图片。','请核对来源；不采用该审核结论。',False,'discard_invalid_audit_reference'),
    'AUDIT_INPUT_BUDGET_EXCEEDED': ('证据审核输入超过安全预算。','请缩小资料范围；尚不能判断证据是否足够。',False,'reduce_context_preserving_required_evidence'),
    'AUDIT_BUDGET_EXHAUSTED': ('剩余处理时间不足以完成证据审核。','请重试；这不代表资料不足。',False,'stop_without_asserting_evidence_absence'),
    'AUDIT_FAILED': ('证据审核执行异常，审核未完成。','请提供请求编号排查；不采用未验证结论。',False,'report_audit_failure_not_evidence_gap'),
    'AUDIT_INPUT_INVALID': ('证据审核输入准备失败，尚未调用审核模型。','请提供请求编号排查；这不代表模型输出格式错误或资料不足。',False,'repair_audit_input_not_model_output'),
    'REPLAN_INPUT_BUDGET_EXCEEDED': ('恢复规划输入超过安全预算，尚未生成恢复计划。','已保留有效证据；请根据请求编号排查输入包装。',False,'reduce_replan_metadata_not_task'),
    'REPLAN_INPUT_INVALID': ('恢复规划输入准备失败，未执行补取证。','请根据请求编号排查；这不代表原始资料不足。',False,'repair_replan_input'),
    'CONTEXT_INTEGRITY_FAILED': ('上下文证据完整性检查未通过。','请提供请求编号排查；不会使用损坏或不完整的证据组合。',False,'repair_context_ids_hashes_or_protected_groups'),
    'REQUEST_INVALID': ('请求参数或文件格式不符合要求。','请检查输入和文件格式。',False,'ask_for_valid_input'),
    'DUPLICATE_ATTACHMENT_CONTENT': ('该文件内容已添加，请勿重复上传。','请移除重复文件；同名但内容不同的文件可以上传。',False,'remove_duplicate_attachment'),
    'AUTH_REQUIRED': ('需要登录后执行此操作。','请先登录。',False,'request_authentication'),
    'ACCESS_DENIED': ('当前账号无权访问该资料或工具。','请联系管理员授权。',False,'do_not_bypass_permissions'),
    'NOT_FOUND': ('请求的资源不存在。','请检查资源或重新上传。',False,'request_available_resource'),
    'ATTACHMENT_EXPIRED': ('附件会话已过期或不可用。','请重新上传附件。',False,'request_reupload'),
    'FILE_TOO_LARGE': ('上传文件超过大小限制。','请分批上传或缩小文件。',False,'request_smaller_input'),
    'UNSUPPORTED_FORMAT': ('当前无法处理这种文件格式。','请转换为支持的格式后上传。',False,'request_supported_format'),
    'PARSE_FAILED': ('文件解析失败。','请检查文件是否损坏或加密，必要时换一个版本。',False,'use_other_available_evidence'),
    'OCR_FAILED': ('部分图像文字未能识别。','请提供更清晰图片，或核对未识别页面。',False,'mark_visual_coverage_incomplete'),
    'VISUAL_COVERAGE_INCOMPLETE': ('附件的视觉内容尚未全部处理。','已解析内容仍可使用；涉及未处理图片或页面的结论需要进一步核验。',False,'do_not_claim_full_document_visual_coverage'),
    'RETRIEVAL_EMPTY': ('没有找到足够的相关证据。','请补充相关资料或明确查询目标。',False,'replan_once_or_report_insufficient_evidence'),
    'RETRIEVAL_FAILED': ('本地资料检索暂时失败。','请稍后重试；持续失败请联系管理员。',True,'use_other_available_evidence'),
    'WEB_NOT_CONFIGURED': ('联网服务尚未配置。','请管理员检查联网配置。',False,'continue_with_local_evidence_only'),
    'WEB_QUOTA_EXHAUSTED': ('本日联网额度已用完。','可使用本地资料，或等待额度恢复。',False,'do_not_retry_web'),
    'WEB_TIMEOUT': ('联网查询超时。','可稍后重试，已找到的本地资料仍可使用。',True,'continue_with_local_evidence_only'),
    'WEB_AUTH_FAILED': ('联网接口认证失败。','请管理员检查API凭据和权限。',False,'do_not_retry_web'),
    'WEB_RATE_LIMITED': ('联网接口暂时限流。','请稍后重试，本地资料仍可使用。',True,'defer_web_request'),
    'WEB_NETWORK_FAILED': ('无法连接联网服务。','请管理员检查网络和代理连接。',True,'continue_with_local_evidence_only'),
    'WEB_PROVIDER_ERROR': ('联网服务端暂时异常。','请稍后重试。',True,'continue_with_local_evidence_only'),
    'WEB_RESPONSE_INVALID': ('联网接口返回的数据格式异常。','请管理员检查接口协议。',False,'discard_invalid_tool_output'),
    'WEB_FAILED': ('联网请求失败。','请稍后重试或联系管理员检查接口。',True,'continue_with_local_evidence_only'),
    'WEB_EMPTY': ('联网没有返回可用资料。','请调整查询范围或补充来源。',False,'report_no_public_evidence'),
    'MODEL_OOM': ('本地GPU显存不足，本次生成未完成。','请稍后重试；可减少同时分析的图片或文件。',False,'do_not_repeat_same_gpu_workload'),
    'MODEL_LOAD_FAILED': ('本地模型未能加载。','请管理员检查模型文件和运行环境。',False,'report_model_unavailable'),
    'GENERATION_TIMEOUT': ('本地回答生成达到处理时限。','已找到的来源可继续查看；稍后可重试。',True,'return_available_sources_without_inventing_answer'),
    'OUTPUT_TOKEN_LIMIT': ('回答触及输出长度上限，未能完整生成。','可分项提问；已找到的来源仍可查看。',False,'recover_complete_fields_only'),
    'OUTPUT_INVALID_JSON': ('模型输出格式不完整，本次汇总未完成。','请查看已有来源，或稍后重试。',True,'recover_complete_fields_only'),
    'OUTPUT_SCHEMA_INVALID': ('模型输出缺少必要字段。','请查看已有来源，或稍后重试。',True,'return_available_sources_without_inventing_answer'),
    'GROUNDING_FAILED': ('生成内容未通过证据或引用校验。','请核对已有资料；系统不会把未验证内容作为事实返回。',False,'omit_unsupported_claims'),
    'ANSWER_PARTIAL': ('部分生成内容缺乏数字证据，已移除；当前回答可能不完整。','请核对来源或补充缺失部分的证据。',False,'report_partial_coverage_do_not_claim_complete_answer'),
    'CONTEXT_BUDGET_EXCEEDED': ('当前输入超过本机安全处理预算。','请聚焦问题或分批分析附件。',False,'reduce_context_without_dropping_required_evidence'),
    'SERVICE_BUSY': ('本地服务正在处理其他请求。','请稍后重试。',True,'wait_for_capacity'),
    'REQUEST_TIMEOUT': ('本次请求已达到整体处理时限。','请稍后重试，或分项处理。',True,'stop_tools_and_report_partial_progress'),
    'REQUEST_CANCELLED': ('本次请求已取消。','需要时可重新提交。',False,'stop_processing'),
    'PLANNER_FAILED': ('工具规划未完成，已使用保守处理方式。','可明确所需资料或查询目标。',False,'use_safe_fallback_no_new_permissions'),
    'RETRY_POLICY_FAILED': ('自动恢复判断出现异常，已保留首轮结果。','请根据请求编号联系管理员；不要把首轮结果视为已通过完整恢复检查。',False,'preserve_first_response_no_blind_retry'),
    'INTERNAL_ERROR': ('服务内部出现异常，本次操作未完成。','请提供错误编号给管理员排查。',False,'report_failure_without_fabrication'),
}


def error_info(code, stage='request', exception_type=None):
    code = code if code in ERRORS else 'INTERNAL_ERROR'
    message, action, retryable, model_action = ERRORS[code]
    return dict(code=code, stage=stage, message=message, action=action,
                retryable=retryable, model_action=model_action, exception_type=exception_type)


def classify(exc, stage='request'):
    kind=type(exc).__name__; value=str(exc).lower()
    if value == 'unregistered_workflow_node':
        return error_info('WORKFLOW_NODE_UNREGISTERED',stage,kind)
    if getattr(exc, 'code', None) in ERRORS:
        info = error_info(exc.code, stage, kind)
        field = getattr(exc, 'field', None)
        if field in {'intent','answerable','customer_reply','citations','next_action','key_points',
                     'normalized_terms','missing_information','risk_warnings','image_observations',
                     'tools','status','supported_evidence_ids','target_document_ids','gap_ids',
                     'revised_plan','action','used_source_ids','plan','audit','text',
                     'reason','retain_source_tools','candidate_asset_ids','visual_need','source_constraint',
                     'gap_type','visual_purpose','visual_reason','web_can_resolve','schema','tables','sections',
                     'table_id','section_id','mappings','label','target','confidence','scope','source_codes',
                     'target_code','case_reference','retrieval_query','search_targets','task_type','period_year',
                     'document_type','title','text_blocks','findings','chart_type','recovery_need'}:
            info['field'] = field
            info['message'] += ' 字段：' + field + '。'
        if getattr(exc,'role',None):
            info['role']=exc.role
            info['attempts']=getattr(exc,'attempts',None)
            info['message'] += ' 阶段：' + exc.role + '；已尝试' + str(info['attempts']) + '次。'
        return info
    if kind == 'ModelCallTimeout' or value == 'model_call_timeout':
        return error_info('MODEL_CALL_TIMEOUT',stage,kind)
    if kind == 'RequestCancelled' or value == 'request_cancelled':
        return error_info('REQUEST_CANCELLED', stage, kind)
    if 'unknown_target_document_id' in value:
        return error_info('PLAN_DOCUMENT_SCOPE_INVALID', stage, kind)
    if (stage == 'document_summary' and kind not in {'RequestBudgetExceeded', 'RequestCancelled'}
            and 'outofmemory' not in kind.lower() and 'out of memory' not in value and 'cancel' not in value):
        return error_info('DOCUMENT_SUMMARY_FAILED', stage, kind)
    if stage == 'public_web_search':
        status = getattr(exc, 'http_status', None)
        provider_code = getattr(exc, 'error_code', None)
        code = ({401:'WEB_AUTH_FAILED',403:'WEB_AUTH_FAILED',429:'WEB_RATE_LIMITED'}.get(status)
                or {'network_error':'WEB_NETWORK_FAILED','timeout':'WEB_TIMEOUT',
                    'invalid_json':'WEB_RESPONSE_INVALID'}.get(provider_code))
        if not code and isinstance(status, int) and status >= 500:
            code = 'WEB_PROVIDER_ERROR'
        if code:
            return error_info(code, stage, kind)
    if 'outofmemory' in kind.lower() or 'out of memory' in value: code='MODEL_OOM'
    elif kind=='RequestBudgetExceeded': code='REQUEST_TIMEOUT'
    elif 'timeout' in kind.lower(): code='WEB_TIMEOUT' if stage=='public_web_search' else 'REQUEST_TIMEOUT'
    elif stage in {'evidence_evaluation','replan_request'} and 'audit_input_budget_exceeded' in value:
        code='REPLAN_INPUT_BUDGET_EXCEEDED' if stage=='replan_request' else 'AUDIT_INPUT_BUDGET_EXCEEDED'
    elif 'answer_review_actual_input_budget_exceeded' in value: code='CONTEXT_BUDGET_EXCEEDED'
    elif 'answer_review_admission_budget_exceeded' in value: code='CONTEXT_BUDGET_EXCEEDED'
    elif 'answer_review_admission_invariant_failed' in value: code='AUDIT_INPUT_BUDGET_EXCEEDED'
    elif 'answer_review_admission_budget_exceeded' in value: code='CONTEXT_BUDGET_EXCEEDED'
    elif 'answer_review_admission_invariant_failed' in value: code='AUDIT_INPUT_BUDGET_EXCEEDED'
    elif 'prompt_budget' in value or 'context_budget' in value: code='CONTEXT_BUDGET_EXCEEDED'
    elif stage=='evidence_evaluation' and ('reference' in value or 'evidence_id' in value): code='AUDIT_INVALID_REFERENCE'
    elif stage=='evidence_evaluation' and kind=='JSONDecodeError': code='AUDIT_INVALID_JSON'
    elif stage=='evidence_evaluation' and kind=='ValueError':
        journal = current_execution.get()
        code = 'AUDIT_INPUT_INVALID' if journal is not None and not journal.audit_invoked else 'AUDIT_SCHEMA_INVALID'
    elif stage=='evidence_evaluation': code='AUDIT_FAILED'
    elif stage=='replan_request' and kind=='ValueError': code='REPLAN_INPUT_INVALID'
    elif kind=='JSONDecodeError': code='OUTPUT_INVALID_JSON'
    elif kind=='ValueError' and str(exc) in {'模型输出中没有 JSON 对象','模型输出不是 JSON 对象','模型输出中没有可恢复的 JSON 对象','截断 JSON 尚未完整输出回答与引用'}: code='OUTPUT_INVALID_JSON'
    elif 'context_integrity_failed' in value: code='CONTEXT_INTEGRITY_FAILED'
    elif 'grounded_output' in value or 'unsupported_numeric' in value or 'unsupported_promotional' in value: code='GROUNDING_FAILED'
    elif 'general_chat_output_invalid' in value: code='OUTPUT_SCHEMA_INVALID'
    elif 'unplanned_tool' in value or 'not_authorised' in value or isinstance(exc,PermissionError): code='ACCESS_DENIED'
    elif stage=='model_load': code='MODEL_LOAD_FAILED'
    elif stage=='plan_request': code='PLANNER_FAILED'
    elif stage=='public_web_search': code='WEB_FAILED'
    elif stage=='customer_documents': code='PARSE_FAILED'
    elif stage=='company_rag': code='RETRIEVAL_FAILED'
    elif stage=='visual_inspection': code='OCR_FAILED'
    elif stage=='validate_answer': code='GROUNDING_FAILED'
    else: code='INTERNAL_ERROR'
    return error_info(code,stage,kind)


@dataclass
class ExecutionStatus:
    request_id: str = field(default_factory=lambda:uuid.uuid4().hex)
    state: str = 'received'
    stage: str = 'request'
    history: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    timings: list = field(default_factory=list)
    stage_details: list = field(default_factory=list)
    # Content-bearing parsed drafts are separate from content-free live summaries.
    answer_versions: list = field(default_factory=list)
    # Owner/admin-only parsed visual text, not part of status/list summaries.
    visual_label_versions: list = field(default_factory=list)
    summary_progress: dict = field(default_factory=dict)
    model_contracts: list = field(default_factory=list)
    audit_invoked: bool = False
    final_generation_invoked: bool = False
    agent_request_id: str | None = None
    _stage_started: float = field(default_factory=time.perf_counter)
    _request_started: float = field(default_factory=time.perf_counter)
    _finished_at: float | None = None

    def enter(self, stage):
        if self.state in {'completed','degraded','failed','cancelled'}: return
        now = time.perf_counter()
        self.timings.append({'stage':self.stage,'elapsed_ms':round((now-self._stage_started)*1000,2)})
        self._stage_started = now
        self.stage=stage
        self.state={'plan_request':'planning','guard_tools':'checking','compose_evidence':'composing',
                    'generate_answer':'generating','validate_answer':'validating'}.get(stage,'executing')
        self.history.append({'state':self.state,'stage':stage})
        self.history=self.history[-40:]

    def finish(self, success):
        if self.state in {'completed','degraded','failed','cancelled'}: return
        self._finished_at = time.perf_counter()
        self.timings.append({'stage':self.stage,'elapsed_ms':round((self._finished_at-self._stage_started)*1000,2)})
        self.state='cancelled' if any(e['code']=='REQUEST_CANCELLED' for e in self.errors) else (
            'degraded' if success and self.errors else 'completed' if success else 'failed')

    def snapshot(self):
        from backend.sales.token_usage import usage
        contracts=[]
        for contract in list(self.model_contracts):
            copied=dict(contract,attempts=[dict(a) for a in contract.get('attempts',[])])
            for attempt in copied['attempts']:
                if attempt.get('status')=='running' and attempt.get('started_unix_seconds'):
                    attempt['elapsed_seconds']=round(max(0,time.time()-attempt['started_unix_seconds']),3)
            contracts.append(copied)
        return {'request_id':self.request_id,'state':self.state,'stage':self.stage,
                'transitions':self.history,'errors':self.errors,
                'stage_details':[{**{k:v for k,v in row.items() if not k.startswith('_')},
                    'output':{**row.get('output',{}),'token_usage':usage(self.model_contracts[row.get('_contract_start',len(self.model_contracts)):])}
                        if row.get('status')=='running' else row.get('output',{}),
                    'token_usage':usage(self.model_contracts[row.get('_contract_start',len(self.model_contracts)):])
                        if row.get('status')=='running' else row.get('token_usage')}
                    for row in self.stage_details],
                'token_usage': usage(self.model_contracts),
                'summary_progress': dict(self.summary_progress),
                'model_contracts': contracts,
                'audit_invoked':self.audit_invoked,
                'final_generation_invoked':self.final_generation_invoked,
                'elapsed_ms':round(((self._finished_at or time.perf_counter())-self._request_started)*1000,2),
                'stage_timings':list(self.timings) if self._finished_at is not None else [*self.timings, {'stage':self.stage,
                    'elapsed_ms':round((time.perf_counter()-self._stage_started)*1000,2),'in_progress':True}]}


current_execution = ContextVar('execution_status',default=None)


@contextmanager
def timed_phase(name):
    started = time.perf_counter()
    try:
        yield
    finally:
        journal = current_execution.get()
        if journal:
            journal.timings.append({'phase':name,'stage':journal.stage,
                                   'elapsed_ms':round((time.perf_counter()-started)*1000,2)})


def timed_function(name):
    def decorate(function):
        @wraps(function)
        def measured(*args, **kwargs):
            with timed_phase(name):
                return function(*args, **kwargs)
        return measured
    return decorate


def report_error(exc=None, *, code=None, stage=None, details=None):
    journal=current_execution.get()
    stage=stage or (journal.stage if journal else 'request')
    info=error_info(code,stage,type(exc).__name__ if exc else None) if code else classify(exc,stage)
    if details is not None:
        info['details'] = details
    if journal and info['code'] == 'CONTEXT_INTEGRITY_FAILED':
        existing = next((e for e in journal.errors if e['code'] == info['code']), None)
        if existing is not None:
            if details is not None:
                existing['details'] = details
            return existing
    if journal and info not in journal.errors: journal.errors.append(info)
    return info


def enter_stage(stage):
    journal=current_execution.get()
    if journal: journal.enter(stage)


def model_tool_errors():
    journal=current_execution.get()
    return [{k:e[k] for k in ('code','stage','message','retryable','model_action')} for e in (journal.errors if journal else [])]


def install_error_protocol(app, log_path: Path):
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException
    from starlette.responses import JSONResponse

    def response_error(request, info, status, headers=None):
        journal=getattr(request.state,'execution',None)
        if journal:
            if info not in journal.errors: journal.errors.append(info)
            journal.finish(False)
        return JSONResponse(status_code=status,headers=headers,
            content={'detail':info['message'], 'error':info,
                     'request_id':journal.request_id if journal else None})

    @app.exception_handler(HTTPException)
    async def http_error(request,exc):
        mapping={401:'AUTH_REQUIRED',403:'ACCESS_DENIED',404:'NOT_FOUND',410:'ATTACHMENT_EXPIRED',
                 413:'FILE_TOO_LARGE',415:'UNSUPPORTED_FORMAT',422:'REQUEST_INVALID',
                 429:'SERVICE_BUSY',499:'REQUEST_CANCELLED',504:'REQUEST_TIMEOUT'}
        code=mapping.get(exc.status_code,'REQUEST_INVALID' if exc.status_code<500 else 'INTERNAL_ERROR')
        if getattr(exc, 'code', None) in ERRORS:
            code = exc.code
        detail=str(exc.detail).lower()
        if exc.status_code==400:
            if 'expired' in detail or '过期' in detail: code='ATTACHMENT_EXPIRED'
            elif 'unsupported' in detail or '不支持' in detail: code='UNSUPPORTED_FORMAT'
            elif 'parse' in detail or '解析' in detail: code='PARSE_FAILED'
        journal = getattr(request.state, 'execution', None)
        info = journal.errors[-1] if exc.status_code >= 500 and journal and journal.errors else error_info(code)
        return response_error(request,info,exc.status_code,exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request,exc):
        return response_error(request,error_info('REQUEST_INVALID'),422)

    @app.middleware('http')
    async def execution_scope(request,call_next):
        journal=ExecutionStatus(); request.state.execution=journal
        token=current_execution.set(journal)
        tracked = request.method == 'POST' and request.url.path == '/api/copilot/answer'
        if tracked:
            from backend.sales.admin_traces import register_live
            register_live(journal)
        try:
            try: response=await call_next(request)
            except Exception as exc:
                response=response_error(request,report_error(exc),500)
            journal.finish(response.status_code<400)
            response.headers['X-Request-ID']=journal.request_id
            return response
        finally:
            if tracked:
                from backend.sales.admin_traces import unregister_live
                unregister_live(journal.request_id)
            if journal.errors:
                try:
                    log_path.parent.mkdir(parents=True,exist_ok=True)
                    with log_path.open('a',encoding='utf-8') as log:
                        log.write(json.dumps({'time':time.time(),**journal.snapshot()},ensure_ascii=False)+'\n')
                except OSError: pass
            current_execution.reset(token)
