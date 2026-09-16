"""Safe, request-scoped execution/error protocol shared by API and graph.

Never put exception messages, prompts, credentials or local paths in this protocol.
"""
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import time
import uuid
from pathlib import Path

# code: user message, next action, retry allowed, model recovery action
ERRORS = {
    'REQUEST_INVALID': ('请求参数或文件格式不符合要求。','请检查输入和文件格式。',False,'ask_for_valid_input'),
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
    elif kind=='JSONDecodeError': code='OUTPUT_INVALID_JSON'
    elif kind=='ValueError' and str(exc) in {'模型输出中没有 JSON 对象','模型输出不是 JSON 对象','模型输出中没有可恢复的 JSON 对象','截断 JSON 尚未完整输出回答与引用'}: code='OUTPUT_INVALID_JSON'
    elif 'prompt_budget' in value or 'context_budget' in value: code='CONTEXT_BUDGET_EXCEEDED'
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

    def enter(self, stage):
        if self.state in {'completed','degraded','failed','cancelled'}: return
        self.stage=stage
        self.state={'plan_request':'planning','guard_tools':'checking','compose_evidence':'composing',
                    'generate_answer':'generating','validate_answer':'validating'}.get(stage,'executing')
        self.history.append({'state':self.state,'stage':stage})
        self.history=self.history[-40:]

    def finish(self, success):
        if self.state in {'completed','degraded','failed','cancelled'}: return
        self.state='cancelled' if any(e['code']=='REQUEST_CANCELLED' for e in self.errors) else (
            'degraded' if success and self.errors else 'completed' if success else 'failed')

    def snapshot(self):
        return {'request_id':self.request_id,'state':self.state,'stage':self.stage,
                'transitions':self.history,'errors':self.errors}


current_execution = ContextVar('execution_status',default=None)


def report_error(exc=None, *, code=None, stage=None):
    journal=current_execution.get()
    stage=stage or (journal.stage if journal else 'request')
    info=error_info(code,stage) if code else classify(exc,stage)
    if journal and info not in journal.errors: journal.errors.append(info)
    return info


def enter_stage(stage):
    journal=current_execution.get()
    if journal: journal.enter(stage)


def model_tool_errors():
    journal=current_execution.get()
    return [{k:e[k] for k in ('code','stage','retryable','model_action')} for e in (journal.errors if journal else [])]


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
        try:
            try: response=await call_next(request)
            except Exception as exc:
                response=response_error(request,report_error(exc),500)
            journal.finish(response.status_code<400)
            response.headers['X-Request-ID']=journal.request_id
            return response
        finally:
            if journal.errors:
                try:
                    log_path.parent.mkdir(parents=True,exist_ok=True)
                    with log_path.open('a',encoding='utf-8') as log:
                        log.write(json.dumps({'time':time.time(),**journal.snapshot()},ensure_ascii=False)+'\n')
                except OSError: pass
            current_execution.reset(token)
