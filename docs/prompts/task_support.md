# task_support.py完整源码快照

[源文件](../../backend/sales/task_support.py)

```python
"""Generic task support contracts. Model judgements are not factual gold."""
import re

AUDIT_OUTPUT_CONTRACT = '''输出协议（优先于省略可选字段的建议）：只输出一个JSON对象。
必须包含status和supported_evidence_ids；存在需要恢复的missing_aspects时才必须包含recovery_need。
recovery_need是字符串，只能为none、lookup、summary、visual、web、clarification、unknown。
status=supported时recovery_need只能省略或为none，missing_aspects必须省略或为空；不能同时声称全部支持又请求恢复。有实际缺口须改为partial/unsupported/uncertain，并根据最小处理选择recovery_need，无法判断用unknown。不得输出预算字段。
不要抄写示例ID，supported_evidence_ids只使用本轮输入允许的ID；无支持ID时使用空数组。
多文件任务把每个实际支持文件的ID放入supported_evidence_ids，由代码按文件分组；不必额外生成document_support。candidate_asset_ids仅用于客户视觉恢复，原样复制visual_candidates.asset_id（包含来源前缀），不能填证据ID或图名；展示图片无需输出该字段。
文字充分的结构示例：{"status":"supported","supported_evidence_ids":["T1"]}。
有限概括且需要继续摘要的结构示例：{"status":"partial","supported_evidence_ids":["T1"],"recovery_need":"summary","missing_aspects":["尚未覆盖的当前任务要求"]}。
需定位原文的结构示例：{"status":"unsupported","supported_evidence_ids":[],"recovery_need":"lookup","missing_aspects":["所问对象的具体条件"]}。
示例仅说明字段结构，不是当前任务的事实或预设审核结论。'''


AUDIT_RETRY_CONTRACT = (
    '沿用原审核协议。status仅supported/partial/unsupported/uncertain；'
    'supported_evidence_ids为允许ID数组；recovery_need必须是字符串，'
    '仅none/lookup/summary/visual/web/clarification/unknown。'
    'supported不得同时有缺口或恢复需求。'
)


def audit_execution_failed(value):
    """Technical failure is not a verdict that the evidence lacks support."""
    return (str(value.get('error_code') or '').startswith('AUDIT_')
            or value.get('error_code') == 'CONTEXT_BUDGET_EXCEEDED'
            or value.get('reason') == 'support_audit_failed')

AUDIT_PROMPT = '''你是证据支持审核器，检查当前用户问题与本轮文字证据，不生成答案/选择工具/思维链；后置轻量审核另用其协议。
用户原话与task_update全部有效要求是权威：追加保留、纠正替换冲突、明确取消才删除；历史仅补指代，旧助手/规划/审核不是事实。planned_task是原交付，execution_task是补取证，lookup恢复不改变summary/comparison交付；Requirement仅候选绑定，不新增维度。
附件按attachment_task/required_documents后端范围：all逐所选文件当前操作，点名不排除其余，explicit才限定。original_question保留原话，question为落实范围后的任务，仍核对原话对象/关系/条件/否定/时间；不混实体或增加维度。summary逐文件支持主要内容，lookup具体取证，不能一文件证明另一文件。
status=supported/partial/unsupported/uncertain，分别全部/部分/无支持/无法确定。多对象多维度分别判断，比较不能只支持一方。supported_evidence_ids只列真实支持片段，不认证整篇正文；共同列出各支持文件ID，程序分组，不输出document_support。相关词/排序/抓取/候选覆盖不是语义支持，缺候选不证明资料没有。
company_capabilities仅六类能力地图；actual_execution_scope对照原任务，明确错类/对象才scope_mismatch，可suggested_categories但不保证存在或授权。company_directory_scope/family_filter/target_names限定集合，不加入关联介绍其他系列。ambiguous不伪装唯一对象，missing_target_names仅本次缺代表。
directory_collection支持登记名称/属性，directory_aggregate支持CPU同粒度登记统计，核对分组/未知桶/覆盖，不当真实全量。列举不额外要求参数/全文，施工/技术/条款解释需正文。summary允许有限介绍，未读全文不新增缺口。
derived_summary/coverage/summary_status限制结论：摘要只支持保留事实，结构概况仅用途/结构，不证明缺失数值/合计/趋势；零成功摘要包不表示已概括正文，partial/prompt_excerpted不认证全文。
所有来源核对实体、指标、单位、条件、否定、版本及时间；宣传不是独立共识，排名/极值需范围比较。current_date_china解释相对日期，历史数据不能支持当前结论；发布日期/抓取日期/source_date/mentioned_dates不是事实日期。区间起始日不代表全部记录，核对目标日期对应行和值及表头，不能因开始于前日排除或区间包含就通过，省略年份须正文依据。
混合日期/对象/条件ID用可选support_spans映射（真实ID对应连续原文数组）限定支持，保留表头/条件，不改写拼接，没有适用片段不选ID。同对象/指标/范围冲突可conflict_groups，双方ID均保留；不同时间/版本/条件不当冲突，第二轮不自动更可信。
审核只看文字。attachment_visuals仅客户清单/附近文字/解析状态，display_images仅展示；客户visual_description只支持自身已识别可见事实，OCR只支持文字，元数据只支持登记身份/描述，不移植公司标签到客户。公司永远标签+关联文字，禁止重看原图；不足不否定已有文字；company_only时visual_need=not_required、visual_purpose=display_only，不visual恢复。
需理解客户颜色/纹理/布局/连接关系而无描述时，选允许customer候选ID，recovery_need=visual。仅展示不识图、已有描述不重看；失败/预算不足不证明原图无信息，不否定其他支持。用真实evidence_id，不用asset_id冒充。
恢复后重新判断全部当前证据，旧缺口不权威不照抄，有效旧证据保留，新轮次不自动覆盖。missing_aspects最多2项，仅真实尚未支持要求，写实体/关系/维度/条件，不混旧ID/免责声明，不断言全库没有。
recovery_budget可信、不改写，不改变支持判断；摘要耗尽不证明原文不存在或lookup不可用。有缺口选最小recovery_need=lookup/summary/visual/web/clarification/unknown，充分none；web_can_resolve仅可能帮助，不授权/保证或代替私有资料。稳定闲聊/写作可无需库证据，陌生/私有/实时不用记忆猜。
只输出精简JSON，必填status、supported_evidence_ids，supported/partial至少一条支持ID；可选support_spans、conflict_groups、missing_aspects、recovery_need、reason、problem、suggested_categories、visual_need(required/not_required/uncertain)、candidate_asset_ids、visual_reason、visual_purpose(display_only/content_inspection/label_dispute/unreadable/uncertain)、source_constraint(evidence_only/general_knowledge_allowed/uncertain)、gap_type(none/current_public_fact/public_fact/local_evidence/ambiguous_request/visual_input/answer_mismatch/scope_mismatch/unknown)、web_can_resolve。解释各一句，不输出预算/error_code/revised_plan/替换查询。''' + '\n' + AUDIT_OUTPUT_CONTRACT

REPLAN_PROMPT = '''你是有限恢复规划器，只输出JSON，不改变用户原任务，最多一轮工具恢复；协议"action":"execute|stop"。
核对原问题、task_update有效要求、actual scope及proof，审核意见不是权威。保留有效目标、维度、附件范围、来源/图片需求，纠正仅替换冲突，不恢复取消内容；previous_plan为原交付、execution_task为上次动作，lookup补搜不改summary/comparison交付。
有明确缺口、新可执行动作和预算才execute，否则stop，不因答案短或库可能有资料重跑。按能力地图/权限选最小动作，可纠正来源/类别；即使审核错误给出 web_can_resolve=false也核对proof，不盲从且不无依据补搜。
company_rag查六类，customer_documents查现有附件，public_web_search须授权且相关，不是固定兜底，不能替代私有项目事实或发送私有内容。公司图只补标签/关联文字、永不看像素；客户视觉用customer_documents、document_visual_required=true及lookup，不重跑全文摘要。有客户附件可用时revised_plan必填document_visual_required布尔值。
目录列举/统计summary+collection.list/count，正文缺口lookup；summary_available=false/do_not_repeat_summary=true不重复附件summary，预算只由recovery_budget决定，耗尽不证明原文没有；attempted_operations同动作不执行。
queries逐要求完整独立，保留实体/关系/条件/事实日期，不拼重复查询。日期来自原任务/current_date_china，不取旧网页日期；企业用语料语言、附件用内容语言、Web用问题语言，历史仅补指代，保留有效证据。
外层仅action、gap_ids、reason、retain_source_tools、revised_plan；execute复制真实gap_ids，revised_plan为首次Planner契约对象不是数组，stop为null。retain_source_tools仅仍相关成功本地来源；reason一句，不编ID/SQL/预算/status；categories仅企业使用，企业summary明确collection，source_operations仅混合操作。'''

CHAT_SELF_CHECK = '''
当前问题是唯一主任务；历史只辅助明确指代和仍适用的用户条件，不得把旧话题强加给新问题。
不要把陌生或含糊的短语擅自归属于历史中的实体；无法确定时澄清，不编造身份。
额外输出 task_support: {"status":"supported|partial|uncertain|unsupported",
"needs_verification":false,"ambiguous":false,"reason":"简短风险原因"}。
此字段只是风险自检，不是事实认证。含义不明、归属不确定、看不清图像或需要核实的冷门/实时事实，
必须标 uncertain 并说明具体疑点。没有执行搜索，不得说已查到、未找到或搜索不到公开资料。
文字与图片同时存在时结合读取，区分文字/OCR来源、直接可见信息和推断；不能凭外观推断隐藏性能。
'''


def candidate_risk(parsed, searched=False):
    check = parsed.get('task_support')
    reasons = []
    if not isinstance(check, dict):
        reasons.append('missing_candidate_support_check')
    elif (check.get('status') not in {'supported', 'partial'} or
          check.get('needs_verification') is True or check.get('ambiguous') is True):
        reasons.append(str(check.get('reason') or 'candidate_support_uncertain')[:240])
    reply = str(parsed.get('customer_reply') or '')
    if not searched and re.search(r'(?:未找到|没有找到|查到|搜索到|检索到).{0,12}(?:公开资料|网页|网上|网络)|(?:网上|联网|搜索).{0,10}(?:发现|找到|显示)', reply):
        reasons.append('SEARCH_CLAIM_WITHOUT_SEARCH')
    return reasons


def audit_time_available(budget, minimum=12):
    import time
    return budget is None or (not budget.expired() and budget.deadline-time.monotonic() >= minimum)


def supported_web_sources(sources, audit):
    """An evaluated source is not automatically an approved generation source."""
    if audit.get('status') not in {'supported', 'partial'}:
        return sources
    ids = set(audit.get('supported_evidence_ids') or [])
    return [bounded_support(source, audit, 'source_id', 'excerpt') for source in sources if source.get('source_id') in ids]


def supported_evidence_rows(rows, audit):
    """All source types share the same audited generation allowlist."""
    if audit.get('status') not in {'supported', 'partial'}:
        return []
    ids = set(audit.get('supported_evidence_ids') or [])
    return [bounded_support(row, audit, 'evidence_id', 'text') for row in rows if row.get('evidence_id') in ids]


def bounded_support(row, audit, id_key, text_key):
    spans = audit.get('support_spans', {}).get(row.get(id_key))
    if not spans:
        return row
    # Membership was checked against the immutable audit input. Never synthesize
    # dates, values or a new source identity; provenance remains on the row.
    import hashlib
    text='\n[…]\n'.join(spans)
    selected={**row, text_key:text, 'audited_span_selection':True}
    if text_key=='text':
        selected['audited_span_source_sha256']=row.get('prepared_content_sha256') or hashlib.sha256(str(row.get('text') or '').encode()).hexdigest()
        selected['prepared_content_sha256']=hashlib.sha256(text.encode()).hexdigest()
    return selected


def unavailable_web_response(meta, language):
    status = meta.get('status')
    codes = {'quota_exhausted':'WEB_QUOTA_EXHAUSTED','not_configured':'WEB_NOT_CONFIGURED',
             'timeout':'WEB_TIMEOUT','failed':'WEB_FAILED','error':'WEB_FAILED'}
    code = meta.get('error_code') or codes.get(status)
    if not code or not str(code).startswith('WEB_') or code == 'WEB_EMPTY':
        return None
    english = language == 'English'
    quota = code == 'WEB_QUOTA_EXHAUSTED'
    reply = ('The search quota is exhausted; this request could not be verified online.' if quota else
             'Online verification is unavailable for this request.') if english else (
             '联网查询预算已耗尽，本次未能完成公开资料核验。' if quota else '联网工具执行失败，本次未能完成公开资料核验。')
    reply += (' This does not mean the requested information does not exist.' if english else '这不表示所问信息不存在。')
    return {'intent':'unknown','answerable':False,'customer_reply':reply,'normalized_terms':[],
            'key_points':[],'citations':[],'missing_information':['Online verification' if english else '所问公开资料的实际核验'],
            'risk_warnings':[],'next_action':'Retry when search is available, or provide sources.' if english else '可在联网恢复后重试，或提供可核验来源。',
            'image_observations':[],'visual_assets':[],'online_sources':[],
            'retrieval':{'result_count':0,'supporting_results':[],'visual_count':0},
            'meta':{'model_used':False,'mode':'web_tool_unavailable','online_search':meta,
                    'error':{'code':code,'stage':'public_web_search','retryable':not quota}}}


FACT_MEANING_POLICY = '''
组织回答必须保留证据中数值的指标含义、对象、时间、范围、单位和适用条件。
不得把极值改成当前值，把预测改成实测，把局部记录扩大成整体结论，
把某型号性能推广至全部产品，或把厂家宣传改成独立验证结论。
current_date_china 只用于解释问题的相对时间，不把历史记录自动变成当前事实。
conflict_groups列出的来源用于说明差异，不能因为排序靠前或补搜轮次更晚就选择一方宣称已确认。
页面发布日期、正文事实时间和统计截止时间不同；事实时间不明确就明确限定，不补造。
所有来源均按正文目标日期对应记录回答，不把页面更新日、抓取日或区间起始日当成该记录日期。区间覆盖目标日但未给对应事实值时仍须说明缺口；存在对应行时不能仅因页面日期较旧或区间起始较早就宣称不存在。
证据被审核选中只代表含有支持片段，不代表整篇所有内容均适用。
只使用对应当前任务的片段；不得混用不同对象、日期或范围的数据。
evidence_evaluation.missing_aspects是仍未支持的具体要求：不得给这些要求填写猜测值后再附免责声明。partial时只回答真正已支持部分，并明确未支持部分。support_spans限制对应ID的可用事实范围，reason说明的时间、版本及条件限制不能丢失。
资料明确给出观测对象或截至时间时，回答相关数值须保留这些必要限定。
无法确定指标含义或时效时说明不确定，不通过改写丢掉限制，不擅自推断缺失关系。
'''
```
