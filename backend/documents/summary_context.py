"""Early and final overview budgets share the same final-answer envelope."""
from backend.documents.summaries import model_coverage
from backend.sales.evidence_packing import model_evidence_view
from backend.sales.answer_coverage import coverage_contract
import hashlib
import re


FILE_RULE = '逐文件回答本轮要求，引用该文件的可见证据；未处理或缺少内容要明确说明，不得省略文件。'
SUMMARY_RULE = ('摘要不是完整性证明。processing_status表示实际处理范围，summary_status表示内容来源。'
    'partial/missing只能局部概括；table_structure_only只能介绍结构，不能推断合计、趋势或分布。'
    'structural_fallback_only/extractive_fallback_only不是成功模型摘要。'
    'visual_pixels_reviewed=false不代表看过图片。不得宣称部分证据是全文完整总结。')


def required_document_view(documents):
    return [{k: d[k] for k in ('document_id', 'document_name', 'aspect_index', 'requirement') if k in d}
            | {'parse_warnings': list((d.get('parse_quality') or {}).get('warnings') or [])[:5]}
            for d in documents]


def final_summary_payload(*, question, language, date, evidence, aspects, required,
                          conversation=None, task_memory=None, evaluation=None, document_selection='all'):
    # The audit record, source coordinates and full coverage manifest remain
    # server-side. Only the validated verdict fields constrain generation.
    verdict = {k: v for k, v in (evaluation or {}).items() if k in
        {'status', 'problem', 'missing_aspects', 'supported_evidence_ids', 'document_support', 'gap_type', 'error_code'}}
    rows = [model_evidence_view(r) for r in evidence]
    from backend.documents.document_tasks import attachment_task_table
    return {'customer_question': question,
        'attachment_task':attachment_task_table(required,evidence,operation='summary',selection=document_selection),
        'answer_language': language, 'current_date_china': date,
        'conversation_context': conversation or [], 'task_memory': task_memory or {},
        'evidence': rows, 'audited_evidence_ids': [r['evidence_id'] for r in rows],
        'evidence_evaluation': verdict, 'answer_coverage_contract': coverage_contract(aspects),
        'required_documents': required_document_view(required),
        'file_coverage_instruction': FILE_RULE, 'summary_coverage_contract': SUMMARY_RULE,
        'rules': ['只用当前证据作事实结论；客户、企业与网页来源不混淆。冲突需明确说明。'
                  '未选中内容不等于原文不存在；无法支持的维度要明确缺口。']}


def allocate_summary_context(*, audit_tokens, final_tokens, audit_limit, final_limit,
                             file_count, maximum_text_tokens=2800, additional_source_tokens=0):
    """Count empty wire envelopes before spending summary calls.

    Covers per-file IDs/coverage plus the future compact support verdict.
    Exact checks still run after packets are built. Output generation has a
    separate budget; this reserve is for input metadata, not answer tokens.
    """
    metadata_reserve = 280 * max(1, file_count)
    verdict_reserve = 512
    audit_fixed = audit_tokens([])
    final_fixed = final_tokens([])
    available = min(audit_limit - audit_fixed - metadata_reserve,
                    final_limit - final_fixed - metadata_reserve - verdict_reserve)
    available -= additional_source_tokens
    text_budget = min(maximum_text_tokens, max(0, available))
    audit = {'policy': 'allocate_before_summary_using_both_wire_envelopes',
        'audit_input_limit': audit_limit, 'final_input_limit': final_limit,
        'audit_fixed_prompt_tokens': audit_fixed, 'final_fixed_prompt_tokens': final_fixed,
        'file_metadata_reserve_tokens': metadata_reserve, 'verdict_input_reserve_tokens': verdict_reserve,
        'additional_source_reserve_tokens': additional_source_tokens,
        'evidence_text_budget': text_budget, 'target_file_count': file_count,
        'per_file_base_tokens': text_budget // max(1, file_count)}
    if text_budget < 96 * max(1, file_count):
        raise ValueError('context_budget_exceeded_before_summary')
    return text_budget, audit


def compact_overview(row, count):
    """Retain whole excerpt units before audit, never certify completeness."""
    original = str(row.get('text') or '')
    target = int(count(original) * .65)
    if target < 96: return None
    units = re.split(r'\n\n+|(?<=[。！？])', original)
    kept = []
    for unit in units:
        if unit.strip() and count('\n'.join([*kept, unit])) <= target:
            kept.append(unit)
    if not kept or len(kept) == len([u for u in units if u.strip()]): return None
    text = '本轮有限片段（因输入预算省略部分概括，非完整总结）：\n' + '\n'.join(kept)
    if count(text) >= count(original): return None
    coverage = {**(row.get('coverage') or {}), 'prompt_excerpted': True}
    return {**row, 'text':text, 'coverage':coverage,
        'prepared_content_sha256': hashlib.sha256(text.encode()).hexdigest()}
