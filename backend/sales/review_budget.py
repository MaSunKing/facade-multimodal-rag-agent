"""CPU admission for a complete draft and lightweight review, before Final.

The draft remains natural language; neither its cap nor permission comes from
Planner. Metadata admission never drops labels or task constraints silently.
"""
from copy import deepcopy

AUDIT_INPUT_TOKEN_LIMIT = 8000
AUDIT_OUTPUT_TOKEN_LIMIT = 384
DRAFT_BOUNDARY_RESERVE = 128


def admit_actual_review(payload, count, *, retry_reserve, ceiling=AUDIT_INPUT_TOKEN_LIMIT):
    """Measure the complete actual draft; compact metadata only if needed."""
    from backend.sales.review_execution import compact_review_metadata
    value = deepcopy(payload)
    initial = count(value)
    limit = ceiling
    if initial > limit:
        value = compact_review_metadata(value)
    actual = count(value)
    without_reply = deepcopy(value)
    without_reply.setdefault('final_draft', {})['customer_reply'] = ''
    metadata = count(without_reply)
    audit = {'policy':'actual_review_input_after_draft', 'input_ceiling':ceiling,
             'initial_input_tokens':initial, 'actual_input_tokens':actual,
             'metadata_tokens':metadata, 'draft_token_delta':actual-metadata,
             'draft_characters':len(value.get('final_draft',{}).get('customer_reply','')),
             'format_retry_reserved_tokens':0,
             'format_retry_estimated_extra_tokens':retry_reserve,
             'required_input_tokens':actual,
             'retry_policy':'measure_actual_retry_only_if_needed',
             'metadata_compacted':initial > limit, 'labels_omitted':0,
             'budget_satisfied':actual <= limit}
    if not audit['budget_satisfied']:
        failure = ValueError('answer_review_actual_input_budget_exceeded')
        failure.budget_audit = audit
        raise failure
    return value, audit


def admit_review(payload, count, *, output_limit, retry_reserve,
                 ceiling=AUDIT_INPUT_TOKEN_LIMIT, strict=True):
    from backend.sales.review_execution import compact_review_metadata
    value = deepcopy(payload)
    value.setdefault('final_draft', {})['customer_reply'] = ''
    value['prepared_support_contract'] = True
    initial = count(value)
    available = ceiling - output_limit - retry_reserve - DRAFT_BOUNDARY_RESERVE
    if initial > available:
        value = compact_review_metadata(value)
    actual = count(value)
    audit = {'policy': 'review_admission_before_final', 'input_ceiling': ceiling,
             'metadata_tokens': actual, 'initial_metadata_tokens': initial,
             'draft_output_reserved_tokens': output_limit,
             'format_retry_reserved_tokens': retry_reserve,
             'boundary_reserved_tokens': DRAFT_BOUNDARY_RESERVE,
             'required_input_tokens': actual + output_limit + retry_reserve + DRAFT_BOUNDARY_RESERVE,
             'available_metadata_tokens': available,
             'budget_satisfied': actual <= available,
             'labels_omitted': 0}
    if strict and not audit['budget_satisfied']:
        failure = ValueError('answer_review_admission_budget_exceeded')
        failure.budget_audit = audit
        raise failure
    return value, audit


def fill_admitted_review(base, reply, count, admission):
    value = deepcopy(base)
    value['final_draft']['customer_reply'] = reply
    limit = admission['input_ceiling'] - admission['format_retry_reserved_tokens']
    actual = count(value)
    if actual > limit:
        # Invariant guard, not a format retry or permission to trim the answer.
        raise ValueError('answer_review_admission_invariant_failed')
    return value, actual
