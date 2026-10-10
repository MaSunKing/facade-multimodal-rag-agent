"""Backend-owned resource state; a model describes needs, not remaining funds."""
import math
import time


RECOVERY_NEEDS = {'none', 'lookup', 'summary', 'visual', 'web', 'clarification', 'unknown'}


def budget_view(request_budget, summary_audit=None, *, tools=(), customer_visual_available=False):
    summary = getattr(request_budget, 'summary_state', None)
    audit = summary_audit or {}
    state = None
    if summary is not None:
        state = {
            'remaining_calls': max(0, summary.max_calls - summary.calls),
            'remaining_input_tokens': max(0, summary.max_input_tokens - summary.input_tokens),
            'remaining_output_tokens': max(0, summary.max_output_tokens - summary.output_tokens),
            'remaining_seconds': round(summary.remaining_seconds, 3) if math.isfinite(summary.remaining_seconds) else None,
        }
    remaining = None if request_budget is None or not request_budget.has_deadline else max(0, request_budget.deadline - time.monotonic())
    final_reserve = 0 if remaining is None else 60
    recovery_available = (remaining is None or remaining >= final_reserve + 45) and not (
        request_budget and (request_budget.cancelled.is_set() or any(
            not (r['stage'] == 'plan_request' and r['action'] == 'repair_query_language')
            for r in request_budget.recoveries)))
    summary_available = summary is None or all(v is None or v > 0 for v in state.values())
    summary_available = summary_available and not audit.get('budget_exhausted', False)
    return {'summary': state, 'summary_available': bool(summary_available),
        'lookup_available': bool(recovery_available and set(tools) & {'customer_documents', 'company_rag'}),
        'web_available': bool(recovery_available and 'public_web_search' in tools),
        'visual_available': bool(recovery_available and ('visual_inspection' in tools or
            customer_visual_available and 'customer_documents' in tools)),
        'recovery_available': bool(recovery_available), 'final_reserve_seconds': final_reserve,
        'request_remaining_seconds': None if remaining is None else round(remaining, 3),
        'authority': 'backend_only'}


def validate_budget_view(value):
    if not isinstance(value, dict) or value.get('authority') != 'backend_only':
        raise ValueError('recovery_budget_authority_invalid')
    for field in ('summary_available', 'lookup_available', 'web_available', 'visual_available', 'recovery_available'):
        if type(value.get(field)) is not bool:
            raise ValueError('recovery_budget_field_invalid:' + field)
    numbers = {'final_reserve_seconds': value.get('final_reserve_seconds')}
    if value.get('request_remaining_seconds') is not None:
        numbers['request_remaining_seconds'] = value['request_remaining_seconds']
    if value.get('summary') is not None:
        if not isinstance(value['summary'], dict): raise ValueError('summary_budget_type_invalid')
        for field in ('remaining_calls', 'remaining_input_tokens', 'remaining_output_tokens', 'remaining_seconds'):
            numbers[field] = value['summary'].get(field)
            if field=='remaining_seconds' and numbers[field] is None:
                del numbers[field]
                continue
            if field != 'remaining_seconds' and type(numbers[field]) is not int:
                raise ValueError('summary_budget_integer_invalid:' + field)
    for field, number in numbers.items():
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            raise ValueError('recovery_budget_number_invalid:' + field)
    if value['summary_available'] and value.get('summary') is not None and any(v is not None and v <= 0 for v in value['summary'].values()):
        raise ValueError('summary_budget_state_inconsistent')
    if not value['recovery_available'] and any(value[k] for k in ('lookup_available','web_available','visual_available')):
        raise ValueError('recovery_budget_state_inconsistent')
    return value


def recovery_allowed(value, need):
    validate_budget_view(value)
    if need not in RECOVERY_NEEDS: raise ValueError('recovery_need_invalid')
    if not value['recovery_available']: return False
    if need in {'none', 'clarification'}: return False
    if need == 'summary': return value['summary_available'] and value['lookup_available']
    if need == 'lookup': return value['lookup_available']
    if need == 'web': return value['web_available']
    if need == 'visual': return value['visual_available']
    # Do not ask an expensive planner to repeat an exhausted summary without
    # an explicitly identified alternative method.
    return value['summary_available'] and any(value[k] for k in ('lookup_available', 'web_available', 'visual_available'))
