"""Exact pre-generation packing: the admitted wire is the generation wire."""
from copy import deepcopy
import hashlib


def preflight_final_input(rows, *, build_payload, render, ceiling, review_tokens=None, review_limit=8000, **packing):
    """Count complete system+chat input; freeze it before any generation.

    render must be the same serializer/tokenizer used by the source composer.
    No guessed reserve, model call or second evidence selection is involved.
    """
    from backend.sales.context_engine import pack_task_context
    original = deepcopy(rows)
    # A list is a directory projection, not a request for every stored property.
    # Perform this lossless membership projection even when the full envelope
    # fits: otherwise incidental catalogue attributes encourage long repetition.
    directory_projection = packing.get('collection_action') == 'list' and not packing.get('goals')
    if directory_projection:
        from backend.sales.directory_evidence import compact_directory_packets
        original = compact_directory_packets(original, packing.get('question',''), packing.get('goals',()))

    def measured(selected):
        # An unbounded render only projects wire metadata; it cannot truncate.
        return render(build_payload(selected), 2**31 - 1)

    initial = measured(original)[1]['actual_prompt_tokens']
    selected, packing_audit = pack_task_context(original,
        audit_tokens=review_tokens or (lambda _: 0), audit_limit=review_limit,
        generation_tokens=lambda items: measured(items)[1]['actual_prompt_tokens'],
        ceiling=ceiling, **packing)
    text, wire_audit = measured(selected)
    actual = wire_audit['actual_prompt_tokens']
    review_actual = review_tokens(selected) if review_tokens else 0
    satisfied = actual <= ceiling and review_actual <= review_limit
    packing_audit.update(initial_complete_final_prompt_tokens=initial,
        initial_budget_satisfied=initial <= ceiling,
        actual_prompt_tokens=actual, budget_satisfied=satisfied,
        input_policy='complete_final_envelope_frozen_before_generation',
        format_retry_reserved_tokens=0)
    packing_audit.update(audit_prompt_tokens=review_actual, audit_input_limit=review_limit,
        fits_audit=review_actual <= review_limit, fits_final=actual <= ceiling,
        joint_budget_policy=('final_envelope_only_review_measured_after_draft' if review_tokens is None
                             else 'final_envelope_and_review_metadata_plus_full_draft_reserve'),
        blocking_budget=('final_input' if actual > ceiling else 'answer_review' if review_actual > review_limit else None))
    packing_audit['directory_projection'] = 'requested_attributes_only' if directory_projection else 'unchanged'
    wire_audit.update(max_prompt_tokens=ceiling, budget_satisfied=satisfied,
        preflight_prompt_tokens=actual,
        final_payload_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
        input_policy=packing_audit['input_policy'],
        support_budget_preflight=packing_audit)
    return selected, text, wire_audit, packing_audit
