"""Answer degradation after validation, independent of domain and routing."""
from __future__ import annotations
from typing import Any


def reject_stripped_numeric_answer(original: dict[str, Any], filtered: dict[str, Any],
                                   audit: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """A surviving preamble is not a supported answer; keep failure auditable."""
    result, checked = dict(filtered), dict(audit)
    before = str(original.get('customer_reply') or '').strip()
    after = str(filtered.get('customer_reply') or '').strip()
    if before != after and len(after) < max(40, int(len(before) * 0.45)):
        result.update(answerable=False, citations=[], key_points=[],
                      customer_reply='当前证据不足以核验所需数值，无法可靠完成该问题。',
                      missing_information=['所需数值及计算依据未通过证据核验'])
        checked.update(numeric_answer_removed=True, passed=False)
    return result, checked
