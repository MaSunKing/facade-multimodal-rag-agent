"""Lossless source-text cleanup and a lean model wire view.

Canonical evidence and ranking telemetry stay server-side. No fact, number,
formula or condition is inferred by this module.
"""
from __future__ import annotations

import re
from typing import Any

_CELL = re.compile(r"\b([A-Z]{1,3})\d+(?:\[[^\]\n]*\])?=")
_INTERNAL = {
    'source_refs', 'goal_support_scores', 'retrieval_score', 'score', 'lexical_score',
    'semantic_reranker_score', 'source_local_rank', 'original_ordinal',
    'content_sha256', 'semantic_context_score', 'goal_rerank_applied',
    'goal_coverage_method', 'window_offset',
    'window_estimated_tokens', 'original_chunk_estimated_tokens',
    'retrieval_aspect', 'retrieval_aspects', 'ranking_text',
    'protected_goal_ids', 'protected_goal_terms',
}


def compact_source_text(text: str) -> str:
    """Prune only redundant provenance and headers of absent cell columns."""
    text = re.sub(r"\[ROW\s+source=[^\]]*\]\s*", "[ROW] ", text)
    # Block kind and native metadata can themselves carry facts (rules,
    # formulas, image properties). Remove provenance only, not the whole tag.
    text = re.sub(r"(\[BLOCK[^\]\n]*?) source=.*?(?= metadata=|\])", r"\1", text)
    columns = set(_CELL.findall(text))
    if columns:
        def narrow_header(match: re.Match) -> str:
            parts = match.group(1).split(' | ')
            kept = [part for part in parts if part.split('=', 1)[0].strip() in columns]
            return '[COLUMNS] ' + ' | '.join(kept) if kept else match.group(0)
        text = re.sub(r"(?m)^\[COLUMNS\]\s*([^\n]*)", narrow_header, text)
    return text


def model_evidence_view(item: dict[str, Any]) -> dict[str, Any]:
    """Keep source content/IDs and integrity groups, omit internal telemetry."""
    return {key: value for key, value in item.items() if key not in _INTERNAL}


def model_payload_view(payload: dict[str, Any], evidence_key: str) -> dict[str, Any]:
    return {**payload, evidence_key: [model_evidence_view(item)
                                    for item in payload.get(evidence_key, [])]}
