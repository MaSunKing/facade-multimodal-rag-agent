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
    'content_sha256', 'prepared_content_sha256', 'semantic_context_score', 'goal_rerank_applied',
    'audited_span_source_sha256', 'audited_span_selection',
    'goal_coverage_method', 'window_offset',
    'window_estimated_tokens', 'original_chunk_estimated_tokens',
    'retrieval_aspect', 'retrieval_aspects', 'ranking_text',
    'protected_goal_ids', 'protected_goal_terms',
    'shared_semantic_relevance', 'duplicate_provenance',
    'web_original_rank',
    'original_source_refs', 'citations',
    'support_audit',  # Retrieval score/heuristic telemetry, not source facts.
    'directory_expected_ids', 'directory_member_ids',
    'directory_members',  # Canonical structured members are represented by lean text.
    'directory_semantic_selection',  # Ranking telemetry stays in the trace.
    'entity_coverage_ids',  # Trusted directory scope/coverage is passed once in the envelope.
    'retrieval_requirements',  # Full repeated backend bindings; IDs stay on each row.
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
    result = {key: value for key, value in item.items() if key not in _INTERNAL}
    if result.get('coverage'):
        from backend.documents.summaries import model_coverage
        result['coverage'] = model_coverage(result['coverage'])
    if item.get('overview_content') or item.get('derived_summary'):
        result.pop('source_group', None)
        result.pop('original_chunk_id', None)
    return result


def nonempty_metadata(value):
    """Omit empty optional metadata, preserving False/0 and all source values."""
    if isinstance(value, dict):
        return {key:nonempty_metadata(item) for key,item in value.items()
                if item is not None and item != '' and item != [] and item != {}}
    if isinstance(value,list):
        return [nonempty_metadata(item) for item in value]
    return value


def model_payload_view(payload: dict[str, Any], evidence_key: str) -> dict[str, Any]:
    return {**payload, evidence_key: [model_evidence_view(item)
                                    for item in payload.get(evidence_key, [])]}
