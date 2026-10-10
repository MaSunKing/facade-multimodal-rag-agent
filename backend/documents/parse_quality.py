"""Cheap attachment capability diagnostics, not semantic support or routing."""
from __future__ import annotations

from typing import Any


def summarize_parse_quality(document: Any) -> dict[str, Any]:
    chunks = [c for c in document.chunks if c.get('kind') != 'document_index']
    text = '\n'.join(str(c.get('text') or '') for c in chunks)
    coverage = document.visual_coverage or {}
    visuals = document.visuals
    ready = [v for v in visuals if v.image_bytes]
    native_pages = coverage.get('native_text_indexed_page_numbers', [])
    pending = coverage.get('ocr_index_pending_page_numbers', [])
    failures = coverage.get('ocr_failure_codes', [])
    confidences = [s.get('extraction_confidence') for s in document.source_locations.values()
                   if isinstance(s.get('extraction_confidence'), (int, float))]
    low_confidence = sum(c < 0.7 for c in confidences)
    warnings: list[str] = []
    if not text.strip():
        warnings.append('no_extracted_answer_text')
    if coverage.get('coverage_complete') is False:
        warnings.append('incomplete_page_rendering')
    if pending:
        warnings.append('ocr_pages_pending')
    if failures:
        warnings.append('ocr_failed')
    if len(ready) < len(visuals):
        warnings.append('some_visual_pixels_unavailable')
    if low_confidence:
        warnings.append('low_extraction_confidence')
    if '\ufffd' in text:
        warnings.append('text_replacement_characters')
    ocr_count = sum(bool(v.metadata.get('ocr_literal_text')) for v in visuals)
    if ocr_count:
        warnings.append('ocr_candidates_unverified')
    return {
        'document_id': document.document_id,
        'text_available': bool(text.strip()),
        'answer_chunk_count': len(chunks),
        'text_characters': len(text),
        'structured_row_count': sum(s.get('row_index') is not None
                                    for s in document.source_locations.values()),
        'visual_count': len(visuals),
        'ready_visual_count': len(ready),
        'visual_samples': [{'visual_id': v.visual_id, 'kind': v.kind,
                            'page_number': v.source.get('page_number')}
                           for v in ready[:8]],
        'page_count': coverage.get('page_count'),
        'native_text_page_count': len(native_pages),
        'ocr_candidate_visual_count': ocr_count,
        'ocr_pending_page_count': len(pending),
        'ocr_pending_page_samples': pending[:8],
        'rendering_complete': coverage.get('coverage_complete'),
        'next_page_start': coverage.get('next_page_start'),
        'coverage_stop_reason': coverage.get('stop_reason'),
        'low_confidence_location_count': low_confidence,
        'warnings': warnings,
        'scope': 'parse_capabilities_only_not_question_support',
    }
