"""Lossless navigation adapter; never infer a chapter that was not parsed."""
from __future__ import annotations
import hashlib
import json


def stable_id(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   default=str).encode()).hexdigest()[:24]


def build_hierarchy(document_id, name, chunks, locations=None, *, source_origin='customer'):
    sections = {}
    for ordinal, chunk in enumerate(chunks):
        refs = chunk.get('source_refs') or []
        pointers = [dict(ref) if isinstance(ref, dict) else (locations or {}).get(ref, {}) for ref in refs]
        pointer = next((p for p in pointers if p), {})
        sheet = pointer.get('sheet_name') or pointer.get('source_sheet')
        heading = pointer.get('section_title') or pointer.get('section_heading')
        page = pointer.get('page_number') or pointer.get('source_page') or pointer.get('page')
        label = sheet or heading or (f'page:{page}' if page is not None else None)
        # No chapter metadata: the native parser block is the fallback unit.
        label = label or str(chunk.get('chunk_id') or chunk.get('id') or f'block:{ordinal}')
        key = stable_id([document_id, label])
        section = sections.setdefault(key, {'section_id': key, 'document_id': document_id,
            'label': str(label), 'kind': 'sheet' if sheet else 'section' if heading else 'page' if page is not None else 'block',
            'evidence_ids': [], 'source_refs': []})
        evidence_id = str(chunk.get('chunk_id') or chunk.get('id') or f'block:{ordinal}')
        section['evidence_ids'].append(evidence_id)
        for ref in refs:
            if ref not in section['source_refs']:
                section['source_refs'].append(ref)
    return {'schema_version': 'document-hierarchy-v1', 'document_id': document_id,
            'document_name': name, 'source_origin': source_origin,
            'sections': list(sections.values()), 'role': 'navigation_not_factual_summary'}
