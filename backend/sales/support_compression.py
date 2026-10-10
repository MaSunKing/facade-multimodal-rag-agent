"""CPU-only extractive support snapshots. Never rewrite facts or inspect pixels."""
import copy
import hashlib
import json
import re
from backend.sales.evidence_gate import terms


def compress_support(rows, question, goals=(), *, max_chars=1000):
    result, decisions, duplicates = [], [], {}
    queries = [terms(q) for q in dict.fromkeys([question, *goals]) if terms(q)]
    for original in rows:
        row = copy.deepcopy(original)
        text = str(row.get('text') or '')
        # Tables, overview summaries, conflicts and visual observations are
        # indivisible here. Their upstream processors own structural coverage.
        group = str(row.get('packing_group_id') or '')
        protected = bool(row.get('visual_description') or row.get('source_type') == 'visual'
            or row.get('directory_collection')
            or row.get('derived_summary') or row.get('overview_content')
            or row.get('protected_relation_types') or group.startswith(('conflict_', 'relation_'))
            or re.search(r'(?m)^\s*(?:\[ROW|\[COLUMNS|\[BLOCK|\|)', text)
            or row.get('table') or row.get('structured_rows'))
        # Identical source span, identity and text only; never fuzzy-merge facts
        # across different sources or remove one side of a protected relation.
        identity = (row.get('source_type'), row.get('document_id'), row.get('original_chunk_id'),
                    json.dumps(row.get('source_refs') or [],sort_keys=True), row.get('facts_eligible'),
                    row.get('claim_scope'),row.get('evidence_scope'), text)
        if not protected and row.get('document_id') and row.get('original_chunk_id'):
            if identity in duplicates:
                prior = duplicates[identity]
                for field in ('protected_goal_ids','protected_goal_terms','citations'):
                    if field in row:
                        prior[field] = list(prior.get(field) or [])
                        prior[field].extend(v for v in (row[field] or []) if v not in prior[field])
                if row.get('required_document_coverage'):
                    prior['required_document_coverage'] = True
                decisions.append({'evidence_id':row['evidence_id'],'reason':'identical_source_span_duplicate',
                                  'duplicate_of':prior['evidence_id'],'original_chars':len(text),'selected_chars':0})
                continue
            duplicates[identity] = row
        spans = [(m.start(), m.end(), m.group()) for m in re.finditer(r'[^\n]+(?:\n|$)', text)]
        if len(spans) <= 2 and len(text) > max_chars:
            # Complete sentences, not arbitrary character prefixes; decimal
            # points inside values are not sentence boundaries.
            spans = [(m.start(),m.end(),m.group()) for m in re.finditer(
                r'.+?(?:[。！？]|[.!?](?=\s|$)|\n+|$)',text,re.S)]
        kept = list(range(len(spans)))
        reason = 'protected_structure_or_visual' if protected else 'short_or_no_safe_boundary'
        if not protected and len(text) > max_chars and len(spans) > 2 and queries:
            scores = [max((len(q & terms(p[2])) for q in queries), default=0) for p in spans]
            selected = set()
            for query in queries:
                hits = [len(query & terms(p[2])) for p in spans]
                best = max(hits, default=0)
                if best:
                    i = max(range(len(spans)), key=lambda n:hits[n])
                    selected.update(range(max(0,i-1), min(len(spans),i+2)))
            if selected:
                # Keep the whole matching paragraph/sentence and neighbours:
                # numbers/units/qualifiers are never sliced within that unit.
                # Unrelated numeric paragraphs are not globally mandatory.
                used = sum(len(spans[i][2]) for i in selected)
                for i in sorted(range(len(spans)), key=lambda n:(-scores[n],n)):
                    if i not in selected and scores[i] and used + len(spans[i][2]) <= max_chars:
                        selected.add(i); used += len(spans[i][2])
                kept = sorted(selected)
                reason = 'question_windows_with_qualifier_paragraphs'
        # Repeated prose can be repeated events (logs/transactions), not noise.
        # Remove repetitions only with an explicit upstream boilerplate mark.
        if not protected and row.get('repeat_is_boilerplate') is True:
            seen = set()
            unique = []
            for i in kept:
                key = spans[i][2].strip()
                if key not in seen:
                    seen.add(key); unique.append(i)
            kept = unique
        chosen = ''.join(spans[i][2] if not n or i == kept[n-1]+1 else '\n[... source excerpt ...]\n'+spans[i][2]
                         for n,i in enumerate(kept)) if spans else text
        if chosen != text:
            row['text'] = chosen
            row['preview_truncated'] = True
            row['support_excerpt'] = True
        row['prepared_content_sha256'] = hashlib.sha256(str(row.get('text') or '').encode()).hexdigest()
        decisions.append({'evidence_id':row['evidence_id'], 'reason':reason,
            'original_chars':len(text), 'selected_chars':len(str(row.get('text') or '')),
            'source_spans':[[spans[i][0],spans[i][1]] for i in kept],
            'omission_is_not_source_absence':True})
        result.append(row)
    return result, {'policy':'cpu_extract_complete_paragraphs_v1', 'decisions':decisions,
        'original_chars':sum(len(str(r.get('text') or '')) for r in rows),
        'selected_chars':sum(len(str(r.get('text') or '')) for r in result), 'model_calls':0}
