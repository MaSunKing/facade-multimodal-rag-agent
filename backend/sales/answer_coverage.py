"""Auditable answer checklist and normative-claim guard, without another LLM.

Literal coverage is a contract check, not semantic accuracy. Evidence existence
and a quoted answer span must both be checked; model self-report is insufficient.
"""
from __future__ import annotations
import re
from typing import Any

_NORM = re.compile(r'符合(?:[\w\u4e00-\u9fff]{0,12})?(?:规范|标准)|满足(?:[\w\u4e00-\u9fff]{0,12})?(?:标准|规范)|'
                   r'\b(?:compliant|complies with|meets (?:the )?(?:standards?|requirements?))\b', re.I)
_STANDARD = re.compile(r'\b(?:GB(?:/T)?|JGJ(?:/T)?|ASTM|ISO|EN|BS)\s*[-:]?\s*\d{2,}|'
                       r'标准编号|规范编号', re.I)


def canonicalise_known_citation_ids(result: dict, evidence: dict) -> dict:
    """Wrap exact known IDs without guessing or changing a cited source."""
    output = dict(result)
    citations = []
    for citation in result.get('citations') or []:
        if isinstance(citation, str):
            if citation not in evidence:
                raise ValueError('citation_id_not_in_model_input')
            citations.append(dict(evidence_id=citation))
        else:
            citations.append(citation)
    output['citations'] = citations
    return output


def coverage_contract(aspects: list[str]) -> dict[str, Any]:
    return dict(requested_aspects=aspects, instruction=(
        'Address each requested aspect separately. Distinguish source facts from conditional proposals. '
        'Use brief numbered answer sections, not merely a file introduction. Missing measurements prevent '
        'a numeric conclusion, not an explanation of required inputs or a clearly labelled conditional method. '
        'Logos and background illustrations must not replace the requested content analysis. '
        'If evidence is missing, explicitly state that limitation rather than omit the aspect. '
        'When requested_aspects is nonempty also output answer_aspect_coverage: '
        'a list of {aspect_index (0-based), status (answered|evidence_missing), '
        'evidence_ids (actual cited IDs), answer_quote (exact 3-8 word span in customer_reply)}. '
        'Prioritise customer_reply, citations and this checklist before optional key_points/image_observations. '
        'Do not assert compliance with a standard unless a cited standard and its applicable requirements establish it.'))


def audit_answer_coverage(result: dict[str, Any], aspects: list[str], evidence: dict[str, dict]) -> dict[str, Any]:
    reply = str(result.get('customer_reply') or '')
    cited = {str(c.get('evidence_id')) for c in result.get('citations', []) if isinstance(c, dict)}
    covered, missing, invalid = [], [], []
    rows = result.get('answer_aspect_coverage') or []
    rows = rows if isinstance(rows, list) else []
    for index, aspect in enumerate(aspects):
        valid = False
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get('aspect_index'), int) or isinstance(row.get('aspect_index'), bool) or row.get('aspect_index') != index:
                continue
            quote = str(row.get('answer_quote') or '').strip()
            ids = row.get('evidence_ids') or []
            ids = ids if isinstance(ids, list) else []
            actual = bool(ids and all(isinstance(i, str) and i in evidence and i in cited for i in ids))
            status = row.get('status')
            # Missing evidence must be stated in the visible answer, but is
            # not counted as a supported answer to that requested dimension.
            if quote and quote in reply and status == 'evidence_missing':
                missing.append(index); valid = True; break
            if quote and quote in reply and status == 'answered' and actual:
                covered.append(index); valid = True; break
        if not valid:
            invalid.append(index)
    return dict(requested_aspects=aspects, covered_aspect_indices=covered,
        evidence_missing_aspect_indices=missing, unaccounted_aspect_indices=invalid,
        complete=not invalid, method='citation_and_exact_answer_span_contract_not_semantic_gold')


def guard_normative_claims(result: dict[str, Any], evidence: dict[str, dict]) -> tuple[dict[str, Any], dict[str, Any]]:
    cited = {str(c.get('evidence_id')) for c in result.get('citations', []) if isinstance(c, dict)}
    standards = [i for i in cited if i in evidence and (
        _STANDARD.search(str(evidence[i].get('text') or ''))
        or evidence[i].get('source_authority') in {'T1_standard', 'T1_national_standard'})]
    removed = []
    def clean(value: str) -> str:
        # Never turn a negative/conditional statement into affirmative advice.
        sentences = re.split(r'(?<=[。！？!?\n])|(?<=\.)\s+', value)
        kept = []
        for sentence in sentences:
            claim_supported = any(
                re.sub(r'\s+', '', sentence).strip() in re.sub(r'\s+', '', str(evidence[i].get('text') or ''))
                for i in standards)
            if _NORM.search(sentence) and not claim_supported and not re.search(
                r'不(?:能|应|代表|构成|保证)|未(?:核验|确认)|无法|需(?:核验|确认)|'
                r'\b(?:not|cannot|unverified|verify|no evidence)\b', sentence, re.I):
                removed.append(sentence.strip())
            else:
                kept.append(sentence)
        return ''.join(kept).strip()
    output = dict(result)
    for field in ('customer_reply', 'next_action'):
        output[field] = clean(str(output.get(field) or ''))
    for field in ('key_points', 'image_observations'):
        output[field] = [clean(str(v)) for v in output.get(field, []) if clean(str(v))]
    if removed:
        output['risk_warnings'] = [*output.get('risk_warnings', []), '规范符合性未获明确标准证据支持，相关肯定结论已移除。']
    if removed and not output.get('customer_reply'):
        output.update(answerable=False, customer_reply='当前证据不足以确认规范符合性。')
    return output, dict(removed_claims=list(dict.fromkeys(removed)), cited_standard_evidence_ids=standards,
        standard_presence_is_not_compliance_verdict=True)
