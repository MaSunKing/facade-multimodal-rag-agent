"""Deterministic output repair and hard checks; no truth/number matching."""
from copy import deepcopy
import json
import re


def program_image_presentation(response):
    """Remove only known delivery-asset Markdown links before audit/publication.

    Preserve image reasoning, captions, Context and server-selected assets.
    Code examples and ordinary links are not presentation artifacts.
    """
    if response.get('meta', {}).get('final_output_format') != 'natural_language':
        return response
    assets = response.get('visual_assets', [])
    targets = {str(a[k]) for a in assets if isinstance(a, dict)
               for k in ('asset_id', 'visual_id', 'evidence_id', 'url', 'image_url', 'path')
               if a.get(k)}
    if not targets:
        return response
    removed = 0
    def link(match):
        nonlocal removed
        target = match.group(2).strip().strip('<>')
        if target not in targets:
            return match.group(0)
        removed += 1
        return match.group(1)  # Keep descriptive words, not a broken link.
    body = response.get('customer_reply', '')
    pieces = re.split(r'(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`)', body)
    for index in range(0, len(pieces), 2):
        pieces[index] = re.sub(r'!?\[([^\]\n]*)\]\(([^)\n]*)\)', link, pieces[index])
    if not removed:
        return response
    return {**response, 'customer_reply': ''.join(pieces),
            'meta': {**response['meta'], 'image_presentation': {
                'renderer': 'backend_assets', 'removed_delivery_links': removed,
                'image_labels_retained_in_context': True}}}


class OutputContractError(ValueError):
    def __init__(self, code, field=None):
        self.code = code
        self.field = field
        super().__init__(code)


def natural_language_answer(raw, evidence_ids=()):
    """Backend envelope for an unpublished text draft; no JSON interpretation.

    answerable=True is provisional compatibility state, NOT model judgment.
    Post-draft review determines the published state. Even requested JSON/code
    is answer content here and cannot set IDs, permissions or envelope fields.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise OutputContractError('OUTPUT_REPLY_EMPTY', 'customer_reply')
    return {'intent': 'unknown', 'normalized_terms': [], 'answerable': True,
        'customer_reply': raw.strip(), 'key_points': [],
        'citations': [{'evidence_id': eid} for eid in sorted(set(evidence_ids))],
        'missing_information': [], 'risk_warnings': [], 'next_action': '',
        'image_observations': []}


def reviewed_answer_state(response):
    """Map accepted coverage to API state, never manufacture support claims."""
    meta = response.get('meta', {})
    if meta.get('final_output_format') != 'natural_language':
        return response
    review = meta.get('answer_review', {})
    if review.get('decision') != 'pass':
        return response  # Recovery/withholding owns rejected drafts.
    status = review.get('coverage_status', review.get('status'))
    answerable = status in {'complete', 'partial', 'supported'}
    return {**response, 'answerable': answerable,
        'citations': response.get('citations', []) if answerable else [],
        'meta': {**meta, 'answer_state_basis': 'post_draft_task_review_not_fact_verification',
                 'draft_state': 'reviewed'}}


def parse_or_repair(raw, parse, recover, *, hit_output_limit=False):
    """Recover only complete JSON fields, without invoking any model."""
    try:
        return parse(raw), False
    except (ValueError, json.JSONDecodeError):
        try:
            return recover(raw), True
        except (ValueError, json.JSONDecodeError) as exc:
            code = 'OUTPUT_TOKEN_LIMIT' if hit_output_limit else 'OUTPUT_INVALID_JSON'
            if not hit_output_limit and str(raw).strip().startswith('['):
                code = 'OUTPUT_NOT_OBJECT'
            raise OutputContractError(code) from exc


def repair_and_validate(value, evidence_ids, intents, *, allow_image_only=False, program_references=False):
    if not isinstance(value, dict):
        raise OutputContractError('OUTPUT_NOT_OBJECT')
    value = deepcopy(value)
    if program_references:
        # References describe the actual generation Context, NOT claim support.
        # Ignore legacy model selection; identity and scope are backend owned.
        value['citations'] = ([{'evidence_id': eid} for eid in sorted(set(evidence_ids))]
                             if value.get('answerable') is True else [])
    required = {'intent', 'normalized_terms', 'answerable', 'customer_reply', 'key_points',
                'citations', 'missing_information', 'risk_warnings', 'next_action', 'image_observations'}
    # Optional presentation containers may safely default to empty. Core content
    # and citation selection must never be guessed or copied from another source.
    for field in ('normalized_terms', 'key_points', 'missing_information', 'risk_warnings', 'image_observations'):
        value.setdefault(field, [])
    value.setdefault('next_action', '')
    value.setdefault('intent', 'unknown')
    for field in sorted(required):
        if field not in value:
            raise OutputContractError('OUTPUT_FIELD_MISSING', field)
    if set(value) - required - {'answer_aspect_coverage'}:
        raise OutputContractError('OUTPUT_FIELD_UNEXPECTED')
    for field in ('customer_reply', 'next_action', 'intent'):
        if not isinstance(value[field], str):
            raise OutputContractError('OUTPUT_FIELD_TYPE_INVALID', field)
    if not isinstance(value['answerable'], bool):
        raise OutputContractError('OUTPUT_FIELD_TYPE_INVALID', 'answerable')
    if value['intent'] not in intents:
        raise OutputContractError('OUTPUT_FIELD_VALUE_INVALID', 'intent')
    if not value['customer_reply'].strip():
        raise OutputContractError('OUTPUT_REPLY_EMPTY', 'customer_reply')
    for field in ('key_points', 'missing_information', 'risk_warnings', 'image_observations'):
        if not isinstance(value[field], list) or any(not isinstance(v, str) for v in value[field]):
            raise OutputContractError('OUTPUT_FIELD_TYPE_INVALID', field)
    terms = value['normalized_terms']
    if not isinstance(terms, list) or any(not isinstance(v, dict)
        or not isinstance(v.get('term'), str) or not isinstance(v.get('normalized'), str) for v in terms):
        raise OutputContractError('OUTPUT_FIELD_TYPE_INVALID', 'normalized_terms')
    if not isinstance(value['citations'], list):
        raise OutputContractError('OUTPUT_FIELD_TYPE_INVALID', 'citations')
    citations = []
    for row in value['citations']:
        if isinstance(row, str):
            row = {'evidence_id': row}  # exact ID wrapping, never source inference
        if not isinstance(row, dict) or set(row) != {'evidence_id'} or not isinstance(row['evidence_id'], str):
            raise OutputContractError('OUTPUT_CITATION_FORMAT_INVALID', 'citations')
        if row['evidence_id'] not in evidence_ids:
            raise OutputContractError('OUTPUT_CITATION_UNKNOWN', 'citations')
        if row not in citations:
            citations.append(row)
    value['citations'] = citations
    if not value['answerable'] and (citations or value['key_points']):
        raise OutputContractError('OUTPUT_STATE_INCONSISTENT', 'answerable')
    if not program_references and value['answerable'] and not citations and not (allow_image_only and value['image_observations']):
        raise OutputContractError('OUTPUT_CITATION_MISSING', 'citations')
    # This optional model checklist is diagnostic, never proof of semantic
    # coverage. A malformed checklist cannot destroy an otherwise valid reply.
    coverage = value.get('answer_aspect_coverage')
    if coverage is not None:
        if not isinstance(coverage, list):
            value.pop('answer_aspect_coverage')
        else:
            value['answer_aspect_coverage'] = [{k:r[k] for k in ('aspect_index','status','answer_quote','evidence_ids')} for r in coverage if isinstance(r, dict)
                and type(r.get('aspect_index')) is int and r['aspect_index'] >= 0
                and r.get('status') in ('answered', 'evidence_missing')
                and isinstance(r.get('answer_quote'), str)
                and isinstance(r.get('evidence_ids'), list)
                and all(isinstance(i, str) and i in evidence_ids for i in r['evidence_ids'])]
    return value
