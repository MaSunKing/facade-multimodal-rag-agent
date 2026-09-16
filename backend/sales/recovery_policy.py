"""CPU-only recovery decisions over explicit source/coverage diagnostics.

This module does not infer business intent from query keywords. A tool must
already be requested by the semantic plan. Missing literal target strings alone
are NOT proof of missing evidence; translations and synonyms can differ.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any


def normalise_anchor(value: str) -> str:
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(value))).casefold()


def content_search_anchor(value: str) -> str:
    """Remove query-control/source-position words, not business semantics."""
    tokens = str(value).split()
    controls = {'document', 'documents', 'file', 'files', 'source', 'sources',
                'theme', 'themes', 'topic', 'topics', 'statement', 'statements',
                'comparison', 'compare', 'in', 'of', 'the', 'and', 'from', 'about'}
    kept = [token for token in tokens if token.casefold().strip('.,:') not in controls and not token.isdigit()]
    return ' '.join(kept).strip()


def query_language_mismatch(query: str, excerpts: str) -> bool:
    """Detect untranslated search concepts, not just the presence of NASA/IDs.

    Conservative English-source heuristic; other scripts are not relabelled
    English. Proper names may remain Chinese when most query concepts are Latin.
    """
    latin = len(re.findall(r'[A-Za-z]', excerpts))
    han = len(re.findall(r'[\u4e00-\u9fff]', excerpts))
    if latin <= max(60, 3 * han):
        return False
    query_han = len(re.findall(r'[\u4e00-\u9fff]', query))
    query_words = re.findall(r'[A-Za-z]{3,}', query)
    return not query.strip() or not query_words or query_han > max(3, len(query_words) * 2)


@dataclass(frozen=True)
class RecoveryDecision:
    actions: tuple[tuple[str, str], ...] = ()
    gaps: tuple[dict[str, Any], ...] = ()
    blocked: tuple[dict[str, Any], ...] = ()

    def audit(self) -> dict[str, Any]:
        return {'policy': 'bounded_source_gap_recovery_v1',
                'actions': [{'tool': tool, 'action': action} for tool, action in self.actions],
                'gaps': list(self.gaps), 'blocked': list(self.blocked),
                'literal_target_absence_is_not_gold': True}


def assess_source_gaps(plan: Any, statuses: dict[str, dict[str, Any]], *,
                       web_allowed: bool) -> RecoveryDecision:
    """Bundle affected read-only tools into one repair, preserving ready ones.

    Statuses come from tool execution, not text in untrusted documents. A
    confirmed no-support label/invalid input is terminal, not a reason to search
    forever. current-fact repair requires BOTH semantic need and user consent.
    """
    get = plan.get if isinstance(plan, dict) else lambda key, default=None: getattr(plan, key, default)
    requested = set(get('tools', []) or [])
    actions, gaps, blocked = [], [], []
    for tool, status in statuses.items():
        if tool not in requested and not (tool == 'customer_visual' and 'customer_documents' in requested
                                         and get('document_visual_required') is True):
            continue
        state = status.get('state', 'ready')
        if state == 'ready':
            continue
        gap = {'tool': tool, 'reason': state}
        gaps.append(gap)
        if state in {'unavailable', 'invalid_input', 'permission_denied', 'quota_exhausted',
                     'confirmed_no_support', 'authentication_failed'}:
            blocked.append(gap)
            continue
        if tool == 'public_web_search' and not (web_allowed and get('requires_public_web', False)):
            blocked.append({**gap, 'reason': 'web_not_authorised_or_not_required'})
            continue
        action = {'empty': 'targeted_retrieval', 'index_only': 'broaden_structure_lookup',
                  'transient_error': 'retry_transient_read',
                  'verified_relation_missing': 'repair_target_coverage',
                  'missing_visual_input': 'select_related_visual',
                  'current_fact_missing': 'refresh_public_fact'}.get(state)
        if action:
            actions.append((tool, action))
        else:
            blocked.append({**gap, 'reason': 'no_verified_safe_repair_signal'})
    return RecoveryDecision(tuple(actions), tuple(gaps), tuple(blocked))


def source_diagnostics(results: dict[str, Any], *, visual_required: bool = False) -> dict[str, dict[str, Any]]:
    """Convert actual tool outputs into typed gap states (not semantic gold)."""
    statuses = {}
    if 'customer_documents' in results:
        documents = results['customer_documents']
        evidence = documents.get('evidence', [])
        state = ('unavailable' if documents.get('status') == 'session_not_found' else
                 'empty' if not evidence else
                 'invalid_input' if not isinstance(evidence, list) or any(not isinstance(item, dict) for item in evidence) else
                 'index_only' if all(item.get('evidence_scope') == 'document_index' for item in evidence) else 'ready')
        statuses['customer_documents'] = {'state': state}
        if documents.get('error'):
            statuses['customer_documents'] = {'state': _source_error_state(documents['error'])}
        if visual_required and not documents.get('selected_visuals'):
            statuses['customer_visual'] = {'state': 'missing_visual_input' if documents.get('input_snapshot', {}).get('available_visual_count') else 'unavailable'}
    if 'company_rag' in results:
        statuses['company_rag'] = {'state': 'ready' if results['company_rag'].get('text_evidence') else 'empty'}
        if results['company_rag'].get('error'):
            statuses['company_rag'] = {'state': _source_error_state(results['company_rag']['error'])}
    if 'public_web_search' in results:
        sources, meta = results['public_web_search']
        state = ('ready' if sources else
                 {'not_configured': 'unavailable', 'quota_exhausted': 'quota_exhausted',
                  'not_requested': 'permission_denied', 'failed': 'unavailable'}.get(meta.get('status'), 'current_fact_missing'))
        statuses['public_web_search'] = {'state': state}
    return statuses


def _source_error_state(info: dict[str, Any]) -> str:
    code = info.get('code')
    if code in {'ACCESS_DENIED', 'AUTH_REQUIRED', 'WEB_AUTH_FAILED'}:
        return 'permission_denied'
    if code == 'WEB_QUOTA_EXHAUSTED':
        return 'quota_exhausted'
    if code in {'NOT_FOUND', 'ATTACHMENT_EXPIRED'}:
        return 'unavailable'
    if info.get('retryable') and info.get('exception_type') in {'TimeoutError', 'ConnectionError'}:
        return 'transient_error'
    return 'invalid_input'


def read_source_step(step: Any, fallback: dict[str, Any]):
    """Capture one failed source without discarding unrelated good evidence.

    Existing cursor transient-read retry uses the shared slot. Cancellation and
    global deadline always propagate; they are never disguised as missing data.
    """
    from backend.request_budget import RequestBudgetExceeded
    from backend.sales.runtime_status import report_error
    try:
        return (yield step)
    except RequestBudgetExceeded:
        raise
    except Exception as exc:
        return {**fallback, 'error': report_error(exc, stage=step.node)}


def answer_document_coverage(plan: Any, result: dict[str, Any], visible: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Check document acknowledgment, NOT full natural-language completeness.

    Only whole/cross-document answers imply all supplied content sources should
    be addressed. Single-file lookup may legitimately choose just one file.
    Never require citations to a file that packing removed from model input.
    """
    scope = getattr(plan, 'document_scope', 'unknown')
    expected = sorted({str(item.get('document_name')) for item in visible.values()
                       if item.get('document_name') and item.get('evidence_scope') == 'content'})
    cited = {str(item.get('document_name')) for item in result.get('citations', [])}
    reply = normalise_anchor(result.get('customer_reply', ''))
    covered = {name for name in expected if name in cited or normalise_anchor(name) in reply}
    missing = sorted(set(expected)-covered) if scope in {'whole_document', 'cross_document'} and result.get('answerable') and len(expected)>1 else []
    return {'expected_visible_documents': expected, 'addressed_documents': sorted(covered),
            'verified_missing_documents': missing, 'scope': scope,
            'semantic_assertion_completeness_verified': False}


def repair_source_bundle(plan: Any, statuses: dict[str, dict[str, Any]], results: dict[str, Any],
                         operations: dict[str, Any], *, web_allowed: bool):
    """One pre-generation repair checkpoint; retain good initial tool results.

    Callers supply read-only WorkflowSteps. They must use existing authorised
    local handles, not raw instructions or paths from a document.
    """
    from backend.request_budget import reserve_recovery, current_budget, check_budget, RequestBudgetExceeded
    from backend.sales.staged_execution import WorkflowStep
    from backend.sales.runtime_status import report_error
    yield WorkflowStep('assess_coverage')
    decision = assess_source_gaps(plan, statuses, web_allowed=web_allowed)
    audit = {**decision.audit(), 'initial_statuses': statuses, 'attempted': False,
             'completed_tools': [], 'failed_tools': []}
    if not decision.actions:
        audit['reason'] = 'coverage_ready_or_no_safe_repair'
        return results, audit
    applicable = [(tool, action) for tool, action in decision.actions if tool in operations]
    if not applicable:
        audit['reason'] = 'repair_operation_unavailable'
        return results, audit
    action_name = ';'.join(f'{tool}:{action}' for tool, action in applicable)
    if not reserve_recovery('assess_coverage', action_name, minimum_seconds=40):
        budget = current_budget.get()
        audit['reason'] = 'shared_recovery_slot_used' if budget and budget.recoveries else 'remaining_time_insufficient'
        return results, audit
    audit.update(attempted=True, reason='verified_source_gap')
    repaired = dict(results)
    for tool, _action in applicable:
        step = operations[tool]
        try:
            check_budget()
            value = yield step
        except RequestBudgetExceeded:
            raise
        except Exception as exc:
            info = report_error(exc, stage=step.node)
            audit['failed_tools'].append({'tool': tool, 'code': info['code']})
            # A failed supplement must not erase usable first-pass evidence.
        else:
            repaired[tool] = value
            audit['completed_tools'].append(tool)
    audit['final_statuses'] = source_diagnostics(repaired)
    audit['remaining_gaps'] = assess_source_gaps(plan, audit['final_statuses'], web_allowed=web_allowed).audit()['gaps']
    audit['tool_completion_is_not_semantic_accuracy'] = True
    return repaired, audit
