"""Content-free summaries plus owner/admin-only parsed-answer detail (no model calls)."""
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from backend import access_control
from backend.sales.runtime_status import current_execution

_lock = threading.Lock()
_active = {}
_checkpoint_workers = {}


def record_visual_labels(phase, labels, *, round_number=None):
    """Persist parsed text separately; never put it in global/list summaries."""
    journal = current_execution.get()
    if not journal or phase not in {'visual_tool', 'final_context', 'answer_audit_input'}:
        return
    from backend.sales.stage_details import safe
    import hashlib
    import re
    def clean(value):
        if isinstance(value, str):
            # Redact before any bound: secrets may cross preview boundaries.
            value=re.sub(r'(?i)bearer\s+\S+|\bsk-[A-Za-z0-9_-]{12,}', '[credential redacted]',value)
            value=re.sub(r'(?i)(?:api[_-]?key|password|secret|token)\s*[:=]\s*\S+', '[credential redacted]',value)
            return re.sub(r'[A-Za-z]:[\\/][^\s"<>]+', '[local path redacted]',value)
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k in {
                'labels','description','visible_text','relations','uncertainties',
                'title','category','visual_role','product_name','product_family','product_form',
                'color_name','variant_or_code','surface_finish','insulation_type','project_names',
                'cities','provinces','caption','section_heading','node_system_references',
                'installation_method','node_kind','label_basis'}}
        if isinstance(value, list): return [clean(v) for v in value[:12]]
        return value if value is None or isinstance(value,(bool,int,float)) else None
    selected=[]
    for item in labels[:48]:
        row={k:item[k] for k in ('asset_id','evidence_id','document_id','source_type',
            'label_basis','in_draft_context','display_delivered') if k in item}
        row=safe(row)
        row['labels']=clean(item.get('labels') or {})
        row['label_hash']=hashlib.sha256(json.dumps(item.get('labels') or {},
            ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        selected.append(row)
    record={'phase':phase,'round':round_number,'labels':selected,
            'text_only':True,'redacted':True,'omitted_label_count':max(0,len(labels)-48)}
    journal.visual_label_versions.append(record)
    journal.visual_label_versions=journal.visual_label_versions[-12:]
    try:
        with access_control._connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS request_visual_labels '
                '(request_id TEXT PRIMARY KEY, recorded_at TEXT NOT NULL, versions_json TEXT NOT NULL)')
            db.execute('INSERT OR REPLACE INTO request_visual_labels VALUES (?,?,?)',
                (journal.request_id,datetime.now(timezone.utc).isoformat(),
                 json.dumps(journal.visual_label_versions,ensure_ascii=False)))
            db.execute('DELETE FROM request_visual_labels WHERE request_id NOT IN '
                '(SELECT request_id FROM request_visual_labels ORDER BY recorded_at DESC LIMIT 300)')
    except (OSError,sqlite3.Error,TypeError,ValueError):
        pass  # Diagnostic storage must not fail an answer.


def _visual_detail(result):
    if not result: return result
    identity=(result.get('summary') or {}).get('execution',{}).get('request_id') or result['request_id']
    with access_control._connect() as db:
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='request_visual_labels'").fetchone():
            row=db.execute('SELECT versions_json FROM request_visual_labels WHERE request_id IN (?,?) '
                'ORDER BY recorded_at DESC LIMIT 1',(identity,result['request_id'])).fetchone()
            if row: result['visual_label_versions']=json.loads(row[0])
    return result


def record_answer_version(response, round_number):
    """Parsed answer only, never prompt/evidence/pixels/hidden reasoning."""
    journal = current_execution.get()
    if not journal: return None
    from backend.sales.stage_details import safe
    text = str(response.get('customer_reply') or '')
    # Reuse credential/path redaction without the 600-character metadata limit.
    import re
    text = re.sub(r'(?i)bearer\s+\S+|\bsk-[A-Za-z0-9_-]{12,}', '[credential redacted]', text)
    text = re.sub(r'(?i)(?:api[_-]?key|password|secret|token)\s*[:=]\s*\S+', '[credential redacted]', text)
    text = re.sub(r'[A-Za-z]:[\\/][^\s"<>]+', '[local path redacted]', text)
    row = {'kind':'revision' if response.get('meta',{}).get('answer_revision_count') else 'draft',
           'round':round_number, 'publication':'internal_not_approved',
           'customer_reply':text[:16000], 'truncated':len(text)>16000,
           'answerable':response.get('answerable'),
           'citation_ids':[str(c.get('evidence_id'))[:160] for c in response.get('citations',[]) if isinstance(c,dict)][:80],
           'missing_information':safe(response.get('missing_information',[]))}
    journal.answer_versions.append(row)
    journal.answer_versions=journal.answer_versions[-6:]
    return row


def record_answer_review(version, review):
    if version is not None:
        from backend.sales.stage_details import safe
        version['review']=safe({k:review[k] for k in ('decision','coverage_status','reason','revision','proof','missing_aspects','error_code','label_consistency') if k in review})
        version['publication']='approved' if review.get('decision')=='pass' else 'withheld'


def persist_live_checkpoint(journal, started):
    """Persist bounded diagnostic metadata, never prompts or attachment bytes."""
    snapshot = journal.snapshot()
    record = {'request_id': journal.request_id, 'agent_request_id':journal.agent_request_id,
        'started_at':started, 'checkpoint':True, 'observed_at':datetime.now(timezone.utc).isoformat(),
        'status':snapshot['state'], 'summary':{'stage_details':snapshot.get('stage_details',[])},
        'execution':_project(snapshot, ('request_id','state','stage','elapsed_ms','stage_timings',
            'summary_progress','model_contracts','token_usage','audit_invoked','final_generation_invoked'))}
    try:
        with access_control._connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS live_execution_checkpoints '
                '(request_id TEXT PRIMARY KEY, agent_request_id TEXT, started_at TEXT NOT NULL, summary_json TEXT NOT NULL)')
            db.execute('INSERT OR REPLACE INTO live_execution_checkpoints VALUES (?,?,?,?)',
                (journal.request_id,journal.agent_request_id,started,json.dumps(record,ensure_ascii=False)))
            db.execute('DELETE FROM live_execution_checkpoints WHERE request_id NOT IN '
                '(SELECT request_id FROM live_execution_checkpoints ORDER BY started_at DESC LIMIT 300)')
    except (OSError,sqlite3.Error,TypeError,ValueError):
        # Diagnostics must not terminate the user workflow.
        pass


def _checkpoint_loop(journal, started, stop):
    while not stop.wait(2):
        from backend.task_cancellation import consume_local_cancellation
        consume_local_cancellation(journal.request_id)
        persist_live_checkpoint(journal,started)


def checkpoint_trace(request_id):
    with access_control._connect() as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='live_execution_checkpoints'").fetchone(): return None
        row=db.execute('SELECT summary_json FROM live_execution_checkpoints WHERE request_id=? OR agent_request_id=?',
            (request_id,request_id)).fetchone()
    if not row: return None
    value=json.loads(row[0])
    # This process no longer has the request: a checkpoint is not a live task
    # or a completed response. Keep the last observed stage for investigation.
    value.update(live=False,status='interrupted',last_observed_status=value['status'])
    value['execution']['state']='interrupted'
    return value


def _project(value, keys):
    return {k: value[k] for k in keys if k in value} if isinstance(value, dict) else {}


def correction_view(value):
    """Keep actionable codes/indices, not question text or source excerpts."""
    output = _project(value, ('action', 'invoked', 'accepted', 'syntax_changes', 'input_tokens', 'output_limit', 'error_type', 'error_code', 'policy', 'runtime_profile'))
    for phase in ('initial', 'after', 'final'):
        output[phase] = _project((value or {}).get(phase), ('missing_documents', 'missing_aspects',
                                     'citation_errors', 'schema_errors', 'failures', 'status', 'method',
                                     'semantic_coverage_verified', 'checklist_unaccounted_document_ids',
                                     'checklist_unaccounted_aspect_indices'))
    output['deferred'] = [_project(d, ('aspect_index', 'reason')) for d in (value or {}).get('deferred', [])]
    # Counts/codes pinpoint the failing check without exposing customer facts.
    checks = (value or {}).get('grounding_checks') or {}
    if checks:
        def check_view(row):
            projected = _project(row, ('passed', 'lexical_overlap_term_count', 'answer_term_count', 'cited_evidence_ids'))
            for key in ('unsupported_numeric_claims', 'unsupported_promotional_claims',
                        'incompatible_standard_scope_claims', 'unsupported_normative_claims'):
                projected[key + '_count'] = len(row.get(key) or [])
            return projected
        output['grounding_checks'] = _project(checks, ('passed', 'failures', 'method',
            'original_issues_not_patch_failures', 'final_validation_still_required'))
        output['grounding_checks'].update(original=check_view(checks.get('original') or {}),
            merged=check_view(checks.get('merged') or {}),
            additions=[{**_project(row, ('aspect_index', 'status', 'evidence_ids', 'failure_codes')),
                        'checks': check_view(row.get('checks') or {})} for row in checks.get('additions', [])])
    return output


def trace_summary(response):
    """Whitelist structure, not arbitrary prompts/reasons/attachment contents."""
    meta = response.get('meta') or {}
    orchestration = meta.get('orchestration') or {}
    plans = orchestration.get('tool_plan_details') or []
    if isinstance(plans, dict):
        plans = [plans]
    summary = {
        'mode': meta.get('mode'), 'answerable': response.get('answerable'),
        'answer_review': _project(meta.get('answer_review'), ('decision','status','coverage_status','review_policy',
            'fact_support_verified','round','reason','error_code',
            'elapsed_ms','token_usage')),
        'answer_revision_count': meta.get('answer_revision_count',0),
        'tools': orchestration.get('tools') or [],
        'tool_rounds': orchestration.get('tool_rounds'),
        'planning_rounds': orchestration.get('planning_rounds'),
        'plans': [_project(p, ('tools', 'task_type', 'intent', 'document_scope',
                   'product_overview', 'wants_visuals', 'visual_scope', 'visual_task',
                   'requires_public_web', 'web_search_profile')) for p in plans if isinstance(p, dict)],
        'node_trace': [_project(n, ('node', 'round', 'elapsed_ms', 'reused', 'operation_error','token_usage'))
                       for n in (orchestration.get('node_trace') or []) if isinstance(n, dict)][:80],
        'execution': _project(meta.get('execution'), ('request_id', 'state', 'stage', 'elapsed_ms', 'stage_timings', 'audit_invoked', 'final_generation_invoked', 'model_contracts','token_usage')),
        'errors': [_project(e, ('code', 'stage', 'retryable', 'exception_type'))
                   for e in ((meta.get('execution') or {}).get('errors') or [])][:20],
        'evidence_gate': _project(meta.get('evidence_gate'), ('status', 'error_code', 'visual_need',
                       'evaluation_method', 'reused', 'gap_type', 'web_can_resolve', 'elapsed_ms', 'recovery_need')),
        'task_support_audit': _project(meta.get('task_support_audit'), ('status', 'error_code', 'visual_need', 'gap_type')),
        'replanning': _project(meta.get('replanning'), ('attempted', 'status', 'error_code', 'action', 'elapsed_ms',
            'invoked', 'reason', 'recovery_need', 'recovery_budget')),
        'online_search': _project(meta.get('online_search'), ('status', 'error_code', 'cache_hit',
                          'attempts', 'source_profile', 'api_calls_for_query')),
        'citation_ids': [str(c.get('evidence_id'))[:160] for c in response.get('citations', [])
                         if isinstance(c, dict) and c.get('evidence_id')][:80],
        'visual_count': len(response.get('visual_assets') or []),
    }
    replan = meta.get('replanning') or {}
    summary['replanning'].update(_project(replan, ('error_type', 'changed_query_requires_execution', 'retained_source_tools')))
    if replan.get('reason') in {'shared_recovery_or_time_budget_exhausted', 'no_authorised_tools'}:
        summary['replanning']['reason'] = replan['reason']
    summary['replanning']['feedback'] = _project(replan.get('feedback'), ('gap_type', 'error_code', 'web_can_resolve'))
    summary['evidence_gate'].update(_project(meta.get('evidence_gate'), ('error_type', 'output_tokens', 'output_limit', 'max_new_tokens', 'model_action', 'runtime_profile')))
    summary['evidence_gate']['format_repair'] = _project((meta.get('evidence_gate') or {}).get('format_repair'),
                                                       ('attempts', 'initial_error_code', 'succeeded'))
    summary['planner'] = _project(orchestration, ('planner_latency_ms', 'planner_model_load_ms',
        'planner_generation_ms', 'planner_cache_hit', 'planner_input_tokens', 'planner_output_tokens',
        'bounded_retry_attempted', 'max_execution_recoveries', 'max_query_normalizations'))
    summary['recovery_actions'] = [_project(a, ('stage', 'action', 'round'))
                                  for a in (orchestration.get('recovery_actions') or []) if isinstance(a, dict)]
    summary['guard_diagnostics'] = [_project(a, ('tool','code','action'))
                                    for a in (orchestration.get('guard_diagnostics') or []) if isinstance(a, dict)]
    summary['query_policy'] = orchestration.get('query_policy')
    generation = meta.get('generation_input_audit') or meta.get('generation_input') or {}
    summary['document_summary'] = meta.get('document_summary')
    summary['context'] = _project(generation, ('actual_prompt_tokens', 'max_prompt_tokens',
                       'kept_evidence_ids', 'removed_evidence_ids'))
    summary['generation'] = _project(meta.get('generation_audit'), ('input_tokens', 'output_tokens',
                      'max_new_tokens', 'hit_output_limit', 'generation_seconds', 'json_recovered'))
    summary['generation'].update(_project(generation, ('generated_tokens', 'generation_seconds',
        'hit_output_limit', 'runtime_profile')))
    if generation.get('coverage_correction'):
        summary['generation']['coverage_correction'] = correction_view(generation['coverage_correction'])
    summary['visual_input_manifest'] = [_project(v, ('image_number', 'evidence_id', 'asset_id', 'sha256', 'origin'))
                  for v in (generation.get('visual_input_manifest') or []) if isinstance(v, dict)][:10]
    return summary


def save_summary(request_id, response):
    # Observability must never turn a successful answer into a failed request.
    try:
        with access_control._connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS admin_request_traces '
                       '(request_id TEXT PRIMARY KEY REFERENCES agent_requests(request_id) ON DELETE CASCADE, summary_json TEXT NOT NULL)')
            summary=trace_summary(response)
            journal=current_execution.get()
            if journal: summary['stage_details']=journal.snapshot().get('stage_details',[])
            db.execute('INSERT OR REPLACE INTO admin_request_traces VALUES (?,?)',
                       (request_id, json.dumps(summary, ensure_ascii=False)))
            if journal and journal.answer_versions:
                db.execute('CREATE TABLE IF NOT EXISTS request_answer_versions '
                    '(request_id TEXT PRIMARY KEY REFERENCES agent_requests(request_id) ON DELETE CASCADE, versions_json TEXT NOT NULL)')
                db.execute('INSERT OR REPLACE INTO request_answer_versions VALUES (?,?)',
                    (request_id,json.dumps(journal.answer_versions,ensure_ascii=False)))
    except (OSError, sqlite3.Error, TypeError, ValueError):
        pass


def register_live(journal):
    started=datetime.now(timezone.utc).isoformat()
    stop=threading.Event()
    with _lock:
        _active[journal.request_id] = (journal, started, time.monotonic())
        _checkpoint_workers[journal.request_id]=stop
    persist_live_checkpoint(journal,started)
    threading.Thread(target=_checkpoint_loop,args=(journal,started,stop),daemon=True,
        name='trace-checkpoint-'+journal.request_id[:8]).start()


def bind_trace_owner(request_id, owner, client_task_id=None):
    if not owner:
        return
    with access_control._connect() as db:
        db.execute('CREATE TABLE IF NOT EXISTS request_trace_owners '
                   '(request_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, client_task_id TEXT)')
        db.execute('INSERT OR REPLACE INTO request_trace_owners VALUES (?,?,?)',
                   (request_id,owner,client_task_id))


def owned_trace(owner, request_id):
    # Ownership is backend-bound at request admission, never supplied by a
    # model or inferred from a guessed request UUID. Old unowned traces deny.
    with access_control._connect() as db:
        exists=db.execute("SELECT 1 FROM sqlite_master WHERE name='request_trace_owners'").fetchone()
        row=db.execute('SELECT request_id FROM request_trace_owners WHERE owner_id=? '
            'AND (request_id=? OR client_task_id=?)', (owner,request_id,request_id)).fetchone() if exists else None
        if row is None and exists and db.execute("SELECT 1 FROM sqlite_master WHERE name='admin_request_traces'").fetchone():
            # UI lists agent IDs after completion, while admission ownership
            # belongs to the HTTP ID. Resolve only through the persisted link.
            row=db.execute("SELECT o.request_id FROM request_trace_owners o JOIN admin_request_traces t "
                "ON json_extract(t.summary_json,'$.execution.request_id')=o.request_id "
                  "WHERE o.owner_id=? AND t.request_id=?",(owner,request_id)).fetchone()
        if row is None and exists and db.execute("SELECT 1 FROM sqlite_master WHERE name='live_execution_checkpoints'").fetchone():
            row=db.execute('SELECT o.request_id FROM request_trace_owners o JOIN live_execution_checkpoints c '
                'ON c.request_id=o.request_id WHERE o.owner_id=? AND c.agent_request_id=?', (owner,request_id)).fetchone()
    return read_trace(row[0]) if row else None


def owned_traces(owner, limit=30):
    with access_control._connect() as db:
        exists=db.execute("SELECT 1 FROM sqlite_master WHERE name='request_trace_owners'").fetchone()
        ids=[r[0] for r in db.execute('SELECT request_id FROM request_trace_owners WHERE owner_id=? '
            'ORDER BY rowid DESC LIMIT ?', (owner,max(1,min(int(limit),100))))] if exists else []
    rows=[r for i in ids if (r:=read_trace(i)) is not None]
    return {'active':[r for r in rows if r.get('live')],
            'requests':[r for r in rows if not r.get('live')]}


def unregister_live(request_id):
    with _lock:
        tracked = _active.pop(request_id, None)
        stop=_checkpoint_workers.pop(request_id,None)
        if stop: stop.set()
    if tracked:
        journal, started, _ = tracked
        persist_live_checkpoint(journal,started)
        snapshot = journal.snapshot()
        # Persist even queue timeout/cancellation, which may occur before the
        # Agent creates a req_ record. Keep only bounded status/timing fields.
        record = {'request_id': request_id, 'started_at': started, 'status': snapshot['state'],
                  'summary':{'stage_details':snapshot.get('stage_details',[])},
                  'total_latency_ms': snapshot['elapsed_ms'],
                  'execution': {**_project(snapshot, ('state', 'stage', 'elapsed_ms', 'stage_timings', 'summary_progress', 'model_contracts','token_usage', 'audit_invoked', 'final_generation_invoked')),
                     'errors': [_project(e, ('code', 'stage', 'exception_type', 'retryable')) for e in snapshot['errors']]}}
        try:
            with access_control._connect() as db:
                db.execute('CREATE TABLE IF NOT EXISTS admin_http_traces (request_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, summary_json TEXT NOT NULL)')
                db.execute('INSERT OR REPLACE INTO admin_http_traces VALUES (?,?,?)',
                           (request_id, started, json.dumps(record, ensure_ascii=False)))
                if journal.agent_request_id:
                    db.execute('CREATE TABLE IF NOT EXISTS admin_request_traces '
                        '(request_id TEXT PRIMARY KEY REFERENCES agent_requests(request_id) ON DELETE CASCADE, summary_json TEXT NOT NULL)')
                    interrupted=trace_summary({'meta':{'mode':'request_interrupted','execution':snapshot}})
                    interrupted['stage_details']=snapshot.get('stage_details',[])
                    # A timeout may happen before the graph's final response.
                    # Do not overwrite a completed richer response summary.
                    db.execute('INSERT OR IGNORE INTO admin_request_traces VALUES (?,?)',
                               (journal.agent_request_id,json.dumps(interrupted,ensure_ascii=False)))
                db.execute('DELETE FROM admin_http_traces WHERE request_id NOT IN '
                           '(SELECT request_id FROM admin_http_traces ORDER BY started_at DESC LIMIT 300)')
        except (OSError, sqlite3.Error):
            pass


def active_traces():
    with _lock:
        rows = list(_active.values())
    return [{'request_id': j.request_id, 'started_at': started, 'status': 'running',
             'summary':{'stage_details':j.snapshot().get('stage_details',[])},
             'live': True, 'execution': {**_project(j.snapshot(), ('state', 'stage', 'elapsed_ms', 'stage_timings', 'summary_progress', 'model_contracts','token_usage', 'audit_invoked', 'final_generation_invoked')),
                'errors': [_project(e, ('code', 'stage', 'exception_type', 'retryable')) for e in j.errors]}}
            for j, started, stamp in rows]


def list_traces(limit=30):
    access_control.ensure_schema()
    with access_control._connect() as db:
        rows = [dict(r) for r in db.execute('SELECT request_id,status,principal_kind,started_at,finished_at,'
                'total_latency_ms,error_type FROM agent_requests ORDER BY started_at DESC,request_id DESC LIMIT ?',
                (max(1, min(int(limit), 100)),))]
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        linked = set()
        if 'admin_request_traces' in tables:
            for row in rows:
                saved = db.execute('SELECT summary_json FROM admin_request_traces WHERE request_id=?', (row['request_id'],)).fetchone()
                if saved:
                    linked.add((json.loads(saved[0]).get('execution') or {}).get('request_id'))
        if 'admin_http_traces' in tables:
            for record in db.execute('SELECT request_id,summary_json FROM admin_http_traces ORDER BY started_at DESC LIMIT ?', (max(1, min(int(limit),100)),)):
                if record[0] not in linked:
                    rows.append(json.loads(record[1]))
        rows.sort(key=lambda r: r['started_at'], reverse=True)
        rows = rows[:max(1, min(int(limit),100))]
    return {'active': active_traces(), 'requests': rows}


def read_trace(request_id):
    for row in active_traces():
        if row['request_id'] == request_id:
            with _lock:
                active = _active.get(request_id)
                if active:
                    row['answer_versions'] = list(active[0].answer_versions)
                    row['visual_label_versions'] = list(active[0].visual_label_versions)
            return row
    access_control.ensure_schema()
    with access_control._connect() as db:
        row = db.execute('SELECT request_id,status,principal_kind,started_at,finished_at,total_latency_ms,error_type '
                         'FROM agent_requests WHERE request_id=?', (request_id,)).fetchone()
        if row is None:
            table = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='admin_request_traces'").fetchone()
            if table:
                row = db.execute('SELECT a.request_id,a.status,a.principal_kind,a.started_at,a.finished_at,a.total_latency_ms,a.error_type '
                    'FROM agent_requests a JOIN admin_request_traces t ON a.request_id=t.request_id '
                    "WHERE json_extract(t.summary_json,'$.execution.request_id')=?", (request_id,)).fetchone()
                if row is not None:
                    request_id = row['request_id']
        if row is None:
            table = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='admin_http_traces'").fetchone()
            saved_http = db.execute('SELECT summary_json FROM admin_http_traces WHERE request_id=?', (request_id,)).fetchone() if table else None
            return _visual_detail(json.loads(saved_http[0]) if saved_http else checkpoint_trace(request_id))
        result = dict(row)
        # Old requests can be reconstructed from bounded, redacted snapshots.
        snapshot = db.execute('SELECT snapshot_path FROM evidence_snapshots WHERE request_id=?', (request_id,)).fetchone()
        table = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='admin_request_traces'").fetchone()
        saved = db.execute('SELECT summary_json FROM admin_request_traces WHERE request_id=?', (request_id,)).fetchone() if table else None
        versions_table=db.execute("SELECT 1 FROM sqlite_master WHERE name='request_answer_versions'").fetchone()
        versions=db.execute('SELECT versions_json FROM request_answer_versions WHERE request_id=?',(request_id,)).fetchone() if versions_table else None
    if versions:
        result['answer_versions']=json.loads(versions[0])
    if saved:
        result['summary'] = json.loads(saved[0])
    elif (checkpoint := checkpoint_trace(request_id)) is not None:
        return _visual_detail(checkpoint)
    elif snapshot and snapshot[0]:
        root = (access_control.ROOT / 'runtime' / 'request_snapshots').resolve()
        path = (access_control.ROOT / snapshot[0]).resolve()
        if path.is_relative_to(root) and path.is_file() and path.stat().st_size <= 2_000_000:
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
                result['summary'] = trace_summary({'meta': payload, 'citations': payload.get('citations', []),
                                                   'visual_assets': payload.get('visual_assets', [])})
                result['summary']['context'] = _project(payload.get('context_snapshot_audit'),
                    ('actual_prompt_tokens', 'max_prompt_tokens', 'kept_evidence_ids', 'removed_evidence_ids',
                     'candidate_count', 'after_dedup_count', 'coverage_sufficient_before_generation', 'integrity_valid'))
            except (OSError, ValueError):
                result['trace_unavailable'] = True
    return _visual_detail(result)
