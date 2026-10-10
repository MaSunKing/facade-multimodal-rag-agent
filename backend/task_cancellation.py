"""Owner-scoped cooperative cancellation; never release a live worker's slot."""
import threading
import time
import json
from pathlib import Path

from backend.request_budget import RequestBudget

_lock = threading.RLock()
_active: dict[tuple[str, str], RequestBudget] = {}
_pending: dict[tuple[str, str], float] = {}
_request_budgets = {}
CONTROL_DIRECTORY = Path(__file__).resolve().parents[1] / 'runtime' / 'task_controls'


def bind_request_budget(request_id, budget):
    with _lock: _request_budgets[request_id]=budget


def unbind_request_budget(request_id, budget):
    with _lock:
        if _request_budgets.get(request_id) is budget: _request_budgets.pop(request_id,None)


def consume_local_cancellation(request_id):
    """Local OS-user control only. No HTTP route can write this inbox."""
    import re
    if not re.fullmatch(r'[a-f0-9]{32}',request_id): return False
    path=CONTROL_DIRECTORY / ('cancel-'+request_id+'.json')
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 512: return False
        payload=json.loads(path.read_text(encoding='utf-8-sig'))
        if payload != {'request_id':request_id,'action':'cancel'}: return False
        with _lock:
            budget=_request_budgets.get(request_id)
            if budget is None: return False
            budget.cancelled.set()
        path.unlink()
        return True
    except (OSError,ValueError): return False


def _purge():
    now = time.monotonic()
    for key in list(_pending):
        if now - _pending[key] > 600:
            del _pending[key]


def register_task(owner: str, task_id: str, budget: RequestBudget):
    key = (owner, task_id)
    with _lock:
        _purge()
        if key in _active:
            raise ValueError('duplicate_active_task_id')
        if key in _pending:
            budget.cancelled.set()
        _active[key] = budget


def unregister_task(owner: str, task_id: str, budget: RequestBudget):
    with _lock:
        key = (owner, task_id)
        if _active.get(key) is budget:
            del _active[key]


def cancel_task(owner: str, task_id: str) -> str:
    with _lock:
        _purge()
        key = (owner, task_id)
        if key not in _pending and len(_pending) >= 256:
            del _pending[min(_pending, key=_pending.get)]
        _pending[key] = time.monotonic()
        budget = _active.get(key)
        if budget is not None:
            budget.cancelled.set()
        # Do not reveal whether another owner's task exists.
        return 'cancellation_requested'


def task_status(owner: str, task_id: str) -> dict:
    """Worker-owned completion, never another owner's task existence."""
    with _lock:
        budget = _active.get((owner, task_id))
        return {'active': budget is not None,
                'status': 'cancelling' if budget and budget.cancelled.is_set() else 'running' if budget else 'idle'}
