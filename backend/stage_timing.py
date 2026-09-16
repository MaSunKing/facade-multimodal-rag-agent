"""Local fine-grained timings; logs contain stage names/numbers, never content."""
from __future__ import annotations
import json
import os
import time
import threading
from pathlib import Path
from contextlib import contextmanager
_log_lock = threading.Lock()


@contextmanager
def stage_timer(audit: dict, stage: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        ms = round((time.perf_counter() - start) * 1000, 3)
        audit[stage] = round(float(audit.get(stage, 0)) + ms, 3)
        if os.getenv('FACADE_RAG_DEBUG') == '1':
            event = {'stage':stage, 'elapsed_ms':ms, 'monotonic_time':time.monotonic(), 'thread_id':threading.get_ident()}
            print('FACADE_STAGE_TIMING ' + json.dumps(event), flush=True)
            log = os.getenv('FACADE_STAGE_TIMING_FILE')
            if log:
                try:
                    with _log_lock, Path(log).open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(event)+'\n')
                except OSError:
                    # Observability failure must not discard original evidence.
                    print('FACADE_STAGE_TIMING_LOG_UNAVAILABLE', flush=True)


def timed_call(audit: dict, stage: str, function, *args, **kwargs):
    with stage_timer(audit, stage):
        return function(*args, **kwargs)
