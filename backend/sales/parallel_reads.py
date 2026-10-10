"""Bounded read-only concurrency; context/ACL/cancellation survive thread hops.

Only independent recall operations use this helper. No generator, tool recovery,
shared request mutation or GPU generation is run here. Results are joined before
fusion; a running worker is not abandoned when a caller cancels.
"""
from concurrent.futures import ThreadPoolExecutor, wait
from contextvars import copy_context
import time
from backend.request_budget import check_budget

_READ_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix='rag-read')


def parallel_reads(operations):
    if len(operations) != 2:
        raise ValueError('parallel_reads_requires_two_independent_operations')
    check_budget()
    def invoke(operation):
        check_budget()
        started = time.perf_counter()
        try:
            return {'value': operation(), 'error': None,
                    'elapsed_ms': round((time.perf_counter()-started)*1000, 2)}
        except Exception as exc:
            from backend.request_budget import RequestBudgetExceeded
            if isinstance(exc, RequestBudgetExceeded): raise
            return {'value': None, 'error': exc,
                    'elapsed_ms': round((time.perf_counter()-started)*1000, 2)}
    started = time.perf_counter()
    futures = {}
    try:
        for name, operation in operations.items():
            futures[name] = _READ_POOL.submit(copy_context().run, invoke, operation)
        while not all(f.done() for f in futures.values()):
            check_budget()
            wait(tuple(futures.values()), timeout=.1)
        check_budget()
        results = {name: f.result() for name, f in futures.items()}
        audit = {'policy': 'bounded_independent_recall', 'max_workers': 2,
                 'elapsed_ms': round((time.perf_counter()-started)*1000, 2),
                 'branches': {name: {'elapsed_ms': row['elapsed_ms'],
                                      'error_type': type(row['error']).__name__ if row['error'] else None}
                              for name, row in results.items()}}
        return results, audit
    finally:
        for future in futures.values(): future.cancel()
        # Native embedding calls are cooperative, not forcibly killed. Keep the
        # request slot until every started read has released its model/ACL scope.
        wait(tuple(futures.values()))
