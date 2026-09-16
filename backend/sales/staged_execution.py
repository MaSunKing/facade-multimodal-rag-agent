"""Request-local continuations for migrating tested answer paths into graph nodes.

No checkpointing: cursors may hold temporary images and local model references.
Only node names/timings, never these references, are exposed in API metadata.
"""
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Iterator
from backend.sales.runtime_status import report_error
from backend.request_budget import check_budget, reserve_recovery


@dataclass
class WorkflowStep:
    node: str
    operation: Callable[..., Any] | None = None
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)


class WorkflowCursor:
    def __init__(self, steps: Iterator):
        self.steps = steps
        self.pending = None
        self.result = None
        self.done = False
        self.operation_error = None
        self.last_value = None

    def advance(self, value=None, error=None):
        try:
            self.pending = self.steps.throw(error) if error else self.steps.send(value)
            if not isinstance(self.pending, WorkflowStep):
                raise TypeError('Answer workflow must yield WorkflowStep')
        except StopIteration as finished:
            self.result, self.done, self.pending = finished.value, True, None

    def execute(self):
        step = self.pending
        self.operation_error = None
        self.last_value = None
        try:
            try:
                value = step.operation(*step.args, **step.kwargs) if step.operation else None
            except (TimeoutError, ConnectionError):
                # Read-only retrieval only. GPU generation, parsing defects,
                # authentication and writes are never replayed unchanged.
                if step.node not in {'company_rag','customer_documents'} or not reserve_recovery(
                        step.node, 'retry_transient_read'):
                    raise
                check_budget()
                value = step.operation(*step.args, **step.kwargs)
        except Exception as exc:
            self.operation_error = type(exc).__name__
            report_error(exc,stage=step.node)
            # Existing path-specific error handling remains authoritative.
            self.advance(error=exc)
        else:
            self.last_value = value
            self.advance(value)

    def close(self):
        self.steps.close()


def staged_answer(function):
    """Keep the synchronous compatibility API; the graph consumes .steps()."""
    @wraps(function)
    def synchronous(*args, **kwargs):
        cursor = WorkflowCursor(function(*args, **kwargs))
        try:
            cursor.advance()
            while not cursor.done:
                cursor.execute()
            return cursor.result
        finally:
            cursor.close()
    synchronous.steps = function
    return synchronous
