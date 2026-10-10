"""Cooperative request deadline; cancellation never releases a running worker slot."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import threading
import time
import math


class RequestBudgetExceeded(RuntimeError):
    pass


class RequestCancelled(RequestBudgetExceeded):
    pass


class ModelCallTimeout(RuntimeError):
    pass


@dataclass
class RequestBudget:
    deadline: float
    cancelled: threading.Event = field(default_factory=threading.Event)
    recoveries: list = field(default_factory=list)
    recovery_lock: threading.Lock = field(default_factory=threading.Lock)
    rerank_cache: dict = field(default_factory=dict)
    summary_state: object | None = None  # Shared by initial execution and Replan.

    @classmethod
    def for_answer(cls, *, has_attachments: bool, timeout_seconds: float | None = None):
        # Normal QA remains unlimited; an explicit caller deadline (evaluation)
        # is opt-in. Cancellation and finite recovery counts remain active.
        return cls(time.monotonic() + timeout_seconds if timeout_seconds is not None else float('inf'))

    @property
    def has_deadline(self) -> bool:
        return math.isfinite(self.deadline)

    def expired(self) -> bool:
        return self.cancelled.is_set() or time.monotonic() >= self.deadline


current_budget: ContextVar[RequestBudget | None] = ContextVar("request_budget", default=None)


def check_budget() -> None:
    budget = current_budget.get()
    if budget is not None and budget.cancelled.is_set():
        raise RequestCancelled("request_cancelled")
    if budget is not None and budget.expired():
        raise RequestBudgetExceeded("request_budget_exceeded")


def reserve_recovery(stage: str, action: str, minimum_seconds: float = 20) -> bool:
    """One execution recovery plus one bounded query normalization.

    Query normalization does not steal the evidence recovery slot. Everything
    shares the original deadline; source supplementation, answer repair and
    OOM repacking still compete for one execution recovery.
    """
    budget = current_budget.get()
    if budget is None:
        return True  # Non-HTTP callers retain their own bounded loop limits.
    with budget.recovery_lock:
        planning = stage == 'plan_request' and action == 'repair_query_language'
        same_category = [item for item in budget.recoveries
                         if (item['stage'] == 'plan_request'
                             and item['action'] == 'repair_query_language') == planning]
        if (budget.expired() or budget.deadline-time.monotonic() < minimum_seconds
                or len(same_category) >= 1
                or any(item['stage'] == stage for item in budget.recoveries)):
            return False
        budget.recoveries.append({'stage':stage,'action':action})
        return True


@contextmanager
def budget_scope(budget: RequestBudget):
    token = current_budget.set(budget)
    try:
        check_budget()
        yield
    finally:
        current_budget.reset(token)
