"""Cooperative request deadline; cancellation never releases a running worker slot."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import threading
import time


class RequestBudgetExceeded(RuntimeError):
    pass


@dataclass
class RequestBudget:
    deadline: float
    cancelled: threading.Event = field(default_factory=threading.Event)
    recoveries: list = field(default_factory=list)
    recovery_lock: threading.Lock = field(default_factory=threading.Lock)

    def expired(self) -> bool:
        return self.cancelled.is_set() or time.monotonic() >= self.deadline


current_budget: ContextVar[RequestBudget | None] = ContextVar("request_budget", default=None)


def check_budget() -> None:
    budget = current_budget.get()
    if budget is not None and budget.expired():
        raise RequestBudgetExceeded("request_budget_exceeded")


def reserve_recovery(stage: str, action: str, minimum_seconds: float = 20) -> bool:
    """One recovery per stage, two across the request; never extend deadline."""
    budget = current_budget.get()
    if budget is None:
        return True  # Non-HTTP callers retain their own bounded loop limits.
    with budget.recovery_lock:
        if (budget.expired() or budget.deadline-time.monotonic() < minimum_seconds
                or len(budget.recoveries) >= 2
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
