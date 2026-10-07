# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded request admission, independent of execution and transport adapters."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from vane.execution.udf_admission import AdmissionLease
from vane.execution.udf_lifecycle import ExecutionCancelledError


def _timeout(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


@dataclass(frozen=True)
class RequestAdmissionLimits:
    max_active_requests: int
    max_queued_requests: int
    queue_timeout: float = 30.0

    def __post_init__(self) -> None:
        if type(self.max_active_requests) is not int or self.max_active_requests <= 0:
            raise ValueError("max_active_requests must be a positive integer")
        if type(self.max_queued_requests) is not int or self.max_queued_requests < 0:
            raise ValueError("max_queued_requests must be a non-negative integer")
        _timeout(self.queue_timeout, "queue_timeout")


class RequestQueueFull(RuntimeError):
    """The request queue has no remaining entry."""


class RequestQueueTimeout(TimeoutError):
    """A request's queue deadline expired before admission."""


class RequestCancelled(ExecutionCancelledError):
    """A request was cancelled before its result was returned."""


class RequestExecutionTimeout(TimeoutError):
    """A claimed request exceeded its execution deadline."""


RequestCancellationReason = Literal["cancelled", "admission_timeout", "execution_timeout"]


class RuntimeRequestAdmission:
    """FIFO admission; a claimed lease lasts through confirmed query cleanup.

    Only ticket metadata lives here. No plans, input buffers, model borrows,
    callbacks, worker slots or byte reservations belong to the queue.
    """

    def __init__(self, limits: RequestAdmissionLimits) -> None:
        if not isinstance(limits, RequestAdmissionLimits):
            raise TypeError("request_limit must be RequestAdmissionLimits")
        self.limits = limits
        self._condition = threading.Condition()
        self._queued: dict[int, RequestTicket] = {}
        self._active: dict[int, RequestTicket] = {}
        self._next_id = 0
        self._draining = False
        self._closed = False
        self._admitted = self._completed = self._cancelled = self._timed_out = self._rejected = 0
        self._drained = 0
        self._execution_timed_out = 0
        self._queue_wait_seconds = 0.0
        self._executed = self._failed_executions = 0
        self._execution_seconds = self._cleanup_seconds = 0.0

    def _dispatch_locked(self) -> None:
        now = time.monotonic()
        changed = False
        for key, ticket in tuple(self._queued.items()):
            if now >= ticket._deadline:
                del self._queued[key]
                ticket._state = "timed_out"
                self._timed_out += 1
                changed = True
        if not self._draining:
            for key, ticket in tuple(self._queued.items()):
                if self._can_admit(ticket._scope):
                    del self._queued[key]
                    self._admit_locked(ticket, now)
                    changed = True
        if changed:
            self._condition.notify_all()

    def _admit_locked(self, ticket: RequestTicket, now: float) -> None:
        ticket._state = "ready"
        ticket._admitted_at = now
        ticket._queue_wait = max(0.0, now - ticket._created)
        self._active[ticket.request_id] = ticket
        self._admitted += 1
        self._queue_wait_seconds += ticket._queue_wait

    def _can_admit(self, scope: RequestAdmissionScope | None) -> bool:
        return len(self._active) < self.limits.max_active_requests and (
            scope is None or sum(t._scope is scope for t in self._active.values()) < scope.limits.max_active_requests
        )

    def scope(self, limits: RequestAdmissionLimits) -> RequestAdmissionScope:
        return RequestAdmissionScope(self, limits)

    def request(
        self, *, queue_timeout: float | None = None, _scope: RequestAdmissionScope | None = None
    ) -> RequestTicket:
        timeout = _timeout(self.limits.queue_timeout if queue_timeout is None else queue_timeout, "queue_timeout")
        with self._condition:
            if self._draining or (_scope is not None and _scope._draining):
                raise RuntimeError("request admission is draining")
            self._dispatch_locked()
            if not self._can_admit(_scope):
                if len(self._queued) >= self.limits.max_queued_requests or (
                    _scope is not None
                    and sum(t._scope is _scope for t in self._queued.values()) >= _scope.limits.max_queued_requests
                ):
                    self._rejected += 1
                    raise RequestQueueFull("runtime request queue is full")
                if timeout == 0:
                    self._timed_out += 1
                    raise RequestQueueTimeout("request queue deadline expired")
            self._next_id += 1
            ticket = RequestTicket(self, self._next_id, timeout, _scope)
            if self._can_admit(_scope):
                self._admit_locked(ticket, time.monotonic())
            else:
                self._queued[ticket.request_id] = ticket
            return ticket

    def require_open(self) -> None:
        with self._condition:
            if self._draining:
                raise RuntimeError("request admission is draining")

    def require_claimed(self, ticket: RequestTicket | None) -> None:
        """Authorize preparation only for this runtime's live execution owner."""
        with self._condition:
            if (
                ticket is None
                or ticket._runtime is not self
                or self._active.get(ticket.request_id) is not ticket
                or ticket._state != "running"
            ):
                raise RuntimeError("request preparation requires a live claim from this runtime")
            if ticket._cancel_reason is not None:
                raise ticket.cancellation_error()

    def _release(self, ticket: RequestTicket) -> None:
        with self._condition:
            if self._active.pop(ticket.request_id, None) is None:
                return
            now = time.monotonic()
            ticket._finish_execution_locked(now, failed=False)
            ticket._released_at = now
            assert ticket._execution_finished_at is not None
            self._cleanup_seconds += max(0.0, now - ticket._execution_finished_at)
            if ticket._cancel_reason == "admission_timeout":
                ticket._state = "timed_out"
                self._timed_out += 1
            elif ticket._cancel_reason == "execution_timeout":
                ticket._state = "execution_timed_out"
                self._execution_timed_out += 1
            elif ticket._cancel_reason is not None:
                ticket._state = "cancelled"
                self._cancelled += 1
            else:
                ticket._state = "finished"
                self._completed += 1
            self._dispatch_locked()
            self._condition.notify_all()

    def drain(self, *, _scope: RequestAdmissionScope | None = None) -> None:
        with self._condition:
            if _scope is None:
                self._draining = True
            else:
                _scope._draining = True
            for ticket in (*self._queued.values(), *self._active.values()):
                if _scope is not None and ticket._scope is not _scope:
                    continue
                if ticket._state in {"queued", "ready"}:
                    ticket._state = "drained"
                    self._drained += 1
            self._queued = {key: ticket for key, ticket in self._queued.items() if ticket._state != "drained"}
            self._active = {key: ticket for key, ticket in self._active.items() if ticket._state != "drained"}
            self._dispatch_locked()
            self._condition.notify_all()

    def close(self, *, timeout: float = 0.0) -> None:
        deadline = time.monotonic() + _timeout(timeout, "request close timeout")
        self.drain()
        with self._condition:
            while self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("request runtime still has active execution or cleanup")
                self._condition.wait(min(remaining, threading.TIMEOUT_MAX))
            self._closed = True

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            self._dispatch_locked()
            return {
                "max_active_requests": self.limits.max_active_requests,
                "max_queued_requests": self.limits.max_queued_requests,
                "active_requests": len(self._active),
                "ready_requests": sum(ticket._state == "ready" for ticket in self._active.values()),
                "running_requests": sum(ticket._state == "running" for ticket in self._active.values()),
                "cancelling_requests": sum(ticket._cancel_reason is not None for ticket in self._active.values()),
                "queued_requests": len(self._queued),
                "admitted_requests": self._admitted,
                "completed_requests": self._completed,
                "cancelled_requests": self._cancelled,
                "drained_requests": self._drained,
                "timed_out_requests": self._timed_out,
                "execution_timed_out_requests": self._execution_timed_out,
                "rejected_requests": self._rejected,
                "queue_wait_seconds": self._queue_wait_seconds,
                "executed_requests": self._executed,
                "failed_executions": self._failed_executions,
                "execution_seconds": self._execution_seconds,
                "cleanup_seconds": self._cleanup_seconds,
                "draining": self._draining,
                "closed": self._closed,
            }


class RequestAdmissionScope:
    """Session limits enforced atomically with the enclosing service queue.

    FIFO order is preserved within each session. A session at capacity does not
    prevent another session from using an available service slot. Historical
    execution counters belong to the enclosing service; this view reports its
    own live occupancy only.
    """

    def __init__(self, runtime: RuntimeRequestAdmission, limits: RequestAdmissionLimits) -> None:
        if (
            limits.max_active_requests > runtime.limits.max_active_requests
            or limits.max_queued_requests > runtime.limits.max_queued_requests
        ):
            raise ValueError("session admission exceeds service capacity")
        self.runtime, self.limits = runtime, limits
        self._draining = False
        self._closed = False

    def request(self, *, queue_timeout: float | None = None) -> RequestTicket:
        return self.runtime.request(queue_timeout=queue_timeout, _scope=self)

    def drain(self) -> None:
        self.runtime.drain(_scope=self)

    def close(self, *, timeout: float = 0) -> None:
        deadline = time.monotonic() + _timeout(timeout, "request close timeout")
        self.drain()
        with self.runtime._condition:
            while any(t._scope is self for t in self.runtime._active.values()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("session still has active execution or cleanup")
                self.runtime._condition.wait(min(remaining, threading.TIMEOUT_MAX))
            self._closed = True

    def snapshot(self) -> dict[str, Any]:
        with self.runtime._condition:
            self.runtime._dispatch_locked()
            active = [t for t in self.runtime._active.values() if t._scope is self]
            return {
                "max_active_requests": self.limits.max_active_requests,
                "max_queued_requests": self.limits.max_queued_requests,
                "active_requests": len(active),
                "ready_requests": sum(t._state == "ready" for t in active),
                "running_requests": sum(t._state == "running" for t in active),
                "cancelling_requests": sum(t._cancel_reason is not None for t in active),
                "queued_requests": sum(t._scope is self for t in self.runtime._queued.values()),
                "draining": self._draining or self.runtime._draining,
                "closed": self._closed,
            }


class RequestTicket:
    def __init__(
        self,
        runtime: RuntimeRequestAdmission,
        request_id: int,
        timeout: float,
        scope: RequestAdmissionScope | None = None,
    ) -> None:
        self._runtime = runtime
        self._scope = scope
        self.request_id = request_id
        self._created = time.monotonic()
        self._deadline = self._created + timeout
        self._state = "queued"
        self._queue_wait = 0.0
        self._cancel_reason: RequestCancellationReason | None = None
        self._claimed_at: float | None = None
        self._admitted_at: float | None = None
        self._execution_finished_at: float | None = None
        self._released_at: float | None = None
        self._deferred_admission = False

    def _finish_admission_locked(self, now: float) -> None:
        if self._deferred_admission:
            waited = max(0.0, now - self._created)
            self._runtime._queue_wait_seconds += waited - self._queue_wait
            self._queue_wait = waited
            self._deferred_admission = False

    def _finish_execution_locked(self, now: float, *, failed: bool) -> None:
        if self._execution_finished_at is not None or (self._claimed_at is None and not self._deferred_admission):
            return
        self._finish_admission_locked(now)
        self._execution_finished_at = now
        if self._claimed_at is None:
            return
        self._runtime._executed += 1
        self._runtime._failed_executions += int(failed)
        self._runtime._execution_seconds += max(0.0, now - self._claimed_at)

    def finish_execution(self, *, failed: bool = False) -> None:
        """Record an adapter's execution boundary once, without releasing its slot.

        Preparation and cancellation callbacks are execution time. Confirmed
        cleanup is measured separately, including time waiting for retries.
        No exception, plan, or per-request history is retained by the counters.
        """
        with self._runtime._condition:
            self._finish_execution_locked(time.monotonic(), failed=failed)

    def timing_snapshot(self) -> dict[str, float | None]:
        """Completed intervals only; unstarted or unfinished intervals are None."""
        with self._runtime._condition:
            return {
                "queue_wait_seconds": self._queue_wait if self._admitted_at is not None else None,
                "execution_seconds": (
                    max(0.0, self._execution_finished_at - self._claimed_at)
                    if self._execution_finished_at is not None and self._claimed_at is not None
                    else None
                ),
                "cleanup_seconds": (
                    max(0.0, self._released_at - self._execution_finished_at)
                    if self._released_at is not None and self._execution_finished_at is not None
                    else None
                ),
            }

    @property
    def state(self) -> str:
        with self._runtime._condition:
            self._runtime._dispatch_locked()
            return "cancelling" if self._state == "running" and self._cancel_reason is not None else self._state

    @property
    def claimed_at(self) -> float:
        with self._runtime._condition:
            if self._claimed_at is None:
                raise RuntimeError("request execution has not been claimed")
            return self._claimed_at

    @property
    def admission_deadline(self) -> float:
        return self._deadline

    def start_execution(self) -> float:
        """Start execution after the adapter obtains its worker reservation."""
        with self._runtime._condition:
            if self._cancel_reason is not None:
                raise self.cancellation_error()
            if self._state != "running" or self._execution_finished_at is not None:
                raise RuntimeError("request has no live execution lease")
            if self._claimed_at is None:
                now = time.monotonic()
                if now >= self._deadline:
                    self._cancel_reason = "admission_timeout"
                    raise RequestQueueTimeout("worker resource admission deadline exceeded")
                self._finish_admission_locked(now)
                self._claimed_at = now
            return self._claimed_at

    @property
    def cancellation_reason(self) -> RequestCancellationReason | None:
        with self._runtime._condition:
            return self._cancel_reason

    def cancellation_error(self) -> RequestCancelled | RequestQueueTimeout | RequestExecutionTimeout:
        with self._runtime._condition:
            if self._cancel_reason == "admission_timeout":
                return RequestQueueTimeout("worker resource admission deadline exceeded")
            if self._cancel_reason == "execution_timeout":
                return RequestExecutionTimeout("request execution deadline exceeded")
            return RequestCancelled("request execution cancelled")

    def cancel_running(self, *, reason: RequestCancellationReason = "cancelled") -> bool:
        """Record cancellation without returning the execution's cleanup lease."""
        if reason not in {"cancelled", "admission_timeout", "execution_timeout"}:
            raise ValueError("unknown request cancellation reason")
        with self._runtime._condition:
            if self._state != "running" or self._cancel_reason is not None:
                return False
            self._cancel_reason = reason
            return True

    def take(self, *, before_claim: Callable[[], None] | None = None, defer_execution: bool = False) -> AdmissionLease:
        """Wait for this ticket, then transfer its slot to one execution.

        An optional metadata reservation runs under the admission lock only
        when this ticket is ready. A refusal leaves it ready for retry. The
        reservation must not wait or call execution/transport callbacks.
        """
        runtime = self._runtime
        with runtime._condition:
            while True:
                runtime._dispatch_locked()
                if self._state == "ready":
                    lease = AdmissionLease(
                        request_id=str(self.request_id),
                        retained_input_bytes=0,
                        lease={},
                        _release_callback=lambda: runtime._release(self),
                    )
                    if before_claim is not None:
                        before_claim()
                    self._state = "running"
                    self._deferred_admission = defer_execution
                    if not defer_execution:
                        self._claimed_at = time.monotonic()
                    return lease
                if self._state == "timed_out":
                    raise RequestQueueTimeout("request queue deadline expired")
                if self._state == "cancelled":
                    raise RequestCancelled("request cancelled before execution")
                if self._state == "drained":
                    raise RequestCancelled("request cancelled by runtime drain")
                if self._state != "queued":
                    raise RuntimeError("request ticket has already been taken")
                runtime._condition.wait(min(threading.TIMEOUT_MAX, max(0.0, self._deadline - time.monotonic())))

    def cancel(self) -> bool:
        """Cancel queued/unclaimed work; a running execution owns its lease."""
        runtime = self._runtime
        with runtime._condition:
            runtime._dispatch_locked()
            if self._state not in {"queued", "ready"}:
                return False
            runtime._queued.pop(self.request_id, None)
            runtime._active.pop(self.request_id, None)
            self._state = "cancelled"
            self._cancel_reason = "cancelled"
            runtime._cancelled += 1
            runtime._dispatch_locked()
            runtime._condition.notify_all()
            return True
