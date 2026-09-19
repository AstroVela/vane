# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded, query-fair task admission independent of worker placement.

The backend supplies a nonblocking capacity adapter. This policy acquires that
capacity and a runtime task allowance together, without parking an allowance
behind a busy worker pool. Ray query/generation authorization remains separate.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any

from vane.execution._diagnostics import bounded_utf8_text, exception_message_from_args, safe_exception_type_name
from vane.execution.udf_admission import AdmissionCapacity, AdmissionLease
from vane.execution.udf_lifecycle import ExecutionCancellationScope


def _failure_message(error: BaseException) -> str:
    message = exception_message_from_args(error)
    detail = bounded_utf8_text(message, 2048) if message is not None else "no simple diagnostic message"
    return f"{safe_exception_type_name(error)}: {detail}"


@dataclass(frozen=True)
class TaskAdmissionLimits:
    max_running_tasks: int
    max_queued_tasks: int

    def __post_init__(self) -> None:
        for name, minimum in (("max_running_tasks", 1), ("max_queued_tasks", 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


class TaskAdmissionQueueFull(RuntimeError):
    """The bounded pending-task queue is full; the caller may retry later."""


class RuntimeTaskAdmission:
    def __init__(self, limits: TaskAdmissionLimits) -> None:
        if not isinstance(limits, TaskAdmissionLimits):
            raise TypeError("task_limit must be TaskAdmissionLimits")
        self._limits = limits
        self._condition = threading.Condition()
        self._queries: set[QueryTaskAdmission] = set()
        self._waiting: deque[QueryTaskAdmission] = deque()
        self._leases: dict[str, RuntimeAdmissionAuthority] = {}
        self._running: set[str] = set()
        self._suspended: set[str] = set()
        self._resuming: deque[str] = deque()
        self._draining = False
        self._closed = False
        # All queries in this runtime share one pool-arbitration identity.
        self._capacity_wakeup = self._wake

    def open_query(self) -> QueryTaskAdmission:
        with self._condition:
            if self._draining or self._closed:
                raise RuntimeError("runtime task admission is draining or closed")
            query = QueryTaskAdmission(self)
            self._queries.add(query)
            return query

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "max_running_tasks": self._limits.max_running_tasks,
                "max_queued_tasks": self._limits.max_queued_tasks,
                "running_tasks": len(self._running),
                "ready_tasks": len(self._leases) - len(self._running) - len(self._suspended),
                "waiting_tasks": len(self._suspended),
                "resuming_tasks": len(self._resuming),
                "queued_tasks": sum(len(q._pending) for q in self._waiting),
                "queries": len(self._queries),
                "draining": self._draining,
                "closed": self._closed,
            }

    def drain(self) -> None:
        with self._condition:
            self._draining = True

    def close(self, *, timeout: float = 0.0) -> None:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("task admission close timeout must be finite and non-negative")
        deadline = time.monotonic() + timeout
        with self._condition:
            self._draining = True
            while self._queries:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("runtime task admission still has active queries or tasks")
                self._condition.wait(remaining)
            self._closed = True

    def _remove_pending_locked(self, authority: RuntimeAdmissionAuthority) -> None:
        query = authority._query
        query._pending.remove(authority)
        if not query._pending:
            self._waiting.remove(query)

    def _retire_locked(self, authority: RuntimeAdmissionAuthority) -> None:
        query = authority._query
        if authority._state == "closed" and not authority._tokens:
            query._authorities.discard(authority)
        self._retire_query_locked(query)

    def _retire_query_locked(self, query: QueryTaskAdmission) -> None:
        if query._closed and not query._authorities:
            self._queries.discard(query)
            self._condition.notify_all()

    def _dispatch_locked(self) -> list[RuntimeAdmissionAuthority]:
        wakeups: list[RuntimeAdmissionAuthority] = []
        # A backend closing must wake a blocked dispatcher even at full quota.
        for query in list(self._waiting):
            for authority in list(query._pending):
                try:
                    if authority._capacity.state()["state"] == "closed":
                        raise RuntimeError("backend admission capacity is closed")
                except BaseException as exc:
                    authority._error = _failure_message(exc)
                    authority._state = "failed"
                    self._remove_pending_locked(authority)
                    wakeups.append(authority)

        # Already submitted tasks own a worker and possibly input data. Resume
        # them before admitting fresh work once their transport can progress.
        while self._resuming and self._has_capacity_locked():
            token = self._resuming.popleft()
            self._suspended.remove(token)
            self._running.add(token)
            self._condition.notify_all()

        while self._waiting and self._has_capacity_locked():
            granted = False
            # One grant per eligible query per round. Within a query, skip busy
            # pools so one model cannot block an unrelated UDF or downstream work.
            for _ in range(len(self._waiting)):
                if not self._has_capacity_locked():
                    break
                query = self._waiting.popleft()
                for _ in range(len(query._pending)):
                    authority = query._pending.popleft()
                    try:
                        base = authority._capacity.try_acquire(authority._retained)
                    except BaseException as exc:
                        authority._error = _failure_message(exc)
                        authority._state = "failed"
                        wakeups.append(authority)
                        continue
                    if base is None:
                        query._pending.append(authority)
                        continue
                    token = uuid.uuid4().hex
                    self._leases[token] = authority
                    authority._tokens.add(token)
                    authority._ready = AdmissionLease(
                        request_id=authority._request_id,
                        retained_input_bytes=authority._retained,
                        lease=base.lease,
                        driver=base.driver,
                        _release_callback=base.release,
                        _execution_finished_callback=partial(self._complete_execution, token, base),
                        _capacity_wait_context=partial(self._suspend_for_wait, token),
                    )
                    authority._ready_token = token
                    authority._state = "ready"
                    wakeups.append(authority)
                    granted = True
                    break
                if query._pending:
                    self._waiting.append(query)
            if not granted:
                break
        return wakeups

    def _complete_execution(self, token: str, base: AdmissionLease) -> None:
        try:
            base.complete_execution()
        finally:
            self._finish(token)

    def _has_capacity_locked(self) -> bool:
        return len(self._leases) - len(self._suspended) < self._limits.max_running_tasks

    def _wake_resume_waiters(self) -> None:
        with self._condition:
            self._condition.notify_all()

    @contextmanager
    def _suspend_for_wait(self, token: str, scope: ExecutionCancellationScope) -> Iterator[None]:
        scope.raise_if_cancelled("runtime capacity wait")
        with self._condition:
            if token not in self._running:
                raise RuntimeError("only a running task can suspend execution capacity")
            self._running.remove(token)
            self._suspended.add(token)
            wakeups = self._dispatch_locked()
        self._notify(wakeups)
        # A failed/cancelled transport proceeds to backend cleanup. It must not
        # reacquire capacity just to terminate, or release its physical owner.
        yield
        unregister = scope.register_cancel_wakeup(self._wake_resume_waiters)
        try:
            with self._condition:
                scope.raise_if_cancelled("runtime capacity resume")
                if token not in self._leases:
                    raise RuntimeError("cannot resume a completed task")
                self._resuming.append(token)
                wakeups = self._dispatch_locked()
            self._notify(wakeups)
            with self._condition:
                while token in self._suspended:
                    scope.raise_if_cancelled("runtime capacity resume")
                    self._condition.wait()
                scope.raise_if_cancelled("runtime capacity resume")
                if token not in self._leases:
                    raise RuntimeError("cannot resume a completed task")
        finally:
            unregister()
            with self._condition:
                if token in self._resuming:
                    self._resuming.remove(token)

    def _notify(self, authorities: list[RuntimeAdmissionAuthority]) -> None:
        for authority in dict.fromkeys(authorities):
            with self._condition:
                callback = authority._wakeup
            if callback is not None:
                try:
                    callback()
                except BaseException as exc:
                    # Do not retain callback tracebacks (and entire requests), or
                    # prevent notifications to other queries after one fails.
                    with self._condition:
                        authority._error = f"admission wakeup failed: {_failure_message(exc)}"

    def _wake(self) -> None:
        with self._condition:
            wakeups = self._dispatch_locked()
        self._notify(wakeups)

    def _finish(self, token: str) -> None:
        with self._condition:
            authority = self._leases.pop(token, None)
            if authority is None:
                return
            self._running.discard(token)
            self._suspended.discard(token)
            if token in self._resuming:
                self._resuming.remove(token)
            self._condition.notify_all()
            authority._tokens.discard(token)
            self._retire_locked(authority)
            wakeups = self._dispatch_locked()
        self._notify(wakeups)


class QueryTaskAdmission:
    """Query-owned preparation resource; shutdown never closes shared models."""

    def __init__(self, runtime: RuntimeTaskAdmission) -> None:
        self._runtime = runtime
        self._authorities: set[RuntimeAdmissionAuthority] = set()
        self._pending: deque[RuntimeAdmissionAuthority] = deque()
        self._closed = False

    def create_authority(self, capacity: AdmissionCapacity) -> RuntimeAdmissionAuthority:
        with self._runtime._condition:
            if self._closed:
                raise RuntimeError("query task admission is closed")
            authority = RuntimeAdmissionAuthority(self, capacity)
            self._authorities.add(authority)
        try:
            capacity.register_capacity_wakeup(self._runtime._capacity_wakeup)
        except BaseException:
            authority.close()
            raise
        return authority

    def shutdown(self, *, kill: bool = False) -> None:
        with self._runtime._condition:
            self._closed = True
            authorities = list(self._authorities)
            self._runtime._retire_query_locked(self)
        errors = []
        for authority in authorities:
            try:
                authority.close()
            except BaseException as exc:
                errors.append(_failure_message(exc))
        if errors:
            raise RuntimeError("query admission cleanup failed: " + "; ".join(errors))

    def cleanup_pending(self) -> bool:
        with self._runtime._condition:
            return self in self._runtime._queries


class RuntimeAdmissionAuthority:
    """The existing dispatcher protocol, backed by query and runtime ownership."""

    def __init__(self, query: QueryTaskAdmission, capacity: AdmissionCapacity) -> None:
        self._query = query
        self._runtime = query._runtime
        self._capacity = capacity
        self._state = "idle"
        self._request_id = ""
        self._retained = 0
        self._ready: AdmissionLease | None = None
        self._ready_token = ""
        self._tokens: set[str] = set()
        self._error: str | None = None
        self._wakeup: Callable[[], None] | None = None

    def register_wakeup(self, callback: Callable[[], None]) -> None:
        with self._runtime._condition:
            self._wakeup = callback
            notify = self._state in {"ready", "failed"}
        if notify:
            self._runtime._notify([self])

    def request(self, retained_input_bytes: int) -> bool:
        retained = int(retained_input_bytes)
        if retained < 0:
            raise ValueError("retained_input_bytes must be >= 0")
        runtime = self._runtime
        with runtime._condition:
            if self._error:
                raise RuntimeError(self._error)
            if self._state == "closed" or self._query._closed:
                raise RuntimeError("query task admission is closed")
            if self._state != "idle":
                return False
            self._state = "requested"
            self._request_id = uuid.uuid4().hex
            self._retained = retained
            if not self._query._pending:
                runtime._waiting.append(self._query)
            self._query._pending.append(self)
            wakeups = runtime._dispatch_locked()
            full = self._state == "requested" and (
                sum(len(q._pending) for q in runtime._waiting) > runtime._limits.max_queued_tasks
            )
            if full:
                runtime._remove_pending_locked(self)
                self._state = "idle"
                self._retained = 0
                self._request_id = ""
        runtime._notify(wakeups)
        if full:
            raise TaskAdmissionQueueFull("runtime pending UDF task queue is full")
        return True

    def state(self) -> dict[str, Any]:
        with self._runtime._condition:
            if self._error:
                raise RuntimeError(self._error)
            return {"state": self._state, "available": self._state == "ready", "retained_input_bytes": self._retained}

    def take(self, retained_input_bytes: int) -> AdmissionLease:
        with self._runtime._condition:
            if self._error:
                raise RuntimeError(self._error)
            if self._state != "ready" or self._ready is None:
                raise RuntimeError("runtime admission lease is not ready")
            if int(retained_input_bytes) != self._retained:
                raise RuntimeError("runtime admission retained input bytes do not match")
            lease = self._ready
            self._runtime._running.add(self._ready_token)
            self._ready = None
            self._ready_token = ""
            self._state = "idle"
            self._retained = 0
            self._request_id = ""
            return lease

    def close(self) -> None:
        with self._runtime._condition:
            if self._state == "closed":
                return
            if self._state == "requested":
                self._runtime._remove_pending_locked(self)
            ready = self._ready
            self._ready = None
            self._ready_token = ""
            self._state = "closed"
            self._retained = 0
            self._wakeup = None
            self._error = None
            self._runtime._retire_locked(self)
        try:
            self._capacity.close()
        finally:
            if ready is not None:
                ready.release()


__all__ = ["QueryTaskAdmission", "RuntimeTaskAdmission", "TaskAdmissionLimits", "TaskAdmissionQueueFull"]
