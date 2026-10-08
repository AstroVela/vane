# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Application ownership of the embedded query service.

The service core owns sessions and their shared execution resources. A remote
submission protocol can host this same core later; this module exposes only
the application-owned, in-process coordinator.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from vane.execution.cleanup_deadline import cleanup_deadline, cleanup_timeout
from vane.execution.pipelined_plan import RayResources
from vane.execution.query_runtime import QueryResources
from vane.execution.request_admission import RequestAdmissionLimits, RuntimeRequestAdmission
from vane.execution.result_delivery import ResultDeliveryLimits, RuntimeResultDelivery


class _CloseAttempt:
    def __init__(self, deadline: float) -> None:
        self.deadline = deadline
        self.done = threading.Event()
        self.error: BaseException | None = None


class QueryService:
    """One application service, independent of individual SQL sessions."""

    def __init__(self, resources: RayResources) -> None:
        from vane.execution.pipelined_runtime import WorkerPool

        self.service_id = uuid.uuid4().hex
        self.resources = resources
        self.pool = WorkerPool(resources)
        self.admission = RuntimeRequestAdmission(
            RequestAdmissionLimits(resources.max_active_queries, resources.max_queued_queries)
        )
        self.delivery = RuntimeResultDelivery(
            ResultDeliveryLimits(resources.max_results, resources.result_buffer_bytes)
        )
        self.lock = threading.RLock()
        self.cleanup_lock = threading.Lock()
        self.close_attempt: _CloseAttempt | None = None
        self.sessions: dict[str, Any] = {}
        self.stores: dict[str, Any] = {}
        self.closing = False
        self.closed = False

    def session(self, execution: str, resources: QueryResources | None) -> Any:
        from vane.execution.pipelined_runtime import RayQueryRuntime

        limits = (
            resources
            if resources is not None
            else QueryResources(
                self.resources.max_active_queries,
                self.resources.max_queued_queries,
                self.resources.max_results,
                self.resources.result_buffer_bytes,
            )
        )
        if type(limits) is not QueryResources:
            raise TypeError("session resources must be QueryResources; configure workers on Runtime")
        for field in ("max_active_queries", "max_queued_queries", "max_results", "result_buffer_bytes"):
            if getattr(limits, field) > getattr(self.resources, field):
                raise ValueError(f"session {field} exceeds the Runtime capacity")
        with self.lock:
            if self.closing:
                raise RuntimeError("query service is closing")
            session = RayQueryRuntime(self, execution, limits)
            self.sessions[session.session_id] = session
            return session

    def retire(self, session: Any) -> None:
        with self.lock:
            self.sessions.pop(session.session_id, None)

    def exchange_store(self, name: str) -> Any:
        from vane.execution.fte_store import StorePool

        config = next((s for s in self.resources.exchange_stores if s.name == name), None)
        if config is None:
            raise ValueError(f"unknown registered exchange store: {name}")
        with self.lock:
            if name not in self.stores:
                self.stores[name] = StorePool(config)
            return self.stores[name]

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            sessions = tuple(self.sessions.values())
        return {
            "service_id": self.service_id,
            "sessions": {session.session_id: session.resource_snapshot() for session in sessions},
            "request_admission": self.admission.snapshot(),
            "result_delivery": self.delivery.snapshot(),
            "workers": self.pool.admission.snapshot(),
            "result_service": self.pool.results.snapshot(),
            "closing": self.closing,
            "closed": self.closed,
        }

    def close(self, deadline: float) -> None:
        # Cleanup can enter native code or a slow RPC submission. The service
        # retains one attempt so a caller's deadline never abandons its owners
        # or starts overlapping cleanup. A later close can wait or retry.
        self.closing = True
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            if not self.cleanup_lock.acquire(timeout=min(remaining, threading.TIMEOUT_MAX)):
                raise TimeoutError("service cleanup is pending; retry Runtime.close()")
            try:
                if self.closed:
                    return
                attempt = self.close_attempt
                started = attempt is None or attempt.done.is_set()
                if started:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("service cleanup is pending; retry Runtime.close()")
                    attempt = _CloseAttempt(deadline)
                    self.close_attempt = attempt
                    thread = threading.Thread(
                        target=self._run_close, args=(attempt,), name="vane-service-close", daemon=True
                    )
                    try:
                        thread.start()
                    except BaseException:
                        self.close_attempt = None
                        raise
            finally:
                self.cleanup_lock.release()
            assert attempt is not None
            remaining = max(0.0, deadline - time.monotonic())
            if not attempt.done.wait(min(remaining, threading.TIMEOUT_MAX)):
                raise TimeoutError("service cleanup is pending; retry Runtime.close()")
            if attempt.error is None:
                return
            if started:
                raise attempt.error
            # This invocation joined a previous attempt. Its failure may have
            # exhausted the previous caller's budget; retry with our remainder.

    def _run_close(self, attempt: _CloseAttempt) -> None:
        try:
            with cleanup_deadline(attempt.deadline):
                self._close()
        except BaseException as error:
            attempt.error = error.with_traceback(None)
        finally:
            attempt.done.set()

    def _close(self) -> None:
        self.admission.drain()
        with self.lock:
            sessions = tuple(self.sessions.values())
        errors = []
        for session in sessions:
            cleanup_timeout(10)
            try:
                session.drain()
            except BaseException as error:
                errors.append(error)
        for session in sessions:
            timeout = cleanup_timeout(10)
            try:
                session.close_session(timeout=timeout)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise RuntimeError("service cleanup is pending; retry Runtime.close()") from errors[0]
        self.delivery.close(timeout=cleanup_timeout(10))
        self.admission.close(timeout=cleanup_timeout(10))
        cleanup_timeout(10)
        self.pool.close()
        self.closed = True


class Runtime:
    """Own a lazy Ray query service shared by explicitly created SQL sessions.

    Constructing a Runtime does not initialize Ray or start actors. connect()
    creates its service core; the first query starts its Ray processes. Closing
    a connection closes only that session. close() shuts down all sessions and
    the shared processes, and can be retried when cleanup is still pending.
    """

    def __init__(self, resources: RayResources | None = None) -> None:
        self.resources = RayResources() if resources is None else resources
        if not isinstance(self.resources, RayResources):
            raise TypeError("Runtime resources must be RayResources")
        self._lock = threading.RLock()
        self._service: QueryService | None = None
        self._closing = False

    def _check_open(self) -> None:
        if self._closing:
            if self._service is not None:
                self._service.closing = True
            raise RuntimeError("Runtime is closed")

    def _new_session(self, execution: str, resources: QueryResources | None) -> Any:
        with self._lock:
            self._check_open()
            if self._service is None:
                self._service = QueryService(self.resources)
            # Construction can yield the GIL to a close that times out on our lock.
            self._check_open()
            return self._service.session(execution, resources)

    def connect(
        self,
        database: Any = ":memory:",
        *,
        read_only: bool = False,
        config: dict[str, Any] | None = None,
        execution: str = "pipelined",
        resources: QueryResources | None = None,
    ) -> Any:
        from vane._native import connect

        connection = connect(
            database,
            read_only=read_only,
            config={} if config is None else config,
            backend="ray",
            runtime=self,
            execution=execution,
            resources=resources,
        )
        with self._lock:
            closing = self._closing
        if closing:
            connection.close()
            raise RuntimeError("Runtime closed while opening a session")
        return connection

    def resource_snapshot(self) -> dict[str, Any]:
        with self._lock:
            service = self._service
        return {"started": service is not None, "service": None if service is None else service.snapshot()}

    def close(self, *, timeout: float = 10) -> None:
        started = time.monotonic()
        from vane._native.execution_runtime import check_entry
        from vane.execution.request_admission import _timeout

        check_entry()
        deadline = started + _timeout(timeout, "Runtime close timeout")
        self._closing = True
        # Fence existing sessions even if another caller is still holding the
        # Runtime lock. _new_session also fences a core constructed concurrently.
        service = self._service
        if service is not None:
            service.closing = True
        remaining = max(0.0, deadline - time.monotonic())
        if not self._lock.acquire(timeout=min(remaining, threading.TIMEOUT_MAX)):
            raise TimeoutError("Runtime cleanup is pending; retry Runtime.close()")
        try:
            service = self._service
        finally:
            self._lock.release()
        if service is not None:
            service.close(deadline)

    def __enter__(self) -> Runtime:
        with self._lock:
            self._check_open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
