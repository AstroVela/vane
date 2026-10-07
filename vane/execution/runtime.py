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

from vane.execution.pipelined_plan import RayResources
from vane.execution.query_runtime import QueryResources
from vane.execution.request_admission import RequestAdmissionLimits, RuntimeRequestAdmission
from vane.execution.result_delivery import ResultDeliveryLimits, RuntimeResultDelivery


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

    def close(self, timeout: float) -> None:
        with self.cleanup_lock:
            with self.lock:
                if self.closed:
                    return
                self.closing = True
                sessions = tuple(self.sessions.values())
            self.admission.drain()
            errors = []
            for session in sessions:
                try:
                    session.drain()
                except BaseException as error:
                    errors.append(error)
            deadline = time.monotonic() + timeout
            for session in sessions:
                try:
                    session.close(timeout=max(0.0, deadline - time.monotonic()))
                except BaseException as error:
                    errors.append(error)
            if errors:
                raise RuntimeError("service cleanup is pending; retry Runtime.close()") from errors[0]
            self.delivery.close(timeout=max(0.0, deadline - time.monotonic()))
            self.admission.close(timeout=max(0.0, deadline - time.monotonic()))
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

    def _new_session(self, execution: str, resources: QueryResources | None) -> Any:
        with self._lock:
            if self._closing:
                raise RuntimeError("Runtime is closed")
            if self._service is None:
                self._service = QueryService(self.resources)
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
        from vane._native.execution_runtime import check_entry
        from vane.execution.request_admission import _timeout

        check_entry()
        timeout = _timeout(timeout, "Runtime close timeout")
        with self._lock:
            self._closing = True
            service = self._service
        if service is not None:
            service.close(timeout)

    def __enter__(self) -> Runtime:
        with self._lock:
            if self._closing:
                raise RuntimeError("Runtime is closed")
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
