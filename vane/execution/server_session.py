# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Transport-independent ownership of leased, remote SQL sessions."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from vane.execution.pipelined_plan import RayResources
from vane.execution.query_options import QueryExecutionOptions, RayExecution
from vane.execution.query_runtime import QueryResources
from vane.execution.request_admission import _timeout
from vane.execution.runtime import Runtime
from vane.execution.server_query import ServerQuery


class SessionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message)
        self.code = code

    def __str__(self) -> str:
        return str(self.args[1])


@dataclass(frozen=True)
class SessionLimits:
    max_sessions: int = 64
    max_query_handles: int = 128
    lease_seconds: float = 60
    cleanup_timeout: float = 5
    maintenance_interval: float = 0.1

    def __post_init__(self) -> None:
        if type(self.max_sessions) is not int or not 0 < self.max_sessions <= 1024:
            raise ValueError("max_sessions must be between 1 and 1024")
        if type(self.max_query_handles) is not int or not 0 < self.max_query_handles <= 4096:
            raise ValueError("max_query_handles must be between 1 and 4096")
        for name in ("lease_seconds", "cleanup_timeout", "maintenance_interval"):
            if not 0 < _timeout(getattr(self, name), name) <= 86400:
                raise ValueError(f"{name} must be positive and at most one day")


@dataclass
class _Session:
    session_id: str
    expires_at: float
    owner: Any = None
    opening: bool = True
    closing: bool = False
    cleaning: bool = False
    retry_at: float = 0
    error: BaseException | None = None
    sequence: int = 0
    queries: dict[int, ServerQuery] = field(default_factory=dict)


class _SessionRuntime(Runtime):
    """Publish the native session owner before connection construction can fail."""

    def __init__(self, resources: RayResources | None) -> None:
        super().__init__(resources)
        self.opening: ContextVar[_Session | None] = ContextVar("vane_opening_session", default=None)

    def _new_session(self, execution: str, resources: QueryResources | None) -> Any:
        record = self.opening.get()
        if record is None:
            raise RuntimeError("server connections must be opened through SessionService")
        owner = super()._new_session(execution, resources)
        record.owner = owner
        return owner


class SessionService:
    """Own native sessions until cleanup completes, including abandoned opens.

    The registry lock only protects metadata. Native connection creation and
    cleanup run outside it. Each closing session owns at most one cleanup
    thread, so a blocked close cannot prevent another session's lease renewal.
    """

    def __init__(
        self,
        *,
        database: str = ":memory:",
        read_only: bool = False,
        config: dict[str, Any] | None = None,
        resources: RayResources | None = None,
        limits: SessionLimits | None = None,
        gateway: Any = None,
    ) -> None:
        self.limits = SessionLimits() if limits is None else limits
        if not isinstance(self.limits, SessionLimits):
            raise TypeError("limits must be SessionLimits")
        self.server_id = uuid.uuid4().hex
        self.runtime = _SessionRuntime(resources)
        self.gateway = gateway
        self._database = database
        self._read_only = read_only
        self._config = dict(config or {})
        self._condition = threading.Condition()
        self._sessions: dict[str, _Session] = {}
        self._closing = False
        self._closed = False
        self._maintenance = threading.Thread(target=self._maintain, name="vane-session-leases", daemon=True)
        self._maintenance.start()

    def open_session(self, *, execution: str = "pipelined", resources: QueryResources | None = None) -> dict[str, Any]:
        if execution not in ("pipelined", "fte"):
            raise ValueError("execution must be 'pipelined' or 'fte'")
        if resources is not None and type(resources) is not QueryResources:
            raise TypeError("session resources must be QueryResources")
        with self._condition:
            if self._closing:
                raise SessionError("SERVER_CLOSING", "server is closing")
            if len(self._sessions) >= self.limits.max_sessions:
                raise SessionError("SESSION_CAPACITY", "server session capacity is full")
            session = _Session(secrets.token_urlsafe(32), time.monotonic() + self.limits.lease_seconds)
            self._sessions[session.session_id] = session
        # Publish the owner before entering native code. A simultaneous server
        # close waits for this open, even if its RPC caller has already left.
        opening = self.runtime.opening.set(session)
        try:
            connection = self.runtime.connect(
                self._database,
                read_only=self._read_only,
                config=self._config,
                execution=execution,
                resources=resources,
            )
        except BaseException:
            with self._condition:
                # Keep the captured owner even if native connect's own cleanup
                # raised. Maintenance must confirm cleanup before retiring it.
                session.opening = False
                session.closing = True
                self._condition.notify_all()
            raise
        finally:
            self.runtime.opening.reset(opening)
        with self._condition:
            assert session.owner is connection.query_runtime
            session.opening = False
            if self._closing or session.closing or time.monotonic() >= session.expires_at:
                session.closing = True
                self._condition.notify_all()
                raise SessionError("SESSION_EXPIRED", "session expired while opening")
            return self._handle(session)

    def _handle(self, session: _Session) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "session_id": session.session_id,
            "lease_seconds": self.limits.lease_seconds,
            "state": "OPEN",
        }

    def _check_server(self, server_id: str) -> None:
        if server_id != self.server_id:
            raise SessionError("SERVER_CHANGED", "session belongs to a different server instance")

    def renew_session(self, server_id: str, session_id: str) -> dict[str, Any]:
        with self._condition:
            self._check_server(server_id)
            session = self._sessions.get(session_id)
            if self._closing:
                raise SessionError("SERVER_CLOSING", "server is closing")
            if session is None or session.opening or session.closing:
                raise SessionError("SESSION_EXPIRED", "session is unknown or closing")
            now = time.monotonic()
            if now >= session.expires_at:
                session.closing = True
                self._condition.notify_all()
                raise SessionError("SESSION_EXPIRED", "session lease expired")
            session.expires_at = now + self.limits.lease_seconds
            return self._handle(session)

    def close_session(self, server_id: str, session_id: str) -> dict[str, str]:
        with self._condition:
            self._check_server(server_id)
            session = self._sessions.get(session_id)
            if session is None:
                return {"state": "CLOSED"}
            session.closing = True
            self._condition.notify_all()
            # Acceptance is distinct from completion. The slot and native
            # owner remain registered until cleanup has actually succeeded.
            return {"state": "CLOSING"}

    def _live_session(self, server_id: str, session_id: str) -> _Session:
        self._check_server(server_id)
        session = self._sessions.get(session_id)
        if self._closing or session is None or session.opening or session.closing:
            raise SessionError("SESSION_EXPIRED", "session is unknown or closing")
        if time.monotonic() >= session.expires_at:
            session.closing = True
            self._condition.notify_all()
            raise SessionError("SESSION_EXPIRED", "session lease expired")
        return session

    def execute(
        self,
        server_id: str,
        session_id: str,
        *,
        sequence: int,
        sql: str,
        options: QueryExecutionOptions | None = None,
        rows_per_batch: int = 1024,
    ) -> dict[str, Any]:
        if self.gateway is None:
            raise SessionError("QUERY_UNAVAILABLE", "native result gateway is not configured")
        if type(sequence) is not int or not 1 <= sequence <= 2**53 - 1:
            raise ValueError("sequence must be an integer between 1 and 2**53-1")
        if not isinstance(sql, str) or not sql.strip() or len(sql.encode("utf-8")) > 32768:
            raise ValueError("SQL must contain 1 to 32768 UTF-8 bytes")
        if type(rows_per_batch) is not int or not 0 < rows_per_batch <= 2048:
            raise ValueError("rows_per_batch must be between 1 and 2048")
        if options is not None and not isinstance(options.target, RayExecution):
            raise ValueError("remote queries require RayExecution options")
        fingerprint = hashlib.sha256(
            json.dumps([sql, None if options is None else options.to_dict(), rows_per_batch], sort_keys=True).encode()
        ).hexdigest()
        with self._condition:
            session = self._live_session(server_id, session_id)
            previous = session.queries.get(sequence)
            if previous is not None:
                if previous.fingerprint != fingerprint:
                    raise SessionError("SUBMISSION_CONFLICT", "submission identity has different SQL or options")
                return previous.snapshot()
            if sequence <= session.sequence:
                raise SessionError("QUERY_RETIRED", "submission has already been closed")
            if sequence != session.sequence + 1:
                raise SessionError("SUBMISSION_ORDER", "submissions must use consecutive sequence numbers")
            if sum(len(s.queries) for s in self._sessions.values()) >= self.limits.max_query_handles:
                raise SessionError("QUERY_CAPACITY", "close completed query handles before submitting more SQL")
            query = ServerQuery(sequence, fingerprint, session.owner, self.gateway, self.limits.cleanup_timeout)
            session.queries[sequence] = query
            session.sequence = sequence
        thread = threading.Thread(
            target=query.run, args=(sql, options, rows_per_batch), name="vane-server-query", daemon=True
        )
        try:
            thread.start()
        except BaseException as error:
            with query.lock:
                query.state = "FAILED"
                query.error = {"code": "QUERY_FAILED", "message": str(error)[:4096]}
                query.done.set()
        return query.snapshot()

    def query_action(self, server_id: str, session_id: str, query_id: int, operation: str) -> dict[str, Any]:
        if type(query_id) is not int or not 1 <= query_id <= 2**53 - 1:
            raise ValueError("invalid query_id")
        if operation not in {"status", "cancel", "finish", "close"}:
            raise ValueError("invalid query operation")
        with self._condition:
            session = self._live_session(server_id, session_id)
            query = session.queries.get(query_id)
            if query is None:
                if query_id <= session.sequence and operation == "close":
                    return {"query_id": query_id, "state": "CLOSED", "cleaned": True}
                raise SessionError("QUERY_RETIRED", "query is unknown or already closed")
            if operation != "status":
                query.request(operation)
            if query.release_requested and query.done.is_set():
                del session.queries[query_id]
                return {"query_id": query_id, "state": "CLOSED", "cleaned": True}
            return query.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            sessions = tuple(self._sessions.values())
            return {
                "server_id": self.server_id,
                "state": "CLOSED" if self._closed else "DRAINING" if self._closing else "READY",
                "max_sessions": self.limits.max_sessions,
                "max_query_handles": self.limits.max_query_handles,
                "queries": sum(len(s.queries) for s in sessions),
                "sessions": len(sessions),
                "opening": sum(s.opening for s in sessions),
                "closing": sum(s.closing for s in sessions),
                "cleanup_pending": sum(s.error is not None for s in sessions),
                "lease_seconds": self.limits.lease_seconds,
            }

    def _maintain(self) -> None:
        while True:
            with self._condition:
                if self._closing and not self._sessions:
                    return
                now = time.monotonic()
                pending = []
                for session in tuple(self._sessions.values()):
                    if self._closing or now >= session.expires_at:
                        session.closing = True
                    if not session.closing or session.opening or session.cleaning or now < session.retry_at:
                        continue
                    session.cleaning = True
                    pending.append(session)
                if not pending:
                    self._condition.wait(self.limits.maintenance_interval)
                    continue
            # Thread startup may itself wait for the interpreter/OS. Do not
            # hold the registry lock while starting any cleanup operation.
            for session in pending:
                thread = threading.Thread(target=self._cleanup, args=(session,), name="vane-session-close", daemon=True)
                try:
                    thread.start()
                except BaseException as error:
                    with self._condition:
                        session.cleaning = False
                        session.error = error.with_traceback(None)
                        session.retry_at = time.monotonic() + self.limits.maintenance_interval
                        self._condition.notify_all()

    def _cleanup(self, session: _Session) -> None:
        error: BaseException | None = None
        try:
            deadline = time.monotonic() + self.limits.cleanup_timeout
            for query in session.queries.values():
                query.request("close")
            for query in session.queries.values():
                if not query.done.wait(max(0, deadline - time.monotonic())):
                    raise TimeoutError("session query cleanup is pending")
            if session.owner is not None:
                session.owner.close_session(timeout=max(0, deadline - time.monotonic()))
        except BaseException as caught:
            error = caught.with_traceback(None)
        with self._condition:
            session.cleaning = False
            session.error = error
            if error is None:
                del self._sessions[session.session_id]
            else:
                session.retry_at = time.monotonic() + self.limits.maintenance_interval
            self._condition.notify_all()

    def close(self, *, timeout: float = 10) -> None:
        deadline = time.monotonic() + _timeout(timeout, "server close timeout")
        self._closing = True
        remaining = max(0, deadline - time.monotonic())
        if not self._condition.acquire(timeout=min(remaining, threading.TIMEOUT_MAX)):
            raise TimeoutError("session cleanup is pending; retry Server.close()")
        try:
            self._condition.notify_all()
            while self._sessions:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("session cleanup is pending; retry Server.close()")
                self._condition.wait(min(remaining, threading.TIMEOUT_MAX))
        finally:
            self._condition.release()
        self.runtime.close(timeout=max(0, deadline - time.monotonic()))
        self._maintenance.join(min(max(0, deadline - time.monotonic()), threading.TIMEOUT_MAX))
        if self._maintenance.is_alive():
            raise TimeoutError("session maintenance is stopping; retry Server.close()")
        self._closed = True
