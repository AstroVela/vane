# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Execute an admitted local request and retain ownership through cleanup."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from vane.execution.request_admission import RequestCancellationReason, RequestTicket, _timeout
from vane.execution.request_deadline import RequestExecutionDeadline
from vane.execution.udf_actor_pool_lifecycle import actor_pool_cleanup_pending, rollback_actor_pools
from vane.execution.udf_admission import AdmissionLease
from vane.execution.udf_lifecycle import ExecutionCancellationScope

if TYPE_CHECKING:
    from vane.execution.udf_local_model import LocalModelRuntime


class _NativeRequestCancellation:
    """Bind interruption after query startup, and fence callbacks before reuse."""

    def __init__(self, cancellation: ExecutionCancellationScope) -> None:
        self._cancellation = cancellation
        self._lock = threading.Lock()
        self._conn: Any = None
        self._active = True
        self._unregister = cancellation.register_cancel_wakeup(self._interrupt)

    def _interrupt(self) -> None:
        with self._lock:
            if self._active and self._conn is not None:
                self._conn.interrupt()

    def started(self, conn: Any) -> None:
        with self._lock:
            if not self._active:
                return
            self._conn = conn
            # DuckDB resets the interrupt flag during startup. Replay an
            # earlier cancellation only after that reset, on the actual cursor.
            if self._cancellation.is_set():
                conn.interrupt()

    def close(self) -> None:
        with self._lock:
            # Unregister alone cannot fence a callback already copied by cancel.
            self._active = False
            self._conn = None
        self._unregister()


def _execute_native(conn: Any, plan: Any, *, cancellation: ExecutionCancellationScope) -> Any:
    from vane._ray_cxx import require_ray_cxx_attr

    binding = _NativeRequestCancellation(cancellation)
    try:
        return require_ray_cxx_attr("DistributedPhysicalPlanRunner")().execute_native(
            conn, plan, native_execution_started=binding.started
        )
    finally:
        binding.close()


def _shutdown_resource(resource: Any, *, kill: bool) -> None:
    resource.shutdown(kill=kill)
    if actor_pool_cleanup_pending(resource):
        raise RuntimeError("request resource cleanup is still in progress")


class LocalModelRequest:
    """One request ticket, executed once on a caller-owned, independent cursor.

    Cancellation interrupts native execution and only this request's UDF scopes.
    A request slot is returned only after its query resources confirm cleanup.
    """

    def __init__(self, runtime: LocalModelRuntime, ticket: RequestTicket) -> None:
        self._runtime = runtime
        self._ticket = ticket
        self._lock = threading.Lock()
        self._used = False
        self._executing = False
        self._cleaning = False
        self._lease: AdmissionLease | None = None
        self._resources: list[Any] = []
        self._cancellation = ExecutionCancellationScope(uuid.uuid4().hex, 1)
        self._cancel_finished = threading.Event()
        self._cancel_finished.set()
        self._deadline: RequestExecutionDeadline | None = None

    @property
    def state(self) -> str:
        return self._ticket.state

    @property
    def cancellation_reason(self) -> RequestCancellationReason | None:
        return self._ticket.cancellation_reason

    def cancel(self) -> bool:
        """Cancel once; keep running work charged until cleanup is confirmed."""
        with self._lock:
            if self._ticket.cancel():
                return True
            if not self._begin_cancellation_locked("cancelled"):
                return False
        self._dispatch_cancellation("cancelled")
        return True

    def _begin_cancellation_locked(self, reason: RequestCancellationReason) -> bool:
        if not self._executing or not self._ticket.cancel_running(reason=reason):
            return False
        self._cancel_finished.clear()
        return True

    def _dispatch_cancellation(self, reason: RequestCancellationReason) -> None:
        try:
            self._cancellation.cancel(
                "local request execution deadline exceeded"
                if reason == "execution_timeout"
                else "local request cancelled"
            )
        finally:
            self._cancel_finished.set()

    def _expire_deadline(self) -> None:
        with self._lock:
            if self._deadline is None or not self._deadline.expired():
                return
            if not self._begin_cancellation_locked("execution_timeout"):
                return
        self._dispatch_cancellation("execution_timeout")

    def _finish_execution(self) -> bool:
        while True:
            with self._lock:
                # A delayed watcher must not publish an overdue result. The
                # same lock arbitrates completion, manual cancel and expiry.
                expire = (
                    self._deadline is not None
                    and self._deadline.expired()
                    and self._begin_cancellation_locked("execution_timeout")
                )
                # Native interruption may return before cancellation has closed
                # buffered UDF results. Its callbacks remain execution owners.
                cancelled = self._ticket.state == "cancelling"
                if not expire and (not cancelled or self._cancel_finished.is_set()):
                    self._executing = False
                    if self._deadline is not None:
                        self._deadline.close()
                    self._cancellation.finish()
                    return cancelled
            if expire:
                self._dispatch_cancellation("execution_timeout")
            else:
                self._cancel_finished.wait()

    def execute(
        self, plan: Any, bindings: Mapping[str, str], *, conn: Any, execution_timeout: float | None = None
    ) -> Any:
        """Admit, prepare and execute a bound plan; retain returned output owners."""
        timeout = None if execution_timeout is None else _timeout(execution_timeout, "execution_timeout")
        with self._lock:
            if self._used:
                raise RuntimeError("a local request can only execute once")
            self._used = True
            self._executing = True
        try:
            self._lease = self._ticket.take()
            if timeout is not None:
                with self._lock:
                    self._deadline = RequestExecutionDeadline(self._ticket.claimed_at, timeout, self._expire_deadline)
                self._deadline.start()
                self._expire_deadline()
            self._cancellation.raise_if_cancelled("local request preparation")
            self._resources = self._runtime._prepare(
                plan, bindings, conn=conn, request_ticket=self._ticket, request_cancellation=self._cancellation
            )
            self._expire_deadline()
            # Cancellation is recorded before its scope is signalled. Honor
            # that decision even while the dispatcher has not resumed yet.
            if self._ticket.cancellation_reason is not None:
                raise self._ticket.cancellation_error()
            self._cancellation.raise_if_cancelled("local request preparation")
            result = _execute_native(conn, plan, cancellation=self._cancellation)
        except BaseException as error:
            with self._lock:
                self._resources.extend(getattr(error, "owned_actor_pools", ()))
            cancelled = self._finish_execution()
            primary: BaseException
            if cancelled and isinstance(error, Exception):
                primary = self._ticket.cancellation_error()
                if isinstance(error, type(primary)):
                    primary = error
                else:
                    primary.__cause__ = error
            else:
                primary = error
            try:
                self.shutdown(kill=True)
            except BaseException as cleanup_error:
                raise primary from cleanup_error
            raise primary
        else:
            if self._finish_execution():
                try:
                    self.shutdown(kill=True)
                except BaseException as cleanup_error:
                    raise self._ticket.cancellation_error() from cleanup_error
                raise self._ticket.cancellation_error()
            self.shutdown()
            return result

    def shutdown(self, *, kill: bool = False) -> None:
        """Cancel unstarted work or retry cleanup after execution has returned."""
        self._ticket.cancel()
        with self._lock:
            if self._executing:
                # A cancelled waiter owns no execution. Its take() will wake
                # and raise; do not interfere with a claimed execution lease.
                if self._ticket.state in {"cancelled", "drained", "timed_out"}:
                    return
                raise RuntimeError("request execution must finish before cleanup")
            if self._cleaning:
                raise RuntimeError("request cleanup is still in progress")
            self._cleaning = True
            resources = self._resources
        errors: list[BaseException] = []
        try:
            remaining = rollback_actor_pools(
                resources,
                RuntimeError("request cleanup"),
                shutdown=lambda resource: _shutdown_resource(resource, kill=kill),
                cleanup_pending=actor_pool_cleanup_pending,
                record_error=errors.append,
            )
            with self._lock:
                self._resources = remaining
            self._runtime._retain_request_cleanup(self, pending=bool(remaining))
            if not remaining and self._lease is not None:
                self._lease.release()
            if errors:
                raise RuntimeError("request cleanup failed; retry request.shutdown() or runtime.close()") from errors[0]
        finally:
            with self._lock:
                self._cleaning = False

    def __enter__(self) -> LocalModelRequest:
        return self

    def __exit__(self, _type: object, error: BaseException | None, _traceback: object) -> None:
        try:
            self.shutdown(kill=error is not None)
        except BaseException as cleanup_error:
            if error is not None:
                raise error from cleanup_error
            raise
