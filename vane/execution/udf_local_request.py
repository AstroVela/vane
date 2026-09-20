# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Execute an admitted local request and retain ownership through cleanup."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from vane.execution.request_admission import RequestTicket
from vane.execution.udf_actor_pool_lifecycle import actor_pool_cleanup_pending, rollback_actor_pools
from vane.execution.udf_admission import AdmissionLease

if TYPE_CHECKING:
    from vane.execution.udf_local_model import LocalModelRuntime


def _execute_native(conn: Any, plan: Any) -> Any:
    from vane._ray_cxx import require_ray_cxx_attr

    return require_ray_cxx_attr("DistributedPhysicalPlanRunner")().execute_native(conn, plan)


def _shutdown_resource(resource: Any, *, kill: bool) -> None:
    resource.shutdown(kill=kill)
    if actor_pool_cleanup_pending(resource):
        raise RuntimeError("request resource cleanup is still in progress")


class LocalModelRequest:
    """One request ticket, executed once on a caller-owned, independent cursor.

    Queue cancellation never interrupts a running query. Once execution starts,
    native interruption and UDF cancellation retain their existing ownership.
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

    @property
    def state(self) -> str:
        return self._ticket.state

    def cancel(self) -> bool:
        return self._ticket.cancel()

    def execute(self, plan: Any, bindings: Mapping[str, str], *, conn: Any) -> Any:
        """Admit, prepare and execute a bound plan; retain returned output owners."""
        with self._lock:
            if self._used:
                raise RuntimeError("a local request can only execute once")
            self._used = True
            self._executing = True
        try:
            self._lease = self._ticket.take()
            self._resources = self._runtime._prepare(plan, bindings, conn=conn, request_ticket=self._ticket)
            result = _execute_native(conn, plan)
        except BaseException as error:
            with self._lock:
                self._resources.extend(getattr(error, "owned_actor_pools", ()))
                self._executing = False
            try:
                self.shutdown(kill=True)
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise
        else:
            with self._lock:
                self._executing = False
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
