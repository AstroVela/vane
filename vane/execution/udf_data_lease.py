# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Runtime-scoped accounting and optional strict admission of live UDF data.

The ledger records identities and sizes rather than data buffers or callers.
Input borrows and cleanup owners last through successful transport cleanup
after backend completion. Output owners can outlive a query or runtime,
including through zero-copy consumer views.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from vane.execution.byte_budget import ByteBudgetUsage, byte_budget_block_reason
from vane.execution.data_lifecycle import _OUTPUT_STATES, OutputBlockLeaseOwner
from vane.execution.udf_data_admission import DataAdmissionCapacityError, DataAdmissionLimits, DataBatchTooLarge
from vane.execution.udf_input_cleanup import TaskOutputGrants

if TYPE_CHECKING:
    from vane.execution.local_resource_graph import LocalResourceUnitContext
    from vane.execution.ref_bundle import LocalShmBudgetManager


@dataclass(frozen=True)
class DataAllocation:
    provider: str
    allocation_id: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError("data allocation requires a provider")
        if not isinstance(self.allocation_id, str) or not self.allocation_id:
            raise ValueError("data allocation requires an identity")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("data allocation size must be a non-negative integer")

    @property
    def key(self) -> tuple[str, str]:
        return self.provider, self.allocation_id


@dataclass(frozen=True)
class _DataLease:
    lease_id: str
    query_id: str
    allocation: DataAllocation
    role: str
    state: str
    resource_unit: LocalResourceUnitContext | None = None


@dataclass
class _QueryState:
    closed: bool = False
    tasks: int = 0
    reservations: int = 0


class RuntimeDataLedger:
    def __init__(self, limits: DataAdmissionLimits | None = None) -> None:
        if limits is not None and not isinstance(limits, DataAdmissionLimits):
            raise TypeError("data_limit must be DataAdmissionLimits")
        self.limits = limits
        self._condition = threading.Condition()
        self._queries: dict[str, _QueryState] = {}
        self._leases: dict[str, _DataLease] = {}
        self._allocations: dict[tuple[str, str], tuple[DataAllocation, int]] = {}
        self._retained_bytes = 0
        self._draining = False
        self._closed = False
        self._reservations: set[DataTaskReservation] = set()
        self._pending_tasks: set[TaskDataScope] = set()

    def _reserved_bytes_locked(self) -> int:
        return sum(r.input_remaining + r.output_remaining for r in self._reservations)

    def open_query(self) -> QueryDataScope:
        with self._condition:
            if self._draining:
                raise RuntimeError("runtime data accounting is draining or closed")
            query_id = uuid.uuid4().hex
            self._queries[query_id] = _QueryState()
            return QueryDataScope(self, query_id)

    def drain(self) -> None:
        with self._condition:
            self._draining = True

    def close(self, *, timeout: float = 0.0) -> None:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("data accounting close timeout must be finite and non-negative")
        deadline = time.monotonic() + timeout
        with self._condition:
            self._draining = True
            while self._queries:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("runtime data accounting still has active queries or tasks")
                self._condition.wait(remaining)
            # Consumer outputs remain valid and accounted after close.
            self._closed = True

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            inputs = {lease.allocation for lease in self._leases.values() if lease.role == "input"}
            outputs = {lease.allocation for lease in self._leases.values() if lease.role == "output"}
            snapshot: dict[str, Any] = {
                "retained_bytes": self._retained_bytes,
                "input_bytes": sum(a.size_bytes for a in inputs),
                "output_bytes": sum(a.size_bytes for a in outputs),
                "allocations": len(self._allocations),
                "leases": len(self._leases),
                "queries": len(self._queries),
                "tasks": sum(query.tasks for query in self._queries.values()),
                "output_state_bytes": {
                    state: sum(
                        a.size_bytes
                        for a in {
                            lease.allocation
                            for lease in self._leases.values()
                            if lease.role == "output" and lease.state == state
                        }
                    )
                    for state in _OUTPUT_STATES[:-1]
                },
                "draining": self._draining,
                "closed": self._closed,
            }
            if self.limits is not None:
                snapshot.update(
                    limit_bytes=self.limits.max_bytes,
                    max_task_input_bytes=self.limits.max_task_input_bytes,
                    max_task_output_bytes=self.limits.max_task_output_bytes,
                    reserved_bytes=self._reserved_bytes_locked(),
                    input_reserved_bytes=sum(r.input_remaining for r in self._reservations),
                    output_reserved_bytes=sum(r.output_remaining for r in self._reservations),
                    reservations=len(self._reservations),
                    usage_bytes=snapshot["retained_bytes"] + self._reserved_bytes_locked(),
                )
            return snapshot

    def _validate_allocation_locked(self, allocation: DataAllocation) -> None:
        existing = self._allocations.get(allocation.key)
        if existing is not None and existing[0] != allocation:
            raise ValueError("data allocation identity has a different size")

    def unit_snapshots(self) -> dict[str, dict[str, Any]]:
        """Attribute live owners; shared allocations remain charged once globally."""
        from vane.execution.udf_resource_usage import empty_unit_data

        with self._condition:
            leases_by_unit: dict[LocalResourceUnitContext, list[_DataLease]] = {}
            allocation_units: dict[tuple[str, str], set[LocalResourceUnitContext | None]] = {}
            for lease in self._leases.values():
                allocation_units.setdefault(lease.allocation.key, set()).add(lease.resource_unit)
                if lease.resource_unit is not None:
                    leases_by_unit.setdefault(lease.resource_unit, []).append(lease)
            reservations_by_unit: dict[LocalResourceUnitContext, list[DataTaskReservation]] = {}
            for reservation in self._reservations:
                if reservation.resource_unit is not None:
                    reservations_by_unit.setdefault(reservation.resource_unit, []).append(reservation)
            pending_by_unit: dict[LocalResourceUnitContext, int] = {}
            for task in self._pending_tasks:
                if task.resource_unit is not None:
                    pending_by_unit[task.resource_unit] = pending_by_unit.get(task.resource_unit, 0) + 1
            result = {}
            for unit in leases_by_unit.keys() | reservations_by_unit.keys() | pending_by_unit.keys():
                leases = leases_by_unit.get(unit, [])
                allocations = {lease.allocation for lease in leases}
                inputs = {lease.allocation for lease in leases if lease.role == "input"}
                outputs = {lease.allocation for lease in leases if lease.role == "output"}
                reservations = reservations_by_unit.get(unit, [])
                input_reserved = sum(r.input_remaining for r in reservations)
                output_reserved = sum(r.output_remaining for r in reservations)
                usage = empty_unit_data()
                usage.update(
                    retained_bytes=sum(a.size_bytes for a in allocations),
                    input_bytes=sum(a.size_bytes for a in inputs),
                    output_bytes=sum(a.size_bytes for a in outputs),
                    shared_retained_bytes=sum(a.size_bytes for a in allocations if len(allocation_units[a.key]) > 1),
                    allocations=len(allocations),
                    leases=len(leases),
                    input_reserved_bytes=input_reserved,
                    output_reserved_bytes=output_reserved,
                    reserved_bytes=input_reserved + output_reserved,
                    reservations=len(reservations),
                    cleanup_pending_tasks=pending_by_unit.get(unit, 0),
                )
                result[unit.resource_unit_id] = {"identity": unit.to_dict(), "usage": usage}
            return result

    def _acquire_locked(
        self,
        query_id: str,
        allocation: DataAllocation,
        role: str,
        state: str,
        resource_unit: LocalResourceUnitContext | None = None,
    ) -> _DataLease:
        self._validate_allocation_locked(allocation)
        count = self._allocations.get(allocation.key, (allocation, 0))[1]
        lease = _DataLease(uuid.uuid4().hex, query_id, allocation, role, state, resource_unit)
        self._allocations[allocation.key] = allocation, count + 1
        if count == 0:
            self._retained_bytes += allocation.size_bytes
        self._leases[lease.lease_id] = lease
        return lease

    def _release_locked(self, lease_id: str) -> bool:
        lease = self._leases.pop(lease_id, None)
        if lease is None:
            return False
        allocation, count = self._allocations[lease.allocation.key]
        if count == 1:
            del self._allocations[allocation.key]
            self._retained_bytes -= allocation.size_bytes
        else:
            self._allocations[allocation.key] = allocation, count - 1
        return True

    def transition_output_block(self, lease_id: str, state: str) -> bool:
        if state not in _OUTPUT_STATES[:-1]:
            raise ValueError(f"invalid output lease state: {state}")
        with self._condition:
            lease = self._leases.get(lease_id)
            if lease is None:
                return False
            if lease.role != "output" or _OUTPUT_STATES.index(state) != _OUTPUT_STATES.index(lease.state) + 1:
                raise ValueError("output leases must advance one state at a time")
            self._leases[lease_id] = replace(lease, state=state)
            return True

    def release_output_block(self, lease_id: str) -> bool:
        with self._condition:
            lease = self._leases.get(lease_id)
            if lease is None or lease.role != "output":
                return False
            return self._release_locked(lease_id)

    def _retire_query_locked(self, query_id: str) -> None:
        query = self._queries[query_id]
        if query.closed and query.tasks == 0 and query.reservations == 0:
            del self._queries[query_id]
            self._condition.notify_all()


class QueryDataScope:
    """Preparation owner; shutdown fences new tasks, not consumer outputs."""

    def __init__(self, ledger: RuntimeDataLedger, query_id: str) -> None:
        self._ledger = ledger
        self.query_id = query_id

    @property
    def limits(self) -> DataAdmissionLimits | None:
        return self._ledger.limits

    def reserve_task(self, *, resource_unit: LocalResourceUnitContext | None = None) -> DataTaskReservation:
        from vane.execution.ref_bundle import local_shm_budget_manager

        with self._ledger._condition:
            query = self._ledger._queries.get(self.query_id)
            if query is None or query.closed:
                raise RuntimeError("query data scope is closed")
            limits = self.limits
            if limits is None:
                raise RuntimeError("data reservation requires a byte limit")
            usage = self._ledger._retained_bytes + self._ledger._reserved_bytes_locked()
            reason = byte_budget_block_reason(
                ByteBudgetUsage(limits.max_bytes, 0, usage, 0),
                limits.task_bytes,
                request_kind="task",
                usage_bytes=usage,
                limit_bytes=limits.max_bytes,
                shared_used_bytes=0,
                shared_pool_bytes=0,
            )
            if reason is not None:
                raise DataAdmissionCapacityError(
                    requested=limits.task_bytes, usage=usage, limit=limits.max_bytes, owner="runtime"
                )
            # Neither owner waits. A transport refusal publishes no runtime
            # reservation; its existing allocations and grants remain intact.
            transport = local_shm_budget_manager().reserve_task_bytes(
                limits.max_task_input_bytes, limits.max_task_output_bytes
            )
            reservation = DataTaskReservation(self._ledger, self.query_id, transport, resource_unit)
            self._ledger._reservations.add(reservation)
            query.reservations += 1
            return reservation

    def open_task(
        self, reservation: DataTaskReservation | None = None, *, resource_unit: LocalResourceUnitContext | None = None
    ) -> TaskDataScope:
        with self._ledger._condition:
            query = self._ledger._queries.get(self.query_id)
            if query is None or query.closed:
                raise RuntimeError("query data scope is closed")
            if self.limits is not None:
                if (
                    reservation is None
                    or reservation not in self._ledger._reservations
                    or reservation.query_id != self.query_id
                    or reservation.resource_unit != resource_unit
                    or reservation.attached
                ):
                    raise RuntimeError("task requires its own live byte admission reservation")
                reservation.attached = True
            query.tasks += 1
            return TaskDataScope(self._ledger, self.query_id, reservation, resource_unit)

    def shutdown(self, *, kill: bool = False) -> None:
        unused = []
        pending = []
        with self._ledger._condition:
            query = self._ledger._queries.get(self.query_id)
            if query is not None:
                query.closed = True
                unused = [
                    r
                    for r in self._ledger._reservations
                    if r.query_id == self.query_id and (not r.attached or r.finished)
                ]
                pending = [task for task in self._ledger._pending_tasks if task._query_id == self.query_id]
                self._ledger._retire_query_locked(self.query_id)
        error: BaseException | None = None
        for task in pending:
            try:
                task.finish()
            except BaseException as exc:
                if error is None:
                    error = exc
        for reservation in unused:
            try:
                reservation.release()
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    def cleanup_pending(self) -> bool:
        with self._ledger._condition:
            return self.query_id in self._ledger._queries


_active_data_task: ContextVar[TaskDataScope | None] = ContextVar("vane_udf_data_task", default=None)


def current_data_task() -> TaskDataScope | None:
    return _active_data_task.get()


class TaskDataScope:
    def __init__(
        self,
        ledger: RuntimeDataLedger,
        query_id: str,
        reservation: DataTaskReservation | None = None,
        resource_unit: LocalResourceUnitContext | None = None,
    ) -> None:
        self._ledger = ledger
        self._query_id = query_id
        self.resource_unit = resource_unit
        self._inputs: dict[tuple[str, str], str] = {}
        self._input_transports: dict[tuple[LocalShmBudgetManager, int], None] = {}
        self._output_grants = TaskOutputGrants()
        self._finished = False
        self._inputs_released = False
        self._finishing = False
        self.reservation = reservation
        self._output_bytes = 0

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _active_data_task.set(self)
        try:
            yield
        finally:
            _active_data_task.reset(token)

    def hold_inputs(self, allocations: Iterable[DataAllocation]) -> None:
        unique: dict[tuple[str, str], DataAllocation] = {}
        for allocation in allocations:
            if allocation.key in unique and unique[allocation.key] != allocation:
                raise ValueError("data allocation identity has a different size")
            unique[allocation.key] = allocation
        with self._ledger._condition:
            if self._finished:
                raise RuntimeError("task data scope is finished")
            for allocation in unique.values():
                self._ledger._validate_allocation_locked(allocation)
            additions = {key: a for key, a in unique.items() if key not in self._inputs}
            if self.reservation is not None:
                existing_bytes = sum(
                    self._ledger._leases[lease].allocation.size_bytes for lease in self._inputs.values()
                )
                requested = existing_bytes + sum(a.size_bytes for a in additions.values())
                limits = self._ledger.limits
                assert limits is not None
                if requested > limits.max_task_input_bytes:
                    raise DataBatchTooLarge("input", requested, limits.max_task_input_bytes)
                self.reservation.input_remaining -= sum(
                    a.size_bytes for key, a in additions.items() if key not in self._ledger._allocations
                )
            for key, allocation in unique.items():
                if key not in self._inputs:
                    lease = self._ledger._acquire_locked(
                        self._query_id, allocation, "input", "task_input", self.resource_unit
                    )
                    self._inputs[key] = lease.lease_id

    def own_output(self, allocation: DataAllocation) -> OutputDataLeaseOwner:
        with self._ledger._condition:
            if self._finished:
                raise RuntimeError("task data scope is finished")
            self._ledger._validate_allocation_locked(allocation)
            if self.reservation is not None:
                limits = self._ledger.limits
                assert limits is not None
                requested = self._output_bytes + allocation.size_bytes
                if requested > limits.max_task_output_bytes:
                    raise DataBatchTooLarge("output", requested, limits.max_task_output_bytes)
                if allocation.key not in self._ledger._allocations:
                    self.reservation.output_remaining -= allocation.size_bytes
                self._output_bytes = requested
            lease = self._ledger._acquire_locked(
                self._query_id, allocation, "output", "generator_pending", self.resource_unit
            )
            return OutputDataLeaseOwner(self._ledger, lease)

    def hold_input_transport(self, manager: LocalShmBudgetManager, lease_id: int) -> None:
        """Own transport cleanup before a lease can be submitted or fail setup."""
        with self._ledger._condition:
            if self._finished:
                raise RuntimeError("task data scope is finished")
            self._input_transports[manager, lease_id] = None

    def hold_output_grant(self, manager: LocalShmBudgetManager, grant_id: int) -> None:
        with self._ledger._condition:
            if self._finished:
                raise RuntimeError("task data scope is finished")
            self._output_grants.hold(manager, grant_id)

    def finish(self) -> None:
        with self._ledger._condition:
            if self._finishing:
                return
            self._finishing = True
            self._finished = True
            if not self._inputs_released:
                self._ledger._pending_tasks.add(self)
            if self.reservation is not None:
                self.reservation.finished = True
        error: BaseException | None = None
        try:
            # A failed worker future does not prove its input transport is
            # clean. Keep the input borrows and query alive through retries.
            # Transport callbacks must run outside the ledger lock.
            for manager, lease_id in tuple(self._input_transports):
                try:
                    manager.cancel_input_lease(lease_id, name="task-input-cleanup")
                    if manager.input_lease_pending(lease_id):
                        raise RuntimeError("task input transport cleanup is still in progress")
                except BaseException as exc:
                    if error is None:
                        error = exc
                else:
                    del self._input_transports[manager, lease_id]
            try:
                self._output_grants.release()
            except BaseException as exc:
                if error is None:
                    error = exc
            if error is None:
                with self._ledger._condition:
                    if not self._inputs_released:
                        for data_lease_id in self._inputs.values():
                            self._ledger._release_locked(data_lease_id)
                        self._inputs.clear()
                        self._inputs_released = True
                        self._ledger._pending_tasks.remove(self)
                        self._ledger._queries[self._query_id].tasks -= 1
                        self._ledger._retire_query_locked(self._query_id)
            if self.reservation is not None:
                try:
                    self.reservation.release()
                except BaseException as exc:
                    if error is None:
                        error = exc
        finally:
            with self._ledger._condition:
                self._finishing = False
        if error is not None:
            raise error


class DataTaskReservation:
    """Unused task bytes; conversion to data owners never double charges bytes."""

    def __init__(
        self,
        ledger: RuntimeDataLedger,
        query_id: str,
        transport: Any,
        resource_unit: LocalResourceUnitContext | None = None,
    ) -> None:
        self._ledger = ledger
        self.query_id = query_id
        self.resource_unit = resource_unit
        self.transport = transport
        assert ledger.limits is not None
        self.input_remaining = ledger.limits.max_task_input_bytes
        self.output_remaining = ledger.limits.max_task_output_bytes
        self.attached = False
        self.finished = False
        self._releasing = False

    def release(self) -> None:
        with self._ledger._condition:
            if self not in self._ledger._reservations or self._releasing:
                return
            self._releasing = True
        try:
            # Wakes run outside the ledger lock. Retain the query and runtime
            # reservation until the transport confirms cleanup, including retries.
            self.transport.release()
        except BaseException:
            with self._ledger._condition:
                self._releasing = False
            raise
        with self._ledger._condition:
            self._ledger._reservations.remove(self)
            self._ledger._queries[self.query_id].reservations -= 1
            self._ledger._retire_query_locked(self.query_id)


class OutputDataLeaseOwner(OutputBlockLeaseOwner):
    def __init__(self, ledger: RuntimeDataLedger, lease: _DataLease) -> None:
        super().__init__(ledger, lease)
        self._ledger = ledger

    def fork(self) -> OutputDataLeaseOwner:
        """Hold the same allocation for a consumer view, even after query close."""
        with self._lock, self._ledger._condition:
            if self._released:
                raise RuntimeError("output data lease is released")
            lease = self._ledger._leases[self._lease_id]
            fork = self._ledger._acquire_locked(
                lease.query_id, lease.allocation, "output", "external_consumer", lease.resource_unit
            )
            return OutputDataLeaseOwner(self._ledger, fork)
