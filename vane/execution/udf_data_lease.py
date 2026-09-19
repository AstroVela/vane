# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Runtime-scoped accounting of live UDF data, without admission policy.

The ledger contains only identities and sizes, never data buffers or callers.
Input borrows last through backend completion. Output owners can outlive a
query or runtime, including through zero-copy consumer views.
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
from typing import Any

from vane.execution.data_lifecycle import _OUTPUT_STATES, OutputBlockLeaseOwner


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


@dataclass
class _QueryState:
    closed: bool = False
    tasks: int = 0


class RuntimeDataLedger:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._queries: dict[str, _QueryState] = {}
        self._leases: dict[str, _DataLease] = {}
        self._allocations: dict[tuple[str, str], tuple[DataAllocation, int]] = {}
        self._draining = False
        self._closed = False

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
            return {
                "retained_bytes": sum(a.size_bytes for a, _ in self._allocations.values()),
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

    def _validate_allocation_locked(self, allocation: DataAllocation) -> None:
        existing = self._allocations.get(allocation.key)
        if existing is not None and existing[0] != allocation:
            raise ValueError("data allocation identity has a different size")

    def _acquire_locked(self, query_id: str, allocation: DataAllocation, role: str, state: str) -> _DataLease:
        self._validate_allocation_locked(allocation)
        count = self._allocations.get(allocation.key, (allocation, 0))[1]
        lease = _DataLease(uuid.uuid4().hex, query_id, allocation, role, state)
        self._allocations[allocation.key] = allocation, count + 1
        self._leases[lease.lease_id] = lease
        return lease

    def _release_locked(self, lease_id: str) -> bool:
        lease = self._leases.pop(lease_id, None)
        if lease is None:
            return False
        allocation, count = self._allocations[lease.allocation.key]
        if count == 1:
            del self._allocations[allocation.key]
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
        if query.closed and query.tasks == 0:
            del self._queries[query_id]
            self._condition.notify_all()


class QueryDataScope:
    """Preparation owner; shutdown fences new tasks, not consumer outputs."""

    def __init__(self, ledger: RuntimeDataLedger, query_id: str) -> None:
        self._ledger = ledger
        self.query_id = query_id

    def open_task(self) -> TaskDataScope:
        with self._ledger._condition:
            query = self._ledger._queries.get(self.query_id)
            if query is None or query.closed:
                raise RuntimeError("query data scope is closed")
            query.tasks += 1
            return TaskDataScope(self._ledger, self.query_id)

    def shutdown(self, *, kill: bool = False) -> None:
        with self._ledger._condition:
            query = self._ledger._queries.get(self.query_id)
            if query is not None:
                query.closed = True
                self._ledger._retire_query_locked(self.query_id)

    def cleanup_pending(self) -> bool:
        with self._ledger._condition:
            return self.query_id in self._ledger._queries


_active_data_task: ContextVar[TaskDataScope | None] = ContextVar("vane_udf_data_task", default=None)


def current_data_task() -> TaskDataScope | None:
    return _active_data_task.get()


class TaskDataScope:
    def __init__(self, ledger: RuntimeDataLedger, query_id: str) -> None:
        self._ledger = ledger
        self._query_id = query_id
        self._inputs: dict[tuple[str, str], str] = {}
        self._finished = False

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
            for key, allocation in unique.items():
                if key not in self._inputs:
                    lease = self._ledger._acquire_locked(self._query_id, allocation, "input", "task_input")
                    self._inputs[key] = lease.lease_id

    def own_output(self, allocation: DataAllocation) -> OutputDataLeaseOwner:
        with self._ledger._condition:
            if self._finished:
                raise RuntimeError("task data scope is finished")
            lease = self._ledger._acquire_locked(self._query_id, allocation, "output", "generator_pending")
            return OutputDataLeaseOwner(self._ledger, lease)

    def finish(self) -> None:
        with self._ledger._condition:
            if self._finished:
                return
            self._finished = True
            for lease_id in self._inputs.values():
                self._ledger._release_locked(lease_id)
            self._inputs.clear()
            self._ledger._queries[self._query_id].tasks -= 1
            self._ledger._retire_query_locked(self._query_id)


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
            fork = self._ledger._acquire_locked(lease.query_id, lease.allocation, "output", "external_consumer")
            return OutputDataLeaseOwner(self._ledger, fork)
