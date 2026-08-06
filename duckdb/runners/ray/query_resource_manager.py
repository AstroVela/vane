# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from duckdb.runners.ray.admission_ledger import BoundedSet
from duckdb.runners.ray.query_resource_graph import (
    QueryAllocation,
    QueryResourceGraph,
    ResourceUnitSpec,
    ResourceVector,
)

_SOFT_TASK_BLOCK_REASONS = {
    "query_soft_cpu",
    "query_soft_gpu",
    "query_soft_heap_bytes",
    "query_soft_object_store_bytes",
    "unit_soft_cpu",
    "unit_soft_gpu",
    "unit_soft_heap_bytes",
    "unit_soft_object_store_bytes",
    "unit_soft_limit",
}
_SOFT_OUTPUT_BLOCK_REASONS = {
    "query_soft_object_store_bytes",
    "unit_soft_object_store_bytes",
    "unit_soft_limit",
}
_OUTPUT_STATES = (
    "generator_pending",
    "unit_queue",
    "downstream_input",
    "external_consumer",
    "released",
)
_RESOURCE_FIELDS = ("cpu", "gpu", "heap_bytes", "object_store_bytes")
_EPSILON = 1e-9
_TERMINAL_IDENTITY_REPLAY_CAPACITY = 65_536
_SOFT_RESERVATION_WARNING_DELAY_S = 60.0

logger = logging.getLogger(__name__)


def _resource_with_object_store(resources: ResourceVector, object_store_bytes: int) -> ResourceVector:
    return ResourceVector(
        cpu=resources.cpu,
        gpu=resources.gpu,
        heap_bytes=resources.heap_bytes,
        object_store_bytes=object_store_bytes,
    )


def _positive_difference(left: ResourceVector, right: ResourceVector) -> ResourceVector:
    return ResourceVector(
        cpu=max(0.0, left.cpu - right.cpu),
        gpu=max(0.0, left.gpu - right.gpu),
        heap_bytes=max(0, left.heap_bytes - right.heap_bytes),
        object_store_bytes=max(0, left.object_store_bytes - right.object_store_bytes),
    )


@dataclass(frozen=True)
class TaskRequest:
    query_id: str
    resource_unit_id: str
    task_id: str
    attempt_id: str
    node_id: str | None
    retained_input_bytes: int | None = None


@dataclass(frozen=True)
class TaskLease:
    lease_id: str
    query_id: str
    resource_unit_id: str
    task_id: str
    attempt_id: str
    node_id: str | None
    execution_slot_id: str
    actor_index: int | None
    resources: ResourceVector
    output_window_bytes: int
    liveness: bool
    allocation_generation: int


@dataclass(frozen=True)
class TaskGrant:
    granted: bool
    lease: TaskLease | None = None
    blocked_reason: str = ""
    fatal: bool = False
    liveness: bool = False
    admission_epoch: int = 0


@dataclass(frozen=True)
class _TaskAdmissionPlan:
    resources: ResourceVector = field(default_factory=ResourceVector)
    output_window_bytes: int = 0
    node_id: str | None = None
    actor_index: int | None = None


@dataclass(frozen=True)
class _ObjectStoreUnitBudget:
    task_reserved_bytes: int
    output_reserved_bytes: int
    task_internal_usage_bytes: int
    output_usage_bytes: int

    @property
    def task_budget_usage_bytes(self) -> int:
        return self.task_internal_usage_bytes + max(
            0,
            self.output_usage_bytes - self.output_reserved_bytes,
        )

    @property
    def task_reserved_remaining_bytes(self) -> int:
        return max(0, self.task_reserved_bytes - self.task_budget_usage_bytes)

    @property
    def output_reserved_remaining_bytes(self) -> int:
        return max(0, self.output_reserved_bytes - self.output_usage_bytes)

    @property
    def shared_used_bytes(self) -> int:
        return max(0, self.task_budget_usage_bytes - self.task_reserved_bytes)


@dataclass(frozen=True)
class _ObjectStoreBudgetState:
    limit_bytes: int
    ineligible_usage_bytes: int
    shared_pool_bytes: int
    shared_used_bytes: int
    query_usage_bytes: int
    reservation_unit_ids: tuple[str, ...]
    units: dict[str, _ObjectStoreUnitBudget]

    @property
    def shared_remaining_bytes(self) -> int:
        return max(0, self.shared_pool_bytes - self.shared_used_bytes)


@dataclass(frozen=True)
class _QueuedTaskEvaluation:
    rank: tuple[int, int, int]
    request: TaskRequest
    reason: str | None
    fatal: bool
    plan: _TaskAdmissionPlan

    @property
    def key(self) -> tuple[str, str]:
        return (str(self.request.task_id), str(self.request.attempt_id))


@dataclass(frozen=True)
class OutputBlockRequest:
    query_id: str
    producer_unit_id: str
    task_lease_id: str
    attempt_id: str
    block_id: str
    size_bytes: int


@dataclass(frozen=True)
class OutputBlockLease:
    lease_id: str
    query_id: str
    producer_unit_id: str
    task_lease_id: str
    attempt_id: str
    block_id: str
    node_id: str | None
    size_bytes: int
    state: str
    liveness: bool
    allocation_generation: int


@dataclass(frozen=True)
class OutputBlockGrant:
    granted: bool
    lease: OutputBlockLease | None = None
    blocked_reason: str = ""
    fatal: bool = False
    liveness: bool = False


class OutputBlockLeaseOwner:
    """Shared lifetime owner carried with one query-produced ObjectRef."""

    def __init__(self, manager: RayQueryResourceManager, lease: OutputBlockLease) -> None:
        self._manager = manager
        self._lease_id = str(lease.lease_id)
        self._state = str(lease.state)
        self._released = False
        self._lock = threading.Lock()

    @property
    def lease_id(self) -> str:
        return self._lease_id

    @property
    def state(self) -> str:
        with self._lock:
            return "released" if self._released else self._state

    def transition_to(self, state: str) -> bool:
        target = str(state)
        if target not in _OUTPUT_STATES or target == "released":
            raise ValueError(f"invalid output lease owner transition target: {target}")
        with self._lock:
            if self._released:
                return False
            current_index = _OUTPUT_STATES.index(self._state)
            target_index = _OUTPUT_STATES.index(target)
            if target_index < current_index:
                raise ValueError(f"output lease owner cannot move backward: {self._state} -> {target}")
            while current_index < target_index:
                next_state = _OUTPUT_STATES[current_index + 1]
                if not self._manager.transition_output_block(self._lease_id, next_state):
                    self._released = True
                    return False
                self._state = next_state
                current_index += 1
            return True

    def release(self) -> bool:
        with self._lock:
            if self._released:
                return False
            released = self._manager.release_output_block(self._lease_id)
            self._released = True
            self._state = "released"
            return bool(released)

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            # Query teardown may already have canceled and removed the manager's
            # leases. Destructors cannot safely surface that idempotent race.
            pass


@dataclass
class _ResourceUnitState:
    spec: ResourceUnitSpec
    runnable: bool = False
    actor_ready: bool = False
    queued_input_bytes: int = 0
    pending_task_count: int = 0
    queued_output_bytes: int = 0
    pending_output_count: int = 0
    bytes_task_outputs_generated: int = 0
    num_task_outputs_generated: int = 0
    num_outputs_of_finished_tasks: int = 0
    num_tasks_finished: int = 0
    completed: bool = False


class RayQueryResourceManager:
    """Own Ray execution and streaming-output resources for one query DAG.

    DuckDB owns native-fragment CPU, heap, scheduling, and spill. Native units
    remain in this graph only so Ray-side object-store windows, task leases,
    backpressure, and liveness can be coordinated across native/UDF edges.
    """

    def __init__(
        self,
        graph: QueryResourceGraph,
        allocation: QueryAllocation,
        *,
        admission_open: bool = True,
        reservation_ratio: float = 0.5,
        on_change: Callable[[], None] | None = None,
        on_eligible_units_change: Callable[[tuple[str, ...]], None] | None = None,
    ) -> None:
        ratio = float(reservation_ratio)
        if not math.isfinite(ratio) or ratio < 0 or ratio > 1:
            raise ValueError("reservation_ratio must be in [0, 1]")
        self.graph = graph
        self.allocation = allocation
        self._allocation_admission_open = bool(admission_open)
        self.reservation_ratio = ratio
        self._on_change = on_change
        self._on_eligible_units_change = on_eligible_units_change
        self._lock = threading.RLock()
        self._admission_epoch = 0
        self._units = {
            unit.resource_unit_id: _ResourceUnitState(
                spec=unit,
                actor_ready=unit.backend != "ray_actor",
            )
            for unit in graph.units
        }
        self._direct_downstream_unit_ids: dict[str, set[str]] = {unit.resource_unit_id: set() for unit in graph.units}
        for unit in graph.units:
            for input_unit_id in unit.input_unit_ids:
                self._direct_downstream_unit_ids[input_unit_id].add(unit.resource_unit_id)
        self._topological_unit_ids = graph.topological_unit_ids()
        self._reverse_topological_unit_ids = graph.reverse_topological_unit_ids()
        self._transitive_upstream_unit_ids: dict[str, frozenset[str]] = {}
        for resource_unit_id in self._topological_unit_ids:
            upstream_unit_ids: set[str] = set()
            for input_unit_id in self._units[resource_unit_id].spec.input_unit_ids:
                upstream_unit_ids.add(input_unit_id)
                upstream_unit_ids.update(self._transitive_upstream_unit_ids[input_unit_id])
            self._transitive_upstream_unit_ids[resource_unit_id] = frozenset(upstream_unit_ids)
        self._reverse_topological_rank = {
            resource_unit_id: index for index, resource_unit_id in enumerate(self._reverse_topological_unit_ids)
        }
        self._task_leases: dict[str, TaskLease] = {}
        self._active_attempt_leases: dict[tuple[str, str], str] = {}
        # Submitted includes Ray Core PENDING actors as well as ready actors.
        # Ray Data charges both states to logical operator usage because each
        # handle already owns a real scheduling request even before placement.
        self._submitted_actor_slots: set[tuple[str, int]] = set()
        self._retiring_actor_unit_ids: set[str] = set()
        self._actor_node_by_slot: dict[tuple[str, int], str] = {}
        self._active_actor_slots: dict[tuple[str, int], str] = {}
        self._queued_actor_slot_leases: dict[tuple[str, int], deque[str]] = {}
        self._terminal_attempts = BoundedSet[tuple[str, str]](capacity=_TERMINAL_IDENTITY_REPLAY_CAPACITY)
        self._waiting_task_inputs: dict[tuple[str, str], tuple[TaskRequest, int]] = {}
        self._waiting_output_blocks: dict[str, OutputBlockRequest] = {}
        self._output_leases: dict[str, OutputBlockLease] = {}
        self._output_lease_by_block: dict[str, str] = {}
        self._observed_output_blocks = BoundedSet[str](capacity=_TERMINAL_IDENTITY_REPLAY_CAPACITY)
        self._generated_output_count_by_task_lease: dict[str, int] = {}
        self._terminal_output_blocks = BoundedSet[str](capacity=_TERMINAL_IDENTITY_REPLAY_CAPACITY)
        self._active_liveness_task_lease_ids_by_unit: dict[str, str] = {}
        self._active_liveness_output_lease_ids_by_unit: dict[str, str] = {}
        self._task_liveness_grants_total = 0
        self._output_liveness_grants_total = 0
        self._completed_materialization_barrier_ids: set[str] = set()
        self._external_consumer_waiting = False
        self._soft_allocation_debt_since: float | None = None
        self._soft_allocation_warning_emitted = False
        self._failed = False
        self._failure_reason = ""
        self._cancelled = False
        self._cancel_reason = ""

    def _publish_change_locked(self) -> None:
        """Publish a non-blocking local wakeup after an accounting mutation.

        The callback must only enqueue local work.  It must never perform a Ray
        RPC or wait for another lock; callers invoke it while the manager's
        snapshot is still protected so the driver can coalesce the query into
        its next admission-pump turn without losing the state transition.
        """
        self._admission_epoch += 1
        if self._on_change is not None:
            self._on_change()

    def admission_epoch(self) -> int:
        """Return the edge-trigger generation for descriptor admission.

        FTE uses this to memoize one failed, non-persistent admission probe.
        Descriptors are retried only after a real QRM accounting mutation,
        avoiding both a waiter explosion and a polling loop.
        """
        with self._lock:
            return int(self._admission_epoch)

    def update_unit_state(
        self,
        resource_unit_id: str,
        *,
        runnable: bool,
        completed: bool = False,
    ) -> None:
        unit_key = str(resource_unit_id)
        eligible_callback: Callable[[tuple[str, ...]], None] | None = None
        eligible_unit_ids: tuple[str, ...] = ()
        with self._lock:
            unit = self._units.get(unit_key)
            if unit is None:
                raise KeyError(f"unit is not registered: {unit_key}")
            if unit.completed and not bool(completed):
                raise RuntimeError(f"completed resource unit cannot become incomplete: {unit_key}")
            before = (
                unit.runnable,
                unit.actor_ready,
                unit.queued_input_bytes,
                unit.pending_task_count,
                unit.queued_output_bytes,
                unit.pending_output_count,
                unit.completed,
            )
            unit.runnable = bool(runnable) and not bool(completed)
            self._recompute_unit_queued_input_locked(unit_key)
            self._recompute_unit_queued_output_locked(unit_key)
            unit.completed = bool(completed)
            after = (
                unit.runnable,
                unit.actor_ready,
                unit.queued_input_bytes,
                unit.pending_task_count,
                unit.queued_output_bytes,
                unit.pending_output_count,
                unit.completed,
            )
            if after != before:
                if bool(completed) and not before[-1]:
                    # Completion changes the eligible demand before the
                    # coordinator can publish the corresponding allocation.
                    # Preserve live leases, but fence every new grant from the
                    # old generation during that handoff.
                    self._allocation_admission_open = False
                self._publish_change_locked()
                if bool(completed) and not before[-1]:
                    eligible_callback = self._on_eligible_units_change
                    eligible_unit_ids = self._eligible_resource_unit_ids_locked()
        if eligible_callback is not None:
            eligible_callback(eligible_unit_ids)

    def mark_materialization_barrier_completed_for_node(self, physical_node_id: str) -> bool:
        """Advance the execution phase after a true barrier completes."""

        barrier = self.graph.barrier_for_physical_node(str(physical_node_id))
        eligible_callback: Callable[[tuple[str, ...]], None] | None = None
        eligible_unit_ids: tuple[str, ...] = ()
        with self._lock:
            if barrier.barrier_id in self._completed_materialization_barrier_ids:
                return False
            self._completed_materialization_barrier_ids.add(barrier.barrier_id)
            # The eligible set changes before the coordinator can publish the
            # next phase allocation. Fence only new grants during that short
            # handoff so downstream work cannot consume the previous phase's
            # reservation; live tasks and output leases continue unchanged.
            self._allocation_admission_open = False
            self._publish_change_locked()
            eligible_callback = self._on_eligible_units_change
            eligible_unit_ids = self._eligible_resource_unit_ids_locked()
        if eligible_callback is not None:
            eligible_callback(eligible_unit_ids)
        return True

    def current_eligible_resource_unit_ids(self) -> tuple[str, ...]:
        with self._lock:
            return self._eligible_resource_unit_ids_locked()

    def _eligible_resource_unit_ids_locked(self) -> tuple[str, ...]:
        if self._cancelled:
            return ()
        return tuple(
            resource_unit_id
            for resource_unit_id in self.graph.eligible_resource_unit_ids(self._completed_materialization_barrier_ids)
            if not self._units[resource_unit_id].completed
        )

    def _frontier_materialization_barriers_locked(self) -> tuple[Any, ...]:
        return self.graph.frontier_materialization_barriers(self._completed_materialization_barrier_ids)

    def _materializing_object_store_unlimited_unit_ids_locked(self) -> tuple[str, ...]:
        barriers = self._frontier_materialization_barriers_locked()
        if not barriers:
            return ()
        eligible = set(self._eligible_resource_unit_ids_locked())
        boundary: set[str] = set()
        for barrier in barriers:
            # A blocking materializer must be able to drain its complete input
            # branch even after the direct input's accumulated bytes exceed the
            # soft query budget.  Lifting the cap only on that direct input can
            # deadlock a longer chain: the live materializer keeps a task lease
            # while an upstream UDF needs another task to produce the remaining
            # input, so the downstream-chain task escape cannot run backward.
            # Follow only the explicitly materialized side upstream; deferred
            # inputs such as a broadcast probe remain backpressured.
            pending = list(barrier.materialized_input_unit_ids)
            while pending:
                resource_unit_id = pending.pop()
                if resource_unit_id in boundary or resource_unit_id not in eligible:
                    continue
                boundary.add(resource_unit_id)
                pending.extend(self._units[resource_unit_id].spec.input_unit_ids)
            boundary.add(barrier.materializer_unit_id)
        return tuple(
            resource_unit_id for resource_unit_id in self._topological_unit_ids if resource_unit_id in boundary
        )

    def note_task_waiting(self, request: TaskRequest) -> None:
        """Register real queued work before attempting admission."""
        with self._lock:
            if str(request.query_id) != self.graph.query_id:
                raise ValueError("queued task query_id does not match resource manager")
            resource_unit_id = str(request.resource_unit_id)
            unit = self._units.get(resource_unit_id)
            if unit is None:
                raise KeyError(f"unit is not registered: {resource_unit_id}")
            task_id = str(request.task_id).strip()
            attempt_id = str(request.attempt_id).strip()
            if not task_id or not attempt_id:
                raise ValueError("queued task identity must be non-empty")
            retained = (
                unit.spec.per_task.object_store_bytes
                if request.retained_input_bytes is None
                else int(request.retained_input_bytes)
            )
            if retained < 0:
                raise ValueError("queued task retained input must be non-negative")
            key = (task_id, attempt_id)
            existing = self._waiting_task_inputs.get(key)
            value = (request, retained)
            if existing is not None and existing != value:
                raise ValueError("queued task identity was reused with different work")
            if existing is not None:
                return
            self._waiting_task_inputs[key] = value
            unit.runnable = not unit.completed
            self._recompute_unit_queued_input_locked(resource_unit_id)
            self._publish_change_locked()

    def remove_task_waiter(self, task_id: str, attempt_id: str) -> bool:
        with self._lock:
            value = self._waiting_task_inputs.pop(
                (str(task_id), str(attempt_id)),
                None,
            )
            if value is None:
                return False
            self._recompute_unit_queued_input_locked(str(value[0].resource_unit_id))
            self._publish_change_locked()
            return True

    def mark_task_attempt_terminal(self, task_id: str, attempt_id: str) -> None:
        """Fence a cancelled request identity without retaining driver ownership."""
        key = (str(task_id), str(attempt_id))
        with self._lock:
            if key in self._active_attempt_leases or key in self._waiting_task_inputs:
                raise RuntimeError("cannot terminalize an active or waiting task attempt")
            self._terminal_attempts.add(key)
            self._publish_change_locked()

    def _recompute_unit_queued_input_locked(self, resource_unit_id: str) -> None:
        unit = self._units[resource_unit_id]
        waiting = [
            queued_bytes
            for request, queued_bytes in self._waiting_task_inputs.values()
            if str(request.resource_unit_id) == resource_unit_id
        ]
        unit.pending_task_count = len(waiting)
        unit.queued_input_bytes = sum(waiting)

    def note_output_waiting(self, request: OutputBlockRequest) -> str | None:
        """Register one produced block, or return its fatal identity reason."""

        with self._lock:
            fatal_reason, _unit, _task, _size = self._output_request_identity_error_locked(request)
            if fatal_reason is not None:
                return fatal_reason
            resource_unit_id = str(request.producer_unit_id)
            block_id = str(request.block_id).strip()
            existing = self._waiting_output_blocks.get(block_id)
            if existing is not None and existing != request:
                raise ValueError("queued output block_id was reused with different work")
            if existing is not None:
                return None
            self._record_output_observation_locked(request)
            self._waiting_output_blocks[block_id] = request
            self._recompute_unit_queued_output_locked(resource_unit_id)
            self._publish_change_locked()
            return None

    def remove_output_waiter(self, block_id: str) -> bool:
        with self._lock:
            value = self._waiting_output_blocks.pop(str(block_id), None)
            if value is None:
                return False
            self._recompute_unit_queued_output_locked(str(value.producer_unit_id))
            self._publish_change_locked()
            return True

    def mark_output_block_terminal(self, block_id: str) -> None:
        """Fence a cancelled output identity without retaining driver ownership."""
        key = str(block_id)
        with self._lock:
            if key in self._output_lease_by_block or key in self._waiting_output_blocks:
                raise RuntimeError("cannot terminalize an active or waiting output block")
            self._terminal_output_blocks.add(key)
            self._publish_change_locked()

    def _recompute_unit_queued_output_locked(self, resource_unit_id: str) -> None:
        unit = self._units[resource_unit_id]
        waiting = [
            int(request.size_bytes)
            for request in self._waiting_output_blocks.values()
            if str(request.producer_unit_id) == resource_unit_id
        ]
        leased_queue_bytes = sum(
            lease.size_bytes
            for lease in self._output_leases.values()
            if lease.producer_unit_id == resource_unit_id and lease.state in {"generator_pending", "unit_queue"}
        )
        unit.pending_output_count = len(waiting)
        unit.queued_output_bytes = sum(waiting) + leased_queue_bytes

    def _output_request_identity_error_locked(
        self,
        request: OutputBlockRequest,
    ) -> tuple[str | None, _ResourceUnitState | None, TaskLease | None, int]:
        """Validate immutable output identity without applying resource policy."""

        if self._cancelled:
            return "query_cancelled", None, None, 0
        if self._failed:
            return "query_failed", None, None, 0
        if str(request.query_id) != self.graph.query_id:
            return "query_id_mismatch", None, None, 0
        unit = self._units.get(str(request.producer_unit_id))
        if unit is None:
            return "unit_not_registered", None, None, 0
        block_id = str(request.block_id).strip()
        if not block_id:
            return "invalid_block_id", unit, None, 0
        if block_id in self._terminal_output_blocks:
            return "output_block_terminal", unit, None, 0
        task = self._task_leases.get(str(request.task_lease_id))
        if task is None:
            return "task_lease_not_active", unit, None, 0
        if task.resource_unit_id != unit.spec.resource_unit_id:
            return "task_unit_mismatch", unit, task, 0
        if task.attempt_id != str(request.attempt_id):
            return "task_attempt_mismatch", unit, task, 0
        if block_id in self._output_lease_by_block:
            return "output_block_already_leased", unit, task, 0
        size = int(request.size_bytes)
        if size <= 0:
            return "invalid_output_block_size", unit, task, size
        return None, unit, task, size

    def _record_output_observation_locked(self, request: OutputBlockRequest) -> bool:
        """Learn one generated block size exactly once for dynamic estimation."""

        block_id = str(request.block_id)
        if block_id in self._observed_output_blocks:
            return False
        unit = self._units[str(request.producer_unit_id)]
        unit.num_task_outputs_generated += 1
        unit.bytes_task_outputs_generated += int(request.size_bytes)
        task_lease_id = str(request.task_lease_id)
        self._generated_output_count_by_task_lease[task_lease_id] = (
            self._generated_output_count_by_task_lease.get(task_lease_id, 0) + 1
        )
        self._observed_output_blocks.add(block_id)
        return True

    def _record_task_finished_locked(self, lease: TaskLease) -> None:
        """Fold one terminal task into the Ray Data-style output averages."""

        unit = self._units[lease.resource_unit_id]
        unit.num_tasks_finished += 1
        unit.num_outputs_of_finished_tasks += self._generated_output_count_by_task_lease.pop(
            lease.lease_id,
            0,
        )

    def set_external_consumer_waiting(self, waiting: bool) -> None:
        with self._lock:
            waiting = bool(waiting)
            if self._external_consumer_waiting == waiting:
                return
            self._external_consumer_waiting = waiting
            self._publish_change_locked()

    def set_submitted_actor_slots(
        self,
        resource_unit_id: str,
        actor_indices: set[int] | tuple[int, ...] | list[int] | range,
    ) -> None:
        """Publish the fixed actor handles submitted to Ray Core.

        Submitted slots are logical process usage whether Ray Core reports the
        actor READY or PENDING. Removing a slot is only valid after its task
        leases have drained, normally during an execution-phase transition.
        """

        unit_key = str(resource_unit_id)
        normalized: set[int] = set()
        with self._lock:
            unit = self._units.get(unit_key)
            if unit is None:
                raise KeyError(f"unit is not registered: {unit_key}")
            if unit.spec.backend != "ray_actor":
                raise ValueError(f"unit is not a Ray actor unit: {unit_key}")
            for raw_actor_index in actor_indices:
                if isinstance(raw_actor_index, bool):
                    raise TypeError("actor index must be an integer")
                actor_index = int(raw_actor_index)
                if actor_index < 0 or actor_index >= unit.spec.actor_pool_size:
                    raise ValueError(f"actor index is outside pool {unit_key}: {actor_index}")
                normalized.add(actor_index)

            before = {
                actor_index
                for candidate_unit_id, actor_index in self._submitted_actor_slots
                if candidate_unit_id == unit_key
            }
            if normalized:
                if self._cancelled:
                    raise RuntimeError(f"cannot submit actor slots for a cancelled query: {unit_key}")
                if self._failed:
                    raise RuntimeError(f"cannot submit actor slots for a failed query: {unit_key}")
                if unit_key not in self._eligible_resource_unit_ids_locked():
                    raise RuntimeError(f"cannot submit actor slots outside the current execution phase: {unit_key}")
            if before == normalized:
                return
            if unit_key in self._retiring_actor_unit_ids:
                raise RuntimeError(f"cannot change submitted slots for a retiring actor pool: {unit_key}")
            removed = before - normalized
            for actor_index in removed:
                slot_key = (unit_key, actor_index)
                if self._actor_slot_lease_count_locked(slot_key) > 0:
                    raise RuntimeError(f"cannot remove actor slot with live leases: {unit_key}/{actor_index}")
            self._submitted_actor_slots = {slot for slot in self._submitted_actor_slots if slot[0] != unit_key}
            self._submitted_actor_slots.update((unit_key, actor_index) for actor_index in normalized)
            for slot_key in tuple(self._actor_node_by_slot):
                if slot_key[0] == unit_key and slot_key[1] not in normalized:
                    self._actor_node_by_slot.pop(slot_key, None)
            unit.actor_ready = any(
                candidate_unit_id == unit_key for candidate_unit_id, _actor_index in self._actor_node_by_slot
            )
            self._publish_change_locked()

    def set_ready_actor_slots(
        self,
        resource_unit_id: str,
        actor_node_ids: dict[int, str],
    ) -> None:
        """Publish the subset of a Ray Core actor pool that is currently ready."""

        unit_key = str(resource_unit_id)
        normalized: dict[int, str] = {}
        with self._lock:
            unit = self._units.get(unit_key)
            if unit is None:
                raise KeyError(f"unit is not registered: {unit_key}")
            if unit.spec.backend != "ray_actor":
                raise ValueError(f"unit is not a Ray actor unit: {unit_key}")
            for raw_actor_index, raw_node_id in actor_node_ids.items():
                if isinstance(raw_actor_index, bool):
                    raise TypeError("actor index must be an integer")
                actor_index = int(raw_actor_index)
                node_id = str(raw_node_id or "").strip()
                if actor_index < 0 or actor_index >= unit.spec.actor_pool_size:
                    raise ValueError(f"actor index is outside pool {unit_key}: {actor_index}")
                if not node_id:
                    raise ValueError(f"actor {unit_key}/{actor_index} node_id must be non-empty")
                normalized[actor_index] = node_id
            if normalized:
                if self._cancelled:
                    raise RuntimeError(f"cannot publish ready actor slots for a cancelled query: {unit_key}")
                if unit_key not in self._eligible_resource_unit_ids_locked():
                    raise RuntimeError(
                        f"cannot publish ready actor slots outside the current execution phase: {unit_key}"
                    )
            submitted = {
                actor_index
                for candidate_unit_id, actor_index in self._submitted_actor_slots
                if candidate_unit_id == unit_key
            }
            unknown_ready = sorted(set(normalized) - submitted)
            if unknown_ready:
                raise ValueError(f"ready actor slot was not submitted for {unit_key}: {unknown_ready[0]}")
            before = {
                actor_index: node_id
                for (candidate_unit_id, actor_index), node_id in self._actor_node_by_slot.items()
                if candidate_unit_id == unit_key
            }
            if before == normalized:
                return
            if unit_key in self._retiring_actor_unit_ids:
                raise RuntimeError(f"cannot change ready slots for a retiring actor pool: {unit_key}")
            removed_ready = set(before) - set(normalized)
            for actor_index in removed_ready:
                slot_key = (unit_key, actor_index)
                if self._actor_slot_lease_count_locked(slot_key) > 0:
                    raise RuntimeError(f"cannot remove ready actor slot with live leases: {unit_key}/{actor_index}")
            for slot_key in tuple(self._actor_node_by_slot):
                if slot_key[0] == unit_key:
                    self._actor_node_by_slot.pop(slot_key, None)
            self._actor_node_by_slot.update(
                {(unit_key, actor_index): node_id for actor_index, node_id in normalized.items()}
            )
            unit.actor_ready = bool(normalized)
            self._publish_change_locked()

    def reconcile_actor_runtime_node(
        self,
        resource_unit_id: str,
        actor_index: int,
        node_id: str,
    ) -> tuple[str, ...]:
        """Move one reconstructed actor slot and its live calls atomically.

        Ray may reconstruct an unpinned actor on another node.  The actor
        process is the execution slot, so every active or prefetched call on
        that slot must follow its new physical node before it can publish more
        output.  Existing output leases stay where they were produced.
        """

        unit_key = str(resource_unit_id)
        if isinstance(actor_index, bool):
            raise TypeError("actor index must be an integer")
        slot_index = int(actor_index)
        node_key = str(node_id or "").strip()
        if not node_key:
            raise ValueError("reconstructed actor node_id must be non-empty")
        with self._lock:
            unit = self._units.get(unit_key)
            if unit is None:
                raise KeyError(f"unit is not registered: {unit_key}")
            if unit.spec.backend != "ray_actor":
                raise ValueError(f"unit is not a Ray actor unit: {unit_key}")
            if slot_index < 0 or slot_index >= unit.spec.actor_pool_size:
                raise ValueError(f"actor index is outside pool {unit_key}: {slot_index}")
            slot_key = (unit_key, slot_index)
            if slot_key not in self._submitted_actor_slots:
                raise RuntimeError(f"reconstructed actor slot was not submitted: {unit_key}/{slot_index}")
            if slot_key not in self._actor_node_by_slot:
                raise RuntimeError(f"reconstructed actor slot was not ready: {unit_key}/{slot_index}")
            if self._cancelled:
                raise RuntimeError(f"cannot move an actor slot for a cancelled query: {unit_key}/{slot_index}")
            if unit_key not in self._eligible_resource_unit_ids_locked():
                raise RuntimeError(
                    f"cannot move an actor slot outside the current execution phase: {unit_key}/{slot_index}"
                )
            if unit_key in self._retiring_actor_unit_ids:
                raise RuntimeError(f"cannot move a retiring actor pool: {unit_key}")

            moved_lease_ids: list[str] = []
            for lease_id, lease in tuple(self._task_leases.items()):
                if lease.resource_unit_id != unit_key or lease.actor_index != slot_index:
                    continue
                if lease.node_id != node_key:
                    self._task_leases[lease_id] = replace(lease, node_id=node_key)
                    moved_lease_ids.append(lease_id)

            changed = self._actor_node_by_slot[slot_key] != node_key or bool(moved_lease_ids)
            self._actor_node_by_slot[slot_key] = node_key
            unit.actor_ready = True
            if changed:
                self._publish_change_locked()
            return tuple(sorted(moved_lease_ids))

    def begin_actor_pool_retirement(self, resource_unit_id: str) -> bool:
        """Atomically fence and uncharge one completed-phase actor pool.

        A stale phase callback must not tear down a pool that is eligible in
        the manager's current frontier.  Conversely, a real phase transition
        may only retire a fully drained pool; live or prefetched invocations
        indicate a barrier/lifecycle ordering bug and keep their slot
        ownership intact.
        """

        unit_key = str(resource_unit_id)
        with self._lock:
            unit = self._units.get(unit_key)
            if unit is None:
                raise KeyError(f"unit is not registered: {unit_key}")
            if unit.spec.backend != "ray_actor":
                raise ValueError(f"unit is not a Ray actor unit: {unit_key}")
            if unit_key in self._eligible_resource_unit_ids_locked():
                return False

            live_slots = sorted(
                actor_index
                for candidate_unit_id, actor_index in self._submitted_actor_slots
                if candidate_unit_id == unit_key
                and self._actor_slot_lease_count_locked((candidate_unit_id, actor_index)) > 0
            )
            if live_slots:
                raise RuntimeError(f"cannot retire actor pool with live leases: {unit_key}/{live_slots[0]}")

            if unit_key in self._retiring_actor_unit_ids:
                return True
            self._retiring_actor_unit_ids.add(unit_key)
            for slot_key in tuple(self._actor_node_by_slot):
                if slot_key[0] == unit_key:
                    self._actor_node_by_slot.pop(slot_key, None)
            unit.actor_ready = False
            self._publish_change_locked()
            return True

    def complete_actor_pool_retirement(self, resource_unit_id: str) -> bool:
        """Uncharge submitted actor processes after physical shutdown."""

        unit_key = str(resource_unit_id)
        with self._lock:
            unit = self._units.get(unit_key)
            if unit is None:
                raise KeyError(f"unit is not registered: {unit_key}")
            if unit.spec.backend != "ray_actor":
                raise ValueError(f"unit is not a Ray actor unit: {unit_key}")
            if unit_key not in self._retiring_actor_unit_ids:
                return False
            live_slots = sorted(
                actor_index
                for candidate_unit_id, actor_index in self._submitted_actor_slots
                if candidate_unit_id == unit_key
                and self._actor_slot_lease_count_locked((candidate_unit_id, actor_index)) > 0
            )
            if live_slots:
                raise RuntimeError(
                    f"cannot complete actor pool retirement with live leases: {unit_key}/{live_slots[0]}"
                )
            self._submitted_actor_slots = {slot for slot in self._submitted_actor_slots if slot[0] != unit_key}
            self._retiring_actor_unit_ids.remove(unit_key)
            self._publish_change_locked()
            return True

    def update_allocation(
        self,
        allocation: QueryAllocation,
        *,
        admission_open: bool,
    ) -> None:
        with self._lock:
            if allocation.generation <= self.allocation.generation:
                raise ValueError(
                    "allocation generation must increase: "
                    f"current={self.allocation.generation} new={allocation.generation}"
                )
            self.allocation = allocation
            self._allocation_admission_open = bool(admission_open) and not self._failed and not self._cancelled
            self._publish_change_locked()

    def close_admission(self) -> None:
        """Fence new grants while preserving live leases for ordered teardown."""
        with self._lock:
            if not self._allocation_admission_open:
                return
            self._allocation_admission_open = False
            self._publish_change_locked()

    def try_acquire_task(self, request: TaskRequest) -> TaskGrant:
        with self._lock:
            blocked_reason, fatal, plan = self._normal_task_block_reason_locked(request)
            if blocked_reason is None:
                return self._grant_task_locked(
                    request,
                    plan,
                    liveness=False,
                )
            if fatal or blocked_reason not in _SOFT_TASK_BLOCK_REASONS:
                return TaskGrant(False, blocked_reason=blocked_reason, fatal=fatal)

            liveness_block = self._task_liveness_block_reason_locked(
                request,
                plan,
            )
            if liveness_block is not None:
                return TaskGrant(False, blocked_reason=liveness_block)
            return self._grant_task_locked(
                request,
                plan,
                liveness=True,
            )

    def try_acquire_task_descriptor(self, request: TaskRequest) -> TaskGrant:
        """Atomically admit one non-persistent scheduler descriptor.

        Unlike ``try_acquire_queued_task``, a denied descriptor is never
        registered in QRM.  Existing persistent waiters still participate in
        the same downstream-first arbitration domain and win when their rank
        is higher.  This is the FTE submission-window boundary: QRM owns only
        tasks that received a lease, while the FTE scheduler owns backlog.
        """
        with self._lock:

            def denied(blocked_reason: str, *, fatal: bool = False) -> TaskGrant:
                return TaskGrant(
                    False,
                    blocked_reason=blocked_reason,
                    fatal=fatal,
                    admission_epoch=int(self._admission_epoch),
                )

            def with_epoch(grant: TaskGrant) -> TaskGrant:
                return TaskGrant(
                    granted=grant.granted,
                    lease=grant.lease,
                    blocked_reason=grant.blocked_reason,
                    fatal=grant.fatal,
                    liveness=grant.liveness,
                    admission_epoch=int(self._admission_epoch),
                )

            # A non-root native fragment unit becomes runnable when its first real split
            # descriptor arrives.  Persistent task waiters perform this same
            # transition in note_task_waiting(); descriptor-owned backlog must
            # preserve it even though no individual waiter is registered.
            request_unit = self._units.get(str(request.resource_unit_id))
            if (
                str(request.query_id) == self.graph.query_id
                and request_unit is not None
                and not request_unit.completed
                and not request_unit.runnable
            ):
                request_unit.runnable = True
                self._recompute_unit_queued_input_locked(str(request.resource_unit_id))
                self._publish_change_locked()

            key = (str(request.task_id), str(request.attempt_id))
            if key in self._waiting_task_inputs:
                return denied(
                    "task_descriptor_already_registered",
                    fatal=True,
                )
            reason, fatal, plan = self._normal_task_block_reason_locked(request)
            if fatal:
                return denied(
                    reason or "fatal_task_admission",
                    fatal=True,
                )

            candidate = _QueuedTaskEvaluation(
                rank=self._waiting_task_rank_locked(
                    request,
                    len(self._waiting_task_inputs),
                ),
                request=request,
                reason=reason,
                fatal=False,
                plan=plan,
            )
            evaluations = (
                *self._evaluate_waiting_tasks_locked(),
                candidate,
            )
            preferred = self._select_waiting_task_evaluation_locked(evaluations)
            if preferred is not None and preferred.key != key:
                return denied("admission_turn")
            if preferred is None:
                return denied(reason or "task_not_admissible")
            if reason is None:
                return with_epoch(
                    self._grant_task_locked(
                        request,
                        plan,
                        liveness=False,
                    )
                )
            if reason not in _SOFT_TASK_BLOCK_REASONS:
                return denied(reason)
            liveness_block = self._queued_liveness_block_reason_locked(evaluations)
            if liveness_block is not None:
                return denied(liveness_block)
            return with_epoch(
                self._grant_task_locked(
                    request,
                    plan,
                    liveness=True,
                )
            )

    def try_acquire_queued_task(self, request: TaskRequest) -> TaskGrant:
        """Admit only the deterministic winner from the query's waiter queue."""
        with self._lock:
            key = (str(request.task_id), str(request.attempt_id))
            waiting = self._waiting_task_inputs.get(key)
            if waiting is None:
                return TaskGrant(
                    False,
                    blocked_reason="task_waiter_not_registered",
                    fatal=True,
                )
            if waiting[0] != request:
                return TaskGrant(
                    False,
                    blocked_reason="task_waiter_identity_mismatch",
                    fatal=True,
                )
            evaluations = self._evaluate_waiting_tasks_locked()
            current = next(item for item in evaluations if item.key == key)
            if current.fatal:
                return TaskGrant(
                    False,
                    blocked_reason=current.reason or "fatal_task_admission",
                    fatal=True,
                )
            preferred = self._select_waiting_task_evaluation_locked(evaluations)
            if preferred is not None and preferred.key != key:
                return TaskGrant(False, blocked_reason="admission_turn")
            if preferred is None:
                return TaskGrant(False, blocked_reason=current.reason or "task_not_admissible")
            if current.reason is None:
                return self._grant_task_locked(
                    request,
                    current.plan,
                    liveness=False,
                )
            liveness_block = self._queued_liveness_block_reason_locked(evaluations)
            if liveness_block is not None:
                return TaskGrant(False, blocked_reason=liveness_block)
            return self._grant_task_locked(
                request,
                current.plan,
                liveness=True,
            )

    def try_acquire_next_queued_task(
        self,
        candidate_keys: set[tuple[str, str]],
    ) -> tuple[TaskRequest | None, TaskGrant | None]:
        """Select globally, then resolve one driver-owned waiter in one pass."""
        with self._lock:
            evaluations = self._evaluate_waiting_tasks_locked()
            owned_fatal = [item for item in evaluations if item.fatal and item.key in candidate_keys]
            if owned_fatal:
                fatal_selected = min(owned_fatal, key=lambda item: item.rank)
                return fatal_selected.request, TaskGrant(
                    False,
                    blocked_reason=fatal_selected.reason or "fatal_task_admission",
                    fatal=True,
                )
            selected = self._select_waiting_task_evaluation_locked(evaluations)
            if selected is None:
                return None, None
            if selected.key not in candidate_keys:
                # The selected waiter belongs to the FTE scheduler rather than
                # the driver's Future table. Yield without consuming capacity.
                return selected.request, TaskGrant(
                    False,
                    blocked_reason="admission_turn",
                )
            if selected.reason is None:
                return selected.request, self._grant_task_locked(
                    selected.request,
                    selected.plan,
                    liveness=False,
                )
            request = selected.request
            liveness_block = self._queued_liveness_block_reason_locked(evaluations)
            if liveness_block is not None:
                return request, TaskGrant(False, blocked_reason=liveness_block)
            return request, self._grant_task_locked(
                request,
                selected.plan,
                liveness=True,
            )

    def _normal_task_block_reason_locked(
        self,
        request: TaskRequest,
        *,
        hypothetical: bool = False,
    ) -> tuple[str | None, bool, _TaskAdmissionPlan]:
        empty_plan = _TaskAdmissionPlan()
        if self._cancelled:
            return "query_cancelled", True, empty_plan
        if self._failed:
            return "query_failed", True, empty_plan
        if str(request.query_id) != self.graph.query_id:
            return "query_id_mismatch", True, empty_plan
        unit = self._units.get(str(request.resource_unit_id))
        if unit is None:
            return "unit_not_registered", True, empty_plan
        if unit.completed:
            return "unit_completed", True, empty_plan
        if not self._allocation_admission_open:
            return "allocation_pending", False, empty_plan
        if unit.spec.resource_unit_id not in self._eligible_resource_unit_ids_locked():
            return "materialization_barrier_pending", False, empty_plan
        if not unit.runnable:
            return "unit_not_runnable", False, empty_plan
        if unit.spec.backend == "ray_actor" and not unit.actor_ready:
            return "actor_not_ready", False, empty_plan

        task_key = str(request.task_id).strip()
        attempt_key = str(request.attempt_id).strip()
        if not task_key or not attempt_key:
            return "invalid_task_identity", True, empty_plan
        attempt = (task_key, attempt_key)
        if not hypothetical:
            if attempt in self._terminal_attempts:
                return "attempt_terminal", True, empty_plan
            if attempt in self._active_attempt_leases:
                return "attempt_already_active", True, empty_plan

        active_unit_tasks = self._active_task_count_for_unit_locked(unit.spec.resource_unit_id)
        concurrency_cap = self._unit_concurrency_cap(unit.spec)
        if concurrency_cap is not None and active_unit_tasks >= concurrency_cap:
            return "unit_concurrency", False, empty_plan

        node_id: str | None = None
        actor_index: int | None = None
        if unit.spec.backend == "ray_worker":
            node_id = "" if request.node_id is None else str(request.node_id).strip()
            if not node_id:
                return "ray_worker_node_required", True, empty_plan
        elif unit.spec.backend == "ray_task":
            if request.node_id is not None and str(request.node_id).strip():
                return "ray_task_node_must_be_unset", True, empty_plan
        elif unit.spec.backend == "ray_actor":
            requested_node_id = "" if request.node_id is None else str(request.node_id).strip()
            candidates = [
                (self._actor_slot_lease_count_locked(slot), slot[1], ready_node_id)
                for slot, ready_node_id in self._actor_node_by_slot.items()
                if slot[0] == unit.spec.resource_unit_id
                and (not requested_node_id or ready_node_id == requested_node_id)
                and self._actor_slot_lease_count_locked(slot) < unit.spec.actor_prefetch_depth
            ]
            if not candidates:
                if requested_node_id:
                    return "actor_node_unavailable", True, empty_plan
                return "actor_slot", False, empty_plan
            _lease_count, actor_index, node_id = min(candidates)
        else:
            return "unsupported_backend", True, empty_plan

        retained = (
            unit.spec.per_task.object_store_bytes
            if request.retained_input_bytes is None
            else int(request.retained_input_bytes)
        )
        if retained < 0:
            return "invalid_retained_input_bytes", True, empty_plan
        resources = _resource_with_object_store(unit.spec.per_task, retained)
        pending_output_estimate = self._pending_output_estimate_for_admission_locked(
            unit.spec,
            actor_index=actor_index,
        )
        commitment = _resource_with_object_store(
            resources,
            retained + pending_output_estimate,
        )
        plan = _TaskAdmissionPlan(
            resources=resources,
            output_window_bytes=unit.spec.output_window_bytes,
            node_id=node_id,
            actor_index=actor_index,
        )

        query_usage = self._soft_allocation_usage_locked()
        usage_after = query_usage + commitment
        for field_name in ("cpu", "gpu", "heap_bytes"):
            if getattr(commitment, field_name) <= 0:
                continue
            if getattr(usage_after, field_name) > getattr(self.allocation.resources, field_name) + _EPSILON:
                return f"query_soft_{field_name}", False, plan

        object_store_backpressure_disabled = (
            unit.spec.resource_unit_id in self._materializing_object_store_unlimited_unit_ids_locked()
        )
        soft_reason = self._unit_soft_block_reason_locked(
            unit.spec.resource_unit_id,
            commitment,
            ignore_object_store=object_store_backpressure_disabled,
            object_store_request_kind="task",
        )
        if soft_reason is not None:
            return soft_reason, False, plan
        return None, False, plan

    def _actor_slot_lease_count_locked(self, slot: tuple[str, int]) -> int:
        return int(slot in self._active_actor_slots) + len(self._queued_actor_slot_leases.get(slot, ()))

    @staticmethod
    def _unit_concurrency_cap(spec: ResourceUnitSpec) -> int | None:
        return spec.max_concurrency

    def _pending_output_estimate_per_task_locked(self, spec: ResourceUnitSpec) -> int:
        """Estimate unseen streaming-generator bytes for one running task.

        Native fragment results are materialized atomically, so their protocol
        window is only the initial estimate until exact result bytes replace
        it. Ray UDF streams instead learn the average generated block size at
        runtime, matching Ray Data's treatment of pending generator outputs.
        The declared window caps the unseen-output estimate; it is not a hard
        cap on an actual block.
        """

        if spec.target_output_block_bytes <= 0:
            return 0
        if spec.backend == "ray_worker":
            return int(spec.output_window_bytes)
        state = self._units[spec.resource_unit_id]
        if state.num_task_outputs_generated <= 0:
            if state.num_tasks_finished > 0:
                return 0
            # Before the first real block arrives, charge the declared generator
            # window instead of treating every running task as output-free.  A
            # zero seed can admit a full CPU wave and then create query-wide
            # object-store debt when the first observation retroactively teaches
            # the manager the real per-task pending-output size.
            return int(spec.output_window_bytes)
        pending_blocks = float(spec.generator_buffer_blocks)
        if state.num_tasks_finished > 0:
            pending_blocks = min(
                pending_blocks,
                state.num_outputs_of_finished_tasks / state.num_tasks_finished,
            )
        estimate = math.ceil(state.bytes_task_outputs_generated * pending_blocks / state.num_task_outputs_generated)
        return min(int(spec.output_window_bytes), max(0, estimate))

    def _pending_output_estimate_for_admission_locked(
        self,
        spec: ResourceUnitSpec,
        *,
        actor_index: int | None,
    ) -> int:
        if spec.backend == "ray_actor":
            if actor_index is None:
                raise RuntimeError("Ray actor admission requires a selected actor slot")
            if self._actor_slot_lease_count_locked((spec.resource_unit_id, actor_index)) > 0:
                # A prefetched actor call retains its exact input but cannot
                # populate the single active generator window for that actor.
                return 0
        return self._pending_output_estimate_per_task_locked(spec)

    def _pending_output_estimate_for_lease_locked(self, lease: TaskLease) -> int:
        spec = self._units[lease.resource_unit_id].spec
        if spec.backend == "ray_actor" and lease.lease_id not in self._active_actor_slots.values():
            # Prefetched actor calls retain their inputs, but only the active
            # call can currently populate that actor's generator buffer.
            return 0
        return self._pending_output_estimate_per_task_locked(spec)

    @staticmethod
    def _task_commitment(spec: ResourceUnitSpec) -> ResourceVector:
        """Return non-spillable process resources for one runnable task."""
        return ResourceVector(
            cpu=spec.per_task.cpu,
            gpu=spec.per_task.gpu,
            heap_bytes=spec.per_task.heap_bytes,
        )

    def _dimension_reservation_unit_ids_locked(
        self,
        field_name: str,
        *,
        requested_unit_id: str | None,
    ) -> tuple[str, ...]:
        """Return every resource-owning operator in the current phase.

        Like Ray Data's reservation allocator, phase eligibility rather than
        future node placement defines the reservation set. Operators may
        temporarily receive less than one invocation's minimum; their real
        requests can still enter Ray Core through the bounded liveness escape.
        """

        eligible = set(self._eligible_resource_unit_ids_locked())
        selected = tuple(
            resource_unit_id
            for resource_unit_id in self._topological_unit_ids
            if resource_unit_id in eligible
            and not self._units[resource_unit_id].completed
            and self._unit_uses_dimension(self._units[resource_unit_id].spec, field_name)
        )
        if requested_unit_id is not None and requested_unit_id not in selected:
            raise RuntimeError(f"unit {requested_unit_id} requested undeclared {field_name} capacity")
        return selected

    def _unit_soft_block_reason_locked(
        self,
        resource_unit_id: str,
        request: ResourceVector,
        *,
        ignore_object_store: bool = False,
        excluded_waiting_block_id: str | None = None,
        object_store_request_kind: str = "task",
    ) -> str | None:
        """Enforce protected shares across current eligible resource units."""
        process_fields = ("cpu", "gpu", "heap_bytes")
        usage_by_unit: dict[str, ResourceVector] | None = None
        if any(getattr(request, field_name) > 0 for field_name in process_fields):
            usage_by_unit = {
                key: self._unit_usage_locked(
                    key,
                    excluded_waiting_block_id=excluded_waiting_block_id,
                )
                for key in self._units
            }
        for field_name in process_fields:
            amount = getattr(request, field_name)
            if amount <= 0:
                continue
            assert usage_by_unit is not None
            dimension_unit_ids, reserved_by_unit, limit = self._dimension_reservations_locked(
                field_name,
                requested_unit_id=resource_unit_id,
            )
            reserved = reserved_by_unit[resource_unit_id]
            current_amount = getattr(usage_by_unit[resource_unit_id], field_name)
            if current_amount + amount <= reserved + _EPSILON:
                continue
            shared_pool = max(0.0, limit - sum(reserved_by_unit.values()))
            shared_used = sum(
                max(
                    0.0,
                    getattr(usage_by_unit[key], field_name) - reserved_by_unit[key],
                )
                for key in dimension_unit_ids
            )
            shared_need = amount - max(0.0, reserved - current_amount)
            if shared_used + shared_need > shared_pool + _EPSILON:
                return f"unit_soft_{field_name}"

        if not ignore_object_store and request.object_store_bytes > 0:
            return self._object_store_soft_block_reason_locked(
                resource_unit_id,
                request.object_store_bytes,
                request_kind=object_store_request_kind,
                excluded_waiting_block_id=excluded_waiting_block_id,
                usage_by_unit=usage_by_unit,
            )
        return None

    def _dimension_reservations_locked(
        self,
        field_name: str,
        *,
        requested_unit_id: str | None,
        reservation_limit: int | float | None = None,
    ) -> tuple[tuple[str, ...], dict[str, int | float], int | float]:
        dimension_unit_ids = self._dimension_reservation_unit_ids_locked(
            field_name,
            requested_unit_id=requested_unit_id,
        )
        global_limit = getattr(self.allocation.resources, field_name)
        limit = global_limit if reservation_limit is None else min(global_limit, max(0, reservation_limit))
        if not dimension_unit_ids:
            return dimension_unit_ids, {}, global_limit

        baseline_by_unit = {
            key: self._unit_dimension_commitment(self._units[key].spec, field_name) for key in dimension_unit_ids
        }
        baseline_total = sum(baseline_by_unit.values())
        if baseline_total <= limit + _EPSILON:
            bonus_pool = max(0.0, limit - baseline_total) * self.reservation_ratio
            bonus = bonus_pool / len(dimension_unit_ids)
            reserved_by_unit = {key: baseline + bonus for key, baseline in baseline_by_unit.items()}
        else:
            reservation_pool = limit * self.reservation_ratio
            reserved_by_unit = {
                key: reservation_pool * baseline / baseline_total for key, baseline in baseline_by_unit.items()
            }
        maximum_by_unit = {
            key: self._unit_dimension_maximum_locked(self._units[key].spec, field_name) for key in dimension_unit_ids
        }
        reserved_by_unit = {key: min(reserved, maximum_by_unit[key]) for key, reserved in reserved_by_unit.items()}
        if field_name in {"heap_bytes", "object_store_bytes"}:
            reserved_by_unit = {key: math.floor(reserved) for key, reserved in reserved_by_unit.items()}
        return dimension_unit_ids, reserved_by_unit, global_limit

    def _object_store_output_usage_for_unit_locked(
        self,
        resource_unit_id: str,
        *,
        excluded_waiting_block_id: str | None,
    ) -> int:
        waiting_bytes = sum(
            int(request.size_bytes)
            for block_id, request in self._waiting_output_blocks.items()
            if block_id != excluded_waiting_block_id and str(request.producer_unit_id) == resource_unit_id
        )
        return self._live_output_bytes_for_unit_locked(resource_unit_id) + waiting_bytes

    def _object_store_budget_state_locked(
        self,
        *,
        excluded_waiting_block_id: str | None = None,
        usage_by_unit: dict[str, ResourceVector] | None = None,
    ) -> _ObjectStoreBudgetState:
        """Partition current usage into per-unit protected and shared bytes.

        Completed or otherwise ineligible units keep their exact bytes charged
        and are removed before current-phase reservations and the shared pool
        are calculated. Streaming producers split their protected share evenly
        between task/internal usage and output handoff; only eligible usage
        beyond those shares consumes the common pool.
        """

        reservation_unit_ids = self._dimension_reservation_unit_ids_locked(
            "object_store_bytes",
            requested_unit_id=None,
        )
        object_limit = int(self.allocation.resources.object_store_bytes)
        if usage_by_unit is None:
            usage_by_unit = {
                resource_unit_id: self._unit_usage_locked(
                    resource_unit_id,
                    excluded_waiting_block_id=excluded_waiting_block_id,
                )
                for resource_unit_id in self._units
            }
        total_usage_by_unit = {
            resource_unit_id: usage.object_store_bytes for resource_unit_id, usage in usage_by_unit.items()
        }
        output_usage_by_unit = {
            resource_unit_id: self._object_store_output_usage_for_unit_locked(
                resource_unit_id,
                excluded_waiting_block_id=excluded_waiting_block_id,
            )
            for resource_unit_id in self._units
        }
        reservation_unit_id_set = set(reservation_unit_ids)
        ineligible_usage = sum(
            total_usage
            for resource_unit_id, total_usage in total_usage_by_unit.items()
            if resource_unit_id not in reservation_unit_id_set
        )
        _, total_reserved_by_unit, _ = self._dimension_reservations_locked(
            "object_store_bytes",
            requested_unit_id=None,
            reservation_limit=max(0, object_limit - ineligible_usage),
        )
        units: dict[str, _ObjectStoreUnitBudget] = {}
        shared_used = 0
        for resource_unit_id, unit in self._units.items():
            total_usage = total_usage_by_unit[resource_unit_id]
            output_usage = output_usage_by_unit[resource_unit_id]
            if output_usage > total_usage:
                raise RuntimeError(f"unit {resource_unit_id} output usage exceeds total object-store usage")

            total_reserved = int(total_reserved_by_unit.get(resource_unit_id, 0))
            has_streaming_outputs = bool(
                unit.spec.target_output_block_bytes > 0 and unit.spec.generator_buffer_blocks > 0
            )
            output_reserved = (total_reserved + 1) // 2 if has_streaming_outputs else 0
            budget = _ObjectStoreUnitBudget(
                task_reserved_bytes=total_reserved - output_reserved,
                output_reserved_bytes=output_reserved,
                task_internal_usage_bytes=total_usage - output_usage,
                output_usage_bytes=output_usage,
            )
            units[resource_unit_id] = budget
            if resource_unit_id in reservation_unit_id_set:
                shared_used += budget.shared_used_bytes

        query_usage = self._soft_allocation_usage_locked(
            excluded_waiting_block_id=excluded_waiting_block_id,
        ).object_store_bytes
        accounted_usage = sum(budget.task_internal_usage_bytes + budget.output_usage_bytes for budget in units.values())
        if accounted_usage != query_usage:
            raise RuntimeError(
                "per-unit object-store accounting does not match query usage: "
                f"units={accounted_usage} query={query_usage}"
            )

        shared_pool = max(
            0,
            object_limit - ineligible_usage - sum(int(total_reserved_by_unit[key]) for key in reservation_unit_ids),
        )
        return _ObjectStoreBudgetState(
            limit_bytes=object_limit,
            ineligible_usage_bytes=ineligible_usage,
            shared_pool_bytes=shared_pool,
            shared_used_bytes=shared_used,
            query_usage_bytes=query_usage,
            reservation_unit_ids=reservation_unit_ids,
            units=units,
        )

    def _object_store_soft_block_reason_locked(
        self,
        resource_unit_id: str,
        amount: int,
        *,
        request_kind: str,
        excluded_waiting_block_id: str | None,
        usage_by_unit: dict[str, ResourceVector] | None,
    ) -> str | None:
        """Allow protected progress before considering shared-pool debt."""

        if request_kind not in {"task", "output"}:
            raise ValueError(f"invalid object-store request kind: {request_kind}")
        state = self._object_store_budget_state_locked(
            excluded_waiting_block_id=excluded_waiting_block_id,
            usage_by_unit=usage_by_unit,
        )
        if resource_unit_id not in state.reservation_unit_ids:
            if request_kind == "output":
                # Completion and phase changes fence new tasks immediately but
                # preserve live leases.  Their already-produced ObjectRefs must
                # still cross the driver handoff after the producer stops being
                # reservation-eligible.  The bytes remain fully charged as
                # ineligible usage; there is no current reservation to enforce.
                return None
            raise RuntimeError(f"unit {resource_unit_id} requested undeclared object_store_bytes capacity")

        unit = state.units[resource_unit_id]
        protected_remaining = unit.task_reserved_remaining_bytes
        if request_kind == "output":
            protected_remaining += unit.output_reserved_remaining_bytes
        shared_need = max(0, int(amount) - protected_remaining)
        if shared_need <= 0:
            return None

        query_usage_after = state.query_usage_bytes + int(amount)
        if query_usage_after > state.limit_bytes:
            return "query_soft_object_store_bytes"
        if state.shared_used_bytes + shared_need > state.shared_pool_bytes:
            return "unit_soft_object_store_bytes"
        return None

    @staticmethod
    def _unit_uses_dimension(spec: ResourceUnitSpec, field_name: str) -> bool:
        if field_name == "object_store_bytes":
            return bool(spec.per_task.object_store_bytes or spec.target_output_block_bytes)
        resources = spec.resident_per_actor if spec.backend == "ray_actor" else spec.per_task
        return getattr(resources, field_name) > 0

    def _unit_dimension_commitment(self, spec: ResourceUnitSpec, field_name: str) -> int | float:
        if field_name == "object_store_bytes":
            # Ray Data does not impose a per-task object-store hard minimum.
            # Known inputs/outputs and the learned generator estimate are
            # charged to runtime usage; the default per-unit reservation below
            # supplies the protected progress share.
            return 0
        if spec.backend == "ray_actor":
            return getattr(spec.resident_per_actor, field_name) * spec.actor_pool_size
        return getattr(spec.per_task, field_name)

    def _unit_dimension_maximum_locked(
        self,
        spec: ResourceUnitSpec,
        field_name: str,
    ) -> int | float:
        """Cap a soft share by concurrency and the aggregate budget dimensions."""
        if field_name == "object_store_bytes":
            # Output sizes are data-dependent. Match Ray Data's unbounded
            # operator maximum and let the query-level soft budget, output
            # backpressure, spill, and liveness escape control the dimension.
            return self.allocation.resources.object_store_bytes
        if spec.backend == "ray_actor" and field_name != "object_store_bytes":
            return getattr(spec.resident_per_actor, field_name) * spec.actor_pool_size
        concurrency = self._unit_concurrency_cap(spec)
        if spec.backend == "ray_actor":
            concurrency = spec.actor_pool_size
        coexistence_bound = math.inf if concurrency is None else int(concurrency)
        commitment = self._task_commitment(spec)
        for resource_field in _RESOURCE_FIELDS:
            per_task = getattr(commitment, resource_field)
            if per_task <= 0:
                continue
            limit = getattr(self.allocation.resources, resource_field)
            if resource_field in {"heap_bytes", "object_store_bytes"}:
                dimension_bound = int(limit) // int(per_task)
            else:
                dimension_bound = math.floor((float(limit) + _EPSILON) / float(per_task))
            coexistence_bound = min(coexistence_bound, dimension_bound)
        if math.isinf(coexistence_bound):
            return getattr(self.allocation.resources, field_name)
        per_task_amount = self._unit_dimension_commitment(spec, field_name)
        return per_task_amount * max(0, int(coexistence_bound))

    def _task_liveness_block_reason_locked(
        self,
        request: TaskRequest,
        plan: _TaskAdmissionPlan,
    ) -> str | None:
        del plan
        if not self._task_liveness_chain_available_locked(request.resource_unit_id):
            return "liveness_task_active"
        if self._has_normal_task_candidate_locked():
            return "normal_candidate_available"
        if self._waiting_task_inputs:
            evaluations = self._evaluate_waiting_tasks_locked(hypothetical=True)
            selected = self._select_waiting_task_evaluation_locked(evaluations)
            if selected is None:
                return "no_liveness_candidate"
            if selected.key != (str(request.task_id), str(request.attempt_id)):
                return "liveness_candidate_not_selected"
        return None

    def _task_liveness_chain_available_locked(self, resource_unit_id: str) -> bool:
        """Keep escaped tasks on one downstream dependency chain.

        A query-global-idle escape cannot start a starving downstream UDF while
        upstream tasks retain the object-store budget that the UDF must drain.
        Permit one escaped task per unit along a strict downstream chain. The
        first escape may advance from any live upstream unit; subsequent
        escapes must descend from every active escape, which keeps parallel
        branches closed while allowing a multi-UDF pipeline to drain.
        """

        resource_unit_key = str(resource_unit_id)
        if self._active_task_count_for_unit_locked(resource_unit_key) > 0:
            return False

        active_liveness_unit_ids = set(self._active_liveness_task_lease_ids_by_unit)
        upstream_unit_ids = self._transitive_upstream_unit_ids[resource_unit_key]
        if active_liveness_unit_ids:
            return active_liveness_unit_ids.issubset(upstream_unit_ids)
        if not self._task_leases:
            return True
        return any(lease.resource_unit_id in upstream_unit_ids for lease in self._task_leases.values())

    def _has_normal_task_candidate_locked(self) -> bool:
        # Liveness arbitration must consider real dispatchable work only. A
        # structurally runnable operator can have no input bundle or descriptor
        # at this instant; inventing a hypothetical task for it would suppress
        # the one escape needed by the actual upstream request that can produce
        # its next input. This matches Ray Data's dispatchable-operator check.
        return any(
            not item.fatal and item.reason is None for item in self._evaluate_waiting_tasks_locked(hypothetical=True)
        )

    def _waiting_task_rank_locked(
        self,
        request: TaskRequest,
        registration_order: int,
    ) -> tuple[int, int, int]:
        resource_unit_id = str(request.resource_unit_id)
        return (
            self._reverse_topological_rank[resource_unit_id],
            self._active_task_count_for_unit_locked(resource_unit_id),
            int(registration_order),
        )

    def _evaluate_waiting_tasks_locked(
        self,
        *,
        hypothetical: bool = False,
    ) -> tuple[_QueuedTaskEvaluation, ...]:
        evaluations: list[_QueuedTaskEvaluation] = []
        for order, (request, _) in enumerate(self._waiting_task_inputs.values()):
            reason, fatal, plan = self._normal_task_block_reason_locked(
                request,
                hypothetical=hypothetical,
            )
            evaluations.append(
                _QueuedTaskEvaluation(
                    rank=self._waiting_task_rank_locked(request, order),
                    request=request,
                    reason=reason,
                    fatal=fatal,
                    plan=plan,
                )
            )
        return tuple(evaluations)

    def _select_waiting_task_evaluation_locked(
        self,
        evaluations: tuple[_QueuedTaskEvaluation, ...],
    ) -> _QueuedTaskEvaluation | None:
        normal = [item for item in evaluations if not item.fatal and item.reason is None]
        if normal:
            return min(normal, key=lambda item: item.rank)
        soft = [item for item in evaluations if not item.fatal and item.reason in _SOFT_TASK_BLOCK_REASONS]
        if not soft:
            return None
        soft_unit_ids = {str(item.request.resource_unit_id) for item in soft}
        ordered_unit_ids = [
            resource_unit_id
            for resource_unit_id in self._reverse_topological_unit_ids
            if resource_unit_id in soft_unit_ids
        ]
        starving_unit_ids = [
            resource_unit_id
            for resource_unit_id in ordered_unit_ids
            if self._units[resource_unit_id].pending_task_count > 0
            and self._active_task_count_for_unit_locked(resource_unit_id) == 0
        ]
        preferred_unit_id = (starving_unit_ids or ordered_unit_ids)[0]
        return min(
            (item for item in soft if str(item.request.resource_unit_id) == preferred_unit_id),
            key=lambda item: item.rank,
        )

    def _queued_liveness_block_reason_locked(
        self,
        evaluations: tuple[_QueuedTaskEvaluation, ...],
    ) -> str | None:
        if any(not item.fatal and item.reason is None for item in evaluations):
            return "normal_candidate_available"
        selected = self._select_waiting_task_evaluation_locked(evaluations)
        if selected is None:
            return "no_liveness_candidate"
        if not self._task_liveness_chain_available_locked(selected.request.resource_unit_id):
            return "liveness_task_active"
        return None

    def _preferred_waiting_task_locked(self) -> tuple[TaskRequest | None, str]:
        evaluations = self._evaluate_waiting_tasks_locked(hypothetical=True)
        selected = self._select_waiting_task_evaluation_locked(evaluations)
        if selected is not None:
            grant_class = "normal" if selected.reason is None else "liveness"
            return selected.request, grant_class
        return None, ""

    def _grant_task_locked(
        self,
        request: TaskRequest,
        plan: _TaskAdmissionPlan,
        *,
        liveness: bool,
    ) -> TaskGrant:
        unit = self._units[str(request.resource_unit_id)].spec
        if unit.backend == "ray_task":
            if plan.node_id is not None:
                raise RuntimeError("Ray task leases must leave placement to Ray Core")
        elif not str(plan.node_id or "").strip():
            raise RuntimeError(f"{unit.backend} task lease requires a concrete runtime node")
        resource_unit_id = str(request.resource_unit_id)
        if liveness and not self._task_liveness_chain_available_locked(resource_unit_id):
            raise RuntimeError("cannot grant more than one liveness task per downstream chain unit")

        actor_index = plan.actor_index
        lease_id = uuid.uuid4().hex
        if unit.backend == "ray_actor":
            if actor_index is None:
                raise RuntimeError("Ray actor task admission requires a concrete actor slot")
            actor_slot = (unit.resource_unit_id, int(actor_index))
            if self._actor_slot_lease_count_locked(actor_slot) >= unit.actor_prefetch_depth:
                raise RuntimeError("Ray actor prefetch slot changed during atomic task admission")
            execution_slot_id = f"ray_actor:{unit.resource_unit_id}:{int(actor_index)}"
        else:
            actor_slot = None
            execution_slot_id = f"{unit.backend}:{unit.resource_unit_id}:{lease_id}"
        lease = TaskLease(
            lease_id=lease_id,
            query_id=self.graph.query_id,
            resource_unit_id=resource_unit_id,
            task_id=str(request.task_id),
            attempt_id=str(request.attempt_id),
            node_id=plan.node_id,
            execution_slot_id=execution_slot_id,
            actor_index=actor_index,
            resources=plan.resources,
            output_window_bytes=int(plan.output_window_bytes),
            liveness=liveness,
            allocation_generation=self.allocation.generation,
        )
        self._task_leases[lease.lease_id] = lease
        self._active_attempt_leases[(lease.task_id, lease.attempt_id)] = lease.lease_id
        if actor_slot is not None:
            if actor_slot not in self._active_actor_slots:
                self._active_actor_slots[actor_slot] = lease.lease_id
            else:
                self._queued_actor_slot_leases.setdefault(actor_slot, deque()).append(lease.lease_id)
        self._waiting_task_inputs.pop((lease.task_id, lease.attempt_id), None)
        self._recompute_unit_queued_input_locked(lease.resource_unit_id)
        if liveness:
            self._active_liveness_task_lease_ids_by_unit[lease.resource_unit_id] = lease.lease_id
            self._task_liveness_grants_total += 1
        self._publish_change_locked()
        return TaskGrant(True, lease=lease, liveness=liveness)

    def _on_task_lease_removed_locked(self, lease: TaskLease) -> None:
        if lease.actor_index is None:
            return
        actor_slot = (lease.resource_unit_id, int(lease.actor_index))
        owner = self._active_actor_slots.get(actor_slot)
        queued = self._queued_actor_slot_leases.get(actor_slot)
        if owner == lease.lease_id:
            if queued:
                self._active_actor_slots[actor_slot] = queued.popleft()
                if not queued:
                    self._queued_actor_slot_leases.pop(actor_slot, None)
            else:
                self._active_actor_slots.pop(actor_slot, None)
        elif queued and lease.lease_id in queued:
            queued.remove(lease.lease_id)
            if not queued:
                self._queued_actor_slot_leases.pop(actor_slot, None)
        else:
            raise RuntimeError("Ray actor slot ownership index is inconsistent")

    def release_task_lease(self, lease_id: str, *, attempt_id: str) -> bool:
        return self._release_task_lease(lease_id, attempt_id=attempt_id, terminal=True)

    def abandon_task_lease(self, lease_id: str, *, attempt_id: str) -> bool:
        """Release admission that never reached task submission.

        Abandonment keeps the attempt eligible for a later placement retry;
        terminal release permanently fences replay of that attempt identity.
        """
        return self._release_task_lease(lease_id, attempt_id=attempt_id, terminal=False)

    def finish_task_with_outputs(
        self,
        task_lease_id: str,
        *,
        attempt_id: str,
        outputs: tuple[OutputBlockRequest, ...] | list[OutputBlockRequest],
    ) -> tuple[OutputBlockLease, ...]:
        """Atomically replace an FTE task window with its materialized outputs."""
        lease_key = str(task_lease_id)
        attempt_key = str(attempt_id)
        requests = tuple(outputs)
        with self._lock:
            task = self._task_leases.get(lease_key)
            if task is None:
                raise RuntimeError(f"FTE task lease is not active: {lease_key}")
            if task.attempt_id != attempt_key:
                raise RuntimeError(f"FTE task lease attempt mismatch: lease={task.attempt_id} result={attempt_key}")
            unit = self._units[task.resource_unit_id]
            if unit.spec.backend != "ray_worker":
                raise RuntimeError(
                    f"atomic FTE completion requires a native fragment lease: "
                    f"{task.resource_unit_id} uses {unit.spec.backend}"
                )
            seen_blocks: set[str] = set()
            for request in requests:
                if str(request.query_id) != task.query_id:
                    raise RuntimeError("FTE output query_id does not match its task lease")
                if str(request.producer_unit_id) != task.resource_unit_id:
                    raise RuntimeError("FTE output resource_unit_id does not match its task lease")
                if str(request.task_lease_id) != task.lease_id:
                    raise RuntimeError("FTE output task_lease_id does not match its task lease")
                if str(request.attempt_id) != task.attempt_id:
                    raise RuntimeError("FTE output attempt_id does not match its task lease")
                block_id = str(request.block_id).strip()
                if not block_id:
                    raise RuntimeError("FTE output block_id must be non-empty")
                if block_id in seen_blocks or block_id in self._output_lease_by_block:
                    raise RuntimeError(f"FTE output block_id is not unique: {block_id}")
                seen_blocks.add(block_id)
                size_bytes = int(request.size_bytes)
                if size_bytes <= 0:
                    raise RuntimeError(f"FTE output block size must be positive: {block_id}")

            output_leases = tuple(
                OutputBlockLease(
                    lease_id=uuid.uuid4().hex,
                    query_id=task.query_id,
                    producer_unit_id=task.resource_unit_id,
                    task_lease_id=task.lease_id,
                    attempt_id=task.attempt_id,
                    block_id=str(request.block_id),
                    node_id=task.node_id,
                    size_bytes=int(request.size_bytes),
                    state="unit_queue",
                    liveness=False,
                    allocation_generation=self.allocation.generation,
                )
                for request in requests
            )
            for request in requests:
                self._record_output_observation_locked(request)
            for output_lease in output_leases:
                self._output_leases[output_lease.lease_id] = output_lease
                self._output_lease_by_block[output_lease.block_id] = output_lease.lease_id
            self._recompute_unit_queued_output_locked(task.resource_unit_id)

            self._task_leases.pop(task.lease_id, None)
            self._record_task_finished_locked(task)
            self._on_task_lease_removed_locked(task)
            self._active_attempt_leases.pop((task.task_id, task.attempt_id), None)
            self._terminal_attempts.add((task.task_id, task.attempt_id))
            self._maybe_clear_task_liveness_locked(task.lease_id)
            self._publish_change_locked()
            return output_leases

    def _release_task_lease(self, lease_id: str, *, attempt_id: str, terminal: bool) -> bool:
        lease_key = str(lease_id)
        with self._lock:
            lease = self._task_leases.get(lease_key)
            if lease is None or lease.attempt_id != str(attempt_id):
                return False
            self._task_leases.pop(lease_key, None)
            if terminal:
                self._record_task_finished_locked(lease)
            else:
                self._generated_output_count_by_task_lease.pop(lease.lease_id, None)
            self._on_task_lease_removed_locked(lease)
            self._active_attempt_leases.pop((lease.task_id, lease.attempt_id), None)
            if terminal:
                self._terminal_attempts.add((lease.task_id, lease.attempt_id))
            self._maybe_clear_task_liveness_locked(lease_key)
            self._publish_change_locked()
            return True

    def _maybe_clear_task_liveness_locked(self, task_lease_id: str) -> None:
        """Return one unit's downstream-chain escape when its task terminates."""
        task_key = str(task_lease_id)
        if task_key in self._task_leases:
            return
        for resource_unit_id, lease_id in tuple(self._active_liveness_task_lease_ids_by_unit.items()):
            if lease_id == task_key:
                self._active_liveness_task_lease_ids_by_unit.pop(resource_unit_id)
                return

    def try_acquire_output_block(self, request: OutputBlockRequest) -> OutputBlockGrant:
        with self._lock:
            fatal_reason, _unit, _task, _size = self._output_request_identity_error_locked(request)
            if fatal_reason is not None:
                return OutputBlockGrant(False, blocked_reason=fatal_reason, fatal=True)
            if self._record_output_observation_locked(request):
                self._publish_change_locked()
            blocked_reason, fatal, _ = self._normal_output_block_reason_locked(request)
            if blocked_reason is None:
                return self._grant_output_block_locked(request, liveness=False)
            if fatal or blocked_reason not in _SOFT_OUTPUT_BLOCK_REASONS:
                return OutputBlockGrant(False, blocked_reason=blocked_reason, fatal=fatal)
            if not self._output_liveness_chain_available_locked(request.producer_unit_id):
                return OutputBlockGrant(False, blocked_reason="liveness_output_active")
            if self._has_normal_output_candidate_locked():
                return OutputBlockGrant(False, blocked_reason="normal_output_candidate_available")
            if not self._output_consumer_starving_locked(request.producer_unit_id):
                return OutputBlockGrant(False, blocked_reason="output_liveness_not_needed")
            preferred = self._preferred_output_liveness_unit_locked(request)
            if preferred != str(request.producer_unit_id):
                return OutputBlockGrant(False, blocked_reason="liveness_output_candidate_not_selected")
            return self._grant_output_block_locked(request, liveness=True)

    def try_acquire_next_queued_output_block(
        self,
        candidate_block_ids: set[str],
    ) -> tuple[OutputBlockRequest | None, OutputBlockGrant | None]:
        """Select and resolve one driver-owned output waiter in one lock pass."""
        with self._lock:
            candidate_ids = {str(block_id) for block_id in candidate_block_ids}
            fatal: list[tuple[tuple[int, int], OutputBlockRequest, str, int]] = []
            normal: list[tuple[tuple[int, int], OutputBlockRequest, str, int]] = []
            soft_blocked: list[tuple[tuple[int, int], OutputBlockRequest, str, int]] = []
            for order, (block_id, request) in enumerate(self._waiting_output_blocks.items()):
                if block_id not in candidate_ids:
                    continue
                rank = (
                    self._reverse_topological_rank.get(
                        str(request.producer_unit_id),
                        len(self._reverse_topological_rank),
                    ),
                    order,
                )
                reason, is_fatal, delta = self._normal_output_block_reason_locked(request)
                ranked = (rank, request, "" if reason is None else reason, delta)
                if is_fatal:
                    fatal.append(ranked)
                elif reason is None:
                    normal.append(ranked)
                elif reason in _SOFT_OUTPUT_BLOCK_REASONS:
                    soft_blocked.append(ranked)

            if fatal:
                _, request, reason, _ = min(fatal, key=lambda item: item[0])
                return request, OutputBlockGrant(
                    False,
                    blocked_reason=reason,
                    fatal=True,
                )
            if normal:
                _, request, _, _ = min(normal, key=lambda item: item[0])
                return request, self._grant_output_block_locked(
                    request,
                    liveness=False,
                    initial_state="unit_queue",
                )
            if not soft_blocked:
                return None, None

            liveness_candidates = [
                item
                for item in soft_blocked
                if self._output_consumer_starving_locked(item[1].producer_unit_id)
                and self._output_liveness_chain_available_locked(item[1].producer_unit_id)
            ]
            if not liveness_candidates:
                _, request, _, _ = min(soft_blocked, key=lambda item: item[0])
                return request, OutputBlockGrant(
                    False,
                    blocked_reason=(
                        "liveness_output_active"
                        if self._active_liveness_output_lease_ids_by_unit
                        else "output_liveness_not_needed"
                    ),
                )
            _, request, _, _ = min(liveness_candidates, key=lambda item: item[0])
            return request, self._grant_output_block_locked(
                request,
                liveness=True,
                initial_state="unit_queue",
            )

    def _normal_output_block_reason_locked(
        self,
        request: OutputBlockRequest,
    ) -> tuple[str | None, bool, int]:
        fatal_reason, unit, task, size = self._output_request_identity_error_locked(request)
        if fatal_reason is not None:
            return fatal_reason, True, 0
        assert unit is not None and task is not None
        block_id = str(request.block_id).strip()
        excluded_waiter = block_id if self._waiting_output_blocks.get(block_id) == request else None
        # Estimated objects still buffered inside a running generator are a
        # separate Ray Data usage category from every managed ObjectRef that
        # has already been exposed to the executor. Pulling this block adds its
        # exact bytes; it does not replace the pending-generator estimate.
        delta = size
        object_store_backpressure_disabled = (
            unit.spec.resource_unit_id in self._materializing_object_store_unlimited_unit_ids_locked()
        )
        if delta > 0:
            soft = self._unit_soft_block_reason_locked(
                unit.spec.resource_unit_id,
                ResourceVector(object_store_bytes=delta),
                ignore_object_store=object_store_backpressure_disabled,
                excluded_waiting_block_id=excluded_waiter,
                object_store_request_kind="output",
            )
            if soft is not None:
                return soft, False, delta
        return None, False, delta

    def _has_normal_output_candidate_locked(self) -> bool:
        for resource_unit_id in self.graph.reverse_topological_unit_ids():
            unit = self._units[resource_unit_id]
            if not unit.runnable or unit.completed or unit.queued_output_bytes <= 0:
                continue
            task = next(
                (lease for lease in self._task_leases.values() if lease.resource_unit_id == resource_unit_id),
                None,
            )
            if task is None:
                continue
            target = unit.spec.target_output_block_bytes
            size = min(unit.queued_output_bytes, target if target > 0 else unit.queued_output_bytes)
            if size <= 0:
                continue
            request = OutputBlockRequest(
                query_id=self.graph.query_id,
                producer_unit_id=resource_unit_id,
                task_lease_id=task.lease_id,
                attempt_id=task.attempt_id,
                block_id=f"hypothetical:{resource_unit_id}:{task.lease_id}",
                size_bytes=size,
            )
            reason, _, _ = self._normal_output_block_reason_locked(request)
            if reason is None:
                return True
        return False

    def _preferred_output_liveness_unit_locked(
        self,
        current_request: OutputBlockRequest,
    ) -> str | None:
        candidates: list[str] = []
        for resource_unit_id in self.graph.reverse_topological_unit_ids():
            unit = self._units[resource_unit_id]
            if not unit.runnable or unit.completed:
                continue
            if not self._output_consumer_starving_locked(resource_unit_id):
                continue
            if not self._output_liveness_chain_available_locked(resource_unit_id):
                continue
            if resource_unit_id != str(current_request.producer_unit_id) and unit.queued_output_bytes <= 0:
                continue
            if resource_unit_id == str(current_request.producer_unit_id):
                request = current_request
            else:
                task = next(
                    (lease for lease in self._task_leases.values() if lease.resource_unit_id == resource_unit_id),
                    None,
                )
                if task is None:
                    continue
                target = unit.spec.target_output_block_bytes
                size = min(
                    unit.queued_output_bytes,
                    target if target > 0 else unit.queued_output_bytes,
                )
                if size <= 0:
                    continue
                request = OutputBlockRequest(
                    query_id=self.graph.query_id,
                    producer_unit_id=resource_unit_id,
                    task_lease_id=task.lease_id,
                    attempt_id=task.attempt_id,
                    block_id=f"hypothetical-liveness:{resource_unit_id}:{task.lease_id}",
                    size_bytes=size,
                )
            reason, fatal, _ = self._normal_output_block_reason_locked(request)
            if not fatal and reason in _SOFT_OUTPUT_BLOCK_REASONS:
                candidates.append(resource_unit_id)
        return candidates[0] if candidates else None

    def _output_liveness_chain_available_locked(self, producer_unit_id: str) -> bool:
        """Keep one escape per unit along one strictly downstream dependency chain.

        A producer block can remain in ``unit_queue`` while a downstream UDF
        has already consumed earlier blocks and is waiting to publish its own
        output. A single query-global output token deadlocks that pipeline: the
        downstream output cannot drain, so it can never free capacity for the
        upstream token. Allowing only strict descendants of every active token
        advances that same dependency chain without opening parallel branches
        or admitting a second escaped block from the same producer.
        """

        producer_key = str(producer_unit_id)
        active_unit_ids = set(self._active_liveness_output_lease_ids_by_unit)
        if not active_unit_ids:
            return True
        if producer_key in active_unit_ids:
            return False
        return active_unit_ids.issubset(self._transitive_upstream_unit_ids[producer_key])

    def _output_consumer_starving_locked(self, producer_unit_id: str) -> bool:
        if self._external_consumer_waiting:
            return True
        eligible_unit_ids = set(self._eligible_resource_unit_ids_locked())
        downstream_ids = [
            unit.resource_unit_id
            for unit in self.graph.units
            if str(producer_unit_id) in unit.input_unit_ids
            and unit.resource_unit_id in eligible_unit_ids
            and not self._units[unit.resource_unit_id].completed
        ]
        if not downstream_ids:
            return False
        # A downstream unit with a partial input bundle and no active task is
        # still starving: it needs another producer block before its compute
        # batch can be admitted.  Requiring queued_input_bytes == 0 creates a
        # cross-lease deadlock when the producer's sole liveness lease is the
        # partial bundle already waiting downstream.
        return any(
            self._active_task_count_for_unit_locked(resource_unit_id) == 0 for resource_unit_id in downstream_ids
        )

    def _grant_output_block_locked(
        self,
        request: OutputBlockRequest,
        *,
        liveness: bool,
        initial_state: str = "generator_pending",
    ) -> OutputBlockGrant:
        task = self._task_leases.get(str(request.task_lease_id))
        if task is None:
            raise RuntimeError("cannot grant an output block without an active task lease")
        producer_unit_id = str(request.producer_unit_id)
        if liveness and not self._output_liveness_chain_available_locked(producer_unit_id):
            raise RuntimeError("cannot grant more than one liveness output per downstream chain unit")
        self._record_output_observation_locked(request)
        lease = OutputBlockLease(
            lease_id=uuid.uuid4().hex,
            query_id=self.graph.query_id,
            producer_unit_id=producer_unit_id,
            task_lease_id=str(request.task_lease_id),
            attempt_id=str(request.attempt_id),
            block_id=str(request.block_id),
            node_id=task.node_id,
            size_bytes=int(request.size_bytes),
            state=str(initial_state),
            liveness=liveness,
            allocation_generation=self.allocation.generation,
        )
        self._output_leases[lease.lease_id] = lease
        self._output_lease_by_block[lease.block_id] = lease.lease_id
        self._waiting_output_blocks.pop(lease.block_id, None)
        self._recompute_unit_queued_output_locked(lease.producer_unit_id)
        if liveness:
            self._active_liveness_output_lease_ids_by_unit[lease.producer_unit_id] = lease.lease_id
            self._output_liveness_grants_total += 1
        self._publish_change_locked()
        return OutputBlockGrant(True, lease=lease, liveness=liveness)

    def transition_output_block(self, lease_id: str, state: str) -> bool:
        lease_key = str(lease_id)
        target = str(state)
        if target not in _OUTPUT_STATES or target == "released":
            raise ValueError(f"invalid output lease transition target: {target}")
        with self._lock:
            lease = self._output_leases.get(lease_key)
            if lease is None:
                return False
            current_index = _OUTPUT_STATES.index(lease.state)
            target_index = _OUTPUT_STATES.index(target)
            if target_index != current_index + 1:
                raise ValueError(f"invalid output lease transition: {lease.state} -> {target}")
            self._output_leases[lease_key] = OutputBlockLease(**{**asdict(lease), "state": target})
            if (
                target == "downstream_input"
                and self._active_liveness_output_lease_ids_by_unit.get(lease.producer_unit_id) == lease_key
            ):
                # A liveness escape limits each producer to one full block at
                # a time. The handed-off object stays charged, while returning
                # this unit's token lets a partial downstream batch request its
                # next block.
                self._active_liveness_output_lease_ids_by_unit.pop(lease.producer_unit_id)
            if target == "downstream_input":
                self._maybe_clear_task_liveness_locked(lease.task_lease_id)
            self._recompute_unit_queued_output_locked(lease.producer_unit_id)
            self._publish_change_locked()
            return True

    def release_output_block(self, lease_id: str) -> bool:
        lease_key = str(lease_id)
        with self._lock:
            lease = self._output_leases.pop(lease_key, None)
            if lease is None:
                return False
            self._output_lease_by_block.pop(lease.block_id, None)
            self._terminal_output_blocks.add(lease.block_id)
            if self._active_liveness_output_lease_ids_by_unit.get(lease.producer_unit_id) == lease_key:
                self._active_liveness_output_lease_ids_by_unit.pop(lease.producer_unit_id)
            self._maybe_clear_task_liveness_locked(lease.task_lease_id)
            self._recompute_unit_queued_output_locked(lease.producer_unit_id)
            self._publish_change_locked()
            return True

    def cancel(self, reason: str) -> dict[str, int]:
        with self._lock:
            if self._cancelled:
                return {"task_lease_count": 0, "output_lease_count": 0}
            task_count = len(self._task_leases)
            output_count = len(self._output_leases)
            for lease in self._task_leases.values():
                self._terminal_attempts.add((lease.task_id, lease.attempt_id))
            self._task_leases.clear()
            self._active_attempt_leases.clear()
            self._submitted_actor_slots.clear()
            self._retiring_actor_unit_ids.clear()
            self._actor_node_by_slot.clear()
            self._active_actor_slots.clear()
            self._queued_actor_slot_leases.clear()
            self._waiting_task_inputs.clear()
            for resource_unit_id in self._units:
                self._recompute_unit_queued_input_locked(resource_unit_id)
            self._output_leases.clear()
            self._output_lease_by_block.clear()
            self._observed_output_blocks.clear()
            self._generated_output_count_by_task_lease.clear()
            self._waiting_output_blocks.clear()
            for resource_unit_id in self._units:
                self._recompute_unit_queued_output_locked(resource_unit_id)
            self._active_liveness_task_lease_ids_by_unit.clear()
            self._active_liveness_output_lease_ids_by_unit.clear()
            for unit in self._units.values():
                if unit.spec.backend == "ray_actor":
                    unit.actor_ready = False
            self._allocation_admission_open = False
            self._cancelled = True
            self._cancel_reason = str(reason)
            self._publish_change_locked()
            return {"task_lease_count": task_count, "output_lease_count": output_count}

    def fail(self, reason: str) -> dict[str, int]:
        """Fence terminally failed work without releasing physical ownership.

        A driver-side protocol or actor failure is not proof that already
        submitted Ray work has stopped. Keep every task, output, and actor-
        process lease charged until ordered query teardown supplies
        that proof; only new task/output/actor submission becomes fatal.
        """

        failure_reason = str(reason or "query_failed")
        with self._lock:
            counts = {
                "task_lease_count": len(self._task_leases),
                "output_lease_count": len(self._output_leases),
            }
            if self._cancelled or self._failed:
                return counts
            self._failed = True
            self._failure_reason = failure_reason
            self._allocation_admission_open = False
            self._publish_change_locked()
            return counts

    def _active_task_count_for_unit_locked(self, resource_unit_id: str) -> int:
        return sum(1 for lease in self._task_leases.values() if lease.resource_unit_id == resource_unit_id)

    def _live_output_bytes_for_task_locked(self, task_lease_id: str) -> int:
        task_key = str(task_lease_id)
        return sum(lease.size_bytes for lease in self._output_leases.values() if lease.task_lease_id == task_key)

    def _scoped_output_bytes_for_task_locked(
        self,
        task_lease_id: str,
        *,
        states: set[str],
        include_waiting: bool,
        excluded_waiting_block_id: str | None = None,
    ) -> int:
        task_key = str(task_lease_id)
        leased = sum(
            lease.size_bytes
            for lease in self._output_leases.values()
            if lease.task_lease_id == task_key and lease.state in states
        )
        if not include_waiting:
            return leased
        waiting = sum(
            int(request.size_bytes)
            for block_id, request in self._waiting_output_blocks.items()
            if block_id != excluded_waiting_block_id and str(request.task_lease_id) == task_key
        )
        return leased + waiting

    def _producer_output_bytes_for_task_locked(
        self,
        task_lease_id: str,
        *,
        excluded_waiting_block_id: str | None = None,
    ) -> int:
        return self._scoped_output_bytes_for_task_locked(
            task_lease_id,
            states={"generator_pending", "unit_queue"},
            include_waiting=True,
            excluded_waiting_block_id=excluded_waiting_block_id,
        )

    def _downstream_output_bytes_for_task_locked(self, task_lease_id: str) -> int:
        return self._scoped_output_bytes_for_task_locked(
            task_lease_id,
            states={"downstream_input", "external_consumer"},
            include_waiting=False,
        )

    def _live_output_bytes_for_unit_locked(self, resource_unit_id: str) -> int:
        return sum(
            lease.size_bytes for lease in self._output_leases.values() if lease.producer_unit_id == resource_unit_id
        )

    def _task_usage_locked(
        self,
        lease: TaskLease,
        *,
        excluded_waiting_block_id: str | None = None,
    ) -> ResourceVector:
        output_estimate = self._pending_output_estimate_for_lease_locked(lease)
        producer_output = self._producer_output_bytes_for_task_locked(
            lease.lease_id,
            excluded_waiting_block_id=excluded_waiting_block_id,
        )
        downstream_output = self._downstream_output_bytes_for_task_locked(lease.lease_id)
        return _resource_with_object_store(
            lease.resources,
            lease.resources.object_store_bytes + downstream_output + producer_output + output_estimate,
        )

    def _uncovered_inactive_output_bytes_locked(
        self,
        active_task_ids: set[str],
        *,
        resource_unit_id: str | None = None,
        excluded_waiting_block_id: str | None = None,
    ) -> int:
        leased = sum(
            output.size_bytes
            for output in self._output_leases.values()
            if output.task_lease_id not in active_task_ids
            and (resource_unit_id is None or output.producer_unit_id == resource_unit_id)
        )
        # Failure cleanup can retire a completed task concurrently with
        # cancelling its final driver-owned output waiter.  The ObjectRef still
        # exists during that bounded handoff, so keep its exact bytes charged
        # even after the task estimate disappears.
        waiting = sum(
            int(request.size_bytes)
            for block_id, request in self._waiting_output_blocks.items()
            if block_id != excluded_waiting_block_id
            and str(request.task_lease_id) not in active_task_ids
            and (resource_unit_id is None or str(request.producer_unit_id) == resource_unit_id)
        )
        return leased + waiting

    def _query_usage_locked(
        self,
        *,
        excluded_waiting_block_id: str | None = None,
    ) -> ResourceVector:
        total = ResourceVector()
        active_task_ids = set(self._task_leases)
        for lease in self._task_leases.values():
            total = total + self._task_usage_locked(
                lease,
                excluded_waiting_block_id=excluded_waiting_block_id,
            )
        return total + ResourceVector(
            object_store_bytes=self._uncovered_inactive_output_bytes_locked(
                active_task_ids,
                excluded_waiting_block_id=excluded_waiting_block_id,
            )
        )

    def _soft_allocation_usage_locked(
        self,
        *,
        excluded_waiting_block_id: str | None = None,
    ) -> ResourceVector:
        return (
            self._query_usage_locked(
                excluded_waiting_block_id=excluded_waiting_block_id,
            )
            + self._actor_resident_usage_locked()
        )

    def _update_soft_allocation_warning_locked(
        self,
        debt: ResourceVector,
        *,
        now: float,
    ) -> None:
        if debt.is_zero():
            self._soft_allocation_debt_since = None
            return
        if self._soft_allocation_debt_since is None:
            self._soft_allocation_debt_since = now
            return
        if self._soft_allocation_warning_emitted:
            return
        duration = max(0.0, now - self._soft_allocation_debt_since)
        if duration < _SOFT_RESERVATION_WARNING_DELAY_S:
            return
        self._soft_allocation_warning_emitted = True
        logger.warning(
            "Ray query %s has exceeded its soft resource reservation for %.1fs "
            "(debt=%s, allocation=%s). Existing work and fixed actor pools may "
            "continue, but new work can be backpressured or remain pending in "
            "Ray Core. Persistent CPU/GPU/heap debt usually requires more "
            "cluster resources or lower UDF concurrency; object-store debt may "
            "spill and reduce throughput.",
            self.graph.query_id,
            duration,
            debt.to_dict(),
            self.allocation.resources.to_dict(),
        )

    def _unit_usage_locked(
        self,
        resource_unit_id: str,
        *,
        excluded_waiting_block_id: str | None = None,
    ) -> ResourceVector:
        total = ResourceVector()
        active_task_ids: set[str] = set()
        for lease in self._task_leases.values():
            if lease.resource_unit_id != resource_unit_id:
                continue
            active_task_ids.add(lease.lease_id)
            task_usage = self._task_usage_locked(
                lease,
                excluded_waiting_block_id=excluded_waiting_block_id,
            )
            total = total + task_usage
        total = total + ResourceVector(
            object_store_bytes=self._uncovered_inactive_output_bytes_locked(
                active_task_ids,
                resource_unit_id=resource_unit_id,
                excluded_waiting_block_id=excluded_waiting_block_id,
            )
        )
        if self._units[resource_unit_id].spec.backend == "ray_actor":
            total = total + self._actor_resident_usage_locked(
                resource_unit_id=resource_unit_id,
            )
        return total

    def _actor_resident_usage_locked(
        self,
        *,
        node_id: str | None = None,
        resource_unit_id: str | None = None,
    ) -> ResourceVector:
        total = ResourceVector()
        for candidate_unit_id, actor_index in self._submitted_actor_slots:
            actor_node_id = self._actor_node_by_slot.get((candidate_unit_id, actor_index))
            if node_id is not None and actor_node_id != node_id:
                continue
            if resource_unit_id is not None and candidate_unit_id != resource_unit_id:
                continue
            unit = self._units.get(candidate_unit_id)
            if unit is None:
                raise RuntimeError("submitted actor slot references an unknown query unit")
            total = total + unit.spec.resident_per_actor
        return total

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            usage = self._query_usage_locked()
            soft_allocation_usage = self._soft_allocation_usage_locked()
            soft_allocation_debt = _positive_difference(
                soft_allocation_usage,
                self.allocation.resources,
            )
            now = time.monotonic()
            self._update_soft_allocation_warning_locked(
                soft_allocation_debt,
                now=now,
            )
            frontier_barriers = self._frontier_materialization_barriers_locked()
            eligible_unit_ids = self._eligible_resource_unit_ids_locked()
            object_store_unlimited_unit_ids = self._materializing_object_store_unlimited_unit_ids_locked()
            object_store_budget = self._object_store_budget_state_locked()
            object_store_reservation_unit_ids = set(object_store_budget.reservation_unit_ids)
            preferred_task, grant_class = self._preferred_waiting_task_locked()
            actor_resident_usage_by_node = {
                node_id: self._actor_resident_usage_locked(node_id=node_id)
                for node_id in sorted(set(self._actor_node_by_slot.values()))
            }
            submitted_actor_slots = set(self._submitted_actor_slots)
            ready_actor_slots = set(self._actor_node_by_slot)
            active_task_lease_ids_by_unit = dict(self._active_liveness_task_lease_ids_by_unit)
            active_output_lease_ids_by_unit = dict(self._active_liveness_output_lease_ids_by_unit)
            return {
                "query_id": self.graph.query_id,
                "graph": self.graph.to_dict(),
                "allocation": self.allocation.to_dict(),
                "ray_core_owns_placement": True,
                "usage": usage.to_dict(),
                "soft_allocation_usage": soft_allocation_usage.to_dict(),
                "actor_process_usage": self._actor_resident_usage_locked().to_dict(),
                "actor_process_usage_by_ready_node": {
                    node_id: resources.to_dict() for node_id, resources in actor_resident_usage_by_node.items()
                },
                "submitted_actor_slots": [
                    f"{resource_unit_id}:{actor_index}"
                    for resource_unit_id, actor_index in sorted(submitted_actor_slots)
                ],
                "pending_actor_slots": [
                    f"{resource_unit_id}:{actor_index}"
                    for resource_unit_id, actor_index in sorted(submitted_actor_slots - ready_actor_slots)
                ],
                "ready_actor_slots": {
                    f"{resource_unit_id}:{actor_index}": node_id
                    for (resource_unit_id, actor_index), node_id in sorted(self._actor_node_by_slot.items())
                },
                "retiring_actor_unit_ids": sorted(self._retiring_actor_unit_ids),
                "soft_allocation_debt": soft_allocation_debt.to_dict(),
                "soft_allocation_debt_duration_s": (
                    0.0
                    if self._soft_allocation_debt_since is None
                    else max(0.0, now - self._soft_allocation_debt_since)
                ),
                "soft_allocation_warning_emitted": self._soft_allocation_warning_emitted,
                "soft_object_store_debt_bytes": max(
                    0,
                    soft_allocation_usage.object_store_bytes - self.allocation.resources.object_store_bytes,
                ),
                "allocation_admission_open": self._allocation_admission_open,
                "admission_epoch": int(self._admission_epoch),
                "reservation_ratio": self.reservation_ratio,
                "execution_phase": {
                    "frontier_barrier_ids": [barrier.barrier_id for barrier in frontier_barriers],
                    "eligible_resource_unit_ids": list(eligible_unit_ids),
                    "completed_barrier_ids": sorted(self._completed_materialization_barrier_ids),
                    "object_store_unlimited_unit_ids": list(object_store_unlimited_unit_ids),
                },
                "cancelled": self._cancelled,
                "cancel_reason": self._cancel_reason,
                "failed": self._failed,
                "failure_reason": self._failure_reason,
                "external_consumer_waiting": self._external_consumer_waiting,
                "liveness": {
                    "active_task_lease_ids_by_unit": active_task_lease_ids_by_unit,
                    "active_output_lease_ids_by_unit": active_output_lease_ids_by_unit,
                    "task_grants_total": self._task_liveness_grants_total,
                    "output_grants_total": self._output_liveness_grants_total,
                },
                "admission": {
                    "preferred_task": (
                        None
                        if preferred_task is None
                        else {
                            "task_id": str(preferred_task.task_id),
                            "attempt_id": str(preferred_task.attempt_id),
                            "resource_unit_id": str(preferred_task.resource_unit_id),
                            "grant_class": grant_class,
                        }
                    ),
                    "waiting_tasks": [
                        {
                            "task_id": str(request.task_id),
                            "attempt_id": str(request.attempt_id),
                            "resource_unit_id": str(request.resource_unit_id),
                        }
                        for request, _ in self._waiting_task_inputs.values()
                    ],
                    "reservation_unit_ids": {
                        field_name: list(
                            self._dimension_reservation_unit_ids_locked(
                                field_name,
                                requested_unit_id=None,
                            )
                        )
                        for field_name in _RESOURCE_FIELDS
                    },
                    "object_store": {
                        "limit_bytes": object_store_budget.limit_bytes,
                        "ineligible_usage_bytes": object_store_budget.ineligible_usage_bytes,
                        "shared_pool_bytes": object_store_budget.shared_pool_bytes,
                        "shared_used_bytes": object_store_budget.shared_used_bytes,
                        "shared_remaining_bytes": object_store_budget.shared_remaining_bytes,
                        "query_usage_bytes": object_store_budget.query_usage_bytes,
                    },
                },
                "units": {
                    resource_unit_id: {
                        "runnable": unit.runnable,
                        "actor_ready": unit.actor_ready,
                        "queued_input_bytes": unit.queued_input_bytes,
                        "pending_task_count": unit.pending_task_count,
                        "queued_output_bytes": unit.queued_output_bytes,
                        "pending_output_count": unit.pending_output_count,
                        "bytes_task_outputs_generated": unit.bytes_task_outputs_generated,
                        "num_task_outputs_generated": unit.num_task_outputs_generated,
                        "num_outputs_of_finished_tasks": unit.num_outputs_of_finished_tasks,
                        "num_tasks_finished": unit.num_tasks_finished,
                        "average_output_block_bytes": (
                            None
                            if unit.num_task_outputs_generated == 0
                            else unit.bytes_task_outputs_generated / unit.num_task_outputs_generated
                        ),
                        "average_output_blocks_per_finished_task": (
                            None
                            if unit.num_tasks_finished == 0
                            else unit.num_outputs_of_finished_tasks / unit.num_tasks_finished
                        ),
                        "pending_output_estimate_per_active_task_bytes": (
                            self._pending_output_estimate_per_task_locked(unit.spec)
                        ),
                        "protocol_output_window_max_bytes": unit.spec.output_window_bytes,
                        "completed": unit.completed,
                        "phase_eligible": resource_unit_id in eligible_unit_ids,
                        "object_store_backpressure_disabled": (resource_unit_id in object_store_unlimited_unit_ids),
                        "usage": self._unit_usage_locked(resource_unit_id).to_dict(),
                        "object_store_budget": {
                            "reservation_eligible": (resource_unit_id in object_store_reservation_unit_ids),
                            "task_reserved_bytes": (object_store_budget.units[resource_unit_id].task_reserved_bytes),
                            "output_reserved_bytes": (
                                object_store_budget.units[resource_unit_id].output_reserved_bytes
                            ),
                            "task_internal_usage_bytes": (
                                object_store_budget.units[resource_unit_id].task_internal_usage_bytes
                            ),
                            "output_usage_bytes": (object_store_budget.units[resource_unit_id].output_usage_bytes),
                            "ineligible_usage_bytes": (
                                0
                                if resource_unit_id in object_store_reservation_unit_ids
                                else (
                                    object_store_budget.units[resource_unit_id].task_internal_usage_bytes
                                    + object_store_budget.units[resource_unit_id].output_usage_bytes
                                )
                            ),
                            "shared_used_bytes": (
                                object_store_budget.units[resource_unit_id].shared_used_bytes
                                if resource_unit_id in object_store_reservation_unit_ids
                                else 0
                            ),
                        },
                        "active_task_count": self._active_task_count_for_unit_locked(resource_unit_id),
                    }
                    for resource_unit_id, unit in self._units.items()
                },
                "task_leases": {
                    lease_id: {
                        **asdict(lease),
                        "resources": lease.resources.to_dict(),
                    }
                    for lease_id, lease in self._task_leases.items()
                },
                "active_actor_slots": {
                    f"{resource_unit_id}:{actor_index}": lease_id
                    for (resource_unit_id, actor_index), lease_id in self._active_actor_slots.items()
                },
                "queued_actor_slots": {
                    f"{resource_unit_id}:{actor_index}": list(lease_ids)
                    for (resource_unit_id, actor_index), lease_ids in self._queued_actor_slot_leases.items()
                },
                "output_leases": {lease_id: asdict(lease) for lease_id, lease in self._output_leases.items()},
            }


__all__ = [
    "OutputBlockGrant",
    "OutputBlockLease",
    "OutputBlockLeaseOwner",
    "OutputBlockRequest",
    "RayQueryResourceManager",
    "TaskGrant",
    "TaskLease",
    "TaskRequest",
]
