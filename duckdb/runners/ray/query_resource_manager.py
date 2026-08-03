# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import threading
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from duckdb.runners.ray.admission_ledger import BoundedSet
from duckdb.runners.ray.query_resource_graph import (
    QueryAllocation,
    QueryResourceGraph,
    ResourceUnitSpec,
    ResourceVector,
)

_SOFT_TASK_BLOCK_REASONS = {
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


def _hard_resources(resources: ResourceVector) -> ResourceVector:
    """Drop the spillable object-store flow-control dimension."""
    return ResourceVector(
        cpu=resources.cpu,
        gpu=resources.gpu,
        heap_bytes=resources.heap_bytes,
    )


def _hard_positive_difference(left: ResourceVector, right: ResourceVector) -> ResourceVector:
    return _positive_difference(_hard_resources(left), _hard_resources(right))


def _component_max(resources: tuple[ResourceVector, ...]) -> ResourceVector:
    if not resources:
        return ResourceVector()
    return ResourceVector(
        cpu=max(item.cpu for item in resources),
        gpu=max(item.gpu for item in resources),
        heap_bytes=max(item.heap_bytes for item in resources),
        object_store_bytes=max(item.object_store_bytes for item in resources),
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
    node_id: str
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


@dataclass
class _ContinuationCredit:
    credit_id: str
    parent_task_lease_id: str
    parent_unit_id: str
    eligible_unit_ids: tuple[str, ...]
    reservation_unit_id: str
    node_id: str
    resources: ResourceVector
    allocation_generation: int
    borrowed_by_task_lease_id: str | None = None
    parent_active: bool = True


@dataclass(frozen=True)
class _NewContinuationCreditPlan:
    eligible_unit_ids: tuple[str, ...]
    reservation_unit_id: str
    node_id: str
    resources: ResourceVector


@dataclass(frozen=True)
class _TaskAdmissionPlan:
    resources: ResourceVector = field(default_factory=ResourceVector)
    output_window_bytes: int = 0
    node_id: str = ""
    actor_index: int | None = None
    new_credit: _NewContinuationCreditPlan | None = None
    borrowed_credit_id: str | None = None


@dataclass(frozen=True)
class _DownstreamReservation:
    """One soft shared resource preference for a downstream runnable path.

    Unlike a per-parent continuation credit, this reservation is not charged
    once for every producer. Admission only proves that the bundle remains
    placeable after granting the producer. The real downstream task consumes
    that capacity through its normal task lease when it becomes runnable. If
    preserving the bundle would prevent all progress, the existing bounded
    liveness escape may start one hard-feasible task without preserving it.
    """

    reservation_id: str
    unit_ids: tuple[str, ...]
    resources: ResourceVector
    allowed_node_ids: tuple[str, ...]


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
    node_id: str
    size_bytes: int
    state: str
    liveness: bool
    allocation_generation: int
    continuation_credit_id: str | None = None


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
        reservation_ratio: float = 0.5,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        ratio = float(reservation_ratio)
        if not math.isfinite(ratio) or ratio <= 0 or ratio > 1:
            raise ValueError("reservation_ratio must be in (0, 1]")
        graph.validate_allocation(allocation)
        self.graph = graph
        self.allocation = allocation
        self._allocation_admission_open = True
        self.reservation_ratio = ratio
        self._on_change = on_change
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
        self._reverse_topological_rank = {
            resource_unit_id: index for index, resource_unit_id in enumerate(self._reverse_topological_unit_ids)
        }
        self._reachable_unit_ids: dict[str, tuple[str, ...]] = {}
        self._reachable_udf_unit_ids: dict[str, tuple[str, ...]] = {}
        for source_unit_id in self._topological_unit_ids:
            reachable: set[str] = set()
            pending = list(self._direct_downstream_unit_ids[source_unit_id])
            while pending:
                resource_unit_id = pending.pop()
                if resource_unit_id in reachable:
                    continue
                reachable.add(resource_unit_id)
                pending.extend(self._direct_downstream_unit_ids[resource_unit_id])
            ordered_reachable = tuple(
                resource_unit_id for resource_unit_id in self._topological_unit_ids if resource_unit_id in reachable
            )
            self._reachable_unit_ids[source_unit_id] = ordered_reachable
            self._reachable_udf_unit_ids[source_unit_id] = tuple(
                resource_unit_id
                for resource_unit_id in ordered_reachable
                if self._units[resource_unit_id].spec.backend in {"ray_task", "ray_actor"}
            )
        self._continuation_parent_unit_ids_by_ray_task: dict[str, tuple[str, ...]] = {}
        for resource_unit_id in self._topological_unit_ids:
            unit = self._units[resource_unit_id].spec
            if unit.backend != "ray_task":
                continue
            parent_unit_ids = tuple(
                parent_unit_id
                for parent_unit_id in self._topological_unit_ids
                if self._units[parent_unit_id].spec.backend == "ray_worker"
                and resource_unit_id in self._reachable_udf_unit_ids[parent_unit_id]
            )
            if parent_unit_ids:
                self._continuation_parent_unit_ids_by_ray_task[resource_unit_id] = parent_unit_ids
        self._started_unit_ids: set[str] = set()
        self._task_leases: dict[str, TaskLease] = {}
        self._continuation_credits: dict[str, _ContinuationCredit] = {}
        self._continuation_credit_by_parent: dict[str, str] = {}
        self._continuation_credit_by_borrower: dict[str, str] = {}
        self._active_attempt_leases: dict[tuple[str, str], str] = {}
        self._active_actor_slots: dict[tuple[str, int], str] = {}
        self._queued_actor_slot_leases: dict[tuple[str, int], deque[str]] = {}
        self._terminal_attempts = BoundedSet[tuple[str, str]](capacity=_TERMINAL_IDENTITY_REPLAY_CAPACITY)
        self._waiting_task_inputs: dict[tuple[str, str], tuple[TaskRequest, int]] = {}
        self._waiting_output_blocks: dict[str, OutputBlockRequest] = {}
        self._output_leases: dict[str, OutputBlockLease] = {}
        self._output_lease_by_block: dict[str, str] = {}
        self._terminal_output_blocks = BoundedSet[str](capacity=_TERMINAL_IDENTITY_REPLAY_CAPACITY)
        self._active_liveness_task_lease_id: str | None = None
        self._active_liveness_path_task_lease_ids: set[str] = set()
        self._active_liveness_output_lease_id: str | None = None
        self._task_liveness_grants_total = 0
        self._output_liveness_grants_total = 0
        self._external_consumer_waiting = False
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
        actor_ready: bool | None = None,
        completed: bool = False,
    ) -> None:
        unit_key = str(resource_unit_id)
        with self._lock:
            unit = self._units.get(unit_key)
            if unit is None:
                raise KeyError(f"unit is not registered: {unit_key}")
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
            if actor_ready is not None:
                unit.actor_ready = bool(actor_ready)
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
                self._publish_change_locked()

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
            if retained < 0 or retained > unit.spec.per_task.object_store_bytes:
                raise ValueError("queued task retained input is outside its unit resource spec")
            key = (task_id, attempt_id)
            existing = self._waiting_task_inputs.get(key)
            queued = max(1, retained)
            value = (request, queued)
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

    def note_output_waiting(self, request: OutputBlockRequest) -> None:
        with self._lock:
            if str(request.query_id) != self.graph.query_id:
                raise ValueError("queued output query_id does not match resource manager")
            resource_unit_id = str(request.producer_unit_id)
            if resource_unit_id not in self._units:
                raise KeyError(f"unit is not registered: {resource_unit_id}")
            block_id = str(request.block_id).strip()
            size_bytes = int(request.size_bytes)
            if not block_id or size_bytes <= 0:
                raise ValueError("queued output requires a non-empty block_id and positive size")
            existing = self._waiting_output_blocks.get(block_id)
            if existing is not None and existing != request:
                raise ValueError("queued output block_id was reused with different work")
            self._waiting_output_blocks[block_id] = request
            self._recompute_unit_queued_output_locked(resource_unit_id)
            self._publish_change_locked()

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

    def set_external_consumer_waiting(self, waiting: bool) -> None:
        with self._lock:
            waiting = bool(waiting)
            if self._external_consumer_waiting == waiting:
                return
            self._external_consumer_waiting = waiting
            self._publish_change_locked()

    def set_unit_actor_ready(self, resource_unit_id: str, ready: bool) -> None:
        with self._lock:
            unit = self._units.get(str(resource_unit_id))
            if unit is None:
                raise KeyError(f"unit is not registered: {resource_unit_id}")
            if unit.spec.backend != "ray_actor":
                raise ValueError(f"unit is not a Ray actor unit: {resource_unit_id}")
            ready = bool(ready)
            if unit.actor_ready == ready:
                return
            unit.actor_ready = ready
            self._publish_change_locked()

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
            self.graph.validate_allocation(
                allocation,
                require_full_minimum=False,
            )
            self.allocation = allocation
            self._allocation_admission_open = bool(admission_open)
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
            if self._active_liveness_task_lease_id is not None and not self._continues_active_liveness_path_locked(
                str(request.resource_unit_id),
                selected.plan,
            ):
                return request, TaskGrant(
                    False,
                    blocked_reason="liveness_task_active",
                )
            if not self._active_tasks_allow_liveness_locked(
                str(request.resource_unit_id),
                selected.plan,
            ):
                return request, TaskGrant(
                    False,
                    blocked_reason="liveness_not_needed",
                )
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
        if not self._allocation_admission_open:
            return "allocation_pending", False, empty_plan
        if str(request.query_id) != self.graph.query_id:
            return "query_id_mismatch", True, empty_plan
        unit = self._units.get(str(request.resource_unit_id))
        if unit is None:
            return "unit_not_registered", True, empty_plan
        if unit.completed:
            return "unit_completed", True, empty_plan
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

        free_actor_indices_by_node: dict[str, list[int]] | None = None
        if unit.spec.backend == "ray_actor":
            free_actor_indices_by_node = {}
            for placement in self.allocation.actor_placements:
                if placement.resource_unit_id != unit.spec.resource_unit_id:
                    continue
                slot_key = (unit.spec.resource_unit_id, placement.actor_index)
                if self._actor_slot_lease_count_locked(slot_key) >= unit.spec.actor_prefetch_depth:
                    continue
                free_actor_indices_by_node.setdefault(placement.node_id, []).append(placement.actor_index)
            if not free_actor_indices_by_node:
                return "actor_slot", False, empty_plan

        retained = (
            unit.spec.per_task.object_store_bytes
            if request.retained_input_bytes is None
            else int(request.retained_input_bytes)
        )
        if retained < 0:
            return "invalid_retained_input_bytes", True, empty_plan
        resources = _resource_with_object_store(unit.spec.per_task, retained)
        output_window = unit.spec.output_window_bytes
        commitment = _resource_with_object_store(resources, retained + output_window)
        hard_commitment = _hard_resources(commitment)
        if hard_commitment.exceeded_dimensions(_hard_resources(self.allocation.resources)):
            return "task_exceeds_query_allocation", True, empty_plan

        query_usage = self._query_usage_locked()
        debt = _hard_positive_difference(query_usage, self.allocation.resources)
        if not debt.is_zero():
            return "allocation_debt", False, empty_plan

        borrowed_credit = self._available_continuation_credit_locked(
            unit.spec,
            requested_node_id=request.node_id,
            resources=resources,
            output_window_bytes=output_window,
            query_usage=query_usage,
        )
        credit_template = self._continuation_credit_template_locked(unit.spec)
        credit_resources = ResourceVector() if credit_template is None else credit_template[2]
        admission_commitment = commitment + credit_resources
        if borrowed_credit is not None:
            # Query/node usage already contains the idle credit and any
            # historical output bytes above its object-store window.  Predict
            # the exact post-borrow task usage using the same aggregate-live-
            # output formula as _task_usage_locked, then add only the delta.
            # Comparing the new task in isolation can grant within the cap and
            # become over-cap immediately after ownership transfer.
            prospective_task_usage = self._borrowed_task_usage_locked(
                resources,
                output_window,
                borrowed_credit,
            )
            idle_output_excess = self._idle_continuation_output_excess_locked(
                borrowed_credit,
            )
            admission_commitment = _positive_difference(
                prospective_task_usage,
                idle_output_excess,
            )

        usage_after = query_usage + admission_commitment
        hard_exceeded = _hard_resources(usage_after).exceeded_dimensions(_hard_resources(self.allocation.resources))
        if hard_exceeded:
            reason = "continuation_capacity" if credit_template is not None else f"hard_{hard_exceeded[0]}"
            return reason, False, empty_plan
        query_soft_blocked = (
            admission_commitment.object_store_bytes > 0
            and usage_after.object_store_bytes > self.allocation.resources.object_store_bytes
        )
        downstream_reservations = self._downstream_reservations_locked(unit.spec)
        downstream_reservation_soft_blocked = False
        if downstream_reservations:
            remaining = _hard_resources(self.allocation.resources) - _hard_resources(query_usage + admission_commitment)
            reservation_total = self._downstream_reservation_total(downstream_reservations)
            if not reservation_total.fits_within(remaining):
                downstream_reservation_soft_blocked = True

        new_credit_plan: _NewContinuationCreditPlan | None = None
        if credit_template is not None:
            eligible_unit_ids, reservation_unit_id, credit_resources = credit_template
            node_id, credit_node_id, node_blocked_reason, node_fatal = self._select_task_and_credit_nodes_locked(
                request.node_id,
                hard_commitment,
                credit_resources,
                allow_unallocated_task_node=unit.spec.backend == "ray_worker",
                downstream_reservations=downstream_reservations,
            )
            if node_blocked_reason == "continuation_node_capacity" and downstream_reservations:
                node_id, credit_node_id, node_blocked_reason, node_fatal = self._select_task_and_credit_nodes_locked(
                    request.node_id,
                    hard_commitment,
                    credit_resources,
                    allow_unallocated_task_node=unit.spec.backend == "ray_worker",
                    downstream_reservations=(),
                )
                if node_blocked_reason is None:
                    downstream_reservation_soft_blocked = True
            if node_blocked_reason is None:
                new_credit_plan = _NewContinuationCreditPlan(
                    eligible_unit_ids=eligible_unit_ids,
                    reservation_unit_id=reservation_unit_id,
                    node_id=credit_node_id,
                    resources=credit_resources,
                )
        else:
            allowed_node_ids = (
                {borrowed_credit.node_id}
                if borrowed_credit is not None
                else (set(free_actor_indices_by_node) if free_actor_indices_by_node is not None else None)
            )
            node_id, node_blocked_reason, node_fatal = self._select_task_node_locked(
                request.node_id,
                _hard_resources(admission_commitment),
                allowed_node_ids=allowed_node_ids,
                allow_unallocated_node=unit.spec.backend == "ray_worker",
                downstream_reservations=downstream_reservations,
            )
            if node_blocked_reason == "continuation_node_capacity" and downstream_reservations:
                node_id, node_blocked_reason, node_fatal = self._select_task_node_locked(
                    request.node_id,
                    _hard_resources(admission_commitment),
                    allowed_node_ids=allowed_node_ids,
                    allow_unallocated_node=unit.spec.backend == "ray_worker",
                )
                if node_blocked_reason is None:
                    downstream_reservation_soft_blocked = True
        if node_blocked_reason is not None:
            return node_blocked_reason, node_fatal, empty_plan

        actor_index = None
        if free_actor_indices_by_node is not None:
            candidates = free_actor_indices_by_node.get(node_id, [])
            if not candidates:
                raise RuntimeError("selected actor node has no free concrete actor slot")
            actor_index = min(
                candidates,
                key=lambda index: (
                    self._actor_slot_lease_count_locked((unit.spec.resource_unit_id, index)),
                    index,
                ),
            )
        plan = _TaskAdmissionPlan(
            resources=resources,
            output_window_bytes=output_window,
            node_id=node_id,
            actor_index=actor_index,
            new_credit=new_credit_plan,
            borrowed_credit_id=(None if borrowed_credit is None else borrowed_credit.credit_id),
        )
        if downstream_reservation_soft_blocked:
            return "unit_soft_limit", False, plan
        if active_unit_tasks > 0 and any(
            downstream_id not in self._started_unit_ids
            and self._units[downstream_id].runnable
            and not self._units[downstream_id].completed
            for downstream_id in self._direct_downstream_unit_ids[unit.spec.resource_unit_id]
        ):
            return "unit_soft_limit", False, plan
        if query_soft_blocked:
            return "query_soft_object_store_bytes", False, plan
        if borrowed_credit is None:
            soft_reason = self._unit_soft_block_reason_locked(
                unit.spec.resource_unit_id,
                commitment,
            )
            if soft_reason is not None:
                return soft_reason, False, plan
        return None, False, plan

    def _select_task_node_locked(
        self,
        requested_node_id: str | None,
        commitment: ResourceVector,
        *,
        allowed_node_ids: set[str] | None,
        allow_unallocated_node: bool = False,
        downstream_reservations: tuple[_DownstreamReservation, ...] = (),
    ) -> tuple[str, str | None, bool]:
        allocation_by_node = {
            item.node_id: _hard_resources(item.resources) for item in self.allocation.node_allocations
        }
        requested = "" if requested_node_id is None else str(requested_node_id).strip()
        external_requested = bool(requested and requested not in allocation_by_node)
        if external_requested and (
            not allow_unallocated_node
            or not commitment.is_zero()
            or (allowed_node_ids is not None and requested not in allowed_node_ids)
        ):
            return "", "node_not_allocated", True

        static_candidates = (
            [requested]
            if external_requested
            else [
                node_id
                for node_id, capacity in allocation_by_node.items()
                if commitment.fits_within(capacity)
                and (not requested or node_id == requested)
                and (allowed_node_ids is None or node_id in allowed_node_ids)
            ]
        )
        if not static_candidates:
            return "", "task_does_not_fit_allocated_node", True

        available_candidates: list[tuple[str, ResourceVector]] = []
        debt_detected = False
        for node_id in static_candidates:
            capacity = allocation_by_node.get(node_id, ResourceVector())
            usage = _hard_resources(self._node_usage_locked(node_id))
            if not usage.fits_within(capacity):
                debt_detected = True
                continue
            remaining = capacity - usage
            if commitment.fits_within(remaining):
                available_candidates.append((node_id, remaining))
        if not available_candidates:
            return "", "node_allocation_debt" if debt_detected else "node_capacity", False

        if downstream_reservations:
            safe_candidates = [
                candidate
                for candidate in available_candidates
                if self._placement_preserves_downstream_locked(
                    candidate[0],
                    commitment,
                    downstream_reservations,
                    allocation_by_node,
                )
            ]
            if not safe_candidates:
                return "", "continuation_node_capacity", False
            available_candidates = safe_candidates

        node_id, _ = min(
            available_candidates,
            key=lambda item: (
                item[1].gpu - commitment.gpu,
                item[1].cpu - commitment.cpu,
                item[1].heap_bytes - commitment.heap_bytes,
                item[1].object_store_bytes - commitment.object_store_bytes,
                item[0],
            ),
        )
        return node_id, None, False

    def _select_task_and_credit_nodes_locked(
        self,
        requested_node_id: str | None,
        task_commitment: ResourceVector,
        credit_resources: ResourceVector,
        *,
        allow_unallocated_task_node: bool = False,
        downstream_reservations: tuple[_DownstreamReservation, ...],
    ) -> tuple[str, str, str | None, bool]:
        """Place one parent task and its continuation ownership atomically."""
        allocation_by_node = {
            item.node_id: _hard_resources(item.resources) for item in self.allocation.node_allocations
        }
        requested = "" if requested_node_id is None else str(requested_node_id).strip()
        external_requested = bool(requested and requested not in allocation_by_node)
        if external_requested and (not allow_unallocated_task_node or not task_commitment.is_zero()):
            return "", "", "node_not_allocated", True

        task_nodes = (
            [requested]
            if external_requested
            else [
                node_id
                for node_id, capacity in allocation_by_node.items()
                if task_commitment.fits_within(capacity) and (not requested or node_id == requested)
            ]
        )
        if not task_nodes:
            return "", "", "task_does_not_fit_allocated_node", True
        credit_nodes = [
            node_id for node_id, capacity in allocation_by_node.items() if credit_resources.fits_within(capacity)
        ]
        if not credit_nodes:
            return "", "", "continuation_does_not_fit_allocated_node", True

        candidates: list[tuple[str, str, dict[str, ResourceVector]]] = []
        debt_detected = False
        for task_node_id in task_nodes:
            for credit_node_id in credit_nodes:
                proposed: dict[str, ResourceVector] = {}
                if task_node_id in allocation_by_node:
                    proposed[task_node_id] = task_commitment
                proposed[credit_node_id] = proposed.get(credit_node_id, ResourceVector()) + credit_resources
                remaining_by_node: dict[str, ResourceVector] = {}
                feasible = True
                for node_id, capacity in allocation_by_node.items():
                    usage = _hard_resources(self._node_usage_locked(node_id))
                    if not usage.fits_within(capacity):
                        if node_id in proposed:
                            debt_detected = True
                            feasible = False
                            break
                        continue
                    usage_after = usage + proposed.get(node_id, ResourceVector())
                    if not usage_after.fits_within(capacity):
                        feasible = False
                        break
                    remaining_by_node[node_id] = capacity - usage_after
                if not feasible:
                    continue
                if downstream_reservations and not self._remaining_preserves_downstream_locked(
                    remaining_by_node,
                    downstream_reservations,
                ):
                    continue
                candidates.append((task_node_id, credit_node_id, remaining_by_node))

        if not candidates:
            reason = "node_allocation_debt" if debt_detected else "continuation_node_capacity"
            return "", "", reason, False

        task_node_id, credit_node_id, _ = min(
            candidates,
            key=lambda item: (
                item[2].get(item[0], ResourceVector()).gpu,
                item[2].get(item[0], ResourceVector()).cpu,
                item[2].get(item[0], ResourceVector()).heap_bytes,
                item[2].get(item[0], ResourceVector()).object_store_bytes,
                item[0],
                item[1],
            ),
        )
        return task_node_id, credit_node_id, None, False

    def _placement_preserves_downstream_locked(
        self,
        parent_node_id: str,
        parent_commitment: ResourceVector,
        reservations: tuple[_DownstreamReservation, ...],
        allocation_by_node: dict[str, ResourceVector],
    ) -> bool:
        remaining_by_node: dict[str, ResourceVector] = {}
        for node_id, capacity in allocation_by_node.items():
            usage = _hard_resources(self._node_usage_locked(node_id))
            if node_id == parent_node_id:
                usage = usage + parent_commitment
            if usage.fits_within(capacity):
                remaining_by_node[node_id] = capacity - usage

        return self._remaining_preserves_downstream_locked(
            remaining_by_node,
            reservations,
        )

    def _remaining_preserves_downstream_locked(
        self,
        remaining_by_node: dict[str, ResourceVector],
        reservations: tuple[_DownstreamReservation, ...],
    ) -> bool:
        if not reservations:
            return True

        node_ids = tuple(sorted(remaining_by_node))
        node_index = {node_id: index for index, node_id in enumerate(node_ids)}
        ordered = tuple(
            sorted(
                reservations,
                key=lambda reservation: (
                    len(reservation.allowed_node_ids),
                    -reservation.resources.dominant_share(self.allocation.resources),
                    reservation.reservation_id,
                ),
            )
        )
        initial = tuple(remaining_by_node[node_id] for node_id in node_ids)
        failed_states: set[tuple[int, tuple[ResourceVector, ...]]] = set()

        def place(index: int, remaining: tuple[ResourceVector, ...]) -> bool:
            if index >= len(ordered):
                return True
            state = (index, remaining)
            if state in failed_states:
                return False
            reservation = ordered[index]
            candidates: list[tuple[int, ResourceVector]] = []
            for node_id in reservation.allowed_node_ids:
                position = node_index.get(node_id)
                if position is None:
                    continue
                capacity = remaining[position]
                if reservation.resources.fits_within(capacity):
                    candidates.append((position, capacity - reservation.resources))
            candidates.sort(
                key=lambda item: (
                    item[1].dominant_share(self.allocation.resources),
                    item[1].gpu,
                    item[1].cpu,
                    item[1].heap_bytes,
                    item[1].object_store_bytes,
                    node_ids[item[0]],
                )
            )
            for position, capacity_after in candidates:
                updated = list(remaining)
                updated[position] = capacity_after
                if place(index + 1, tuple(updated)):
                    return True
            failed_states.add(state)
            return False

        return place(0, initial)

    def _actor_slot_lease_count_locked(self, slot: tuple[str, int]) -> int:
        return int(slot in self._active_actor_slots) + len(self._queued_actor_slot_leases.get(slot, ()))

    @staticmethod
    def _unit_concurrency_cap(spec: ResourceUnitSpec) -> int | None:
        return spec.max_concurrency

    @staticmethod
    def _task_commitment(spec: ResourceUnitSpec) -> ResourceVector:
        """Return non-spillable process resources for one runnable task."""
        return ResourceVector(
            cpu=spec.per_task.cpu,
            gpu=spec.per_task.gpu,
            heap_bytes=spec.per_task.heap_bytes,
        )

    def _continuation_credit_template_locked(
        self,
        parent: ResourceUnitSpec,
    ) -> tuple[tuple[str, ...], str, ResourceVector] | None:
        if parent.backend != "ray_worker":
            return None
        eligible_specs = tuple(
            self._units[resource_unit_id].spec
            for resource_unit_id in self._reachable_udf_unit_ids[parent.resource_unit_id]
            if self._units[resource_unit_id].spec.backend == "ray_task" and not self._units[resource_unit_id].completed
        )
        if not eligible_specs:
            return None
        commitments = tuple(self._task_commitment(spec) for spec in eligible_specs)
        resources = _component_max(commitments)
        reservation_spec = max(
            eligible_specs,
            key=lambda spec: (
                self._task_commitment(spec).dominant_share(self.allocation.resources),
                -self._topological_unit_ids.index(spec.resource_unit_id),
            ),
        )
        return (
            tuple(spec.resource_unit_id for spec in eligible_specs),
            reservation_spec.resource_unit_id,
            resources,
        )

    def _available_continuation_credit_locked(
        self,
        unit: ResourceUnitSpec,
        *,
        requested_node_id: str | None,
        resources: ResourceVector,
        output_window_bytes: int,
        query_usage: ResourceVector,
    ) -> _ContinuationCredit | None:
        if unit.backend != "ray_task":
            return None
        if not self._continuation_parent_unit_ids_by_ray_task.get(unit.resource_unit_id):
            return None
        requested = "" if requested_node_id is None else str(requested_node_id).strip()
        commitment = self._task_commitment(unit)
        allocation_by_node = {
            item.node_id: _hard_resources(item.resources) for item in self.allocation.node_allocations
        }
        candidates = [
            credit
            for credit in self._continuation_credits.values()
            if credit.parent_active
            and credit.borrowed_by_task_lease_id is None
            and unit.resource_unit_id in credit.eligible_unit_ids
            and commitment.fits_within(credit.resources)
            and (not requested or credit.node_id == requested)
            and credit.node_id in allocation_by_node
        ]
        if not candidates:
            return None

        def candidate_rank(credit: _ContinuationCredit) -> tuple[Any, ...]:
            prospective_task_usage = self._borrowed_task_usage_locked(
                resources,
                output_window_bytes,
                credit,
            )
            idle_output_excess = self._idle_continuation_output_excess_locked(
                credit,
            )
            incremental = _positive_difference(
                prospective_task_usage,
                idle_output_excess,
            )
            query_exceeded = _hard_resources(query_usage + incremental).exceeded_dimensions(
                _hard_resources(self.allocation.resources)
            )
            node_usage = _hard_resources(self._node_usage_locked(credit.node_id))
            node_exceeded = _hard_resources(node_usage + incremental).exceeded_dimensions(
                allocation_by_node[credit.node_id]
            )
            soft_debt = max(
                0,
                query_usage.object_store_bytes
                + incremental.object_store_bytes
                - self.allocation.resources.object_store_bytes,
            )
            feasible = not query_exceeded and not node_exceeded
            return (
                0 if feasible else 1,
                len(query_exceeded) + len(node_exceeded),
                soft_debt,
                incremental.dominant_share(self.allocation.resources),
                incremental.gpu,
                incremental.cpu,
                incremental.heap_bytes,
                incremental.object_store_bytes,
                credit.credit_id,
            )

        # Historical outputs differ per credit, so selecting by insertion
        # order can repeatedly choose an infeasible borrower while another
        # idle credit is immediately grantable. Rank the full prospective
        # query+node transition and use credit_id only as the stable tie-break.
        return min(candidates, key=candidate_rank)

    def _downstream_reservations_locked(
        self,
        parent: ResourceUnitSpec,
    ) -> tuple[_DownstreamReservation, ...]:
        """Return soft shared bundles that prefer a drainable producer path.

        Native fragments do not consume Vane-owned process resources. Nested
        Ray tasks use the explicit per-parent continuation credit above. A
        streaming actor boundary additionally needs one slot for every inactive
        downstream Ray-task continuation: the actor input producer, actor call,
        and downstream consumer can all remain live while generator output is
        drained, so the transferable component-max process credit alone is not
        enough. Actor process resources are already resident; invocation
        input/output bytes use the query's dynamic object-store budget.

        Reservations include latent units: waiting for a unit to become
        runnable is too late to bias upstream fanout. They remain soft so an
        allocation at the graph's hard minimum can still start through the
        bounded liveness escape.
        """
        reachable = tuple(
            self._units[resource_unit_id]
            for resource_unit_id in self._reachable_unit_ids[parent.resource_unit_id]
            if not self._units[resource_unit_id].completed
        )
        reservations: list[_DownstreamReservation] = []
        allocation_node_ids = tuple(sorted(allocation.node_id for allocation in self.allocation.node_allocations))

        def is_after_streaming_actor(resource_unit_id: str) -> bool:
            if parent.backend == "ray_actor":
                return True
            return any(
                candidate.spec.backend == "ray_actor"
                and resource_unit_id in self._reachable_unit_ids[candidate.spec.resource_unit_id]
                for candidate in reachable
            )

        actor_continuations = tuple(
            unit.spec
            for unit in reachable
            if unit.spec.backend == "ray_task"
            and is_after_streaming_actor(unit.spec.resource_unit_id)
            and self._active_task_count_for_unit_locked(unit.spec.resource_unit_id) == 0
        )
        for spec in actor_continuations:
            resources = self._task_commitment(spec)
            if resources.is_zero():
                continue
            reservations.append(
                _DownstreamReservation(
                    reservation_id=f"actor_continuation:{parent.resource_unit_id}:{spec.resource_unit_id}",
                    unit_ids=(spec.resource_unit_id,),
                    resources=resources,
                    allowed_node_ids=allocation_node_ids,
                )
            )

        for unit in reachable:
            spec = unit.spec
            if spec.backend != "ray_actor":
                continue
            if self._active_task_count_for_unit_locked(spec.resource_unit_id) > 0:
                continue
            resources = self._task_commitment(spec)
            if resources.is_zero():
                continue
            free_node_ids = tuple(
                sorted(
                    {
                        placement.node_id
                        for placement in self.allocation.actor_placements
                        if placement.resource_unit_id == spec.resource_unit_id
                        and (spec.resource_unit_id, placement.actor_index) not in self._active_actor_slots
                    }
                )
            )
            reservations.append(
                _DownstreamReservation(
                    reservation_id=f"actor:{spec.resource_unit_id}",
                    unit_ids=(spec.resource_unit_id,),
                    resources=resources,
                    allowed_node_ids=free_node_ids,
                )
            )
        return tuple(reservations)

    @staticmethod
    def _downstream_reservation_total(
        reservations: tuple[_DownstreamReservation, ...],
    ) -> ResourceVector:
        total = ResourceVector()
        for reservation in reservations:
            total = total + reservation.resources
        return total

    def _dimension_live_demand_unit_ids_locked(
        self,
        field_name: str,
        *,
        requested_unit_id: str | None,
    ) -> tuple[str, ...]:
        unit_ids: list[str] = []
        for resource_unit_id, unit in self._units.items():
            if not unit.runnable or unit.completed:
                continue
            task_demand = (
                (requested_unit_id is not None and resource_unit_id == requested_unit_id)
                or unit.pending_task_count > 0
                or self._active_task_count_for_unit_locked(resource_unit_id) > 0
            )
            output_demand = (
                unit.pending_output_count > 0
                or unit.queued_output_bytes > 0
                or any(lease.producer_unit_id == resource_unit_id for lease in self._output_leases.values())
            )
            if not task_demand and not (field_name == "object_store_bytes" and output_demand):
                continue
            if self._unit_dimension_commitment(unit.spec, field_name) > 0:
                unit_ids.append(resource_unit_id)
        return tuple(unit_ids)

    def _dimension_continuation_unit_ids_locked(
        self,
        field_name: str,
        *,
        live_unit_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        live = set(live_unit_ids)
        continuation_ids: set[str] = set()
        for source_unit_id in live_unit_ids:
            if self._units[source_unit_id].spec.backend != "ray_worker":
                continue
            for resource_unit_id in self._reachable_udf_unit_ids[source_unit_id]:
                unit = self._units[resource_unit_id]
                if (
                    resource_unit_id not in live
                    and not unit.completed
                    and self._unit_dimension_commitment(unit.spec, field_name) > 0
                ):
                    continuation_ids.add(resource_unit_id)
        return tuple(
            resource_unit_id for resource_unit_id in self._topological_unit_ids if resource_unit_id in continuation_ids
        )

    def _dimension_reservation_unit_ids_locked(
        self,
        field_name: str,
        *,
        requested_unit_id: str | None,
    ) -> tuple[str, ...]:
        live_unit_ids = self._dimension_live_demand_unit_ids_locked(
            field_name,
            requested_unit_id=requested_unit_id,
        )
        if requested_unit_id is not None and self._units[requested_unit_id].spec.backend != "ray_worker":
            # A concrete downstream UDF borrows latent continuation reserves.
            # It releases its own lease before a later continuation needs them.
            return live_unit_ids
        continuation_unit_ids = self._dimension_continuation_unit_ids_locked(
            field_name,
            live_unit_ids=live_unit_ids,
        )
        selected = set(live_unit_ids) | set(continuation_unit_ids)
        return tuple(
            resource_unit_id for resource_unit_id in self._topological_unit_ids if resource_unit_id in selected
        )

    def _unit_soft_block_reason_locked(
        self,
        resource_unit_id: str,
        request: ResourceVector,
    ) -> str | None:
        """Enforce dynamic shares among units with real runnable demand.

        DuckDB owns native execution phases. QRM therefore does not infer a
        static barrier scope: every active or queued unit participates in the
        dimensions it actually requests, and unused capacity remains shared.
        """
        usage_by_unit = {key: self._unit_usage_locked(key) for key in self._units}
        current = usage_by_unit[resource_unit_id]
        limits = self.allocation.resources
        for field_name in _RESOURCE_FIELDS:
            amount = getattr(request, field_name)
            if amount <= 0:
                continue
            dimension_unit_ids = self._dimension_reservation_unit_ids_locked(
                field_name,
                requested_unit_id=resource_unit_id,
            )
            if resource_unit_id not in dimension_unit_ids:
                raise RuntimeError(f"unit {resource_unit_id} requested undeclared {field_name} capacity")
            limit = getattr(limits, field_name)
            minimum_by_unit = {
                key: self._unit_dimension_commitment(self._units[key].spec, field_name) for key in dimension_unit_ids
            }
            minimum_total = sum(minimum_by_unit.values())
            if minimum_total <= limit + _EPSILON:
                bonus_pool = max(0.0, limit - minimum_total) * self.reservation_ratio
                bonus = bonus_pool / len(dimension_unit_ids)
                reserved_by_unit = {key: minimum + bonus for key, minimum in minimum_by_unit.items()}
            else:
                reservation_pool = limit * self.reservation_ratio
                reserved_by_unit = {
                    key: reservation_pool * minimum / minimum_total for key, minimum in minimum_by_unit.items()
                }
            maximum_by_unit = {
                key: self._unit_dimension_maximum_locked(self._units[key].spec, field_name)
                for key in dimension_unit_ids
            }
            reserved_by_unit = {key: min(reserved, maximum_by_unit[key]) for key, reserved in reserved_by_unit.items()}
            if field_name in {"heap_bytes", "object_store_bytes"}:
                reserved_by_unit = {key: math.floor(reserved) for key, reserved in reserved_by_unit.items()}
            reserved = reserved_by_unit[resource_unit_id]
            current_amount = getattr(current, field_name)
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
        return None

    @staticmethod
    def _unit_dimension_commitment(spec: ResourceUnitSpec, field_name: str) -> int | float:
        if field_name == "object_store_bytes":
            return int(spec.per_task.object_store_bytes) + int(spec.output_window_bytes)
        return getattr(spec.per_task, field_name)

    def _unit_dimension_maximum_locked(
        self,
        spec: ResourceUnitSpec,
        field_name: str,
    ) -> int | float:
        """Cap a soft share by tasks that can coexist under every hard dimension."""
        concurrency = self._unit_concurrency_cap(spec)
        if spec.backend == "ray_actor":
            concurrency = spec.actor_pool_size
        hard_bound = math.inf if concurrency is None else int(concurrency)
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
            hard_bound = min(hard_bound, dimension_bound)
        if math.isinf(hard_bound):
            return getattr(self.allocation.resources, field_name)
        per_task_amount = self._unit_dimension_commitment(spec, field_name)
        return per_task_amount * max(0, int(hard_bound))

    def _active_tasks_allow_liveness_locked(
        self,
        resource_unit_id: str,
        plan: _TaskAdmissionPlan,
    ) -> bool:
        """Allow one escape only when it can drain an active upstream path."""
        if not self._task_leases:
            return True
        unit_key = str(resource_unit_id)
        if self._active_task_count_for_unit_locked(unit_key) > 0:
            return False
        if plan.borrowed_credit_id is not None:
            return True
        return any(unit_key in self._reachable_unit_ids[lease.resource_unit_id] for lease in self._task_leases.values())

    def _continues_active_liveness_path_locked(
        self,
        resource_unit_id: str,
        plan: _TaskAdmissionPlan,
    ) -> bool:
        """Return whether a soft-blocked task continues the one escape path."""
        active_lease_id = self._active_liveness_task_lease_id
        if active_lease_id is None:
            return False
        unit_key = str(resource_unit_id)
        unit = self._units.get(unit_key)
        if unit is None:
            return False
        active_path_leases = tuple(
            lease
            for lease_id, lease in self._task_leases.items()
            if lease_id in self._active_liveness_path_task_lease_ids
        )
        frontier = tuple(
            lease
            for lease in active_path_leases
            if not any(
                other.lease_id != lease.lease_id
                and other.resource_unit_id in self._reachable_unit_ids[lease.resource_unit_id]
                for other in active_path_leases
            )
        )
        continues_from_frontier = any(
            unit_key in self._reachable_unit_ids[lease.resource_unit_id] for lease in frontier
        )
        credit_id = plan.borrowed_credit_id
        if credit_id is not None:
            credit = self._continuation_credits.get(credit_id)
            return (
                continues_from_frontier
                and credit is not None
                and credit.parent_active
                and credit.parent_task_lease_id in self._active_liveness_path_task_lease_ids
                and credit.borrowed_by_task_lease_id is None
            )
        return continues_from_frontier

    def _task_liveness_block_reason_locked(
        self,
        request: TaskRequest,
        plan: _TaskAdmissionPlan,
    ) -> str | None:
        if self._active_liveness_task_lease_id is not None and not self._continues_active_liveness_path_locked(
            str(request.resource_unit_id),
            plan,
        ):
            return "liveness_task_active"
        if self._has_normal_task_candidate_locked():
            return "normal_candidate_available"
        preferred = self._preferred_liveness_unit_locked()
        if preferred is None:
            return "no_liveness_candidate"
        if preferred != str(request.resource_unit_id):
            return "liveness_candidate_not_selected"
        if not self._active_tasks_allow_liveness_locked(
            str(request.resource_unit_id),
            plan,
        ):
            return "liveness_not_needed"
        return None

    def _has_normal_task_candidate_locked(self) -> bool:
        if self._waiting_task_inputs:
            for waiting_request, _ in self._waiting_task_inputs.values():
                reason, _, _ = self._normal_task_block_reason_locked(
                    waiting_request,
                    hypothetical=True,
                )
                if reason is None:
                    return True
            return False
        for resource_unit_id, unit in self._units.items():
            if not unit.runnable or unit.completed:
                continue
            request = TaskRequest(
                query_id=self.graph.query_id,
                resource_unit_id=resource_unit_id,
                task_id=f"hypothetical:{resource_unit_id}",
                attempt_id="hypothetical",
                node_id=None,
                retained_input_bytes=unit.spec.per_task.object_store_bytes,
            )
            reason, _, _ = self._normal_task_block_reason_locked(
                request,
                hypothetical=True,
            )
            if reason is None:
                return True
        return False

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
        if self._active_liveness_task_lease_id is not None:
            path_candidates = [
                item
                for item in soft
                if self._continues_active_liveness_path_locked(
                    str(item.request.resource_unit_id),
                    item.plan,
                )
                and self._active_tasks_allow_liveness_locked(
                    str(item.request.resource_unit_id),
                    item.plan,
                )
            ]
            if path_candidates:
                soft = path_candidates
        soft_unit_ids = {str(item.request.resource_unit_id) for item in soft}
        ordered_unit_ids = [
            resource_unit_id
            for resource_unit_id in self._reverse_topological_unit_ids
            if resource_unit_id in soft_unit_ids
        ]
        starving_unit_ids = [
            resource_unit_id
            for resource_unit_id in ordered_unit_ids
            if self._units[resource_unit_id].queued_input_bytes > 0
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
        if self._active_liveness_task_lease_id is not None and not self._continues_active_liveness_path_locked(
            str(selected.request.resource_unit_id),
            selected.plan,
        ):
            return "liveness_task_active"
        if not self._active_tasks_allow_liveness_locked(
            str(selected.request.resource_unit_id),
            selected.plan,
        ):
            return "liveness_not_needed"
        return None

    def _preferred_waiting_task_locked(self) -> tuple[TaskRequest | None, str]:
        evaluations = self._evaluate_waiting_tasks_locked(hypothetical=True)
        selected = self._select_waiting_task_evaluation_locked(evaluations)
        if selected is not None:
            grant_class = "normal" if selected.reason is None else "liveness"
            return selected.request, grant_class
        return None, ""

    def _preferred_liveness_unit_locked(self) -> str | None:
        if self._waiting_task_inputs:
            waiting_candidates: list[str] = []
            for resource_unit_id in self.graph.reverse_topological_unit_ids():
                for request, _ in self._waiting_task_inputs.values():
                    if str(request.resource_unit_id) != resource_unit_id:
                        continue
                    reason, fatal, plan = self._normal_task_block_reason_locked(
                        request,
                        hypothetical=True,
                    )
                    path_allowed = self._active_liveness_task_lease_id is None or (
                        self._continues_active_liveness_path_locked(
                            resource_unit_id,
                            plan,
                        )
                        and self._active_tasks_allow_liveness_locked(
                            resource_unit_id,
                            plan,
                        )
                    )
                    if not fatal and reason in _SOFT_TASK_BLOCK_REASONS and path_allowed:
                        waiting_candidates.append(resource_unit_id)
                        break
            if not waiting_candidates:
                return None
            starving = [
                resource_unit_id
                for resource_unit_id in waiting_candidates
                if self._units[resource_unit_id].queued_input_bytes > 0
                and self._active_task_count_for_unit_locked(resource_unit_id) == 0
            ]
            return (starving or waiting_candidates)[0]
        runnable_candidates: list[str] = []
        for resource_unit_id in self.graph.reverse_topological_unit_ids():
            unit = self._units[resource_unit_id]
            if not unit.runnable or unit.completed:
                continue
            request = TaskRequest(
                query_id=self.graph.query_id,
                resource_unit_id=resource_unit_id,
                task_id=f"hypothetical:{resource_unit_id}",
                attempt_id="hypothetical",
                node_id=None,
                retained_input_bytes=unit.spec.per_task.object_store_bytes,
            )
            reason, fatal, plan = self._normal_task_block_reason_locked(
                request,
                hypothetical=True,
            )
            path_allowed = self._active_liveness_task_lease_id is None or (
                self._continues_active_liveness_path_locked(
                    resource_unit_id,
                    plan,
                )
                and self._active_tasks_allow_liveness_locked(
                    resource_unit_id,
                    plan,
                )
            )
            if not fatal and reason in _SOFT_TASK_BLOCK_REASONS and path_allowed:
                runnable_candidates.append(resource_unit_id)
        if not runnable_candidates:
            return None
        starving = [
            resource_unit_id
            for resource_unit_id in runnable_candidates
            if self._units[resource_unit_id].queued_input_bytes > 0
            and self._active_task_count_for_unit_locked(resource_unit_id) == 0
        ]
        return (starving or runnable_candidates)[0]

    def _grant_task_locked(
        self,
        request: TaskRequest,
        plan: _TaskAdmissionPlan,
        *,
        liveness: bool,
    ) -> TaskGrant:
        if not str(plan.node_id).strip():
            raise RuntimeError("cannot grant a query task lease without a concrete Ray node")
        if plan.new_credit is not None and plan.borrowed_credit_id is not None:
            raise RuntimeError("task admission cannot create and borrow a continuation credit")
        continues_active_liveness = liveness and self._continues_active_liveness_path_locked(
            str(request.resource_unit_id),
            plan,
        )
        if liveness and self._active_liveness_task_lease_id is not None and not continues_active_liveness:
            raise RuntimeError("task liveness path changed during atomic admission")
        if liveness and self._active_liveness_task_lease_id is None and self._active_liveness_path_task_lease_ids:
            raise RuntimeError("task liveness path has no active root")
        unit = self._units[str(request.resource_unit_id)].spec
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
            resource_unit_id=str(request.resource_unit_id),
            task_id=str(request.task_id),
            attempt_id=str(request.attempt_id),
            node_id=str(plan.node_id),
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
        if plan.new_credit is not None:
            new_credit = _ContinuationCredit(
                credit_id=uuid.uuid4().hex,
                parent_task_lease_id=lease.lease_id,
                parent_unit_id=lease.resource_unit_id,
                eligible_unit_ids=plan.new_credit.eligible_unit_ids,
                reservation_unit_id=plan.new_credit.reservation_unit_id,
                node_id=plan.new_credit.node_id,
                resources=plan.new_credit.resources,
                allocation_generation=self.allocation.generation,
            )
            self._continuation_credits[new_credit.credit_id] = new_credit
            self._continuation_credit_by_parent[lease.lease_id] = new_credit.credit_id
        elif plan.borrowed_credit_id is not None:
            borrowed_credit = self._continuation_credits.get(plan.borrowed_credit_id)
            if (
                borrowed_credit is None
                or not borrowed_credit.parent_active
                or borrowed_credit.borrowed_by_task_lease_id is not None
                or lease.resource_unit_id not in borrowed_credit.eligible_unit_ids
                or borrowed_credit.node_id != lease.node_id
            ):
                raise RuntimeError("continuation credit changed during atomic task admission")
            borrowed_credit.borrowed_by_task_lease_id = lease.lease_id
            self._continuation_credit_by_borrower[lease.lease_id] = borrowed_credit.credit_id
        self._waiting_task_inputs.pop((lease.task_id, lease.attempt_id), None)
        self._recompute_unit_queued_input_locked(lease.resource_unit_id)
        self._started_unit_ids.add(lease.resource_unit_id)
        if liveness:
            if self._active_liveness_task_lease_id is None:
                self._active_liveness_task_lease_id = lease.lease_id
            self._active_liveness_path_task_lease_ids.add(lease.lease_id)
            self._task_liveness_grants_total += 1
        self._publish_change_locked()
        return TaskGrant(True, lease=lease, liveness=liveness)

    def _delete_continuation_credit_locked(self, credit_id: str) -> None:
        credit = self._continuation_credits.pop(str(credit_id), None)
        if credit is None:
            return
        if self._continuation_credit_by_parent.get(credit.parent_task_lease_id) == credit.credit_id:
            self._continuation_credit_by_parent.pop(credit.parent_task_lease_id, None)
        borrower = credit.borrowed_by_task_lease_id
        if borrower is not None:
            self._continuation_credit_by_borrower.pop(borrower, None)

    def _on_task_lease_removed_locked(self, lease: TaskLease) -> None:
        if lease.actor_index is not None:
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
        self._maybe_return_continuation_credit_locked(lease.lease_id)

        owned_credit_id = self._continuation_credit_by_parent.pop(
            lease.lease_id,
            None,
        )
        if owned_credit_id is not None:
            credit = self._continuation_credits.get(owned_credit_id)
            if credit is None or credit.parent_task_lease_id != lease.lease_id:
                raise RuntimeError("continuation parent index is inconsistent")
            credit.parent_active = False
            if credit.borrowed_by_task_lease_id is None:
                self._delete_continuation_credit_locked(credit.credit_id)

    def _maybe_return_continuation_credit_locked(self, task_lease_id: str) -> None:
        task_key = str(task_lease_id)
        borrowed_credit_id = self._continuation_credit_by_borrower.get(task_key)
        if borrowed_credit_id is None:
            return
        if task_key in self._task_leases or any(
            output.task_lease_id == task_key and output.state not in {"downstream_input", "external_consumer"}
            for output in self._output_leases.values()
        ):
            return
        # Handoff is the continuation ownership boundary.  The physical output
        # lease remains attached to borrowed_credit_id and therefore stays in
        # the credit's aggregate object-store accounting, but the next child
        # task may reuse the compute/heap reservation.  Holding the borrower
        # until physical release deadlocks whenever downstream needs more than
        # one upstream block to form its compute batch.
        self._continuation_credit_by_borrower.pop(task_key, None)
        credit = self._continuation_credits.get(borrowed_credit_id)
        if credit is None or credit.borrowed_by_task_lease_id != task_key:
            raise RuntimeError("continuation borrower index is inconsistent")
        credit.borrowed_by_task_lease_id = None
        if not credit.parent_active:
            self._delete_continuation_credit_locked(credit.credit_id)

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
            total_output_bytes = sum(int(request.size_bytes) for request in requests)
            if total_output_bytes > task.output_window_bytes:
                raise RuntimeError(
                    f"FTE output bytes {total_output_bytes} exceed task window {task.output_window_bytes}: "
                    f"query={task.query_id} unit={task.resource_unit_id} task_lease={task.lease_id} "
                    f"attempt={task.attempt_id}"
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
                target_bytes = int(unit.spec.target_output_block_bytes)
                if target_bytes <= 0 or size_bytes > target_bytes:
                    raise RuntimeError(
                        f"FTE output block {block_id} size {size_bytes} exceeds unit target {target_bytes}"
                    )

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
                    continuation_credit_id=None,
                )
                for request in requests
            )
            for output_lease in output_leases:
                self._output_leases[output_lease.lease_id] = output_lease
                self._output_lease_by_block[output_lease.block_id] = output_lease.lease_id
            self._recompute_unit_queued_output_locked(task.resource_unit_id)

            self._task_leases.pop(task.lease_id, None)
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
            self._on_task_lease_removed_locked(lease)
            self._active_attempt_leases.pop((lease.task_id, lease.attempt_id), None)
            if terminal:
                self._terminal_attempts.add((lease.task_id, lease.attempt_id))
            self._maybe_clear_task_liveness_locked(lease_key)
            self._publish_change_locked()
            return True

    def _maybe_clear_task_liveness_locked(self, task_lease_id: str) -> None:
        """Return one escape after its continuation path leaves the producer."""
        task_key = str(task_lease_id)
        if task_key not in self._active_liveness_path_task_lease_ids:
            return
        if task_key in self._task_leases:
            return
        if any(
            output.task_lease_id == task_key and output.state in {"generator_pending", "unit_queue"}
            for output in self._output_leases.values()
        ):
            return
        self._active_liveness_path_task_lease_ids.remove(task_key)
        if not self._active_liveness_path_task_lease_ids:
            self._active_liveness_task_lease_id = None

    def try_acquire_output_block(self, request: OutputBlockRequest) -> OutputBlockGrant:
        with self._lock:
            blocked_reason, fatal, _ = self._normal_output_block_reason_locked(request)
            if blocked_reason is None:
                return self._grant_output_block_locked(request, liveness=False)
            if fatal or blocked_reason not in _SOFT_OUTPUT_BLOCK_REASONS:
                return OutputBlockGrant(False, blocked_reason=blocked_reason, fatal=fatal)
            if self._active_liveness_output_lease_id is not None:
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

            _, request, _, _ = min(soft_blocked, key=lambda item: item[0])
            if self._active_liveness_output_lease_id is not None:
                return request, OutputBlockGrant(
                    False,
                    blocked_reason="liveness_output_active",
                )
            if not self._output_consumer_starving_locked(request.producer_unit_id):
                return request, OutputBlockGrant(
                    False,
                    blocked_reason="output_liveness_not_needed",
                )
            return request, self._grant_output_block_locked(
                request,
                liveness=True,
                initial_state="unit_queue",
            )

    def _normal_output_block_reason_locked(
        self,
        request: OutputBlockRequest,
    ) -> tuple[str | None, bool, int]:
        if self._cancelled:
            return "query_cancelled", True, 0
        if str(request.query_id) != self.graph.query_id:
            return "query_id_mismatch", True, 0
        unit = self._units.get(str(request.producer_unit_id))
        if unit is None:
            return "unit_not_registered", True, 0
        block_id = str(request.block_id).strip()
        if not block_id:
            return "invalid_block_id", True, 0
        if block_id in self._terminal_output_blocks:
            return "output_block_terminal", True, 0
        task = self._task_leases.get(str(request.task_lease_id))
        if task is None:
            return "task_lease_not_active", True, 0
        if task.resource_unit_id != unit.spec.resource_unit_id:
            return "task_unit_mismatch", True, 0
        if task.attempt_id != str(request.attempt_id):
            return "task_attempt_mismatch", True, 0
        if block_id in self._output_lease_by_block:
            return "output_block_already_leased", True, 0
        size = int(request.size_bytes)
        if size <= 0:
            return "invalid_output_block_size", True, 0

        live = self._live_output_bytes_for_task_locked(task.lease_id)
        before = max(task.output_window_bytes, live)
        after = max(task.output_window_bytes, live + size)
        delta = after - before
        object_limit = self.allocation.resources.object_store_bytes
        usage_after = self._query_usage_locked() + ResourceVector(object_store_bytes=delta)
        if delta > 0 and usage_after.object_store_bytes > object_limit:
            return "query_soft_object_store_bytes", False, delta
        if delta > 0:
            soft = self._unit_soft_block_reason_locked(
                unit.spec.resource_unit_id,
                ResourceVector(object_store_bytes=delta),
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

    def _output_consumer_starving_locked(self, producer_unit_id: str) -> bool:
        if self._external_consumer_waiting:
            return True
        downstream_ids = [
            unit.resource_unit_id for unit in self.graph.units if str(producer_unit_id) in unit.input_unit_ids
        ]
        if not downstream_ids:
            return False
        # A downstream unit with a partial input bundle and no active task is
        # still starving: it needs another producer block before its compute
        # batch can be admitted.  Requiring queued_input_bytes == 0 creates a
        # cross-lease deadlock when the producer's sole liveness lease is the
        # partial bundle already waiting downstream.
        return all(
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
        lease = OutputBlockLease(
            lease_id=uuid.uuid4().hex,
            query_id=self.graph.query_id,
            producer_unit_id=str(request.producer_unit_id),
            task_lease_id=str(request.task_lease_id),
            attempt_id=str(request.attempt_id),
            block_id=str(request.block_id),
            node_id=task.node_id,
            size_bytes=int(request.size_bytes),
            state=str(initial_state),
            liveness=liveness,
            allocation_generation=self.allocation.generation,
            continuation_credit_id=self._continuation_credit_by_borrower.get(task.lease_id),
        )
        self._output_leases[lease.lease_id] = lease
        self._output_lease_by_block[lease.block_id] = lease.lease_id
        self._waiting_output_blocks.pop(lease.block_id, None)
        self._recompute_unit_queued_output_locked(lease.producer_unit_id)
        if liveness:
            self._active_liveness_output_lease_id = lease.lease_id
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
            if target == "downstream_input" and self._active_liveness_output_lease_id == lease_key:
                # A liveness escape limits producer-side progress to one full
                # block at a time. The handed-off object stays charged, while
                # returning the token lets a partial downstream batch request
                # its next block.
                self._active_liveness_output_lease_id = None
            if target == "downstream_input":
                self._maybe_clear_task_liveness_locked(lease.task_lease_id)
                self._maybe_return_continuation_credit_locked(lease.task_lease_id)
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
            if self._active_liveness_output_lease_id == lease_key:
                self._active_liveness_output_lease_id = None
            self._maybe_clear_task_liveness_locked(lease.task_lease_id)
            self._maybe_return_continuation_credit_locked(lease.task_lease_id)
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
            self._continuation_credits.clear()
            self._continuation_credit_by_parent.clear()
            self._continuation_credit_by_borrower.clear()
            self._active_attempt_leases.clear()
            self._active_actor_slots.clear()
            self._queued_actor_slot_leases.clear()
            self._waiting_task_inputs.clear()
            for resource_unit_id in self._units:
                self._recompute_unit_queued_input_locked(resource_unit_id)
            self._output_leases.clear()
            self._output_lease_by_block.clear()
            self._waiting_output_blocks.clear()
            for resource_unit_id in self._units:
                self._recompute_unit_queued_output_locked(resource_unit_id)
            self._active_liveness_task_lease_id = None
            self._active_liveness_path_task_lease_ids.clear()
            self._active_liveness_output_lease_id = None
            self._allocation_admission_open = False
            self._cancelled = True
            self._cancel_reason = str(reason)
            self._publish_change_locked()
            return {"task_lease_count": task_count, "output_lease_count": output_count}

    def _active_task_count_for_unit_locked(self, resource_unit_id: str) -> int:
        return sum(1 for lease in self._task_leases.values() if lease.resource_unit_id == resource_unit_id)

    def _live_output_bytes_for_task_locked(self, task_lease_id: str) -> int:
        task_key = str(task_lease_id)
        credit_id = self._continuation_credit_by_borrower.get(task_key)
        if credit_id is not None:
            # A continuation credit is a shared window for sequential nested
            # tasks in one parent fragment.  Include objects handed off by
            # earlier borrowers so the next output grant sees the aggregate
            # physical footprint and charges any overflow exactly once.
            return self._live_output_bytes_for_credit_locked(credit_id)
        return sum(lease.size_bytes for lease in self._output_leases.values() if lease.task_lease_id == task_key)

    def _live_output_bytes_for_credit_locked(self, credit_id: str) -> int:
        credit_key = str(credit_id)
        return sum(
            lease.size_bytes for lease in self._output_leases.values() if lease.continuation_credit_id == credit_key
        )

    def _borrowed_task_usage_locked(
        self,
        resources: ResourceVector,
        output_window_bytes: int,
        credit: _ContinuationCredit,
    ) -> ResourceVector:
        live_output = self._live_output_bytes_for_credit_locked(credit.credit_id)
        combined_commitment = _resource_with_object_store(
            resources,
            resources.object_store_bytes + max(int(output_window_bytes), live_output),
        )
        return _positive_difference(combined_commitment, credit.resources)

    def _idle_continuation_output_excess_locked(
        self,
        credit: _ContinuationCredit,
    ) -> ResourceVector:
        live_output = self._live_output_bytes_for_credit_locked(credit.credit_id)
        return ResourceVector(
            object_store_bytes=max(
                0,
                live_output - credit.resources.object_store_bytes,
            )
        )

    def _live_output_bytes_for_unit_locked(self, resource_unit_id: str) -> int:
        return sum(
            lease.size_bytes for lease in self._output_leases.values() if lease.producer_unit_id == resource_unit_id
        )

    def _task_usage_locked(self, lease: TaskLease) -> ResourceVector:
        credit_id = self._continuation_credit_by_borrower.get(lease.lease_id)
        if credit_id is None:
            live_output = self._live_output_bytes_for_task_locked(lease.lease_id)
            return _resource_with_object_store(
                lease.resources,
                lease.resources.object_store_bytes + max(lease.output_window_bytes, live_output),
            )
        credit = self._continuation_credits.get(credit_id)
        if credit is None or credit.borrowed_by_task_lease_id != lease.lease_id:
            raise RuntimeError("continuation credit accounting index is inconsistent")
        return self._borrowed_task_usage_locked(
            lease.resources,
            lease.output_window_bytes,
            credit,
        )

    def _uncovered_inactive_output_bytes_locked(
        self,
        active_task_ids: set[str],
        *,
        node_id: str | None = None,
        resource_unit_id: str | None = None,
    ) -> int:
        if resource_unit_id is not None:
            uncovered = sum(
                output.size_bytes
                for output in self._output_leases.values()
                if output.producer_unit_id == resource_unit_id
                and (
                    output.continuation_credit_id is None
                    or output.continuation_credit_id not in self._continuation_credits
                )
                and output.task_lease_id not in active_task_ids
            )
            return uncovered + self._continuation_output_excess_by_unit_locked().get(
                resource_unit_id,
                0,
            )
        uncovered = 0
        outputs_by_idle_credit: dict[str, int] = {}
        for output in self._output_leases.values():
            if node_id is not None and output.node_id != node_id:
                continue
            credit_id = output.continuation_credit_id
            credit = self._continuation_credits.get(credit_id or "")
            if credit is not None:
                borrower = credit.borrowed_by_task_lease_id
                if borrower is not None and borrower in self._task_leases:
                    # The active borrower's task usage accounts the aggregate
                    # output bytes for this credit, including prior borrowers.
                    continue
                outputs_by_idle_credit[credit.credit_id] = (
                    outputs_by_idle_credit.get(credit.credit_id, 0) + output.size_bytes
                )
                continue
            if output.task_lease_id not in active_task_ids:
                uncovered += output.size_bytes
        for credit_id, output_bytes in outputs_by_idle_credit.items():
            credit = self._continuation_credits.get(credit_id)
            if credit is None:
                uncovered += output_bytes
            else:
                uncovered += max(0, output_bytes - credit.resources.object_store_bytes)
        return uncovered

    def _continuation_output_excess_by_unit_locked(self) -> dict[str, int]:
        outputs_by_credit_and_unit: dict[str, dict[str, int]] = {}
        for output in self._output_leases.values():
            credit_id = output.continuation_credit_id
            if credit_id is None or credit_id not in self._continuation_credits:
                continue
            credit = self._continuation_credits[credit_id]
            borrower = credit.borrowed_by_task_lease_id
            if borrower is not None and borrower in self._task_leases:
                # Active borrower usage owns retained input plus aggregate
                # outputs. Attribute that exact delta to the borrower unit;
                # only idle-credit overflow needs cross-unit distribution.
                continue
            by_unit = outputs_by_credit_and_unit.setdefault(credit_id, {})
            by_unit[output.producer_unit_id] = by_unit.get(output.producer_unit_id, 0) + output.size_bytes

        topological_rank = {
            resource_unit_id: index for index, resource_unit_id in enumerate(self._topological_unit_ids)
        }
        excess_by_unit: dict[str, int] = {}
        for credit_id, bytes_by_unit in outputs_by_credit_and_unit.items():
            credit = self._continuation_credits[credit_id]
            remaining_covered = int(credit.resources.object_store_bytes)
            for resource_unit_id in sorted(
                bytes_by_unit,
                key=lambda key: (topological_rank.get(key, len(topological_rank)), key),
            ):
                output_bytes = bytes_by_unit[resource_unit_id]
                covered = min(output_bytes, remaining_covered)
                remaining_covered -= covered
                excess = output_bytes - covered
                if excess > 0:
                    excess_by_unit[resource_unit_id] = excess_by_unit.get(resource_unit_id, 0) + excess
        return excess_by_unit

    def _query_usage_locked(self) -> ResourceVector:
        total = self._actor_resident_usage_locked()
        active_task_ids = set(self._task_leases)
        for credit in self._continuation_credits.values():
            total = total + credit.resources
        for lease in self._task_leases.values():
            total = total + self._task_usage_locked(lease)
        total = total + ResourceVector(object_store_bytes=self._uncovered_inactive_output_bytes_locked(active_task_ids))
        return total

    def _node_usage_locked(self, node_id: str) -> ResourceVector:
        node_key = str(node_id)
        total = self._actor_resident_usage_locked(node_id=node_key)
        active_task_ids: set[str] = set()
        for credit in self._continuation_credits.values():
            if credit.node_id == node_key:
                total = total + credit.resources
        for lease in self._task_leases.values():
            if lease.node_id != node_key:
                continue
            active_task_ids.add(lease.lease_id)
            total = total + self._task_usage_locked(lease)
        total = total + ResourceVector(
            object_store_bytes=self._uncovered_inactive_output_bytes_locked(
                active_task_ids,
                node_id=node_key,
            )
        )
        return total

    def _unit_usage_locked(self, resource_unit_id: str) -> ResourceVector:
        total = self._actor_resident_usage_locked(resource_unit_id=resource_unit_id)
        active_task_ids: set[str] = set()
        for credit in self._continuation_credits.values():
            if credit.reservation_unit_id == resource_unit_id:
                total = total + credit.resources
        for lease in self._task_leases.values():
            if lease.resource_unit_id != resource_unit_id:
                continue
            active_task_ids.add(lease.lease_id)
            task_usage = self._task_usage_locked(lease)
            total = total + task_usage
        total = total + ResourceVector(
            object_store_bytes=self._uncovered_inactive_output_bytes_locked(
                active_task_ids,
                resource_unit_id=resource_unit_id,
            )
        )
        return total

    def _actor_resident_usage_locked(
        self,
        *,
        node_id: str | None = None,
        resource_unit_id: str | None = None,
    ) -> ResourceVector:
        total = ResourceVector()
        for placement in self.allocation.actor_placements:
            if node_id is not None and placement.node_id != node_id:
                continue
            if resource_unit_id is not None and placement.resource_unit_id != resource_unit_id:
                continue
            unit = self._units.get(placement.resource_unit_id)
            if unit is None:
                raise RuntimeError("actor placement references an unknown query unit")
            total = total + unit.spec.resident_per_actor
        return total

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            usage = self._query_usage_locked()
            preferred_task, grant_class = self._preferred_waiting_task_locked()
            allocation_by_node = {item.node_id: item.resources for item in self.allocation.node_allocations}
            reservation_source_unit_ids = {lease.resource_unit_id for lease in self._task_leases.values()} | {
                str(request.resource_unit_id) for request, _ in self._waiting_task_inputs.values()
            }
            downstream_reservations: dict[str, tuple[_DownstreamReservation, ...]] = {}
            for resource_unit_id in self._topological_unit_ids:
                unit = self._units[resource_unit_id]
                if resource_unit_id not in reservation_source_unit_ids or unit.completed:
                    continue
                reservations = self._downstream_reservations_locked(unit.spec)
                if reservations:
                    downstream_reservations[resource_unit_id] = reservations
            node_ids = sorted(
                set(allocation_by_node)
                | {lease.node_id for lease in self._task_leases.values()}
                | {lease.node_id for lease in self._output_leases.values()}
                | {credit.node_id for credit in self._continuation_credits.values()}
            )
            node_usage = {node_id: self._node_usage_locked(node_id) for node_id in node_ids}
            return {
                "query_id": self.graph.query_id,
                "graph": self.graph.to_dict(),
                "allocation": self.allocation.to_dict(),
                "usage": usage.to_dict(),
                "allocation_debt": _hard_positive_difference(
                    usage,
                    self.allocation.resources,
                ).to_dict(),
                "soft_object_store_debt_bytes": max(
                    0,
                    usage.object_store_bytes - self.allocation.resources.object_store_bytes,
                ),
                "allocation_admission_open": self._allocation_admission_open,
                "admission_epoch": int(self._admission_epoch),
                "node_usage": {node_id: resources.to_dict() for node_id, resources in node_usage.items()},
                "node_allocation_debt": {
                    node_id: _hard_positive_difference(
                        resources,
                        allocation_by_node.get(node_id, ResourceVector()),
                    ).to_dict()
                    for node_id, resources in node_usage.items()
                },
                "node_soft_object_store_debt_bytes": {
                    node_id: max(
                        0,
                        resources.object_store_bytes
                        - allocation_by_node.get(
                            node_id,
                            ResourceVector(),
                        ).object_store_bytes,
                    )
                    for node_id, resources in node_usage.items()
                },
                "reservation_ratio": self.reservation_ratio,
                "cancelled": self._cancelled,
                "cancel_reason": self._cancel_reason,
                "external_consumer_waiting": self._external_consumer_waiting,
                "liveness": {
                    "active_task_lease_id": self._active_liveness_task_lease_id,
                    "active_output_lease_id": self._active_liveness_output_lease_id,
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
                    "live_demand_unit_ids": {
                        field_name: list(
                            self._dimension_live_demand_unit_ids_locked(
                                field_name,
                                requested_unit_id=None,
                            )
                        )
                        for field_name in _RESOURCE_FIELDS
                    },
                    "continuation_unit_ids": {
                        field_name: list(
                            self._dimension_continuation_unit_ids_locked(
                                field_name,
                                live_unit_ids=self._dimension_live_demand_unit_ids_locked(
                                    field_name,
                                    requested_unit_id=None,
                                ),
                            )
                        )
                        for field_name in _RESOURCE_FIELDS
                    },
                    "reservation_unit_ids": {
                        field_name: list(
                            self._dimension_reservation_unit_ids_locked(
                                field_name,
                                requested_unit_id=None,
                            )
                        )
                        for field_name in _RESOURCE_FIELDS
                    },
                    "downstream_reservations": {
                        resource_unit_id: [
                            {
                                "reservation_id": reservation.reservation_id,
                                "unit_ids": list(reservation.unit_ids),
                                "resources": reservation.resources.to_dict(),
                                "allowed_node_ids": list(reservation.allowed_node_ids),
                            }
                            for reservation in reservations
                        ]
                        for resource_unit_id, reservations in downstream_reservations.items()
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
                        "completed": unit.completed,
                        "usage": self._unit_usage_locked(resource_unit_id).to_dict(),
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
                "continuation_credits": {
                    credit_id: {
                        **asdict(credit),
                        "resources": credit.resources.to_dict(),
                    }
                    for credit_id, credit in self._continuation_credits.items()
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
