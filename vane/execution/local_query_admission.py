# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Local execution adapter for the query policy also used by Ray.

The policy owns task/input/output commitments. Subprocess pools own physical
workers, and the SHM store owns allocations. Policy grants precede backend
placement, just as Ray grants precede submission to Ray Core.
"""

from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from vane.execution.data_lifecycle import OutputBlockLeaseOwner
from vane.execution.local_query_coordinator import LocalQueryCoordinator
from vane.execution.query_resource_policy import (
    OutputBlockLease,
    OutputBlockRequest,
    QueryResourceManager,
    TaskLease,
    TaskRequest,
)
from vane.execution.query_resource_spec import QueryAllocation, QueryResourceGraph, ResourceUnitSpec
from vane.execution.resource_graph import MaterializationBarrierSpec
from vane.execution.resource_graph_metadata import (
    _node_sort_key,
    _normalize_metadata,
    materialization_barrier_id_for_node,
    native_fragment_unit_id_for_node,
    udf_unit_id_for_node,
    validate_udf_node_ids,
)
from vane.execution.resources import ResourceVector
from vane.execution.udf_admission import AdmissionAuthority, AdmissionLease, LocalSlotAdmissionAuthority
from vane.execution.udf_lifecycle import ExecutionCancellationScope
from vane.execution.udf_local_actor_admission import LocalActorExecutionSlotPool
from vane.execution.udf_resource_policy import udf_resource_spec


def build_local_admission_graph(
    metadata: Mapping[str, Any],
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    query_id: str,
    env: Mapping[str, str],
) -> tuple[QueryResourceGraph, dict[str, str]]:
    _, nodes, terminal_node_ids = _normalize_metadata(metadata)
    bindings = validate_udf_node_ids(metadata, metadata["udf_node_ids"])
    output_ids = {
        node_id: (
            udf_unit_id_for_node(query_id, node_id)
            if node["udf_payload"] is not None
            else native_fragment_unit_id_for_node(query_id, node_id)
        )
        for node_id, node in nodes.items()
    }
    units = []
    barriers = []
    by_binding = {}
    for node_id in sorted(nodes, key=_node_sort_key):
        node = nodes[node_id]
        native_id = native_fragment_unit_id_for_node(query_id, node_id)
        # Local native rows stay in DuckDB; they do not create a separate
        # managed object-store task/window between the UDF boundaries.
        units.append(
            ResourceUnitSpec(
                query_id=query_id,
                resource_unit_id=native_id,
                physical_node_id=f"node:{node_id}:native-fragment",
                unit_kind="native_fragment",
                backend="local_native",
                input_unit_ids=tuple(output_ids[parent] for parent in node["input_node_ids"]),
                per_task=ResourceVector(),
                target_output_block_bytes=0,
                generator_buffer_blocks=0,
                max_concurrency=None,
            )
        )
        if node["is_materialization_barrier"]:
            barriers.append(
                MaterializationBarrierSpec(
                    query_id=query_id,
                    barrier_id=materialization_barrier_id_for_node(query_id, node_id),
                    physical_node_id=node_id,
                    materializer_unit_id=native_id,
                    materialized_input_unit_ids=tuple(
                        output_ids[parent] for parent in node["materialized_input_node_ids"]
                    ),
                )
            )
        if node["udf_payload"] is not None:
            binding = bindings[node_id]
            payload = payloads[binding]
            if payload["execution_backend"] not in {"subprocess_task", "subprocess_actor"}:
                raise ValueError("local admission requires subprocess UDFs")
            spec = udf_resource_spec(
                query_id=query_id,
                resource_unit_id=output_ids[node_id],
                physical_node_id=f"node:{node_id}:udf",
                input_unit_ids=(native_id,),
                payload=payload,
                env=env,
            )
            units.append(spec)
            by_binding[binding] = spec.resource_unit_id
    if set(by_binding) != set(payloads):
        raise ValueError("local admission graph does not match the prepared UDF bindings")
    graph = QueryResourceGraph(
        query_id=query_id,
        plan_digest="sha256:pending",
        units=tuple(units),
        terminal_unit_ids=tuple(output_ids[node_id] for node_id in terminal_node_ids),
        materialization_barriers=tuple(barriers),
    )
    return replace(graph, plan_digest=graph.normalized_digest()), by_binding


@dataclass
class _OutputWait:
    request: OutputBlockRequest
    ready: threading.Event
    lease: OutputBlockLease | None = None


@dataclass(frozen=True)
class LocalQueryAdmissionBinding:
    query: LocalQueryAdmission
    unit_id: str


class LocalQueryAdmission:
    """Drive the shared nonblocking policy on local capacity/lifetime events."""

    def __init__(
        self,
        graph: QueryResourceGraph,
        allocation: QueryAllocation,
        *,
        coordinator: LocalQueryCoordinator | None = None,
        gpu_devices: tuple[str, ...] = (),
    ) -> None:
        self.owner_pid = os.getpid()
        self.graph = graph
        self.gpu_devices = frozenset(gpu_devices)
        if allocation.resources.gpu != len(self.gpu_devices):
            raise ValueError("local GPU capacity must match its device inventory")
        self._lock = threading.RLock()
        self._pump_lock = threading.Lock()
        self._event = threading.Event()
        self._closed = False
        self._error: BaseException | None = None
        self._authorities: set[LocalQueryAdmissionAuthority] = set()
        self._pending: dict[tuple[str, str], LocalQueryAdmissionAuthority] = {}
        self._outputs: dict[str, _OutputWait] = {}
        self._live_tasks: dict[str, str] = {}
        self._producers: dict[str, list[LocalQueryAdmissionAuthority]] = {}
        self._completed_units: set[str] = set()
        self._published_actors: set[str] = set()
        self._coordinator = coordinator
        self.manager = QueryResourceManager(
            graph,
            allocation,
            reservation_ratio=float(os.environ.get("VANE_QUERY_RESOURCE_RESERVATION_RATIO", "0.5")),
            on_change=self._event.set,
            on_eligible_units_change=self._reopen_frontier,
        )
        for unit in graph.units:
            self.manager.update_unit_state(unit.resource_unit_id, runnable=True)
        self._sync_actor_slots()
        self.manager.seal_native_fragment_production()
        self.manager.set_external_consumer_waiting(True)
        if coordinator is not None:
            coordinator.register(self, allocation.resources)
        self._thread = threading.Thread(target=self._run, name="vane-local-query-admission", daemon=True)
        try:
            self._thread.start()
        except BaseException:
            if coordinator is not None:
                coordinator.remove(self)
            raise

    def _reopen_frontier(self, _eligible: tuple[str, ...], _epoch: int) -> None:
        # The manager invokes this under its lock. Never enter adapter locks
        # or backend callbacks here; the admission thread reopens the fence.
        self._event.set()

    def _prepare_unit(self, unit_id: str) -> None:
        # A request beyond a blocking native operator is positive evidence
        # that DuckDB has completed that materializer. Native execution owns
        # this ordering; no speculative phase advancement is made at prepare.
        upstream: set[str] = set()
        pending = list(self.graph.unit_by_id(unit_id).input_unit_ids)
        while pending:
            key = pending.pop()
            if key not in upstream:
                upstream.add(key)
                pending.extend(self.graph.unit_by_id(key).input_unit_ids)
        for barrier in self.graph.materialization_barriers:
            if barrier.materializer_unit_id in upstream:
                self.manager.mark_materialization_barrier_completed_for_node(barrier.physical_node_id)

    def create_authority(
        self, unit_id: str, base: AdmissionAuthority, physical: LocalSlotAdmissionAuthority
    ) -> LocalQueryAdmissionAuthority:
        with self._lock:
            self._raise_error_locked()
            if unit_id in self._completed_units:
                raise RuntimeError("local UDF production already completed")
            authority = LocalQueryAdmissionAuthority(self, unit_id, base, physical)
            self._authorities.add(authority)
            self._producers.setdefault(unit_id, []).append(authority)
        base.register_wakeup(authority._backend_wakeup)
        return authority

    def _raise_error_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError("local query admission failed") from self._error
        if self._closed:
            raise RuntimeError("local query admission is closed")

    def _run(self) -> None:
        while True:
            self._event.wait(timeout=self._coordinator.refresh_interval_s if self._coordinator is not None else 1.0)
            self._event.clear()
            with self._lock:
                if self._closed:
                    return
            try:
                self._drive()
            except BaseException as exc:
                with self._lock:
                    self._error = exc
                    authorities = tuple(self._authorities)
                    outputs = tuple(self._outputs.values())
                for output in outputs:
                    output.ready.set()
                for authority in authorities:
                    try:
                        authority._notify()
                    except BaseException:
                        # Preserve the admission error and wake every peer.
                        pass
                return

    def _drive(self) -> None:
        with self._pump_lock:
            if self._coordinator is not None:
                notifications = self._coordinator.drive(self)
            else:
                self._sync_actor_slots()
                notifications = self._pump()
        # Native/user callbacks may immediately submit more work. Invoke them
        # after both the per-query pump and shared coordinator locks are gone.
        error = None
        for authority in notifications:
            try:
                authority._notify()
            except BaseException as exc:
                error = error or exc
        if error is not None:
            raise error

    def _sync_actor_slots(self) -> None:
        with self._lock:
            if self._closed:
                return
            eligible = set(self.manager.current_eligible_resource_unit_ids())
            for unit_id in tuple(self._published_actors - eligible):
                self.manager.set_submitted_actor_slots(unit_id, set())
                self._published_actors.remove(unit_id)
            for unit_id in eligible - self._published_actors:
                unit = self.graph.unit_by_id(unit_id)
                if unit.execution_kind == "actor":
                    self.manager.set_submitted_actor_slots(unit_id, set(range(unit.actor_pool_size)))
                    self.manager.set_ready_actor_slots(unit_id, {i: "local" for i in range(unit.actor_pool_size)})
                    self._published_actors.add(unit_id)

    def _pump(self) -> set[LocalQueryAdmissionAuthority]:
        # Policy callbacks only set the event. User/native wakeups and backend
        # acquisition run outside the policy manager's lock.
        notifications: set[LocalQueryAdmissionAuthority] = set()
        while True:
            changed = False
            frontier = self.manager.pending_allocation_frontier()
            if frontier is not None and self._coordinator is None:
                allocation = self.manager.allocation
                self.manager.update_allocation(
                    replace(allocation, generation=allocation.generation + 1),
                    reopen_fence_epoch=frontier[1],
                )
            with self._lock:
                if self._closed:
                    return notifications
                outputs = set(self._outputs)
            if outputs:
                output_request, output_grant = self.manager.try_acquire_next_queued_output_block(outputs)
                if output_request is not None and output_grant is not None:
                    if output_grant.fatal:
                        raise RuntimeError(f"local output admission failed: {output_grant.blocked_reason}")
                    if output_grant.granted:
                        assert output_grant.lease is not None
                        with self._lock:
                            waiter = self._outputs.get(output_request.block_id)
                            if waiter is None:
                                self.manager.release_output_block(output_grant.lease.lease_id)
                            else:
                                waiter.lease = output_grant.lease
                                waiter.ready.set()
                        changed = True
            with self._lock:
                candidates = set(self._pending)
                actor_pools = {
                    authority.unit_id: authority.physical._pool
                    for authority in self._pending.values()
                    if isinstance(authority.physical._pool, LocalActorExecutionSlotPool)
                }
            for unit_id, pool in actor_pools.items():
                self.manager.set_actor_slot_loads(unit_id, pool.actor_loads())
            if candidates:
                request, grant = self.manager.try_acquire_next_queued_task(candidates)
                if request is not None and grant is not None:
                    if grant.fatal:
                        raise RuntimeError(f"local task admission failed: {grant.blocked_reason}")
                    if grant.granted:
                        assert grant.lease is not None
                        with self._lock:
                            authority = self._pending.pop((request.task_id, request.attempt_id), None)
                            if authority is None or authority._state == "closed":
                                self.manager.abandon_task_lease(grant.lease.lease_id, attempt_id=grant.lease.attempt_id)
                            else:
                                authority._policy_lease = grant.lease
                        changed = True
            with self._lock:
                authorities = tuple(self._authorities)
            for authority in authorities:
                prepared = authority._prepare_backend()
                changed = prepared or changed
                if prepared or authority._backend_changed.is_set():
                    authority._backend_changed.clear()
                    notifications.add(authority)
            if not changed:
                return notifications

    def own_output(
        self,
        task: TaskLease,
        size_bytes: int,
        scope: ExecutionCancellationScope,
        admission: AdmissionLease,
    ) -> OutputBlockLeaseOwner:
        request = OutputBlockRequest(
            query_id=self.graph.query_id,
            producer_unit_id=task.resource_unit_id,
            task_lease_id=task.lease_id,
            attempt_id=task.attempt_id,
            block_id=uuid.uuid4().hex,
            size_bytes=size_bytes,
        )
        waiter = _OutputWait(request, threading.Event())
        unregister = scope.register_cancel_wakeup(waiter.ready.set)
        try:
            with self._lock:
                self._raise_error_locked()
                scope.raise_if_cancelled("local query output admission")
                grant = self.manager.try_acquire_output_block(request)
                if grant.fatal:
                    raise RuntimeError(f"local output admission failed: {grant.blocked_reason}")
                if grant.granted:
                    assert grant.lease is not None
                    return OutputBlockLeaseOwner(self.manager, grant.lease)
                self._outputs[request.block_id] = waiter
                error = self.manager.note_output_waiting(request)
                if error is not None:
                    raise RuntimeError(f"local output identity failed: {error}")
            self._event.set()
            with admission.suspend_for_wait(scope):
                waiter.ready.wait()
            scope.raise_if_cancelled("local query output admission")
            with self._lock:
                self._raise_error_locked()
                assert waiter.lease is not None
                owner = OutputBlockLeaseOwner(self.manager, waiter.lease)
                waiter.lease = None
                return owner
        finally:
            unregister()
            with self._lock:
                self._outputs.pop(request.block_id, None)
                self.manager.remove_output_waiter(request.block_id)
                if waiter.lease is not None:
                    self.manager.release_output_block(waiter.lease.lease_id)

    def finish_task(self, lease: TaskLease) -> None:
        with self._lock:
            if lease.lease_id not in self._live_tasks:
                return
            self._live_tasks.pop(lease.lease_id)
            self.manager.release_task_lease(lease.lease_id, attempt_id=lease.attempt_id)
            self._finish_unit_locked(lease.resource_unit_id)

    def _finish_unit_locked(self, unit_id: str) -> None:
        if unit_id in self._completed_units or unit_id in self._live_tasks.values():
            return
        producers = self._producers.get(unit_id, ())
        if producers and all(authority._finished_submitting for authority in producers):
            self._completed_units.add(unit_id)
            self.manager.update_unit_state(unit_id, runnable=False, completed=True)

    def snapshot(self) -> dict[str, Any]:
        return self.manager.snapshot()

    def shutdown(self, *, kill: bool = False) -> None:
        if self.owner_pid != os.getpid():
            return
        with self._lock:
            self._closed = True
            authorities = tuple(self._authorities)
            outputs = tuple(self._outputs.values())
            self.manager.close_admission()
        self._event.set()
        for output in outputs:
            output.ready.set()
        cleanup_error = None
        for authority in authorities:
            try:
                authority.close()
            except BaseException as error:
                cleanup_error = error
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise TimeoutError("local query admission thread still owns cleanup")
        if cleanup_error is not None:
            raise cleanup_error
        if self.cleanup_pending():
            raise RuntimeError("local query admission still owns active executions")
        if self._coordinator is not None:
            self._coordinator.remove(self)

    def cleanup_pending(self) -> bool:
        if self.owner_pid != os.getpid():
            return False
        with self._lock:
            return bool(self._authorities or self._live_tasks or self._outputs)


class LocalQueryAdmissionAuthority:
    """One native dispatcher's request, through policy then physical capacity."""

    def __init__(
        self,
        query: LocalQueryAdmission,
        unit_id: str,
        base: AdmissionAuthority,
        physical: LocalSlotAdmissionAuthority,
    ) -> None:
        self.query = query
        self.unit_id = unit_id
        self.base = base
        self.physical = physical
        self._state = "idle"
        self._request: TaskRequest | None = None
        self._policy_lease: TaskLease | None = None
        self._backend_requested = False
        self._backend_changed = threading.Event()
        self._wakeup: Callable[[], None] | None = None
        self._base_closed = False
        self._finished_submitting = False

    def register_wakeup(self, callback: Callable[[], None] | None) -> None:
        with self.query._lock:
            self._wakeup = callback

    def _backend_wakeup(self) -> None:
        # Backend locks may be held here. Defer native callbacks and query-lock
        # acquisition, including byte-timeout and partial-input-flush signals.
        self._backend_changed.set()
        self.query._event.set()

    def _notify(self) -> None:
        with self.query._lock:
            callback = self._wakeup
        if callback is not None:
            callback()

    def request(self, retained_input_bytes: int) -> bool:
        retained = int(retained_input_bytes)
        if retained < 0:
            raise ValueError("retained_input_bytes must be >= 0")
        with self.query._lock:
            self.query._raise_error_locked()
            if self._state != "idle":
                return False
            request = TaskRequest(
                query_id=self.query.graph.query_id,
                resource_unit_id=self.unit_id,
                task_id=uuid.uuid4().hex,
                attempt_id="0",
                node_id=None,
                retained_input_bytes=retained,
            )
            self.query._prepare_unit(self.unit_id)
            self.query.manager.note_task_waiting(request)
            self.query._pending[(request.task_id, request.attempt_id)] = self
            self._request = request
            self._state = "requested"
        self.query._event.set()
        self.query._drive()
        return True

    def _prepare_backend(self) -> bool:
        with self.query._lock:
            if self._state != "requested" or self._policy_lease is None:
                return False
            lease = self._policy_lease
            assert self._request is not None
            retained = int(self._request.retained_input_bytes or 0)
            submit = not self._backend_requested
            if submit:
                self._backend_requested = True
                if lease.actor_index is not None:
                    self.physical.select_actor(lease.actor_index)
                self.base.request(retained)
            state = self.base.state()
            if not state["available"]:
                return submit
            self._state = "ready"
        return True

    def state(self) -> dict[str, Any]:
        with self.query._lock:
            if self._state != "closed":
                self.query._raise_error_locked()
            backend = self.base.state()
            state = self._state
            if state == "requested" and self._backend_requested:
                state = str(backend["state"])
                if state == "ready":
                    state = "requested"  # The pump owns the ready handoff.
            return {
                **backend,
                "state": state,
                "available": self._state == "ready",
                "retained_input_bytes": 0 if self._request is None else self._request.retained_input_bytes,
            }

    def diagnostic_state(self) -> str:
        with self.query._lock:
            if self._state == "requested" and self._backend_requested:
                observe = getattr(self.base, "diagnostic_state", None)
                if callable(observe):
                    return str(observe())
            return self._state

    def take(self, retained_input_bytes: int) -> AdmissionLease:
        with self.query._lock:
            self.query._raise_error_locked()
            if self._state != "ready" or self._policy_lease is None or self._request is None:
                raise RuntimeError("local query admission is not ready")
            if self._request.retained_input_bytes != int(retained_input_bytes):
                raise RuntimeError("local query retained input changed after admission")
            policy = self._policy_lease
            base = self.base.take(retained_input_bytes)
            self.query._live_tasks[policy.lease_id] = self.unit_id
            self._policy_lease = None
            self._request = None
            self._backend_requested = False
            self._state = "idle"

        def complete() -> None:
            try:
                base.complete_execution()
            finally:
                self.query.finish_task(policy)

        def release() -> None:
            try:
                base.release()
            finally:
                self.query.finish_task(policy)

        return AdmissionLease(
            request_id=base.request_id,
            retained_input_bytes=base.retained_input_bytes,
            lease={
                **base.lease,
                "query_policy": self.query,
                "query_policy_task": policy,
                "actor_index": policy.actor_index,
            },
            _execution_finished_callback=complete,
            _release_callback=release,
            _capacity_wait_context=base.suspend_for_wait,
        )

    def finished_submitting(self) -> None:
        self.close()
        with self.query._lock:
            self._finished_submitting = True
            self.query._finish_unit_locked(self.unit_id)

    def close(self) -> None:
        if self.query.owner_pid != os.getpid():
            return
        with self.query._lock:
            if self._base_closed:
                return
            self._state = "closed"
            request, self._request = self._request, None
            if request is not None:
                self.query._pending.pop((request.task_id, request.attempt_id), None)
                self.query.manager.remove_task_waiter(request.task_id, request.attempt_id)
            # Keep a granted policy lease and this cleanup owner until the
            # backend has successfully returned its ready reservation.
            self.base.close()
            self._base_closed = True
            policy, self._policy_lease = self._policy_lease, None
            self.query._authorities.discard(self)
            if policy is not None:
                self.query.manager.abandon_task_lease(policy.lease_id, attempt_id=policy.attempt_id)
        self._notify()
