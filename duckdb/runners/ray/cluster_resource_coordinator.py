# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import math
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from duckdb.runners.ray.query_resource_graph import (
    ActorPlacement,
    NodeResourceAllocation,
    QueryAllocation,
    ResourceVector,
)
from duckdb.runners.ray.worker_memory import build_ray_node_memory_layout

_EPSILON = 1e-9


class ElasticGpuDemandError(ValueError):
    code = "ELASTIC_GPU_UNSUPPORTED"

    def __init__(
        self,
        query_id: str,
        minimum_gpu: float,
        desired_gpu: float,
    ) -> None:
        self.query_id = str(query_id)
        self.minimum_gpu = float(minimum_gpu)
        self.desired_gpu = float(desired_gpu)
        super().__init__(self.query_id, self.minimum_gpu, self.desired_gpu)

    def __str__(self) -> str:
        return (
            f"query {self.query_id} GPU resources cannot be elastic: "
            f"minimum={self.minimum_gpu:g}, desired={self.desired_gpu:g}"
        )

    def to_dict(self) -> dict[str, str | float]:
        return {
            "code": self.code,
            "query_id": self.query_id,
            "resource": "gpu",
            "minimum": self.minimum_gpu,
            "desired": self.desired_gpu,
        }


def _sum_resources(resources: Sequence[ResourceVector]) -> ResourceVector:
    total = ResourceVector()
    for item in resources:
        total = total + item
    return total


def _replace_resource(vector: ResourceVector, field_name: str, value: float) -> ResourceVector:
    payload = vector.to_dict()
    payload[field_name] = value
    return ResourceVector.from_dict(payload)


def _positive_difference(left: ResourceVector, right: ResourceVector) -> ResourceVector:
    return ResourceVector(
        cpu=max(0.0, left.cpu - right.cpu),
        gpu=max(0.0, left.gpu - right.gpu),
        heap_bytes=max(0, left.heap_bytes - right.heap_bytes),
        object_store_bytes=max(0, left.object_store_bytes - right.object_store_bytes),
    )


def _hard_positive_difference(left: ResourceVector, right: ResourceVector) -> ResourceVector:
    """Return allocation debt for resources that cannot be relieved by spill."""
    return ResourceVector(
        cpu=max(0.0, left.cpu - right.cpu),
        gpu=max(0.0, left.gpu - right.gpu),
        heap_bytes=max(0, left.heap_bytes - right.heap_bytes),
    )


@dataclass(frozen=True)
class NodeCapacity:
    node_id: str
    resources: ResourceVector
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        node_id = str(self.node_id).strip()
        if not node_id:
            raise ValueError("node_id must be non-empty")
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "labels", tuple(sorted({str(item) for item in self.labels if str(item)})))

    def to_dict(self) -> dict[str, Any]:
        return {
            "resources": self.resources.to_dict(),
            "labels": list(self.labels),
        }


@dataclass(frozen=True)
class ActorResourceBundle:
    resource_unit_id: str
    actor_index: int
    resources: ResourceVector

    def __post_init__(self) -> None:
        resource_unit_id = str(self.resource_unit_id).strip()
        actor_index = int(self.actor_index)
        if not resource_unit_id:
            raise ValueError("actor resource bundle resource_unit_id must be non-empty")
        if actor_index < 0:
            raise ValueError("actor resource bundle actor_index must be >= 0")
        if self.resources.is_zero():
            raise ValueError("actor resource bundle must own non-zero resources")
        if self.resources.object_store_bytes != 0:
            raise ValueError("actor resource bundles may not hard-reserve object-store bytes")
        object.__setattr__(self, "resource_unit_id", resource_unit_id)
        object.__setattr__(self, "actor_index", actor_index)


@dataclass(frozen=True)
class QueryDemand:
    """Hard placement minimum plus elastic resource targets.

    CPU, GPU, and heap in ``minimum`` are placement requirements. The
    object-store components are soft flow-control budgets; spillable task
    input/output windows must not be copied into ``minimum`` or the placement
    bundles.
    """

    query_id: str
    minimum: ResourceVector
    desired: ResourceVector
    weight: float = 1.0
    priority: int = 0
    actor_bundles: tuple[ActorResourceBundle, ...] = ()
    task_bundles: tuple[ResourceVector, ...] = ()

    def __post_init__(self) -> None:
        query_id = str(self.query_id).strip()
        if not query_id:
            raise ValueError("query_id must be non-empty")
        if not self.minimum.fits_within(self.desired):
            exceeded = self.minimum.exceeded_dimensions(self.desired)
            raise ValueError(f"minimum query demand exceeds desired demand for {', '.join(exceeded)}")
        if not math.isclose(self.minimum.gpu, self.desired.gpu, rel_tol=0.0, abs_tol=_EPSILON):
            raise ElasticGpuDemandError(
                query_id=query_id,
                minimum_gpu=self.minimum.gpu,
                desired_gpu=self.desired.gpu,
            )
        weight = float(self.weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("weight must be finite and > 0")
        actor_bundles = tuple(self.actor_bundles)
        task_bundles = tuple(self.task_bundles)
        if self.minimum.object_store_bytes != 0:
            raise ValueError("minimum query resources may not hard-reserve object-store bytes")
        if any(bundle.object_store_bytes != 0 for bundle in task_bundles):
            raise ValueError("task resource bundles may not hard-reserve object-store bytes")
        actor_keys = [(bundle.resource_unit_id, bundle.actor_index) for bundle in actor_bundles]
        if len(set(actor_keys)) != len(actor_keys):
            raise ValueError("actor resource bundles contain duplicate resource-unit/index identities")
        actor_total = _sum_resources([bundle.resources for bundle in actor_bundles])
        task_total = _sum_resources(task_bundles)
        bundle_total = actor_total + task_total
        if bundle_total != self.minimum:
            raise ValueError("actor_bundles plus task_bundles must exactly equal minimum query resources")
        if not bundle_total.fits_within(self.desired):
            raise ValueError("minimum placement bundles exceed desired query resources")
        if actor_total.gpu > self.desired.gpu + _EPSILON:
            raise ValueError("actor bundles exceed desired GPU resources")
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "priority", int(self.priority))
        object.__setattr__(self, "actor_bundles", actor_bundles)
        object.__setattr__(self, "task_bundles", task_bundles)


@dataclass
class _QueryState:
    demand: QueryDemand
    sequence: int
    observed_usage: ResourceVector = field(default_factory=ResourceVector)
    allocation: QueryAllocation = field(
        default_factory=lambda: QueryAllocation(
            resources=ResourceVector(),
            node_allocations=(),
            actor_placements=(),
            generation=1,
        )
    )
    node_allocations: dict[str, ResourceVector] = field(default_factory=dict)
    actor_placements: tuple[ActorPlacement, ...] = ()
    allocation_debt: ResourceVector = field(default_factory=ResourceVector)
    state: str = "PENDING_RESOURCES"
    rejection_reason: str = ""
    actor_placement_lost: bool = False
    expires_at: float = 0.0


def read_ray_node_capacities(
    ray_module: Any,
    *,
    object_store_fraction: float = 0.5,
    heap_reserve_bytes_per_node: int = 0,
) -> tuple[NodeCapacity, ...]:
    """Read the only supported capacity source: live resources reported by Ray.

    This function is deliberately separate from coordinator locks. A slow GCS
    request can delay a refresh, but it cannot block query admission already
    operating on the last complete capacity snapshot.
    """
    fraction = float(object_store_fraction)
    if not math.isfinite(fraction) or fraction <= 0 or fraction > 1:
        raise ValueError("object_store_fraction must be in (0, 1]")
    heap_reserve = int(heap_reserve_bytes_per_node)
    if heap_reserve < 0:
        raise ValueError("heap_reserve_bytes_per_node must be >= 0")

    try:
        raw_nodes = ray_module.nodes()
    except Exception as exc:
        raise RuntimeError(f"failed to read Ray node capacity: {exc}") from exc

    capacities: list[NodeCapacity] = []
    for raw_node in raw_nodes:
        if not bool(raw_node.get("Alive", True)):
            continue
        resources = dict(raw_node.get("Resources") or {})
        cpu = max(0.0, float(resources.get("CPU", 0) or 0))
        gpu = max(0.0, float(resources.get("GPU", 0) or 0))
        if cpu <= 0 and gpu <= 0:
            continue
        node_id = str(raw_node.get("NodeID") or raw_node.get("NodeManagerAddress") or "").strip()
        if not node_id:
            raise ValueError("alive Ray node with schedulable resources is missing NodeID")
        ray_heap = max(0, int(float(resources.get("memory", 0) or 0)))
        ray_store = max(0, int(float(resources.get("object_store_memory", 0) or 0)))
        memory_layout = build_ray_node_memory_layout(ray_heap)
        labels = [str(key) for key, value in resources.items() if str(key).startswith("node:") and float(value) > 0]
        labels.extend(f"{key}={value}" for key, value in sorted(dict(raw_node.get("Labels") or {}).items()))
        capacities.append(
            NodeCapacity(
                node_id=node_id,
                resources=ResourceVector(
                    cpu=cpu,
                    gpu=gpu,
                    heap_bytes=max(0, memory_layout.task_heap_capacity_bytes - heap_reserve),
                    object_store_bytes=math.floor(ray_store * fraction),
                ),
                labels=tuple(labels),
            )
        )
    return tuple(sorted(capacities, key=lambda item: item.node_id))


class ClusterQueryResourceCoordinator:
    """Deterministic cluster-to-query allocation authority.

    Ray I/O is intentionally absent from this class. Callers refresh a complete
    immutable node-capacity snapshot, then all allocation and lease bookkeeping
    happens under one local lock.
    """

    def __init__(
        self,
        node_capacities: Sequence[NodeCapacity],
        *,
        heartbeat_timeout_s: float = 30.0,
    ) -> None:
        timeout = float(heartbeat_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("heartbeat_timeout_s must be finite and > 0")
        self._lock = threading.RLock()
        self._heartbeat_timeout_s = timeout
        self._nodes = self._normalize_nodes(node_capacities)
        self._queries: dict[str, _QueryState] = {}
        self._next_sequence = 0
        self._generation = 0

    @staticmethod
    def _normalize_nodes(node_capacities: Sequence[NodeCapacity]) -> dict[str, NodeCapacity]:
        nodes: dict[str, NodeCapacity] = {}
        for node in node_capacities:
            if node.node_id in nodes:
                raise ValueError(f"duplicate Ray node capacity: {node.node_id}")
            nodes[node.node_id] = node
        return dict(sorted(nodes.items()))

    def register_query(self, demand: QueryDemand, *, now: float | None = None) -> QueryAllocation:
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            if demand.query_id in self._queries:
                raise ValueError(f"query already registered: {demand.query_id}")
            previous_queries = self._queries
            previous_next_sequence = self._next_sequence
            previous_generation = self._generation
            state = _QueryState(
                demand=demand,
                sequence=previous_next_sequence,
                expires_at=timestamp + self._heartbeat_timeout_s,
            )
            staged_queries = copy.deepcopy(previous_queries)
            staged_queries[demand.query_id] = state
            try:
                self._queries = staged_queries
                self._next_sequence = previous_next_sequence + 1
                self._rebalance_locked()
            except BaseException:
                # Rebalancing updates every query allocation in place. Restore
                # the exact pre-registration objects and counters after any
                # failure so no partial allocation, heartbeat, or lease state
                # can escape.
                self._queries = previous_queries
                self._next_sequence = previous_next_sequence
                self._generation = previous_generation
                raise
            return state.allocation

    def refresh_query(
        self,
        query_id: str,
        *,
        observed_usage: ResourceVector,
        generation: int,
        now: float | None = None,
        demand: QueryDemand | None = None,
    ) -> QueryAllocation:
        query_key = str(query_id)
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            state = self._queries.get(query_key)
            if state is None:
                raise KeyError(f"query is not registered: {query_key}")
            self._require_generation(state, generation)
            if demand is not None:
                if demand.query_id != query_key:
                    raise ValueError("refresh demand query_id mismatch")
                state.demand = demand
            state.observed_usage = observed_usage
            state.expires_at = timestamp + self._heartbeat_timeout_s
            self._rebalance_locked()
            return state.allocation

    def refresh_queries(
        self,
        *,
        observed_usage_by_query: Mapping[str, ResourceVector],
        generations: Mapping[str, int],
        now: float | None = None,
    ) -> dict[str, QueryAllocation]:
        """Atomically refresh every live query from one coordinator snapshot.

        A multi-query driver must not refresh queries one by one because each
        rebalance advances every allocation generation.  Validating the full
        batch before mutation also prevents a stale query from partially
        extending other heartbeat deadlines.
        """

        timestamp = time.monotonic() if now is None else float(now)
        usage = {str(query_id): value for query_id, value in observed_usage_by_query.items()}
        generation_by_query = {str(query_id): int(generation) for query_id, generation in generations.items()}
        if set(usage) != set(generation_by_query):
            raise ValueError("refresh query usage and generation sets must match")
        with self._lock:
            expected = set(self._queries)
            if set(usage) != expected:
                missing = sorted(expected - set(usage))
                unknown = sorted(set(usage) - expected)
                details = []
                if missing:
                    details.append("missing=" + ",".join(missing))
                if unknown:
                    details.append("unknown=" + ",".join(unknown))
                raise ValueError("refresh query set mismatch: " + " ".join(details))
            for query_id in sorted(expected):
                state = self._queries[query_id]
                self._require_generation(state, generation_by_query[query_id])
                if not isinstance(usage[query_id], ResourceVector):
                    raise TypeError(f"observed usage for query {query_id} must be ResourceVector")
            for query_id in sorted(expected):
                state = self._queries[query_id]
                state.observed_usage = usage[query_id]
                state.expires_at = timestamp + self._heartbeat_timeout_s
            if expected:
                self._rebalance_locked()
            return {query_id: self._queries[query_id].allocation for query_id in sorted(expected)}

    def heartbeat(self, query_id: str, generation: int, *, now: float | None = None) -> QueryAllocation:
        query_key = str(query_id)
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            state = self._queries.get(query_key)
            if state is None:
                raise KeyError(f"query is not registered: {query_key}")
            self._require_generation(state, generation)
            state.expires_at = timestamp + self._heartbeat_timeout_s
            return state.allocation

    def release_query(self, query_id: str, generation: int) -> bool:
        query_key = str(query_id)
        with self._lock:
            state = self._queries.get(query_key)
            if state is None or int(generation) != state.allocation.generation:
                return False
            self._queries.pop(query_key, None)
            self._rebalance_locked()
            return True

    def expire_queries(self, *, now: float | None = None) -> tuple[str, ...]:
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            expired = tuple(
                sorted(query_id for query_id, state in self._queries.items() if state.expires_at <= timestamp)
            )
            for query_id in expired:
                self._queries.pop(query_id, None)
            if expired:
                self._rebalance_locked()
            return expired

    def update_node_capacities(
        self,
        node_capacities: Sequence[NodeCapacity],
        *,
        now: float | None = None,
    ) -> None:
        del now  # Capacity generations, not wall clock, order this update.
        normalized = self._normalize_nodes(node_capacities)
        with self._lock:
            self._nodes = normalized
            self._rebalance_locked()

    @staticmethod
    def _require_generation(state: _QueryState, generation: int) -> None:
        if int(generation) != state.allocation.generation:
            raise ValueError(
                f"stale allocation generation: got {int(generation)}, current {state.allocation.generation}"
            )

    def _rebalance_locked(self) -> None:
        self._generation += 1
        generation = self._generation
        remaining = {node_id: node.resources for node_id, node in self._nodes.items()}
        node_allocations_by_query: dict[str, dict[str, ResourceVector]] = {query_id: {} for query_id in self._queries}
        actor_placements_by_query: dict[str, tuple[ActorPlacement, ...]] = {query_id: () for query_id in self._queries}
        admitted: list[_QueryState] = []

        # Eager actor placements become non-preemptible as soon as admission
        # publishes them.  The driver creates actors immediately after this
        # allocation, so moving the placement during the startup gap would make
        # the virtual lease disagree with Ray's hard NodeAffinity placement.
        pinned_query_ids: set[str] = set()
        lost_actor_placement_query_ids: set[str] = set()
        for state in sorted(self._queries.values(), key=lambda item: (item.sequence, item.demand.query_id)):
            if state.actor_placement_lost:
                lost_actor_placement_query_ids.add(state.demand.query_id)
                continue
            if not state.actor_placements:
                continue
            trial_remaining = dict(remaining)
            placement_valid = True
            actor_bundles = {
                (bundle.resource_unit_id, bundle.actor_index): bundle.resources for bundle in state.demand.actor_bundles
            }
            pinned_allocations: dict[str, ResourceVector] = {}
            for placement in state.actor_placements:
                vector = actor_bundles.get((placement.resource_unit_id, placement.actor_index))
                capacity = trial_remaining.get(placement.node_id)
                if vector is None or capacity is None or not vector.fits_within(capacity):
                    placement_valid = False
                    break
                trial_remaining[placement.node_id] = capacity - vector
                pinned_allocations[placement.node_id] = (
                    pinned_allocations.get(placement.node_id, ResourceVector()) + vector
                )
            if not placement_valid:
                state.actor_placement_lost = True
                lost_actor_placement_query_ids.add(state.demand.query_id)
                continue
            query_id = state.demand.query_id
            remaining = trial_remaining
            pinned_query_ids.add(query_id)
            actor_placements_by_query[query_id] = state.actor_placements
            node_allocations_by_query[query_id] = pinned_allocations

        ordered = sorted(
            self._queries.values(),
            key=lambda state: (-state.demand.priority, state.sequence, state.demand.query_id),
        )
        for state in ordered:
            query_id = state.demand.query_id
            if query_id in lost_actor_placement_query_ids:
                continue
            trial_remaining = dict(remaining)
            trial_allocations: dict[str, ResourceVector] = dict(node_allocations_by_query[query_id])
            if query_id not in pinned_query_ids:
                actor_ok = True
                new_actor_placements: list[ActorPlacement] = []
                for actor_bundle in state.demand.actor_bundles:
                    node_id = self._place_bundle(actor_bundle.resources, trial_remaining)
                    if node_id is None:
                        actor_ok = False
                        break
                    new_actor_placements.append(
                        ActorPlacement(
                            resource_unit_id=actor_bundle.resource_unit_id,
                            actor_index=actor_bundle.actor_index,
                            node_id=node_id,
                        )
                    )
                    trial_allocations[node_id] = (
                        trial_allocations.get(node_id, ResourceVector()) + actor_bundle.resources
                    )
                if not actor_ok:
                    continue
                actor_placements_by_query[query_id] = tuple(new_actor_placements)

            task_ok = True
            for task_bundle in state.demand.task_bundles:
                node_id = self._place_bundle(task_bundle, trial_remaining)
                if node_id is None:
                    task_ok = False
                    break
                trial_allocations[node_id] = trial_allocations.get(node_id, ResourceVector()) + task_bundle
            if not task_ok:
                continue

            remaining = trial_remaining
            node_allocations_by_query[query_id] = trial_allocations
            admitted.append(state)

        total_capacity = _sum_resources([node.resources for node in self._nodes.values()])
        extra_capacity = _sum_resources(list(remaining.values()))
        extras = self._weighted_drf_extras(admitted, extra_capacity, total_capacity)

        # Place the aggregate DRF result onto concrete nodes. GPU demand is
        # fixed in the bundles above; only CPU and memory extras are divisible.
        for state in admitted:
            query_id = state.demand.query_id
            extra = extras.get(query_id, ResourceVector())
            placed = self._place_divisible(extra, remaining)
            if placed is None:
                # Aggregate feasibility should make this impossible because the
                # divisible dimensions may be split independently across nodes.
                raise RuntimeError(f"failed to place feasible DRF allocation for query {query_id}")
            for node_id, vector in placed.items():
                allocations = node_allocations_by_query[query_id]
                allocations[node_id] = allocations.get(node_id, ResourceVector()) + vector

        admitted_ids = {state.demand.query_id for state in admitted}
        for query_id, state in self._queries.items():
            if query_id in admitted_ids:
                state.state = "RUNNING"
                state.rejection_reason = ""
            elif query_id in lost_actor_placement_query_ids:
                state.state = "ACTOR_PLACEMENT_LOST"
                state.rejection_reason = "allocated Ray actor node is no longer available"
            else:
                state.state = "PENDING_RESOURCES"
                state.rejection_reason = "minimum query resource bundles are not currently feasible"
            resources = _sum_resources(list(node_allocations_by_query[query_id].values()))
            state.allocation = QueryAllocation(
                resources=resources,
                node_allocations=tuple(
                    NodeResourceAllocation(node_id=node_id, resources=vector)
                    for node_id, vector in sorted(node_allocations_by_query[query_id].items())
                    if not vector.is_zero()
                ),
                actor_placements=actor_placements_by_query[query_id],
                generation=generation,
            )
            state.node_allocations = node_allocations_by_query[query_id]
            state.actor_placements = actor_placements_by_query[query_id]
            state.allocation_debt = _hard_positive_difference(state.observed_usage, resources)
            if not state.allocation_debt.is_zero() and state.state != "ACTOR_PLACEMENT_LOST":
                state.state = "ALLOCATION_DEBT"
                state.rejection_reason = "live query hard-resource leases exceed the current allocation"

    @staticmethod
    def _place_bundle(bundle: ResourceVector, remaining: dict[str, ResourceVector]) -> str | None:
        candidates = [node_id for node_id, capacity in remaining.items() if bundle.fits_within(capacity)]
        if not candidates:
            return None
        node_id = min(
            candidates,
            key=lambda candidate: (
                remaining[candidate].gpu - bundle.gpu,
                remaining[candidate].cpu - bundle.cpu,
                remaining[candidate].heap_bytes - bundle.heap_bytes,
                candidate,
            ),
        )
        remaining[node_id] = remaining[node_id] - bundle
        return node_id

    @staticmethod
    def _place_divisible(
        request: ResourceVector,
        remaining: dict[str, ResourceVector],
    ) -> dict[str, ResourceVector] | None:
        if request.gpu > _EPSILON:
            raise AssertionError("divisible placement received fixed GPU resources")
        trial_remaining = dict(remaining)
        allocations = {node_id: ResourceVector() for node_id in remaining}
        for field_name in ("cpu", "heap_bytes", "object_store_bytes"):
            needed = float(getattr(request, field_name))
            if needed <= _EPSILON:
                continue
            candidates = sorted(
                trial_remaining,
                key=lambda node_id: (-float(getattr(trial_remaining[node_id], field_name)), node_id),
            )
            for node_id in candidates:
                available = float(getattr(trial_remaining[node_id], field_name))
                if available <= _EPSILON:
                    continue
                amount = min(needed, available)
                if field_name != "cpu":
                    amount = int(amount)
                if amount <= 0:
                    continue
                allocations[node_id] = _replace_resource(
                    allocations[node_id],
                    field_name,
                    getattr(allocations[node_id], field_name) + amount,
                )
                trial_remaining[node_id] = _replace_resource(
                    trial_remaining[node_id],
                    field_name,
                    getattr(trial_remaining[node_id], field_name) - amount,
                )
                needed -= amount
                if needed <= _EPSILON:
                    break
            if needed > _EPSILON:
                return None
        remaining.clear()
        remaining.update(trial_remaining)
        return {node_id: vector for node_id, vector in allocations.items() if not vector.is_zero()}

    @staticmethod
    def _weighted_drf_extras(
        admitted: Sequence[_QueryState],
        extra_capacity: ResourceVector,
        total_capacity: ResourceVector,
    ) -> dict[str, ResourceVector]:
        if not admitted:
            return {}

        # Object-store bytes are a soft flow-control budget. Allocate them
        # independently so saturation in a hard DRF dimension (for example,
        # all CPU consumed by query minima) cannot give one admitted query the
        # entire store tail while another receives zero.
        hard_extra_capacity = ResourceVector(
            cpu=extra_capacity.cpu,
            gpu=extra_capacity.gpu,
            heap_bytes=extra_capacity.heap_bytes,
        )
        hard_total_capacity = ResourceVector(
            cpu=total_capacity.cpu,
            gpu=total_capacity.gpu,
            heap_bytes=total_capacity.heap_bytes,
        )
        # Water-fill each query's final weighted dominant share. Hard minima
        # are already allocated, so fairness must start from those shares
        # instead of treating every query's remaining headroom as a fresh zero.
        states_by_id = {state.demand.query_id: state for state in admitted}
        minimum: dict[str, ResourceVector] = {
            state.demand.query_id: ResourceVector(
                cpu=state.demand.minimum.cpu,
                gpu=state.demand.minimum.gpu,
                heap_bytes=state.demand.minimum.heap_bytes,
            )
            for state in admitted
        }
        desired: dict[str, ResourceVector] = {
            state.demand.query_id: ResourceVector(
                cpu=state.demand.desired.cpu,
                gpu=state.demand.desired.gpu,
                heap_bytes=state.demand.desired.heap_bytes,
            )
            for state in admitted
        }
        headroom = {query_id: desired[query_id] - minimum[query_id] for query_id in states_by_id}
        minimum_share = {query_id: minimum[query_id].dominant_share(hard_total_capacity) for query_id in states_by_id}
        desired_share = {query_id: desired[query_id].dominant_share(hard_total_capacity) for query_id in states_by_id}
        finite_limits = [
            desired_share[query_id] / states_by_id[query_id].demand.weight
            for query_id in headroom
            if not headroom[query_id].is_zero() and math.isfinite(desired_share[query_id])
        ]

        def minimum_factor_at(query_id: str, level: float) -> float:
            """Return the least headroom fraction that reaches a weighted share."""
            room = headroom[query_id]
            if room.is_zero():
                return 0.0
            weight = states_by_id[query_id].demand.weight
            target_share = level * weight
            if minimum_share[query_id] >= target_share - _EPSILON:
                return 0.0
            factors: list[float] = []
            for field_name in ("cpu", "gpu", "heap_bytes"):
                room_amount = float(getattr(room, field_name))
                if room_amount <= 0:
                    continue
                capacity = float(getattr(hard_total_capacity, field_name))
                if capacity <= 0:
                    return 0.0
                base_amount = float(getattr(minimum[query_id], field_name))
                factors.append((target_share * capacity - base_amount) / room_amount)
            return min(1.0, max(0.0, min(factors, default=0.0)))

        def allocation_at(level: float) -> dict[str, ResourceVector]:
            return {query_id: room.scale(minimum_factor_at(query_id, level)) for query_id, room in headroom.items()}

        def feasible(level: float) -> bool:
            total = _sum_resources(list(allocation_at(level).values()))
            return total.fits_within(hard_extra_capacity)

        if finite_limits:
            low = 0.0
            high = max(finite_limits)
            if feasible(high):
                low = high
            else:
                for _ in range(80):
                    middle = (low + high) / 2.0
                    if feasible(middle):
                        low = middle
                    else:
                        high = middle
            allocated = allocation_at(low)

            # A query can have headroom in a non-dominant dimension. Fill as
            # much of that plateau as capacity permits without raising its
            # weighted dominant share above the water level. This preserves
            # max-min fairness while avoiding stranded divisible capacity.
            remaining = _positive_difference(
                hard_extra_capacity,
                _sum_resources(list(allocated.values())),
            )
            for state in sorted(
                admitted,
                key=lambda item: (-item.demand.priority, item.sequence, item.demand.query_id),
            ):
                query_id = state.demand.query_id
                unit_headroom = headroom[query_id]
                if unit_headroom.is_zero():
                    continue
                current_factor = minimum_factor_at(query_id, low)
                share_ceiling = max(minimum_share[query_id], low * state.demand.weight)
                upper_factors: list[float] = []
                for field_name in ("cpu", "gpu", "heap_bytes"):
                    room_amount = float(getattr(unit_headroom, field_name))
                    if room_amount <= 0:
                        continue
                    capacity = float(getattr(hard_total_capacity, field_name))
                    if capacity <= 0:
                        upper_factors.append(0.0)
                        continue
                    base_amount = float(getattr(minimum[query_id], field_name))
                    upper_factors.append((share_ceiling * capacity - base_amount) / room_amount)
                maximum_factor = min(1.0, max(0.0, min(upper_factors, default=0.0)))
                if maximum_factor <= current_factor + _EPSILON:
                    continue

                candidate = unit_headroom.scale(maximum_factor)
                delta = candidate - allocated[query_id]
                if not delta.fits_within(remaining):
                    lower = current_factor
                    upper = maximum_factor
                    for _ in range(80):
                        middle = (lower + upper) / 2.0
                        middle_delta = unit_headroom.scale(middle) - allocated[query_id]
                        if middle_delta.fits_within(remaining):
                            lower = middle
                        else:
                            upper = middle
                    candidate = unit_headroom.scale(lower)
                    delta = candidate - allocated[query_id]
                allocated[query_id] = candidate
                remaining = remaining - delta
        else:
            allocated = {state.demand.query_id: ResourceVector() for state in admitted}

        # Byte flooring can leave a small tail. Assign it deterministically to
        # the currently lowest weighted dominant share without exceeding demand.
        used = _sum_resources(list(allocated.values()))
        remaining = _positive_difference(hard_extra_capacity, used)
        for field_name in ("heap_bytes",):
            tail = int(getattr(remaining, field_name))
            if tail <= 0:
                continue

            def weighted_hard_share(state: _QueryState) -> float:
                query_id = state.demand.query_id
                minimum = state.demand.minimum
                extra = allocated[query_id]
                current = ResourceVector(
                    cpu=minimum.cpu + extra.cpu,
                    gpu=minimum.gpu + extra.gpu,
                    heap_bytes=minimum.heap_bytes + extra.heap_bytes,
                )
                return current.dominant_share(hard_total_capacity) / state.demand.weight

            candidates = sorted(
                admitted,
                key=lambda state: (
                    weighted_hard_share(state),
                    -state.demand.priority,
                    state.sequence,
                ),
            )
            for state in candidates:
                query_id = state.demand.query_id
                field_headroom = int(getattr(headroom[query_id], field_name) - getattr(allocated[query_id], field_name))
                amount = min(tail, max(0, field_headroom))
                if amount <= 0:
                    continue
                allocated[query_id] = _replace_resource(
                    allocated[query_id],
                    field_name,
                    getattr(allocated[query_id], field_name) + amount,
                )
                tail -= amount
                if tail == 0:
                    break

        object_store_extras = ClusterQueryResourceCoordinator._weighted_object_store_extras(
            admitted,
            int(extra_capacity.object_store_bytes),
        )
        for state in admitted:
            query_id = state.demand.query_id
            allocated[query_id] = _replace_resource(
                allocated[query_id],
                "object_store_bytes",
                object_store_extras.get(query_id, 0),
            )
        return allocated

    @staticmethod
    def _weighted_object_store_extras(
        admitted: Sequence[_QueryState],
        extra_capacity_bytes: int,
    ) -> dict[str, int]:
        """Weighted max-min allocation for the spillable query budget."""
        capacity = max(0, int(extra_capacity_bytes))
        if not admitted or capacity <= 0:
            return {}

        minimum = {state.demand.query_id: int(state.demand.minimum.object_store_bytes) for state in admitted}
        desired = {state.demand.query_id: int(state.demand.desired.object_store_bytes) for state in admitted}
        limits = [
            desired[state.demand.query_id] / state.demand.weight
            for state in admitted
            if desired[state.demand.query_id] > minimum[state.demand.query_id]
        ]
        if not limits:
            return {}

        def allocation_at(level: float) -> dict[str, int]:
            result: dict[str, int] = {}
            for state in admitted:
                query_id = state.demand.query_id
                target = min(
                    desired[query_id],
                    max(
                        minimum[query_id],
                        level * state.demand.weight,
                    ),
                )
                result[query_id] = min(
                    desired[query_id] - minimum[query_id],
                    max(0, math.floor(target - minimum[query_id])),
                )
            return result

        def feasible(level: float) -> bool:
            return sum(allocation_at(level).values()) <= capacity

        low = 0.0
        high = max(limits)
        if feasible(high):
            low = high
        else:
            for _ in range(80):
                middle = (low + high) / 2.0
                if feasible(middle):
                    low = middle
                else:
                    high = middle
        allocated = allocation_at(low)

        # Flooring leaves at most a small integer tail. Hand out individual
        # bytes at the lowest weighted final budget to preserve fairness and a
        # deterministic priority/arrival-order tie break.
        tail = capacity - sum(allocated.values())
        while tail > 0:
            candidates = [
                state
                for state in admitted
                if allocated[state.demand.query_id] < desired[state.demand.query_id] - minimum[state.demand.query_id]
            ]
            if not candidates:
                break
            selected = min(
                candidates,
                key=lambda state: (
                    (minimum[state.demand.query_id] + allocated[state.demand.query_id]) / state.demand.weight,
                    -state.demand.priority,
                    state.sequence,
                    state.demand.query_id,
                ),
            )
            allocated[selected.demand.query_id] += 1
            tail -= 1
        return allocated

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "generation": self._generation,
                "heartbeat_timeout_s": self._heartbeat_timeout_s,
                "nodes": {node_id: node.to_dict() for node_id, node in self._nodes.items()},
                "queries": {
                    query_id: {
                        "state": state.state,
                        "priority": state.demand.priority,
                        "weight": state.demand.weight,
                        "allocation": state.allocation.to_dict(),
                        "observed_usage": state.observed_usage.to_dict(),
                        "allocation_debt": state.allocation_debt.to_dict(),
                        "soft_object_store_debt_bytes": max(
                            0,
                            state.observed_usage.object_store_bytes - state.allocation.resources.object_store_bytes,
                        ),
                        "can_admit_new_tasks": state.state == "RUNNING" and state.allocation_debt.is_zero(),
                        "rejection_reason": state.rejection_reason,
                        "node_allocations": {
                            node_id: vector.to_dict() for node_id, vector in sorted(state.node_allocations.items())
                        },
                        "expires_at": state.expires_at,
                    }
                    for query_id, state in sorted(self._queries.items())
                },
            }


__all__ = [
    "ActorResourceBundle",
    "ClusterQueryResourceCoordinator",
    "ElasticGpuDemandError",
    "NodeCapacity",
    "QueryDemand",
    "read_ray_node_capacities",
]
