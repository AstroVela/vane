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

from duckdb.runners.ray.query_execution_graph import (
    ActorPlacement,
    NodeResourceAllocation,
    QueryAllocation,
    ResourceVector,
)
from duckdb.runners.ray.worker_memory import build_ray_node_memory_layout

_EPSILON = 1e-9


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
    stage_id: str
    actor_index: int
    resources: ResourceVector

    def __post_init__(self) -> None:
        stage_id = str(self.stage_id).strip()
        actor_index = int(self.actor_index)
        if not stage_id:
            raise ValueError("actor resource bundle stage_id must be non-empty")
        if actor_index < 0:
            raise ValueError("actor resource bundle actor_index must be >= 0")
        if self.resources.is_zero():
            raise ValueError("actor resource bundle must own non-zero resources")
        object.__setattr__(self, "stage_id", stage_id)
        object.__setattr__(self, "actor_index", actor_index)


@dataclass(frozen=True)
class QueryDemand:
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
        weight = float(self.weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("weight must be finite and > 0")
        actor_bundles = tuple(self.actor_bundles)
        task_bundles = tuple(self.task_bundles)
        actor_keys = [(bundle.stage_id, bundle.actor_index) for bundle in actor_bundles]
        if len(set(actor_keys)) != len(actor_keys):
            raise ValueError("actor resource bundles contain duplicate stage/index identities")
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
                (bundle.stage_id, bundle.actor_index): bundle.resources for bundle in state.demand.actor_bundles
            }
            pinned_allocations: dict[str, ResourceVector] = {}
            for placement in state.actor_placements:
                vector = actor_bundles.get((placement.stage_id, placement.actor_index))
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
                for bundle in state.demand.actor_bundles:
                    node_id = self._place_bundle(bundle.resources, trial_remaining)
                    if node_id is None:
                        actor_ok = False
                        break
                    new_actor_placements.append(
                        ActorPlacement(
                            stage_id=bundle.stage_id,
                            actor_index=bundle.actor_index,
                            node_id=node_id,
                        )
                    )
                    trial_allocations[node_id] = trial_allocations.get(node_id, ResourceVector()) + bundle.resources
                if not actor_ok:
                    continue
                actor_placements_by_query[query_id] = tuple(new_actor_placements)

            task_ok = True
            for bundle in state.demand.task_bundles:
                node_id = self._place_bundle(bundle, trial_remaining)
                if node_id is None:
                    task_ok = False
                    break
                trial_allocations[node_id] = trial_allocations.get(node_id, ResourceVector()) + bundle
            if not task_ok:
                continue

            remaining = trial_remaining
            node_allocations_by_query[query_id] = trial_allocations
            admitted.append(state)

        total_capacity = _sum_resources([node.resources for node in self._nodes.values()])
        extra_capacity = _sum_resources(list(remaining.values()))
        extras = self._weighted_drf_extras(admitted, extra_capacity, total_capacity)

        # Place the aggregate DRF result onto concrete nodes. Each dimension is
        # divisible; indivisible actor bundles were already placed above.
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
            state.allocation_debt = _positive_difference(state.observed_usage, resources)
            if not state.allocation_debt.is_zero() and state.state != "ACTOR_PLACEMENT_LOST":
                state.state = "ALLOCATION_DEBT"
                state.rejection_reason = "live query leases exceed the current allocation"

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
            return None
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
        if not admitted or extra_capacity.is_zero():
            return {}

        headroom: dict[str, ResourceVector] = {
            state.demand.query_id: state.demand.desired - state.demand.minimum for state in admitted
        }
        dominant: dict[str, float] = {
            query_id: vector.dominant_share(total_capacity) for query_id, vector in headroom.items()
        }
        states_by_id = {state.demand.query_id: state for state in admitted}
        finite_limits = [
            dominant[query_id] / states_by_id[query_id].demand.weight
            for query_id in headroom
            if dominant[query_id] > 0 and math.isfinite(dominant[query_id])
        ]
        if not finite_limits:
            return {}

        def allocation_at(level: float) -> dict[str, ResourceVector]:
            result: dict[str, ResourceVector] = {}
            for query_id, room in headroom.items():
                room_dominant = dominant[query_id]
                if room_dominant <= 0 or not math.isfinite(room_dominant):
                    result[query_id] = ResourceVector()
                    continue
                weight = states_by_id[query_id].demand.weight
                factor = min(1.0, level * weight / room_dominant)
                result[query_id] = room.scale(factor)
            return result

        def feasible(level: float) -> bool:
            total = _sum_resources(list(allocation_at(level).values()))
            return total.fits_within(extra_capacity)

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

        # Byte flooring can leave a small tail. Assign it deterministically to
        # the currently lowest weighted dominant share without exceeding demand.
        used = _sum_resources(list(allocated.values()))
        remaining = _positive_difference(extra_capacity, used)
        for field_name in ("heap_bytes", "object_store_bytes"):
            tail = int(getattr(remaining, field_name))
            if tail <= 0:
                continue
            candidates = sorted(
                admitted,
                key=lambda state: (
                    (state.demand.minimum + allocated[state.demand.query_id]).dominant_share(total_capacity)
                    / state.demand.weight,
                    -state.demand.priority,
                    state.sequence,
                ),
            )
            for state in candidates:
                query_id = state.demand.query_id
                room = int(getattr(headroom[query_id], field_name) - getattr(allocated[query_id], field_name))
                amount = min(tail, max(0, room))
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
    "NodeCapacity",
    "QueryDemand",
    "read_ray_node_capacities",
]
