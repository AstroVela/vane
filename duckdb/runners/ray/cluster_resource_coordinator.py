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

from duckdb.runners.ray.query_resource_graph import QueryAllocation, ResourceVector
from duckdb.runners.ray.worker_memory import build_ray_node_memory_layout

_EPSILON = 1e-9
_RESOURCE_FIELDS = ("cpu", "gpu", "heap_bytes", "object_store_bytes")
_INTEGER_RESOURCE_FIELDS = {"heap_bytes", "object_store_bytes"}


def _sum_resources(resources: Sequence[ResourceVector]) -> ResourceVector:
    total = ResourceVector()
    for item in resources:
        total = total + item
    return total


def _positive_difference(left: ResourceVector, right: ResourceVector) -> ResourceVector:
    return ResourceVector(
        cpu=max(0.0, left.cpu - right.cpu),
        gpu=max(0.0, left.gpu - right.gpu),
        heap_bytes=max(0, left.heap_bytes - right.heap_bytes),
        object_store_bytes=max(0, left.object_store_bytes - right.object_store_bytes),
    )


@dataclass(frozen=True)
class NodeCapacity:
    """One live Ray node used only to derive the cluster-wide soft budget."""

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
class QueryDemand:
    """A query's current-phase soft resource target.

    Demand is intentionally aggregate and divisible. It is not a placement
    bundle and never prevents concrete Ray tasks or actors from being
    submitted. Ray Core owns physical feasibility, pending demand, and
    autoscaling; the coordinator only divides the observed cluster capacity
    into weighted soft backpressure budgets.
    """

    query_id: str
    desired: ResourceVector
    weight: float = 1.0
    priority: int = 0

    def __post_init__(self) -> None:
        query_id = str(self.query_id).strip()
        if not query_id:
            raise ValueError("query_id must be non-empty")
        weight = float(self.weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("weight must be finite and > 0")
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "priority", int(self.priority))


@dataclass
class _QueryState:
    demand: QueryDemand
    sequence: int
    observed_usage: ResourceVector = field(default_factory=ResourceVector)
    allocation: QueryAllocation = field(
        default_factory=lambda: QueryAllocation(
            resources=ResourceVector(),
            generation=1,
        )
    )
    soft_allocation_debt: ResourceVector = field(default_factory=ResourceVector)
    expires_at: float = 0.0


def read_ray_node_capacities(
    ray_module: Any,
    *,
    object_store_fraction: float = 0.5,
    heap_reserve_bytes_per_node: int = 0,
) -> tuple[NodeCapacity, ...]:
    """Read live Ray capacity without inventing host-resource fallbacks."""

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
    """Divide cluster capacity into query-level soft budgets.

    The coordinator deliberately performs no bin packing and has no pending
    admission state. Even a zero-budget query remains runnable so one bounded
    QRM liveness grant can submit real demand to Ray Core and its autoscaler.
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
        demands_by_query: Mapping[str, QueryDemand] | None = None,
        now: float | None = None,
    ) -> dict[str, QueryAllocation]:
        """Atomically refresh every live query from one driver snapshot."""

        timestamp = time.monotonic() if now is None else float(now)
        usage = {str(query_id): value for query_id, value in observed_usage_by_query.items()}
        generation_by_query = {str(query_id): int(generation) for query_id, generation in generations.items()}
        demands = (
            None
            if demands_by_query is None
            else {str(query_id): demand for query_id, demand in demands_by_query.items()}
        )
        if set(usage) != set(generation_by_query):
            raise ValueError("refresh query usage and generation sets must match")
        if demands is not None and set(demands) != set(usage):
            raise ValueError("refresh query demand and usage sets must match")
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
                if demands is not None:
                    demand = demands[query_id]
                    if not isinstance(demand, QueryDemand):
                        raise TypeError(f"demand for query {query_id} must be QueryDemand")
                    if demand.query_id != query_id:
                        raise ValueError(f"refresh demand query_id mismatch for query {query_id}")
            for query_id in sorted(expected):
                state = self._queries[query_id]
                if demands is not None:
                    state.demand = demands[query_id]
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

    def query_state(self, query_id: str, generation: int) -> str:
        """Return the generation-fenced state; registered queries always run."""

        query_key = str(query_id)
        with self._lock:
            state = self._queries.get(query_key)
            if state is None:
                raise KeyError(f"query is not registered: {query_key}")
            self._require_generation(state, generation)
            return "RUNNING"

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
        del now
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
        total_capacity = _sum_resources(tuple(node.resources for node in self._nodes.values()))
        ordered = tuple(
            sorted(
                self._queries.values(),
                key=lambda state: (-state.demand.priority, state.sequence, state.demand.query_id),
            )
        )
        allocations = self._weighted_soft_allocations(ordered, total_capacity)
        for state in ordered:
            resources = allocations.get(state.demand.query_id, ResourceVector())
            state.allocation = QueryAllocation(resources=resources, generation=generation)
            state.soft_allocation_debt = _positive_difference(state.observed_usage, resources)

    @staticmethod
    def _weighted_field_allocations(
        states: Sequence[_QueryState],
        capacity: int | float,
        field_name: str,
    ) -> dict[str, int | float]:
        available = max(0.0, float(capacity))
        result: dict[str, int | float] = {state.demand.query_id: 0 for state in states}
        candidates = [state for state in states if float(getattr(state.demand.desired, field_name)) > 0]
        if not candidates or available <= 0:
            return result

        total_desired = sum(float(getattr(state.demand.desired, field_name)) for state in candidates)
        if total_desired <= available + _EPSILON:
            for state in candidates:
                result[state.demand.query_id] = getattr(state.demand.desired, field_name)
            return result

        high = max(float(getattr(state.demand.desired, field_name)) / state.demand.weight for state in candidates)

        def allocation_at(level: float) -> dict[str, float]:
            return {
                state.demand.query_id: min(
                    float(getattr(state.demand.desired, field_name)),
                    level * state.demand.weight,
                )
                for state in candidates
            }

        low = 0.0
        for _ in range(80):
            middle = (low + high) / 2.0
            if sum(allocation_at(middle).values()) <= available:
                low = middle
            else:
                high = middle
        raw = allocation_at(low)

        if field_name not in _INTEGER_RESOURCE_FIELDS:
            for query_id, amount in raw.items():
                result[query_id] = amount
            return result

        integer_capacity = int(capacity)
        for query_id, amount in raw.items():
            result[query_id] = math.floor(amount)
        remaining = integer_capacity - sum(int(value) for value in result.values())
        while remaining > 0:
            eligible = [
                state
                for state in candidates
                if int(result[state.demand.query_id]) < int(getattr(state.demand.desired, field_name))
            ]
            if not eligible:
                break
            selected = min(
                eligible,
                key=lambda state: (
                    float(result[state.demand.query_id]) / state.demand.weight,
                    -state.demand.priority,
                    state.sequence,
                    state.demand.query_id,
                ),
            )
            result[selected.demand.query_id] = int(result[selected.demand.query_id]) + 1
            remaining -= 1
        return result

    @classmethod
    def _weighted_soft_allocations(
        cls,
        states: Sequence[_QueryState],
        capacity: ResourceVector,
    ) -> dict[str, ResourceVector]:
        by_field = {
            field_name: cls._weighted_field_allocations(states, getattr(capacity, field_name), field_name)
            for field_name in _RESOURCE_FIELDS
        }
        return {
            state.demand.query_id: ResourceVector(
                cpu=float(by_field["cpu"][state.demand.query_id]),
                gpu=float(by_field["gpu"][state.demand.query_id]),
                heap_bytes=int(by_field["heap_bytes"][state.demand.query_id]),
                object_store_bytes=int(by_field["object_store_bytes"][state.demand.query_id]),
            )
            for state in states
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "generation": self._generation,
                "heartbeat_timeout_s": self._heartbeat_timeout_s,
                "nodes": {node_id: node.to_dict() for node_id, node in self._nodes.items()},
                "queries": {
                    query_id: {
                        "state": "RUNNING",
                        "priority": state.demand.priority,
                        "weight": state.demand.weight,
                        "allocation": state.allocation.to_dict(),
                        "observed_usage": state.observed_usage.to_dict(),
                        "soft_allocation_debt": state.soft_allocation_debt.to_dict(),
                        "expires_at": state.expires_at,
                    }
                    for query_id, state in sorted(self._queries.items())
                },
            }


__all__ = [
    "ClusterQueryResourceCoordinator",
    "NodeCapacity",
    "QueryDemand",
    "read_ray_node_capacities",
]
