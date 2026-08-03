# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

_RESOURCE_UNIT_KIND_BY_BACKEND = {
    "ray_worker": "native_fragment",
    "ray_task": "ray_task_udf",
    "ray_actor": "ray_actor_pool",
}


def _strict_fields(payload: Mapping[str, Any], expected: tuple[str, ...], type_name: str) -> None:
    actual = set(payload)
    expected_set = set(expected)
    unknown = sorted(actual - expected_set)
    missing = sorted(expected_set - actual)
    if unknown:
        raise ValueError(f"{type_name} has unknown fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{type_name} is missing required fields: {', '.join(missing)}")


@dataclass(frozen=True)
class ResourceVector:
    """Resources owned by a query, resource unit, task, or output window.

    CPU and GPU are Ray logical resources and may be fractional. Byte resources
    use integer accounting units. Query allocations treat CPU, GPU, and heap as
    hard ownership while object-store bytes are a flow-control budget: live
    ObjectRefs may temporarily exceed it under the bounded liveness policy and
    Ray Core may spill them. A vector is never allowed to carry negative
    capacity; subtraction that would underflow is a control-plane bug.
    """

    cpu: float = 0.0
    gpu: float = 0.0
    heap_bytes: int = 0
    object_store_bytes: int = 0

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "cpu",
        "gpu",
        "heap_bytes",
        "object_store_bytes",
    )

    def __post_init__(self) -> None:
        for name in ("cpu", "gpu"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
            object.__setattr__(self, name, value)
        for name in ("heap_bytes", "object_store_bytes"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
            object.__setattr__(self, name, value)

    def __add__(self, other: ResourceVector) -> ResourceVector:
        if not isinstance(other, ResourceVector):
            return NotImplemented
        return ResourceVector(
            cpu=self.cpu + other.cpu,
            gpu=self.gpu + other.gpu,
            heap_bytes=self.heap_bytes + other.heap_bytes,
            object_store_bytes=self.object_store_bytes + other.object_store_bytes,
        )

    def __sub__(self, other: ResourceVector) -> ResourceVector:
        if not isinstance(other, ResourceVector):
            return NotImplemented
        values = {
            "cpu": self.cpu - other.cpu,
            "gpu": self.gpu - other.gpu,
            "heap_bytes": self.heap_bytes - other.heap_bytes,
            "object_store_bytes": self.object_store_bytes - other.object_store_bytes,
        }
        underflow = [name for name, value in values.items() if value < 0]
        if underflow:
            raise ValueError(f"resource subtraction underflow: {', '.join(underflow)}")
        return ResourceVector(
            cpu=values["cpu"],
            gpu=values["gpu"],
            heap_bytes=int(values["heap_bytes"]),
            object_store_bytes=int(values["object_store_bytes"]),
        )

    def scale(self, factor: float) -> ResourceVector:
        factor = float(factor)
        if not math.isfinite(factor) or factor < 0:
            raise ValueError("resource scale factor must be finite and >= 0")
        return ResourceVector(
            cpu=self.cpu * factor,
            gpu=self.gpu * factor,
            heap_bytes=math.floor(self.heap_bytes * factor),
            object_store_bytes=math.floor(self.object_store_bytes * factor),
        )

    def fits_within(self, capacity: ResourceVector) -> bool:
        return all(getattr(self, name) <= getattr(capacity, name) for name in self._FIELDS)

    def exceeded_dimensions(self, capacity: ResourceVector) -> tuple[str, ...]:
        return tuple(name for name in self._FIELDS if getattr(self, name) > getattr(capacity, name))

    def dominant_share(self, capacity: ResourceVector) -> float:
        shares: list[float] = []
        for name in self._FIELDS:
            demand = float(getattr(self, name))
            available = float(getattr(capacity, name))
            if demand <= 0:
                shares.append(0.0)
            elif available <= 0:
                shares.append(math.inf)
            else:
                shares.append(demand / available)
        return max(shares, default=0.0)

    def is_zero(self) -> bool:
        return all(getattr(self, name) == 0 for name in self._FIELDS)

    def to_dict(self) -> dict[str, int | float]:
        return {
            "cpu": self.cpu,
            "gpu": self.gpu,
            "heap_bytes": self.heap_bytes,
            "object_store_bytes": self.object_store_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResourceVector:
        values = dict(payload)
        _strict_fields(values, cls._FIELDS, cls.__name__)
        return cls(
            cpu=float(values["cpu"]),
            gpu=float(values["gpu"]),
            heap_bytes=int(values["heap_bytes"]),
            object_store_bytes=int(values["object_store_bytes"]),
        )


@dataclass(frozen=True)
class NodeResourceAllocation:
    """One query's allocation attribution on one Ray node.

    CPU, GPU, and heap are enforced when a task or actor is placed on this
    node. The object-store component contributes to the query-wide soft
    flow-control budget; per-node object-store debt is observable but is not a
    separate QRM admission limit. Ray Core remains responsible for node-local
    spill and OOM protection.
    """

    node_id: str
    resources: ResourceVector

    _FIELDS: ClassVar[tuple[str, ...]] = ("node_id", "resources")

    def __post_init__(self) -> None:
        node_id = str(self.node_id).strip()
        if not node_id:
            raise ValueError("node allocation node_id must be non-empty")
        if self.resources.is_zero():
            raise ValueError(f"node allocation {node_id} must own non-zero resources")
        object.__setattr__(self, "node_id", node_id)

    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "resources": self.resources.to_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> NodeResourceAllocation:
        values = dict(payload)
        _strict_fields(values, cls._FIELDS, cls.__name__)
        return cls(
            node_id=str(values["node_id"]),
            resources=ResourceVector.from_dict(values["resources"]),
        )


@dataclass(frozen=True)
class ActorPlacement:
    """Coordinator-selected placement for one query-owned Ray actor."""

    resource_unit_id: str
    actor_index: int
    node_id: str

    _FIELDS: ClassVar[tuple[str, ...]] = ("resource_unit_id", "actor_index", "node_id")

    def __post_init__(self) -> None:
        resource_unit_id = str(self.resource_unit_id).strip()
        node_id = str(self.node_id).strip()
        actor_index = int(self.actor_index)
        if not resource_unit_id:
            raise ValueError("actor placement resource_unit_id must be non-empty")
        if actor_index < 0:
            raise ValueError("actor placement actor_index must be >= 0")
        if not node_id:
            raise ValueError("actor placement node_id must be non-empty")
        object.__setattr__(self, "resource_unit_id", resource_unit_id)
        object.__setattr__(self, "actor_index", actor_index)
        object.__setattr__(self, "node_id", node_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_unit_id": self.resource_unit_id,
            "actor_index": self.actor_index,
            "node_id": self.node_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ActorPlacement:
        values = dict(payload)
        _strict_fields(values, cls._FIELDS, cls.__name__)
        return cls(
            resource_unit_id=str(values["resource_unit_id"]),
            actor_index=int(values["actor_index"]),
            node_id=str(values["node_id"]),
        )


def _resource_vectors_equivalent(left: ResourceVector, right: ResourceVector) -> bool:
    return (
        math.isclose(left.cpu, right.cpu, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(left.gpu, right.gpu, rel_tol=0.0, abs_tol=1e-9)
        and left.heap_bytes == right.heap_bytes
        and left.object_store_bytes == right.object_store_bytes
    )


def _hard_resources(resources: ResourceVector) -> ResourceVector:
    """Return the non-spillable portion of a query resource vector."""
    return ResourceVector(
        cpu=resources.cpu,
        gpu=resources.gpu,
        heap_bytes=resources.heap_bytes,
    )


@dataclass(frozen=True)
class QueryAllocation:
    resources: ResourceVector
    node_allocations: tuple[NodeResourceAllocation, ...]
    actor_placements: tuple[ActorPlacement, ...]
    generation: int

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "resources",
        "node_allocations",
        "actor_placements",
        "generation",
    )

    def __post_init__(self) -> None:
        generation = int(self.generation)
        if generation <= 0:
            raise ValueError("generation must be > 0")
        node_allocations = tuple(self.node_allocations)
        actor_placements = tuple(self.actor_placements)
        node_ids = [allocation.node_id for allocation in node_allocations]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("query allocation contains duplicate node_id entries")
        aggregate = ResourceVector()
        for node_allocation in node_allocations:
            aggregate = aggregate + node_allocation.resources
        if not _resource_vectors_equivalent(aggregate, self.resources):
            raise ValueError("query allocation resources must equal the sum of node_allocations")
        placement_keys = [(placement.resource_unit_id, placement.actor_index) for placement in actor_placements]
        if len(set(placement_keys)) != len(placement_keys):
            raise ValueError("query allocation contains duplicate actor placements")
        unknown_nodes = sorted({placement.node_id for placement in actor_placements} - set(node_ids))
        if unknown_nodes:
            raise ValueError("actor placement references unallocated node_id: " + ", ".join(unknown_nodes))
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "node_allocations", node_allocations)
        object.__setattr__(self, "actor_placements", actor_placements)

    def resources_for_node(self, node_id: str) -> ResourceVector:
        node_key = str(node_id)
        for allocation in self.node_allocations:
            if allocation.node_id == node_key:
                return allocation.resources
        raise KeyError(f"query has no allocation on Ray node {node_key!r}")

    def actor_node_ids_for_unit(self, resource_unit_id: str) -> tuple[str, ...]:
        unit_key = str(resource_unit_id)
        placements = sorted(
            (placement for placement in self.actor_placements if placement.resource_unit_id == unit_key),
            key=lambda placement: placement.actor_index,
        )
        return tuple(placement.node_id for placement in placements)

    def to_dict(self) -> dict[str, Any]:
        return {
            "resources": self.resources.to_dict(),
            "node_allocations": [allocation.to_dict() for allocation in self.node_allocations],
            "actor_placements": [placement.to_dict() for placement in self.actor_placements],
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> QueryAllocation:
        values = dict(payload)
        _strict_fields(values, cls._FIELDS, cls.__name__)
        return cls(
            resources=ResourceVector.from_dict(values["resources"]),
            node_allocations=tuple(NodeResourceAllocation.from_dict(item) for item in values["node_allocations"]),
            actor_placements=tuple(ActorPlacement.from_dict(item) for item in values["actor_placements"]),
            generation=int(values["generation"]),
        )


@dataclass(frozen=True)
class ResourceUnitSpec:
    """Resource accounting for one independently scheduled execution family.

    A unit represents native fragment tasks, Ray task UDFs, or a Ray actor
    pool. It does not gate pipeline execution; ``is_barrier`` only marks the
    materialization boundary used to calculate the current reservation scope.
    """

    query_id: str
    resource_unit_id: str
    physical_node_id: str
    unit_kind: str
    backend: str
    input_unit_ids: tuple[str, ...]
    per_task: ResourceVector
    target_output_block_bytes: int
    generator_buffer_blocks: int
    max_concurrency: int | None
    resident_per_actor: ResourceVector = field(default_factory=ResourceVector)
    actor_pool_size: int = 0
    actor_prefetch_depth: int = 1
    is_barrier: bool = False

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "query_id",
        "resource_unit_id",
        "physical_node_id",
        "unit_kind",
        "backend",
        "input_unit_ids",
        "per_task",
        "target_output_block_bytes",
        "generator_buffer_blocks",
        "max_concurrency",
        "resident_per_actor",
        "actor_pool_size",
        "actor_prefetch_depth",
        "is_barrier",
    )

    @property
    def output_window_bytes(self) -> int:
        return int(self.target_output_block_bytes) * int(self.generator_buffer_blocks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "resource_unit_id": self.resource_unit_id,
            "physical_node_id": self.physical_node_id,
            "unit_kind": self.unit_kind,
            "backend": self.backend,
            "input_unit_ids": list(self.input_unit_ids),
            "per_task": self.per_task.to_dict(),
            "target_output_block_bytes": int(self.target_output_block_bytes),
            "generator_buffer_blocks": int(self.generator_buffer_blocks),
            "max_concurrency": None if self.max_concurrency is None else int(self.max_concurrency),
            "resident_per_actor": self.resident_per_actor.to_dict(),
            "actor_pool_size": int(self.actor_pool_size),
            "actor_prefetch_depth": int(self.actor_prefetch_depth),
            "is_barrier": self.is_barrier,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResourceUnitSpec:
        values = dict(payload)
        _strict_fields(values, cls._FIELDS, cls.__name__)
        max_concurrency = values["max_concurrency"]
        return cls(
            query_id=str(values["query_id"]),
            resource_unit_id=str(values["resource_unit_id"]),
            physical_node_id=str(values["physical_node_id"]),
            unit_kind=str(values["unit_kind"]),
            backend=str(values["backend"]),
            input_unit_ids=tuple(str(item) for item in values["input_unit_ids"]),
            per_task=ResourceVector.from_dict(values["per_task"]),
            target_output_block_bytes=int(values["target_output_block_bytes"]),
            generator_buffer_blocks=int(values["generator_buffer_blocks"]),
            max_concurrency=None if max_concurrency is None else int(max_concurrency),
            resident_per_actor=ResourceVector.from_dict(values["resident_per_actor"]),
            actor_pool_size=int(values["actor_pool_size"]),
            actor_prefetch_depth=int(values["actor_prefetch_depth"]),
            is_barrier=values["is_barrier"],
        )


@dataclass(frozen=True)
class QueryResourceGraph:
    """Dependency graph for query resource accounting and admission."""

    query_id: str
    plan_digest: str
    units: tuple[ResourceUnitSpec, ...]
    terminal_unit_ids: tuple[str, ...]

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "query_id",
        "plan_digest",
        "units",
        "terminal_unit_ids",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", str(self.query_id).strip())
        object.__setattr__(self, "plan_digest", str(self.plan_digest).strip())
        object.__setattr__(self, "units", tuple(self.units))
        object.__setattr__(self, "terminal_unit_ids", tuple(str(item) for item in self.terminal_unit_ids))
        self._validate()

    def _validate(self) -> None:
        if not self.query_id:
            raise ValueError("query_id must be non-empty")
        if not self.plan_digest:
            raise ValueError("plan_digest must be non-empty")
        if not self.units:
            raise ValueError("query resource graph must contain at least one unit")
        if not self.terminal_unit_ids:
            raise ValueError("query resource graph must contain at least one terminal unit")

        by_id: dict[str, ResourceUnitSpec] = {}
        physical_nodes: dict[str, str] = {}
        for unit in self.units:
            self._validate_unit(unit)
            if unit.resource_unit_id in by_id:
                raise ValueError(f"duplicate resource_unit_id: {unit.resource_unit_id}")
            if unit.physical_node_id in physical_nodes:
                raise ValueError(
                    "duplicate physical_node_id: "
                    f"{unit.physical_node_id} used by {physical_nodes[unit.physical_node_id]} and {unit.resource_unit_id}"
                )
            by_id[unit.resource_unit_id] = unit
            physical_nodes[unit.physical_node_id] = unit.resource_unit_id

        if len(set(self.terminal_unit_ids)) != len(self.terminal_unit_ids):
            raise ValueError("terminal_unit_ids must be unique")
        for terminal in self.terminal_unit_ids:
            if terminal not in by_id:
                raise ValueError(f"terminal unit is not registered: {terminal}")

        downstream: dict[str, set[str]] = {resource_unit_id: set() for resource_unit_id in by_id}
        for unit in self.units:
            if len(set(unit.input_unit_ids)) != len(unit.input_unit_ids):
                raise ValueError(f"unit {unit.resource_unit_id} has duplicate input_unit_ids")
            for input_unit_id in unit.input_unit_ids:
                if input_unit_id not in by_id:
                    raise ValueError(f"unit {unit.resource_unit_id} references missing input unit {input_unit_id}")
                if input_unit_id == unit.resource_unit_id:
                    raise ValueError(f"query resource graph contains a cycle at unit {unit.resource_unit_id}")
                downstream[input_unit_id].add(unit.resource_unit_id)

        ordered = self._topological_order(by_id, downstream)
        if len(ordered) != len(by_id):
            raise ValueError("query resource graph contains a cycle")

        for terminal in self.terminal_unit_ids:
            if downstream[terminal]:
                raise ValueError(
                    f"terminal unit {terminal} has downstream units: {', '.join(sorted(downstream[terminal]))}"
                )

        reaches_terminal = set(self.terminal_unit_ids)
        for resource_unit_id in reversed(ordered):
            if any(child in reaches_terminal for child in downstream[resource_unit_id]):
                reaches_terminal.add(resource_unit_id)
        missing_terminal_path = sorted(set(by_id) - reaches_terminal)
        if missing_terminal_path:
            raise ValueError(f"unit {missing_terminal_path[0]} does not reach a terminal unit")

    def _validate_unit(self, unit: ResourceUnitSpec) -> None:
        if str(unit.query_id).strip() != self.query_id:
            raise ValueError(
                f"unit {unit.resource_unit_id or '<empty>'} query_id {unit.query_id!r} does not match {self.query_id!r}"
            )
        if not str(unit.resource_unit_id).strip():
            raise ValueError("resource_unit_id must be non-empty")
        if not str(unit.resource_unit_id).startswith("resource:"):
            raise ValueError(f"resource_unit_id must use stable 'resource:' identity: {unit.resource_unit_id}")
        if not str(unit.physical_node_id).strip():
            raise ValueError(f"unit {unit.resource_unit_id} physical_node_id must be non-empty")
        unit_kind = str(unit.unit_kind).strip()
        if not unit_kind:
            raise ValueError(f"unit {unit.resource_unit_id} unit_kind must be non-empty")
        backend = str(unit.backend).strip()
        if not backend:
            raise ValueError(f"unit {unit.resource_unit_id} backend must be non-empty")
        expected_unit_kind = _RESOURCE_UNIT_KIND_BY_BACKEND.get(backend)
        if expected_unit_kind is None:
            raise ValueError(f"unit {unit.resource_unit_id} has unsupported backend {backend!r}")
        if unit_kind != expected_unit_kind:
            raise ValueError(
                f"unit {unit.resource_unit_id} kind {unit_kind!r} does not match "
                f"backend {backend!r}; expected {expected_unit_kind!r}"
            )
        if int(unit.target_output_block_bytes) < 0:
            raise ValueError(f"unit {unit.resource_unit_id} target_output_block_bytes must be >= 0")
        if int(unit.generator_buffer_blocks) < 0:
            raise ValueError(f"unit {unit.resource_unit_id} generator_buffer_blocks must be >= 0")
        target = int(unit.target_output_block_bytes)
        blocks = int(unit.generator_buffer_blocks)
        if target == 0 and blocks != 0:
            raise ValueError(
                f"unit {unit.resource_unit_id} target_output_block_bytes and generator_buffer_blocks must both be zero"
            )
        if target > 0 and blocks <= 0:
            raise ValueError(
                f"unit {unit.resource_unit_id} target_output_block_bytes and generator_buffer_blocks must both be positive"
            )
        if unit.max_concurrency is not None and int(unit.max_concurrency) <= 0:
            raise ValueError(f"unit {unit.resource_unit_id} max_concurrency must be > 0")
        if not isinstance(unit.is_barrier, bool):
            raise ValueError(f"unit {unit.resource_unit_id} is_barrier must be a boolean")

        process_resources = unit.resident_per_actor if unit.backend == "ray_actor" else unit.per_task
        if process_resources.cpu <= 0 and process_resources.gpu <= 0:
            raise ValueError(f"unit {unit.resource_unit_id} process commitment must request CPU or GPU resources")
        if process_resources.heap_bytes <= 0:
            raise ValueError(f"unit {unit.resource_unit_id} process commitment must request non-zero heap_bytes")

        actor_pool_size = int(unit.actor_pool_size)
        actor_prefetch_depth = int(unit.actor_prefetch_depth)
        if unit.backend == "ray_actor":
            if actor_pool_size <= 0:
                raise ValueError(f"ray_actor unit {unit.resource_unit_id} actor_pool_size must be > 0")
            if actor_prefetch_depth <= 0:
                raise ValueError(f"ray_actor unit {unit.resource_unit_id} actor_prefetch_depth must be > 0")
            if unit.max_concurrency is not None:
                raise ValueError(f"ray_actor unit {unit.resource_unit_id} concurrency is owned by concrete actor slots")
            if unit.per_task.cpu or unit.per_task.gpu or unit.per_task.heap_bytes:
                raise ValueError(
                    f"ray_actor unit {unit.resource_unit_id} invocation resources may only contain object-store bytes"
                )
        elif actor_pool_size != 0:
            raise ValueError(f"actor_pool_size is only valid for ray_actor units: {unit.resource_unit_id}")
        elif actor_prefetch_depth != 1:
            raise ValueError(f"actor_prefetch_depth is only configurable for ray_actor units: {unit.resource_unit_id}")
        elif not unit.resident_per_actor.is_zero():
            raise ValueError(f"resident_per_actor is only valid for ray_actor units: {unit.resource_unit_id}")
        if unit.backend == "ray_task" and unit.max_concurrency is not None:
            raise ValueError(f"ray_task unit {unit.resource_unit_id} concurrency is owned by resource credit")

    @staticmethod
    def _topological_order(
        by_id: Mapping[str, ResourceUnitSpec],
        downstream: Mapping[str, set[str]],
    ) -> tuple[str, ...]:
        indegree = {resource_unit_id: len(unit.input_unit_ids) for resource_unit_id, unit in by_id.items()}
        ready = [resource_unit_id for resource_unit_id, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        ordered: list[str] = []
        while ready:
            resource_unit_id = heapq.heappop(ready)
            ordered.append(resource_unit_id)
            for child in sorted(downstream[resource_unit_id]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, child)
        return tuple(ordered)

    def unit_by_id(self, resource_unit_id: str) -> ResourceUnitSpec:
        key = str(resource_unit_id)
        for unit in self.units:
            if unit.resource_unit_id == key:
                return unit
        raise KeyError(f"unknown resource_unit_id {key!r}")

    def unit_id_for_physical_node(self, physical_node_id: str) -> str:
        key = str(physical_node_id)
        for unit in self.units:
            if unit.physical_node_id == key:
                return unit.resource_unit_id
        raise KeyError(f"unknown physical_node_id {key!r}")

    def topological_unit_ids(self) -> tuple[str, ...]:
        by_id = {unit.resource_unit_id: unit for unit in self.units}
        downstream: dict[str, set[str]] = {resource_unit_id: set() for resource_unit_id in by_id}
        for unit in self.units:
            for parent in unit.input_unit_ids:
                downstream[parent].add(unit.resource_unit_id)
        return self._topological_order(by_id, downstream)

    def reverse_topological_unit_ids(self) -> tuple[str, ...]:
        return tuple(reversed(self.topological_unit_ids()))

    def downstream_native_fragment_unit_ids_requiring_separate_slot(
        self,
        source_unit_id: str,
    ) -> tuple[str, ...]:
        """Return downstream native fragment units separated by a remote-process boundary.

        A native fragment task can normally hand its capacity directly to a
        downstream native fragment after it finishes. That capacity is not
        transferable while the fragment is blocked inside a nested Ray task
        or actor invocation: the producer remains alive until the remote
        continuation returns. Every native fragment reachable after such a
        boundary therefore shares one additional progress slot that admission
        must keep placeable.

        The source itself counts as a boundary when it is not a Ray worker.
        Results follow the graph's deterministic topological order.  For a
        join, one boundary-crossing input path is sufficient to require the
        separate slot.
        """
        source = self.unit_by_id(source_unit_id)
        by_id = {unit.resource_unit_id: unit for unit in self.units}
        downstream: dict[str, set[str]] = {resource_unit_id: set() for resource_unit_id in by_id}
        for unit in self.units:
            for input_unit_id in unit.input_unit_ids:
                downstream[input_unit_id].add(unit.resource_unit_id)

        crossed_remote_process: dict[str, bool] = {source.resource_unit_id: source.backend != "ray_worker"}
        ordered = self.topological_unit_ids()
        for resource_unit_id in ordered:
            crossed = crossed_remote_process.get(resource_unit_id)
            if crossed is None:
                continue
            for child_id in downstream[resource_unit_id]:
                child_crossed = crossed or by_id[child_id].backend != "ray_worker"
                crossed_remote_process[child_id] = crossed_remote_process.get(child_id, False) or child_crossed

        return tuple(
            resource_unit_id
            for resource_unit_id in ordered
            if resource_unit_id != source.resource_unit_id
            and by_id[resource_unit_id].backend == "ray_worker"
            and crossed_remote_process.get(resource_unit_id, False)
        )

    def task_identity(self, resource_unit_id: str, *, partition_id: int | str, attempt_id: int | str) -> str:
        unit = self.unit_by_id(resource_unit_id)
        partition = str(partition_id).strip()
        attempt = str(attempt_id).strip()
        if not partition:
            raise ValueError("partition_id must be non-empty")
        if not attempt:
            raise ValueError("attempt_id must be non-empty")
        return f"task:{unit.resource_unit_id}:partition:{partition}:attempt:{attempt}"

    def validate_allocation(
        self,
        allocation: QueryAllocation,
        *,
        require_full_minimum: bool = True,
    ) -> None:
        hard_capacity = _hard_resources(allocation.resources)
        for unit in self.units:
            task_commitment = _hard_resources(unit.per_task)
            actor_resident = unit.resident_per_actor if unit.backend == "ray_actor" else ResourceVector()
            placement_commitment = actor_resident + task_commitment
            exceeded = placement_commitment.exceeded_dimensions(hard_capacity)
            if require_full_minimum and exceeded:
                raise ValueError(
                    f"unit {unit.resource_unit_id} maximum task exceeds query allocation for {', '.join(exceeded)}"
                )
            if (
                require_full_minimum
                and not task_commitment.is_zero()
                and not any(
                    placement_commitment.fits_within(_hard_resources(node_allocation.resources))
                    for node_allocation in allocation.node_allocations
                )
            ):
                raise ValueError(f"unit {unit.resource_unit_id} maximum task does not fit any allocated Ray node")
            if unit.backend == "ray_actor":
                actor_pool = actor_resident.scale(unit.actor_pool_size)
                exceeded_actor = actor_pool.exceeded_dimensions(hard_capacity)
                if require_full_minimum and exceeded_actor:
                    raise ValueError(
                        f"unit {unit.resource_unit_id} actor pool exceeds query allocation for {', '.join(exceeded_actor)}"
                    )
                placements = [
                    placement
                    for placement in allocation.actor_placements
                    if placement.resource_unit_id == unit.resource_unit_id
                ]
                if require_full_minimum and len(placements) != unit.actor_pool_size:
                    raise ValueError(
                        f"unit {unit.resource_unit_id} requires exactly {unit.actor_pool_size} actor placements"
                    )
                expected_indices = set(range(unit.actor_pool_size))
                placement_indices = {placement.actor_index for placement in placements}
                if placements and placement_indices != expected_indices:
                    raise ValueError(
                        f"unit {unit.resource_unit_id} actor placement indices must be contiguous from zero"
                    )
                for placement in placements:
                    if not placement_commitment.fits_within(
                        _hard_resources(allocation.resources_for_node(placement.node_id))
                    ):
                        raise ValueError(
                            f"unit {unit.resource_unit_id} actor {placement.actor_index} does not fit "
                            f"allocated Ray node {placement.node_id}"
                        )

        known_actor_unit_ids = {unit.resource_unit_id for unit in self.units if unit.backend == "ray_actor"}
        unknown_actor_units = sorted(
            {placement.resource_unit_id for placement in allocation.actor_placements} - known_actor_unit_ids
        )
        if unknown_actor_units:
            raise ValueError("actor placement references non-actor unit: " + ", ".join(unknown_actor_units))
        if require_full_minimum:
            actor_commitment_by_node: dict[str, ResourceVector] = {}
            for placement in allocation.actor_placements:
                unit = self.unit_by_id(placement.resource_unit_id)
                actor_commitment_by_node[placement.node_id] = (
                    actor_commitment_by_node.get(
                        placement.node_id,
                        ResourceVector(),
                    )
                    + unit.resident_per_actor
                )
            for node_id, commitment in actor_commitment_by_node.items():
                if not commitment.fits_within(_hard_resources(allocation.resources_for_node(node_id))):
                    raise ValueError(f"cumulative actor placements do not fit allocated Ray node {node_id}")

    def normalized_digest(self) -> str:
        payload = self.to_dict()
        payload["plan_digest"] = ""
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "plan_digest": self.plan_digest,
            "units": [unit.to_dict() for unit in self.units],
            "terminal_unit_ids": list(self.terminal_unit_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> QueryResourceGraph:
        values = dict(payload)
        _strict_fields(values, cls._FIELDS, cls.__name__)
        return cls(
            query_id=str(values["query_id"]),
            plan_digest=str(values["plan_digest"]),
            units=tuple(ResourceUnitSpec.from_dict(item) for item in values["units"]),
            terminal_unit_ids=tuple(str(item) for item in values["terminal_unit_ids"]),
        )


__all__ = [
    "ActorPlacement",
    "NodeResourceAllocation",
    "QueryAllocation",
    "QueryResourceGraph",
    "ResourceVector",
    "ResourceUnitSpec",
]
