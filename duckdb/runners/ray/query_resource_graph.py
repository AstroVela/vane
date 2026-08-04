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
_RESOURCE_ABS_TOLERANCE = 1e-12
_RESOURCE_REL_TOLERANCE = 1e-12


def _float_resource_leq(value: float, capacity: float) -> bool:
    return value <= capacity or math.isclose(
        value,
        capacity,
        rel_tol=_RESOURCE_REL_TOLERANCE,
        abs_tol=_RESOURCE_ABS_TOLERANCE,
    )


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
    """Resources requested by work or attributed to a soft query budget.

    CPU and GPU are Ray logical resources and may be fractional. Byte resources
    use integer accounting units. On a concrete task or actor, CPU, GPU, and
    declared heap become real Ray Core scheduling requests. On a
    ``QueryAllocation`` all four fields are soft admission/reservation shares:
    existing work is not revoked after an overage, and one bounded liveness
    path may escape a soft block. Object-store bytes additionally describe
    spillable flow rather than process memory. A vector is never allowed to
    carry negative capacity; subtraction that would underflow is a
    control-plane bug.
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
        underflow = [name for name in ("heap_bytes", "object_store_bytes") if values[name] < 0]
        underflow.extend(
            name
            for name in ("cpu", "gpu")
            if not _float_resource_leq(float(getattr(other, name)), float(getattr(self, name)))
        )
        if underflow:
            raise ValueError(f"resource subtraction underflow: {', '.join(underflow)}")
        return ResourceVector(
            cpu=max(0.0, values["cpu"]),
            gpu=max(0.0, values["gpu"]),
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
        return (
            _float_resource_leq(self.cpu, capacity.cpu)
            and _float_resource_leq(self.gpu, capacity.gpu)
            and self.heap_bytes <= capacity.heap_bytes
            and self.object_store_bytes <= capacity.object_store_bytes
        )

    def exceeded_dimensions(self, capacity: ResourceVector) -> tuple[str, ...]:
        return tuple(
            name
            for name in self._FIELDS
            if (
                not _float_resource_leq(float(getattr(self, name)), float(getattr(capacity, name)))
                if name in {"cpu", "gpu"}
                else getattr(self, name) > getattr(capacity, name)
            )
        )

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

    QRM uses CPU, GPU, and heap attribution to choose a feasible logical slot
    before submission; Ray Core remains the physical placement authority and
    actors are not pinned to these nodes. The object-store component contributes
    only to the query-wide soft flow-control budget. Per-node debt is
    observable, while Ray Core remains responsible for node-local spill and OOM
    protection.
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
    generation: int

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "resources",
        "node_allocations",
        "generation",
    )

    def __post_init__(self) -> None:
        generation = int(self.generation)
        if generation <= 0:
            raise ValueError("generation must be > 0")
        node_allocations = tuple(self.node_allocations)
        node_ids = [allocation.node_id for allocation in node_allocations]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("query allocation contains duplicate node_id entries")
        aggregate = ResourceVector()
        for node_allocation in node_allocations:
            aggregate = aggregate + node_allocation.resources
        if not _resource_vectors_equivalent(aggregate, self.resources):
            raise ValueError("query allocation resources must equal the sum of node_allocations")
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "node_allocations", node_allocations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "resources": self.resources.to_dict(),
            "node_allocations": [allocation.to_dict() for allocation in self.node_allocations],
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> QueryAllocation:
        values = dict(payload)
        _strict_fields(values, cls._FIELDS, cls.__name__)
        return cls(
            resources=ResourceVector.from_dict(values["resources"]),
            node_allocations=tuple(NodeResourceAllocation.from_dict(item) for item in values["node_allocations"]),
            generation=int(values["generation"]),
        )


@dataclass(frozen=True)
class MaterializationBarrierSpec:
    """A true distributed boundary inside one physical materializer node.

    Barriers are execution events, not resource-owning operators.  The
    materializer remains a normal native resource unit so its managed object
    flow can be accounted without inventing a separate Stage abstraction.
    Before completion only ``materialized_input_unit_ids`` feed that node;
    after completion the same node may emit final work from its remaining
    inputs before downstream execution continues.
    """

    query_id: str
    barrier_id: str
    physical_node_id: str
    materializer_unit_id: str
    materialized_input_unit_ids: tuple[str, ...]

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "query_id",
        "barrier_id",
        "physical_node_id",
        "materializer_unit_id",
        "materialized_input_unit_ids",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", str(self.query_id).strip())
        object.__setattr__(self, "barrier_id", str(self.barrier_id).strip())
        object.__setattr__(self, "physical_node_id", str(self.physical_node_id).strip())
        object.__setattr__(self, "materializer_unit_id", str(self.materializer_unit_id).strip())
        object.__setattr__(
            self,
            "materialized_input_unit_ids",
            tuple(str(item).strip() for item in self.materialized_input_unit_ids),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "barrier_id": self.barrier_id,
            "physical_node_id": self.physical_node_id,
            "materializer_unit_id": self.materializer_unit_id,
            "materialized_input_unit_ids": list(self.materialized_input_unit_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MaterializationBarrierSpec:
        values = dict(payload)
        _strict_fields(values, cls._FIELDS, cls.__name__)
        return cls(
            query_id=str(values["query_id"]),
            barrier_id=str(values["barrier_id"]),
            physical_node_id=str(values["physical_node_id"]),
            materializer_unit_id=str(values["materializer_unit_id"]),
            materialized_input_unit_ids=tuple(str(item) for item in values["materialized_input_unit_ids"]),
        )


@dataclass(frozen=True)
class ResourceUnitSpec:
    """Resource accounting for one independently scheduled execution family.

    Native-fragment units retain graph identity and managed object-flow
    windows, but do not own process resources: the node-local shared DuckDB
    instance schedules their threads and manages their working memory. Ray
    task and actor units own the process resources declared by their UDFs.
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
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", str(self.query_id).strip())
        object.__setattr__(self, "resource_unit_id", str(self.resource_unit_id).strip())
        object.__setattr__(self, "physical_node_id", str(self.physical_node_id).strip())
        object.__setattr__(self, "unit_kind", str(self.unit_kind).strip())
        object.__setattr__(self, "backend", str(self.backend).strip())
        object.__setattr__(self, "input_unit_ids", tuple(str(item).strip() for item in self.input_unit_ids))
        object.__setattr__(self, "target_output_block_bytes", int(self.target_output_block_bytes))
        object.__setattr__(self, "generator_buffer_blocks", int(self.generator_buffer_blocks))
        object.__setattr__(
            self,
            "max_concurrency",
            None if self.max_concurrency is None else int(self.max_concurrency),
        )
        object.__setattr__(self, "actor_pool_size", int(self.actor_pool_size))
        object.__setattr__(self, "actor_prefetch_depth", int(self.actor_prefetch_depth))

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
        )


@dataclass(frozen=True)
class QueryResourceGraph:
    """Dependency graph for query resource accounting and admission."""

    query_id: str
    plan_digest: str
    units: tuple[ResourceUnitSpec, ...]
    terminal_unit_ids: tuple[str, ...]
    materialization_barriers: tuple[MaterializationBarrierSpec, ...] = ()

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "query_id",
        "plan_digest",
        "units",
        "materialization_barriers",
        "terminal_unit_ids",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", str(self.query_id).strip())
        object.__setattr__(self, "plan_digest", str(self.plan_digest).strip())
        object.__setattr__(self, "units", tuple(self.units))
        object.__setattr__(self, "materialization_barriers", tuple(self.materialization_barriers))
        object.__setattr__(self, "terminal_unit_ids", tuple(str(item).strip() for item in self.terminal_unit_ids))
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

        barrier_ids: set[str] = set()
        barrier_nodes: set[str] = set()
        barrier_units: set[str] = set()
        for barrier in self.materialization_barriers:
            if str(barrier.query_id).strip() != self.query_id:
                raise ValueError(
                    f"barrier {barrier.barrier_id or '<empty>'} query_id {barrier.query_id!r} "
                    f"does not match {self.query_id!r}"
                )
            barrier_id = str(barrier.barrier_id).strip()
            physical_node_id = str(barrier.physical_node_id).strip()
            materializer_unit_id = str(barrier.materializer_unit_id).strip()
            if barrier_id != f"barrier:{self.query_id}:node:{physical_node_id}":
                raise ValueError(f"invalid materialization barrier identity: {barrier_id!r}")
            if not physical_node_id:
                raise ValueError(f"barrier {barrier_id} physical_node_id must be non-empty")
            if materializer_unit_id not in by_id:
                raise ValueError(f"barrier {barrier_id} references missing materializer unit {materializer_unit_id}")
            materializer = by_id[materializer_unit_id]
            if materializer.backend != "ray_worker":
                raise ValueError(f"barrier {barrier_id} materializer must be a native fragment unit")
            expected_materializer_physical_node_id = f"node:{physical_node_id}:native-fragment"
            if materializer.physical_node_id != expected_materializer_physical_node_id:
                raise ValueError(
                    f"barrier {barrier_id} physical node {physical_node_id!r} does not match "
                    f"materializer {materializer_unit_id} physical node {materializer.physical_node_id!r}"
                )
            materialized_input_unit_ids = tuple(
                str(resource_unit_id).strip() for resource_unit_id in barrier.materialized_input_unit_ids
            )
            if not materialized_input_unit_ids:
                raise ValueError(f"barrier {barrier_id} must materialize at least one input unit")
            if len(set(materialized_input_unit_ids)) != len(materialized_input_unit_ids):
                raise ValueError(f"barrier {barrier_id} has duplicate materialized input units")
            materializer_inputs = set(materializer.input_unit_ids)
            for input_unit_id in materialized_input_unit_ids:
                if input_unit_id not in materializer_inputs:
                    raise ValueError(
                        f"barrier {barrier_id} materialized input {input_unit_id} "
                        f"is not a direct input of {materializer_unit_id}"
                    )
            if barrier_id in barrier_ids:
                raise ValueError(f"duplicate materialization barrier_id: {barrier_id}")
            if physical_node_id in barrier_nodes:
                raise ValueError(f"duplicate materialization barrier physical_node_id: {physical_node_id}")
            if materializer_unit_id in barrier_units:
                raise ValueError(f"duplicate materialization barrier unit: {materializer_unit_id}")
            barrier_ids.add(barrier_id)
            barrier_nodes.add(physical_node_id)
            barrier_units.add(materializer_unit_id)

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
        process_resources = unit.resident_per_actor if unit.backend == "ray_actor" else _hard_resources(unit.per_task)
        if unit.backend == "ray_worker":
            if not process_resources.is_zero():
                raise ValueError(f"native fragment unit {unit.resource_unit_id} process resources are owned by DuckDB")
        else:
            if process_resources.cpu <= 0 and process_resources.gpu <= 0:
                raise ValueError(f"unit {unit.resource_unit_id} process commitment must request CPU or GPU resources")

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

    def barrier_for_physical_node(self, physical_node_id: str) -> MaterializationBarrierSpec:
        key = str(physical_node_id)
        for barrier in self.materialization_barriers:
            if barrier.physical_node_id == key:
                return barrier
        raise KeyError(f"unknown materialization barrier physical_node_id {key!r}")

    def ordered_materialization_barriers(self) -> tuple[MaterializationBarrierSpec, ...]:
        rank = {resource_unit_id: index for index, resource_unit_id in enumerate(self.topological_unit_ids())}
        return tuple(
            sorted(
                self.materialization_barriers,
                key=lambda barrier: (rank[barrier.materializer_unit_id], barrier.barrier_id),
            )
        )

    def eligible_resource_unit_ids(self, completed_barrier_ids: set[str] | frozenset[str]) -> tuple[str, ...]:
        """Return units not hidden behind an unfinished true barrier.

        The returned set is the current execution phase, not the set of tasks
        that happen to be running.  Parallel branches up to their respective
        first unfinished barriers remain eligible together.  Once a barrier
        completes, reverse traversal stops at that boundary so the preceding
        phase is retired instead of being reserved again.
        """

        completed = {str(barrier_id) for barrier_id in completed_barrier_ids}
        ordered = self.topological_unit_ids()
        by_id = {unit.resource_unit_id: unit for unit in self.units}
        barrier_by_materializer = {barrier.materializer_unit_id: barrier for barrier in self.materialization_barriers}
        pending = tuple(
            barrier for barrier in self.ordered_materialization_barriers() if barrier.barrier_id not in completed
        )
        pending_materializer_unit_ids = {barrier.materializer_unit_id for barrier in pending}

        direct_downstream: dict[str, set[str]] = {resource_unit_id: set() for resource_unit_id in ordered}
        for unit in self.units:
            for input_unit_id in unit.input_unit_ids:
                direct_downstream[input_unit_id].add(unit.resource_unit_id)
        blocked: set[str] = set()
        todo = [
            downstream_unit_id
            for barrier in pending
            for downstream_unit_id in direct_downstream[barrier.materializer_unit_id]
        ]
        while todo:
            resource_unit_id = todo.pop()
            if resource_unit_id in blocked:
                continue
            blocked.add(resource_unit_id)
            todo.extend(direct_downstream[resource_unit_id])

        # Every unfinished barrier whose materialized side is reachable is a
        # target for the current phase.  The materializer itself can be
        # structurally downstream of another barrier through a deferred input
        # (for example, a broadcast build whose probe side contains a sort),
        # while its independent materialized side is already executable.
        # Treating the whole materializer as blocked would serialize those two
        # materializations and would not match the task streams spawned by the
        # distributed executor.
        targets = {
            barrier.materializer_unit_id
            for barrier in pending
            if not any(
                input_unit_id in blocked or input_unit_id in pending_materializer_unit_ids
                for input_unit_id in barrier.materialized_input_unit_ids
            )
        }
        # A terminal is also a target when no unfinished barrier withholds it.
        targets.update(
            terminal_unit_id for terminal_unit_id in self.terminal_unit_ids if terminal_unit_id not in blocked
        )
        # A pending barrier on one branch can also block a downstream join fed
        # by a branch whose own barrier has already completed.  Keep the
        # unblocked side of every blocked edge eligible so that branch can
        # stream up to the join's bounded input instead of being retired until
        # the other branch catches up.
        for blocked_unit_id in blocked:
            unit = by_id[blocked_unit_id]
            barrier = barrier_by_materializer.get(blocked_unit_id)
            if barrier is not None and barrier.barrier_id not in completed:
                # A pending barrier's pre-materialization work is represented
                # by targeting the materializer above.  Its deferred inputs do
                # not execute until the barrier completes.
                continue
            retired_inputs = set() if barrier is None else set(barrier.materialized_input_unit_ids)
            targets.update(
                input_unit_id
                for input_unit_id in unit.input_unit_ids
                if input_unit_id not in blocked and input_unit_id not in retired_inputs
            )

        eligible: set[str] = set()
        todo = list(targets)
        while todo:
            resource_unit_id = todo.pop()
            if resource_unit_id in eligible:
                continue
            eligible.add(resource_unit_id)
            unit = by_id[resource_unit_id]
            barrier = barrier_by_materializer.get(resource_unit_id)
            if barrier is None:
                todo.extend(unit.input_unit_ids)
            elif barrier.barrier_id in completed:
                materialized_inputs = set(barrier.materialized_input_unit_ids)
                todo.extend(
                    input_unit_id for input_unit_id in unit.input_unit_ids if input_unit_id not in materialized_inputs
                )
            else:
                todo.extend(barrier.materialized_input_unit_ids)
        return tuple(resource_unit_id for resource_unit_id in ordered if resource_unit_id in eligible)

    def frontier_materialization_barriers(
        self,
        completed_barrier_ids: set[str] | frozenset[str],
    ) -> tuple[MaterializationBarrierSpec, ...]:
        """Return all unfinished barriers on the current parallel frontier."""

        completed = {str(barrier_id) for barrier_id in completed_barrier_ids}
        eligible = set(self.eligible_resource_unit_ids(completed))
        return tuple(
            barrier
            for barrier in self.ordered_materialization_barriers()
            if barrier.barrier_id not in completed and barrier.materializer_unit_id in eligible
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
        eligible_unit_ids: tuple[str, ...] | None = None,
    ) -> None:
        eligible = (
            {unit.resource_unit_id for unit in self.units}
            if eligible_unit_ids is None
            else {str(resource_unit_id) for resource_unit_id in eligible_unit_ids}
        )
        unknown = sorted(eligible - {unit.resource_unit_id for unit in self.units})
        if unknown:
            raise ValueError(f"allocation validation references unknown eligible unit: {unknown[0]}")
        hard_capacity = _hard_resources(allocation.resources)
        for unit in self.units:
            task_commitment = _hard_resources(unit.per_task)
            if unit.backend == "ray_task" and unit.resource_unit_id in eligible:
                exceeded = task_commitment.exceeded_dimensions(hard_capacity)
                if exceeded:
                    raise ValueError(
                        f"unit {unit.resource_unit_id} maximum task exceeds query allocation for {', '.join(exceeded)}"
                    )
                if not task_commitment.is_zero() and not any(
                    task_commitment.fits_within(_hard_resources(node_allocation.resources))
                    for node_allocation in allocation.node_allocations
                ):
                    raise ValueError(f"unit {unit.resource_unit_id} maximum task does not fit any allocated Ray node")

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
            "materialization_barriers": [barrier.to_dict() for barrier in self.materialization_barriers],
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
            materialization_barriers=tuple(
                MaterializationBarrierSpec.from_dict(item) for item in values["materialization_barriers"]
            ),
            terminal_unit_ids=tuple(str(item) for item in values["terminal_unit_ids"]),
        )


__all__ = [
    "MaterializationBarrierSpec",
    "NodeResourceAllocation",
    "QueryAllocation",
    "QueryResourceGraph",
    "ResourceVector",
    "ResourceUnitSpec",
]
