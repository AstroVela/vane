# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from vane.execution.resource_graph import (
    MaterializationBarrierSpec as MaterializationBarrierSpec,
)
from vane.execution.resource_graph import (
    ResourceGraph,
    _strict_fields,
)

# Keep the original import path valid for callers and serialized references.
from vane.execution.resources import ResourceVector as ResourceVector

_RESOURCE_UNIT_KIND_BY_BACKEND = {
    "ray_worker": "native_fragment",
    "ray_task": "ray_task_udf",
    "ray_actor": "ray_actor_pool",
}


def _process_resources(resources: ResourceVector) -> ResourceVector:
    """Return the non-spillable portion of a query resource vector."""
    return ResourceVector(
        cpu=resources.cpu,
        gpu=resources.gpu,
        heap_bytes=resources.heap_bytes,
    )


@dataclass(frozen=True)
class QueryAllocation:
    """One driver's aggregate soft budget for a query.

    This is deliberately not a Ray placement claim. Concrete tasks and actors
    carry their real resource requests to Ray Core, which owns node selection,
    pending demand, and autoscaling. Vane uses this vector only to decide when
    to apply query/operator backpressure.
    """

    resources: ResourceVector
    generation: int

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "resources",
        "generation",
    )

    def __post_init__(self) -> None:
        generation = int(self.generation)
        if generation <= 0:
            raise ValueError("generation must be > 0")
        object.__setattr__(self, "generation", generation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "resources": self.resources.to_dict(),
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> QueryAllocation:
        values = dict(payload)
        _strict_fields(values, cls._FIELDS, cls.__name__)
        return cls(
            resources=ResourceVector.from_dict(values["resources"]),
            generation=int(values["generation"]),
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
class QueryResourceGraph(ResourceGraph[ResourceUnitSpec]):
    """Ray resource policy on the shared dependency graph."""

    def _validate_unit(self, unit: ResourceUnitSpec) -> None:
        super()._validate_unit(unit)
        unit_kind = str(unit.unit_kind).strip()
        backend = str(unit.backend).strip()
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
        process_resources = (
            unit.resident_per_actor if unit.backend == "ray_actor" else _process_resources(unit.per_task)
        )
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
    "QueryAllocation",
    "QueryResourceGraph",
    "ResourceVector",
    "ResourceUnitSpec",
]
