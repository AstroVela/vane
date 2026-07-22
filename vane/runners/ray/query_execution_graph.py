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
    """Resources owned by a query, stage, task, or output window.

    CPU and GPU are Ray logical resources and may be fractional. Byte resources
    are exact integer ownership quantities. A vector is never allowed to carry
    negative capacity; subtraction that would underflow is a control-plane bug.
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
        return ResourceVector(**values)

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
    """One query's hard resource ownership on one concrete Ray node."""

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

    stage_id: str
    actor_index: int
    node_id: str

    _FIELDS: ClassVar[tuple[str, ...]] = ("stage_id", "actor_index", "node_id")

    def __post_init__(self) -> None:
        stage_id = str(self.stage_id).strip()
        node_id = str(self.node_id).strip()
        actor_index = int(self.actor_index)
        if not stage_id:
            raise ValueError("actor placement stage_id must be non-empty")
        if actor_index < 0:
            raise ValueError("actor placement actor_index must be >= 0")
        if not node_id:
            raise ValueError("actor placement node_id must be non-empty")
        object.__setattr__(self, "stage_id", stage_id)
        object.__setattr__(self, "actor_index", actor_index)
        object.__setattr__(self, "node_id", node_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "actor_index": self.actor_index,
            "node_id": self.node_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ActorPlacement:
        values = dict(payload)
        _strict_fields(values, cls._FIELDS, cls.__name__)
        return cls(
            stage_id=str(values["stage_id"]),
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
        placement_keys = [(placement.stage_id, placement.actor_index) for placement in actor_placements]
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

    def actor_node_ids_for_stage(self, stage_id: str) -> tuple[str, ...]:
        stage_key = str(stage_id)
        placements = sorted(
            (placement for placement in self.actor_placements if placement.stage_id == stage_key),
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
class StageResourceSpec:
    query_id: str
    stage_id: str
    physical_node_id: str
    stage_kind: str
    backend: str
    input_stage_ids: tuple[str, ...]
    per_task: ResourceVector
    target_output_block_bytes: int
    generator_buffer_blocks: int
    max_concurrency: int | None
    resident_per_actor: ResourceVector = field(default_factory=ResourceVector)
    actor_min_size: int = 0
    actor_max_size: int = 0
    actor_prefetch_depth: int = 1
    spill_mode: str = "streaming"

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "query_id",
        "stage_id",
        "physical_node_id",
        "stage_kind",
        "backend",
        "input_stage_ids",
        "per_task",
        "target_output_block_bytes",
        "generator_buffer_blocks",
        "max_concurrency",
        "resident_per_actor",
        "actor_min_size",
        "actor_max_size",
        "actor_prefetch_depth",
        "spill_mode",
    )

    @property
    def output_window_bytes(self) -> int:
        return int(self.target_output_block_bytes) * int(self.generator_buffer_blocks)

    @property
    def is_ray_process(self) -> bool:
        return self.backend in {"ray_task", "ray_actor", "ray_worker"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "stage_id": self.stage_id,
            "physical_node_id": self.physical_node_id,
            "stage_kind": self.stage_kind,
            "backend": self.backend,
            "input_stage_ids": list(self.input_stage_ids),
            "per_task": self.per_task.to_dict(),
            "target_output_block_bytes": int(self.target_output_block_bytes),
            "generator_buffer_blocks": int(self.generator_buffer_blocks),
            "max_concurrency": None if self.max_concurrency is None else int(self.max_concurrency),
            "resident_per_actor": self.resident_per_actor.to_dict(),
            "actor_min_size": int(self.actor_min_size),
            "actor_max_size": int(self.actor_max_size),
            "actor_prefetch_depth": int(self.actor_prefetch_depth),
            "spill_mode": self.spill_mode,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StageResourceSpec:
        values = dict(payload)
        _strict_fields(values, cls._FIELDS, cls.__name__)
        max_concurrency = values["max_concurrency"]
        return cls(
            query_id=str(values["query_id"]),
            stage_id=str(values["stage_id"]),
            physical_node_id=str(values["physical_node_id"]),
            stage_kind=str(values["stage_kind"]),
            backend=str(values["backend"]),
            input_stage_ids=tuple(str(item) for item in values["input_stage_ids"]),
            per_task=ResourceVector.from_dict(values["per_task"]),
            target_output_block_bytes=int(values["target_output_block_bytes"]),
            generator_buffer_blocks=int(values["generator_buffer_blocks"]),
            max_concurrency=None if max_concurrency is None else int(max_concurrency),
            resident_per_actor=ResourceVector.from_dict(values["resident_per_actor"]),
            actor_min_size=int(values["actor_min_size"]),
            actor_max_size=int(values["actor_max_size"]),
            actor_prefetch_depth=int(values["actor_prefetch_depth"]),
            spill_mode=str(values["spill_mode"]),
        )


@dataclass(frozen=True)
class QueryExecutionGraph:
    query_id: str
    plan_digest: str
    stages: tuple[StageResourceSpec, ...]
    terminal_stage_ids: tuple[str, ...]

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "query_id",
        "plan_digest",
        "stages",
        "terminal_stage_ids",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", str(self.query_id).strip())
        object.__setattr__(self, "plan_digest", str(self.plan_digest).strip())
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "terminal_stage_ids", tuple(str(item) for item in self.terminal_stage_ids))
        self._validate()

    def _validate(self) -> None:
        if not self.query_id:
            raise ValueError("query_id must be non-empty")
        if not self.plan_digest:
            raise ValueError("plan_digest must be non-empty")
        if not self.stages:
            raise ValueError("query graph must contain at least one stage")
        if not self.terminal_stage_ids:
            raise ValueError("query graph must contain at least one terminal stage")

        by_id: dict[str, StageResourceSpec] = {}
        physical_nodes: dict[str, str] = {}
        for stage in self.stages:
            self._validate_stage(stage)
            if stage.stage_id in by_id:
                raise ValueError(f"duplicate stage_id: {stage.stage_id}")
            if stage.physical_node_id in physical_nodes:
                raise ValueError(
                    "duplicate physical_node_id: "
                    f"{stage.physical_node_id} used by {physical_nodes[stage.physical_node_id]} and {stage.stage_id}"
                )
            by_id[stage.stage_id] = stage
            physical_nodes[stage.physical_node_id] = stage.stage_id

        if len(set(self.terminal_stage_ids)) != len(self.terminal_stage_ids):
            raise ValueError("terminal_stage_ids must be unique")
        for terminal in self.terminal_stage_ids:
            if terminal not in by_id:
                raise ValueError(f"terminal stage is not registered: {terminal}")

        downstream: dict[str, set[str]] = {stage_id: set() for stage_id in by_id}
        for stage in self.stages:
            if len(set(stage.input_stage_ids)) != len(stage.input_stage_ids):
                raise ValueError(f"stage {stage.stage_id} has duplicate input_stage_ids")
            for input_stage_id in stage.input_stage_ids:
                if input_stage_id not in by_id:
                    raise ValueError(f"stage {stage.stage_id} references missing input stage {input_stage_id}")
                if input_stage_id == stage.stage_id:
                    raise ValueError(f"query graph contains a cycle at stage {stage.stage_id}")
                downstream[input_stage_id].add(stage.stage_id)

        ordered = self._topological_order(by_id, downstream)
        if len(ordered) != len(by_id):
            raise ValueError("query graph contains a cycle")

        for terminal in self.terminal_stage_ids:
            if downstream[terminal]:
                raise ValueError(
                    f"terminal stage {terminal} has downstream stages: {', '.join(sorted(downstream[terminal]))}"
                )

        reaches_terminal = set(self.terminal_stage_ids)
        for stage_id in reversed(ordered):
            if any(child in reaches_terminal for child in downstream[stage_id]):
                reaches_terminal.add(stage_id)
        missing_terminal_path = sorted(set(by_id) - reaches_terminal)
        if missing_terminal_path:
            raise ValueError(f"stage {missing_terminal_path[0]} does not reach a terminal stage")

    def _validate_stage(self, stage: StageResourceSpec) -> None:
        if str(stage.query_id).strip() != self.query_id:
            raise ValueError(
                f"stage {stage.stage_id or '<empty>'} query_id {stage.query_id!r} does not match {self.query_id!r}"
            )
        if not str(stage.stage_id).strip():
            raise ValueError("stage_id must be non-empty")
        if not str(stage.stage_id).startswith("stage:"):
            raise ValueError(f"stage_id must use stable 'stage:' identity: {stage.stage_id}")
        if not str(stage.physical_node_id).strip():
            raise ValueError(f"stage {stage.stage_id} physical_node_id must be non-empty")
        if not str(stage.stage_kind).strip():
            raise ValueError(f"stage {stage.stage_id} stage_kind must be non-empty")
        if not str(stage.backend).strip():
            raise ValueError(f"stage {stage.stage_id} backend must be non-empty")
        if int(stage.target_output_block_bytes) < 0:
            raise ValueError(f"stage {stage.stage_id} target_output_block_bytes must be >= 0")
        if int(stage.generator_buffer_blocks) < 0:
            raise ValueError(f"stage {stage.stage_id} generator_buffer_blocks must be >= 0")
        target = int(stage.target_output_block_bytes)
        blocks = int(stage.generator_buffer_blocks)
        if target == 0 and blocks != 0:
            raise ValueError(
                f"stage {stage.stage_id} target_output_block_bytes and generator_buffer_blocks must both be zero"
            )
        if target > 0 and blocks <= 0:
            raise ValueError(
                f"stage {stage.stage_id} target_output_block_bytes and generator_buffer_blocks must both be positive"
            )
        if stage.max_concurrency is not None and int(stage.max_concurrency) <= 0:
            raise ValueError(f"stage {stage.stage_id} max_concurrency must be > 0")
        if stage.spill_mode not in {"streaming", "barrier"}:
            raise ValueError(f"stage {stage.stage_id} has invalid spill_mode {stage.spill_mode!r}")

        process_resources = stage.resident_per_actor if stage.backend == "ray_actor" else stage.per_task
        if stage.is_ray_process:
            if process_resources.cpu <= 0 and process_resources.gpu <= 0:
                raise ValueError(f"Ray stage {stage.stage_id} must request CPU or GPU resources")
            if process_resources.heap_bytes <= 0:
                raise ValueError(f"Ray stage {stage.stage_id} must request non-zero heap_bytes")

        actor_min = int(stage.actor_min_size)
        actor_max = int(stage.actor_max_size)
        actor_prefetch_depth = int(stage.actor_prefetch_depth)
        if stage.backend == "ray_actor":
            if actor_min <= 0:
                raise ValueError(f"ray_actor stage {stage.stage_id} actor_min_size must be > 0")
            if actor_max < actor_min:
                raise ValueError(f"ray_actor stage {stage.stage_id} actor_max_size must be >= actor_min_size")
            if actor_prefetch_depth <= 0:
                raise ValueError(f"ray_actor stage {stage.stage_id} actor_prefetch_depth must be > 0")
            if stage.max_concurrency is not None:
                raise ValueError(f"ray_actor stage {stage.stage_id} concurrency is owned by concrete actor slots")
            if stage.per_task.cpu or stage.per_task.gpu or stage.per_task.heap_bytes:
                raise ValueError(
                    f"ray_actor stage {stage.stage_id} invocation resources may only contain object-store bytes"
                )
        elif actor_min != 0 or actor_max != 0:
            raise ValueError(f"actor bounds are only valid for ray_actor stages: {stage.stage_id}")
        elif actor_prefetch_depth != 1:
            raise ValueError(f"actor_prefetch_depth is only configurable for ray_actor stages: {stage.stage_id}")
        elif not stage.resident_per_actor.is_zero():
            raise ValueError(f"resident_per_actor is only valid for ray_actor stages: {stage.stage_id}")
        if stage.backend == "ray_task" and stage.max_concurrency is not None:
            raise ValueError(f"ray_task stage {stage.stage_id} concurrency is owned by resource credit")

    @staticmethod
    def _topological_order(
        by_id: Mapping[str, StageResourceSpec],
        downstream: Mapping[str, set[str]],
    ) -> tuple[str, ...]:
        indegree = {stage_id: len(stage.input_stage_ids) for stage_id, stage in by_id.items()}
        ready = [stage_id for stage_id, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        ordered: list[str] = []
        while ready:
            stage_id = heapq.heappop(ready)
            ordered.append(stage_id)
            for child in sorted(downstream[stage_id]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, child)
        return tuple(ordered)

    def stage_by_id(self, stage_id: str) -> StageResourceSpec:
        key = str(stage_id)
        for stage in self.stages:
            if stage.stage_id == key:
                return stage
        raise KeyError(f"unknown stage_id {key!r}")

    def stage_id_for_physical_node(self, physical_node_id: str) -> str:
        key = str(physical_node_id)
        for stage in self.stages:
            if stage.physical_node_id == key:
                return stage.stage_id
        raise KeyError(f"unknown physical_node_id {key!r}")

    def topological_stage_ids(self) -> tuple[str, ...]:
        by_id = {stage.stage_id: stage for stage in self.stages}
        downstream: dict[str, set[str]] = {stage_id: set() for stage_id in by_id}
        for stage in self.stages:
            for parent in stage.input_stage_ids:
                downstream[parent].add(stage.stage_id)
        return self._topological_order(by_id, downstream)

    def reverse_topological_stage_ids(self) -> tuple[str, ...]:
        return tuple(reversed(self.topological_stage_ids()))

    def downstream_fte_stage_ids_requiring_separate_slot(
        self,
        source_stage_id: str,
    ) -> tuple[str, ...]:
        """Return downstream FTE stages separated by a non-FTE boundary.

        An FTE lease can normally hand its capacity directly to a downstream
        FTE stage after it finishes.  That capacity is not transferable while
        the FTE is blocked inside a nested Ray task or actor invocation: the
        producer remains alive until the non-FTE continuation returns.  Every
        FTE reachable after such a boundary therefore shares one additional
        progress slot that admission must keep placeable.

        The source itself counts as a boundary when it is not a Ray worker.
        Results follow the graph's deterministic topological order.  For a
        join, one boundary-crossing input path is sufficient to require the
        separate slot.
        """
        source = self.stage_by_id(source_stage_id)
        by_id = {stage.stage_id: stage for stage in self.stages}
        downstream: dict[str, set[str]] = {stage_id: set() for stage_id in by_id}
        for stage in self.stages:
            for input_stage_id in stage.input_stage_ids:
                downstream[input_stage_id].add(stage.stage_id)

        crossed_non_fte: dict[str, bool] = {source.stage_id: source.backend != "ray_worker"}
        ordered = self.topological_stage_ids()
        for stage_id in ordered:
            crossed = crossed_non_fte.get(stage_id)
            if crossed is None:
                continue
            for child_id in downstream[stage_id]:
                child_crossed = crossed or by_id[child_id].backend != "ray_worker"
                crossed_non_fte[child_id] = crossed_non_fte.get(child_id, False) or child_crossed

        return tuple(
            stage_id
            for stage_id in ordered
            if stage_id != source.stage_id
            and by_id[stage_id].backend == "ray_worker"
            and crossed_non_fte.get(stage_id, False)
        )

    def task_identity(self, stage_id: str, *, partition_id: int | str, attempt_id: int | str) -> str:
        stage = self.stage_by_id(stage_id)
        partition = str(partition_id).strip()
        attempt = str(attempt_id).strip()
        if not partition:
            raise ValueError("partition_id must be non-empty")
        if not attempt:
            raise ValueError("attempt_id must be non-empty")
        return f"task:{stage.stage_id}:partition:{partition}:attempt:{attempt}"

    def validate_allocation(
        self,
        allocation: QueryAllocation,
        *,
        require_full_minimum: bool = True,
    ) -> None:
        capacity = allocation.resources
        for stage in self.stages:
            task_commitment = ResourceVector(
                cpu=stage.per_task.cpu,
                gpu=stage.per_task.gpu,
                heap_bytes=stage.per_task.heap_bytes,
                object_store_bytes=(stage.per_task.object_store_bytes + stage.output_window_bytes),
            )
            actor_resident = stage.resident_per_actor if stage.backend == "ray_actor" else ResourceVector()
            placement_commitment = actor_resident + task_commitment
            output_window = stage.output_window_bytes
            if require_full_minimum and capacity.object_store_bytes > 0 and output_window > capacity.object_store_bytes:
                raise ValueError(
                    f"stage {stage.stage_id} output window {output_window} exceeds query object-store allocation "
                    f"{capacity.object_store_bytes}"
                )
            total_object_store = stage.per_task.object_store_bytes + output_window
            if (
                require_full_minimum
                and capacity.object_store_bytes > 0
                and total_object_store > capacity.object_store_bytes
            ):
                raise ValueError(
                    f"stage {stage.stage_id} retained input plus output window {total_object_store} exceeds "
                    f"query object-store allocation {capacity.object_store_bytes}"
                )
            exceeded = placement_commitment.exceeded_dimensions(capacity)
            if require_full_minimum and exceeded:
                raise ValueError(
                    f"stage {stage.stage_id} maximum task exceeds query allocation for {', '.join(exceeded)}"
                )
            if (
                require_full_minimum
                and not task_commitment.is_zero()
                and not any(
                    placement_commitment.fits_within(node_allocation.resources)
                    for node_allocation in allocation.node_allocations
                )
            ):
                raise ValueError(f"stage {stage.stage_id} maximum task does not fit any allocated Ray node")
            if stage.backend == "ray_actor":
                actor_minimum = placement_commitment.scale(stage.actor_min_size)
                exceeded_actor = actor_minimum.exceeded_dimensions(capacity)
                if require_full_minimum and exceeded_actor:
                    raise ValueError(
                        f"stage {stage.stage_id} actor minimum exceeds query allocation for {', '.join(exceeded_actor)}"
                    )
                placements = [
                    placement for placement in allocation.actor_placements if placement.stage_id == stage.stage_id
                ]
                if require_full_minimum and len(placements) != stage.actor_min_size:
                    raise ValueError(f"stage {stage.stage_id} requires exactly {stage.actor_min_size} actor placements")
                expected_indices = set(range(stage.actor_min_size))
                placement_indices = {placement.actor_index for placement in placements}
                if placements and placement_indices != expected_indices:
                    raise ValueError(f"stage {stage.stage_id} actor placement indices must be contiguous from zero")
                for placement in placements:
                    if not placement_commitment.fits_within(allocation.resources_for_node(placement.node_id)):
                        raise ValueError(
                            f"stage {stage.stage_id} actor {placement.actor_index} does not fit "
                            f"allocated Ray node {placement.node_id}"
                        )

        known_stage_ids = {stage.stage_id for stage in self.stages if stage.backend == "ray_actor"}
        unknown_actor_stages = sorted(
            {placement.stage_id for placement in allocation.actor_placements} - known_stage_ids
        )
        if unknown_actor_stages:
            raise ValueError("actor placement references non-actor stage: " + ", ".join(unknown_actor_stages))
        if require_full_minimum:
            actor_commitment_by_node: dict[str, ResourceVector] = {}
            for placement in allocation.actor_placements:
                stage = self.stage_by_id(placement.stage_id)
                invocation = ResourceVector(
                    object_store_bytes=(stage.per_task.object_store_bytes + stage.output_window_bytes)
                )
                actor_commitment_by_node[placement.node_id] = (
                    actor_commitment_by_node.get(
                        placement.node_id,
                        ResourceVector(),
                    )
                    + stage.resident_per_actor
                    + invocation
                )
            for node_id, commitment in actor_commitment_by_node.items():
                if not commitment.fits_within(allocation.resources_for_node(node_id)):
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
            "stages": [stage.to_dict() for stage in self.stages],
            "terminal_stage_ids": list(self.terminal_stage_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> QueryExecutionGraph:
        values = dict(payload)
        _strict_fields(values, cls._FIELDS, cls.__name__)
        return cls(
            query_id=str(values["query_id"]),
            plan_digest=str(values["plan_digest"]),
            stages=tuple(StageResourceSpec.from_dict(item) for item in values["stages"]),
            terminal_stage_ids=tuple(str(item) for item in values["terminal_stage_ids"]),
        )


__all__ = [
    "ActorPlacement",
    "NodeResourceAllocation",
    "QueryAllocation",
    "QueryExecutionGraph",
    "ResourceVector",
    "StageResourceSpec",
]
