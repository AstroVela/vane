# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from vane.execution.resource_graph_metadata import (
    _node_sort_key,
    _normalize_metadata,
    _positive_int,
    materialization_barrier_id_for_node,
)
from vane.execution.resource_graph_metadata import (
    native_fragment_unit_id_for_fragment as native_fragment_unit_id_for_fragment,
)
from vane.execution.resource_graph_metadata import (
    native_fragment_unit_id_for_node as native_fragment_unit_id_for_node,
)
from vane.execution.resource_graph_metadata import (
    udf_unit_id_for_node as udf_unit_id_for_node,
)
from vane.execution.resources import udf_process_resources
from vane.runners.ray.cluster_resource_coordinator import NodeCapacity, QueryDemand
from vane.runners.ray.query_resource_graph import (
    MaterializationBarrierSpec,
    QueryResourceGraph,
    ResourceUnitSpec,
    ResourceVector,
)

_DEFAULT_TARGET_OUTPUT_BLOCK_BYTES = 128 * 1024**2
_DEFAULT_RAY_ACTOR_PREFETCH_DEPTH = 2
_GENERATOR_BUFFER_BLOCKS = 2


def _resource_dimension(resources: ResourceVector, field_name: str) -> int | float:
    value = getattr(resources, field_name)
    if not isinstance(value, (int, float)):
        raise TypeError(f"resource field {field_name!r} must be numeric, got {value!r}")
    return value


def _env_positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    return int(default) if raw is None or not str(raw).strip() else _positive_int(raw, name)


def _udf_unit(
    query_id: str,
    node: Mapping[str, Any],
    input_unit_id: str,
    env: Mapping[str, str],
) -> ResourceUnitSpec | None:
    payload = node["udf_payload"]
    if payload is None:
        return None
    backend = str(payload.get("execution_backend") or "").strip()
    if backend not in {"ray_task", "ray_actor"}:
        return None
    node_id = str(node["node_id"])
    expected_unit_id = udf_unit_id_for_node(query_id, node_id)
    actual_unit_id = str(payload.get("resource_unit_id") or "").strip()
    if not actual_unit_id:
        raise ValueError(f"Ray UDF node {node_id} is missing pre-registered resource_unit_id")
    if actual_unit_id != expected_unit_id:
        raise ValueError(
            f"Ray UDF node {node_id} resource_unit_id mismatch: got {actual_unit_id!r}, expected {expected_unit_id!r}"
        )
    payload_query_id = str(payload.get("query_id") or "").strip()
    if payload_query_id and payload_query_id != query_id:
        raise ValueError(f"Ray UDF node {node_id} query_id mismatch: got {payload_query_id!r}, expected {query_id!r}")
    process_resources = udf_process_resources(payload)
    target = _positive_int(
        payload.get(
            "udf_output_target_max_bytes",
            _env_positive_int(env, "VANE_TARGET_OUTPUT_BLOCK_BYTES", _DEFAULT_TARGET_OUTPUT_BLOCK_BYTES),
        ),
        "udf_output_target_max_bytes",
    )
    input_window = _positive_int(
        payload.get(
            "udf_task_input_max_bytes",
            _env_positive_int(env, "VANE_TARGET_OUTPUT_BLOCK_BYTES", _DEFAULT_TARGET_OUTPUT_BLOCK_BYTES),
        ),
        "udf_task_input_max_bytes",
    )
    if backend == "ray_actor":
        actor_size = _positive_int(payload.get("actor_pool_size"), "actor_pool_size")
        actor_prefetch_depth = _env_positive_int(
            env,
            "VANE_RAY_ACTOR_PREFETCH_DEPTH",
            _DEFAULT_RAY_ACTOR_PREFETCH_DEPTH,
        )
        max_concurrency = None
        actor_pool_size = actor_size
        resident_per_actor = process_resources
        invocation_resources = ResourceVector(object_store_bytes=input_window)
    else:
        max_concurrency = None
        actor_pool_size = 0
        actor_prefetch_depth = 1
        resident_per_actor = ResourceVector()
        invocation_resources = process_resources + ResourceVector(object_store_bytes=input_window)
    return ResourceUnitSpec(
        query_id=query_id,
        resource_unit_id=expected_unit_id,
        physical_node_id=f"node:{node_id}:udf",
        unit_kind="ray_actor_pool" if backend == "ray_actor" else "ray_task_udf",
        backend=backend,
        input_unit_ids=(input_unit_id,),
        per_task=invocation_resources,
        target_output_block_bytes=target,
        generator_buffer_blocks=_GENERATOR_BUFFER_BLOCKS,
        max_concurrency=max_concurrency,
        resident_per_actor=resident_per_actor,
        actor_pool_size=actor_pool_size,
        actor_prefetch_depth=actor_prefetch_depth,
    )


def build_query_resource_graph(
    metadata: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
) -> QueryResourceGraph:
    environment = os.environ if env is None else env
    query_id, nodes, terminal_node_ids = _normalize_metadata(metadata)
    native_fragment_target = _env_positive_int(
        environment,
        "VANE_TARGET_OUTPUT_BLOCK_BYTES",
        _DEFAULT_TARGET_OUTPUT_BLOCK_BYTES,
    )

    output_unit_by_node: dict[str, str] = {}
    for node_id, node in nodes.items():
        udf_payload = node["udf_payload"]
        has_remote_udf = udf_payload is not None and str(udf_payload.get("execution_backend") or "").strip() in {
            "ray_task",
            "ray_actor",
        }
        output_unit_by_node[node_id] = (
            udf_unit_id_for_node(query_id, node_id)
            if has_remote_udf
            else native_fragment_unit_id_for_node(query_id, node_id)
        )

    units: list[ResourceUnitSpec] = []
    barriers: list[MaterializationBarrierSpec] = []
    for node_id in sorted(nodes, key=_node_sort_key):
        node = nodes[node_id]
        native_fragment_unit_id = native_fragment_unit_id_for_node(query_id, node_id)
        input_unit_ids = tuple(output_unit_by_node[parent] for parent in node["input_node_ids"])
        is_sink = bool(node["is_sink"])
        units.append(
            ResourceUnitSpec(
                query_id=query_id,
                resource_unit_id=native_fragment_unit_id,
                physical_node_id=f"node:{node_id}:native-fragment",
                unit_kind="native_fragment",
                backend="ray_worker",
                input_unit_ids=input_unit_ids,
                # All native fragments on a Ray node execute inside one shared
                # DuckDB DatabaseInstance. Its TaskScheduler, BufferManager,
                # TemporaryMemoryManager, and memory_limit own native process
                # resources; Vane only accounts cross-process object flow.
                per_task=ResourceVector(),
                target_output_block_bytes=0 if is_sink else native_fragment_target,
                generator_buffer_blocks=0 if is_sink else _GENERATOR_BUFFER_BLOCKS,
                max_concurrency=int(node["num_partitions"]),
            )
        )
        if bool(node["is_materialization_barrier"]):
            barriers.append(
                MaterializationBarrierSpec(
                    query_id=query_id,
                    barrier_id=materialization_barrier_id_for_node(query_id, node_id),
                    physical_node_id=node_id,
                    materializer_unit_id=native_fragment_unit_id,
                    materialized_input_unit_ids=tuple(
                        output_unit_by_node[parent] for parent in node["materialized_input_node_ids"]
                    ),
                )
            )
        udf_unit = _udf_unit(
            query_id,
            node,
            native_fragment_unit_id,
            environment,
        )
        if udf_unit is not None:
            units.append(udf_unit)

    terminals = tuple(output_unit_by_node[node_id] for node_id in terminal_node_ids)
    preliminary = QueryResourceGraph(
        query_id=query_id,
        plan_digest="sha256:pending",
        units=tuple(units),
        terminal_unit_ids=terminals,
        materialization_barriers=tuple(barriers),
    )
    return QueryResourceGraph(
        query_id=query_id,
        plan_digest=preliminary.normalized_digest(),
        units=preliminary.units,
        terminal_unit_ids=preliminary.terminal_unit_ids,
        materialization_barriers=preliminary.materialization_barriers,
    )


def _task_scheduling_request(unit: ResourceUnitSpec) -> ResourceVector:
    """Return the concrete Ray Core request for one task invocation.

    Retained inputs and generator output windows are spillable pipeline data.
    QRM accounts them dynamically instead of copying them into the process
    request or a query-level placement model.
    """
    return ResourceVector(
        cpu=unit.per_task.cpu,
        gpu=unit.per_task.gpu,
        heap_bytes=unit.per_task.heap_bytes,
    )


def _sum_node_capacities(node_capacities: Sequence[NodeCapacity]) -> ResourceVector:
    total = ResourceVector()
    for node in node_capacities:
        total = total + node.resources
    return total


def build_query_demand(
    graph: QueryResourceGraph,
    node_capacities: Sequence[NodeCapacity],
    *,
    eligible_unit_ids: tuple[str, ...] | None = None,
    weight: float = 1.0,
    priority: int = 0,
) -> QueryDemand:
    nodes = tuple(node_capacities)
    node_ids = [node.node_id for node in nodes]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("node_capacities contains duplicate Ray node IDs")
    cluster_capacity = _sum_node_capacities(nodes)
    eligible = set(graph.eligible_resource_unit_ids(set()) if eligible_unit_ids is None else eligible_unit_ids)
    unknown_eligible = sorted(eligible - {unit.resource_unit_id for unit in graph.units})
    if unknown_eligible:
        raise ValueError(f"query demand references unknown eligible unit: {unknown_eligible[0]}")
    ray_tasks: list[ResourceVector] = []
    actor_processes: list[ResourceVector] = []
    for unit in graph.units:
        if unit.resource_unit_id not in eligible:
            continue
        commitment = _task_scheduling_request(unit)
        if unit.backend == "ray_task":
            ray_tasks.append(commitment)
        elif unit.backend == "ray_actor":
            actor_processes.extend(unit.resident_per_actor for _actor_index in range(unit.actor_pool_size))
    actor_maximum = ResourceVector()
    for actor_process in actor_processes:
        actor_maximum = actor_maximum + actor_process

    def elastic_target(field_name: str) -> int | float:
        task_uses_dimension = any(_resource_dimension(task, field_name) > 0 for task in ray_tasks)
        if task_uses_dimension:
            elastic = _resource_dimension(cluster_capacity, field_name)
        else:
            elastic = min(
                _resource_dimension(actor_maximum, field_name),
                _resource_dimension(cluster_capacity, field_name),
            )
        return elastic

    desired = ResourceVector(
        # CPU/GPU/declared heap are aggregate soft targets only. Concrete UDF
        # calls carry their real resource shape to Ray Core, including when a
        # current cluster snapshot cannot fit that shape yet.
        cpu=elastic_target("cpu"),
        gpu=elastic_target("gpu"),
        heap_bytes=int(elastic_target("heap_bytes")),
        object_store_bytes=cluster_capacity.object_store_bytes,
    )
    return QueryDemand(
        query_id=graph.query_id,
        desired=desired,
        weight=weight,
        priority=priority,
    )


__all__ = [
    "build_query_demand",
    "build_query_resource_graph",
    "materialization_barrier_id_for_node",
    "native_fragment_unit_id_for_fragment",
    "native_fragment_unit_id_for_node",
    "udf_unit_id_for_node",
]
