# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from duckdb.runners.ray.cluster_resource_coordinator import NodeCapacity, QueryDemand
from duckdb.runners.ray.query_resource_graph import (
    MaterializationBarrierSpec,
    QueryResourceGraph,
    ResourceUnitSpec,
    ResourceVector,
)

_DEFAULT_TARGET_OUTPUT_BLOCK_BYTES = 128 * 1024**2
_DEFAULT_RAY_ACTOR_PREFETCH_DEPTH = 2
_GENERATOR_BUFFER_BLOCKS = 2
_TOP_LEVEL_FIELDS = ("query_id", "nodes", "terminal_node_ids")
_NODE_FIELDS = (
    "node_id",
    "node_name",
    "input_node_ids",
    "is_sink",
    "is_materialization_barrier",
    "materialized_input_node_ids",
    "num_partitions",
    "udf_payload",
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


def _node_sort_key(node_id: str) -> tuple[int, int | str]:
    value = str(node_id)
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _nonnegative_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative number") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return parsed


def _env_positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    return int(default) if raw is None or not str(raw).strip() else _positive_int(raw, name)


def native_fragment_unit_id_for_node(query_id: str, node_id: str | int) -> str:
    query = str(query_id).strip()
    node = str(node_id).strip()
    if not query or not node:
        raise ValueError("query_id and node_id must be non-empty")
    return f"resource:{query}:fragment:node:{node}"


def udf_unit_id_for_node(query_id: str, node_id: str | int) -> str:
    query = str(query_id).strip()
    node = str(node_id).strip()
    if not query or not node:
        raise ValueError("query_id and node_id must be non-empty")
    return f"resource:{query}:udf:node:{node}"


def materialization_barrier_id_for_node(query_id: str, node_id: str | int) -> str:
    query = str(query_id).strip()
    node = str(node_id).strip()
    if not query or not node:
        raise ValueError("query_id and node_id must be non-empty")
    return f"barrier:{query}:node:{node}"


def native_fragment_unit_id_for_fragment(query_id: str, fragment_id: str) -> str:
    query = str(query_id).strip()
    fragment = str(fragment_id).strip()
    prefix = f"{query}:node:"
    if not fragment.startswith(prefix):
        if fragment.endswith(":node:") or ":node:" not in fragment:
            raise ValueError(f"invalid native fragment_id: {fragment}")
        raise ValueError(f"fragment {fragment!r} does not belong to query {query!r}")
    node_id = fragment[len(prefix) :]
    if not node_id or ":" in node_id:
        raise ValueError(f"invalid native fragment_id: {fragment}")
    return native_fragment_unit_id_for_node(query, node_id)


def _normalize_metadata(metadata: Mapping[str, Any]) -> tuple[str, dict[str, dict[str, Any]], tuple[str, ...]]:
    payload = dict(metadata)
    _strict_fields(payload, _TOP_LEVEL_FIELDS, "resource unit metadata")
    query_id = str(payload["query_id"]).strip()
    if not query_id:
        raise ValueError("resource unit metadata query_id must be non-empty")
    nodes: dict[str, dict[str, Any]] = {}
    for raw_node in payload["nodes"]:
        node = dict(raw_node)
        _strict_fields(node, _NODE_FIELDS, "resource unit node")
        node_id = str(node["node_id"]).strip()
        if not node_id:
            raise ValueError("resource unit node_id must be non-empty")
        if node_id in nodes:
            raise ValueError(f"duplicate resource unit node_id: {node_id}")
        node["node_id"] = node_id
        node["node_name"] = str(node["node_name"]).strip()
        if not node["node_name"]:
            raise ValueError(f"resource unit node {node_id} node_name must be non-empty")
        node["input_node_ids"] = tuple(str(item).strip() for item in node["input_node_ids"])
        node["num_partitions"] = _positive_int(node["num_partitions"], "num_partitions")
        node["is_sink"] = bool(node["is_sink"])
        node["is_materialization_barrier"] = bool(node["is_materialization_barrier"])
        node["materialized_input_node_ids"] = tuple(str(item).strip() for item in node["materialized_input_node_ids"])
        if len(set(node["materialized_input_node_ids"])) != len(node["materialized_input_node_ids"]):
            raise ValueError(f"resource unit node {node_id} has duplicate materialized input node ids")
        if node["is_materialization_barrier"] != bool(node["materialized_input_node_ids"]):
            raise ValueError(
                f"resource unit node {node_id} must declare materialized inputs "
                "if and only if it is a materialization barrier"
            )
        if node["udf_payload"] is not None and not isinstance(node["udf_payload"], Mapping):
            raise TypeError(f"resource unit node {node_id} udf_payload must be a mapping or None")
        node["udf_payload"] = None if node["udf_payload"] is None else dict(node["udf_payload"])
        nodes[node_id] = node

    for node_id, node in nodes.items():
        for input_node_id in node["input_node_ids"]:
            if input_node_id not in nodes:
                raise ValueError(f"resource unit node {node_id} references missing input node {input_node_id}")
        for input_node_id in node["materialized_input_node_ids"]:
            if input_node_id not in node["input_node_ids"]:
                raise ValueError(
                    f"resource unit node {node_id} materialized input {input_node_id} is not a direct input"
                )
    terminal_node_ids = tuple(str(item).strip() for item in payload["terminal_node_ids"])
    if not terminal_node_ids:
        raise ValueError("resource unit metadata must contain terminal_node_ids")
    for terminal in terminal_node_ids:
        if terminal not in nodes:
            raise ValueError(f"terminal node is not registered: {terminal}")
    return query_id, nodes, tuple(sorted(set(terminal_node_ids), key=_node_sort_key))


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
    cpu = _nonnegative_float(payload.get("cpus", 1.0), "cpus")
    gpu = _nonnegative_float(payload.get("gpus", 0.0), "gpus")
    if cpu <= 0 and gpu <= 0:
        raise ValueError(f"Ray UDF node {node_id} must request CPU or GPU resources")
    declared_heap = payload.get("memory_bytes")
    heap_bytes = 0 if declared_heap is None else _positive_int(declared_heap, "memory_bytes")
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
        actor_prefetch_depth = (
            1
            if bool(payload.get("stateful", False))
            else _env_positive_int(
                env,
                "VANE_RAY_ACTOR_PREFETCH_DEPTH",
                _DEFAULT_RAY_ACTOR_PREFETCH_DEPTH,
            )
        )
        max_concurrency = None
        actor_pool_size = actor_size
        resident_per_actor = ResourceVector(
            cpu=cpu,
            gpu=gpu,
            heap_bytes=heap_bytes,
        )
        invocation_resources = ResourceVector(object_store_bytes=input_window)
    else:
        max_concurrency = None
        actor_pool_size = 0
        actor_prefetch_depth = 1
        resident_per_actor = ResourceVector()
        invocation_resources = ResourceVector(
            cpu=cpu,
            gpu=gpu,
            heap_bytes=heap_bytes,
            object_store_bytes=input_window,
        )
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
        task_uses_dimension = any(getattr(task, field_name) > 0 for task in ray_tasks)
        if task_uses_dimension:
            elastic = getattr(cluster_capacity, field_name)
        else:
            elastic = min(
                getattr(actor_maximum, field_name),
                getattr(cluster_capacity, field_name),
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
