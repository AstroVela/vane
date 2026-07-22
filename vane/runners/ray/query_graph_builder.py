# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Any

from vane.runners.ray.cluster_resource_coordinator import (
    ActorResourceBundle,
    QueryDemand,
)
from vane.runners.ray.query_execution_graph import (
    QueryExecutionGraph,
    ResourceVector,
    StageResourceSpec,
)

_GIB = 1024**3
_MIB = 1024**2
_DEFAULT_FTE_TASK_HEAP_BYTES = 2 * _GIB
_DEFAULT_FTE_UDF_DRIVER_HEAP_BYTES = 512 * _MIB
_DEFAULT_UDF_TASK_HEAP_BYTES = 2 * _GIB
_DEFAULT_UDF_ACTOR_HEAP_BYTES = 4 * _GIB
_DEFAULT_TARGET_OUTPUT_BLOCK_BYTES = 128 * _MIB
_DEFAULT_RAY_ACTOR_PREFETCH_DEPTH = 2
_GENERATOR_BUFFER_BLOCKS = 2
_TOP_LEVEL_FIELDS = ("query_id", "nodes", "terminal_node_ids")
_NODE_FIELDS = (
    "node_id",
    "node_name",
    "input_node_ids",
    "is_sink",
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


def _positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive number")
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


def fte_stage_id_for_node(query_id: str, node_id: str | int) -> str:
    query = str(query_id).strip()
    node = str(node_id).strip()
    if not query or not node:
        raise ValueError("query_id and node_id must be non-empty")
    return f"stage:{query}:node:{node}:fte"


def udf_stage_id_for_node(query_id: str, node_id: str | int) -> str:
    query = str(query_id).strip()
    node = str(node_id).strip()
    if not query or not node:
        raise ValueError("query_id and node_id must be non-empty")
    return f"stage:{query}:node:{node}:udf"


def fte_stage_id_for_fragment(query_id: str, fragment_id: str) -> str:
    query = str(query_id).strip()
    fragment = str(fragment_id).strip()
    prefix = f"{query}:node:"
    if not fragment.startswith(prefix):
        if fragment.endswith(":node:") or ":node:" not in fragment:
            raise ValueError(f"invalid FTE fragment_id: {fragment}")
        raise ValueError(f"fragment {fragment!r} does not belong to query {query!r}")
    node_id = fragment[len(prefix) :]
    if not node_id or ":" in node_id:
        raise ValueError(f"invalid FTE fragment_id: {fragment}")
    return fte_stage_id_for_node(query, node_id)


def _normalize_metadata(metadata: Mapping[str, Any]) -> tuple[str, dict[str, dict[str, Any]], tuple[str, ...]]:
    payload = dict(metadata)
    _strict_fields(payload, _TOP_LEVEL_FIELDS, "execution stage metadata")
    query_id = str(payload["query_id"]).strip()
    if not query_id:
        raise ValueError("execution stage metadata query_id must be non-empty")
    nodes: dict[str, dict[str, Any]] = {}
    for raw_node in payload["nodes"]:
        node = dict(raw_node)
        _strict_fields(node, _NODE_FIELDS, "execution stage node")
        node_id = str(node["node_id"]).strip()
        if not node_id:
            raise ValueError("execution stage node_id must be non-empty")
        if node_id in nodes:
            raise ValueError(f"duplicate execution stage node_id: {node_id}")
        node["node_id"] = node_id
        node["node_name"] = str(node["node_name"]).strip()
        if not node["node_name"]:
            raise ValueError(f"execution stage node {node_id} node_name must be non-empty")
        node["input_node_ids"] = tuple(str(item).strip() for item in node["input_node_ids"])
        node["num_partitions"] = _positive_int(node["num_partitions"], "num_partitions")
        node["is_sink"] = bool(node["is_sink"])
        if node["udf_payload"] is not None and not isinstance(node["udf_payload"], Mapping):
            raise TypeError(f"execution stage node {node_id} udf_payload must be a mapping or None")
        node["udf_payload"] = None if node["udf_payload"] is None else dict(node["udf_payload"])
        nodes[node_id] = node

    for node_id, node in nodes.items():
        for input_node_id in node["input_node_ids"]:
            if input_node_id not in nodes:
                raise ValueError(f"execution stage node {node_id} references missing input node {input_node_id}")
    terminal_node_ids = tuple(str(item).strip() for item in payload["terminal_node_ids"])
    if not terminal_node_ids:
        raise ValueError("execution stage metadata must contain terminal_node_ids")
    for terminal in terminal_node_ids:
        if terminal not in nodes:
            raise ValueError(f"terminal node is not registered: {terminal}")
    return query_id, nodes, tuple(sorted(set(terminal_node_ids), key=_node_sort_key))


def _udf_stage(
    query_id: str,
    node: Mapping[str, Any],
    input_stage_id: str,
    env: Mapping[str, str],
    *,
    downstream_input_window_bytes: int = 0,
) -> StageResourceSpec | None:
    payload = node["udf_payload"]
    if payload is None:
        return None
    backend = str(payload.get("execution_backend") or "").strip()
    if backend not in {"ray_task", "ray_actor"}:
        return None
    node_id = str(node["node_id"])
    expected_stage_id = udf_stage_id_for_node(query_id, node_id)
    actual_stage_id = str(payload.get("stage_id") or "").strip()
    if not actual_stage_id:
        raise ValueError(f"Ray UDF node {node_id} is missing pre-registered stage_id")
    if actual_stage_id != expected_stage_id:
        raise ValueError(
            f"Ray UDF node {node_id} stage_id mismatch: got {actual_stage_id!r}, expected {expected_stage_id!r}"
        )
    payload_query_id = str(payload.get("query_id") or "").strip()
    if payload_query_id and payload_query_id != query_id:
        raise ValueError(f"Ray UDF node {node_id} query_id mismatch: got {payload_query_id!r}, expected {query_id!r}")
    cpu = _nonnegative_float(payload.get("cpus", 1.0), "cpus")
    gpu = _nonnegative_float(payload.get("gpus", 0.0), "gpus")
    if cpu <= 0 and gpu <= 0:
        raise ValueError(f"Ray UDF node {node_id} must request CPU or GPU resources")
    default_heap = (
        _env_positive_int(env, "VANE_UDF_ACTOR_HEAP_BYTES", _DEFAULT_UDF_ACTOR_HEAP_BYTES)
        if backend == "ray_actor"
        else _env_positive_int(env, "VANE_UDF_TASK_HEAP_BYTES", _DEFAULT_UDF_TASK_HEAP_BYTES)
    )
    heap_bytes = _positive_int(payload.get("memory_bytes", default_heap), "memory_bytes")
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
    retention_window = max(
        target * _GENERATOR_BUFFER_BLOCKS,
        int(downstream_input_window_bytes),
    )
    retention_blocks = max(
        _GENERATOR_BUFFER_BLOCKS,
        math.ceil(retention_window / target),
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
        actor_min = actor_size
        actor_max = actor_size
        resident_per_actor = ResourceVector(
            cpu=cpu,
            gpu=gpu,
            heap_bytes=heap_bytes,
        )
        invocation_resources = ResourceVector(object_store_bytes=input_window)
    else:
        max_concurrency = None
        actor_min = 0
        actor_max = 0
        actor_prefetch_depth = 1
        resident_per_actor = ResourceVector()
        invocation_resources = ResourceVector(
            cpu=cpu,
            gpu=gpu,
            heap_bytes=heap_bytes,
            object_store_bytes=input_window,
        )
    return StageResourceSpec(
        query_id=query_id,
        stage_id=expected_stage_id,
        physical_node_id=f"node:{node_id}:udf",
        stage_kind="udf",
        backend=backend,
        input_stage_ids=(input_stage_id,),
        per_task=invocation_resources,
        target_output_block_bytes=target,
        generator_buffer_blocks=retention_blocks,
        max_concurrency=max_concurrency,
        resident_per_actor=resident_per_actor,
        actor_min_size=actor_min,
        actor_max_size=actor_max,
        actor_prefetch_depth=actor_prefetch_depth,
        spill_mode="streaming",
    )


def build_query_execution_graph(
    metadata: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
) -> QueryExecutionGraph:
    environment = os.environ if env is None else env
    query_id, nodes, terminal_node_ids = _normalize_metadata(metadata)
    fte_heap = _env_positive_int(environment, "VANE_FTE_TASK_HEAP_BYTES", _DEFAULT_FTE_TASK_HEAP_BYTES)
    fte_target = _env_positive_int(
        environment,
        "VANE_TARGET_OUTPUT_BLOCK_BYTES",
        _DEFAULT_TARGET_OUTPUT_BLOCK_BYTES,
    )
    fte_udf_driver_heap = _env_positive_int(
        environment,
        "VANE_FTE_UDF_DRIVER_HEAP_BYTES",
        max(
            _DEFAULT_FTE_UDF_DRIVER_HEAP_BYTES,
            fte_target * _GENERATOR_BUFFER_BLOCKS,
        ),
    )

    output_stage_by_node: dict[str, str] = {}
    remote_udf_driver_node_ids: set[str] = set()
    downstream_node_ids: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for child_id, child in nodes.items():
        for parent_id in child["input_node_ids"]:
            downstream_node_ids[parent_id].append(child_id)

    def remote_udf_submit_window(node_id: str) -> int | None:
        payload = nodes[node_id]["udf_payload"]
        if payload is None or str(payload.get("execution_backend") or "").strip() not in {
            "ray_task",
            "ray_actor",
        }:
            return None
        default_window = payload.get(
            "udf_output_target_max_bytes",
            _env_positive_int(environment, "VANE_TARGET_OUTPUT_BLOCK_BYTES", _DEFAULT_TARGET_OUTPUT_BLOCK_BYTES),
        )
        return _positive_int(
            payload.get("udf_task_input_max_bytes", default_window),
            "udf_task_input_max_bytes",
        )

    downstream_input_windows: dict[str, int] = {}
    for source_id in nodes:
        pending = list(downstream_node_ids[source_id])
        visited: set[str] = set()
        windows: list[int] = []
        while pending:
            downstream_id = pending.pop()
            if downstream_id in visited:
                continue
            visited.add(downstream_id)
            window = remote_udf_submit_window(downstream_id)
            if window is not None:
                windows.append(window)
                continue
            pending.extend(downstream_node_ids[downstream_id])
        downstream_input_windows[source_id] = max(windows, default=0)

    for node_id, node in nodes.items():
        udf_payload = node["udf_payload"]
        has_remote_udf = udf_payload is not None and str(udf_payload.get("execution_backend") or "") in {
            "ray_task",
            "ray_actor",
        }
        output_stage_by_node[node_id] = (
            udf_stage_id_for_node(query_id, node_id) if has_remote_udf else fte_stage_id_for_node(query_id, node_id)
        )
        if has_remote_udf:
            # Distributed task fragments terminate at the native node feeding
            # a remote UDF; the UDF node's own FTE stage is a logical wrapper.
            # Both are orchestration stages, while the separately leased UDF
            # process owns the standalone heap commitment.
            remote_udf_driver_node_ids.add(node_id)
            remote_udf_driver_node_ids.update(node["input_node_ids"])

    stages: list[StageResourceSpec] = []
    for node_id in sorted(nodes, key=_node_sort_key):
        node = nodes[node_id]
        fte_stage_id = fte_stage_id_for_node(query_id, node_id)
        input_stage_ids = tuple(output_stage_by_node[parent] for parent in node["input_node_ids"])
        is_sink = bool(node["is_sink"])
        remote_udf_driver = node_id in remote_udf_driver_node_ids
        stages.append(
            StageResourceSpec(
                query_id=query_id,
                stage_id=fte_stage_id,
                physical_node_id=f"node:{node_id}:fte",
                stage_kind="sink" if is_sink else "fte",
                backend="ray_worker",
                input_stage_ids=input_stage_ids,
                # A Ray UDF node runs its user code in a separately leased Ray
                # process.  The parent FTE task is an in-process orchestration
                # continuation in the shared RayWorkerActor, so charging the
                # full standalone-process default again double-counts heap.
                # Its incremental commitment is instead bounded by the paired
                # stream window, with a conservative 512 MiB floor.  Native
                # FTE stages retain the 2 GiB default for joins/sorts/spill.
                per_task=ResourceVector(
                    cpu=1,
                    heap_bytes=fte_udf_driver_heap if remote_udf_driver else fte_heap,
                ),
                target_output_block_bytes=0 if is_sink else fte_target,
                generator_buffer_blocks=0 if is_sink else _GENERATOR_BUFFER_BLOCKS,
                max_concurrency=int(node["num_partitions"]),
                spill_mode="barrier" if is_sink else "streaming",
            )
        )
        udf_stage = _udf_stage(
            query_id,
            node,
            fte_stage_id,
            environment,
            downstream_input_window_bytes=downstream_input_windows[node_id],
        )
        if udf_stage is not None:
            stages.append(udf_stage)

    terminals = tuple(output_stage_by_node[node_id] for node_id in terminal_node_ids)
    preliminary = QueryExecutionGraph(
        query_id=query_id,
        plan_digest="sha256:pending",
        stages=tuple(stages),
        terminal_stage_ids=terminals,
    )
    return QueryExecutionGraph(
        query_id=query_id,
        plan_digest=preliminary.normalized_digest(),
        stages=preliminary.stages,
        terminal_stage_ids=preliminary.terminal_stage_ids,
    )


def _task_commitment(stage: StageResourceSpec) -> ResourceVector:
    return ResourceVector(
        cpu=stage.per_task.cpu,
        gpu=stage.per_task.gpu,
        heap_bytes=stage.per_task.heap_bytes,
        object_store_bytes=stage.per_task.object_store_bytes + stage.output_window_bytes,
    )


def _component_max(resources: list[ResourceVector]) -> ResourceVector:
    if not resources:
        return ResourceVector()
    return ResourceVector(
        cpu=max(item.cpu for item in resources),
        gpu=max(item.gpu for item in resources),
        heap_bytes=max(item.heap_bytes for item in resources),
        object_store_bytes=max(item.object_store_bytes for item in resources),
    )


def build_query_demand(
    graph: QueryExecutionGraph,
    cluster_capacity: ResourceVector,
    *,
    weight: float = 1.0,
    priority: int = 0,
) -> QueryDemand:
    actor_bundles: list[ActorResourceBundle] = []
    fte_tasks: list[ResourceVector] = []
    ray_tasks: list[ResourceVector] = []
    downstream_fte_tasks: list[ResourceVector] = []
    desired_gpu = 0.0
    stage_by_id = {stage.stage_id: stage for stage in graph.stages}
    for stage in graph.stages:
        commitment = _task_commitment(stage)
        if stage.backend == "ray_actor":
            actor_bundle = stage.resident_per_actor + commitment
            desired_gpu += stage.resident_per_actor.gpu * stage.actor_max_size
            actor_bundles.extend(
                ActorResourceBundle(
                    stage_id=stage.stage_id,
                    actor_index=actor_index,
                    resources=actor_bundle,
                )
                for actor_index in range(stage.actor_min_size)
            )
        elif stage.backend == "ray_task":
            ray_tasks.append(commitment)
        elif stage.backend == "ray_worker":
            fte_tasks.append(commitment)
    downstream_fte_stage_ids = {
        downstream_stage_id
        for stage in graph.stages
        if stage.backend != "ray_worker"
        for downstream_stage_id in (graph.downstream_fte_stage_ids_requiring_separate_slot(stage.stage_id))
    }
    downstream_fte_tasks.extend(
        _task_commitment(stage_by_id[stage_id])
        for stage_id in graph.topological_stage_ids()
        if stage_id in downstream_fte_stage_ids
    )
    minimum = ResourceVector()
    for bundle in actor_bundles:
        minimum = minimum + bundle.resources
    task_bundles = tuple(
        bundle
        # Reserve the nested Ray process before its parent FTE bundle.  The
        # minimum is component-wise identical either way on one node, but the
        # order is placement-significant on heterogeneous/multi-node clusters
        # and must preserve continuation capacity.
        # A non-FTE streaming producer can remain alive while a downstream FTE
        # task drains it. Keep that progress slot separate from the FTE task
        # that invoked the producer; QRM enforces the same shared reservation
        # dynamically before admitting additional producers.
        for bundle in (
            _component_max(ray_tasks),
            _component_max(fte_tasks),
            _component_max(downstream_fte_tasks),
        )
        if not bundle.is_zero()
    )
    for bundle in task_bundles:
        minimum = minimum + bundle
    desired = ResourceVector(
        cpu=cluster_capacity.cpu,
        # GPU task commitments are indivisible task bundles just like their
        # CPU/heap commitments. Keep enough GPU allocation for the minimum
        # runnable task set in addition to any fixed actor residency.
        gpu=min(cluster_capacity.gpu, max(desired_gpu, minimum.gpu)),
        heap_bytes=cluster_capacity.heap_bytes,
        object_store_bytes=cluster_capacity.object_store_bytes,
    )
    return QueryDemand(
        query_id=graph.query_id,
        minimum=minimum,
        desired=desired,
        weight=weight,
        priority=priority,
        actor_bundles=tuple(actor_bundles),
        task_bundles=task_bundles,
    )


__all__ = [
    "build_query_demand",
    "build_query_execution_graph",
    "fte_stage_id_for_fragment",
    "fte_stage_id_for_node",
    "udf_stage_id_for_node",
]
