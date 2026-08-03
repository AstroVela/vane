# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Any

from duckdb.runners.ray.cluster_resource_coordinator import (
    ActorResourceBundle,
    QueryDemand,
)
from duckdb.runners.ray.query_resource_graph import (
    QueryResourceGraph,
    ResourceUnitSpec,
    ResourceVector,
)

_GIB = 1024**3
_MIB = 1024**2
_DEFAULT_NATIVE_FRAGMENT_TASK_HEAP_BYTES = 2 * _GIB
_DEFAULT_NATIVE_FRAGMENT_UDF_DRIVER_HEAP_BYTES = 512 * _MIB
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
    "is_blocking_materializing",
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
        node["is_blocking_materializing"] = bool(node["is_blocking_materializing"])
        if node["udf_payload"] is not None and not isinstance(node["udf_payload"], Mapping):
            raise TypeError(f"resource unit node {node_id} udf_payload must be a mapping or None")
        node["udf_payload"] = None if node["udf_payload"] is None else dict(node["udf_payload"])
        nodes[node_id] = node

    for node_id, node in nodes.items():
        for input_node_id in node["input_node_ids"]:
            if input_node_id not in nodes:
                raise ValueError(f"resource unit node {node_id} references missing input node {input_node_id}")
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
    *,
    downstream_input_window_bytes: int = 0,
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
        generator_buffer_blocks=retention_blocks,
        max_concurrency=max_concurrency,
        resident_per_actor=resident_per_actor,
        actor_pool_size=actor_pool_size,
        actor_prefetch_depth=actor_prefetch_depth,
        is_barrier=False,
    )


def build_query_resource_graph(
    metadata: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
) -> QueryResourceGraph:
    environment = os.environ if env is None else env
    query_id, nodes, terminal_node_ids = _normalize_metadata(metadata)
    native_fragment_heap = _env_positive_int(
        environment, "VANE_NATIVE_FRAGMENT_TASK_HEAP_BYTES", _DEFAULT_NATIVE_FRAGMENT_TASK_HEAP_BYTES
    )
    native_fragment_target = _env_positive_int(
        environment,
        "VANE_TARGET_OUTPUT_BLOCK_BYTES",
        _DEFAULT_TARGET_OUTPUT_BLOCK_BYTES,
    )
    native_fragment_udf_driver_heap = _env_positive_int(
        environment,
        "VANE_NATIVE_FRAGMENT_UDF_DRIVER_HEAP_BYTES",
        max(
            _DEFAULT_NATIVE_FRAGMENT_UDF_DRIVER_HEAP_BYTES,
            native_fragment_target * _GENERATOR_BUFFER_BLOCKS,
        ),
    )

    output_unit_by_node: dict[str, str] = {}
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
        output_unit_by_node[node_id] = (
            udf_unit_id_for_node(query_id, node_id)
            if has_remote_udf
            else native_fragment_unit_id_for_node(query_id, node_id)
        )
        if has_remote_udf:
            # Distributed task fragments terminate at the native node feeding
            # a remote UDF; the UDF node's native fragment unit is a logical
            # wrapper.
            # Both are orchestration units, while the separately leased UDF
            # process owns the standalone heap commitment.
            remote_udf_driver_node_ids.add(node_id)
            remote_udf_driver_node_ids.update(node["input_node_ids"])

    units: list[ResourceUnitSpec] = []
    for node_id in sorted(nodes, key=_node_sort_key):
        node = nodes[node_id]
        native_fragment_unit_id = native_fragment_unit_id_for_node(query_id, node_id)
        input_unit_ids = tuple(output_unit_by_node[parent] for parent in node["input_node_ids"])
        is_sink = bool(node["is_sink"])
        is_blocking_materializing = bool(node["is_blocking_materializing"])
        remote_udf_driver = node_id in remote_udf_driver_node_ids
        units.append(
            ResourceUnitSpec(
                query_id=query_id,
                resource_unit_id=native_fragment_unit_id,
                physical_node_id=f"node:{node_id}:native-fragment",
                unit_kind="native_fragment",
                backend="ray_worker",
                input_unit_ids=input_unit_ids,
                # A Ray UDF node runs its user code in a separately leased Ray
                # process. The parent native fragment task is an in-process
                # orchestration continuation in the shared RayWorkerActor, so
                # charging the full standalone-process default again
                # double-counts heap.
                # Its incremental commitment is instead bounded by the paired
                # stream window, with a conservative 512 MiB floor. Native
                # fragment units retain the 2 GiB default for joins/sorts/spill.
                per_task=ResourceVector(
                    cpu=1,
                    heap_bytes=native_fragment_udf_driver_heap if remote_udf_driver else native_fragment_heap,
                ),
                target_output_block_bytes=0 if is_sink else native_fragment_target,
                generator_buffer_blocks=0 if is_sink else _GENERATOR_BUFFER_BLOCKS,
                max_concurrency=int(node["num_partitions"]),
                is_barrier=is_blocking_materializing,
            )
        )
        udf_unit = _udf_unit(
            query_id,
            node,
            native_fragment_unit_id,
            environment,
            downstream_input_window_bytes=downstream_input_windows[node_id],
        )
        if udf_unit is not None:
            units.append(udf_unit)

    terminals = tuple(output_unit_by_node[node_id] for node_id in terminal_node_ids)
    preliminary = QueryResourceGraph(
        query_id=query_id,
        plan_digest="sha256:pending",
        units=tuple(units),
        terminal_unit_ids=terminals,
    )
    return QueryResourceGraph(
        query_id=query_id,
        plan_digest=preliminary.normalized_digest(),
        units=preliminary.units,
        terminal_unit_ids=preliminary.terminal_unit_ids,
    )


def _hard_task_commitment(unit: ResourceUnitSpec) -> ResourceVector:
    """Return resources that must be placed before a task can run.

    Retained inputs and generator output windows are spillable pipeline data.
    They are controlled dynamically by QRM against the query's object-store
    budget instead of being reserved for every possible task at registration.
    """
    return ResourceVector(
        cpu=unit.per_task.cpu,
        gpu=unit.per_task.gpu,
        heap_bytes=unit.per_task.heap_bytes,
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
    graph: QueryResourceGraph,
    cluster_capacity: ResourceVector,
    *,
    weight: float = 1.0,
    priority: int = 0,
) -> QueryDemand:
    actor_bundles: list[ActorResourceBundle] = []
    native_fragment_tasks: list[ResourceVector] = []
    ray_tasks: list[ResourceVector] = []
    downstream_native_fragment_tasks: list[ResourceVector] = []
    unit_by_id = {unit.resource_unit_id: unit for unit in graph.units}
    for unit in graph.units:
        commitment = _hard_task_commitment(unit)
        if unit.backend == "ray_actor":
            actor_bundles.extend(
                ActorResourceBundle(
                    resource_unit_id=unit.resource_unit_id,
                    actor_index=actor_index,
                    resources=unit.resident_per_actor,
                )
                for actor_index in range(unit.actor_pool_size)
            )
        elif unit.backend == "ray_task":
            ray_tasks.append(commitment)
        elif unit.backend == "ray_worker":
            native_fragment_tasks.append(commitment)
    downstream_native_fragment_unit_ids = {
        downstream_unit_id
        for unit in graph.units
        if unit.backend != "ray_worker"
        for downstream_unit_id in (
            graph.downstream_native_fragment_unit_ids_requiring_separate_slot(unit.resource_unit_id)
        )
    }
    downstream_native_fragment_tasks.extend(
        _hard_task_commitment(unit_by_id[resource_unit_id])
        for resource_unit_id in graph.topological_unit_ids()
        if resource_unit_id in downstream_native_fragment_unit_ids
    )
    minimum = ResourceVector()
    for required_actor in actor_bundles:
        minimum = minimum + required_actor.resources
    task_bundles = tuple(
        bundle
        # Reserve the nested Ray process before its parent native fragment
        # bundle. The minimum is component-wise identical either way on one node, but the
        # order is placement-significant on heterogeneous/multi-node clusters
        # and must preserve continuation capacity.
        # A remote-process streaming producer can remain alive while a
        # downstream native fragment task drains it. Keep that progress slot
        # separate from the native fragment task that invoked the producer;
        # QRM enforces the same shared reservation dynamically before
        # admitting additional producers.
        for bundle in (
            _component_max(ray_tasks),
            _component_max(native_fragment_tasks),
            _component_max(downstream_native_fragment_tasks),
        )
        if not bundle.is_zero()
    )
    for task_bundle in task_bundles:
        minimum = minimum + task_bundle
    desired = ResourceVector(
        cpu=cluster_capacity.cpu,
        # GPU commitments remain fixed indivisible bundles. Only CPU and
        # memory headroom participate in elastic DRF allocation.
        gpu=minimum.gpu,
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
    "build_query_resource_graph",
    "native_fragment_unit_id_for_fragment",
    "native_fragment_unit_id_for_node",
    "udf_unit_id_for_node",
]
