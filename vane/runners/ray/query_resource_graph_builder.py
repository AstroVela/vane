# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from vane.execution.query_resource_spec import (
    MaterializationBarrierSpec,
    QueryResourceGraph,
    ResourceUnitSpec,
    ResourceVector,
)
from vane.execution.resource_graph_metadata import (
    _node_sort_key,
    _normalize_metadata,
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
from vane.execution.udf_resource_policy import (
    DEFAULT_TARGET_OUTPUT_BLOCK_BYTES as _DEFAULT_TARGET_OUTPUT_BLOCK_BYTES,
)
from vane.execution.udf_resource_policy import (
    env_positive_int as _env_positive_int,
)
from vane.execution.udf_resource_policy import (
    udf_resource_spec,
)
from vane.execution.udf_stream_backpressure import STREAM_BUFFER_BLOCKS


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
    return udf_resource_spec(
        query_id=query_id,
        resource_unit_id=expected_unit_id,
        physical_node_id=f"node:{node_id}:udf",
        input_unit_ids=(input_unit_id,),
        payload=payload,
        env=env,
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
                generator_buffer_blocks=0 if is_sink else STREAM_BUFFER_BLOCKS,
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


__all__ = [
    "build_query_resource_graph",
    "materialization_barrier_id_for_node",
    "native_fragment_unit_id_for_fragment",
    "native_fragment_unit_id_for_node",
    "udf_unit_id_for_node",
]
