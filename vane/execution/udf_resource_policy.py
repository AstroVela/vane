# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Identical UDF resource declarations and defaults for every backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vane.execution.query_resource_spec import ResourceUnitSpec
from vane.execution.resource_graph_metadata import _positive_int
from vane.execution.resources import ResourceVector, udf_process_resources
from vane.execution.udf_stream_backpressure import STREAM_BUFFER_BLOCKS

DEFAULT_TARGET_OUTPUT_BLOCK_BYTES = 128 * 1024**2
DEFAULT_ACTOR_PREFETCH_DEPTH = 2


def env_positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    return int(default) if raw is None or not str(raw).strip() else _positive_int(raw, name)


def actor_prefetch_depth(env: Mapping[str, str]) -> int:
    return env_positive_int(env, "VANE_UDF_ACTOR_PREFETCH_DEPTH", DEFAULT_ACTOR_PREFETCH_DEPTH)


def udf_resource_spec(
    *,
    query_id: str,
    resource_unit_id: str,
    physical_node_id: str,
    input_unit_ids: tuple[str, ...],
    payload: Mapping[str, Any],
    env: Mapping[str, str],
) -> ResourceUnitSpec:
    backend = str(payload["execution_backend"])
    kinds = {
        "ray_task": "ray_task_udf",
        "ray_actor": "ray_actor_pool",
        "subprocess_task": "subprocess_task_udf",
        "subprocess_actor": "subprocess_actor_pool",
    }
    actor = backend in {"ray_actor", "subprocess_actor"}
    resources = udf_process_resources(dict(payload))
    default_target = env_positive_int(env, "VANE_TARGET_OUTPUT_BLOCK_BYTES", DEFAULT_TARGET_OUTPUT_BLOCK_BYTES)
    target = _positive_int(payload.get("udf_output_target_max_bytes", default_target), "udf_output_target_max_bytes")
    input_window = _positive_int(payload.get("udf_task_input_max_bytes", default_target), "udf_task_input_max_bytes")
    return ResourceUnitSpec(
        query_id=query_id,
        resource_unit_id=resource_unit_id,
        physical_node_id=physical_node_id,
        unit_kind=kinds[backend],
        backend=backend,
        input_unit_ids=input_unit_ids,
        per_task=(ResourceVector() if actor else resources) + ResourceVector(object_store_bytes=input_window),
        target_output_block_bytes=target,
        generator_buffer_blocks=STREAM_BUFFER_BLOCKS,
        max_concurrency=None,
        resident_per_actor=resources if actor else ResourceVector(),
        actor_pool_size=_positive_int(payload.get("actor_pool_size"), "actor_pool_size") if actor else 0,
        actor_prefetch_depth=actor_prefetch_depth(env) if actor else 1,
    )
