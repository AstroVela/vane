# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any

from vane.execution.udf_ray_config import (
    payload_num_cpus,
)


def is_ray_worker_process() -> bool:
    import ray
    from ray._private import worker as ray_worker

    return ray.is_initialized() and ray_worker.global_worker.mode == ray_worker.WORKER_MODE


def is_vane_worker_process() -> bool:
    return is_ray_worker_process() and os.getenv("VANE_WORKER") is not None


def collect_actor_env_overrides() -> dict[str, str]:
    overrides = {k: v for k, v in os.environ.items() if k.startswith("VANE_")}
    for key in ("PYTHONPATH", "PYTHONWARNINGS"):
        value = os.environ.get(key)
        if value:
            overrides[key] = value
    for key, value in os.environ.items():
        if key in overrides:
            continue
        if key.startswith(("AWS_", "DUCKDB_", "S3FS_", "VANE_")):
            overrides[key] = value
    return overrides


def normalize_actor_pool_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Normalize distributed actor-pool payloads before actor creation/use.

    Ordinary stateful UDFs use direct actor calls.
    """
    normalized = dict(payload or {})
    if normalized.get("ray_actor_pool_name") is not None:
        raise ValueError("Ray UDF named actor pools have been removed; ray_actor_pool_name is unsupported")
    return normalized


def normalize_actor_node_ids(
    value: Any | None,
    *,
    expected_count: int | None = None,
) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        node_ids = [value]
    else:
        try:
            node_ids = [str(item) if item is not None else "" for item in value]
        except TypeError as exc:
            raise ValueError("actor node IDs must be a string or iterable of strings") from exc
    if expected_count is not None and expected_count >= 0 and len(node_ids) != expected_count:
        raise ValueError(f"actor node IDs count {len(node_ids)} does not match expected actor count {expected_count}")
    if any(not node_id.strip() for node_id in node_ids):
        raise ValueError("actor node IDs must be non-empty strings")
    return node_ids


def resolve_actor_num_cpus(payload: dict[str, Any]) -> float:
    return payload_num_cpus(payload)
