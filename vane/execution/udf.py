# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Unified Python UDF executor routing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vane.execution._common import ensure_table as _ensure_table

if TYPE_CHECKING:
    from vane.execution.unified_executor import UDFExecutor as BaseUDFExecutor

_DEFAULTS: dict[str, Any] = {
    "max_task_retries": None,
    "ray_options": {},
}
_ALLOWED_OPTIONS = frozenset(
    {
        *_DEFAULTS,
        "actor_handles",
        "actor_node_ids",
        "actor_dispatch_indices",
        "local_actor_pool",
    }
)


def normalize_options(options: Any | None) -> dict[str, Any]:
    merged = dict(_DEFAULTS)
    if options is None:
        options = {}
    if type(options) is not dict:
        raise TypeError("UDF executor options must be a dict")
    unknown = sorted(set(options) - _ALLOWED_OPTIONS)
    if unknown:
        raise ValueError("unknown UDF executor options: %s" % ", ".join(unknown))
    merged.update(options)
    retries = merged["max_task_retries"]
    if retries is not None and (type(retries) is not int or retries < 0):
        raise ValueError("max_task_retries must be a non-negative integer")
    ray_options = merged["ray_options"]
    if type(ray_options) is not dict:
        raise TypeError("UDF executor ray_options must be a dict")
    merged["ray_options"] = dict(ray_options)
    _reject_runtime_overrides(merged)
    return merged


def _reject_runtime_overrides(options: dict[str, Any]) -> None:
    ray_options = options["ray_options"]
    ray_resource_keys = sorted(
        key
        for key in (
            "_generator_backpressure_num_objects",
            "max_restarts",
            "max_retries",
            "max_task_retries",
            "memory",
            "num_cpus",
            "num_gpus",
            "scheduling_strategy",
        )
        if key in ray_options
    )
    if ray_resource_keys:
        raise ValueError("UDF runtime ray_options cannot override payload resources: %s" % ", ".join(ray_resource_keys))


def build_executor(payload: dict[str, Any], _options: dict[str, Any] | None = None) -> BaseUDFExecutor:
    if payload is None:
        raise ValueError("UDF payload is required")
    call_mode = str(payload.get("call_mode") or "").strip().lower()
    if call_mode not in ("map_batches", "map_batches_rows", "flat_map", "map"):
        raise ValueError("UDF payload.call_mode must be one of: map_batches, map_batches_rows, flat_map, map")

    backend = str(payload.get("execution_backend") or "").strip().lower()
    if backend not in ("subprocess_task", "subprocess_actor", "ray_task", "ray_actor"):
        raise ValueError(
            "UDF payload.execution_backend must be one of: subprocess_task, subprocess_actor, ray_task, ray_actor"
        )

    options = normalize_options(_options)

    if backend in ("subprocess_task", "subprocess_actor"):
        gpus = float(payload.get("gpus") or 0.0)
        if gpus > 0:
            raise ValueError("GPU resources require a Ray UDF backend")
        from vane.execution.udf_subprocess import UDFExecutor

        return UDFExecutor(payload, options=options)

    from vane.execution.udf_ray import build_ray_executor

    return build_ray_executor(payload, options)


__all__ = ["_ensure_table", "build_executor", "normalize_options"]
