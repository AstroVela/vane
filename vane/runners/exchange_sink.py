# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_PLAN_IDENTITY = "plan"
_TASK_IDENTITY = "task"
_IDENTITY_SOURCES = {_PLAN_IDENTITY, _TASK_IDENTITY}
_CONFIG_KEYS = {
    "identity_source",
    "plan_task_partition_id",
    "query_id",
    "output_location_prefix",
    "output_partition_count",
}


def _non_negative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def normalize_exchange_sink_config(exchange_sink_config: Any) -> dict[str, Any]:
    if not isinstance(exchange_sink_config, Mapping):
        raise TypeError("exchange_sink_config must be a mapping")

    identity_source = exchange_sink_config.get("identity_source")
    if not isinstance(identity_source, str) or identity_source not in _IDENTITY_SOURCES:
        raise ValueError("exchange_sink_config identity_source must be 'plan' or 'task'")
    unexpected_keys = set(exchange_sink_config) - _CONFIG_KEYS
    if unexpected_keys:
        unexpected_fields = ", ".join(sorted(repr(key) for key in unexpected_keys))
        raise ValueError(f"exchange_sink_config contains unexpected fields: {unexpected_fields}")

    query_id = exchange_sink_config.get("query_id")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError("exchange_sink_config query_id must be non-empty")
    output_location_prefix = exchange_sink_config.get("output_location_prefix")
    if not isinstance(output_location_prefix, str) or not output_location_prefix:
        raise ValueError("exchange_sink_config output_location_prefix must be non-empty")
    output_partition_count = _non_negative_integer(
        "exchange_sink_config output_partition_count",
        exchange_sink_config.get("output_partition_count"),
    )
    if output_partition_count == 0:
        raise ValueError("exchange_sink_config output_partition_count must be positive")

    result: dict[str, Any] = {
        "identity_source": identity_source,
        "query_id": query_id,
        "output_location_prefix": output_location_prefix,
        "output_partition_count": output_partition_count,
    }
    plan_task_partition_id = exchange_sink_config.get("plan_task_partition_id")
    if identity_source == _PLAN_IDENTITY:
        if plan_task_partition_id is None:
            raise ValueError("plan-identity exchange_sink_config requires plan_task_partition_id")
        result["plan_task_partition_id"] = _non_negative_integer(
            "exchange_sink_config plan_task_partition_id",
            plan_task_partition_id,
        )
    elif "plan_task_partition_id" in exchange_sink_config:
        raise ValueError("task-identity exchange_sink_config cannot carry plan_task_partition_id")
    return result


def bind_exchange_sink_instance(
    exchange_sink_config: Any,
    *,
    attempt_id: int,
    task_partition_id: int | None = None,
) -> dict[str, Any]:
    config = normalize_exchange_sink_config(exchange_sink_config)
    attempt_id = _non_negative_integer("attempt_id", attempt_id)
    if config["identity_source"] == _PLAN_IDENTITY:
        if task_partition_id is not None:
            raise ValueError("plan-identity exchange sink binding cannot override task_partition_id")
        bound_task_partition_id = int(config["plan_task_partition_id"])
    else:
        if task_partition_id is None:
            raise ValueError("task-identity exchange sink binding requires task_partition_id")
        bound_task_partition_id = _non_negative_integer("task_partition_id", task_partition_id)

    output_location = f"{config['output_location_prefix']}__sink_{bound_task_partition_id}__attempt_{attempt_id}"
    return {
        "sink_handle": {"task_partition_id": bound_task_partition_id},
        "task_partition_id": bound_task_partition_id,
        "attempt_id": attempt_id,
        "query_id": config["query_id"],
        "output_partition_count": config["output_partition_count"],
        "output_location": output_location,
    }
