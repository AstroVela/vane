# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_CONFIG_KEYS = {
    "query_id",
    "output_location_prefix",
    "output_partition_count",
    "preserve_order",
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

    preserve_order = exchange_sink_config.get("preserve_order", False)
    if not isinstance(preserve_order, bool):
        raise TypeError("exchange_sink_config preserve_order must be a boolean")

    result: dict[str, Any] = {
        "query_id": query_id,
        "output_location_prefix": output_location_prefix,
        "output_partition_count": output_partition_count,
    }
    if preserve_order:
        result["preserve_order"] = True
    return result


def bind_exchange_sink_instance(
    exchange_sink_config: Any,
    *,
    attempt_id: int,
    task_partition_id: int,
    source_task_order: int | None = None,
) -> dict[str, Any]:
    config = normalize_exchange_sink_config(exchange_sink_config)
    attempt_id = _non_negative_integer("attempt_id", attempt_id)
    task_partition_id = _non_negative_integer("task_partition_id", task_partition_id)

    if config.get("preserve_order", False):
        source_task_order = _non_negative_integer("source_task_order", source_task_order)
    elif source_task_order is not None:
        raise ValueError("source_task_order requires an order-preserving exchange sink")

    output_location = f"{config['output_location_prefix']}__sink_{task_partition_id}__attempt_{attempt_id}"
    payload = {
        "sink_handle": {"task_partition_id": task_partition_id},
        "attempt_id": attempt_id,
        "query_id": config["query_id"],
        "output_partition_count": config["output_partition_count"],
        "output_location": output_location,
    }
    if source_task_order is not None:
        payload["source_task_order"] = source_task_order
    return payload
