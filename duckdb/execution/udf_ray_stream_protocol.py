# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa
from _duckdb import __standard_vector_size__ as DUCKDB_STANDARD_VECTOR_SIZE  # type: ignore[import-not-found]

from duckdb.execution._common import ensure_table, estimate_table_bytes

RAY_UDF_STREAM_PROTOCOL_VERSION = 1
RAY_UDF_STREAM_OBJECTS_PER_BLOCK = 2
RAY_UDF_STREAM_BUFFER_BLOCKS = 2
RAY_UDF_GENERATOR_BACKPRESSURE_OBJECTS = RAY_UDF_STREAM_OBJECTS_PER_BLOCK * RAY_UDF_STREAM_BUFFER_BLOCKS


def validate_task_runtime_node(payload: dict[str, Any]) -> str:
    if str(payload.get("execution_backend") or "").strip() != "ray_task":
        raise RuntimeError("task runtime node discovery requires the ray_task backend")
    if str(payload.get("node_id") or "").strip():
        raise RuntimeError("Ray task payload must leave physical placement to Ray Core")
    import ray

    actual_node_id = str(ray.get_runtime_context().get_node_id() or "").strip()
    if not actual_node_id:
        raise RuntimeError("Ray runtime context did not expose the executing node_id")
    return actual_node_id


_RAW_METADATA_FIELDS = {
    "protocol_version",
    "query_id",
    "producer_unit_id",
    "task_lease_id",
    "attempt_id",
    "block_id",
    "size_bytes",
    "num_rows",
    "names",
}
_ERROR_METADATA_FIELDS = {
    "protocol_version",
    "event_kind",
    "query_id",
    "producer_unit_id",
    "task_lease_id",
    "attempt_id",
    "exception_type",
    "exception_message",
    "exception_details",
}
_MAX_ERROR_TEXT_CHARS = 16 * 1024
_PROVIDER_CAPABILITY_DETAIL_FIELDS = {
    "kind",
    "provider",
    "model",
    "capability",
    "original_error_summary",
}


def _stream_identity(payload: dict[str, Any]) -> tuple[str, str, str, str]:
    query_id = str(payload.get("query_id") or "").strip()
    resource_unit_id = str(payload.get("resource_unit_id") or "").strip()
    task_lease_id = str(payload.get("task_lease_id") or "").strip()
    attempt_id = str(payload.get("attempt_id") or "").strip()
    if not query_id or not resource_unit_id or not task_lease_id or not attempt_id:
        raise RuntimeError("Ray UDF stream output requires query_id, resource_unit_id, task_lease_id, and attempt_id")
    return query_id, resource_unit_id, task_lease_id, attempt_id


def task_payload_with_lease(payload: dict[str, Any], lease: dict[str, Any]) -> dict[str, Any]:
    """Bind a remote UDF invocation to its pre-admitted task lease."""
    merged = dict(payload)
    query_id = str(lease.get("query_id") or "").strip()
    resource_unit_id = str(lease.get("resource_unit_id") or "").strip()
    lease_id = str(lease.get("lease_id") or "").strip()
    attempt_id = str(lease.get("attempt_id") or "").strip()
    execution_slot_id = str(lease.get("execution_slot_id") or "").strip()
    output_window_bytes = int(lease.get("output_window_bytes") or 0)
    if not query_id or not resource_unit_id or not lease_id or not attempt_id or not execution_slot_id:
        raise ValueError("task lease is missing query, resource unit, lease, attempt, or execution slot identity")
    if str(merged.get("query_id") or "").strip() != query_id:
        raise ValueError("UDF payload query_id does not match task lease")
    if str(merged.get("resource_unit_id") or "").strip() != resource_unit_id:
        raise ValueError("UDF payload resource_unit_id does not match task lease")
    backend = str(merged.get("execution_backend") or "").strip()
    raw_actor_index = lease.get("actor_index")
    if backend == "ray_actor":
        node_id = str(lease.get("node_id") or "").strip()
        if not node_id:
            raise ValueError("Ray actor task lease is missing its runtime node")
        if isinstance(raw_actor_index, bool) or not isinstance(raw_actor_index, int) or raw_actor_index < 0:
            raise ValueError("Ray actor task lease is missing a valid actor_index")
        expected_slot_id = f"ray_actor:{resource_unit_id}:{raw_actor_index}"
        if execution_slot_id != expected_slot_id:
            raise ValueError(
                "Ray actor task lease execution slot does not match actor_index: "
                f"slot={execution_slot_id} expected={expected_slot_id}"
            )
    elif backend == "ray_task":
        if lease.get("node_id") is not None:
            raise ValueError("Ray task lease must leave physical placement to Ray Core")
        if raw_actor_index is not None:
            raise ValueError("Ray task lease must not contain actor_index")
        expected_slot_id = f"ray_task:{resource_unit_id}:{lease_id}"
        if execution_slot_id != expected_slot_id:
            raise ValueError(
                "Ray task lease execution slot does not match lease_id: "
                f"slot={execution_slot_id} expected={expected_slot_id}"
            )
    else:
        raise ValueError(f"unsupported distributed UDF execution backend: {backend!r}")
    target_bytes = int(merged.get("udf_output_target_max_bytes") or 0)
    if target_bytes <= 0:
        raise ValueError("UDF payload udf_output_target_max_bytes must be positive")
    minimum_window_bytes = target_bytes * RAY_UDF_STREAM_BUFFER_BLOCKS
    if output_window_bytes < minimum_window_bytes or output_window_bytes % target_bytes != 0:
        raise ValueError(
            "UDF task lease output window is not a valid multiple of the registered block target: "
            f"lease={output_window_bytes} target={target_bytes} minimum={minimum_window_bytes}"
        )
    merged["task_lease_id"] = lease_id
    merged["attempt_id"] = attempt_id
    if backend == "ray_actor":
        merged["node_id"] = node_id
    else:
        merged.pop("node_id", None)
    merged["execution_slot_id"] = execution_slot_id
    merged["actor_index"] = raw_actor_index
    merged["output_window_bytes"] = output_window_bytes
    return merged


def _output_block_target_bytes(payload: dict[str, Any]) -> int:
    target_bytes = int(payload.get("udf_output_target_max_bytes") or 0)
    output_window_bytes = int(payload.get("output_window_bytes") or 0)
    if target_bytes <= 0:
        raise RuntimeError("Ray UDF stream output requires a positive udf_output_target_max_bytes")
    minimum_window_bytes = target_bytes * RAY_UDF_STREAM_BUFFER_BLOCKS
    if output_window_bytes < minimum_window_bytes or output_window_bytes % target_bytes != 0:
        raise RuntimeError(
            "Ray UDF stream output window is not a valid multiple of its registered block target: "
            f"window={output_window_bytes} target={target_bytes} minimum={minimum_window_bytes}"
        )
    return target_bytes


def iter_bounded_stream_blocks(
    table: pa.Table,
    payload: dict[str, Any],
) -> Iterator[pa.Table]:
    """Split one producer result around its soft byte and hard row targets."""
    block = ensure_table(table)
    target_bytes = _output_block_target_bytes(payload)
    size_bytes = max(1, int(estimate_table_bytes(block)))
    if size_bytes <= target_bytes and block.num_rows <= DUCKDB_STANDARD_VECTOR_SIZE:
        yield block
        return
    if block.num_rows <= 1:
        # One row is the smallest publishable Arrow block. Let the query
        # resource manager account its exact size and use the bounded
        # full-block liveness escape when it exceeds the soft target.
        yield block
        return

    offset = 0
    while offset < block.num_rows:
        remaining = block.num_rows - offset
        low = 1
        high = min(remaining, DUCKDB_STANDARD_VECTOR_SIZE)
        best_rows = 0
        best_block: pa.Table | None = None
        while low <= high:
            candidate_rows = (low + high) // 2
            candidate = block.slice(offset, candidate_rows)
            candidate_bytes = max(1, int(estimate_table_bytes(candidate)))
            if candidate_bytes <= target_bytes:
                best_rows = candidate_rows
                best_block = candidate
                low = candidate_rows + 1
            else:
                high = candidate_rows - 1

        if best_block is None:
            single_row = block.slice(offset, 1)
            yield single_row
            offset += 1
            continue
        yield best_block
        offset += best_rows


def make_stream_block_metadata(
    table: pa.Table,
    payload: dict[str, Any],
    *,
    output_index: int,
) -> dict[str, Any]:
    """Build the bounded control object paired with one direct Ray block."""
    block = ensure_table(table)
    query_id, resource_unit_id, task_lease_id, attempt_id = _stream_identity(payload)
    index = int(output_index)
    if index < 0:
        raise ValueError("output_index must be >= 0")
    return {
        "protocol_version": RAY_UDF_STREAM_PROTOCOL_VERSION,
        "query_id": query_id,
        "producer_unit_id": resource_unit_id,
        "task_lease_id": task_lease_id,
        "attempt_id": attempt_id,
        "block_id": f"block:{task_lease_id}:{index}",
        # Zero-row Arrow objects still occupy Ray object-store metadata and
        # must have non-zero ownership for the lease model.
        "size_bytes": max(1, int(estimate_table_bytes(block))),
        "num_rows": int(block.num_rows),
        "names": list(block.schema.names),
    }


def make_stream_error_pair(
    payload: dict[str, Any],
    exc: BaseException,
) -> tuple[pa.Table, dict[str, Any]]:
    """Encode an application failure as one bounded block/control pair."""
    query_id, resource_unit_id, task_lease_id, attempt_id = _stream_identity(payload)
    exception_type = type(exc).__name__[:256] or "Exception"
    exception_message = str(exc)
    if len(exception_message) > _MAX_ERROR_TEXT_CHARS:
        exception_message = exception_message[:_MAX_ERROR_TEXT_CHARS] + "…<truncated>"
    exception_details = None
    from vane.ai.provider import ProviderCapabilityError

    if isinstance(exc, ProviderCapabilityError):
        exception_details = {
            "kind": "provider_capability",
            "provider": exc.provider,
            "model": exc.model,
            "capability": exc.capability,
            "original_error_summary": exc.original_error_summary,
        }
    sentinel = pa.table({})
    return sentinel, {
        "protocol_version": RAY_UDF_STREAM_PROTOCOL_VERSION,
        "event_kind": "error",
        "query_id": query_id,
        "producer_unit_id": resource_unit_id,
        "task_lease_id": task_lease_id,
        "attempt_id": attempt_id,
        "exception_type": exception_type,
        "exception_message": exception_message,
        "exception_details": exception_details,
    }


def validate_stream_block_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise TypeError(f"Ray UDF stream metadata must be a dict, got {type(metadata).__name__}")
    unknown = sorted(set(metadata) - _RAW_METADATA_FIELDS)
    missing = sorted(_RAW_METADATA_FIELDS - set(metadata))
    if unknown or missing:
        details = []
        if unknown:
            details.append(f"unknown={','.join(unknown)}")
        if missing:
            details.append(f"missing={','.join(missing)}")
        raise ValueError(f"invalid Ray UDF stream metadata: {' '.join(details)}")
    result = dict(metadata)
    if int(result["protocol_version"]) != RAY_UDF_STREAM_PROTOCOL_VERSION:
        raise ValueError(f"unsupported Ray UDF stream protocol version: {result['protocol_version']}")
    for name in (
        "query_id",
        "producer_unit_id",
        "task_lease_id",
        "attempt_id",
        "block_id",
    ):
        value = str(result[name] or "").strip()
        if not value:
            raise ValueError(f"Ray UDF stream metadata {name} must be non-empty")
        result[name] = value
    result["size_bytes"] = int(result["size_bytes"])
    result["num_rows"] = int(result["num_rows"])
    if result["size_bytes"] <= 0:
        raise ValueError("Ray UDF stream metadata size_bytes must be > 0")
    if result["num_rows"] < 0:
        raise ValueError("Ray UDF stream metadata num_rows must be >= 0")
    if not isinstance(result["names"], list) or not all(isinstance(name, str) for name in result["names"]):
        raise TypeError("Ray UDF stream metadata names must be a list of strings")
    return result


def validate_stream_error_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise TypeError(f"Ray UDF stream error metadata must be a dict, got {type(metadata).__name__}")
    unknown = sorted(set(metadata) - _ERROR_METADATA_FIELDS)
    missing = sorted(_ERROR_METADATA_FIELDS - set(metadata))
    if unknown or missing:
        details = []
        if unknown:
            details.append(f"unknown={','.join(unknown)}")
        if missing:
            details.append(f"missing={','.join(missing)}")
        raise ValueError(f"invalid Ray UDF stream error metadata: {' '.join(details)}")
    result = dict(metadata)
    if int(result["protocol_version"]) != RAY_UDF_STREAM_PROTOCOL_VERSION:
        raise ValueError(f"unsupported Ray UDF stream protocol version: {result['protocol_version']}")
    if result["event_kind"] != "error":
        raise ValueError("Ray UDF stream error metadata event_kind must be 'error'")
    for name in (
        "query_id",
        "producer_unit_id",
        "task_lease_id",
        "attempt_id",
        "exception_type",
    ):
        value = str(result[name] or "").strip()
        if not value:
            raise ValueError(f"Ray UDF stream error metadata {name} must be non-empty")
        result[name] = value
    result["exception_message"] = str(result["exception_message"] or "")
    if len(result["exception_message"]) > _MAX_ERROR_TEXT_CHARS + len("…<truncated>"):
        raise ValueError("Ray UDF stream error metadata exception_message is too large")
    details = result["exception_details"]
    if details is not None:
        if not isinstance(details, dict) or set(details) != _PROVIDER_CAPABILITY_DETAIL_FIELDS:
            raise ValueError("Ray UDF stream provider capability details have invalid fields")
        if details["kind"] != "provider_capability":
            raise ValueError("Ray UDF stream provider capability details have invalid kind")
        for name in ("provider", "model", "capability"):
            value = details[name]
            if not isinstance(value, str) or not value or len(value) > _MAX_ERROR_TEXT_CHARS:
                raise ValueError(f"Ray UDF stream provider capability detail {name} is invalid")
        summary = details["original_error_summary"]
        if summary is not None and (not isinstance(summary, str) or len(summary) > _MAX_ERROR_TEXT_CHARS):
            raise ValueError("Ray UDF stream provider capability original error summary is invalid")
    return result


__all__ = [
    "RAY_UDF_GENERATOR_BACKPRESSURE_OBJECTS",
    "RAY_UDF_STREAM_BUFFER_BLOCKS",
    "RAY_UDF_STREAM_OBJECTS_PER_BLOCK",
    "RAY_UDF_STREAM_PROTOCOL_VERSION",
    "iter_bounded_stream_blocks",
    "make_stream_block_metadata",
    "make_stream_error_pair",
    "task_payload_with_lease",
    "validate_stream_block_metadata",
    "validate_stream_error_metadata",
]
