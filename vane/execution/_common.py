# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for unified UDF executor modules."""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Iterable

_UDF_CALLABLE_CACHE_LOCK = threading.Lock()
_UDF_CALLABLE_CACHE: OrderedDict[str, Any] = OrderedDict()
_UDF_CALLABLE_CACHE_STATS = {
    "python_udf_callable_cache_hit": 0,
    "python_udf_callable_cache_miss": 0,
    "python_udf_callable_cache_bypass": 0,
}
_TRUTHY_FALSE_VALUES = ("", "0", "false", "no", "off")


# ── Table helpers ────────────────────────────────────────────────────────────


def ensure_table(rows: Any) -> pa.Table:
    """Coerce various Arrow types to pa.Table."""
    if isinstance(rows, pa.Table):
        return rows
    if isinstance(rows, pa.RecordBatch):
        return pa.Table.from_batches([rows])
    if isinstance(rows, pa.RecordBatchReader):
        return pa.Table.from_batches(list(rows))
    raise TypeError("rows must be a pyarrow Table, RecordBatch, or RecordBatchReader")


def estimate_table_bytes(table: pa.Table) -> int:
    """Return a positive retained-byte estimate for every non-empty table."""
    return max(0, int(table.nbytes), int(table.num_rows))


def iter_table_batches(table: pa.Table, batch_size: int | None) -> Iterable[pa.Table]:
    """Yield sub-tables of at most *batch_size* rows."""
    if batch_size is None or batch_size <= 0 or table.num_rows <= batch_size:
        yield table
        return
    for start in range(0, table.num_rows, batch_size):
        yield table.slice(start, batch_size)


# ── Pickle deserialization ───────────────────────────────────────────────────


def load_udf_from_payload(payload: dict[str, Any]) -> Any:
    """Deserialize the UDF callable from a payload's function_pickle field."""
    from vane import pickle as duckdb_pickle

    function_pickle = _payload_pickle_bytes(payload)
    return duckdb_pickle.loads(function_pickle)


def _payload_pickle_bytes(payload: dict[str, Any]) -> bytes:
    function_pickle = payload.get("function_pickle")
    if type(function_pickle) is not bytes:
        raise TypeError("function_pickle must be bytes in a UDF payload")
    return function_pickle


def _cache_component(payload: dict[str, Any], key: str) -> bytes:
    value = payload.get(key)
    if value is None:
        return b"<none>"
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    return repr(value).encode("utf-8", errors="replace")


def udf_cache_key(payload: dict[str, Any]) -> str:
    """Return a stable cache key for a serialized UDF payload."""
    digest = hashlib.sha256()
    digest.update(_payload_pickle_bytes(payload))
    for key in (
        "call_mode",
        "scalar_udf_type",
        "execution_backend",
        "null_handling",
        "exception_handling",
        "output_schema",
    ):
        digest.update(b"\0")
        digest.update(key.encode())
        digest.update(b"=")
        digest.update(_cache_component(payload, key))
    return digest.hexdigest()


def clear_udf_callable_cache() -> None:
    """Clear the process-local UDF callable cache."""
    with _UDF_CALLABLE_CACHE_LOCK:
        _UDF_CALLABLE_CACHE.clear()
        for key in _UDF_CALLABLE_CACHE_STATS:
            _UDF_CALLABLE_CACHE_STATS[key] = 0


def udf_callable_cache_stats() -> dict[str, int]:
    """Return process-local callable cache counters."""
    with _UDF_CALLABLE_CACHE_LOCK:
        return dict(_UDF_CALLABLE_CACHE_STATS)


def callable_cache_enabled(payload: dict[str, Any]) -> bool:
    if payload.get("side_effects"):
        return False
    return os.getenv("VANE_CPU_UDF_CALLABLE_CACHE", "").strip().lower() not in _TRUTHY_FALSE_VALUES


def load_udf_from_payload_cached(payload: dict[str, Any], max_entries: int | None = None) -> Any:
    """Deserialize a UDF callable with a process-local LRU cache."""
    if payload.get("side_effects"):
        with _UDF_CALLABLE_CACHE_LOCK:
            _UDF_CALLABLE_CACHE_STATS["python_udf_callable_cache_bypass"] += 1
        return load_udf_from_payload(payload)
    if max_entries is None:
        max_entries = 64
    if max_entries <= 0:
        with _UDF_CALLABLE_CACHE_LOCK:
            _UDF_CALLABLE_CACHE_STATS["python_udf_callable_cache_bypass"] += 1
        return load_udf_from_payload(payload)

    key = udf_cache_key(payload)
    with _UDF_CALLABLE_CACHE_LOCK:
        if key in _UDF_CALLABLE_CACHE:
            cached = _UDF_CALLABLE_CACHE[key]
            _UDF_CALLABLE_CACHE.move_to_end(key)
            _UDF_CALLABLE_CACHE_STATS["python_udf_callable_cache_hit"] += 1
            return cached

    udf = load_udf_from_payload(payload)
    with _UDF_CALLABLE_CACHE_LOCK:
        if key in _UDF_CALLABLE_CACHE:
            cached = _UDF_CALLABLE_CACHE[key]
            _UDF_CALLABLE_CACHE.move_to_end(key)
            _UDF_CALLABLE_CACHE_STATS["python_udf_callable_cache_hit"] += 1
            return cached
        _UDF_CALLABLE_CACHE[key] = udf
        _UDF_CALLABLE_CACHE_STATS["python_udf_callable_cache_miss"] += 1
        while len(_UDF_CALLABLE_CACHE) > max_entries:
            _UDF_CALLABLE_CACHE.popitem(last=False)
    return udf


__all__ = [
    "callable_cache_enabled",
    "clear_udf_callable_cache",
    "ensure_table",
    "iter_table_batches",
    "load_udf_from_payload",
    "load_udf_from_payload_cached",
    "udf_cache_key",
    "udf_callable_cache_stats",
]
