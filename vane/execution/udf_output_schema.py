# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from typing import Any

import pyarrow as pa


def _arrow_type_from_name(type_name: str) -> pa.DataType:
    normalized = str(type_name or "").strip().upper()
    if normalized in ("BOOLEAN", "BOOL"):
        return pa.bool_()
    if normalized in ("TINYINT", "INT8"):
        return pa.int8()
    if normalized in ("UTINYINT", "UINT8"):
        return pa.uint8()
    if normalized in ("SMALLINT", "INT16"):
        return pa.int16()
    if normalized in ("USMALLINT", "UINT16"):
        return pa.uint16()
    if normalized in ("INTEGER", "INT", "INT32"):
        return pa.int32()
    if normalized in ("UINTEGER", "UINT", "UINT32"):
        return pa.uint32()
    if normalized in ("BIGINT", "INT64", "LONG"):
        return pa.int64()
    if normalized in ("UBIGINT", "UINT64", "ULONG"):
        return pa.uint64()
    if normalized in ("FLOAT", "FLOAT32", "REAL"):
        return pa.float32()
    if normalized in ("DOUBLE", "FLOAT64"):
        return pa.float64()
    if normalized in ("VARCHAR", "STRING"):
        return pa.string()
    if normalized in ("BLOB", "BINARY"):
        return pa.binary()
    if normalized == "DATE":
        return pa.date32()
    if normalized == "TIME":
        return pa.time64("us")
    if normalized in ("TIMESTAMP", "TIMESTAMP_NS", "TIMESTAMP_MS", "TIMESTAMP_S"):
        unit = {
            "TIMESTAMP_NS": "ns",
            "TIMESTAMP_MS": "ms",
            "TIMESTAMP_S": "s",
        }.get(normalized, "us")
        return pa.timestamp(unit)

    decimal = re.fullmatch(r"DECIMAL\((\d+),\s*(\d+)\)", normalized)
    if decimal:
        return pa.decimal128(int(decimal.group(1)), int(decimal.group(2)))

    try:
        return _arrow_type_from_duckdb_type(type_name)
    except Exception as exc:
        raise ValueError(f"unsupported UDF output type for empty output: {type_name!r}") from exc


def _arrow_type_from_duckdb_type(type_name: str) -> pa.DataType:
    import vane

    return _arrow_type_from_duckdb_pytype(vane.type(type_name))


def _arrow_type_from_duckdb_pytype(dt: Any) -> pa.DataType:
    type_id = str(dt.id)
    basic = {
        "varchar": pa.string,
        "integer": pa.int32,
        "bigint": pa.int64,
        "smallint": pa.int16,
        "tinyint": pa.int8,
        "uinteger": lambda: pa.uint32(),
        "ubigint": lambda: pa.uint64(),
        "usmallint": lambda: pa.uint16(),
        "utinyint": lambda: pa.uint8(),
        "float": pa.float32,
        "double": pa.float64,
        "boolean": pa.bool_,
        "blob": pa.binary,
        "timestamp": lambda: pa.timestamp("us"),
        "timestamp_s": lambda: pa.timestamp("s"),
        "timestamp_ms": lambda: pa.timestamp("ms"),
        "timestamp_ns": lambda: pa.timestamp("ns"),
        "date": pa.date32,
        "time": lambda: pa.time64("us"),
        "interval": lambda: pa.duration("us"),
        "json": pa.string,
        "hugeint": lambda: pa.decimal128(38, 0),
    }
    factory = basic.get(type_id)
    if factory is not None:
        return factory()

    if type_id == "decimal":
        children = dict(dt.children)
        return pa.decimal128(int(children["precision"]), int(children["scale"]))
    if type_id == "list":
        return pa.list_(_arrow_type_from_duckdb_pytype(dt.children[0][1]))
    if type_id == "array":
        children = dict(dt.children)
        return pa.list_(_arrow_type_from_duckdb_pytype(children["child"]), list_size=int(children["size"]))
    if type_id == "struct":
        return pa.struct([(name, _arrow_type_from_duckdb_pytype(child_dt)) for name, child_dt in dt.children])
    if type_id == "map":
        children = dict(dt.children)
        return pa.map_(
            _arrow_type_from_duckdb_pytype(children["key"]), _arrow_type_from_duckdb_pytype(children["value"])
        )

    raise ValueError(f"unsupported DuckDB type id for empty UDF output: {type_id!r}")


def _arrow_type_from_output_schema_entry(entry: dict[str, Any]) -> pa.DataType:
    kind = str(entry.get("kind") or "").strip().lower()
    if kind == "tensor":
        dtype = _arrow_type_from_name(str(entry.get("dtype") or ""))
        shape = [int(dim) for dim in entry.get("shape") or []]
        if not shape:
            return dtype
        if any(dim <= 0 for dim in shape):
            raise ValueError(f"tensor output shape must contain positive dimensions: {shape!r}")
        return pa.fixed_shape_tensor(dtype, tuple(shape))
    return _arrow_type_from_name(str(entry.get("type") or ""))


def empty_output_table_from_schema(output_schema: Any) -> pa.Table:
    if not output_schema:
        raise ValueError("empty UDF output requires payload.output_schema")
    arrays = {}
    for entry in output_schema:
        if not isinstance(entry, dict):
            raise ValueError("payload.output_schema entries must be dicts")
        name = str(entry.get("name") or "")
        arrays[name] = pa.array([], type=_arrow_type_from_output_schema_entry(entry))
    return pa.table(arrays)


def empty_output_table_from_payload(payload: dict[str, Any] | None) -> pa.Table:
    payload = payload or {}
    return empty_output_table_from_schema(payload.get("output_schema"))


__all__ = ["empty_output_table_from_payload", "empty_output_table_from_schema"]
