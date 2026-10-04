# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from typing import Any

import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]


class _ArrowOpaqueCompatType(pa.ExtensionType):
    """Backport Arrow's canonical opaque type for supported older PyArrow releases."""

    def __init__(self, storage_type: pa.DataType, type_name: str, vendor_name: str) -> None:
        self.type_name = type_name
        self.vendor_name = vendor_name
        super().__init__(storage_type, "arrow.opaque")

    def __arrow_ext_serialize__(self) -> bytes:
        return json.dumps(
            {"type_name": self.type_name, "vendor_name": self.vendor_name},
            separators=(",", ":"),
        ).encode()

    @classmethod
    def __arrow_ext_deserialize__(
        cls,
        storage_type: pa.DataType,
        serialized: bytes,
    ) -> _ArrowOpaqueCompatType:
        metadata = json.loads(serialized.decode())
        return cls(storage_type, metadata["type_name"], metadata["vendor_name"])

    def __reduce__(self) -> tuple[Any, tuple[pa.DataType, str, str]]:
        return type(self), (self.storage_type, self.type_name, self.vendor_name)


def _duckdb_bit_arrow_type() -> pa.DataType:
    opaque = getattr(pa, "opaque", None)
    if callable(opaque):
        return opaque(pa.binary(), "bit", "DuckDB")
    return _ArrowOpaqueCompatType(pa.binary(), "bit", "DuckDB")


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


def _duckdb_pytype_contains_governed(dt: Any) -> bool:
    is_file = getattr(dt, "is_file", None)
    is_image = getattr(dt, "is_image", None)
    if (callable(is_file) and is_file()) or (callable(is_image) and is_image()):
        return True
    type_id = str(dt.id)
    if type_id in ("list", "array", "tensor"):
        children = dict(dt.children)
        child = children["dtype"] if type_id == "tensor" else children["child"]
        return _duckdb_pytype_contains_governed(child)
    if type_id in ("struct", "union", "map"):
        return any(_duckdb_pytype_contains_governed(child) for _, child in dt.children)
    return False


def _arrow_type_from_duckdb_pytype(dt: Any) -> pa.DataType:
    type_name = str(dt)
    if type_name.startswith("FIXEDBINARY(") and type_name.endswith(")"):
        return pa.binary(int(type_name[12:-1]))
    if dt.is_image():
        from vane._image import image_arrow_type

        return image_arrow_type(dt)
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
        "bit": _duckdb_bit_arrow_type,
        "timestamp": lambda: pa.timestamp("us"),
        "timestamp_s": lambda: pa.timestamp("s"),
        "timestamp_ms": lambda: pa.timestamp("ms"),
        "timestamp_ns": lambda: pa.timestamp("ns"),
        "date": pa.date32,
        "time": lambda: pa.time64("us"),
        "time_ns": lambda: pa.time64("ns"),
        "interval": pa.month_day_nano_interval,
        "json": pa.string,
        # Arrow has no 128-bit integer type.  decimal128 is limited to 38
        # digits and DuckDB does not import decimal256, so decimal strings
        # preserve the complete HUGEINT and UHUGEINT domains.
        "hugeint": pa.string,
        "uhugeint": pa.string,
        "bignum": pa.string,
        "uuid": pa.string,
        "timestamp with time zone": lambda: pa.timestamp("us", tz="UTC"),
        "time with time zone": pa.string,
        "enum": pa.string,
        "null": pa.null,
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
    if type_id == "tensor":
        children = dict(dt.children)
        from vane._tensor import tensor_arrow_type

        return tensor_arrow_type(
            _arrow_type_from_duckdb_pytype(children["dtype"]),
            children["shape"],
        )
    if type_id == "struct":
        return pa.struct([(name, _arrow_type_from_duckdb_pytype(child_dt)) for name, child_dt in dt.children])
    if type_id == "union":
        if _duckdb_pytype_contains_governed(dt):
            raise ValueError(
                "UNION values containing governed logical types are not supported at Python UDF boundaries"
            )
        return pa.union(
            [pa.field(name, _arrow_type_from_duckdb_pytype(child_dt)) for name, child_dt in dt.children if name],
            mode="sparse",
        )
    if type_id == "map":
        children = dict(dt.children)
        return pa.map_(
            _arrow_type_from_duckdb_pytype(children["key"]),
            _arrow_type_from_duckdb_pytype(children["value"]),
        )

    raise ValueError(f"unsupported DuckDB type id for empty UDF output: {type_id!r}")


def _arrow_type_from_output_schema_entry(entry: dict[str, Any]) -> pa.DataType:
    kind = str(entry.get("kind") or "").strip().lower()
    if kind == "tensor":
        from vane._tensor import tensor_arrow_type

        dtype = _arrow_type_from_name(str(entry.get("dtype") or ""))
        shape = entry.get("shape") or []
        if not shape:
            return dtype
        if all(dim is not None for dim in shape) and any(dim <= 0 for dim in shape):
            raise ValueError(f"tensor output shape must contain positive dimensions: {shape!r}")
        return tensor_arrow_type(dtype, shape)
    return _arrow_type_from_name(str(entry.get("type") or ""))


def _materialized_type_supported(dtype: pa.DataType, *, top_level: bool = True) -> bool:
    if isinstance(dtype, pa.FixedShapeTensorType):
        return top_level and (pa.types.is_integer(dtype.value_type) or pa.types.is_floating(dtype.value_type))
    if pa.types.is_list(dtype) or pa.types.is_fixed_size_list(dtype):
        return _materialized_type_supported(dtype.value_type, top_level=False)
    if pa.types.is_struct(dtype):
        return all(_materialized_type_supported(field.type, top_level=False) for field in dtype)
    return any(
        check(dtype)
        for check in (
            pa.types.is_null,
            pa.types.is_boolean,
            pa.types.is_integer,
            pa.types.is_floating,
            pa.types.is_string,
            pa.types.is_binary,
        )
    )


def materialized_output_schema(payload: dict[str, Any]) -> pa.Schema:
    """Resolve the types supported by the materialized column encoder."""
    import vane

    entries = payload.get("output_schema")
    if not entries:
        raise ValueError("UDF requires output_schema")
    fields = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"]:
            raise ValueError("UDF output_schema entries require a non-empty column name")
        name = entry["name"]
        if str(entry.get("kind") or "").lower() != "tensor":
            logical_type = vane.type(str(entry.get("type") or ""))
            if _duckdb_pytype_contains_governed(logical_type):
                raise TypeError(f"UDF output column {name!r} does not support FILE or IMAGE")
        dtype = _arrow_type_from_output_schema_entry(entry)
        if not _materialized_type_supported(dtype):
            raise TypeError(f"UDF output column {name!r} has unsupported type {dtype}")
        fields.append(pa.field(name, dtype))
    names = [field.name for field in fields]
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("UDF output column names must be unambiguous")
    return pa.schema(fields)


def columns_to_output_table(result: Any, schema: pa.Schema, *, udf_name: str) -> pa.Table:
    """Encode materialized columns on the actor thread without inferring types."""
    import numpy as np

    from vane._tensor import _NUMPY_DTYPES

    boundary = f"UDF {udf_name!r} output"
    if not isinstance(result, dict):
        raise TypeError(f"{boundary} must be a materialized dict, got {type(result).__name__}")
    if set(result) != set(schema.names):
        raise ValueError(f"{boundary} must contain exactly the declared columns {schema.names}")
    arrays = []
    row_count = None
    for field in schema:
        values = result[field.name]
        column_boundary = f"{boundary} column {field.name!r} (expected {field.type})"
        if not isinstance(values, (list, tuple, np.ndarray)):
            raise TypeError(f"{column_boundary} requires a materialized list, tuple or ndarray")
        if isinstance(field.type, pa.FixedShapeTensorType):
            if not isinstance(values, np.ndarray):
                raise TypeError(f"{column_boundary} requires a NumPy ndarray")
            if (
                values.ndim != len(field.type.shape) + 1
                or tuple(values.shape[1:]) != tuple(field.type.shape)
                or values.dtype != _NUMPY_DTYPES[field.type.value_type]
                or not values.flags.c_contiguous
            ):
                raise ValueError(
                    f"{column_boundary} requires matching dtype, shape and C-contiguous storage; "
                    f"got dtype={values.dtype}, shape={values.shape}"
                )
        elif isinstance(values, np.ndarray) and values.ndim != 1:
            raise ValueError(f"{column_boundary} requires a one-dimensional ndarray, got shape={values.shape}")
        length = len(values)
        if row_count is not None and length != row_count:
            raise ValueError(f"{column_boundary} has {length} rows, expected {row_count}")
        row_count = length
        try:
            if isinstance(field.type, pa.FixedShapeTensorType) and length:
                array = pa.FixedShapeTensorArray.from_numpy_ndarray(values)
            else:
                array = pa.array(values if length else [], type=field.type)
        except (TypeError, ValueError, OverflowError, pa.ArrowException):
            # Arrow error messages may include complete input values.
            raise ValueError(f"{column_boundary} could not encode {type(values).__name__}") from None
        arrays.append(array)
    return pa.Table.from_arrays(arrays, schema=schema)


def empty_output_table_from_schema(output_schema: Any, *, output_contract_types: Any = None) -> pa.Table:
    if not output_schema:
        raise ValueError("empty UDF output requires payload.output_schema")
    entries = list(output_schema)
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("payload.output_schema entries must be dicts")

    from vane.execution.udf_file_contract import FileUDFContract

    contract_payload = {"udf_name": "<empty>", "output_schema": entries}
    if output_contract_types is not None:
        contract_payload["output_contract_types"] = output_contract_types
    file_contract = FileUDFContract.from_payload(contract_payload)
    output_names = [str(entry.get("name") or "") for entry in entries]
    logical_table = (
        file_contract.native_output_rows_to_table([], output_names) if file_contract.has_governed_outputs else None
    )

    arrays = {}
    for index, entry in enumerate(entries):
        name = str(entry.get("name") or "")
        if logical_table is not None and file_contract.output_types[index] is not None:
            arrays[name] = logical_table.column(index)
            continue
        try:
            arrays[name] = pa.array([], type=_arrow_type_from_output_schema_entry(entry))
        except Exception:
            if logical_table is None:
                raise
            kind = str(entry.get("kind") or "duckdb_type").strip().lower()
            if kind != "duckdb_type":
                raise
            import vane

            vane.type(str(entry.get("type") or ""))
            arrays[name] = logical_table.column(index)
    return pa.table(arrays)


def empty_output_table_from_payload(payload: dict[str, Any] | None) -> pa.Table:
    payload = payload or {}
    return empty_output_table_from_schema(
        payload.get("output_schema"),
        output_contract_types=payload.get("output_contract_types"),
    )


__all__ = ["empty_output_table_from_payload", "empty_output_table_from_schema"]
