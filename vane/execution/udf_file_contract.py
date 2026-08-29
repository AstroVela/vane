# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit FILE contracts at Python UDF boundaries.

Arrow transports FILE values as their canonical five-field STRUCT.  The
logical FILE identity is carried separately in the UDF payload, so workers can
validate values before user code runs and restore ``vane.File`` objects for
row UDFs without changing generic STRUCT behavior.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]

from vane.execution.udf_output_schema import _arrow_type_from_duckdb_pytype

_FILE_FIELDS = ("url", "content_type", "position", "size", "checksum")


def _invalid_input(message: str) -> Exception:
    import vane

    return vane.InvalidInputException(message)


def _type_id(dtype: Any) -> str:
    return str(dtype.id)


def _is_file_type(dtype: Any) -> bool:
    is_file = getattr(dtype, "is_file", None)
    return bool(callable(is_file) and is_file())


def _type_children(dtype: Any) -> dict[str, Any]:
    return dict(dtype.children)


def _sequence_child(dtype: Any) -> Any:
    return _type_children(dtype)["child"]


def _contains_file(dtype: Any | None) -> bool:
    if dtype is None:
        return False
    if _is_file_type(dtype):
        return True
    type_id = _type_id(dtype)
    if type_id in ("list", "array"):
        return _contains_file(_sequence_child(dtype))
    if type_id in ("struct", "union", "map"):
        return any(_contains_file(child) for _, child in dtype.children)
    return False


def contains_file_type(dtype: Any) -> bool:
    """Return whether a DuckDB Python type contains an explicit FILE leaf."""
    return _contains_file(dtype)


def _parse_file_type(type_name: Any, *, field: str) -> Any | None:
    if not isinstance(type_name, str) or re.search(r"\bFILE\b", type_name, flags=re.IGNORECASE) is None:
        return None

    import vane

    try:
        dtype = vane.type(type_name)
    except Exception as exc:
        raise _invalid_input(f"UDF payload field {field!r} contains an invalid SQL type") from exc
    return dtype if _contains_file(dtype) else None


def _expected_arrow_type(dtype: Any, *, boundary: str) -> pa.DataType:
    try:
        return _arrow_type_from_duckdb_pytype(dtype)
    except Exception as exc:
        raise _invalid_input(f"{boundary} uses an unsupported type containing FILE: {dtype}") from exc


def _validate_arrow_storage_type(
    actual: pa.DataType,
    dtype: Any,
    *,
    boundary: str,
    path: str,
    allow_untyped_null: bool = False,
) -> None:
    if allow_untyped_null and pa.types.is_null(actual):
        return
    if _is_file_type(dtype):
        valid_file_type = pa.types.is_struct(actual) and len(actual) == len(_FILE_FIELDS)
        if valid_file_type:
            for index, name in enumerate(_FILE_FIELDS):
                field = actual.field(index)
                if field.name != name:
                    valid_file_type = False
                    break
                if name in ("url", "content_type", "checksum"):
                    field_matches = pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
                else:
                    field_matches = pa.types.is_int64(field.type)
                if not field_matches:
                    valid_file_type = False
                    break
        if not valid_file_type:
            raise _invalid_input(
                f"{boundary} FILE at {path} must use the canonical five-field Arrow STRUCT, got {actual}"
            )
        return

    type_id = _type_id(dtype)
    if type_id == "list":
        if not (pa.types.is_list(actual) or pa.types.is_large_list(actual)):
            raise _invalid_input(f"{boundary} value at {path} must use an Arrow list type")
        _validate_arrow_storage_type(
            actual.value_type,
            _sequence_child(dtype),
            boundary=boundary,
            path=f"{path}[]",
            allow_untyped_null=allow_untyped_null,
        )
        return
    if type_id == "array":
        if not pa.types.is_fixed_size_list(actual):
            raise _invalid_input(f"{boundary} value at {path} must use an Arrow fixed-size list type")
        expected_size = int(_type_children(dtype)["size"])
        if actual.list_size != expected_size:
            raise _invalid_input(f"{boundary} value at {path} must have fixed size {expected_size}")
        _validate_arrow_storage_type(
            actual.value_type,
            _sequence_child(dtype),
            boundary=boundary,
            path=f"{path}[]",
            allow_untyped_null=allow_untyped_null,
        )
        return
    if type_id == "struct":
        if not pa.types.is_struct(actual):
            raise _invalid_input(f"{boundary} value at {path} must use an Arrow STRUCT type")
        for name, child in dtype.children:
            if not _contains_file(child):
                continue
            field_index = actual.get_field_index(name)
            if field_index < 0:
                raise _invalid_input(f"{boundary} STRUCT at {path} is missing FILE-bearing field {name!r}")
            _validate_arrow_storage_type(
                actual.field(field_index).type,
                child,
                boundary=boundary,
                path=f"{path}.{name}",
                allow_untyped_null=allow_untyped_null,
            )
        return
    if type_id == "map":
        if not pa.types.is_map(actual):
            raise _invalid_input(f"{boundary} value at {path} must use an Arrow MAP type")
        children = _type_children(dtype)
        if _contains_file(children["key"]):
            _validate_arrow_storage_type(
                actual.key_type,
                children["key"],
                boundary=boundary,
                path=f"{path}.key",
                allow_untyped_null=allow_untyped_null,
            )
        if _contains_file(children["value"]):
            _validate_arrow_storage_type(
                actual.item_type,
                children["value"],
                boundary=boundary,
                path=f"{path}.value",
                allow_untyped_null=allow_untyped_null,
            )
        return
    if type_id == "union":
        raise _invalid_input(f"{boundary} does not yet support UNION values containing FILE")


def _file_from_arrow_value(value: Any, *, boundary: str, path: str) -> Any:
    if not isinstance(value, Mapping):
        raise _invalid_input(f"{boundary} FILE value at {path} must be an Arrow STRUCT")
    if len(value) != len(_FILE_FIELDS) or set(value) != set(_FILE_FIELDS):
        raise _invalid_input(f"{boundary} FILE value at {path} must contain exactly the five FILE fields")

    import vane

    try:
        return vane.File(*(value[field] for field in _FILE_FIELDS))
    except (TypeError, ValueError, OverflowError) as exc:
        raise _invalid_input(f"{boundary} contains an invalid FILE value at {path}: {exc}") from exc


def _validate_arrow_value(value: Any, dtype: Any, *, boundary: str, path: str) -> None:
    if value is None:
        return
    if _is_file_type(dtype):
        _file_from_arrow_value(value, boundary=boundary, path=path)
        return

    type_id = _type_id(dtype)
    if type_id in ("list", "array"):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise _invalid_input(f"{boundary} value at {path} must be a sequence")
        child = _sequence_child(dtype)
        for index, item in enumerate(value):
            _validate_arrow_value(item, child, boundary=boundary, path=f"{path}[{index}]")
        return
    if type_id == "struct":
        if not isinstance(value, Mapping):
            raise _invalid_input(f"{boundary} value at {path} must be an Arrow STRUCT")
        for name, child in dtype.children:
            if _contains_file(child):
                _validate_arrow_value(value.get(name), child, boundary=boundary, path=f"{path}.{name}")
        return
    if type_id == "map":
        children = _type_children(dtype)
        entries = value.items() if isinstance(value, Mapping) else value
        for index, entry in enumerate(entries):
            if not isinstance(entry, Sequence) or len(entry) != 2:
                raise _invalid_input(f"{boundary} MAP value at {path} must contain key/value pairs")
            key, item = entry
            _validate_arrow_value(key, children["key"], boundary=boundary, path=f"{path}[{index}].key")
            _validate_arrow_value(item, children["value"], boundary=boundary, path=f"{path}[{index}].value")
        return
    if type_id == "union":
        raise _invalid_input(f"{boundary} does not yet support UNION values containing FILE")


def _materialize_native_value(value: Any, dtype: Any, *, boundary: str, path: str) -> Any:
    if value is None:
        return None
    if _is_file_type(dtype):
        return _file_from_arrow_value(value, boundary=boundary, path=path)

    type_id = _type_id(dtype)
    if type_id in ("list", "array"):
        child = _sequence_child(dtype)
        values = [
            _materialize_native_value(item, child, boundary=boundary, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
        return tuple(values) if type_id == "array" else values
    if type_id == "struct":
        result = dict(value)
        for name, child in dtype.children:
            if _contains_file(child):
                result[name] = _materialize_native_value(
                    value.get(name), child, boundary=boundary, path=f"{path}.{name}"
                )
        return result
    if type_id == "map":
        children = _type_children(dtype)
        entries = value.items() if isinstance(value, Mapping) else value
        pairs = [
            (
                _materialize_native_value(key, children["key"], boundary=boundary, path=f"{path}[{index}].key"),
                _materialize_native_value(item, children["value"], boundary=boundary, path=f"{path}[{index}].value"),
            )
            for index, (key, item) in enumerate(entries)
        ]
        try:
            return dict(pairs)
        except TypeError:
            return {"key": [key for key, _ in pairs], "value": [item for _, item in pairs]}
    if type_id == "union":
        raise _invalid_input(f"{boundary} does not yet support UNION values containing FILE")
    return value


def _canonicalize_native_output(value: Any, dtype: Any, *, boundary: str, path: str) -> Any:
    if value is None:
        return None
    if _is_file_type(dtype):
        import vane

        if not isinstance(value, vane.File):
            raise _invalid_input(f"{boundary} FILE value at {path} must be vane.File or NULL")
        return {field: getattr(value, field) for field in _FILE_FIELDS}

    type_id = _type_id(dtype)
    if type_id in ("list", "array"):
        if not isinstance(value, (list, tuple)):
            raise _invalid_input(f"{boundary} value at {path} must be a sequence")
        child = _sequence_child(dtype)
        return [
            _canonicalize_native_output(item, child, boundary=boundary, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if type_id == "struct":
        if not isinstance(value, Mapping):
            raise _invalid_input(f"{boundary} value at {path} must be a mapping")
        result = dict(value)
        for name, child in dtype.children:
            if _contains_file(child):
                result[name] = _canonicalize_native_output(
                    value.get(name), child, boundary=boundary, path=f"{path}.{name}"
                )
        return result
    if type_id == "map":
        if not isinstance(value, Mapping):
            raise _invalid_input(f"{boundary} MAP value at {path} must be a mapping")
        children = _type_children(dtype)
        if set(value) == {"key", "value"} and isinstance(value["key"], (list, tuple)) and isinstance(
            value["value"], (list, tuple)
        ):
            entries = zip(value["key"], value["value"], strict=True)
        else:
            entries = value.items()
        return [
            (
                _canonicalize_native_output(key, children["key"], boundary=boundary, path=f"{path}[{index}].key"),
                _canonicalize_native_output(item, children["value"], boundary=boundary, path=f"{path}[{index}].value"),
            )
            for index, (key, item) in enumerate(entries)
        ]
    if type_id == "union":
        raise _invalid_input(f"{boundary} does not yet support UNION values containing FILE")
    return value


def validate_file_arrow_array(
    array: Any,
    dtype: Any,
    *,
    boundary: str,
    allow_untyped_null: bool = False,
) -> None:
    """Validate one Arrow array governed by a logical type containing FILE."""
    _validate_arrow_storage_type(
        array.type,
        dtype,
        boundary=boundary,
        path="column",
        allow_untyped_null=allow_untyped_null,
    )
    for row, value in enumerate(array.to_pylist()):
        _validate_arrow_value(value, dtype, boundary=boundary, path=f"row {row}")


def validate_file_arrow_storage_type(
    actual: pa.DataType,
    dtype: Any,
    *,
    boundary: str,
    allow_untyped_null: bool = False,
) -> None:
    """Validate the Arrow storage shape of a logical type containing FILE."""
    _validate_arrow_storage_type(
        actual,
        dtype,
        boundary=boundary,
        path="column",
        allow_untyped_null=allow_untyped_null,
    )


@dataclass(frozen=True)
class FileUDFContract:
    """Parsed FILE-bearing input and output types for one runtime payload."""

    udf_name: str
    input_types: tuple[Any | None, ...]
    output_types: tuple[Any | None, ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> FileUDFContract:
        raw_inputs = payload.get("input_types") or []
        input_types = (
            tuple(_parse_file_type(type_name, field="input_types") for type_name in raw_inputs)
            if isinstance(raw_inputs, (list, tuple))
            else ()
        )

        output_types: list[Any | None] = []
        method_return_type = payload.get("method_return_type")
        if method_return_type is not None:
            output_types.append(_parse_file_type(method_return_type, field="method_return_type"))
        else:
            output_schema = payload.get("output_schema") or []
            if isinstance(output_schema, (list, tuple)):
                for entry in output_schema:
                    if not isinstance(entry, Mapping):
                        output_types.append(None)
                        continue
                    kind = str(entry.get("kind") or "duckdb_type").lower()
                    output_types.append(
                        _parse_file_type(entry.get("type"), field="output_schema")
                        if kind == "duckdb_type"
                        else None
                    )

        return cls(
            udf_name=str(payload.get("udf_name") or "<unknown>"),
            input_types=input_types,
            output_types=tuple(output_types),
        )

    @property
    def has_file_inputs(self) -> bool:
        return any(dtype is not None for dtype in self.input_types)

    @property
    def has_file_outputs(self) -> bool:
        return any(dtype is not None for dtype in self.output_types)

    def _validate_column_count(self, table: pa.Table, types: tuple[Any | None, ...], *, boundary: str) -> None:
        if types and table.num_columns != len(types):
            raise _invalid_input(
                f"{boundary} has {table.num_columns} columns but its logical contract declares {len(types)}"
            )

    def validate_input_table(self, table: pa.Table) -> None:
        if not self.has_file_inputs:
            return
        boundary = f"UDF {self.udf_name!r} input"
        self._validate_column_count(table, self.input_types, boundary=boundary)
        for index, dtype in enumerate(self.input_types):
            if dtype is not None:
                validate_file_arrow_array(table.column(index), dtype, boundary=f"{boundary} column {index}")

    def materialize_scalar_inputs(self, table: pa.Table) -> list[list[Any]]:
        if not self.has_file_inputs:
            return [column.to_pylist() for column in table.columns]
        boundary = f"UDF {self.udf_name!r} input"
        self._validate_column_count(table, self.input_types, boundary=boundary)
        columns: list[list[Any]] = []
        for index, column in enumerate(table.columns):
            values = column.to_pylist()
            dtype = self.input_types[index]
            if dtype is not None:
                _validate_arrow_storage_type(
                    column.type,
                    dtype,
                    boundary=f"{boundary} column {index}",
                    path="column",
                )
                values = [
                    _materialize_native_value(
                        value,
                        dtype,
                        boundary=f"{boundary} column {index}",
                        path=f"row {row}",
                    )
                    for row, value in enumerate(values)
                ]
            columns.append(values)
        return columns

    def scalar_outputs_to_array(self, outputs: list[Any]) -> pa.Array:
        if not self.has_file_outputs:
            return pa.array(outputs)
        if len(self.output_types) != 1 or self.output_types[0] is None:
            raise _invalid_input(f"UDF {self.udf_name!r} scalar FILE output contract must declare one column")
        dtype = self.output_types[0]
        boundary = f"UDF {self.udf_name!r} output"
        canonical = [
            _canonicalize_native_output(value, dtype, boundary=boundary, path=f"row {row}")
            for row, value in enumerate(outputs)
        ]
        try:
            return pa.array(canonical, type=_expected_arrow_type(dtype, boundary=boundary))
        except Exception:
            raise _invalid_input(f"{boundary} could not be encoded using its declared FILE type") from None

    def validate_output_table(self, table: pa.Table) -> None:
        if not self.has_file_outputs:
            return
        boundary = f"UDF {self.udf_name!r} output"
        self._validate_column_count(table, self.output_types, boundary=boundary)
        for index, dtype in enumerate(self.output_types):
            if dtype is not None:
                validate_file_arrow_array(
                    table.column(index),
                    dtype,
                    boundary=f"{boundary} column {index}",
                    allow_untyped_null=True,
                )


__all__ = [
    "FileUDFContract",
    "contains_file_type",
    "validate_file_arrow_array",
    "validate_file_arrow_storage_type",
]
