# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit FILE contracts at Python UDF boundaries.

Arrow transports FILE values as their canonical five-field STRUCT.  The
logical FILE identity is carried separately in the UDF payload, so workers can
validate values before user code runs and restore ``vane.File`` objects for
row UDFs without changing generic STRUCT behavior.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import time as datetime_time
from typing import Any
from uuid import UUID

import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]
import pyarrow.compute as pc  # type: ignore[import-not-found, import-untyped, unused-ignore]

from vane.execution.udf_output_schema import _arrow_type_from_duckdb_pytype

_FILE_FIELDS = ("url", "content_type", "position", "size", "checksum")
_TENSOR_TYPE_PATTERN = re.compile(r"^TENSOR\((.*),\s*\[([0-9,\s]+)\]\)$", flags=re.IGNORECASE)
_NATIVE_OUTPUT_ENCODED_TYPE_IDS = {
    "bignum",
    "hugeint",
    "time with time zone",
    "uhugeint",
    "uuid",
}


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
    children = _type_children(dtype)
    return children["dtype"] if _type_id(dtype) == "tensor" else children["child"]


def _tensor_shape(dtype: Any) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in _type_children(dtype)["shape"])


def _fixed_sequence_size(dtype: Any) -> int:
    if _type_id(dtype) != "tensor":
        return int(_type_children(dtype)["size"])
    size = 1
    for dimension in _tensor_shape(dtype):
        size *= dimension
    return size


def _contains_file(dtype: Any | None) -> bool:
    if dtype is None:
        return False
    if _is_file_type(dtype):
        return True
    type_id = _type_id(dtype)
    if type_id in ("list", "array", "tensor"):
        return _contains_file(_sequence_child(dtype))
    if type_id in ("struct", "union", "map"):
        return any(_contains_file(child) for _, child in dtype.children)
    return False


def _contains_bit(dtype: Any | None) -> bool:
    if dtype is None:
        return False
    type_id = _type_id(dtype)
    if type_id == "bit":
        return True
    if type_id in ("list", "array", "tensor"):
        return _contains_bit(_sequence_child(dtype))
    if type_id in ("struct", "map"):
        return any(_contains_bit(child) for _, child in dtype.children)
    # UNION tags are not materialized at the Python boundary yet. Treat the
    # whole UNION as opaque instead of annotating only some member storage.
    return False


def _requires_native_output_encoding(dtype: Any | None) -> bool:
    """Return whether native output needs recursive Arrow-safe encoding."""
    if dtype is None:
        return False
    if _is_file_type(dtype):
        return True
    type_id = _type_id(dtype)
    if type_id in _NATIVE_OUTPUT_ENCODED_TYPE_IDS:
        return True
    if type_id in ("list", "array", "tensor"):
        return _requires_native_output_encoding(_sequence_child(dtype))
    if type_id in ("struct", "map"):
        return any(_requires_native_output_encoding(child) for _, child in dtype.children)
    # UNION tags are not materialized at the Python boundary yet. Keep the
    # entire value opaque, matching FILE/BIT input handling.
    return False


def _native_map_key_is_hashable(dtype: Any) -> bool:
    if _is_file_type(dtype):
        return True
    type_id = _type_id(dtype)
    if type_id in ("list", "struct", "map", "tensor", "union"):
        return False
    if type_id == "array":
        return _native_map_key_is_hashable(_sequence_child(dtype))
    return True


def contains_file_type(dtype: Any) -> bool:
    """Return whether a DuckDB Python type contains an explicit FILE leaf."""
    return _contains_file(dtype)


def _parse_file_type(type_name: Any, *, field: str) -> Any | None:
    if not isinstance(type_name, str) or re.search(r"\bFILE\b", type_name, flags=re.IGNORECASE) is None:
        return None
    dtype = _parse_declared_type(type_name, field=field)
    return dtype if _contains_file(dtype) else None


def _parse_declared_type(type_name: Any, *, field: str) -> Any | None:
    if not isinstance(type_name, str):
        return None

    import vane

    try:
        return vane.type(type_name)
    except Exception as exc:
        tensor_match = _TENSOR_TYPE_PATTERN.fullmatch(type_name.strip())
        if tensor_match is None:
            raise _invalid_input(f"UDF payload field {field!r} contains an invalid SQL type") from exc
        child = _parse_declared_type(tensor_match.group(1).strip(), field=field)
        if child is None:
            raise _invalid_input(f"UDF payload field {field!r} contains an invalid TENSOR dtype") from exc
        try:
            shape = tuple(int(dimension.strip()) for dimension in tensor_match.group(2).split(","))
            return vane.tensor_type(child, shape)
        except Exception as tensor_error:
            raise _invalid_input(f"UDF payload field {field!r} contains an invalid TENSOR type") from tensor_error


def _parse_tensor_type(
    entry: Mapping[str, Any],
    *,
    field: str,
    file_only: bool = False,
) -> Any | None:
    child = (
        _parse_file_type(entry.get("dtype"), field=field)
        if file_only
        else _parse_declared_type(entry.get("dtype"), field=field)
    )
    if child is None:
        if file_only:
            return None
        raise _invalid_input(f"UDF payload field {field!r} contains an invalid TENSOR dtype")
    raw_shape = entry.get("shape")
    if not isinstance(raw_shape, (list, tuple)) or not raw_shape:
        raise _invalid_input(f"UDF payload field {field!r} contains an invalid TENSOR shape")
    if any(isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0 for dimension in raw_shape):
        raise _invalid_input(f"UDF payload field {field!r} contains an invalid TENSOR shape")

    import vane

    return vane.tensor_type(child, tuple(raw_shape))


def _expected_arrow_type(dtype: Any, *, boundary: str) -> pa.DataType:
    try:
        return _arrow_type_from_duckdb_pytype(dtype)
    except Exception as exc:
        raise _invalid_input(f"{boundary} uses an unsupported type containing FILE: {dtype}") from exc


def _optional_struct_field_index(
    actual: pa.StructType,
    name: str,
    *,
    boundary: str,
    path: str,
) -> int | None:
    folded_name = name.casefold()
    matches = [index for index, field in enumerate(actual) if field.name.casefold() == folded_name]
    if len(matches) > 1:
        raise _invalid_input(f"{boundary} STRUCT at {path} has ambiguous field names matching {name!r}")
    return matches[0] if matches else None


def _struct_field_index(
    actual: pa.StructType,
    name: str,
    *,
    boundary: str,
    path: str,
) -> int:
    field_index = _optional_struct_field_index(actual, name, boundary=boundary, path=path)
    if field_index is None:
        raise _invalid_input(f"{boundary} STRUCT at {path} is missing FILE-bearing field {name!r}")
    return field_index


def _mapping_field_value(
    value: Mapping[Any, Any],
    name: str,
    *,
    boundary: str,
    path: str,
) -> Any:
    folded_name = name.casefold()
    matches = [key for key in value if isinstance(key, str) and key.casefold() == folded_name]
    if len(matches) > 1:
        raise _invalid_input(f"{boundary} STRUCT at {path} has ambiguous field names matching {name!r}")
    return value[matches[0]] if matches else None


def _is_arrow_string_storage(dtype: pa.DataType) -> bool:
    is_string_view = getattr(pa.types, "is_string_view", None)
    return (
        pa.types.is_string(dtype)
        or pa.types.is_large_string(dtype)
        or (callable(is_string_view) and is_string_view(dtype))
    )


def _is_arrow_binary_storage(dtype: pa.DataType) -> bool:
    is_binary_view = getattr(pa.types, "is_binary_view", None)
    return (
        pa.types.is_binary(dtype)
        or pa.types.is_large_binary(dtype)
        or (callable(is_binary_view) and is_binary_view(dtype))
    )


def _is_arrow_list_storage(dtype: pa.DataType) -> bool:
    is_list_view = getattr(pa.types, "is_list_view", None)
    is_large_list_view = getattr(pa.types, "is_large_list_view", None)
    return (
        pa.types.is_list(dtype)
        or pa.types.is_large_list(dtype)
        or (callable(is_list_view) and is_list_view(dtype))
        or (callable(is_large_list_view) and is_large_list_view(dtype))
    )


def _is_arrow_extension_type(dtype: pa.DataType) -> bool:
    return isinstance(dtype, getattr(pa, "BaseExtensionType", pa.ExtensionType))


def _is_duckdb_bit_arrow_type(dtype: pa.DataType) -> bool:
    return (
        _is_arrow_extension_type(dtype)
        and getattr(dtype, "extension_name", None) == "arrow.opaque"
        and getattr(dtype, "type_name", None) == "bit"
        and getattr(dtype, "vendor_name", None) == "DuckDB"
        and _is_arrow_binary_storage(dtype.storage_type)
    )


def _decode_duckdb_bit_bytes(value: bytes) -> str:
    if len(value) < 2:
        raise ValueError("BIT storage must contain padding metadata and at least one data byte")
    padding = value[0]
    if padding >= 8:
        raise ValueError("BIT storage has invalid padding metadata")
    if padding:
        padding_mask = ((1 << padding) - 1) << (8 - padding)
        if value[1] & padding_mask != padding_mask:
            raise ValueError("BIT storage has invalid padding bits")
    bits = "".join(f"{byte:08b}" for byte in value[1:])
    return bits[padding:]


def _annotate_duckdb_bit_input(array: Any, dtype: Any, *, boundary: str) -> Any:
    """Attach BIT provenance to DuckDB-produced binary input without touching siblings."""
    if not _contains_bit(dtype):
        return array
    if isinstance(array, pa.ChunkedArray):
        if not array.chunks:
            return array
        return pa.chunked_array([_annotate_duckdb_bit_input(chunk, dtype, boundary=boundary) for chunk in array.chunks])

    type_id = _type_id(dtype)
    if type_id == "bit":
        expected = _expected_arrow_type(dtype, boundary=boundary)
        if array.type.equals(expected):
            return array
        if pa.types.is_null(array.type):
            return pa.nulls(len(array), type=expected)
        if _is_duckdb_bit_arrow_type(array.type):
            storage = array.storage
        elif _is_arrow_binary_storage(array.type):
            storage = array
        else:
            return array
        if not storage.type.equals(expected.storage_type):
            storage = storage.cast(expected.storage_type)
        return pa.ExtensionArray.from_storage(expected, storage)

    if type_id == "struct":
        arrays = [array.field(index) for index in range(array.type.num_fields)]
        fields = list(array.type)
        changed = False
        for name, child in dtype.children:
            if not _contains_bit(child):
                continue
            field_index = _struct_field_index(array.type, name, boundary=boundary, path="column")
            annotated = _annotate_duckdb_bit_input(arrays[field_index], child, boundary=boundary)
            if annotated.type.equals(arrays[field_index].type):
                continue
            arrays[field_index] = annotated
            field = fields[field_index]
            fields[field_index] = pa.field(
                field.name,
                annotated.type,
                nullable=field.nullable,
                metadata=field.metadata,
            )
            changed = True
        return pa.StructArray.from_arrays(arrays, fields=fields, mask=array.is_null()) if changed else array

    if type_id == "list":
        if pa.types.is_list(array.type):
            offset_type = pa.int32()
            constructor = pa.ListArray.from_arrays
        if pa.types.is_large_list(array.type):
            offset_type = pa.int64()
            constructor = pa.LargeListArray.from_arrays
        if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
            offsets = [int(offset) for offset in array.offsets.to_pylist()]
            start = offsets[0]
            values = array.values.slice(start, offsets[-1] - start)
            annotated = _annotate_duckdb_bit_input(values, _sequence_child(dtype), boundary=boundary)
            if annotated.type.equals(values.type):
                return array
            normalized_offsets = pa.array([offset - start for offset in offsets], type=offset_type)
            return constructor(normalized_offsets, annotated, mask=array.is_null())
        is_list_view = getattr(pa.types, "is_list_view", None)
        if callable(is_list_view) and is_list_view(array.type):
            values = _annotate_duckdb_bit_input(array.values, _sequence_child(dtype), boundary=boundary)
            if values.type.equals(array.values.type):
                return array
            return pa.ListViewArray.from_arrays(array.offsets, array.sizes, values, mask=array.is_null())
        is_large_list_view = getattr(pa.types, "is_large_list_view", None)
        if callable(is_large_list_view) and is_large_list_view(array.type):
            values = _annotate_duckdb_bit_input(array.values, _sequence_child(dtype), boundary=boundary)
            if values.type.equals(array.values.type):
                return array
            return pa.LargeListViewArray.from_arrays(array.offsets, array.sizes, values, mask=array.is_null())
        return array

    if type_id in ("array", "tensor"):
        storage = array.storage if type_id == "tensor" and _is_arrow_extension_type(array.type) else array
        array_size = storage.type.list_size
        values = storage.values.slice(storage.offset * array_size, len(storage) * array_size)
        annotated = _annotate_duckdb_bit_input(values, _sequence_child(dtype), boundary=boundary)
        if annotated.type.equals(values.type):
            return array
        normalized_storage = pa.FixedSizeListArray.from_arrays(
            annotated,
            array_size,
            mask=storage.is_null(),
        )
        if type_id == "array":
            return normalized_storage
        tensor_type = pa.fixed_shape_tensor(annotated.type, _tensor_shape(dtype))
        return pa.ExtensionArray.from_storage(tensor_type, normalized_storage)

    if type_id == "map":
        children = _type_children(dtype)
        start, length, offsets = _child_window(array)
        source_keys = array.keys.slice(start, length)
        source_items = array.items.slice(start, length)
        keys = (
            _annotate_duckdb_bit_input(source_keys, children["key"], boundary=boundary)
            if _contains_bit(children["key"])
            else source_keys
        )
        items = (
            _annotate_duckdb_bit_input(source_items, children["value"], boundary=boundary)
            if _contains_bit(children["value"])
            else source_items
        )
        if keys.type.equals(source_keys.type) and items.type.equals(source_items.type):
            return array
        normalized_offsets = pa.array([offset - start for offset in offsets], type=pa.int32())
        return pa.MapArray.from_arrays(normalized_offsets, keys, items, mask=array.is_null())

    return array


def _validate_nested_struct_field_sets(
    actual: pa.DataType,
    dtype: Any,
    *,
    boundary: str,
    path: str,
) -> None:
    """Reject lossy STRUCT rebuilding without constraining DuckDB source casts."""
    if pa.types.is_null(actual) or _is_file_type(dtype):
        return

    type_id = _type_id(dtype)
    if type_id == "struct":
        if not pa.types.is_struct(actual):
            return
        actual_names = [field.name.casefold() for field in actual]
        declared_names = [name.casefold() for name, _ in dtype.children]
        if len(set(actual_names)) != len(actual_names):
            raise _invalid_input(f"{boundary} STRUCT at {path} has ambiguous field names")
        if len(set(declared_names)) != len(declared_names) or set(actual_names) != set(declared_names):
            raise _invalid_input(f"{boundary} STRUCT at {path} must contain exactly the declared fields")
        for name, child in dtype.children:
            field_index = _struct_field_index(actual, name, boundary=boundary, path=path)
            _validate_nested_struct_field_sets(
                actual.field(field_index).type,
                child,
                boundary=boundary,
                path=f"{path}.{name}",
            )
        return

    if type_id in ("list", "array", "tensor"):
        actual_storage = actual.storage_type if _is_arrow_extension_type(actual) else actual
        if not (_is_arrow_list_storage(actual_storage) or pa.types.is_fixed_size_list(actual_storage)):
            return
        _validate_nested_struct_field_sets(
            actual_storage.value_type,
            _sequence_child(dtype),
            boundary=boundary,
            path=f"{path}[]",
        )
        return

    if type_id == "map" and pa.types.is_map(actual):
        children = _type_children(dtype)
        _validate_nested_struct_field_sets(
            actual.key_type,
            children["key"],
            boundary=boundary,
            path=f"{path}.key",
        )
        _validate_nested_struct_field_sets(
            actual.item_type,
            children["value"],
            boundary=boundary,
            path=f"{path}.value",
        )
        return

    if type_id == "union" and pa.types.is_union(actual):
        if actual.mode != "sparse":
            raise _invalid_input(f"{boundary} UNION at {path} must use sparse Arrow storage")
        if actual.type_codes != list(range(len(actual))):
            raise _invalid_input(f"{boundary} UNION at {path} type codes must match child ordinals")


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
                    field_matches = _is_arrow_string_storage(field.type)
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
        if not _is_arrow_list_storage(actual):
            raise _invalid_input(f"{boundary} value at {path} must use an Arrow list type")
        _validate_arrow_storage_type(
            actual.value_type,
            _sequence_child(dtype),
            boundary=boundary,
            path=f"{path}[]",
            allow_untyped_null=allow_untyped_null,
        )
        return
    if type_id in ("array", "tensor"):
        actual_storage = actual
        if type_id == "tensor" and _is_arrow_extension_type(actual):
            if (
                actual.extension_name != "arrow.fixed_shape_tensor"
                or tuple(actual.shape) != _tensor_shape(dtype)
                or getattr(actual, "permutation", None) is not None
                or getattr(actual, "dim_names", None) is not None
            ):
                raise _invalid_input(f"{boundary} value at {path} must use its declared Arrow tensor metadata")
            actual_storage = actual.storage_type
        if not pa.types.is_fixed_size_list(actual_storage):
            raise _invalid_input(f"{boundary} value at {path} must use an Arrow fixed-size list type")
        expected_size = _fixed_sequence_size(dtype)
        if actual_storage.list_size != expected_size:
            raise _invalid_input(f"{boundary} value at {path} must have fixed size {expected_size}")
        _validate_arrow_storage_type(
            actual_storage.value_type,
            _sequence_child(dtype),
            boundary=boundary,
            path=f"{path}[]",
            allow_untyped_null=allow_untyped_null,
        )
        return
    if type_id == "struct":
        if not pa.types.is_struct(actual):
            raise _invalid_input(f"{boundary} value at {path} must use an Arrow STRUCT type")
        actual_names = [field.name.casefold() for field in actual]
        declared_names = [name.casefold() for name, _ in dtype.children]
        if len(set(actual_names)) != len(actual_names):
            raise _invalid_input(f"{boundary} STRUCT at {path} has ambiguous field names")
        if len(set(declared_names)) != len(declared_names) or set(actual_names) != set(declared_names):
            raise _invalid_input(f"{boundary} STRUCT at {path} must contain exactly the declared fields")
        for name, child in dtype.children:
            if not _contains_file(child):
                continue
            field_index = _struct_field_index(actual, name, boundary=boundary, path=path)
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


def _active_values(array: pa.Array, parent_active: Sequence[bool] | None) -> list[bool]:
    active = [bool(value) for value in array.is_valid().to_pylist()]
    if parent_active is None:
        return active
    if len(parent_active) != len(array):
        raise RuntimeError("FILE validation received a mismatched parent validity mask")
    return [parent and current for parent, current in zip(parent_active, active, strict=True)]


def _mask_inactive(array: Any, active: Sequence[bool]) -> Any:
    if all(active):
        return array
    indices = pa.array([index if is_active else None for index, is_active in enumerate(active)], type=pa.int64())
    return array.take(indices)


def _child_window(array: Any) -> tuple[int, int, list[int]]:
    offsets = [int(offset) for offset in array.offsets.to_pylist()]
    start = offsets[0]
    return start, offsets[-1] - start, offsets


def _arrow_cast_preserves_values(actual: pa.DataType, expected: pa.DataType, dtype: Any) -> bool:
    """Return whether Arrow can only change storage, not DuckDB cast semantics."""
    if actual.equals(expected):
        return True
    if _is_arrow_extension_type(actual) and actual.storage_type.equals(expected):
        # DuckDB BIT storage includes a padding byte that its BIT-to-BLOB cast
        # removes. Stripping the extension in Arrow would expose internal
        # bytes instead of preserving that declared cast.
        return not _is_duckdb_bit_arrow_type(actual)
    if _is_arrow_string_storage(actual) and _is_arrow_string_storage(expected):
        return True
    if _is_arrow_binary_storage(actual) and _is_arrow_binary_storage(expected):
        return True
    if pa.types.is_signed_integer(actual) and pa.types.is_signed_integer(expected):
        return expected.bit_width >= actual.bit_width
    if pa.types.is_unsigned_integer(actual) and pa.types.is_unsigned_integer(expected):
        return expected.bit_width >= actual.bit_width
    if pa.types.is_unsigned_integer(actual) and pa.types.is_signed_integer(expected):
        return expected.bit_width > actual.bit_width
    if pa.types.is_float32(actual) and pa.types.is_float64(expected):
        return True
    if pa.types.is_timestamp(actual) and pa.types.is_timestamp(expected):
        return (actual.tz is None) == (expected.tz is None)
    if pa.types.is_time(actual) and pa.types.is_time(expected):
        return True
    if pa.types.is_date(actual) and pa.types.is_date(expected):
        return True
    if pa.types.is_decimal(actual) and pa.types.is_decimal(expected):
        return actual.scale == expected.scale and actual.precision <= expected.precision
    return _type_id(dtype) in ("bignum", "hugeint", "uhugeint") and pa.types.is_integer(actual)


def _validate_file_arrow_values(
    array: Any,
    dtype: Any,
    *,
    boundary: str,
    path: str,
    parent_active: Sequence[bool] | None = None,
) -> None:
    """Validate FILE leaves without converting unrelated Arrow children."""
    if isinstance(array, pa.ChunkedArray):
        offset = 0
        for chunk in array.chunks:
            chunk_active = None if parent_active is None else parent_active[offset : offset + len(chunk)]
            _validate_file_arrow_values(
                chunk,
                dtype,
                boundary=boundary,
                path=path,
                parent_active=chunk_active,
            )
            offset += len(chunk)
        return
    if pa.types.is_null(array.type):
        return

    if _is_file_type(dtype):
        for index, value in enumerate(array.to_pylist()):
            if value is None or (parent_active is not None and not parent_active[index]):
                continue
            _file_from_arrow_value(value, boundary=boundary, path=f"{path}[{index}]")
        return

    active = _active_values(array, parent_active)
    type_id = _type_id(dtype)
    if type_id == "struct":
        for name, child in dtype.children:
            if _contains_file(child):
                field_index = _struct_field_index(array.type, name, boundary=boundary, path=path)
                _validate_file_arrow_values(
                    array.field(field_index),
                    child,
                    boundary=boundary,
                    path=f"{path}.{name}",
                    parent_active=active,
                )
        return
    if type_id == "list":
        source = _mask_inactive(array, active)
        _validate_file_arrow_values(
            pc.list_flatten(source),
            _sequence_child(dtype),
            boundary=boundary,
            path=f"{path}[]",
        )
        return
    if type_id in ("array", "tensor"):
        storage = array.storage if type_id == "tensor" and isinstance(array, pa.ExtensionArray) else array
        array_size = _fixed_sequence_size(dtype)
        child_active = [is_active for is_active in active for _ in range(array_size)]
        _validate_file_arrow_values(
            storage.values.slice(storage.offset * array_size, len(storage) * array_size),
            _sequence_child(dtype),
            boundary=boundary,
            path=f"{path}[]",
            parent_active=child_active,
        )
        return
    if type_id == "map":
        start, length, offsets = _child_window(array)
        child_active = [
            is_active for index, is_active in enumerate(active) for _ in range(offsets[index + 1] - offsets[index])
        ]
        children = _type_children(dtype)
        if _contains_file(children["key"]):
            _validate_file_arrow_values(
                array.keys.slice(start, length),
                children["key"],
                boundary=boundary,
                path=f"{path}.key",
                parent_active=child_active,
            )
        if _contains_file(children["value"]):
            _validate_file_arrow_values(
                array.items.slice(start, length),
                children["value"],
                boundary=boundary,
                path=f"{path}.value",
                parent_active=child_active,
            )
        return
    if type_id == "union":
        raise _invalid_input(f"{boundary} does not yet support UNION values containing FILE")


def _materialize_native_value(value: Any, dtype: Any, *, boundary: str, path: str) -> Any:
    if value is None:
        return None
    if _is_file_type(dtype):
        return _file_from_arrow_value(value, boundary=boundary, path=path)

    type_id = _type_id(dtype)
    if type_id == "bit" and isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return _decode_duckdb_bit_bytes(bytes(value))
        except ValueError as exc:
            raise _invalid_input(f"{boundary} contains invalid BIT storage at {path}: {exc}") from exc
    if type_id in ("list", "array", "tensor"):
        child = _sequence_child(dtype)
        values = [
            _materialize_native_value(item, child, boundary=boundary, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
        return tuple(values) if type_id in ("array", "tensor") else values
    if type_id == "struct":
        result = {}
        for name, child in dtype.children:
            child_value = _mapping_field_value(value, name, boundary=boundary, path=path)
            result[name] = (
                _materialize_native_value(child_value, child, boundary=boundary, path=f"{path}.{name}")
                if _contains_file(child) or _contains_bit(child)
                else child_value
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
    if type_id in ("list", "array", "tensor"):
        if not isinstance(value, (list, tuple)):
            raise _invalid_input(f"{boundary} value at {path} must be a sequence")
        if type_id in ("array", "tensor"):
            expected_size = _fixed_sequence_size(dtype)
            if len(value) != expected_size:
                raise _invalid_input(f"{boundary} value at {path} must have fixed size {expected_size}")
        child = _sequence_child(dtype)
        return [
            _canonicalize_native_output(item, child, boundary=boundary, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if type_id == "struct":
        struct_children = list(dtype.children)
        if isinstance(value, Mapping):
            actual_keys: dict[str, Any] = {}
            for key in value:
                if not isinstance(key, str):
                    raise _invalid_input(f"{boundary} STRUCT value at {path} must contain exactly the declared fields")
                folded_key = key.casefold()
                if folded_key in actual_keys:
                    raise _invalid_input(f"{boundary} STRUCT at {path} has ambiguous field names matching {key!r}")
                actual_keys[folded_key] = key
            declared_keys = {name.casefold() for name, _ in struct_children}
            if set(actual_keys) != declared_keys:
                raise _invalid_input(f"{boundary} STRUCT value at {path} must contain exactly the declared fields")
            child_values = [value[actual_keys[name.casefold()]] for name, _ in struct_children]
        elif isinstance(value, tuple):
            if len(value) != len(struct_children):
                raise _invalid_input(
                    f"{boundary} STRUCT value at {path} must contain exactly {len(struct_children)} positional fields"
                )
            child_values = list(value)
        else:
            raise _invalid_input(f"{boundary} value at {path} must be a mapping or positional tuple")
        result = {}
        for (name, child), child_value in zip(struct_children, child_values, strict=True):
            result[name] = (
                _canonicalize_native_output(child_value, child, boundary=boundary, path=f"{path}.{name}")
                if _requires_native_output_encoding(child)
                else child_value
            )
        return result
    if type_id == "map":
        children = _type_children(dtype)
        entries: Any
        if (
            isinstance(value, Mapping)
            and set(value) == {"key", "value"}
            and isinstance(value["key"], (list, tuple))
            and isinstance(value["value"], (list, tuple))
            and not _native_map_key_is_hashable(children["key"])
        ):
            keys = value["key"]
            items = value["value"]
            if len(keys) != len(items):
                raise _invalid_input(f"{boundary} MAP value at {path} must contain equally sized key/value lists")
            entries = zip(keys, items)
        elif isinstance(value, Mapping):
            entries = value.items()
        elif isinstance(value, (list, tuple)):
            entries = value
        else:
            raise _invalid_input(f"{boundary} MAP value at {path} must be a mapping or sequence of pairs")
        canonical_pairs: list[tuple[Any, Any]] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes, bytearray)) or len(entry) != 2:
                raise _invalid_input(f"{boundary} MAP value at {path} must contain key/value pairs")
            key, item = entry
            canonical_pairs.append(
                (
                    _canonicalize_native_output(
                        key,
                        children["key"],
                        boundary=boundary,
                        path=f"{path}[{index}].key",
                    ),
                    _canonicalize_native_output(
                        item,
                        children["value"],
                        boundary=boundary,
                        path=f"{path}[{index}].value",
                    ),
                )
            )
        return canonical_pairs
    if type_id == "union":
        raise _invalid_input(f"{boundary} does not yet support UNION values containing FILE")
    return value


def _canonical_values_to_arrow_array(
    values: Sequence[Any],
    dtype: Any,
    *,
    boundary: str,
) -> pa.Array:
    """Encode FILE and Python-only special leaves without typing ordinary leaves."""
    if _is_file_type(dtype):
        return pa.array(values, type=_expected_arrow_type(dtype, boundary=boundary))
    type_id = _type_id(dtype)
    if type_id in ("hugeint", "uhugeint"):
        encoded: list[str | None] = []
        for value in values:
            if value is None:
                encoded.append(None)
                continue
            if isinstance(value, str):
                encoded.append(value)
                continue
            integer = operator.index(value)
            if type_id == "hugeint":
                in_range = -(1 << 127) <= integer < 1 << 127
            else:
                in_range = 0 <= integer < 1 << 128
            if not in_range:
                raise ValueError(f"{type_id.upper()} output is outside its 128-bit range")
            encoded.append(str(integer))
        return pa.array(encoded, type=pa.string())
    if type_id == "bignum":
        bignum_encoded: list[str | None] = []
        for value in values:
            if value is None:
                bignum_encoded.append(None)
                continue
            try:
                bignum_encoded.append(str(operator.index(value)))
            except TypeError:
                if not isinstance(value, str):
                    raise
                bignum_encoded.append(value)
        return pa.array(bignum_encoded, type=pa.string())
    if type_id == "uuid":
        return pa.array(
            [
                None
                if value is None
                else str(value)
                if isinstance(value, (UUID, str))
                else str(UUID(bytes=bytes(value)))
                for value in values
            ],
            type=pa.string(),
        )
    if type_id == "time with time zone":
        textual_non_null = [isinstance(value, (datetime_time, str)) for value in values if value is not None]
        if all(textual_non_null):
            encoded = [value.isoformat() if isinstance(value, datetime_time) else value for value in values]
            return pa.array(encoded, type=pa.string())
        if any(textual_non_null):
            # Keep values with different DuckDB cast provenance in separate
            # Arrow pieces rather than allowing inference to coerce one into
            # the storage family of another.
            raise TypeError("TIME WITH TIME ZONE output mixes textual and non-textual storage")
        return pa.array(values)
    if not _requires_native_output_encoding(dtype):
        try:
            return pa.array(values)
        except Exception as inference_error:
            try:
                expected_type = _arrow_type_from_duckdb_pytype(dtype)
            except Exception:
                raise inference_error
            return pa.array(values, type=expected_type)

    type_id = _type_id(dtype)
    null_mask = pa.array([value is None for value in values], type=pa.bool_())
    if type_id == "struct":
        arrays = []
        names = []
        for name, child in dtype.children:
            names.append(name)
            child_values = [None if value is None else value.get(name) for value in values]
            arrays.append(_canonical_values_to_arrow_array(child_values, child, boundary=boundary))
        return pa.StructArray.from_arrays(arrays, names=names, mask=null_mask)

    if type_id == "list":
        offsets = [0]
        flattened = []
        for value in values:
            if value is not None:
                flattened.extend(value)
            offsets.append(len(flattened))
        child = _sequence_child(dtype)
        child_array = _canonical_values_to_arrow_array(flattened, child, boundary=boundary)
        return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), child_array, mask=null_mask)

    if type_id in ("array", "tensor"):
        child = _sequence_child(dtype)
        array_size = _fixed_sequence_size(dtype)
        flattened = []
        for value in values:
            flattened.extend([None] * array_size if value is None else value)
        child_array = _canonical_values_to_arrow_array(flattened, child, boundary=boundary)
        storage = pa.FixedSizeListArray.from_arrays(child_array, array_size, mask=null_mask)
        if type_id == "array":
            return storage
        tensor_type = pa.fixed_shape_tensor(child_array.type, _tensor_shape(dtype))
        return pa.ExtensionArray.from_storage(tensor_type, storage)

    if type_id == "map":
        offsets = [0]
        keys = []
        items = []
        for value in values:
            if value is not None:
                for key, item in value:
                    keys.append(key)
                    items.append(item)
            offsets.append(len(keys))
        children = _type_children(dtype)
        key_array = _canonical_values_to_arrow_array(keys, children["key"], boundary=boundary)
        item_array = _canonical_values_to_arrow_array(items, children["value"], boundary=boundary)
        return pa.MapArray.from_arrays(
            pa.array(offsets, type=pa.int32()),
            key_array,
            item_array,
            mask=null_mask,
        )

    raise _invalid_input(f"{boundary} uses an unsupported type containing FILE: {dtype}")


def _native_outputs_to_arrow_array(values: Sequence[Any], dtype: Any, *, boundary: str) -> pa.Array:
    canonical = [
        _canonicalize_native_output(value, dtype, boundary=boundary, path=f"row {row_index}")
        for row_index, value in enumerate(values)
    ]
    try:
        return _canonical_values_to_arrow_array(canonical, dtype, boundary=boundary)
    except Exception:
        raise _invalid_input(f"{boundary} could not be encoded using its declared FILE type") from None


def _native_outputs_to_arrow_arrays(values: Sequence[Any], dtype: Any, *, boundary: str) -> list[Any]:
    """Encode native rows, splitting only when Arrow cannot hold one lossless schema."""
    try:
        return [_native_outputs_to_arrow_array(values, dtype, boundary=boundary)]
    except Exception:
        row_arrays = [_native_outputs_to_arrow_array([value], dtype, boundary=boundary) for value in values]

    groups: list[list[pa.Array]] = []
    for array in row_arrays:
        if groups and array.type.equals(groups[-1][0].type):
            groups[-1].append(array)
        else:
            groups.append([array])
    return [group[0] if len(group) == 1 else pa.concat_arrays(group) for group in groups]


def _normalize_file_arrow_array(
    array: Any,
    dtype: Any,
    *,
    boundary: str,
    parent_active: Sequence[bool] | None = None,
    normalize_value_dependent: bool = True,
) -> Any:
    """Canonicalize a FILE-bearing Arrow value to its declared stable storage."""
    if isinstance(array, pa.ChunkedArray):
        if not array.chunks:
            normalized_empty = _normalize_file_arrow_array(
                pa.array([], type=array.type),
                dtype,
                boundary=boundary,
                parent_active=parent_active,
                normalize_value_dependent=normalize_value_dependent,
            )
            return pa.chunked_array([], type=normalized_empty.type)
        chunks = []
        offset = 0
        for chunk in array.chunks:
            chunk_active = None if parent_active is None else parent_active[offset : offset + len(chunk)]
            chunks.append(
                _normalize_file_arrow_array(
                    chunk,
                    dtype,
                    boundary=boundary,
                    parent_active=chunk_active,
                    normalize_value_dependent=normalize_value_dependent,
                )
            )
            offset += len(chunk)
        normalized_type = chunks[0].type
        if all(chunk.type.equals(normalized_type) for chunk in chunks[1:]):
            return pa.chunked_array(chunks, type=normalized_type)

        # Some safe casts, notably temporal downcasts, depend on the values.
        # Retry every original chunk without those casts so the whole logical
        # column keeps one schema without concatenating or materializing it.
        stable_chunks = []
        offset = 0
        for chunk in array.chunks:
            chunk_active = None if parent_active is None else parent_active[offset : offset + len(chunk)]
            stable_chunks.append(
                _normalize_file_arrow_array(
                    chunk,
                    dtype,
                    boundary=boundary,
                    parent_active=chunk_active,
                    normalize_value_dependent=False,
                )
            )
            offset += len(chunk)
        stable_type = stable_chunks[0].type
        if any(not chunk.type.equals(stable_type) for chunk in stable_chunks[1:]):
            raise RuntimeError("FILE normalization could not stabilize a chunked Arrow column")
        return pa.chunked_array(stable_chunks, type=stable_type)
    active = _active_values(array, parent_active)
    type_id = _type_id(dtype)
    if type_id == "bit":
        source = _mask_inactive(array, active)
        expected = _expected_arrow_type(dtype, boundary=boundary)
        if source.type.equals(expected):
            return source
        if pa.types.is_null(source.type):
            return pa.nulls(len(source), type=expected)
        if _is_duckdb_bit_arrow_type(source.type):
            storage = source.storage
            if not storage.type.equals(expected.storage_type):
                storage = storage.cast(expected.storage_type)
            return pa.ExtensionArray.from_storage(expected, storage)
        # Ordinary binary is a BLOB value returned by user code. Leave it
        # unannotated so DuckDB performs its normal BLOB-to-BIT cast.
        return source
    if type_id == "uuid":
        source = _mask_inactive(array, active)
        if not normalize_value_dependent:
            return source
        values = source.to_pylist()
        if _is_arrow_binary_storage(source.type) and any(
            value is not None and len(bytes(value)) != 16 for value in values
        ):
            return source
        encoded: list[str | None] = []
        for value in values:
            if value is None:
                encoded.append(None)
            elif isinstance(value, UUID):
                encoded.append(str(value))
            elif isinstance(value, (bytes, bytearray, memoryview)):
                encoded.append(str(UUID(bytes=bytes(value))))
            else:
                encoded.append(str(value))
        return pa.array(encoded, type=pa.string())
    if pa.types.is_null(array.type):
        try:
            expected = _expected_arrow_type(dtype, boundary=boundary)
        except Exception:
            # Preserve the promotable Arrow NULL type when a non-FILE sibling
            # has no canonical Arrow mapping.
            return array
        return pa.nulls(len(array), type=expected)
    if _is_file_type(dtype):
        expected = _expected_arrow_type(dtype, boundary=boundary)
        source = _mask_inactive(array, active)
        return source if source.type.equals(expected) else source.cast(expected)

    if type_id == "struct":
        source = _mask_inactive(array, active)
        if not pa.types.is_struct(source.type):
            return source

        arrays = []
        fields = []
        for name, child in dtype.children:
            field_index = _struct_field_index(source.type, name, boundary=boundary, path="column")
            child_array = source.field(field_index)
            child_array = _normalize_file_arrow_array(
                child_array,
                child,
                boundary=boundary,
                parent_active=active,
                normalize_value_dependent=normalize_value_dependent,
            )
            arrays.append(child_array)
            fields.append(pa.field(name, child_array.type))
        return pa.StructArray.from_arrays(arrays, fields=fields, mask=source.is_null())

    if type_id == "list":
        source = _mask_inactive(array, active)
        if not _is_arrow_list_storage(source.type):
            return source

        lengths = [0 if length is None else int(length) for length in pc.list_value_length(source).to_pylist()]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        child_array = _normalize_file_arrow_array(
            pc.list_flatten(source),
            _sequence_child(dtype),
            boundary=boundary,
            normalize_value_dependent=normalize_value_dependent,
        )
        return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), child_array, mask=source.is_null())

    if type_id in ("array", "tensor"):
        source = _mask_inactive(array, active)
        storage = source
        if type_id == "tensor" and isinstance(source, pa.ExtensionArray):
            if (
                source.type.extension_name != "arrow.fixed_shape_tensor"
                or tuple(source.type.shape) != _tensor_shape(dtype)
                or getattr(source.type, "permutation", None) is not None
                or getattr(source.type, "dim_names", None) is not None
            ):
                return source
            storage = source.storage
        array_size = _fixed_sequence_size(dtype)
        child_source: Any
        if _is_arrow_list_storage(storage.type):
            lengths = pc.list_value_length(storage).to_pylist()
            if any(length is not None and int(length) != array_size for length in lengths):
                raise _invalid_input(f"{boundary} ARRAY value must have fixed size {array_size}")

            flattened = pc.list_flatten(storage)
            flattened_index = 0
            child_indices: list[int | None] = []
            for length in lengths:
                if length is None:
                    child_indices.extend([None] * array_size)
                    continue
                child_indices.extend(range(flattened_index, flattened_index + array_size))
                flattened_index += int(length)
            child_source = flattened.take(pa.array(child_indices, type=pa.int64()))
        elif pa.types.is_fixed_size_list(storage.type):
            if storage.type.list_size != array_size:
                raise _invalid_input(f"{boundary} ARRAY value must have fixed size {array_size}")
            child_source = storage.values.slice(storage.offset * array_size, len(storage) * array_size)
        else:
            return source

        child_array = _normalize_file_arrow_array(
            child_source,
            _sequence_child(dtype),
            boundary=boundary,
            parent_active=[is_active for is_active in active for _ in range(array_size)],
            normalize_value_dependent=normalize_value_dependent,
        )
        normalized_storage = pa.FixedSizeListArray.from_arrays(child_array, array_size, mask=source.is_null())
        if type_id == "array":
            return normalized_storage
        tensor_type = pa.fixed_shape_tensor(child_array.type, _tensor_shape(dtype))
        return pa.ExtensionArray.from_storage(tensor_type, normalized_storage)

    if type_id == "map":
        source = _mask_inactive(array, active)
        if not pa.types.is_map(source.type):
            return source

        start, _, offsets = _child_window(source)
        children = _type_children(dtype)
        selected_indices: list[int] = []
        normalized_offsets = [0]
        for index, is_active in enumerate(active):
            if is_active:
                selected_indices.extend(range(offsets[index] - start, offsets[index + 1] - start))
            normalized_offsets.append(len(selected_indices))
        selection = pa.array(selected_indices, type=pa.int64())
        keys = source.keys.slice(start, offsets[-1] - start).take(selection)
        items = source.items.slice(start, offsets[-1] - start).take(selection)
        keys = _normalize_file_arrow_array(
            keys,
            children["key"],
            boundary=boundary,
            normalize_value_dependent=normalize_value_dependent,
        )
        items = _normalize_file_arrow_array(
            items,
            children["value"],
            boundary=boundary,
            normalize_value_dependent=normalize_value_dependent,
        )
        return pa.MapArray.from_arrays(
            pa.array(normalized_offsets, type=pa.int32()),
            keys,
            items,
            mask=source.is_null(),
        )

    source = _mask_inactive(array, active)
    if not normalize_value_dependent:
        return source
    try:
        expected = _arrow_type_from_duckdb_pytype(dtype)
    except Exception:
        return source
    if source.null_count == len(source):
        return pa.nulls(len(source), type=expected)
    if not _arrow_cast_preserves_values(source.type, expected, dtype):
        # Cross-type casts belong to DuckDB. Arrow may accept a cast while
        # assigning it different semantics (for example BLOB to VARCHAR).
        return source
    try:
        return source.cast(expected)
    except Exception:
        return source


def validate_file_arrow_array(
    array: Any,
    dtype: Any,
    *,
    boundary: str,
    allow_untyped_null: bool = False,
) -> None:
    """Validate one Arrow array governed by a logical type containing FILE."""
    _validate_nested_struct_field_sets(
        array.type,
        dtype,
        boundary=boundary,
        path="column",
    )
    _validate_arrow_storage_type(
        array.type,
        dtype,
        boundary=boundary,
        path="column",
        allow_untyped_null=allow_untyped_null,
    )
    _validate_file_arrow_values(array, dtype, boundary=boundary, path="column")


def normalize_file_arrow_array(
    array: Any,
    dtype: Any,
    *,
    boundary: str,
    allow_untyped_null: bool = False,
) -> Any:
    """Validate and canonicalize an Arrow array governed by a FILE type."""
    validate_file_arrow_array(
        array,
        dtype,
        boundary=boundary,
        allow_untyped_null=allow_untyped_null,
    )
    return _normalize_file_arrow_array(array, dtype, boundary=boundary)


@dataclass(frozen=True)
class FileUDFContract:
    """Parsed FILE-bearing input and output types for one runtime payload."""

    udf_name: str
    input_types: tuple[Any | None, ...]
    output_types: tuple[Any | None, ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> FileUDFContract:
        raw_inputs = payload.get("input_types") or []
        raw_input_contracts = payload.get("input_contract_types")
        parsed_input_contracts: tuple[Any | None, ...] | None = None
        if isinstance(raw_inputs, (list, tuple)) and isinstance(raw_input_contracts, (list, tuple)):
            if len(raw_input_contracts) != len(raw_inputs):
                raise _invalid_input("UDF payload input contract count does not match its input type count")
            parsed_input_contracts = tuple(
                _parse_declared_type(type_name, field="input_contract_types") if isinstance(type_name, str) else None
                for type_name in raw_input_contracts
            )
            file_input_types = tuple(
                dtype if dtype is not None and _contains_file(dtype) else None for dtype in parsed_input_contracts
            )
            input_types = file_input_types
        elif isinstance(raw_inputs, (list, tuple)):
            file_input_types = tuple(_parse_file_type(type_name, field="input_types") for type_name in raw_inputs)
            input_types = file_input_types
        else:
            file_input_types = ()
            input_types = ()

        method_return_type = payload.get("method_return_type")
        output_schema = payload.get("output_schema") or []
        raw_output_contracts = payload.get("output_contract_types")
        output_types: list[Any | None] = []
        if raw_output_contracts is not None:
            if not isinstance(raw_output_contracts, (list, tuple)):
                raise _invalid_input("UDF payload output_contract_types must be a list")
            if method_return_type is None and not isinstance(output_schema, (list, tuple)):
                raise _invalid_input("UDF payload output_schema must be a list")
            expected_output_count = 1 if method_return_type is not None else len(output_schema)
            if expected_output_count == 0 or len(raw_output_contracts) != expected_output_count:
                raise _invalid_input("UDF payload output contract count does not match its declared outputs")
            file_output_types: list[Any | None] = []
            for type_name in raw_output_contracts:
                if not isinstance(type_name, str):
                    raise _invalid_input("UDF payload output_contract_types must contain SQL type strings")
                file_output_types.append(_parse_file_type(type_name, field="output_contract_types"))
            if any(dtype is not None for dtype in file_output_types):
                output_types.extend(
                    file_dtype
                    if file_dtype is not None
                    else _parse_declared_type(type_name, field="output_contract_types")
                    for type_name, file_dtype in zip(raw_output_contracts, file_output_types, strict=True)
                )
            else:
                output_types.extend(file_output_types)
        elif method_return_type is not None:
            output_types.append(_parse_file_type(method_return_type, field="method_return_type"))
        else:
            if isinstance(output_schema, (list, tuple)):
                entries: list[Mapping[str, Any] | None] = []
                file_types: list[Any | None] = []
                for entry in output_schema:
                    if not isinstance(entry, Mapping):
                        entries.append(None)
                        file_types.append(None)
                        continue
                    kind = str(entry.get("kind") or "duckdb_type").strip().lower()
                    entries.append(entry)
                    file_types.append(
                        _parse_file_type(entry.get("type"), field="output_schema")
                        if kind == "duckdb_type"
                        else _parse_tensor_type(entry, field="output_schema", file_only=True)
                        if kind == "tensor"
                        else None
                    )
                if any(dtype is not None for dtype in file_types):
                    output_types.extend(
                        file_dtype
                        if file_dtype is not None
                        else (
                            _parse_declared_type(entry.get("type"), field="output_schema")
                            if entry is not None
                            and str(entry.get("kind") or "duckdb_type").strip().lower() == "duckdb_type"
                            else _parse_tensor_type(entry, field="output_schema")
                            if entry is not None and str(entry.get("kind") or "duckdb_type").strip().lower() == "tensor"
                            else None
                        )
                        for entry, file_dtype in zip(entries, file_types, strict=True)
                    )
                else:
                    output_types.extend(file_types)

        has_file_contract = (
            any(dtype is not None for dtype in file_input_types)
            or any(dtype is not None and _contains_file(dtype) for dtype in parsed_input_contracts or ())
            or any(dtype is not None and _contains_file(dtype) for dtype in output_types)
        )
        if has_file_contract and parsed_input_contracts is not None:
            input_types = parsed_input_contracts

        return cls(
            udf_name=str(payload.get("udf_name") or "<unknown>"),
            input_types=input_types,
            output_types=tuple(output_types),
        )

    @property
    def has_file_inputs(self) -> bool:
        return any(dtype is not None and _contains_file(dtype) for dtype in self.input_types)

    @property
    def has_file_outputs(self) -> bool:
        return any(dtype is not None and _contains_file(dtype) for dtype in self.output_types)

    @property
    def has_bit_inputs(self) -> bool:
        return any(dtype is not None and _contains_bit(dtype) for dtype in self.input_types)

    @property
    def requires_input_materialization(self) -> bool:
        return self.has_file_inputs or self.has_bit_inputs

    def _validate_column_count(self, table: pa.Table, types: tuple[Any | None, ...], *, boundary: str) -> None:
        if types and table.num_columns != len(types):
            raise _invalid_input(
                f"{boundary} has {table.num_columns} columns but its logical contract declares {len(types)}"
            )

    def validate_input_table(self, table: pa.Table) -> None:
        if not self.requires_input_materialization:
            return
        boundary = f"UDF {self.udf_name!r} input"
        self._validate_column_count(table, self.input_types, boundary=boundary)
        for index, dtype in enumerate(self.input_types):
            if dtype is not None and _contains_file(dtype):
                validate_file_arrow_array(table.column(index), dtype, boundary=f"{boundary} column {index}")

    def prepare_input_table(self, table: pa.Table) -> pa.Table:
        """Validate FILE inputs and mark DuckDB-produced BIT storage before user code runs."""
        self.validate_input_table(table)
        if not self.has_bit_inputs:
            return table

        boundary = f"UDF {self.udf_name!r} input"
        columns = list(table.columns)
        fields = list(table.schema)
        changed = False
        for index, dtype in enumerate(self.input_types):
            if dtype is None or not _contains_bit(dtype):
                continue
            annotated = _annotate_duckdb_bit_input(
                columns[index],
                dtype,
                boundary=f"{boundary} column {index}",
            )
            if annotated.type.equals(columns[index].type):
                continue
            columns[index] = annotated
            field = fields[index]
            fields[index] = pa.field(
                field.name,
                annotated.type,
                nullable=field.nullable,
                metadata=field.metadata,
            )
            changed = True

        if not changed:
            return table
        return pa.Table.from_arrays(columns, schema=pa.schema(fields, metadata=table.schema.metadata))

    def materialize_scalar_inputs(self, table: pa.Table) -> list[list[Any]]:
        if not self.requires_input_materialization:
            return [column.to_pylist() for column in table.columns]
        table = self.prepare_input_table(table)
        boundary = f"UDF {self.udf_name!r} input"
        self._validate_column_count(table, self.input_types, boundary=boundary)
        columns: list[list[Any]] = []
        for index, column in enumerate(table.columns):
            values = column.to_pylist()
            dtype = self.input_types[index]
            if dtype is not None and (_contains_file(dtype) or _contains_bit(dtype)):
                if _contains_file(dtype):
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
        return _native_outputs_to_arrow_array(outputs, dtype, boundary=boundary)

    def scalar_outputs_to_arrays(self, outputs: list[Any]) -> list[Any]:
        """Encode scalar rows into the minimum number of Arrow-compatible pieces."""
        if not self.has_file_outputs:
            return [pa.array(outputs)]
        if len(self.output_types) != 1 or self.output_types[0] is None:
            raise _invalid_input(f"UDF {self.udf_name!r} scalar FILE output contract must declare one column")
        return _native_outputs_to_arrow_arrays(
            outputs,
            self.output_types[0],
            boundary=f"UDF {self.udf_name!r} output",
        )

    def normalize_scalar_arrow_output(self, output: Any) -> Any:
        if not self.has_file_outputs:
            return output
        if len(self.output_types) != 1 or self.output_types[0] is None:
            raise _invalid_input(f"UDF {self.udf_name!r} scalar FILE output contract must declare one column")
        return normalize_file_arrow_array(
            output,
            self.output_types[0],
            boundary=f"UDF {self.udf_name!r} output",
            allow_untyped_null=True,
        )

    def native_output_rows_to_table(
        self,
        rows: Sequence[Mapping[str, Any]],
        output_names: Sequence[str],
    ) -> pa.Table:
        """Encode row-native outputs while preserving declared FILE leaves."""
        if not self.has_file_outputs:
            return pa.table({name: [row.get(name) for row in rows] for name in output_names})

        boundary = f"UDF {self.udf_name!r} output"
        if len(output_names) != len(self.output_types):
            raise _invalid_input(
                f"{boundary} declares {len(self.output_types)} columns but has {len(output_names)} output names"
            )

        declared_names = {name.casefold(): name for name in output_names}
        if len(declared_names) != len(output_names):
            raise _invalid_input(f"{boundary} has ambiguous output names")
        column_values: dict[str, list[Any]] = {name: [] for name in output_names}
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise _invalid_input(f"{boundary} row {row_index} must be a mapping")
            actual_names: dict[str, Any] = {}
            for name in row:
                if not isinstance(name, str) or name.casefold() in actual_names:
                    raise _invalid_input(f"{boundary} row {row_index} has ambiguous output names")
                actual_names[name.casefold()] = name
            if set(actual_names) != set(declared_names):
                raise _invalid_input(f"{boundary} row {row_index} must contain exactly the declared output fields")
            for folded_name, declared_name in declared_names.items():
                column_values[declared_name].append(row[actual_names[folded_name]])

        arrays: dict[str, Any] = {}
        for index, name in enumerate(output_names):
            dtype = self.output_types[index]
            if dtype is None:
                arrays[name] = column_values[name]
                continue
            arrays[name] = _native_outputs_to_arrow_array(column_values[name], dtype, boundary=boundary)

        table = pa.table(arrays)
        self.validate_output_table(table)
        return table

    def native_output_rows_to_tables(
        self,
        rows: Sequence[Mapping[str, Any]],
        output_names: Sequence[str],
    ) -> list[pa.Table]:
        """Encode row outputs, retaining separate schemas for DuckDB casts."""
        try:
            return [self.native_output_rows_to_table(rows, output_names)]
        except Exception:
            row_tables = [self.native_output_rows_to_table([row], output_names) for row in rows]

        groups: list[list[pa.Table]] = []
        for table in row_tables:
            if groups and table.schema.equals(groups[-1][0].schema):
                groups[-1].append(table)
            else:
                groups.append([table])
        return [group[0] if len(group) == 1 else pa.concat_tables(group) for group in groups]

    def validate_output_table(self, table: pa.Table) -> None:
        if not self.has_file_outputs:
            return
        boundary = f"UDF {self.udf_name!r} output"
        self._validate_column_count(table, self.output_types, boundary=boundary)
        for index, dtype in enumerate(self.output_types):
            if dtype is not None and _contains_file(dtype):
                validate_file_arrow_array(
                    table.column(index),
                    dtype,
                    boundary=f"{boundary} column {index}",
                    allow_untyped_null=True,
                )

    def normalize_output_table(self, table: pa.Table) -> pa.Table:
        """Validate FILE leaves and stabilize their declared output schema."""
        self.validate_output_table(table)
        if not self.has_file_outputs:
            return table

        boundary = f"UDF {self.udf_name!r} output"
        columns = list(table.columns)
        fields = list(table.schema)
        changed = False
        for index, dtype in enumerate(self.output_types):
            if dtype is None:
                continue
            try:
                normalized = _normalize_file_arrow_array(
                    columns[index],
                    dtype,
                    boundary=f"{boundary} column {index}",
                )
            except Exception:
                raise _invalid_input(f"{boundary} column {index} could not normalize its declared storage") from None
            normalized_field = pa.field(fields[index].name, normalized.type)
            if normalized.type.equals(columns[index].type) and fields[index].equals(
                normalized_field, check_metadata=True
            ):
                continue
            columns[index] = normalized
            fields[index] = normalized_field
            changed = True

        if not changed:
            return table
        return pa.Table.from_arrays(columns, schema=pa.schema(fields, metadata=table.schema.metadata))


__all__ = [
    "FileUDFContract",
    "contains_file_type",
    "normalize_file_arrow_array",
    "validate_file_arrow_array",
]
