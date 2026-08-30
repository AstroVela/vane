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
_ARROW_LIST_OFFSET_MAX = (1 << 31) - 1
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


def _contains_native_struct(dtype: Any | None) -> bool:
    if dtype is None or _is_file_type(dtype):
        return False
    type_id = _type_id(dtype)
    if type_id == "struct":
        return True
    if type_id in ("list", "array", "tensor"):
        return _contains_native_struct(_sequence_child(dtype))
    if type_id == "map":
        return any(_contains_native_struct(child) for _, child in dtype.children)
    # Native UNION values remain opaque until their selected tag can be
    # represented explicitly at the Python boundary.
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


def _is_arrow_large_list_storage(dtype: pa.DataType) -> bool:
    is_large_list_view = getattr(pa.types, "is_large_list_view", None)
    return pa.types.is_large_list(dtype) or (callable(is_large_list_view) and is_large_list_view(dtype))


def _is_arrow_list_storage(dtype: pa.DataType) -> bool:
    is_list_view = getattr(pa.types, "is_list_view", None)
    return (
        pa.types.is_list(dtype)
        or (callable(is_list_view) and is_list_view(dtype))
        or _is_arrow_large_list_storage(dtype)
    )


def _is_arrow_list_like_storage(dtype: pa.DataType) -> bool:
    return _is_arrow_list_storage(dtype) or pa.types.is_fixed_size_list(dtype)


def _list_storage_parts(array: Any) -> tuple[list[int | None], Any, bool] | None:
    """Return logical row lengths and active children for Arrow list-like storage."""
    if _is_arrow_list_storage(array.type):
        lengths = [None if length is None else int(length) for length in pc.list_value_length(array).to_pylist()]
        return lengths, pc.list_flatten(array), _is_arrow_large_list_storage(array.type)
    if not pa.types.is_fixed_size_list(array.type):
        return None

    list_size = array.type.list_size
    lengths = [list_size if is_valid else None for is_valid in array.is_valid().to_pylist()]
    values = array.values.slice(array.offset * list_size, len(array) * list_size)
    if all(length is not None for length in lengths):
        return lengths, values, False
    selected = [
        row_index * list_size + child_index
        for row_index, length in enumerate(lengths)
        if length is not None
        for child_index in range(list_size)
    ]
    return lengths, values.take(pa.array(selected, type=pa.int64())), False


def _fixed_sequence_child_source(array: Any, dtype: Any, *, boundary: str) -> Any | None:
    parts = _list_storage_parts(array)
    if parts is None:
        return None
    lengths, flattened, _ = parts
    expected_size = _fixed_sequence_size(dtype)
    if any(length is not None and length != expected_size for length in lengths):
        raise _invalid_input(f"{boundary} {_type_id(dtype).upper()} value must have fixed size {expected_size}")
    if all(length is not None for length in lengths):
        return flattened

    flattened_index = 0
    child_indices: list[int | None] = []
    for length in lengths:
        if length is None:
            child_indices.extend([None] * expected_size)
            continue
        child_indices.extend(range(flattened_index, flattened_index + expected_size))
        flattened_index += length
    return flattened.take(pa.array(child_indices, type=pa.int64()))


def _list_array_from_offsets(
    offsets: Sequence[int],
    values: Any,
    *,
    mask: Any,
    force_large: bool = False,
    value_field: pa.Field | None = None,
) -> Any:
    use_large = force_large or offsets[-1] > _ARROW_LIST_OFFSET_MAX
    offset_type = pa.int64() if use_large else pa.int32()
    constructor = pa.LargeListArray.from_arrays if use_large else pa.ListArray.from_arrays
    target_type = None
    if value_field is not None:
        normalized_field = value_field.with_type(values.type)
        target_type = pa.large_list(normalized_field) if use_large else pa.list_(normalized_field)
    return constructor(pa.array(offsets, type=offset_type), values, type=target_type, mask=mask)


def _null_bitmap_from_mask(mask: Any) -> tuple[Any | None, int]:
    nulls = [bool(value) for value in mask.to_pylist()]
    null_count = sum(nulls)
    if not null_count:
        return None, 0
    validity = pa.array([not is_null for is_null in nulls], type=pa.bool_())
    return validity.buffers()[1], null_count


def _fixed_size_list_array(
    values: Any,
    list_size: int,
    *,
    mask: Any,
    value_field: pa.Field | None = None,
) -> Any:
    # PyArrow 14 has no mask argument on FixedSizeListArray.from_arrays.
    # Build the exact field-bearing type from its validity and child buffers.
    normalized_field = pa.field("item", values.type) if value_field is None else value_field.with_type(values.type)
    target_type = pa.list_(normalized_field, list_size=list_size)
    null_bitmap, null_count = _null_bitmap_from_mask(mask)
    return pa.Array.from_buffers(
        target_type,
        len(mask),
        [null_bitmap],
        null_count=null_count,
        children=[values],
    )


def _map_array_from_offsets(
    offsets: Sequence[int],
    keys: Any,
    items: Any,
    *,
    mask: Any,
    map_type: pa.MapType | None = None,
) -> Any:
    # PyArrow 14 has neither type nor mask arguments on MapArray.from_arrays.
    # The buffer constructor preserves both the parent validity and child fields.
    target_type = pa.map_(keys.type, items.type) if map_type is None else map_type
    entries = pa.StructArray.from_arrays(
        [keys, items],
        fields=[target_type.key_field, target_type.item_field],
    )
    offset_array = pa.array(offsets, type=pa.int32())
    null_bitmap, null_count = _null_bitmap_from_mask(mask)
    return pa.Array.from_buffers(
        target_type,
        len(mask),
        [null_bitmap, offset_array.buffers()[1]],
        null_count=null_count,
        children=[entries],
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
            fields[field_index] = field.with_type(annotated.type)
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
            normalized_field = array.type.value_field.with_type(annotated.type)
            normalized_type = (
                pa.large_list(normalized_field) if pa.types.is_large_list(array.type) else pa.list_(normalized_field)
            )
            return constructor(normalized_offsets, annotated, type=normalized_type, mask=array.is_null())
        is_list_view = getattr(pa.types, "is_list_view", None)
        if callable(is_list_view) and is_list_view(array.type):
            values = _annotate_duckdb_bit_input(array.values, _sequence_child(dtype), boundary=boundary)
            if values.type.equals(array.values.type):
                return array
            return pa.ListViewArray.from_arrays(
                array.offsets,
                array.sizes,
                values,
                type=pa.list_view(array.type.value_field.with_type(values.type)),
                mask=array.is_null(),
            )
        is_large_list_view = getattr(pa.types, "is_large_list_view", None)
        if callable(is_large_list_view) and is_large_list_view(array.type):
            values = _annotate_duckdb_bit_input(array.values, _sequence_child(dtype), boundary=boundary)
            if values.type.equals(array.values.type):
                return array
            return pa.LargeListViewArray.from_arrays(
                array.offsets,
                array.sizes,
                values,
                type=pa.large_list_view(array.type.value_field.with_type(values.type)),
                mask=array.is_null(),
            )
        return array

    if type_id in ("array", "tensor"):
        storage = array.storage if type_id == "tensor" and _is_arrow_extension_type(array.type) else array
        array_size = storage.type.list_size
        values = storage.values.slice(storage.offset * array_size, len(storage) * array_size)
        annotated = _annotate_duckdb_bit_input(values, _sequence_child(dtype), boundary=boundary)
        if annotated.type.equals(values.type):
            return array
        if type_id == "array":
            return _fixed_size_list_array(
                annotated,
                array_size,
                mask=storage.is_null(),
                value_field=storage.type.value_field,
            )
        tensor_type = pa.fixed_shape_tensor(annotated.type, _tensor_shape(dtype))
        normalized_storage = _fixed_size_list_array(
            annotated,
            array_size,
            mask=storage.is_null(),
            value_field=tensor_type.storage_type.value_field,
        )
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
        normalized_offsets = [offset - start for offset in offsets]
        normalized_type = pa.map_(
            array.type.key_field.with_type(keys.type),
            array.type.item_field.with_type(items.type),
            keys_sorted=array.type.keys_sorted,
        )
        return _map_array_from_offsets(
            normalized_offsets,
            keys,
            items,
            mask=array.is_null(),
            map_type=normalized_type,
        )

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
        if not _is_arrow_list_like_storage(actual_storage):
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
        if not _is_arrow_list_like_storage(actual):
            raise _invalid_input(f"{boundary} value at {path} must use an Arrow list-like type")
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
        if not _is_arrow_list_like_storage(actual_storage):
            raise _invalid_input(f"{boundary} value at {path} must use an Arrow list-like type")
        if pa.types.is_fixed_size_list(actual_storage) and actual_storage.list_size != _fixed_sequence_size(dtype):
            raise _invalid_input(f"{boundary} value at {path} must have fixed size {_fixed_sequence_size(dtype)}")
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
        parts = _list_storage_parts(source)
        if parts is None:
            return
        _, child_source, _ = parts
        _validate_file_arrow_values(
            child_source,
            _sequence_child(dtype),
            boundary=boundary,
            path=f"{path}[]",
        )
        return
    if type_id in ("array", "tensor"):
        source = _mask_inactive(array, active)
        storage = source.storage if type_id == "tensor" and isinstance(source, pa.ExtensionArray) else source
        array_size = _fixed_sequence_size(dtype)
        child_source = _fixed_sequence_child_source(storage, dtype, boundary=boundary)
        if child_source is None:
            return
        _validate_file_arrow_values(
            child_source,
            _sequence_child(dtype),
            boundary=boundary,
            path=f"{path}[]",
            parent_active=[is_active for is_active in active for _ in range(array_size)],
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


def _native_struct_child_values(
    value: Any,
    dtype: Any,
    *,
    boundary: str,
    path: str,
    require_struct: bool,
) -> list[tuple[str, Any, Any]] | None:
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
        if len(declared_keys) != len(struct_children) or set(actual_keys) != declared_keys:
            raise _invalid_input(f"{boundary} STRUCT value at {path} must contain exactly the declared fields")
        child_values = [value[actual_keys[name.casefold()]] for name, _ in struct_children]
    elif isinstance(value, tuple):
        if len(value) != len(struct_children):
            raise _invalid_input(
                f"{boundary} STRUCT value at {path} must contain exactly {len(struct_children)} positional fields"
            )
        child_values = list(value)
    elif require_struct:
        raise _invalid_input(f"{boundary} value at {path} must be a mapping or positional tuple")
    else:
        return None
    return [
        (name, child, child_value) for (name, child), child_value in zip(struct_children, child_values, strict=True)
    ]


def _validate_native_nested_struct_field_sets(
    value: Any,
    dtype: Any,
    *,
    boundary: str,
    path: str,
) -> None:
    """Validate native STRUCT shapes while leaving cross-type casts untouched."""
    if value is None or not _contains_native_struct(dtype):
        return

    type_id = _type_id(dtype)
    if type_id == "struct":
        child_values = _native_struct_child_values(
            value,
            dtype,
            boundary=boundary,
            path=path,
            require_struct=False,
        )
        if child_values is None:
            return
        for name, child, child_value in child_values:
            _validate_native_nested_struct_field_sets(
                child_value,
                child,
                boundary=boundary,
                path=f"{path}.{name}",
            )
        return
    if type_id in ("list", "array", "tensor"):
        if not isinstance(value, (list, tuple)):
            return
        child = _sequence_child(dtype)
        for index, item in enumerate(value):
            _validate_native_nested_struct_field_sets(
                item,
                child,
                boundary=boundary,
                path=f"{path}[{index}]",
            )
        return
    if type_id == "map":
        children = _type_children(dtype)
        if (
            isinstance(value, Mapping)
            and set(value) == {"key", "value"}
            and isinstance(value["key"], (list, tuple))
            and isinstance(value["value"], (list, tuple))
            and not _native_map_key_is_hashable(children["key"])
        ):
            entries: Any = zip(value["key"], value["value"])
        elif isinstance(value, Mapping):
            entries = value.items()
        elif isinstance(value, (list, tuple)):
            entries = value
        else:
            return
        for index, entry in enumerate(entries):
            if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes, bytearray)) or len(entry) != 2:
                continue
            key, item = entry
            _validate_native_nested_struct_field_sets(
                key,
                children["key"],
                boundary=boundary,
                path=f"{path}[{index}].key",
            )
            _validate_native_nested_struct_field_sets(
                item,
                children["value"],
                boundary=boundary,
                path=f"{path}[{index}].value",
            )


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
            if not _contains_file(dtype):
                return value
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
        child_values = _native_struct_child_values(
            value,
            dtype,
            boundary=boundary,
            path=path,
            require_struct=_contains_file(dtype),
        )
        if child_values is None:
            return value
        result = {}
        for name, child, child_value in child_values:
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
            if not _contains_file(dtype):
                return value
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


def _native_value_uses_declared_container(value: Any, type_id: str) -> bool:
    if type_id == "struct":
        return isinstance(value, Mapping)
    if type_id in ("list", "array", "tensor", "map"):
        return isinstance(value, (list, tuple))
    return False


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
    if type_id in ("bignum", "hugeint", "uhugeint"):
        encoded: list[str | None] = []
        has_encoded = False
        has_passthrough = False
        for value in values:
            if value is None:
                encoded.append(None)
                continue
            if isinstance(value, str):
                encoded.append(value)
                has_encoded = True
                continue
            try:
                integer = operator.index(value)
            except TypeError:
                encoded.append(None)
                has_passthrough = True
                continue
            if type_id == "hugeint":
                in_range = -(1 << 127) <= integer < 1 << 127
            elif type_id == "uhugeint":
                in_range = 0 <= integer < 1 << 128
            else:
                in_range = True
            if not in_range:
                raise ValueError(f"{type_id.upper()} output is outside its 128-bit range")
            encoded.append(str(integer))
            has_encoded = True
        if has_passthrough:
            if has_encoded:
                raise TypeError(f"{type_id.upper()} output mixes encoded and native numeric storage")
            return pa.array(values)
        return pa.array(encoded, type=pa.string())
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

    if type_id in ("struct", "list", "array", "tensor", "map"):
        container_matches = [
            _native_value_uses_declared_container(value, type_id) for value in values if value is not None
        ]
        if container_matches and not all(container_matches):
            if any(container_matches):
                raise TypeError(f"{type_id.upper()} output mixes declared-container and cross-type storage")
            return pa.array(values)

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
        return _list_array_from_offsets(offsets, child_array, mask=null_mask)

    if type_id in ("array", "tensor"):
        child = _sequence_child(dtype)
        array_size = _fixed_sequence_size(dtype)
        flattened = []
        for value in values:
            flattened.extend([None] * array_size if value is None else value)
        child_array = _canonical_values_to_arrow_array(flattened, child, boundary=boundary)
        storage = _fixed_size_list_array(child_array, array_size, mask=null_mask)
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
        return _map_array_from_offsets(
            offsets,
            key_array,
            item_array,
            mask=null_mask,
        )

    raise _invalid_input(f"{boundary} uses an unsupported type containing FILE: {dtype}")


def _native_outputs_to_arrow_array(values: Sequence[Any], dtype: Any, *, boundary: str) -> pa.Array:
    canonical = []
    for row_index, value in enumerate(values):
        path = f"row {row_index}"
        _validate_native_nested_struct_field_sets(value, dtype, boundary=boundary, path=path)
        canonical.append(_canonicalize_native_output(value, dtype, boundary=boundary, path=path))
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


def _collect_large_list_paths(
    actual: pa.DataType,
    dtype: Any,
    *,
    logical_path: tuple[int, ...],
    result: set[tuple[int, ...]],
) -> None:
    """Collect declared LIST nodes normalized with 64-bit Arrow offsets."""
    if _is_file_type(dtype):
        return

    type_id = _type_id(dtype)
    if type_id == "list":
        if not _is_arrow_list_like_storage(actual):
            return
        if _is_arrow_large_list_storage(actual):
            result.add(logical_path)
        _collect_large_list_paths(
            actual.value_type,
            _sequence_child(dtype),
            logical_path=(*logical_path, 0),
            result=result,
        )
        return
    if type_id in ("array", "tensor"):
        actual_storage = actual.storage_type if type_id == "tensor" and _is_arrow_extension_type(actual) else actual
        if not _is_arrow_list_like_storage(actual_storage):
            return
        _collect_large_list_paths(
            actual_storage.value_type,
            _sequence_child(dtype),
            logical_path=(*logical_path, 0),
            result=result,
        )
        return
    if type_id == "struct" and pa.types.is_struct(actual):
        for child_index, (_, child) in enumerate(dtype.children):
            if child_index >= actual.num_fields:
                return
            _collect_large_list_paths(
                actual.field(child_index).type,
                child,
                logical_path=(*logical_path, child_index),
                result=result,
            )
        return
    if type_id == "map" and pa.types.is_map(actual):
        children = _type_children(dtype)
        _collect_large_list_paths(
            actual.key_type,
            children["key"],
            logical_path=(*logical_path, 0),
            result=result,
        )
        _collect_large_list_paths(
            actual.item_type,
            children["value"],
            logical_path=(*logical_path, 1),
            result=result,
        )


def _normalize_file_arrow_array(
    array: Any,
    dtype: Any,
    *,
    boundary: str,
    parent_active: Sequence[bool] | None = None,
    normalize_value_dependent: bool = True,
    force_large_list_paths: frozenset[tuple[int, ...]] = frozenset(),
    logical_path: tuple[int, ...] = (),
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
                force_large_list_paths=force_large_list_paths,
                logical_path=logical_path,
            )
            return pa.chunked_array([], type=normalized_empty.type)

        def normalize_chunks(
            *,
            value_dependent: bool,
            large_list_paths: frozenset[tuple[int, ...]],
        ) -> list[Any]:
            result = []
            offset = 0
            for chunk in array.chunks:
                chunk_active = None if parent_active is None else parent_active[offset : offset + len(chunk)]
                result.append(
                    _normalize_file_arrow_array(
                        chunk,
                        dtype,
                        boundary=boundary,
                        parent_active=chunk_active,
                        normalize_value_dependent=value_dependent,
                        force_large_list_paths=large_list_paths,
                        logical_path=logical_path,
                    )
                )
                offset += len(chunk)
            return result

        chunks = normalize_chunks(
            value_dependent=normalize_value_dependent,
            large_list_paths=force_large_list_paths,
        )
        normalized_type = chunks[0].type
        if all(chunk.type.equals(normalized_type) for chunk in chunks[1:]):
            return pa.chunked_array(chunks, type=normalized_type)

        # Some safe casts, notably temporal downcasts, depend on the values.
        # Retry every original chunk without those casts so the whole logical
        # column keeps one schema without concatenating or materializing it.
        stable_chunks = normalize_chunks(value_dependent=False, large_list_paths=force_large_list_paths)
        stable_type = stable_chunks[0].type
        if all(chunk.type.equals(stable_type) for chunk in stable_chunks[1:]):
            return pa.chunked_array(stable_chunks, type=stable_type)

        # A ListView chunk can logically flatten beyond INT32_MAX even when
        # another chunk in the same column remains small. Promote each LIST
        # path used by any normalized chunk, then rebuild every chunk with one
        # stable offset width at that path.
        promoted_paths = set(force_large_list_paths)
        for chunk in stable_chunks:
            _collect_large_list_paths(
                chunk.type,
                dtype,
                logical_path=logical_path,
                result=promoted_paths,
            )
        if promoted_paths != set(force_large_list_paths):
            frozen_promoted_paths = frozenset(promoted_paths)
            promoted_chunks = normalize_chunks(
                value_dependent=normalize_value_dependent,
                large_list_paths=frozen_promoted_paths,
            )
            promoted_type = promoted_chunks[0].type
            if all(chunk.type.equals(promoted_type) for chunk in promoted_chunks[1:]):
                return pa.chunked_array(promoted_chunks, type=promoted_type)
            if normalize_value_dependent:
                promoted_chunks = normalize_chunks(
                    value_dependent=False,
                    large_list_paths=frozen_promoted_paths,
                )
                promoted_type = promoted_chunks[0].type
                if all(chunk.type.equals(promoted_type) for chunk in promoted_chunks[1:]):
                    return pa.chunked_array(promoted_chunks, type=promoted_type)

        raise RuntimeError("FILE normalization could not stabilize a chunked Arrow column")
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
        for child_index, (name, child) in enumerate(dtype.children):
            field_index = _struct_field_index(source.type, name, boundary=boundary, path="column")
            source_field = source.type.field(field_index)
            child_array = source.field(field_index)
            child_array = _normalize_file_arrow_array(
                child_array,
                child,
                boundary=boundary,
                parent_active=active,
                normalize_value_dependent=normalize_value_dependent,
                force_large_list_paths=force_large_list_paths,
                logical_path=(*logical_path, child_index),
            )
            arrays.append(child_array)
            fields.append(source_field.with_name(name).with_type(child_array.type))
        return pa.StructArray.from_arrays(arrays, fields=fields, mask=source.is_null())

    if type_id == "list":
        source = _mask_inactive(array, active)
        parts = _list_storage_parts(source)
        if parts is None:
            return source

        lengths, child_source, force_large = parts
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + (0 if length is None else length))
        child_array = _normalize_file_arrow_array(
            child_source,
            _sequence_child(dtype),
            boundary=boundary,
            normalize_value_dependent=normalize_value_dependent,
            force_large_list_paths=force_large_list_paths,
            logical_path=(*logical_path, 0),
        )
        return _list_array_from_offsets(
            offsets,
            child_array,
            mask=source.is_null(),
            force_large=force_large or logical_path in force_large_list_paths,
            value_field=source.type.value_field,
        )

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
        child_source = _fixed_sequence_child_source(storage, dtype, boundary=boundary)
        if child_source is None:
            return source

        child_array = _normalize_file_arrow_array(
            child_source,
            _sequence_child(dtype),
            boundary=boundary,
            parent_active=[is_active for is_active in active for _ in range(array_size)],
            normalize_value_dependent=normalize_value_dependent,
            force_large_list_paths=force_large_list_paths,
            logical_path=(*logical_path, 0),
        )
        if type_id == "array":
            return _fixed_size_list_array(
                child_array,
                array_size,
                mask=source.is_null(),
                value_field=storage.type.value_field,
            )
        tensor_type = pa.fixed_shape_tensor(child_array.type, _tensor_shape(dtype))
        normalized_storage = _fixed_size_list_array(
            child_array,
            array_size,
            mask=source.is_null(),
            value_field=tensor_type.storage_type.value_field,
        )
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
            force_large_list_paths=force_large_list_paths,
            logical_path=(*logical_path, 0),
        )
        items = _normalize_file_arrow_array(
            items,
            children["value"],
            boundary=boundary,
            normalize_value_dependent=normalize_value_dependent,
            force_large_list_paths=force_large_list_paths,
            logical_path=(*logical_path, 1),
        )
        normalized_map_type = pa.map_(
            source.type.key_field.with_type(keys.type),
            source.type.item_field.with_type(items.type),
            keys_sorted=source.type.keys_sorted,
        )
        return _map_array_from_offsets(
            normalized_offsets,
            keys,
            items,
            mask=source.is_null(),
            map_type=normalized_map_type,
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
            fields[index] = fields[index].with_type(annotated.type)
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
            normalized_field = fields[index].with_type(normalized.type)
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
