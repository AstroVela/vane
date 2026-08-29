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


def _native_map_key_is_hashable(dtype: Any) -> bool:
    if _is_file_type(dtype):
        return True
    type_id = _type_id(dtype)
    if type_id in ("list", "struct", "map", "union"):
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

    import vane

    try:
        dtype = vane.type(type_name)
    except Exception as exc:
        raise _invalid_input(f"UDF payload field {field!r} contains an invalid SQL type") from exc
    return dtype if _contains_file(dtype) else None


def _parse_declared_type(type_name: Any, *, field: str) -> Any | None:
    if not isinstance(type_name, str):
        return None

    import vane

    try:
        return vane.type(type_name)
    except Exception as exc:
        raise _invalid_input(f"UDF payload field {field!r} contains an invalid SQL type") from exc


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
    if isinstance(actual, pa.ExtensionType) and actual.storage_type.equals(expected):
        return True
    if (pa.types.is_string(actual) or pa.types.is_large_string(actual)) and (
        pa.types.is_string(expected) or pa.types.is_large_string(expected)
    ):
        return True
    if (pa.types.is_binary(actual) or pa.types.is_large_binary(actual)) and (
        pa.types.is_binary(expected) or pa.types.is_large_binary(expected)
    ):
        return True
    if pa.types.is_signed_integer(actual) and pa.types.is_signed_integer(expected):
        return expected.bit_width >= actual.bit_width
    if pa.types.is_unsigned_integer(actual) and pa.types.is_unsigned_integer(expected):
        return expected.bit_width >= actual.bit_width
    if pa.types.is_unsigned_integer(actual) and pa.types.is_signed_integer(expected):
        return expected.bit_width > actual.bit_width
    if pa.types.is_float32(actual) and pa.types.is_float64(expected):
        return True
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
        start, length, offsets = _child_window(array)
        child_active = [
            is_active for index, is_active in enumerate(active) for _ in range(offsets[index + 1] - offsets[index])
        ]
        _validate_file_arrow_values(
            array.values.slice(start, length),
            _sequence_child(dtype),
            boundary=boundary,
            path=f"{path}[]",
            parent_active=child_active,
        )
        return
    if type_id == "array":
        array_size = int(_type_children(dtype)["size"])
        child_active = [is_active for is_active in active for _ in range(array_size)]
        _validate_file_arrow_values(
            array.values.slice(array.offset * array_size, len(array) * array_size),
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
    if type_id in ("list", "array"):
        child = _sequence_child(dtype)
        values = [
            _materialize_native_value(item, child, boundary=boundary, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
        return tuple(values) if type_id == "array" else values
    if type_id == "struct":
        result = {}
        for name, child in dtype.children:
            child_value = _mapping_field_value(value, name, boundary=boundary, path=path)
            result[name] = (
                _materialize_native_value(child_value, child, boundary=boundary, path=f"{path}.{name}")
                if _contains_file(child)
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
    if type_id in ("list", "array"):
        if not isinstance(value, (list, tuple)):
            raise _invalid_input(f"{boundary} value at {path} must be a sequence")
        if type_id == "array":
            expected_size = int(_type_children(dtype)["size"])
            if len(value) != expected_size:
                raise _invalid_input(f"{boundary} value at {path} must have fixed size {expected_size}")
        child = _sequence_child(dtype)
        return [
            _canonicalize_native_output(item, child, boundary=boundary, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if type_id == "struct":
        if not isinstance(value, Mapping):
            raise _invalid_input(f"{boundary} value at {path} must be a mapping")
        result = {}
        for name, child in dtype.children:
            child_value = _mapping_field_value(value, name, boundary=boundary, path=path)
            result[name] = (
                _canonicalize_native_output(child_value, child, boundary=boundary, path=f"{path}.{name}")
                if _contains_file(child)
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
    """Encode canonical values while typing only branches that contain FILE."""
    if _is_file_type(dtype):
        return pa.array(values, type=_expected_arrow_type(dtype, boundary=boundary))
    type_id = _type_id(dtype)
    if type_id in ("hugeint", "uhugeint"):
        encoded: list[str | None] = []
        for value in values:
            if value is None:
                encoded.append(None)
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
        encoded = [value.isoformat() if isinstance(value, datetime_time) else value for value in values]
        return pa.array(encoded, type=pa.string())
    if not _contains_file(dtype):
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

    if type_id == "array":
        child = _sequence_child(dtype)
        array_size = int(_type_children(dtype)["size"])
        flattened = []
        for value in values:
            flattened.extend([None] * array_size if value is None else value)
        child_array = _canonical_values_to_arrow_array(flattened, child, boundary=boundary)
        return pa.FixedSizeListArray.from_arrays(child_array, array_size, mask=null_mask)

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


def _normalize_file_arrow_array(
    array: Any,
    dtype: Any,
    *,
    boundary: str,
    parent_active: Sequence[bool] | None = None,
) -> Any:
    """Canonicalize a FILE-bearing Arrow value to its declared stable storage."""
    if isinstance(array, pa.ChunkedArray):
        if not array.chunks:
            normalized_empty = _normalize_file_arrow_array(
                pa.array([], type=array.type),
                dtype,
                boundary=boundary,
                parent_active=parent_active,
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
                )
            )
            offset += len(chunk)
        return pa.chunked_array(chunks)
    active = _active_values(array, parent_active)
    type_id = _type_id(dtype)
    if type_id == "uuid":
        source = _mask_inactive(array, active)
        encoded: list[str | None] = []
        for value in source.to_pylist():
            if value is None:
                encoded.append(None)
            elif isinstance(value, UUID):
                encoded.append(str(value))
            elif isinstance(value, (bytes, bytearray, memoryview)):
                raw_value = bytes(value)
                encoded.append(
                    str(UUID(bytes=raw_value))
                    if pa.types.is_fixed_size_binary(source.type) and source.type.byte_width == 16
                    else raw_value.decode()
                )
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
        arrays = []
        fields = []
        for name, child in dtype.children:
            field_index = _optional_struct_field_index(array.type, name, boundary=boundary, path="column")
            if field_index is None:
                child_array = pa.nulls(len(array))
            else:
                child_array = array.field(field_index)
            child_array = _normalize_file_arrow_array(
                child_array,
                child,
                boundary=boundary,
                parent_active=active,
            )
            arrays.append(child_array)
            fields.append(pa.field(name, child_array.type))
        return pa.StructArray.from_arrays(arrays, fields=fields, mask=array.is_null())

    if type_id == "list":
        start, length, offsets = _child_window(array)
        child_array = _normalize_file_arrow_array(
            array.values.slice(start, length),
            _sequence_child(dtype),
            boundary=boundary,
            parent_active=[
                is_active for index, is_active in enumerate(active) for _ in range(offsets[index + 1] - offsets[index])
            ],
        )
        normalized_offsets = pa.array([offset - start for offset in offsets], type=pa.int32())
        return pa.ListArray.from_arrays(normalized_offsets, child_array, mask=array.is_null())

    if type_id == "array":
        array_size = int(_type_children(dtype)["size"])
        child_array = _normalize_file_arrow_array(
            array.values.slice(array.offset * array_size, len(array) * array_size),
            _sequence_child(dtype),
            boundary=boundary,
            parent_active=[is_active for is_active in active for _ in range(array_size)],
        )
        return pa.FixedSizeListArray.from_arrays(child_array, array_size, mask=array.is_null())

    if type_id == "map":
        start, _, offsets = _child_window(array)
        children = _type_children(dtype)
        selected_indices: list[int] = []
        normalized_offsets = [0]
        for index, is_active in enumerate(active):
            if is_active:
                selected_indices.extend(range(offsets[index] - start, offsets[index + 1] - start))
            normalized_offsets.append(len(selected_indices))
        selection = pa.array(selected_indices, type=pa.int64())
        keys = array.keys.slice(start, offsets[-1] - start).take(selection)
        items = array.items.slice(start, offsets[-1] - start).take(selection)
        keys = _normalize_file_arrow_array(keys, children["key"], boundary=boundary)
        items = _normalize_file_arrow_array(items, children["value"], boundary=boundary)
        return pa.MapArray.from_arrays(
            pa.array(normalized_offsets, type=pa.int32()),
            keys,
            items,
            mask=array.is_null(),
        )

    source = _mask_inactive(array, active)
    try:
        expected = _arrow_type_from_duckdb_pytype(dtype)
    except Exception:
        return source
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
                entries: list[Mapping[str, Any] | None] = []
                file_types: list[Any | None] = []
                for entry in output_schema:
                    if not isinstance(entry, Mapping):
                        entries.append(None)
                        file_types.append(None)
                        continue
                    kind = str(entry.get("kind") or "duckdb_type").lower()
                    entries.append(entry)
                    file_types.append(
                        _parse_file_type(entry.get("type"), field="output_schema") if kind == "duckdb_type" else None
                    )
                if any(dtype is not None for dtype in file_types):
                    output_types.extend(
                        file_dtype
                        if file_dtype is not None
                        else (
                            _parse_declared_type(entry.get("type"), field="output_schema")
                            if entry is not None and str(entry.get("kind") or "duckdb_type").lower() == "duckdb_type"
                            else None
                        )
                        for entry, file_dtype in zip(entries, file_types, strict=True)
                    )
                else:
                    output_types.extend(file_types)

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
        return any(dtype is not None and _contains_file(dtype) for dtype in self.output_types)

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
        return _native_outputs_to_arrow_array(outputs, dtype, boundary=boundary)

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

        arrays: dict[str, Any] = {}
        for index, name in enumerate(output_names):
            values = [row.get(name) for row in rows]
            dtype = self.output_types[index]
            if dtype is None:
                arrays[name] = values
                continue
            arrays[name] = _native_outputs_to_arrow_array(values, dtype, boundary=boundary)

        table = pa.table(arrays)
        self.validate_output_table(table)
        return table

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
            if normalized.type.equals(columns[index].type, check_metadata=True) and fields[index].equals(
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
