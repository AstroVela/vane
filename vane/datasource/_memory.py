# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Internal Arrow snapshot tasks for Python-owned Ray relation sources."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

_TARGET_PARTITION_BYTES = 16 * 1024 * 1024
_MAX_PARTITION_ROWS = 1_000_000
_FILE_FIELD_NAMES = ("url", "content_type", "position", "size", "checksum")


class _RayMemorySourceTask:
    """Resolve one query-owned Arrow table and expose it as record batches."""

    def __init__(self, source_id: str, object_ref: Any, column_indices: tuple[int, ...]) -> None:
        self.source_id = str(source_id)
        self.object_ref = object_ref
        self.column_indices = tuple(column_indices)

    def execute(self) -> Iterator[Any]:
        import pyarrow as pa
        import ray

        partition = ray.get(self.object_ref)
        if not isinstance(partition, pa.Table):
            raise TypeError(
                f"Ray memory source {self.source_id!r} resolved to {type(partition).__name__}, expected pyarrow.Table"
            )
        if self.column_indices != tuple(range(partition.num_columns)):
            partition = partition.select(self.column_indices)
        yield from partition.to_batches()


def _append_version_integer(version: bytearray, value: int) -> None:
    version.extend(int(value).to_bytes(8, byteorder="little", signed=False))


def _append_version_bytes(version: bytearray, value: bytes) -> None:
    _append_version_integer(version, len(value))
    version.extend(value)


def _arrow_array_children(value: Any) -> tuple[Any, ...]:
    """Return logical child arrays, including storage outside Array.buffers()."""

    import pyarrow as pa

    if isinstance(value, pa.ExtensionArray):
        return (value.storage,)
    if pa.types.is_dictionary(value.type):
        # Dictionary values are stored outside the index buffers returned by
        # Array.buffers(), so they must be visited explicitly.
        return (value.dictionary,)
    is_list_view = getattr(pa.types, "is_list_view", lambda _: False)
    is_large_list_view = getattr(pa.types, "is_large_list_view", lambda _: False)
    if (
        pa.types.is_list(value.type)
        or pa.types.is_large_list(value.type)
        or pa.types.is_fixed_size_list(value.type)
        or is_list_view(value.type)
        or is_large_list_view(value.type)
        or pa.types.is_map(value.type)
    ):
        return (value.values,)
    if pa.types.is_struct(value.type) or pa.types.is_union(value.type):
        return tuple(value.field(index) for index in range(value.type.num_fields))
    is_run_end_encoded = getattr(pa.types, "is_run_end_encoded", lambda _: False)
    if is_run_end_encoded(value.type):
        return (value.run_ends, value.values)
    return ()


def _append_arrow_array_version(version: bytearray, value: Any) -> None:
    _append_version_bytes(version, str(value.type).encode())
    _append_version_integer(version, len(value))
    _append_version_integer(version, value.offset)
    # Container Array.buffers() flattens descendant buffers. Record only this
    # node's layout; the recursion below visits every logical child exactly once.
    buffers = value.buffers()[: value.type.num_buffers]
    _append_version_integer(version, len(buffers))
    for buffer in buffers:
        _append_version_integer(version, buffer is not None)
        if buffer is not None:
            _append_version_integer(version, buffer.address)
            _append_version_integer(version, buffer.size)
    for child in _arrow_array_children(value):
        _append_arrow_array_version(version, child)


def _arrow_source_version(source: Any) -> bytes:
    """Return a process-local fingerprint for an eager Arrow source's retained buffers."""

    import pyarrow as pa

    if isinstance(source, pa.RecordBatch):
        source = pa.Table.from_batches([source])
    if not isinstance(source, pa.Table):
        raise TypeError(
            f"Arrow source version requires pyarrow.Table or pyarrow.RecordBatch, got {type(source).__name__}"
        )

    version = bytearray()
    _append_version_bytes(version, source.schema.serialize().to_pybytes())
    _append_version_integer(version, source.num_rows)
    _append_version_integer(version, source.num_columns)
    for column in source.columns:
        _append_version_integer(version, column.num_chunks)
        for chunk in column.chunks:
            _append_arrow_array_version(version, chunk)
    return bytes(version)


def _as_arrow_table(source: Any, source_kind: str) -> Any:
    import pyarrow as pa

    if source_kind == "pandas":
        return pa.Table.from_pandas(source, preserve_index=False)
    if source_kind == "numpy":
        if not isinstance(source, dict):
            raise TypeError(f"NumPy memory source must be normalized to a dict, got {type(source).__name__}")
        return pa.Table.from_arrays(
            [pa.array(column, from_pandas=True) for column in source.values()],
            names=[str(name) for name in source],
        )
    if source_kind != "arrow":
        raise ValueError(f"Unsupported Python memory source kind: {source_kind!r}")

    if isinstance(source, pa.Table):
        return source
    if isinstance(source, pa.RecordBatch):
        return pa.Table.from_batches([source])
    raise TypeError(
        "Ray distributed execution only snapshots in-memory pyarrow.Table or pyarrow.RecordBatch sources; "
        f"got {type(source).__name__}. Materialize lazy or streaming Arrow sources explicitly before calling "
        "from_arrow()."
    )


def _select_memory_source_columns(source: Any, source_kind: str, column_indices: tuple[int, ...]) -> Any:
    import pyarrow as pa

    if source_kind == "pandas":
        return source.iloc[:, list(column_indices)]
    if source_kind == "numpy":
        source_names = tuple(source)
        return {source_names[column_index]: source[source_names[column_index]] for column_index in column_indices}
    if isinstance(source, (pa.Table, pa.RecordBatch)):
        return source.select(column_indices)
    return source


def _is_pandas_scan_null(value: Any) -> bool:
    import math

    import pandas as pd

    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float) and math.isnan(value)


def _pandas_scan_varchar_value(value: Any) -> str | None:
    """Match the native pandas_scan VARCHAR fallback for one object value."""

    if _is_pandas_scan_null(value):
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _is_arrow_string_type(data_type: Any) -> bool:
    import pyarrow as pa

    is_string_view = getattr(pa.types, "is_string_view", None)
    return (
        pa.types.is_string(data_type)
        or pa.types.is_large_string(data_type)
        or (is_string_view is not None and is_string_view(data_type))
    )


def _pandas_scan_map_value(value: Any) -> Any:
    if _is_pandas_scan_null(value):
        return None
    if not isinstance(value, dict):
        return value

    if len(value) == 2 and "key" in value and "value" in value:
        keys = value["key"]
        items = value["value"]
        keys_are_sequence = keys is None or (hasattr(keys, "__getitem__") and hasattr(keys, "__len__"))
        items_are_sequence = items is None or (hasattr(items, "__getitem__") and hasattr(items, "__len__"))
        if keys_are_sequence and items_are_sequence:
            if keys is None or items is None:
                return None
            if len(keys) == len(items):
                return list(zip(keys, items, strict=True))
    return list(value.items())


def _pandas_scan_map_array(values: Any, data_type: Any) -> Any:
    import pyarrow as pa

    return pa.array([_pandas_scan_map_value(value) for value in values], type=data_type, from_pandas=True)


def _pandas_scan_file_value(value: Any) -> dict[str, Any] | None:
    if _is_pandas_scan_null(value):
        return None
    return {field_name: getattr(value, field_name) for field_name in _FILE_FIELD_NAMES}


def _pandas_scan_file_array(values: Any, data_type: Any) -> Any:
    import pyarrow as pa

    return pa.array([_pandas_scan_file_value(value) for value in values], type=data_type, from_pandas=True)


def _coerce_object_columns(
    source: Any,
    source_kind: str,
    expected_schema: Any,
    logical_type_tags: tuple[str, ...],
) -> Any:
    """Apply bound pandas_scan semantics before Arrow attempts object inference."""

    import pandas as pd
    import pyarrow as pa

    if source_kind == "pandas":
        converted = None
        for column_index, dtype in enumerate(source.dtypes):
            if not pd.api.types.is_object_dtype(dtype):
                continue
            expected_type = expected_schema.field(column_index).type
            logical_type_tag = logical_type_tags[column_index]
            if logical_type_tag == "VARCHAR" or (logical_type_tag == "UUID" and _is_arrow_string_type(expected_type)):
                column = source.iloc[:, column_index].map(_pandas_scan_varchar_value)
            elif logical_type_tag == "MAP" and pa.types.is_map(expected_type):
                map_array = _pandas_scan_map_array(source.iloc[:, column_index], expected_type)
                column = pd.Series(map_array, dtype=pd.ArrowDtype(expected_type), index=source.index)
            elif logical_type_tag == "FILE" and pa.types.is_struct(expected_type):
                file_array = _pandas_scan_file_array(source.iloc[:, column_index], expected_type)
                column = pd.Series(file_array, dtype=pd.ArrowDtype(expected_type), index=source.index)
            else:
                continue
            if converted is None:
                converted = source.copy(deep=False)
            column_name = source.columns[column_index]
            converted[column_name] = column
        return source if converted is None else converted

    if source_kind == "numpy":
        if not isinstance(source, dict):
            raise TypeError(f"NumPy memory source must be normalized to a dict, got {type(source).__name__}")
        converted = None
        for column_index, (column_name, column) in enumerate(source.items()):
            if getattr(getattr(column, "dtype", None), "kind", None) != "O":
                continue
            expected_type = expected_schema.field(column_index).type
            logical_type_tag = logical_type_tags[column_index]
            if logical_type_tag == "VARCHAR" or (logical_type_tag == "UUID" and _is_arrow_string_type(expected_type)):
                converted_column = [_pandas_scan_varchar_value(value) for value in column]
            elif logical_type_tag == "MAP" and pa.types.is_map(expected_type):
                converted_column = _pandas_scan_map_array(column, expected_type)
            elif logical_type_tag == "FILE" and pa.types.is_struct(expected_type):
                converted_column = _pandas_scan_file_array(column, expected_type)
            else:
                continue
            if converted is None:
                converted = dict(source)
            converted[column_name] = converted_column
        return source if converted is None else converted

    return source


def _memory_source_row_count(source: Any, source_kind: str) -> int:
    import pyarrow as pa

    if source_kind == "pandas":
        return len(source)
    if source_kind == "numpy":
        if not isinstance(source, dict) or not source:
            raise TypeError("NumPy memory source must contain at least one normalized column")
        return len(next(iter(source.values())))
    if isinstance(source, (pa.Table, pa.RecordBatch)):
        return source.num_rows
    raise TypeError(
        "Ray distributed execution only snapshots in-memory pyarrow.Table or pyarrow.RecordBatch sources; "
        f"got {type(source).__name__}. Materialize lazy or streaming Arrow sources explicitly before calling "
        "from_arrow()."
    )


def _truncating_integer_divide(values: Any, divisor: int) -> Any:
    import numpy as np

    quotient = np.abs(values) // divisor
    return np.where(values < 0, -quotient, quotient)


def _duration_to_month_day_nano(column: Any) -> Any:
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    array = column.combine_chunks()
    validity = pc.is_valid(array).to_numpy(zero_copy_only=False)
    values = pc.fill_null(array.view(pa.int64()), 0).to_numpy(zero_copy_only=False)
    if array.type.unit == "ns":
        microseconds = _truncating_integer_divide(values, 1_000)
    elif array.type.unit == "us":
        microseconds = values
    elif array.type.unit == "ms":
        microseconds = values * 1_000
    elif array.type.unit == "s":
        microseconds = values * 1_000_000
    else:
        raise TypeError(f"Unsupported duration unit for a Python memory source: {array.type.unit!r}")

    microseconds_per_day = 24 * 60 * 60 * 1_000_000
    days = _truncating_integer_divide(microseconds, microseconds_per_day)
    remaining_microseconds = microseconds - days * microseconds_per_day
    months = _truncating_integer_divide(days, 30)
    remaining_days = days - months * 30

    intervals: Any = np.empty(
        len(array),
        dtype=[("months", np.int32), ("days", np.int32), ("nanoseconds", np.int64)],
    )
    intervals["months"] = months
    intervals["days"] = remaining_days
    intervals["nanoseconds"] = remaining_microseconds * 1_000
    validity_buffer = None
    if not validity.all():
        validity_buffer = pa.py_buffer(np.packbits(validity, bitorder="little"))
    return pa.Array.from_buffers(
        pa.month_day_nano_interval(),
        len(array),
        [validity_buffer, pa.py_buffer(intervals)],
        null_count=int(len(array) - validity.sum()),
    )


def _coerce_pandas_scan_table(table: Any, expected_schema: Any) -> Any:
    import pyarrow as pa
    import pyarrow.compute as pc

    columns = []
    for column_index, column in enumerate(table.columns):
        expected_type = expected_schema.field(column_index).type
        if pa.types.is_duration(column.type) and expected_type == pa.month_day_nano_interval():
            column = _duration_to_month_day_nano(column)
        elif (
            pa.types.is_timestamp(column.type) and pa.types.is_timestamp(expected_type) and column.type != expected_type
        ):
            # pandas_scan intentionally stores TIMESTAMP_TZ at microsecond
            # precision. Match its truncation semantics for finer input units.
            column = pc.cast(column, expected_type, safe=False)
        columns.append(column)
    normalized = pa.Table.from_arrays(columns, names=table.column_names)
    return normalized.cast(expected_schema, safe=True)


def _materialize_partition_column(column: Any) -> Any:
    import pyarrow as pa
    import pyarrow.compute as pc

    def reencode_dictionary_descendants(value: Any) -> Any:
        if isinstance(value, pa.ExtensionArray):
            storage = value.storage
            normalized_storage = reencode_dictionary_descendants(storage)
            if normalized_storage is storage:
                return value
            return pa.ExtensionArray.from_storage(value.type, normalized_storage)
        if pa.types.is_dictionary(value.type):
            decoded = pc.dictionary_decode(value)
            reencoded = pc.dictionary_encode(decoded).cast(value.type)
            dictionary = reencode_dictionary_descendants(reencoded.dictionary)
            return pa.DictionaryArray.from_arrays(reencoded.indices, dictionary, ordered=value.type.ordered)

        children = _arrow_array_children(value)
        if not children:
            return value
        normalized_children = tuple(reencode_dictionary_descendants(child) for child in children)
        if all(normalized is original for normalized, original in zip(normalized_children, children, strict=True)):
            return value
        return pa.Array.from_buffers(
            value.type,
            len(value),
            value.buffers()[: value.type.num_buffers],
            null_count=value.null_count,
            offset=value.offset,
            children=normalized_children,
        )

    return reencode_dictionary_descendants(column.combine_chunks())


def _snapshot_and_put_memory_source(
    source: Any,
    source_kind: str,
    expected_schema: Any,
    logical_type_tags: tuple[str, ...],
    column_indices: tuple[int, ...],
    include_row_count_column: bool,
) -> tuple[Any, list[Any]]:
    """Materialize selected columns into one canonical Ray-owned Arrow snapshot."""

    import pyarrow as pa
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray must be initialized before planning a Pandas or Arrow in-memory relation")

    if not isinstance(expected_schema, pa.Schema):
        raise TypeError(f"Expected a pyarrow.Schema for a Python memory source, got {type(expected_schema).__name__}")
    if len(logical_type_tags) != len(expected_schema):
        raise ValueError(
            f"Python memory source has {len(logical_type_tags)} bound types, expected {len(expected_schema)}"
        )
    row_count = _memory_source_row_count(source, source_kind) if include_row_count_column else None
    source = _select_memory_source_columns(source, source_kind, column_indices)
    source = _coerce_object_columns(source, source_kind, expected_schema, logical_type_tags)
    table = _as_arrow_table(source, source_kind)
    expected_names = expected_schema.names
    if row_count is not None:
        row_count_type = expected_schema.field(len(expected_names) - 1).type
        table = pa.Table.from_arrays(
            [*table.columns, pa.nulls(row_count, type=row_count_type)],
            names=[*table.column_names, expected_names[-1]],
        )
    if table.num_columns != len(expected_names):
        raise ValueError(
            f"Python memory source has {table.num_columns} columns after Arrow conversion, expected {len(expected_names)}"
        )
    if table.column_names != expected_names:
        table = table.rename_columns(expected_names)
    if source_kind in {"pandas", "numpy"}:
        table = _coerce_pandas_scan_table(table, expected_schema)

    if table.num_rows == 0:
        partition_ranges = [(0, 0)]
    else:
        average_row_bytes = max(1, (table.nbytes + table.num_rows - 1) // table.num_rows)
        rows_per_partition = max(1, min(_MAX_PARTITION_ROWS, _TARGET_PARTITION_BYTES // average_row_bytes))
        partition_ranges = [
            (offset, min(rows_per_partition, table.num_rows - offset))
            for offset in range(0, table.num_rows, rows_per_partition)
        ]

    object_refs = []
    for offset, row_count in partition_ranges:
        sliced = table.slice(offset, row_count)
        # A small table can still be a zero-copy view over much larger buffers.
        # Materializing at the Ray ownership boundary prevents every ObjectRef
        # from retaining or serializing bytes outside its partition.
        partition = pa.Table.from_arrays(
            [_materialize_partition_column(column) for column in sliced.columns], schema=table.schema
        )
        object_refs.append(ray.put(partition))
    return table.schema, object_refs


def _memory_source_tasks(
    source_id: str, object_refs: list[Any], column_indices: tuple[int, ...]
) -> list[_RayMemorySourceTask]:
    return [_RayMemorySourceTask(source_id, object_ref, column_indices) for object_ref in object_refs]


def _memory_source_schema(schema: Any, column_indices: tuple[int, ...]) -> Any:
    import pyarrow as pa

    return pa.schema([schema.field(column_index) for column_index in column_indices], metadata=schema.metadata)
