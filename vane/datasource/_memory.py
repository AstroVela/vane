# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Internal Arrow snapshot tasks for Python-owned Ray relation sources."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from vane.datasource import DataSourceTask

if TYPE_CHECKING:
    from vane._native import _DataSourceExecutionContext

_TARGET_PARTITION_BYTES = 16 * 1024 * 1024
_MAX_PARTITION_ROWS = 1_000_000


class _RayMemorySourceTask(DataSourceTask):
    """Resolve one query-owned Arrow table and expose it as record batches."""

    def __init__(self, source_id: str, partition_index: int, column_indices: tuple[int, ...]) -> None:
        self.source_id = str(source_id)
        self.partition_index = partition_index
        self.column_indices = tuple(column_indices)

    def execute(self) -> Iterator[Any]:
        import pyarrow as pa
        import ray

        from vane import ray_cxx

        partition = ray.get(ray_cxx._lookup_memory_source_ref(self.source_id, self.partition_index))
        if not isinstance(partition, pa.Table):
            raise TypeError(
                f"Ray memory source {self.source_id!r} resolved to {type(partition).__name__}, expected pyarrow.Table"
            )
        if self.column_indices != tuple(range(partition.num_columns)):
            partition = partition.select(self.column_indices)
        yield from partition.to_batches()

    def _execute_with_context(self, execution_context: _DataSourceExecutionContext) -> Iterator[Any]:
        execution_context._check_interrupted()
        for batch in self.execute():
            execution_context._check_interrupted()
            yield batch


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


def _prepare_arrow_memory_source(
    source: Any, column_indices: tuple[int, ...], names: list[str], include_row_count_column: bool
) -> Any:
    import pyarrow as pa

    if isinstance(source, pa.RecordBatch):
        source = pa.Table.from_batches([source])
    if not isinstance(source, pa.Table):
        raise TypeError(
            "Ray supports eager pyarrow.Table and pyarrow.RecordBatch memory sources. "
            "Materialize lazy or streaming Arrow sources explicitly before calling from_arrow()."
        )
    row_count = source.num_rows
    table = source.select(column_indices)
    if include_row_count_column:
        table = pa.Table.from_arrays([*table.columns, pa.nulls(row_count, type=pa.bool_())], names=names)
    elif table.column_names != names:
        table = table.rename_columns(names)
    return table


def _materialize_partition_column(column: Any) -> Any:
    import pyarrow as pa
    import pyarrow.compute as pc

    def trim_dictionary_descendants(value: Any) -> Any:
        if isinstance(value, pa.ExtensionArray):
            storage = value.storage
            normalized_storage = trim_dictionary_descendants(storage)
            if normalized_storage is storage:
                return value
            return pa.ExtensionArray.from_storage(value.type, normalized_storage)
        if pa.types.is_dictionary(value.type):
            # Work on integer indices so dictionary values can themselves be
            # nested or extension arrays. Keep their original dictionary order.
            used = pc.drop_null(pc.unique(value.indices))
            used = pc.take(used, pc.sort_indices(used))
            indices = pc.index_in(value.indices, value_set=used).cast(value.type.index_type)
            dictionary = trim_dictionary_descendants(pc.take(value.dictionary, used))
            return pa.DictionaryArray.from_arrays(indices, dictionary, ordered=value.type.ordered)

        children = _arrow_array_children(value)
        if not children:
            return value
        normalized_children = tuple(trim_dictionary_descendants(child) for child in children)
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

    # Each input chunk may have a different dictionary. Preserve chunk
    # boundaries instead of requiring Arrow to unify nested dictionaries or
    # fit their combined indices into the original (possibly int8) index type.
    return pa.chunked_array(
        [trim_dictionary_descendants(pa.chunked_array([chunk]).combine_chunks()) for chunk in column.chunks],
        type=column.type,
    )


def _snapshot_and_put_memory_source(table: Any) -> tuple[Any, list[Any]]:
    """Copy selected Arrow columns into query-owned Ray partitions."""

    import pyarrow as pa
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray must be initialized before planning a Pandas or Arrow in-memory relation")
    if not isinstance(table, pa.Table):
        raise TypeError(f"Expected a pyarrow.Table memory snapshot, got {type(table).__name__}")

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
    source_id: str, partition_count: int, column_indices: tuple[int, ...]
) -> list[_RayMemorySourceTask]:
    return [_RayMemorySourceTask(source_id, index, column_indices) for index in range(partition_count)]


def _memory_source_schema(schema: Any, column_indices: tuple[int, ...]) -> Any:
    import pyarrow as pa

    return pa.schema([schema.field(column_index) for column_index in column_indices], metadata=schema.metadata)
