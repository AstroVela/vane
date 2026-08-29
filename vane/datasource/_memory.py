# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Internal Arrow snapshot tasks for Python-owned Ray relation sources."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

_TARGET_PARTITION_BYTES = 16 * 1024 * 1024
_MAX_PARTITION_ROWS = 1_000_000


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


def _append_arrow_array_version(version: bytearray, value: Any) -> None:
    import pyarrow as pa

    _append_version_bytes(version, str(value.type).encode())
    _append_version_integer(version, len(value))
    _append_version_integer(version, value.offset)
    buffers = value.buffers()
    _append_version_integer(version, len(buffers))
    for buffer in buffers:
        _append_version_integer(version, buffer is not None)
        if buffer is not None:
            _append_version_integer(version, buffer.address)
            _append_version_integer(version, buffer.size)
    if pa.types.is_dictionary(value.type):
        _append_arrow_array_version(version, value.dictionary)


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
    if isinstance(source, (pa.Table, pa.RecordBatch)):
        return source.select(column_indices)
    return source


def _materialize_partition_column(column: Any) -> Any:
    import pyarrow as pa
    import pyarrow.compute as pc

    if pa.types.is_dictionary(column.type):
        decoded = pc.dictionary_decode(column)
        column = pc.dictionary_encode(decoded).cast(column.type)
    return column.combine_chunks()


def _snapshot_and_put_memory_source(
    source: Any, source_kind: str, expected_schema: Any, column_indices: tuple[int, ...]
) -> tuple[Any, list[Any]]:
    """Materialize selected columns into one canonical Ray-owned Arrow snapshot."""

    import pyarrow as pa
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray must be initialized before planning a Pandas or Arrow in-memory relation")

    source = _select_memory_source_columns(source, source_kind, column_indices)
    table = _as_arrow_table(source, source_kind)
    if not isinstance(expected_schema, pa.Schema):
        raise TypeError(f"Expected a pyarrow.Schema for a Python memory source, got {type(expected_schema).__name__}")
    expected_names = expected_schema.names
    if table.num_columns != len(expected_names):
        raise ValueError(
            f"Python memory source has {table.num_columns} columns after Arrow conversion, expected {len(expected_names)}"
        )
    if table.column_names != expected_names:
        table = table.rename_columns(expected_names)
    if source_kind == "pandas":
        table = table.cast(expected_schema, safe=True)

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
