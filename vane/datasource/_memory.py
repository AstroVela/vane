# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Internal Arrow snapshot tasks for Python-owned Ray relation sources."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

_TARGET_PARTITION_BYTES = 16 * 1024 * 1024
_MAX_PARTITION_ROWS = 1_000_000


class _ArrowStreamCapsule:
    """Adapt a bare Arrow C stream capsule to the PyCapsule protocol."""

    def __init__(self, capsule: Any) -> None:
        self._capsule = capsule

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        del requested_schema
        return self._capsule


class _RayMemorySourceTask:
    """Resolve one query-owned Arrow table and expose it as record batches."""

    def __init__(self, source_id: str, object_ref: Any) -> None:
        self.source_id = str(source_id)
        self.object_ref = object_ref

    def execute(self) -> Iterator[Any]:
        import pyarrow as pa
        import ray

        partition = ray.get(self.object_ref)
        if not isinstance(partition, pa.Table):
            raise TypeError(
                f"Ray memory source {self.source_id!r} resolved to {type(partition).__name__}, expected pyarrow.Table"
            )
        yield from partition.to_batches()


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
    if isinstance(source, pa.RecordBatchReader):
        return source.read_all()

    to_table = getattr(source, "to_table", None)
    if callable(to_table):
        table = to_table()
        if isinstance(table, pa.Table):
            return table

    to_arrow = getattr(source, "to_arrow", None)
    if callable(to_arrow):
        table = to_arrow()
        if isinstance(table, pa.Table):
            return table

    collect = getattr(source, "collect", None)
    if callable(collect):
        collected = collect()
        to_arrow = getattr(collected, "to_arrow", None)
        if callable(to_arrow):
            table = to_arrow()
            if isinstance(table, pa.Table):
                return table

    if type(source).__name__ == "PyCapsule":
        source = _ArrowStreamCapsule(source)
    table = pa.table(source)
    if not isinstance(table, pa.Table):  # pragma: no cover - defensive PyArrow contract check
        raise TypeError(f"Arrow source converted to {type(table).__name__}, expected pyarrow.Table")
    return table


def _snapshot_and_put_memory_source(
    source: Any, source_kind: str, source_id: str, expected_schema: Any
) -> tuple[Any, list[Any], list[_RayMemorySourceTask]]:
    """Materialize one canonical Arrow snapshot and put it in Ray exactly once."""

    import pyarrow as pa
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray must be initialized before planning a Pandas or Arrow in-memory relation")

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
        partitions = [table]
    else:
        average_row_bytes = max(1, (table.nbytes + table.num_rows - 1) // table.num_rows)
        rows_per_partition = max(1, min(_MAX_PARTITION_ROWS, _TARGET_PARTITION_BYTES // average_row_bytes))
        partitions = [
            table.slice(offset, min(rows_per_partition, table.num_rows - offset))
            for offset in range(0, table.num_rows, rows_per_partition)
        ]

    object_refs = [ray.put(partition) for partition in partitions]
    tasks = [_RayMemorySourceTask(source_id, object_ref) for object_ref in object_refs]
    return table.schema, object_refs, tasks
