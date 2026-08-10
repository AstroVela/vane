# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import itertools
import os
import posixpath
import uuid
from collections.abc import Iterable
from typing import Any

import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]

from vane.execution._common import ensure_table
from vane.execution.udf_output_schema import _arrow_type_from_name

_SINK_FLAG = "terminal_arrow_parquet_sink"
_OUTPUT_DIRECTORY = "terminal_arrow_parquet_output_directory"
_FILE_EXTENSION = "terminal_arrow_parquet_file_extension"
_WRITER_OPTIONS = "terminal_arrow_parquet_writer_options"
_EXPECTED_NAMES = "terminal_arrow_parquet_expected_names"
_EXPECTED_TYPES = "terminal_arrow_parquet_expected_types"
_WRITE_EMPTY_FILE = "terminal_arrow_parquet_write_empty_file"
_SINK_PAYLOAD_KEYS = (
    _SINK_FLAG,
    _OUTPUT_DIRECTORY,
    _FILE_EXTENSION,
    _WRITER_OPTIONS,
    _EXPECTED_NAMES,
    _EXPECTED_TYPES,
    _WRITE_EMPTY_FILE,
)

_COLUMN_STATISTICS_TYPE = pa.map_(pa.string(), pa.map_(pa.string(), pa.string()))
_PARTITION_KEYS_TYPE = pa.map_(pa.string(), pa.string())
_COPY_STATISTICS_SCHEMA = pa.schema(
    [
        pa.field("filename", pa.string()),
        pa.field("count", pa.uint64()),
        pa.field("file_size_bytes", pa.uint64()),
        pa.field("footer_size_bytes", pa.uint64()),
        pa.field("column_statistics", _COLUMN_STATISTICS_TYPE),
        pa.field("partition_keys", _PARTITION_KEYS_TYPE),
    ]
)


def terminal_arrow_parquet_sink_enabled(payload: dict[str, Any] | None) -> bool:
    return bool((payload or {}).get(_SINK_FLAG, False))


def terminal_arrow_parquet_sink_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not terminal_arrow_parquet_sink_enabled(payload):
        return None
    assert payload is not None
    return {key: payload[key] for key in _SINK_PAYLOAD_KEYS if key in payload}


def _expected_schema(payload: dict[str, Any]) -> pa.Schema:
    names = [str(value) for value in payload.get(_EXPECTED_NAMES) or []]
    type_names = [str(value) for value in payload.get(_EXPECTED_TYPES) or []]
    if not names or len(names) != len(type_names):
        raise ValueError("terminal Arrow Parquet sink requires matching expected names and types")
    return pa.schema([pa.field(name, _arrow_type_from_name(type_name)) for name, type_name in zip(names, type_names)])


def _row_preserving_output(payload: dict[str, Any]) -> bool:
    call_mode = str(payload.get("call_mode") or "")
    return call_mode == "map_batches_rows" or (call_mode == "map" and payload.get("scalar_arg_count") is not None)


def _normalize_table(table: pa.Table, schema: pa.Schema, *, rename_columns: bool) -> pa.Table:
    result = ensure_table(table)
    if result.schema.names != schema.names:
        if not rename_columns or result.num_columns != len(schema):
            raise ValueError(
                "terminal Arrow Parquet sink output columns do not match COPY schema: "
                f"expected={schema.names!r} got={result.schema.names!r}"
            )
        result = result.rename_columns(schema.names)
    if result.schema.types != schema.types:
        result = result.cast(schema, safe=True)
    return result


def _output_location(output_directory: str, file_extension: str, invocation_id: str) -> tuple[Any, str, str]:
    import pyarrow.fs as pafs  # type: ignore[import-not-found, import-untyped, unused-ignore]

    directory = str(output_directory or "").strip()
    if not directory:
        raise ValueError("terminal Arrow Parquet sink requires an output directory")
    extension = str(file_extension or "")
    if not extension or "/" in extension or "\\" in extension:
        raise ValueError("terminal Arrow Parquet sink requires a path-safe file extension")
    identity = hashlib.sha256(str(invocation_id).encode("utf-8")).hexdigest()[:16]
    file_name = f"part-{identity}-{uuid.uuid4().hex}.{extension}"

    if "://" in directory or directory.startswith("file:"):
        filesystem, filesystem_directory = pafs.FileSystem.from_uri(directory)
        file_path = posixpath.join(filesystem_directory.rstrip("/"), file_name)
        visible_path = directory.rstrip("/") + "/" + file_name
    else:
        filesystem = pafs.LocalFileSystem()
        filesystem_directory = directory
        file_path = os.path.join(filesystem_directory, file_name)
        visible_path = file_path

    filesystem.create_dir(filesystem_directory, recursive=True)
    return filesystem, file_path, visible_path


def _empty_copy_statistics() -> pa.Table:
    return pa.Table.from_arrays(
        [pa.array([], type=field.type) for field in _COPY_STATISTICS_SCHEMA],
        schema=_COPY_STATISTICS_SCHEMA,
    )


def _copy_statistics(
    *,
    filename: str,
    row_count: int,
    file_size_bytes: int,
    footer_size_bytes: int,
) -> pa.Table:
    return pa.Table.from_arrays(
        [
            pa.array([filename], type=pa.string()),
            pa.array([row_count], type=pa.uint64()),
            pa.array([file_size_bytes], type=pa.uint64()),
            pa.array([footer_size_bytes], type=pa.uint64()),
            pa.array([[]], type=_COLUMN_STATISTICS_TYPE),
            pa.array([[]], type=_PARTITION_KEYS_TYPE),
        ],
        schema=_COPY_STATISTICS_SCHEMA,
    )


def _iter_row_group_tables(tables: Iterable[pa.Table], row_group_size: int) -> Iterable[pa.Table]:
    pending: list[pa.Table] = []
    pending_rows = 0
    for table in tables:
        offset = 0
        while offset < table.num_rows:
            remaining = table.num_rows - offset
            if pending_rows == 0 and remaining >= row_group_size:
                yield table.slice(offset, row_group_size)
                offset += row_group_size
                continue

            take_rows = min(row_group_size - pending_rows, remaining)
            pending.append(table.slice(offset, take_rows))
            pending_rows += take_rows
            offset += take_rows
            if pending_rows == row_group_size:
                yield pending[0] if len(pending) == 1 else pa.concat_tables(pending)
                pending = []
                pending_rows = 0

    if pending:
        yield pending[0] if len(pending) == 1 else pa.concat_tables(pending)


def write_terminal_arrow_parquet_output(
    payload: dict[str, Any],
    tables: Iterable[pa.Table],
    *,
    invocation_id: str,
) -> pa.Table:
    import pyarrow.fs as pafs  # type: ignore[import-not-found, import-untyped, unused-ignore]
    import pyarrow.parquet as pq  # type: ignore[import-not-found, import-untyped, unused-ignore]

    if not terminal_arrow_parquet_sink_enabled(payload):
        raise ValueError("terminal Arrow Parquet sink is not enabled")

    schema = _expected_schema(payload)
    rename_columns = _row_preserving_output(payload)
    normalized_tables: Iterable[pa.Table] = (
        _normalize_table(table, schema, rename_columns=rename_columns) for table in tables
    )
    write_empty_file = bool(payload.get(_WRITE_EMPTY_FILE, True))
    if not write_empty_file:
        first_non_empty = next((table for table in normalized_tables if table.num_rows > 0), None)
        if first_non_empty is None:
            return _empty_copy_statistics()
        normalized_tables = itertools.chain((first_non_empty,), normalized_tables)

    raw_options = payload.get(_WRITER_OPTIONS) or {}
    if not isinstance(raw_options, dict):
        raise TypeError("terminal Arrow Parquet writer options must be a dict")
    writer_options = dict(raw_options)
    row_group_size = int(writer_options.pop("row_group_size", 0) or 0)
    if row_group_size <= 0:
        raise ValueError("terminal Arrow Parquet row_group_size must be positive")

    filesystem, file_path, visible_path = _output_location(
        str(payload.get(_OUTPUT_DIRECTORY) or ""),
        str(payload.get(_FILE_EXTENSION) or ""),
        invocation_id,
    )
    row_count = 0
    with filesystem.open_output_stream(file_path) as output_stream:
        with pq.ParquetWriter(output_stream, schema, **writer_options) as writer:
            for table in _iter_row_group_tables(normalized_tables, row_group_size):
                row_count += table.num_rows
                writer.write_table(table, row_group_size=row_group_size)

    file_info = filesystem.get_file_info(file_path)
    if file_info.type != pafs.FileType.File:
        raise RuntimeError(f"terminal Arrow Parquet output is not a file: {visible_path}")
    metadata = pq.ParquetFile(file_path, filesystem=filesystem).metadata
    return _copy_statistics(
        filename=visible_path,
        row_count=row_count,
        file_size_bytes=int(file_info.size),
        footer_size_bytes=int(metadata.serialized_size),
    )


__all__ = [
    "terminal_arrow_parquet_sink_enabled",
    "terminal_arrow_parquet_sink_payload",
    "write_terminal_arrow_parquet_output",
]
