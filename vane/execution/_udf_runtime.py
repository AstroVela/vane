# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""In-process UDF runtime used by Ray and subprocess workers.

Executes Python UDFs directly in the current process.
Supports ``map``, ``flat_map``, ``map_batches``, and ``map_batches_rows`` modes.
"""

from __future__ import annotations

import inspect
import os
from collections import deque
from collections.abc import Iterable
from typing import Any

import pyarrow as pa

from vane import PythonExceptionHandling
from vane._duckdb_func import FunctionNullHandling
from vane.execution._common import (
    ensure_table as _ensure_table,
)
from vane.execution._common import (
    estimate_table_bytes as _estimate_table_bytes,
)
from vane.execution._common import (
    iter_table_batches as _iter_table_batches,
)
from vane.execution._common import (
    load_udf_from_payload,
    load_udf_from_payload_cached,
)
from vane.execution.udf_output_schema import empty_output_table_from_payload as _empty_output_table_from_payload
from vane.execution.udf_ray_config import stream_output_enabled as _stream_output_enabled

# Mirrors DuckDB's STANDARD_VECTOR_SIZE default batch size.
BATCH_SIZE = 2048
DEFAULT_TARGET_MAX_BATCH_BYTES = 128 * 1024 * 1024
TARGET_MAX_BATCH_BYTES_ENV = "VANE_UDF_TARGET_MAX_BATCH_BYTES"


def _load_runtime_callable(
    payload: dict[str, Any],
    *,
    cache_callable: bool = False,
    cache_max_entries: int | None = None,
) -> Any:
    if cache_callable:
        udf = load_udf_from_payload_cached(payload, max_entries=cache_max_entries)
    else:
        udf = load_udf_from_payload(payload)

    backend = str(payload.get("execution_backend") or "").strip().lower()
    is_actor_backend = backend in ("subprocess_actor", "ray_actor")
    is_task_backend = backend in ("subprocess_task", "ray_task")

    if payload.get("stateful"):
        actor_number = payload.get("actor_number")
        if not is_actor_backend:
            raise ValueError("stateful expression UDFs require an actor execution backend")
        if type(actor_number) is not int or actor_number != 1:
            raise ValueError(
                "actor_number must be exactly 1 for stateful vane.cls UDFs; multi-actor state semantics are not defined"
            )

    if is_task_backend:
        if inspect.isclass(udf):
            raise ValueError("task UDF backends require a function, not a callable class")
        if not (inspect.isfunction(udf) or inspect.ismethod(udf)):
            raise ValueError("task UDF backends require a function")
        return udf

    if not is_actor_backend:
        return udf

    if not inspect.isclass(udf):
        raise ValueError("actor UDF backends require a callable class")
    if _has_required_constructor_args(udf):
        raise TypeError(
            "callable class UDF constructors must be zero-argument; "
            "use environment variables or class-level defaults when configuration is required"
        )
    return udf()


def _has_required_constructor_args(cls: type) -> bool:
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"cannot inspect callable class constructor for {cls!r}") from exc
    for parameter in signature.parameters.values():
        if parameter.default is not inspect.Parameter.empty:
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        return True
    return False


# ── Output normalization helpers ─────────────────────────────────────────────


# ── Scalar output helpers ────────────────────────────────────────────────────

_NULL_HANDLING_ERROR = """\
The returned result contained NULL values, but the 'null_handling' was set to DEFAULT.
If you want more control over NULL values then 'null_handling' should be set to SPECIAL.

With DEFAULT all rows containing NULL have been filtered from the UDFs input.
Those rows are automatically set to NULL in the final result.
The UDF is not expected to return NULL values."""


_DEFAULT_NULL_HANDLING = int(FunctionNullHandling.DEFAULT)
_RETURN_NULL = int(PythonExceptionHandling.RETURN_NULL)


def _coerce_scalar_array(output: Any, expected_rows: int) -> pa.Array:
    if isinstance(output, pa.Table):
        table = output
    elif isinstance(output, pa.RecordBatch):
        table = pa.Table.from_batches([output])
    elif isinstance(output, pa.RecordBatchReader):
        table = pa.Table.from_batches(list(output))
    else:
        table = pa.Table.from_arrays([output], names=["_udf_out"])

    if table.num_columns != 1:
        raise ValueError(f"map output must have exactly 1 column, got {table.num_columns}")

    column = table.column(0)
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    if len(column) != expected_rows:
        raise ValueError(f"map output row count {len(column)} does not match {expected_rows}")
    return column


def _build_valid_mask(table: pa.Table) -> list[bool]:
    row_count = table.num_rows
    if row_count == 0:
        return []
    mask = [True] * row_count
    for column in table.columns:
        values = column.to_pylist()
        for idx, value in enumerate(values):
            if value is None:
                mask[idx] = False
    return mask


# ── Stream output utilities ──────────────────────────────────────────────────


def _positive_int_or_none(value: Any, label: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return parsed


def _positive_int_from_env_or_none(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return _positive_int_or_none(raw, name)


def _effective_target_max_batch_bytes(payload: dict[str, Any]) -> int:
    parsed = _positive_int_or_none(payload.get("udf_target_max_batch_bytes"), "udf_target_max_batch_bytes")
    if parsed is not None:
        return parsed
    parsed = _positive_int_from_env_or_none(TARGET_MAX_BATCH_BYTES_ENV)
    if parsed is not None:
        return parsed
    return DEFAULT_TARGET_MAX_BATCH_BYTES


def _effective_output_target_max_bytes(payload: dict[str, Any]) -> int:
    parsed = _positive_int_or_none(payload.get("udf_output_target_max_bytes"), "udf_output_target_max_bytes")
    if parsed is not None:
        return parsed
    return _effective_target_max_batch_bytes(payload)


def _table_nbytes(table: pa.Table) -> int:
    return _estimate_table_bytes(table)


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (numerator + denominator - 1) // denominator


def _estimate_python_value_bytes(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, bool):
        return 1
    if isinstance(value, (int, float)):
        return 8
    if isinstance(value, dict):
        return sum(
            _estimate_python_value_bytes(key) + _estimate_python_value_bytes(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(_estimate_python_value_bytes(item) for item in value)
    return 64


def _estimate_python_row_bytes(row: dict[str, Any]) -> int:
    return max(1, sum(_estimate_python_value_bytes(value) for value in row.values()))


def _effective_batch_size(payload: dict[str, Any]) -> int | None:
    parsed = _positive_int_or_none(payload.get("batch_size"), "batch_size")
    if parsed is not None:
        return parsed
    if str(payload.get("call_mode") or "") == "map_batches":
        return None
    return BATCH_SIZE


def _effective_output_batch_size(payload: dict[str, Any]) -> int | None:
    parsed = _positive_int_or_none(payload.get("output_batch_size"), "output_batch_size")
    if parsed is not None:
        return parsed
    return None


def _iter_output_tables(result: Any) -> Iterable[pa.Table]:
    """Iterate over output batches from a stream UDF result."""
    if result is None:
        return

    if isinstance(result, pa.Table):
        yield result
        return
    if isinstance(result, pa.RecordBatch):
        yield pa.Table.from_batches([result])
        return
    if isinstance(result, pa.RecordBatchReader):
        for rb in result:
            yield pa.Table.from_batches([rb])
        return
    if isinstance(result, dict):
        yield pa.table(result)
        return

    if isinstance(result, Iterable) and not isinstance(result, (str, bytes, bytearray)):
        for item in result:
            if item is None:
                continue
            yield from _iter_output_tables(item)
        return

    raise TypeError("udf output must be Table/RecordBatch/RecordBatchReader/dict, or an iterable yielding those types")


def _iter_table_row_dicts(table: pa.Table) -> Iterable[dict[str, Any]]:
    names = table.schema.names
    columns = table.columns
    for row_idx in range(table.num_rows):
        yield {name: column[row_idx].as_py() for name, column in zip(names, columns, strict=True)}


class RuntimeOutputBuffer:
    def __init__(self, target_rows: int | None, target_bytes: int | None = None) -> None:
        if target_rows is not None:
            target_rows = int(target_rows)
            if target_rows <= 0:
                raise ValueError("output batch size must be a positive integer")
        target_bytes = _positive_int_or_none(target_bytes, "output target bytes")
        if target_rows is None and target_bytes is None:
            raise ValueError("output buffer requires row or byte target")
        self._target_rows = target_rows
        self._target_bytes = target_bytes
        self._tables: deque[pa.Table] = deque()
        self._row_count = 0
        self._byte_count = 0
        self._empty_table: pa.Table | None = None
        self._saw_non_empty = False

    def append(self, table: pa.Table) -> Iterable[pa.Table]:
        table = _ensure_table(table)
        if table.num_rows == 0:
            if not self._saw_non_empty and self._empty_table is None:
                self._empty_table = table
            return

        self._saw_non_empty = True
        self._empty_table = None
        self._tables.append(table)
        self._row_count += table.num_rows
        self._byte_count += _table_nbytes(table)
        while self._should_flush():
            yield self._take(self._take_rows_for_limits())

    def flush(self) -> Iterable[pa.Table]:
        while self._row_count > 0:
            if self._target_bytes is not None and self._byte_count > self._target_bytes:
                yield self._take(self._take_rows_for_limits())
            elif self._target_rows is not None and self._row_count > self._target_rows:
                yield self._take(self._target_rows)
            else:
                yield self._take(self._row_count)
                return
        if not self._saw_non_empty and self._empty_table is not None:
            yield self._empty_table
            self._empty_table = None

    def _should_flush(self) -> bool:
        if self._target_rows is not None and self._row_count >= self._target_rows:
            return True
        return self._target_bytes is not None and self._byte_count >= self._target_bytes

    def _take_rows_for_limits(self) -> int:
        max_rows = self._row_count if self._target_rows is None else min(self._target_rows, self._row_count)
        if self._target_bytes is None:
            return max_rows

        remaining_bytes = self._target_bytes
        selected_rows = 0
        for table in self._tables:
            if selected_rows >= max_rows:
                break
            candidate_rows = min(table.num_rows, max_rows - selected_rows)
            if candidate_rows <= 0:
                continue
            candidate = table if candidate_rows == table.num_rows else table.slice(0, candidate_rows)
            candidate_bytes = _table_nbytes(candidate)
            if candidate_bytes <= remaining_bytes:
                selected_rows += candidate_rows
                remaining_bytes -= candidate_bytes
                continue

            bytes_per_row = max(1, _ceil_div(candidate_bytes, candidate_rows))
            rows_by_bytes = remaining_bytes // bytes_per_row
            if selected_rows == 0:
                selected_rows = max(1, min(candidate_rows, rows_by_bytes))
            else:
                selected_rows += min(candidate_rows, rows_by_bytes)
            break

        return max(1, min(max_rows, selected_rows))

    def _take(self, rows: int) -> pa.Table:
        pieces: list[pa.Table] = []
        remaining = rows
        while remaining > 0:
            table = self._tables[0]
            if table.num_rows <= remaining:
                pieces.append(table)
                self._tables.popleft()
                self._row_count -= table.num_rows
                remaining -= table.num_rows
                continue

            pieces.append(table.slice(0, remaining))
            self._tables[0] = table.slice(remaining)
            self._row_count -= remaining
            remaining = 0

        self._byte_count = sum(_table_nbytes(table) for table in self._tables)
        if len(pieces) == 1:
            return pieces[0]
        return pa.concat_tables(pieces, promote_options="default")


class RuntimeInputBatcher:
    def __init__(self, target_rows: int) -> None:
        target_rows = int(target_rows)
        if target_rows <= 0:
            raise ValueError("input batch size must be a positive integer")
        self._target_rows = target_rows
        self._tables: deque[pa.Table] = deque()
        self._row_count = 0

    def append(self, table: pa.Table) -> Iterable[pa.Table]:
        table = _ensure_table(table)
        if table.num_rows == 0:
            return

        self._tables.append(table)
        self._row_count += table.num_rows
        while self._row_count >= self._target_rows:
            yield self._take(self._target_rows)

    def flush(self) -> Iterable[pa.Table]:
        if self._row_count > 0:
            yield self._take(self._row_count)

    def _take(self, rows: int) -> pa.Table:
        pieces: list[pa.Table] = []
        remaining = rows
        while remaining > 0:
            table = self._tables[0]
            if table.num_rows <= remaining:
                pieces.append(table)
                self._tables.popleft()
                self._row_count -= table.num_rows
                remaining -= table.num_rows
                continue

            pieces.append(table.slice(0, remaining))
            self._tables[0] = table.slice(remaining)
            self._row_count -= remaining
            remaining = 0

        if len(pieces) == 1:
            return pieces[0]
        return pa.concat_tables(pieces, promote_options="default")


# ── Executor ─────────────────────────────────────────────────────────────────


class UDFExecutor:
    """Execute a Python UDF locally according to payload.call_mode.

    Supported call modes:
    - ``map_batches``: ``fn(pa.Table) → pa.Table | Iterator[pa.Table]``
    - ``map_batches_rows``: ``fn(pa.Table) → pa.Table`` with one output row per input row.
    - ``flat_map``:    ``fn(dict) → dict | Iterator[dict]``
    - ``map``:
      ``native`` row calls or ``arrow`` vectorized column calls.
    """

    def __init__(
        self,
        payload: dict[str, Any],
        cache_callable: bool = False,
        cache_max_entries: int | None = None,
    ) -> None:
        if payload is None:
            raise ValueError("UDF payload is required")
        self._queue: deque[pa.Table] = deque()
        self._finished_submitting = False
        self._call_mode = str(payload.get("call_mode") or "")
        if self._call_mode not in ("map_batches", "map_batches_rows", "flat_map", "map"):
            raise ValueError("UDF payload.call_mode must be one of: map_batches, map_batches_rows, flat_map, map")

        if self._call_mode == "map":
            self._init_scalar(payload, cache_callable=cache_callable, cache_max_entries=cache_max_entries)
            return

        self._payload = payload
        self._stream_output = _stream_output_enabled(payload)

        # Output config
        self._output_names: list[str] | None = None
        output_schema = payload.get("output_schema")
        if output_schema:
            self._output_names = [str(entry.get("name") or "") for entry in output_schema]

        # Input names for renaming args columns
        self._input_names: list[str] | None = None
        raw_input_names = payload.get("input_names")
        if raw_input_names:
            self._input_names = list(raw_input_names)

        self._map_fn = _load_runtime_callable(payload)
        self._mode = self._call_mode
        self._is_map_batches = self._call_mode == "map_batches"
        self._is_map_batches_rows = self._call_mode == "map_batches_rows"
        self._is_flat_map = self._call_mode == "flat_map"
        self._is_map = False

        # Batch size for splitting input
        self._batch_size = _effective_batch_size(payload)
        self._output_batch_size = _effective_output_batch_size(payload)
        self._output_target_max_bytes = _effective_output_target_max_bytes(payload)
        preserve_compute_boundaries = payload.get("preserve_compute_batch_boundaries", False)
        if not isinstance(preserve_compute_boundaries, bool):
            raise ValueError("UDF payload field 'preserve_compute_batch_boundaries' must be boolean")
        self._preserve_compute_batch_boundaries = preserve_compute_boundaries
        # When C++ pre-batches input, skip Python re-batching.
        self._prebatched_input = bool(payload.get("prebatched_input", False))
        self._execution_backend = str(payload.get("execution_backend") or "").strip().lower()
        can_flush_compute_tail = self._execution_backend in ("ray_task", "subprocess_task")
        self._input_batcher = (
            RuntimeInputBatcher(self._batch_size)
            if self._is_map_batches
            and can_flush_compute_tail
            and not self._prebatched_input
            and self._batch_size is not None
            and self._batch_size > BATCH_SIZE
            else None
        )

    def _init_scalar(
        self,
        payload: dict[str, Any],
        *,
        cache_callable: bool,
        cache_max_entries: int | None,
    ) -> None:
        self._mode = "map"
        self._is_map_batches = False
        self._is_map_batches_rows = False
        self._is_flat_map = False
        self._is_map = True
        self._scalar_udf_type = str(payload.get("scalar_udf_type") or "native")
        if self._scalar_udf_type not in ("native", "arrow"):
            raise ValueError("scalar_udf_type must be one of: native, arrow")
        self._scalar_output_name = "value"
        self._null_handling = int(payload.get("null_handling") or 0)
        self._exception_handling = int(payload.get("exception_handling") or 0)
        self._batch_size = payload.get("batch_size")
        if self._batch_size is not None:
            self._batch_size = int(self._batch_size)
        self._map_fn = _load_runtime_callable(
            payload,
            cache_callable=cache_callable,
            cache_max_entries=cache_max_entries,
        )

    def warm_up(self) -> None:
        """Run an optional UDF-level warmup hook after deserialization."""
        warm_up = getattr(self._map_fn, "warm_up", None)
        if callable(warm_up):
            warm_up()

    def _rename_args(self, args: pa.Table) -> pa.Table:
        if self._input_names:
            if len(self._input_names) != args.num_columns:
                raise ValueError(
                    f"UDF input_names count {len(self._input_names)} does not match input column count {args.num_columns}"
                )
            args = args.rename_columns(self._input_names)
        return args

    def _iter_map_batches_output_tables(self, result: Any) -> Iterable[pa.Table]:
        if result is None:
            return

        try:
            tables = _iter_output_tables(result)
            for table in tables:
                if table is not None:
                    yield table
            return
        except TypeError as exc:
            raise TypeError(f"map_batches UDF must return pa.Table or Iterator[pa.Table], got {type(result)}") from exc

    def _coerce_row_preserving_batch_output(self, result: Any, expected_rows: int) -> pa.Table:
        if isinstance(result, pa.Table):
            table = result
        elif isinstance(result, pa.RecordBatch):
            table = pa.Table.from_batches([result])
        elif isinstance(result, dict):
            table = pa.table(result)
        elif isinstance(result, Iterable) and not isinstance(result, (str, bytes, bytearray)):
            raise TypeError("row-preserving map_batches UDF must return a single pa.Table, not an iterator")
        else:
            raise TypeError(f"row-preserving map_batches UDF must return pa.Table, got {type(result)}")

        if table.num_rows != expected_rows:
            raise ValueError(
                f"row-preserving map_batches output row count {table.num_rows} does not match input rows {expected_rows}"
            )
        if self._output_names and len(self._output_names) == 1 and table.num_columns != 1:
            raise ValueError(f"row-preserving map_batches output must have exactly 1 column, got {table.num_columns}")
        return table

    def _iter_map_batches_compute_batches(self, args: pa.Table) -> Iterable[pa.Table]:
        if self._prebatched_input:
            yield args
            return
        if self._input_batcher is not None:
            yield from self._input_batcher.append(args)
            return
        yield from _iter_table_batches(args, self._batch_size)

    def _flush_map_batches_compute_batches(self) -> Iterable[pa.Table]:
        if self._input_batcher is not None:
            yield from self._input_batcher.flush()

    def _execute_map_batches_compute_batches(self, batches: Iterable[pa.Table]) -> None:
        results: list[pa.Table] = []
        saw_compute_batch = False
        saw_output = False
        shared_output_buffer = (
            RuntimeOutputBuffer(self._output_batch_size, self._output_target_max_bytes) if self._stream_output else None
        )
        for batch in batches:
            saw_compute_batch = True
            result = self._map_fn(batch)
            if self._is_map_batches_rows:
                results.append(self._coerce_row_preserving_batch_output(result, batch.num_rows))
                continue
            output_tables = self._iter_map_batches_output_tables(result)
            if self._stream_output:
                output_buffer = (
                    RuntimeOutputBuffer(self._output_batch_size, self._output_target_max_bytes)
                    if self._preserve_compute_batch_boundaries
                    else shared_output_buffer
                )
                assert output_buffer is not None
                for table in output_tables:
                    saw_output = True
                    for output in output_buffer.append(table):
                        self._queue.append(output)
                if self._preserve_compute_batch_boundaries:
                    for output in output_buffer.flush():
                        self._queue.append(output)
            else:
                batch_tables = list(output_tables)
                saw_output = saw_output or bool(batch_tables)
                results.extend(batch_tables)
        if shared_output_buffer is not None and not self._preserve_compute_batch_boundaries:
            for output in shared_output_buffer.flush():
                self._queue.append(output)
        if results:
            if len(results) == 1:
                self._queue.append(results[0])
            else:
                self._queue.append(pa.concat_tables(results, promote_options="default"))
        elif saw_compute_batch and not saw_output:
            self._queue.append(_empty_output_table_from_payload(self._payload))

    def _execute_map_batches(self, args: pa.Table) -> None:
        args = self._rename_args(args)
        self._execute_map_batches_compute_batches(self._iter_map_batches_compute_batches(args))

    def iter_submit(self, args: pa.Table) -> Iterable[pa.Table]:
        """Submit one input table and yield output tables as they are produced.

        This is used by Ray streaming-generator actors for block-producing
        table UDFs. The normal submit()/take_ready_result() path remains queue based and
        only returns after the callable finishes.
        """
        args = _ensure_table(args)
        if args.num_rows == 0:
            return

        if self._is_map_batches and self._stream_output:
            args = self._rename_args(args)
            batches = self._iter_map_batches_compute_batches(args)
            saw_compute_batch = False
            saw_output = False
            shared_output_buffer = RuntimeOutputBuffer(self._output_batch_size, self._output_target_max_bytes)
            for batch in batches:
                saw_compute_batch = True
                result = self._map_fn(batch)
                output_buffer = (
                    RuntimeOutputBuffer(self._output_batch_size, self._output_target_max_bytes)
                    if self._preserve_compute_batch_boundaries
                    else shared_output_buffer
                )
                for table in self._iter_map_batches_output_tables(result):
                    if table is not None:
                        saw_output = True
                        yield from output_buffer.append(table)
                if self._preserve_compute_batch_boundaries:
                    yield from output_buffer.flush()
            if not self._preserve_compute_batch_boundaries:
                yield from shared_output_buffer.flush()
            if saw_compute_batch and not saw_output:
                yield _empty_output_table_from_payload(self._payload)
            return

        if self._is_flat_map and self._stream_output:
            emitted = False
            for table in self._iter_flat_map_output_tables(args):
                emitted = True
                yield table
            if not emitted:
                yield _empty_output_table_from_payload(self._payload)
            return

        self.submit(args)
        for table in self.drain_outputs():
            yield table

    def _iter_flat_map_output_tables(self, args: pa.Table) -> Iterable[pa.Table]:
        args = self._rename_args(args)
        output_rows: list[dict[str, Any]] = []
        output_row_bytes = 0
        output_buffer = RuntimeOutputBuffer(self._output_batch_size, self._output_target_max_bytes)
        batches = _iter_table_batches(args, self._batch_size) if not self._prebatched_input else [args]

        def flush_output_rows() -> Iterable[pa.Table]:
            nonlocal output_rows, output_row_bytes
            if not output_rows:
                return
            table = self._flat_map_rows_to_table(output_rows)
            output_rows = []
            output_row_bytes = 0
            yield from output_buffer.append(table)

        def append_output_row(row: dict[str, Any]) -> Iterable[pa.Table]:
            nonlocal output_row_bytes
            output_rows.append(row)
            output_row_bytes += _estimate_python_row_bytes(row)
            row_limit_reached = self._output_batch_size is not None and len(output_rows) >= self._output_batch_size
            byte_limit_reached = output_row_bytes >= self._output_target_max_bytes
            if row_limit_reached or byte_limit_reached:
                yield from flush_output_rows()

        for batch in batches:
            for row_dict in _iter_table_row_dicts(batch):
                result = self._map_fn(row_dict)
                if result is None:
                    continue
                if isinstance(result, dict):
                    yield from append_output_row(result)
                else:
                    for out_row in result:
                        if out_row is None:
                            continue
                        yield from append_output_row(out_row)
        if output_rows:
            yield from flush_output_rows()
        yield from output_buffer.flush()

    def _execute_flat_map(self, args: pa.Table) -> None:
        emitted = False
        for table in self._iter_flat_map_output_tables(args):
            self._queue.append(table)
            emitted = True
        if not emitted:
            self._queue.append(_empty_output_table_from_payload(self._payload))

    def _flat_map_rows_to_table(self, rows: list[dict[str, Any]]) -> pa.Table:
        if self._output_names:
            arrays = {name: [row.get(name) for row in rows] for name in self._output_names}
            return pa.table(arrays)
        return pa.Table.from_pylist(rows)

    def _default_null_handling(self) -> bool:
        return self._null_handling == _DEFAULT_NULL_HANDLING

    def _return_null_on_error(self) -> bool:
        return self._exception_handling == _RETURN_NULL

    def _execute_scalar_native(self, args: pa.Table) -> pa.Array:
        row_count = args.num_rows
        columns = [column.to_pylist() for column in args.columns]
        outputs: list[Any] = []

        for row_idx in range(row_count):
            if self._default_null_handling() and any(col[row_idx] is None for col in columns):
                outputs.append(None)
                continue
            call_args = [col[row_idx] for col in columns]
            try:
                result = self._map_fn(*call_args)
            except Exception:
                if self._return_null_on_error():
                    outputs.append(None)
                    continue
                raise
            if self._default_null_handling() and result is None:
                raise ValueError(_NULL_HANDLING_ERROR)
            outputs.append(result)
        return pa.array(outputs)

    def _execute_scalar_arrow(self, args: pa.Table) -> pa.Array:
        row_count = args.num_rows
        exception_occurred = False

        if self._default_null_handling():
            valid_mask = _build_valid_mask(args)
            valid_indices = [idx for idx, ok in enumerate(valid_mask) if ok]
            if valid_indices:
                args = args.take(pa.array(valid_indices, type=pa.int64()))
            else:
                args = args.slice(0, 0)
        else:
            valid_indices = None

        outputs: list[pa.Array] = []
        for batch in _iter_table_batches(args, self._batch_size):
            try:
                result = self._map_fn(*batch.columns)
            except Exception:
                if self._return_null_on_error():
                    exception_occurred = True
                    result = [None] * batch.num_rows
                else:
                    raise
            outputs.append(_coerce_scalar_array(result, batch.num_rows))

        if outputs:
            if len(outputs) == 1:
                result_array = outputs[0]
            else:
                try:
                    result_array = pa.concat_arrays(outputs)
                except pa.ArrowInvalid as exc:
                    if "offset overflow" in str(exc):
                        result_array = pa.chunked_array(outputs)
                    else:
                        raise
        else:
            result_array = pa.array([])

        if (
            self._default_null_handling()
            and not exception_occurred
            and any(value is None for value in result_array.to_pylist())
        ):
            raise ValueError(_NULL_HANDLING_ERROR)

        if valid_indices is not None:
            values = result_array.to_pylist()
            if len(values) != len(valid_indices):
                raise ValueError("map output row count does not match filtered input")
            full_values: list[Any] = [None] * row_count
            for idx, value in zip(valid_indices, values, strict=False):
                full_values[idx] = value
            result_array = pa.array(full_values)

        return result_array

    def _execute_map(self, args: pa.Table) -> None:
        if self._scalar_udf_type == "arrow":
            outputs = self._execute_scalar_arrow(args)
        else:
            outputs = self._execute_scalar_native(args)
        self._queue.append(pa.table({self._scalar_output_name: outputs}))

    def submit(self, args: pa.Table) -> None:
        args = _ensure_table(args)
        if args.num_rows == 0:
            return
        if self._is_map_batches or self._is_map_batches_rows:
            self._execute_map_batches(args)
        elif self._is_flat_map:
            self._execute_flat_map(args)
        elif self._is_map:
            self._execute_map(args)
        else:
            raise ValueError(f"Unknown UDF mode: {self._mode}")

    def take_ready_result(self) -> pa.Table | None:
        try:
            result = self._queue.popleft()
        except IndexError:
            return None
        return result

    def drain_outputs(self) -> list[pa.Table]:
        results = list(self._queue)
        self._queue.clear()
        return results

    def finished_submitting(self) -> None:
        if self._finished_submitting:
            return
        if self._is_map_batches:
            self._execute_map_batches_compute_batches(self._flush_map_batches_compute_batches())
        self._finished_submitting = True

    def all_tasks_finished(self) -> bool:
        return self._finished_submitting and not self._queue


__all__ = ["UDFExecutor"]
