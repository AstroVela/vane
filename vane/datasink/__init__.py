# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Generic immediate, idempotent Python DataSink contracts.

A sink binds once on the driver. Its bound form is serialized into a replayable
Arrow batch operator, and every worker opens a short-lived writer for one
batch. The first protocol intentionally supports only externally idempotent,
immediately visible writes; Vane never reports exactly-once delivery or remote
rollback.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]

    from vane import DuckDBPyRelation


_MAX_RESULT_METADATA_BYTES = 64 * 1024
_MAX_RESULT_WARNINGS_BYTES = 64 * 1024
_MAX_WRITE_RESULTS = 1_000_000
_MAX_TOTAL_RESULT_BYTES = 64 * 1024 * 1024
_MAX_WRITE_RESULT_WARNINGS = 4
_MAX_SUMMARY_WARNINGS = 16
_MAX_WARNING_BYTES = 4 * 1024
_MAX_UINT64 = (1 << 64) - 1


class CommitProtocol(str, Enum):
    """When writes become externally visible."""

    IMMEDIATE = "immediate"
    TWO_PHASE = "two_phase"


class RetryMode(str, Enum):
    """Whether replay converges on the same external state."""

    IDEMPOTENT = "idempotent"
    NEVER = "never"


class WriteState(str, Enum):
    """State represented by one selected worker result."""

    APPLIED = "applied"
    ABORTED = "aborted"


class WriteOutcome(str, Enum):
    """Driver-visible terminal knowledge about an operation."""

    APPLIED = "applied"
    ABORTED = "aborted"
    UNKNOWN = "unknown"


def _strict_non_negative_int(name: str, value: object, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    suffix = " or None" if allow_none else ""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer{suffix}")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer{suffix}")
    return value


def _strict_uint64(name: str, value: object, *, allow_none: bool = False) -> int | None:
    result = _strict_non_negative_int(name, value, allow_none=allow_none)
    if result is not None and result > _MAX_UINT64:
        suffix = " or None" if allow_none else ""
        raise ValueError(f"{name} must be an unsigned 64-bit integer{suffix}")
    return result


def _plain_json_value(name: str, value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{name} keys must be strings")
        return {key: _plain_json_value(name, item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json_value(name, item) for item in value]
    return value


def _json_mapping(name: str, value: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = {} if value is None else _plain_json_value(name, value)
    if not isinstance(normalized, dict):
        raise TypeError(f"{name} must be a mapping")
    try:
        encoded = json.dumps(normalized, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain finite JSON values") from error
    if len(encoded.encode("utf-8")) > _MAX_RESULT_METADATA_BYTES:
        raise ValueError(f"{name} must serialize to at most 64 KiB")
    return json.loads(encoded)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _safe_error_message(error: BaseException) -> str:
    try:
        return str(error)
    except BaseException:
        return "<error message unavailable>"


def _bounded_warning(value: str) -> str:
    encoded = value.strip().encode("utf-8", "replace")
    if len(encoded) <= _MAX_WARNING_BYTES:
        return encoded.decode("utf-8")
    omission = "…"
    remaining = _MAX_WARNING_BYTES - len(omission.encode("utf-8"))
    prefix_size = remaining // 2
    suffix_size = remaining - prefix_size
    return encoded[:prefix_size].decode("utf-8", "ignore") + omission + encoded[-suffix_size:].decode("utf-8", "ignore")


def _append_warning(warnings: tuple[str, ...], warning: str, *, limit: int) -> tuple[str, ...]:
    normalized = _bounded_warning(warning)
    if not normalized:
        return warnings
    if len(warnings) < limit:
        return warnings + (normalized,)
    if limit == 1:
        return ("additional DataSink warnings omitted",)
    return warnings[: limit - 1] + ("additional DataSink warnings omitted",)


@dataclass(frozen=True)
class EnvironmentSecret:
    """An opaque environment-variable reference resolved only on a worker."""

    variable: str

    def __post_init__(self) -> None:
        if not isinstance(self.variable, str):
            raise TypeError("EnvironmentSecret.variable must be a string")
        variable = self.variable.strip()
        if not variable:
            raise ValueError("EnvironmentSecret.variable must not be empty")
        if "=" in variable or "\x00" in variable:
            raise ValueError("EnvironmentSecret.variable is not a valid environment variable name")
        try:
            variable.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("EnvironmentSecret.variable must contain valid UTF-8") from error
        object.__setattr__(self, "variable", variable)

    def resolve(self) -> str:
        try:
            return os.environ[self.variable]
        except KeyError as error:
            raise RuntimeError(f"required DataSink environment secret is not set: {self.variable}") from error


@dataclass(frozen=True)
class DataSinkCapabilities:
    """Execution guarantees declared by a bound sink."""

    commit_protocol: CommitProtocol
    retry_mode: RetryMode

    def __post_init__(self) -> None:
        object.__setattr__(self, "commit_protocol", CommitProtocol(self.commit_protocol))
        object.__setattr__(self, "retry_mode", RetryMode(self.retry_mode))


@dataclass(frozen=True)
class DataSinkExecutionOptions:
    """Resource and batching requests for the internal Arrow operator."""

    batch_size: int | None = None
    cpus: float | None = None
    gpus: float | None = None
    memory_bytes: int | None = None
    target_max_batch_bytes: int | None = None
    task_input_max_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in ("batch_size", "memory_bytes", "target_max_batch_bytes", "task_input_max_bytes"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer or None")
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer or None")
        for name in ("cpus", "gpus"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite non-negative number or None")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be a finite non-negative number or None")

    def map_batches_kwargs(self) -> dict[str, Any]:
        return {
            name: value
            for name in (
                "batch_size",
                "cpus",
                "gpus",
                "memory_bytes",
                "target_max_batch_bytes",
                "task_input_max_bytes",
            )
            if (value := getattr(self, name)) is not None
        }


@dataclass(frozen=True)
class WriteContext:
    """Stable identity shared by the driver and every replay."""

    operation_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, str):
            raise TypeError("DataSink operation_id must be a string")
        operation_id = self.operation_id.strip()
        if not operation_id:
            raise ValueError("DataSink operation_id must not be empty")
        try:
            operation_id_bytes = operation_id.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("DataSink operation_id must contain valid UTF-8") from error
        if len(operation_id_bytes) > 256:
            raise ValueError("DataSink operation_id must be at most 256 UTF-8 bytes")
        object.__setattr__(self, "operation_id", operation_id)


@dataclass(frozen=True)
class WriteResult:
    """Bounded result produced by one selected worker batch."""

    rows_received: int
    rows_affected: int | None = None
    bytes_received: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    state: WriteState = WriteState.APPLIED

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows_received", _strict_uint64("rows_received", self.rows_received))
        object.__setattr__(
            self,
            "rows_affected",
            _strict_uint64("rows_affected", self.rows_affected, allow_none=True),
        )
        object.__setattr__(self, "bytes_received", _strict_uint64("bytes_received", self.bytes_received))
        object.__setattr__(self, "state", WriteState(self.state))
        metadata = _json_mapping("WriteResult.metadata", self.metadata)
        object.__setattr__(self, "metadata", cast(Mapping[str, Any], _freeze_json(metadata)))
        if isinstance(cast(object, self.warnings), str):
            raise TypeError("WriteResult.warnings must be a sequence of strings")
        warnings = tuple(self.warnings)
        if len(warnings) > _MAX_WRITE_RESULT_WARNINGS:
            raise ValueError("WriteResult.warnings must contain at most four values")
        for warning in warnings:
            if not isinstance(warning, str):
                raise TypeError("WriteResult.warnings must contain only strings")
            if not warning.strip():
                raise ValueError("WriteResult.warnings must not contain empty strings")
            try:
                warning_bytes = warning.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError("WriteResult.warnings must contain valid UTF-8") from error
            if len(warning_bytes) > _MAX_WARNING_BYTES:
                raise ValueError("each WriteResult warning must be at most 4 KiB")
        warnings_json = json.dumps(warnings, separators=(",", ":"))
        if len(warnings_json.encode("utf-8")) > _MAX_RESULT_WARNINGS_BYTES:
            raise ValueError("WriteResult.warnings must serialize to at most 64 KiB")
        object.__setattr__(self, "warnings", warnings)
        if self.state is WriteState.ABORTED and (self.rows_affected != 0 or self.bytes_received != 0):
            raise ValueError("aborted WriteResult requires zero rows_affected and zero bytes_received")

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            type(self),
            (
                self.rows_received,
                self.rows_affected,
                self.bytes_received,
                _plain_json_value("WriteResult.metadata", self.metadata),
                self.warnings,
                self.state,
            ),
        )


@dataclass(frozen=True)
class WriteSummary:
    """Aggregated driver-visible result for one operation."""

    operation_id: str
    outcome: WriteOutcome
    results: tuple[WriteResult, ...]
    rows_received: int
    rows_affected: int | None
    bytes_received: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", WriteContext(self.operation_id).operation_id)
        object.__setattr__(self, "outcome", WriteOutcome(self.outcome))
        results = tuple(self.results)
        if len(results) > _MAX_WRITE_RESULTS:
            raise ValueError("WriteSummary.results must contain at most 1000000 values")
        if any(not isinstance(result, WriteResult) for result in results):
            raise TypeError("WriteSummary.results must contain only WriteResult values")
        object.__setattr__(self, "results", results)
        states = {result.state for result in results}
        if self.outcome is WriteOutcome.APPLIED and WriteState.ABORTED in states:
            raise ValueError("applied WriteSummary must not contain aborted results")
        if self.outcome is WriteOutcome.ABORTED and states != {WriteState.ABORTED}:
            raise ValueError("aborted WriteSummary requires one or more aborted results")
        expected_received = sum(item.rows_received for item in results)
        expected_bytes = sum(item.bytes_received for item in results)
        affected_values = [item.rows_affected for item in results]
        expected_affected = None if any(value is None for value in affected_values) else sum(affected_values)  # type: ignore[arg-type]
        if self.rows_received != expected_received:
            raise ValueError("WriteSummary.rows_received does not match its results")
        if self.rows_affected != expected_affected:
            raise ValueError("WriteSummary.rows_affected does not match its results")
        if self.bytes_received != expected_bytes:
            raise ValueError("WriteSummary.bytes_received does not match its results")
        _strict_non_negative_int("WriteSummary.rows_received", self.rows_received)
        _strict_non_negative_int("WriteSummary.rows_affected", self.rows_affected, allow_none=True)
        _strict_non_negative_int("WriteSummary.bytes_received", self.bytes_received)
        object.__setattr__(
            self, "metadata", cast(Mapping[str, Any], _freeze_json(_json_mapping("metadata", self.metadata)))
        )
        if isinstance(cast(object, self.warnings), str):
            raise TypeError("WriteSummary.warnings must be a sequence of strings")
        warnings = tuple(self.warnings)
        if len(warnings) > _MAX_SUMMARY_WARNINGS:
            raise ValueError("WriteSummary.warnings must contain at most 16 values")
        if any(not isinstance(warning, str) for warning in warnings):
            raise TypeError("WriteSummary.warnings must contain only strings")
        for warning in warnings:
            if not warning.strip():
                raise ValueError("WriteSummary warnings must be non-empty and at most 4 KiB each")
            try:
                warning_bytes = warning.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError("WriteSummary.warnings must contain valid UTF-8") from error
            if len(warning_bytes) > _MAX_WARNING_BYTES:
                raise ValueError("WriteSummary warnings must be non-empty and at most 4 KiB each")
        object.__setattr__(self, "warnings", warnings)

    @property
    def batch_count(self) -> int:
        return len(self.results)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            type(self),
            (
                self.operation_id,
                self.outcome,
                self.results,
                self.rows_received,
                self.rows_affected,
                self.bytes_received,
                _plain_json_value("WriteSummary.metadata", self.metadata),
                self.warnings,
            ),
        )


class DataSinkWriteError(RuntimeError):
    """A DataSink did not finish with a known applied outcome."""

    def __init__(self, summary: WriteSummary, detail: str, *, safe_to_retry: bool) -> None:
        self.summary = summary
        self.operation_id = summary.operation_id
        self.outcome = summary.outcome
        self.safe_to_retry = bool(safe_to_retry)
        self.detail = _bounded_warning(str(detail)) or "DataSink failed without outcome detail"
        super().__init__(
            f"DataSink operation {summary.operation_id} ended with outcome {summary.outcome.value}: {self.detail}"
        )

    def __reduce__(self) -> tuple[Any, tuple[WriteSummary, str, bool]]:
        return (_restore_datasink_write_error, (self.summary, self.detail, self.safe_to_retry))


def _restore_datasink_write_error(
    summary: WriteSummary,
    detail: str,
    safe_to_retry: bool,
) -> DataSinkWriteError:
    return DataSinkWriteError(summary, detail, safe_to_retry=safe_to_retry)


class DataSinkWriter(ABC):
    """Worker-local writer opened for exactly one Arrow compute batch."""

    @abstractmethod
    def write(self, table: pa.Table) -> WriteResult:
        """Apply one batch using idempotent external semantics."""

    def abort(self, error: BaseException) -> None:
        """Release worker-local state after failure; this is not remote rollback."""

    def close(self) -> None:
        """Release worker-local resources."""


class BoundDataSink(ABC):
    """Schema-bound, cloudpickle-serializable sink execution contract."""

    @property
    @abstractmethod
    def capabilities(self) -> DataSinkCapabilities: ...

    @property
    def execution_options(self) -> DataSinkExecutionOptions:
        return DataSinkExecutionOptions()

    def prepare_input(self, relation: DuckDBPyRelation) -> DuckDBPyRelation:
        """Optionally add deterministic, lazy input transformations."""

        return relation

    @abstractmethod
    def open_writer(self, context: WriteContext) -> DataSinkWriter: ...


class BoundKeyedUpsertSink(BoundDataSink):
    """A bound upsert sink whose input keys must be globally unique and non-null."""

    @property
    @abstractmethod
    def key_columns(self) -> Sequence[str]: ...


class DataSink(ABC):
    """Driver-side factory for a schema-bound terminal sink."""

    @abstractmethod
    def bind(self, schema: pa.Schema) -> BoundDataSink: ...


_WIRE_COLUMNS = (
    "operation_id",
    "state",
    "rows_received",
    "rows_affected",
    "bytes_received",
    "metadata_json",
    "warnings_json",
)


def _wire_output_schema() -> dict[str, Any]:
    from vane import sqltypes

    return {
        "operation_id": sqltypes.VARCHAR,
        "state": sqltypes.VARCHAR,
        "rows_received": sqltypes.UBIGINT,
        "rows_affected": sqltypes.UBIGINT,
        "bytes_received": sqltypes.UBIGINT,
        "metadata_json": sqltypes.VARCHAR,
        "warnings_json": sqltypes.VARCHAR,
    }


def _empty_wire_table() -> pa.Table:
    import pyarrow as pa

    return pa.table(
        {
            "operation_id": pa.array([], type=pa.string()),
            "state": pa.array([], type=pa.string()),
            "rows_received": pa.array([], type=pa.uint64()),
            "rows_affected": pa.array([], type=pa.uint64()),
            "bytes_received": pa.array([], type=pa.uint64()),
            "metadata_json": pa.array([], type=pa.string()),
            "warnings_json": pa.array([], type=pa.string()),
        }
    )


def _result_to_wire_table(context: WriteContext, result: WriteResult) -> pa.Table:
    import pyarrow as pa

    return pa.table(
        {
            "operation_id": pa.array([context.operation_id], type=pa.string()),
            "state": pa.array([result.state.value], type=pa.string()),
            "rows_received": pa.array([result.rows_received], type=pa.uint64()),
            "rows_affected": pa.array([result.rows_affected], type=pa.uint64()),
            "bytes_received": pa.array([result.bytes_received], type=pa.uint64()),
            "metadata_json": pa.array(
                [json.dumps(_plain_json_value("metadata", result.metadata), sort_keys=True, separators=(",", ":"))],
                type=pa.string(),
            ),
            "warnings_json": pa.array(
                [json.dumps(result.warnings, separators=(",", ":"))],
                type=pa.string(),
            ),
        }
    )


@dataclass(frozen=True)
class _KeyValidation:
    original_names: tuple[str, ...]
    duplicate_marker: str
    null_marker: str

    def strip_and_validate(self, table: pa.Table) -> tuple[pa.Table, str | None]:
        import pyarrow as pa

        if table.num_rows == 0:
            return (
                pa.Table.from_arrays(
                    [table.column(index) for index in range(len(self.original_names))],
                    names=self.original_names,
                ),
                None,
            )
        duplicate_count = table.column(self.duplicate_marker)[0].as_py()
        null_count = table.column(self.null_marker)[0].as_py()
        if isinstance(duplicate_count, bool) or not isinstance(duplicate_count, int):
            raise RuntimeError("keyed DataSink duplicate validation returned an invalid count")
        if isinstance(null_count, bool) or not isinstance(null_count, int):
            raise RuntimeError("keyed DataSink null validation returned an invalid count")
        reason = None
        if null_count:
            reason = "null_keys"
        elif duplicate_count > 1:
            reason = "duplicate_keys"
        return (
            pa.Table.from_arrays(
                [table.column(index) for index in range(len(self.original_names))],
                names=self.original_names,
            ),
            reason,
        )


class _SinkBatchRuntime:
    def __init__(
        self,
        sink: BoundDataSink,
        context: WriteContext,
        key_validation: _KeyValidation | None,
    ) -> None:
        self._sink = sink
        self._context = context
        self._key_validation = key_validation

    def __call__(self, table: pa.Table) -> pa.Table:
        import pyarrow as pa

        if isinstance(table, pa.RecordBatch):
            table = pa.Table.from_batches([table])
        if not isinstance(table, pa.Table):
            raise TypeError(f"DataSink worker expected pyarrow.Table, got {type(table).__name__}")
        rejection_reason = None
        if self._key_validation is not None:
            table, rejection_reason = self._key_validation.strip_and_validate(table)
        if table.num_rows == 0:
            return _empty_wire_table()
        if rejection_reason is not None:
            return _result_to_wire_table(
                self._context,
                WriteResult(
                    rows_received=table.num_rows,
                    rows_affected=0,
                    metadata={"validation_error": rejection_reason},
                    state=WriteState.ABORTED,
                ),
            )

        writer = self._sink.open_writer(self._context)
        if not isinstance(writer, DataSinkWriter):
            raise TypeError(f"BoundDataSink.open_writer() must return DataSinkWriter, got {type(writer).__name__}")
        primary_error: BaseException | None = None
        result: WriteResult | None = None
        try:
            result = writer.write(table)
            if not isinstance(result, WriteResult):
                raise TypeError(f"DataSinkWriter.write() must return WriteResult, got {type(result).__name__}")
            if result.state is not WriteState.APPLIED:
                raise ValueError("DataSinkWriter.write() must return state='applied'")
            if result.rows_received != table.num_rows:
                raise ValueError(
                    f"DataSinkWriter rows_received={result.rows_received} does not match input rows={table.num_rows}"
                )
        except BaseException as error:
            primary_error = error
            try:
                writer.abort(error)
            except BaseException as abort_error:
                if hasattr(error, "add_note"):
                    error.add_note(
                        "DataSinkWriter.abort() local cleanup also failed: "
                        f"{type(abort_error).__name__}: {_safe_error_message(abort_error)}"
                    )
            raise
        finally:
            try:
                writer.close()
            except BaseException as close_error:
                if primary_error is not None:
                    if hasattr(primary_error, "add_note"):
                        primary_error.add_note(
                            "DataSinkWriter.close() also failed: "
                            f"{type(close_error).__name__}: {_safe_error_message(close_error)}"
                        )
                elif result is not None:
                    result = WriteResult(
                        rows_received=result.rows_received,
                        rows_affected=result.rows_affected,
                        bytes_received=result.bytes_received,
                        metadata=result.metadata,
                        warnings=_append_warning(
                            result.warnings,
                            "DataSinkWriter.close() failed after the external write was applied: "
                            f"{type(close_error).__name__}: {_safe_error_message(close_error)}",
                            limit=_MAX_WRITE_RESULT_WARNINGS,
                        ),
                        state=result.state,
                    )
                else:
                    raise
        if result is None:
            raise RuntimeError("DataSinkWriter completed without a WriteResult")
        return _result_to_wire_table(self._context, result)


def _make_batch_function(
    sink: BoundDataSink,
    context: WriteContext,
    key_validation: _KeyValidation | None,
) -> Any:
    runtime = _SinkBatchRuntime(sink, context, key_validation)

    def execute_datasink_batch(table: pa.Table) -> pa.Table:
        return runtime(table)

    return execute_datasink_batch


def _relation_arrow_schema(relation: DuckDBPyRelation) -> pa.Schema:
    import pyarrow as pa

    from vane.datasource import _convert_duckdb_pytype

    names = list(relation.columns)
    types = list(relation.types)
    if len(names) != len(types):
        raise RuntimeError("relation returned inconsistent column names and types")
    return pa.schema([pa.field(name, _convert_duckdb_pytype(dtype)) for name, dtype in zip(names, types, strict=True)])


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _prepare_key_validation(
    relation: DuckDBPyRelation,
    sink: BoundKeyedUpsertSink,
) -> tuple[DuckDBPyRelation, _KeyValidation]:
    raw_keys = sink.key_columns
    if isinstance(raw_keys, str) or not isinstance(raw_keys, Sequence):
        raise TypeError("BoundKeyedUpsertSink.key_columns must be a non-empty sequence of strings")
    key_columns = tuple(raw_keys)
    if not key_columns or any(not isinstance(key, str) or not key.strip() for key in key_columns):
        raise ValueError("BoundKeyedUpsertSink.key_columns must be a non-empty sequence of non-empty strings")
    key_columns = tuple(key.strip() for key in key_columns)
    if len({key.casefold() for key in key_columns}) != len(key_columns):
        raise ValueError("BoundKeyedUpsertSink.key_columns must not contain duplicates")

    original_names = tuple(relation.columns)
    by_name: dict[str, list[str]] = {}
    for name in original_names:
        by_name.setdefault(name.casefold(), []).append(name)
    for key in key_columns:
        matches = by_name.get(key.casefold(), [])
        if len(matches) != 1:
            raise ValueError(f"keyed DataSink key column {key!r} must match exactly one input column")

    existing = {name.casefold() for name in original_names}
    nonce = uuid.uuid4().hex
    row_count_marker = f"__vane_datasink_key_count_{nonce}"
    null_marker = f"__vane_datasink_null_count_{nonce}"
    duplicate_marker = f"__vane_datasink_max_key_count_{nonce}"
    if any(marker.casefold() in existing for marker in (row_count_marker, null_marker, duplicate_marker)):
        raise RuntimeError("failed to allocate internal keyed DataSink validation columns")

    quoted_keys = ", ".join(_quote_identifier(key) for key in key_columns)
    null_predicate = " OR ".join(f"{_quote_identifier(key)} IS NULL" for key in key_columns)
    first_stage = relation.project(
        f"*, count(*) OVER (PARTITION BY {quoted_keys}) AS {_quote_identifier(row_count_marker)}, "
        f"count(*) FILTER (WHERE {null_predicate}) OVER () AS {_quote_identifier(null_marker)}"
    )
    validated = first_stage.project(
        f"*, max({_quote_identifier(row_count_marker)}) OVER () AS {_quote_identifier(duplicate_marker)}"
    )
    return validated, _KeyValidation(original_names, duplicate_marker, null_marker)


def _write_result_from_mapping(operation_id: str, payload: Mapping[str, Any]) -> WriteResult:
    result_operation_id = payload.get("operation_id")
    if not isinstance(result_operation_id, str) or result_operation_id != operation_id:
        raise RuntimeError(
            f"DataSink result operation_id mismatch: expected {operation_id!r}, got {result_operation_id!r}"
        )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("DataSink result metadata must be a mapping")
    warnings = payload.get("warnings", ())
    if isinstance(warnings, str) or not isinstance(warnings, (list, tuple)):
        raise TypeError("DataSink result warnings must be a list of strings")
    rows_received = cast(int, _strict_uint64("rows_received", payload.get("rows_received")))
    rows_affected = _strict_uint64("rows_affected", payload.get("rows_affected"), allow_none=True)
    bytes_received = cast(int, _strict_uint64("bytes_received", payload.get("bytes_received")))
    return WriteResult(
        rows_received=rows_received,
        rows_affected=rows_affected,
        bytes_received=bytes_received,
        metadata=metadata,
        warnings=tuple(warnings),
        state=WriteState(str(payload.get("state") or "")),
    )


def _result_wire_bytes(operation_id: str, result: WriteResult) -> int:
    metadata = json.dumps(_plain_json_value("metadata", result.metadata), sort_keys=True, separators=(",", ":"))
    warnings = json.dumps(result.warnings, separators=(",", ":"))
    return (
        len(operation_id.encode("utf-8"))
        + len(result.state.value)
        + 3 * 8
        + len(metadata.encode("utf-8"))
        + len(warnings.encode("utf-8"))
    )


def _results_from_native(operation_id: str, payload: Mapping[str, Any]) -> tuple[WriteResult, ...]:
    native_operation_id = payload.get("operation_id")
    outcome_unknown = payload.get("outcome_unknown")
    if not isinstance(outcome_unknown, bool):
        raise TypeError("native DataSink outcome_unknown must be a boolean")
    outcome_aborted = payload.get("outcome_aborted")
    if not isinstance(outcome_aborted, bool):
        raise TypeError("native DataSink outcome_aborted must be a boolean")
    if outcome_unknown and outcome_aborted:
        raise RuntimeError("native DataSink result cannot be both aborted and unknown")
    if native_operation_id != operation_id and not (outcome_unknown and native_operation_id == ""):
        raise RuntimeError(
            f"native DataSink operation_id mismatch: expected {operation_id!r}, got {native_operation_id!r}"
        )
    raw_results = payload.get("write_results")
    if not isinstance(raw_results, list):
        raise TypeError("native DataSink result must contain a write_results list")
    if len(raw_results) > _MAX_WRITE_RESULTS:
        raise ValueError("native DataSink result exceeds the write-result limit")
    results: list[WriteResult] = []
    total_bytes = 0
    for item in raw_results:
        if not isinstance(item, Mapping):
            raise TypeError("native DataSink write results must be mappings")
        result = _write_result_from_mapping(operation_id, item)
        total_bytes += _result_wire_bytes(operation_id, result)
        if total_bytes > _MAX_TOTAL_RESULT_BYTES:
            raise ValueError("native DataSink result exceeds the 64 MiB coordinator payload limit")
        results.append(result)
    states = {result.state for result in results}
    if not outcome_unknown:
        if outcome_aborted and states != {WriteState.ABORTED}:
            raise RuntimeError("known aborted DataSink outcome requires one or more aborted worker results")
        if not outcome_aborted and WriteState.ABORTED in states:
            raise RuntimeError("known applied DataSink outcome must not contain aborted worker results")
    return tuple(results)


def _results_from_arrow(operation_id: str, table: pa.Table) -> tuple[WriteResult, ...]:
    import pyarrow as pa

    if not isinstance(table, pa.Table):
        raise TypeError(f"local-fast DataSink returned {type(table).__name__}, expected pyarrow.Table")
    if tuple(table.column_names) != _WIRE_COLUMNS:
        raise RuntimeError("local-fast DataSink returned an invalid worker result schema")
    if table.num_rows > _MAX_WRITE_RESULTS:
        raise ValueError("local-fast DataSink result exceeds the write-result limit")
    results: list[WriteResult] = []
    total_bytes = 0
    for row in table.to_pylist():
        metadata_json = row["metadata_json"]
        warnings_json = row["warnings_json"]
        if not isinstance(metadata_json, str) or len(metadata_json.encode("utf-8")) > _MAX_RESULT_METADATA_BYTES:
            raise ValueError("DataSink result metadata must be a JSON string of at most 64 KiB")
        if not isinstance(warnings_json, str) or len(warnings_json.encode("utf-8")) > _MAX_RESULT_WARNINGS_BYTES:
            raise ValueError("DataSink result warnings must be a JSON string of at most 64 KiB")
        try:
            metadata = json.loads(metadata_json)
            warnings = json.loads(warnings_json)
        except (TypeError, ValueError) as error:
            raise RuntimeError("DataSink result contains invalid JSON") from error
        result = _write_result_from_mapping(
            operation_id,
            {
                "operation_id": row["operation_id"],
                "state": row["state"],
                "rows_received": row["rows_received"],
                "rows_affected": row["rows_affected"],
                "bytes_received": row["bytes_received"],
                "metadata": metadata,
                "warnings": warnings,
            },
        )
        total_bytes += _result_wire_bytes(operation_id, result)
        if total_bytes > _MAX_TOTAL_RESULT_BYTES:
            raise ValueError("local-fast DataSink result exceeds the 64 MiB coordinator payload limit")
        results.append(result)
    return tuple(results)


def _cleanup_warnings(payload: Mapping[str, Any]) -> tuple[str, ...]:
    raw = payload.get("data_sink_cleanup_warnings", ())
    if isinstance(raw, str):
        raw = (raw,) if raw else ()
    if not isinstance(raw, (list, tuple)) or any(not isinstance(item, str) for item in raw):
        raise TypeError("native DataSink cleanup warnings must be a sequence of strings")
    warnings: tuple[str, ...] = ()
    for item in raw:
        warnings = _append_warning(warnings, item, limit=_MAX_SUMMARY_WARNINGS)
    return warnings


def _summary(
    context: WriteContext,
    outcome: WriteOutcome,
    results: tuple[WriteResult, ...],
    warnings: tuple[str, ...] = (),
) -> WriteSummary:
    summary_warnings = warnings
    for result in results:
        for warning in result.warnings:
            summary_warnings = _append_warning(summary_warnings, warning, limit=_MAX_SUMMARY_WARNINGS)
    affected = [result.rows_affected for result in results]
    rows_affected = None if any(value is None for value in affected) else sum(affected)  # type: ignore[arg-type]
    return WriteSummary(
        operation_id=context.operation_id,
        outcome=outcome,
        results=results,
        rows_received=sum(result.rows_received for result in results),
        rows_affected=rows_affected,
        bytes_received=sum(result.bytes_received for result in results),
        warnings=summary_warnings,
    )


def _unknown_error(
    context: WriteContext,
    results: tuple[WriteResult, ...],
    error: BaseException | str,
    warnings: tuple[str, ...] = (),
) -> DataSinkWriteError:
    detail = error if isinstance(error, str) else f"{type(error).__name__}: {_safe_error_message(error)}"
    return DataSinkWriteError(
        _summary(context, WriteOutcome.UNKNOWN, results, warnings),
        detail,
        safe_to_retry=True,
    )


def _aborted_error(
    context: WriteContext,
    results: tuple[WriteResult, ...],
    detail: str,
    warnings: tuple[str, ...] = (),
) -> DataSinkWriteError:
    return DataSinkWriteError(
        _summary(context, WriteOutcome.ABORTED, results, warnings),
        detail,
        safe_to_retry=True,
    )


def write_datasink(
    relation: DuckDBPyRelation,
    sink: DataSink,
    *,
    operation_id: str | None = None,
) -> WriteSummary:
    """Execute a relation into an immediate, idempotent Python DataSink."""

    import cloudpickle

    from vane import DuckDBPyRelation as RuntimeDuckDBPyRelation
    from vane.runners import get_or_create_runner, get_or_infer_runner_type

    if not isinstance(relation, RuntimeDuckDBPyRelation):
        raise TypeError(f"relation must be DuckDBPyRelation, got {type(relation).__name__}")
    if not isinstance(sink, DataSink):
        raise TypeError(f"sink must be DataSink, got {type(sink).__name__}")
    context = WriteContext(str(uuid.uuid4()) if operation_id is None else operation_id)
    bound = sink.bind(_relation_arrow_schema(relation))
    if not isinstance(bound, BoundDataSink):
        raise TypeError(f"DataSink.bind() must return BoundDataSink, got {type(bound).__name__}")
    capabilities = bound.capabilities
    if not isinstance(capabilities, DataSinkCapabilities):
        raise TypeError("BoundDataSink.capabilities must be DataSinkCapabilities")
    if capabilities.commit_protocol is not CommitProtocol.IMMEDIATE:
        raise ValueError("DataSink currently requires commit_protocol='immediate'")
    if capabilities.retry_mode is not RetryMode.IDEMPOTENT:
        raise ValueError("DataSink currently requires retry_mode='idempotent'")
    options = bound.execution_options
    if not isinstance(options, DataSinkExecutionOptions):
        raise TypeError("BoundDataSink.execution_options must be DataSinkExecutionOptions")
    try:
        cloudpickle.dumps(bound)
    except BaseException as error:
        raise TypeError("bound DataSink must be cloudpickle-serializable before execution") from error

    prepared_relation = bound.prepare_input(relation)
    if not isinstance(prepared_relation, RuntimeDuckDBPyRelation):
        raise TypeError("BoundDataSink.prepare_input() must return DuckDBPyRelation")
    key_validation = None
    if isinstance(bound, BoundKeyedUpsertSink):
        prepared_relation, key_validation = _prepare_key_validation(prepared_relation, bound)
    batch_function = _make_batch_function(bound, context, key_validation)
    try:
        cloudpickle.dumps(batch_function)
    except BaseException as error:
        raise TypeError("DataSink worker callable must be cloudpickle-serializable before execution") from error
    mapped = prepared_relation.map_batches(
        batch_function,
        schema=_wire_output_schema(),
        **options.map_batches_kwargs(),
    )
    terminal = mapped._mark_datasink(context.operation_id)

    results: tuple[WriteResult, ...] = ()
    warnings: tuple[str, ...] = ()
    try:
        runner_type = get_or_infer_runner_type()
        if runner_type == "local-fast":
            results = _results_from_arrow(context.operation_id, terminal.to_arrow_table())
            states = {result.state for result in results}
            if states == {WriteState.ABORTED}:
                raise _aborted_error(
                    context,
                    results,
                    "keyed DataSink input validation rejected the operation before writers opened",
                )
            if WriteState.ABORTED in states:
                raise _unknown_error(context, results, "DataSink results mixed applied and aborted states")
            return _summary(context, WriteOutcome.APPLIED, results)
        native_result = get_or_create_runner().run_datasink(terminal)
        if not isinstance(native_result, Mapping):
            raise TypeError(f"Runner.run_datasink() returned {type(native_result).__name__}, expected a mapping")
        warnings = _cleanup_warnings(native_result)
        results = _results_from_native(context.operation_id, native_result)
        if native_result["outcome_unknown"]:
            detail = native_result.get("outcome_error")
            if not isinstance(detail, str) or not detail:
                detail = "distributed execution may have applied external writes"
            raise _unknown_error(context, results, detail, warnings)
        if native_result["outcome_aborted"]:
            detail = native_result.get("outcome_error")
            if not isinstance(detail, str) or not detail:
                detail = "DataSink input was rejected before writers opened"
            raise _aborted_error(context, results, detail, warnings)
        return _summary(context, WriteOutcome.APPLIED, results, warnings)
    except DataSinkWriteError:
        raise
    except BaseException as error:
        raise _unknown_error(context, results, error, warnings) from error
    finally:
        del terminal
        del mapped
        del prepared_relation


__all__ = [
    "BoundDataSink",
    "BoundKeyedUpsertSink",
    "CommitProtocol",
    "DataSink",
    "DataSinkCapabilities",
    "DataSinkExecutionOptions",
    "DataSinkWriteError",
    "DataSinkWriter",
    "EnvironmentSecret",
    "RetryMode",
    "WriteContext",
    "WriteOutcome",
    "WriteResult",
    "WriteState",
    "WriteSummary",
    "write_datasink",
]
