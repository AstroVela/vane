# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""First-class terminal data sinks for Vane.

``DataSink`` binds once on the driver. Its bound form is serialized with the
query and opens a ``DataSinkWriter`` for every replayable worker batch. Worker
calls return bounded ``WriteResult`` values; only those values cross the
execution boundary. The driver then performs the sink's commit, abort, or
reconciliation protocol and returns a ``WriteSummary``.
"""

from __future__ import annotations

import json
import math
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]

    from vane import DuckDBPyRelation


_MAX_COMMIT_TOKEN_BYTES = 64 * 1024
_MAX_RESULT_METADATA_BYTES = 64 * 1024
_MAX_WRITE_RESULTS = 1_000_000
_MAX_TOTAL_RESULT_BYTES = 64 * 1024 * 1024
_MAX_WRITE_RESULT_WARNINGS = 4
_MAX_SUMMARY_WARNINGS = 16
_MAX_SUMMARY_WARNING_BYTES = 4 * 1024
_MAX_UINT64 = (1 << 64) - 1


class CommitProtocol(str, Enum):
    """How worker writes become externally visible."""

    IMMEDIATE = "immediate"
    TWO_PHASE = "two_phase"


class RetryMode(str, Enum):
    """Whether the engine and caller may replay a sink operation.

    ``IDEMPOTENT`` requires the same external state after any batch is replayed
    and after independent batches complete in a different order. For a
    two-phase sink, repeating prepare, commit, or abort with the same
    ``operation_id`` must also converge on the same external state.
    """

    NEVER = "never"
    IDEMPOTENT = "idempotent"


class WriteState(str, Enum):
    """State represented by one worker ``WriteResult``."""

    APPLIED = "applied"
    PREPARED = "prepared"


class WriteOutcome(str, Enum):
    """Driver-visible terminal outcome for a sink operation."""

    COMMITTED = "committed"
    ABORTED = "aborted"
    UNKNOWN = "unknown"


def _strict_non_negative_int(name: str, value: object, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        suffix = " or None" if allow_none else ""
        raise TypeError(f"{name} must be a non-negative integer{suffix}")
    if value < 0:
        suffix = " or None" if allow_none else ""
        raise ValueError(f"{name} must be a non-negative integer{suffix}")
    return value


def _strict_uint64(name: str, value: object, *, allow_none: bool = False) -> int | None:
    result = _strict_non_negative_int(name, value, allow_none=allow_none)
    if result is not None and result > _MAX_UINT64:
        suffix = " or None" if allow_none else ""
        raise ValueError(f"{name} must be an unsigned 64-bit integer{suffix}")
    return result


def _json_mapping(
    name: str,
    value: Mapping[str, Any] | None,
    *,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    result = {} if value is None else _plain_json_value(name, value)
    if not isinstance(result, dict):
        raise TypeError(f"{name} must be a mapping")
    try:
        encoded = json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain finite JSON values") from exc
    encoded_bytes = encoded.encode("utf-8")
    if max_bytes is not None and len(encoded_bytes) > max_bytes:
        raise ValueError(f"{name} must serialize to at most {max_bytes} UTF-8 bytes")
    # Round-trip to detach nested containers from caller-owned mutable values.
    return json.loads(encoded)


def _plain_json_value(name: str, value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{name} keys must be strings")
        return {key: _plain_json_value(name, item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json_value(name, item) for item in value]
    return value


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _freeze_json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})


def _bounded_warning(value: str) -> str:
    # Cleanup errors are diagnostic and can originate in arbitrary third-party
    # code. Normalize lone surrogates before measuring them so malformed error
    # text cannot replace a known successful external write with a reporting
    # failure.
    encoded = value.encode("utf-8", "replace")
    if len(encoded) <= _MAX_SUMMARY_WARNING_BYTES:
        return encoded.decode("utf-8")
    suffix = "…"
    prefix = encoded[: _MAX_SUMMARY_WARNING_BYTES - len(suffix.encode("utf-8"))].decode("utf-8", "ignore")
    return prefix + suffix


def _safe_exception_message(error: BaseException) -> str:
    try:
        return str(error)
    except BaseException:
        return "<error message unavailable>"


def _append_summary_warning(warnings: tuple[str, ...], warning: str) -> tuple[str, ...]:
    bounded = _bounded_warning(warning.strip())
    if not bounded:
        return warnings
    if len(warnings) < _MAX_SUMMARY_WARNINGS:
        return warnings + (bounded,)
    return warnings[: _MAX_SUMMARY_WARNINGS - 2] + (
        "additional DataSink warnings omitted",
        bounded,
    )


def _append_write_result_warning(warnings: tuple[str, ...], warning: str) -> tuple[str, ...]:
    bounded = _bounded_warning(warning.strip())
    if not bounded:
        return warnings
    if len(warnings) < _MAX_WRITE_RESULT_WARNINGS:
        return warnings + (bounded,)
    return warnings[: _MAX_WRITE_RESULT_WARNINGS - 2] + (
        "additional worker warnings omitted",
        bounded,
    )


def _cleanup_warnings_from_native(payload: Mapping[str, Any]) -> tuple[str, ...]:
    raw_warnings = payload.get("data_sink_cleanup_warnings")
    if not isinstance(raw_warnings, list):
        raise TypeError("native DataSink cleanup warnings must be a list of strings")
    if any(not isinstance(warning, str) for warning in raw_warnings):
        raise TypeError("native DataSink cleanup warnings must contain only strings")
    warnings = tuple(_bounded_warning(warning.strip()) for warning in raw_warnings if warning.strip())
    if len(warnings) <= _MAX_SUMMARY_WARNINGS:
        return warnings
    return warnings[: _MAX_SUMMARY_WARNINGS - 1] + ("additional DataSink cleanup warnings omitted",)


@dataclass(frozen=True)
class DataSinkCapabilities:
    """Execution and commit guarantees declared by a bound sink."""

    commit_protocol: CommitProtocol
    retry_mode: RetryMode
    supports_abort: bool = False
    supports_reconcile: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "commit_protocol", CommitProtocol(self.commit_protocol))
        object.__setattr__(self, "retry_mode", RetryMode(self.retry_mode))
        if not isinstance(self.supports_abort, bool):
            raise TypeError("supports_abort must be a boolean")
        if not isinstance(self.supports_reconcile, bool):
            raise TypeError("supports_reconcile must be a boolean")
        if self.commit_protocol is CommitProtocol.IMMEDIATE and (self.supports_abort or self.supports_reconcile):
            raise ValueError("immediate DataSink cannot declare abort or reconciliation support")
        if self.commit_protocol is CommitProtocol.TWO_PHASE and not self.supports_abort:
            raise ValueError("two-phase DataSink must support operation-scoped abort")


@dataclass(frozen=True)
class DataSinkExecutionOptions:
    """Worker resource and batching requests for the internal Arrow operator."""

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
            checked = _strict_non_negative_int(name, value)
            if checked == 0:
                raise ValueError(f"{name} must be positive when provided")
        for name in ("cpus", "gpus"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite non-negative number")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be a finite non-negative number")

    def map_batches_kwargs(self) -> dict[str, Any]:
        """Return only options understood by ``DuckDBPyRelation.map_batches``."""
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
    """Stable identity shared by the driver and every worker replay."""

    operation_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, str):
            raise TypeError("DataSink operation_id must be a string")
        operation_id = self.operation_id.strip()
        if not operation_id:
            raise ValueError("DataSink operation_id must not be empty")
        if len(operation_id.encode("utf-8")) > 256:
            raise ValueError("DataSink operation_id must be at most 256 UTF-8 bytes")
        object.__setattr__(self, "operation_id", operation_id)


@dataclass(frozen=True)
class WriteResult:
    """Bounded, serializable result produced by one replayable worker batch."""

    state: WriteState
    rows_received: int
    rows_affected: int | None = None
    bytes_received: int = 0
    commit_token: bytes | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", WriteState(self.state))
        object.__setattr__(self, "rows_received", _strict_uint64("rows_received", self.rows_received))
        object.__setattr__(
            self,
            "rows_affected",
            _strict_uint64("rows_affected", self.rows_affected, allow_none=True),
        )
        object.__setattr__(self, "bytes_received", _strict_uint64("bytes_received", self.bytes_received))
        if self.commit_token is not None and not isinstance(self.commit_token, bytes):
            raise TypeError("commit_token must be bytes or None")
        if self.commit_token is not None and len(self.commit_token) > _MAX_COMMIT_TOKEN_BYTES:
            raise ValueError(f"commit_token must be at most {_MAX_COMMIT_TOKEN_BYTES} bytes")
        metadata = _json_mapping(
            "WriteResult.metadata",
            self.metadata,
            max_bytes=_MAX_RESULT_METADATA_BYTES,
        )
        object.__setattr__(self, "metadata", _freeze_json_mapping(metadata))
        if isinstance(cast(object, self.warnings), str):
            raise TypeError("WriteResult.warnings must be a tuple of strings")
        warnings = tuple(self.warnings)
        if len(warnings) > _MAX_WRITE_RESULT_WARNINGS:
            raise ValueError(f"WriteResult.warnings must contain at most {_MAX_WRITE_RESULT_WARNINGS} values")
        for warning in warnings:
            if not isinstance(warning, str):
                raise TypeError("WriteResult.warnings must contain only strings")
            if not warning:
                raise ValueError("WriteResult.warnings must not contain empty strings")
            try:
                warning_bytes = warning.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("WriteResult.warnings must contain valid UTF-8") from exc
            if len(warning_bytes) > _MAX_SUMMARY_WARNING_BYTES:
                raise ValueError(f"each WriteResult warning must be at most {_MAX_SUMMARY_WARNING_BYTES} UTF-8 bytes")
        object.__setattr__(self, "warnings", warnings)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            type(self),
            (
                self.state,
                self.rows_received,
                self.rows_affected,
                self.bytes_received,
                self.commit_token,
                _plain_json_value("WriteResult.metadata", self.metadata),
                self.warnings,
            ),
        )


@dataclass(frozen=True)
class WriteSummary:
    """Aggregated driver-visible result of a sink operation."""

    operation_id: str
    outcome: WriteOutcome
    results: tuple[WriteResult, ...]
    rows_received: int
    rows_affected: int | None
    bytes_received: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        context = WriteContext(self.operation_id)
        object.__setattr__(self, "operation_id", context.operation_id)
        object.__setattr__(self, "outcome", WriteOutcome(self.outcome))
        results = tuple(self.results)
        if any(not isinstance(result, WriteResult) for result in results):
            raise TypeError("WriteSummary.results must contain only WriteResult values")
        object.__setattr__(self, "results", results)
        rows_received = _strict_non_negative_int("rows_received", self.rows_received)
        rows_affected = _strict_non_negative_int("rows_affected", self.rows_affected, allow_none=True)
        bytes_received = _strict_non_negative_int("bytes_received", self.bytes_received)
        if rows_received != sum(result.rows_received for result in results):
            raise ValueError("WriteSummary.rows_received does not match its worker batch results")
        affected_values = [result.rows_affected for result in results]
        expected_affected = (
            None if any(value is None for value in affected_values) else sum(affected_values)  # type: ignore[arg-type]
        )
        if rows_affected != expected_affected:
            raise ValueError("WriteSummary.rows_affected does not match its worker batch results")
        if bytes_received != sum(result.bytes_received for result in results):
            raise ValueError("WriteSummary.bytes_received does not match its worker batch results")
        object.__setattr__(self, "rows_received", rows_received)
        object.__setattr__(self, "rows_affected", rows_affected)
        object.__setattr__(self, "bytes_received", bytes_received)
        object.__setattr__(
            self,
            "metadata",
            _freeze_json_mapping(
                _json_mapping("WriteSummary.metadata", self.metadata, max_bytes=_MAX_RESULT_METADATA_BYTES)
            ),
        )
        if isinstance(cast(object, self.warnings), str):
            raise TypeError("WriteSummary.warnings must be a tuple of strings")
        warnings = tuple(self.warnings)
        if len(warnings) > _MAX_SUMMARY_WARNINGS:
            raise ValueError(f"WriteSummary.warnings must contain at most {_MAX_SUMMARY_WARNINGS} values")
        for warning in warnings:
            if not isinstance(warning, str):
                raise TypeError("WriteSummary.warnings must contain only strings")
            if not warning:
                raise ValueError("WriteSummary.warnings must not contain empty strings")
            if len(warning.encode("utf-8")) > _MAX_SUMMARY_WARNING_BYTES:
                raise ValueError(f"each WriteSummary warning must be at most {_MAX_SUMMARY_WARNING_BYTES} UTF-8 bytes")
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
    """A sink failed after execution may have reached the external service."""

    def __init__(self, summary: WriteSummary, detail: str, *, safe_to_retry: bool) -> None:
        self.summary = summary
        self.operation_id = summary.operation_id
        self.outcome = summary.outcome
        self.safe_to_retry = bool(safe_to_retry)
        self.detail = str(detail)
        super().__init__(
            f"DataSink operation {summary.operation_id} ended with outcome {summary.outcome.value}: {self.detail}"
        )

    def __reduce__(self) -> tuple[Any, tuple[WriteSummary, str, bool]]:
        return (_restore_data_sink_write_error, (self.summary, self.detail, self.safe_to_retry))


def _restore_data_sink_write_error(
    summary: WriteSummary,
    detail: str,
    safe_to_retry: bool,
) -> DataSinkWriteError:
    return DataSinkWriteError(summary, detail, safe_to_retry=safe_to_retry)


class DataSinkWriter(ABC):
    """Worker-local writer opened for one Arrow compute batch."""

    @abstractmethod
    def write(self, table: pa.Table) -> WriteResult:
        """Write one table atomically enough to return a single ``WriteResult``."""
        ...

    def abort(self, error: BaseException) -> None:
        """Release or cancel worker-local state after ``write`` fails."""

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
        """Return the relation that worker writers consume.

        A sink may add internal validation columns or deterministic routing
        operators after binding. The transformed relation remains part of the
        same lazy execution plan as the external write.
        """
        return relation

    @abstractmethod
    def open_writer(self, context: WriteContext) -> DataSinkWriter: ...

    def commit(self, context: WriteContext, results: tuple[WriteResult, ...]) -> Mapping[str, Any] | None:
        """Commit all prepared batches for an operation.

        The result order is unspecified; commit implementations must identify
        prepared work by token or operation identity instead of tuple position.
        """
        raise NotImplementedError("two-phase DataSink must implement commit()")

    def abort(self, context: WriteContext, results: tuple[WriteResult, ...], error: BaseException) -> None:
        """Abort every prepared write for ``context.operation_id``.

        ``results`` is diagnostic and may be incomplete when distributed
        execution fails. An abort implementation must therefore use the
        operation identity as its authoritative transaction scope.
        """
        raise NotImplementedError("DataSink does not implement abort()")

    def reconcile(self, context: WriteContext) -> WriteOutcome:
        raise NotImplementedError("DataSink does not implement reconcile()")


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
    "commit_token",
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
        "commit_token": sqltypes.BLOB,
        "metadata_json": sqltypes.VARCHAR,
        "warnings_json": sqltypes.VARCHAR,
    }


def _result_to_wire_table(context: WriteContext, result: WriteResult) -> pa.Table:
    import pyarrow as pa

    return pa.table(
        {
            "operation_id": pa.array([context.operation_id], type=pa.string()),
            "state": pa.array([result.state.value], type=pa.string()),
            "rows_received": pa.array([result.rows_received], type=pa.uint64()),
            "rows_affected": pa.array([result.rows_affected], type=pa.uint64()),
            "bytes_received": pa.array([result.bytes_received], type=pa.uint64()),
            "commit_token": pa.array([result.commit_token], type=pa.binary()),
            "metadata_json": pa.array(
                [
                    json.dumps(
                        _plain_json_value("WriteResult.metadata", result.metadata),
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ],
                type=pa.string(),
            ),
            "warnings_json": pa.array(
                [json.dumps(result.warnings, allow_nan=False, separators=(",", ":"))],
                type=pa.string(),
            ),
        }
    )


class _SinkBatchRuntime:
    def __init__(
        self,
        sink: BoundDataSink,
        context: WriteContext,
        capabilities: DataSinkCapabilities,
    ) -> None:
        self._sink = sink
        self._context = context
        self._expected_state = (
            WriteState.APPLIED if capabilities.commit_protocol is CommitProtocol.IMMEDIATE else WriteState.PREPARED
        )

    def __call__(self, table: pa.Table) -> pa.Table:
        import pyarrow as pa

        if isinstance(table, pa.RecordBatch):
            table = pa.Table.from_batches([table])
        if not isinstance(table, pa.Table):
            raise TypeError(f"DataSink worker expected pyarrow.Table, got {type(table).__name__}")

        writer = self._sink.open_writer(self._context)
        if not isinstance(writer, DataSinkWriter):
            raise TypeError(f"BoundDataSink.open_writer() must return DataSinkWriter, got {type(writer).__name__}")
        primary_error: BaseException | None = None
        result: WriteResult | None = None
        try:
            result = writer.write(table)
            if not isinstance(result, WriteResult):
                raise TypeError(f"DataSinkWriter.write() must return WriteResult, got {type(result).__name__}")
            if result.state is not self._expected_state:
                raise ValueError(
                    f"DataSinkWriter returned state {result.state.value!r}; expected {self._expected_state.value!r}"
                )
            if result.rows_received != table.num_rows:
                raise ValueError(
                    f"DataSinkWriter rows_received={result.rows_received} does not match input rows={table.num_rows}"
                )
        except BaseException as exc:
            primary_error = exc
            try:
                writer.abort(exc)
            except BaseException as abort_error:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        "DataSinkWriter.abort() also failed: "
                        f"{type(abort_error).__name__}: {_safe_exception_message(abort_error)}"
                    )
            raise
        finally:
            try:
                writer.close()
            except BaseException as close_error:
                if primary_error is None:
                    if result is None:
                        raise
                    result = WriteResult(
                        state=result.state,
                        rows_received=result.rows_received,
                        rows_affected=result.rows_affected,
                        bytes_received=result.bytes_received,
                        commit_token=result.commit_token,
                        metadata=result.metadata,
                        warnings=_append_write_result_warning(
                            result.warnings,
                            "DataSinkWriter.close() failed after write succeeded: "
                            f"{type(close_error).__name__}: {_safe_exception_message(close_error)}",
                        ),
                    )
                elif hasattr(primary_error, "add_note"):
                    primary_error.add_note(
                        "DataSinkWriter.close() also failed: "
                        f"{type(close_error).__name__}: {_safe_exception_message(close_error)}"
                    )
        if result is None:
            raise RuntimeError("DataSinkWriter completed without a WriteResult")
        return _result_to_wire_table(self._context, result)


def _make_batch_function(
    sink: BoundDataSink,
    context: WriteContext,
    capabilities: DataSinkCapabilities,
) -> Any:
    runtime = _SinkBatchRuntime(sink, context, capabilities)

    def execute_data_sink_batch(table: pa.Table) -> pa.Table:
        return runtime(table)

    return execute_data_sink_batch


def _relation_arrow_schema(relation: DuckDBPyRelation) -> pa.Schema:
    import pyarrow as pa

    from vane.datasource import _convert_duckdb_pytype

    names = list(relation.columns)
    types = list(relation.types)
    if len(names) != len(types):
        raise RuntimeError("relation returned inconsistent column names and types")
    return pa.schema([pa.field(name, _convert_duckdb_pytype(dtype)) for name, dtype in zip(names, types, strict=True)])


def _write_result_from_mapping(operation_id: str, payload: Mapping[str, Any]) -> WriteResult:
    result_operation_id = str(payload.get("operation_id") or "")
    if result_operation_id != operation_id:
        raise RuntimeError(
            f"DataSink result operation_id mismatch: expected {operation_id!r}, got {result_operation_id!r}"
        )
    metadata_value = payload.get("metadata", {})
    if not isinstance(metadata_value, Mapping):
        raise TypeError("DataSink result metadata must be a mapping")
    rows_received = payload.get("rows_received")
    if isinstance(rows_received, bool) or not isinstance(rows_received, int):
        raise TypeError("DataSink result rows_received must be a non-negative integer")
    rows_affected = payload.get("rows_affected")
    if rows_affected is not None and (isinstance(rows_affected, bool) or not isinstance(rows_affected, int)):
        raise TypeError("DataSink result rows_affected must be a non-negative integer or None")
    bytes_received = payload.get("bytes_received")
    if isinstance(bytes_received, bool) or not isinstance(bytes_received, int):
        raise TypeError("DataSink result bytes_received must be a non-negative integer")
    raw_warnings = payload.get("warnings", ())
    if isinstance(raw_warnings, str) or not isinstance(raw_warnings, (list, tuple)):
        raise TypeError("DataSink result warnings must be a list of strings")
    return WriteResult(
        state=WriteState(str(payload.get("state") or "")),
        rows_received=rows_received,
        rows_affected=rows_affected,
        bytes_received=bytes_received,
        commit_token=payload.get("commit_token"),
        metadata=metadata_value,
        warnings=tuple(raw_warnings),
    )


def _write_result_wire_bytes(operation_id: str, result: WriteResult) -> int:
    metadata_json = json.dumps(
        _plain_json_value("WriteResult.metadata", result.metadata),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    warnings_json = json.dumps(result.warnings, allow_nan=False, separators=(",", ":"))
    return (
        len(operation_id.encode("utf-8"))
        + len(result.state.value)
        + 3 * 8
        + (0 if result.commit_token is None else len(result.commit_token))
        + len(metadata_json.encode("utf-8"))
        + len(warnings_json.encode("utf-8"))
    )


def _results_from_native(operation_id: str, payload: Mapping[str, Any]) -> tuple[WriteResult, ...]:
    native_operation_id = str(payload.get("operation_id") or "")
    if native_operation_id != operation_id:
        raise RuntimeError(
            f"native DataSink operation_id mismatch: expected {operation_id!r}, got {native_operation_id!r}"
        )
    raw_results = payload.get("write_results")
    if not isinstance(raw_results, list):
        raise TypeError("native DataSink result must contain a write_results list")
    if len(raw_results) > _MAX_WRITE_RESULTS:
        raise ValueError(f"native DataSink result exceeds the {_MAX_WRITE_RESULTS} write-result limit")
    results: list[WriteResult] = []
    total_bytes = 0
    for raw_result in raw_results:
        if not isinstance(raw_result, Mapping):
            raise TypeError("native DataSink write result must be a mapping")
        result = _write_result_from_mapping(operation_id, raw_result)
        total_bytes += _write_result_wire_bytes(operation_id, result)
        if total_bytes > _MAX_TOTAL_RESULT_BYTES:
            raise ValueError("native DataSink result exceeds the 64 MiB coordinator payload limit")
        results.append(result)
    return tuple(results)


def _results_from_arrow(operation_id: str, table: pa.Table) -> tuple[WriteResult, ...]:
    import pyarrow as pa

    if not isinstance(table, pa.Table):
        raise TypeError(f"local-fast DataSink execution must return pyarrow.Table, got {type(table).__name__}")
    if table.num_rows > _MAX_WRITE_RESULTS:
        raise ValueError(f"local-fast DataSink result exceeds the {_MAX_WRITE_RESULTS} write-result limit")
    if table.nbytes > _MAX_TOTAL_RESULT_BYTES:
        raise ValueError("local-fast DataSink result exceeds the 64 MiB coordinator payload limit")
    if tuple(table.column_names) != _WIRE_COLUMNS:
        raise RuntimeError(
            f"local-fast DataSink result schema mismatch: expected {_WIRE_COLUMNS!r}, got {tuple(table.column_names)!r}"
        )
    results: list[WriteResult] = []
    for row in table.to_pylist():
        metadata_json = row["metadata_json"]
        if not isinstance(metadata_json, str):
            raise TypeError("DataSink result metadata_json must be a string")
        if len(metadata_json.encode("utf-8")) > _MAX_RESULT_METADATA_BYTES:
            raise ValueError("DataSink result metadata exceeds 64 KiB")
        try:
            metadata = json.loads(metadata_json)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("DataSink result contains invalid metadata JSON") from exc
        warnings_json = row["warnings_json"]
        if not isinstance(warnings_json, str):
            raise TypeError("DataSink result warnings_json must be a string")
        if len(warnings_json.encode("utf-8")) > _MAX_RESULT_METADATA_BYTES:
            raise ValueError("DataSink result warnings exceed 64 KiB")
        try:
            result_warnings = json.loads(warnings_json)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("DataSink result contains invalid warnings JSON") from exc
        results.append(
            _write_result_from_mapping(
                operation_id,
                {
                    "operation_id": row["operation_id"],
                    "state": row["state"],
                    "rows_received": row["rows_received"],
                    "rows_affected": row["rows_affected"],
                    "bytes_received": row["bytes_received"],
                    "commit_token": row["commit_token"],
                    "metadata": metadata,
                    "warnings": result_warnings,
                },
            )
        )
    return tuple(results)


def _summary(
    context: WriteContext,
    outcome: WriteOutcome,
    results: tuple[WriteResult, ...],
    metadata: Mapping[str, Any] | None = None,
    warnings: tuple[str, ...] = (),
) -> WriteSummary:
    affected_values = [result.rows_affected for result in results]
    rows_affected = None if any(value is None for value in affected_values) else sum(affected_values)  # type: ignore[arg-type]
    summary_warnings = warnings
    for result in results:
        for warning in result.warnings:
            summary_warnings = _append_summary_warning(summary_warnings, warning)
    return WriteSummary(
        operation_id=context.operation_id,
        outcome=outcome,
        results=results,
        rows_received=sum(result.rows_received for result in results),
        rows_affected=rows_affected,
        bytes_received=sum(result.bytes_received for result in results),
        metadata={} if metadata is None else metadata,
        warnings=summary_warnings,
    )


def _raise_execution_error(
    sink: BoundDataSink,
    capabilities: DataSinkCapabilities,
    context: WriteContext,
    results: tuple[WriteResult, ...],
    error: BaseException,
    warnings: tuple[str, ...] = (),
) -> None:
    outcome = WriteOutcome.UNKNOWN
    if capabilities.commit_protocol is CommitProtocol.TWO_PHASE and capabilities.supports_abort:
        try:
            sink.abort(context, results, error)
        except BaseException as abort_error:
            if hasattr(error, "add_note"):
                error.add_note(
                    "BoundDataSink.abort() also failed: "
                    f"{type(abort_error).__name__}: {_safe_exception_message(abort_error)}"
                )
        else:
            outcome = WriteOutcome.ABORTED
    summary = _summary(context, outcome, results, warnings=warnings)
    raise DataSinkWriteError(
        summary,
        f"{type(error).__name__}: {_safe_exception_message(error)}",
        safe_to_retry=capabilities.retry_mode is RetryMode.IDEMPOTENT,
    ) from error


def _finalize_results(
    sink: BoundDataSink,
    capabilities: DataSinkCapabilities,
    context: WriteContext,
    results: tuple[WriteResult, ...],
    warnings: tuple[str, ...] = (),
) -> WriteSummary:
    expected_state = (
        WriteState.APPLIED if capabilities.commit_protocol is CommitProtocol.IMMEDIATE else WriteState.PREPARED
    )
    for result in results:
        if result.state is not expected_state:
            _raise_execution_error(
                sink,
                capabilities,
                context,
                results,
                RuntimeError(f"DataSink result state {result.state.value!r} does not match {expected_state.value!r}"),
                warnings,
            )

    if capabilities.commit_protocol is CommitProtocol.IMMEDIATE:
        return _summary(context, WriteOutcome.COMMITTED, results, warnings=warnings)

    try:
        metadata = sink.commit(context, results)
    except BaseException as commit_error:
        if capabilities.supports_reconcile:
            try:
                outcome = WriteOutcome(sink.reconcile(context))
            except BaseException as reconcile_error:
                if hasattr(commit_error, "add_note"):
                    commit_error.add_note(
                        "BoundDataSink.reconcile() also failed: "
                        f"{type(reconcile_error).__name__}: {_safe_exception_message(reconcile_error)}"
                    )
            else:
                if outcome is WriteOutcome.COMMITTED:
                    return _summary(context, WriteOutcome.COMMITTED, results, {"reconciled": True}, warnings)
                if outcome is WriteOutcome.ABORTED:
                    summary = _summary(context, WriteOutcome.ABORTED, results, {"reconciled": True}, warnings)
                    raise DataSinkWriteError(
                        summary,
                        f"commit failed and reconciliation reported aborted: {_safe_exception_message(commit_error)}",
                        safe_to_retry=True,
                    ) from commit_error
        summary = _summary(context, WriteOutcome.UNKNOWN, results, warnings=warnings)
        raise DataSinkWriteError(
            summary,
            f"commit outcome is unknown: {type(commit_error).__name__}: {_safe_exception_message(commit_error)}",
            safe_to_retry=capabilities.retry_mode is RetryMode.IDEMPOTENT,
        ) from commit_error

    try:
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("BoundDataSink.commit() must return a mapping or None")
        normalized_metadata = _json_mapping(
            "BoundDataSink.commit() metadata",
            metadata,
            max_bytes=_MAX_RESULT_METADATA_BYTES,
        )
    except Exception as metadata_error:
        metadata_warnings = _append_summary_warning(
            warnings,
            "DataSink commit succeeded but its metadata was omitted: "
            f"{type(metadata_error).__name__}: {_safe_exception_message(metadata_error)}",
        )
        return _summary(context, WriteOutcome.COMMITTED, results, warnings=metadata_warnings)
    return _summary(context, WriteOutcome.COMMITTED, results, normalized_metadata, warnings)


def write_datasink(
    relation: DuckDBPyRelation,
    sink: DataSink,
    *,
    operation_id: str | None = None,
) -> WriteSummary:
    """Execute ``relation`` into ``sink`` and return its terminal summary.

    Distributed execution may replay a worker batch. Vane therefore accepts
    only sinks that explicitly declare idempotent replay semantics.
    """
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
    if capabilities.retry_mode is not RetryMode.IDEMPOTENT:
        raise ValueError("Vane DataSink execution requires retry_mode='idempotent'")
    execution_options = bound.execution_options
    if not isinstance(execution_options, DataSinkExecutionOptions):
        raise TypeError("BoundDataSink.execution_options must be DataSinkExecutionOptions")
    prepared_relation = bound.prepare_input(relation)
    if not isinstance(prepared_relation, RuntimeDuckDBPyRelation):
        raise TypeError(
            f"BoundDataSink.prepare_input() must return DuckDBPyRelation, got {type(prepared_relation).__name__}"
        )

    batch_function = _make_batch_function(bound, context, capabilities)
    mapped = prepared_relation.map_batches(
        batch_function,
        schema=_wire_output_schema(),
        **execution_options.map_batches_kwargs(),
    )
    terminal = mapped._mark_data_sink(context.operation_id)
    results: tuple[WriteResult, ...] = ()
    warnings: tuple[str, ...] = ()
    try:
        runner_type = get_or_infer_runner_type()
        if runner_type == "local-fast":
            results = _results_from_arrow(context.operation_id, terminal.to_arrow_table())
        else:
            native_result = get_or_create_runner().run_data_sink(terminal)
            if not isinstance(native_result, Mapping):
                raise TypeError(f"Runner.run_data_sink() must return a mapping, got {type(native_result).__name__}")
            warnings = _cleanup_warnings_from_native(native_result)
            results = _results_from_native(context.operation_id, native_result)
    except BaseException as execution_error:
        _raise_execution_error(bound, capabilities, context, results, execution_error, warnings)
    finally:
        del terminal
        del mapped
        del prepared_relation

    return _finalize_results(bound, capabilities, context, results, warnings)


from vane.datasink.turbopuffer import TurbopufferSink  # noqa: E402

__all__ = [
    "BoundDataSink",
    "CommitProtocol",
    "DataSink",
    "DataSinkCapabilities",
    "DataSinkExecutionOptions",
    "DataSinkWriteError",
    "DataSinkWriter",
    "RetryMode",
    "TurbopufferSink",
    "WriteContext",
    "WriteOutcome",
    "WriteResult",
    "WriteState",
    "WriteSummary",
    "write_datasink",
]
