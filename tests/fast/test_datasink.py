# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pickle
import struct
import sys
import threading
import types
import uuid
from base64 import b64decode
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any

import cloudpickle
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import vane
from vane import runners
from vane.datasink import (
    BoundDataSink,
    CommitProtocol,
    DataSink,
    DataSinkCapabilities,
    DataSinkExecutionOptions,
    DataSinkWriteError,
    DataSinkWriter,
    RetryMode,
    TurbopufferSink,
    WriteContext,
    WriteOutcome,
    WriteResult,
    WriteState,
    WriteSummary,
)


class _AppliedWriter(DataSinkWriter):
    def write(self, table: pa.Table) -> WriteResult:
        return WriteResult(
            state=WriteState.APPLIED,
            rows_received=table.num_rows,
            rows_affected=table.num_rows,
            bytes_received=table.nbytes,
            metadata={"writer": "test"},
        )


class _AppliedBoundSink(BoundDataSink):
    @property
    def capabilities(self) -> DataSinkCapabilities:
        return DataSinkCapabilities(CommitProtocol.IMMEDIATE, RetryMode.IDEMPOTENT)

    @property
    def execution_options(self) -> DataSinkExecutionOptions:
        return DataSinkExecutionOptions(batch_size=128, target_max_batch_bytes=8 * 1024 * 1024)

    def open_writer(self, context: WriteContext) -> DataSinkWriter:
        assert context.operation_id
        return _AppliedWriter()


class _AppliedSink(DataSink):
    def bind(self, schema: pa.Schema) -> BoundDataSink:
        assert isinstance(schema, pa.Schema)
        return _AppliedBoundSink()


class _CloseWarningWriter(_AppliedWriter):
    def close(self) -> None:
        raise RuntimeError("injected worker close failure")


class _CloseWarningBoundSink(_AppliedBoundSink):
    def open_writer(self, context: WriteContext) -> DataSinkWriter:
        assert context.operation_id
        return _CloseWarningWriter()


class _CloseWarningSink(DataSink):
    def bind(self, schema: pa.Schema) -> BoundDataSink:
        assert isinstance(schema, pa.Schema)
        return _CloseWarningBoundSink()


def _fail_upstream_batch(table: pa.Table) -> pa.Table:
    del table
    raise RuntimeError("injected upstream worker failure")


class _PreparedWriter(DataSinkWriter):
    def write(self, table: pa.Table) -> WriteResult:
        return WriteResult(
            state=WriteState.PREPARED,
            rows_received=table.num_rows,
            rows_affected=table.num_rows,
            bytes_received=table.nbytes,
            commit_token=f"rows-{table.num_rows}".encode(),
        )


@dataclass
class _TwoPhaseBoundSink(BoundDataSink):
    fail_commit: bool = False
    reconciled_outcome: WriteOutcome | None = None
    commits: list[tuple[WriteResult, ...]] = field(default_factory=list)
    aborts: list[tuple[WriteResult, ...]] = field(default_factory=list)

    @property
    def capabilities(self) -> DataSinkCapabilities:
        return DataSinkCapabilities(
            CommitProtocol.TWO_PHASE,
            RetryMode.IDEMPOTENT,
            supports_abort=True,
            supports_reconcile=self.reconciled_outcome is not None,
        )

    def open_writer(self, context: WriteContext) -> DataSinkWriter:
        assert context.operation_id
        return _PreparedWriter()

    def commit(self, context: WriteContext, results: tuple[WriteResult, ...]) -> dict[str, Any]:
        self.commits.append(results)
        if self.fail_commit:
            raise RuntimeError("injected commit failure")
        return {"committed_tokens": len(results), "operation_id": context.operation_id}

    def abort(
        self,
        context: WriteContext,
        results: tuple[WriteResult, ...],
        error: BaseException,
    ) -> None:
        del context, error
        self.aborts.append(results)

    def reconcile(self, context: WriteContext) -> WriteOutcome:
        del context
        assert self.reconciled_outcome is not None
        return self.reconciled_outcome


class _TwoPhaseSink(DataSink):
    def __init__(self) -> None:
        self.bound = _TwoPhaseBoundSink()

    def bind(self, schema: pa.Schema) -> BoundDataSink:
        assert isinstance(schema, pa.Schema)
        return self.bound


def _prepared_result(rows: int = 3) -> WriteResult:
    return WriteResult(
        state=WriteState.PREPARED,
        rows_received=rows,
        rows_affected=rows,
        bytes_received=rows * 8,
        commit_token=f"token-{rows}".encode(),
    )


def test_write_result_is_strict_immutable_and_json_bounded():
    result = WriteResult(
        state="applied",
        rows_received=2,
        rows_affected=None,
        bytes_received=16,
        metadata={"nested": {"value": 1}},
        warnings=("worker warning",),
    )

    assert result.state is WriteState.APPLIED
    assert dict(result.metadata) == {"nested": {"value": 1}}
    assert result.warnings == ("worker warning",)
    with pytest.raises(TypeError):
        result.metadata["new"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        result.metadata["nested"]["value"] = 2
    with pytest.raises(TypeError, match="rows_received"):
        WriteResult(WriteState.APPLIED, True)
    with pytest.raises(TypeError, match="finite JSON"):
        WriteResult(WriteState.APPLIED, 1, metadata={"bad": float("nan")})
    with pytest.raises(ValueError, match="65536"):
        WriteResult(WriteState.APPLIED, 1, metadata={"too_large": "x" * (64 * 1024)})
    with pytest.raises(ValueError, match="65536"):
        WriteResult(WriteState.PREPARED, 1, commit_token=b"x" * (64 * 1024 + 1))
    with pytest.raises(ValueError, match="at most 4"):
        WriteResult(WriteState.APPLIED, 1, warnings=("warning",) * 5)
    for field_name in ("rows_received", "rows_affected", "bytes_received"):
        values = {"rows_received": 0, field_name: 1 << 64}
        with pytest.raises(ValueError, match="unsigned 64-bit integer"):
            WriteResult(WriteState.APPLIED, **values)
    with pytest.raises(ValueError, match="rows_received"):
        WriteSummary("operation", WriteOutcome.COMMITTED, (result,), 3, None, 16)
    with pytest.raises(ValueError, match="must not be empty"):
        WriteContext("")

    restored = pickle.loads(pickle.dumps(result))
    assert restored == result
    assert restored.metadata == {"nested": {"value": 1}}


def test_datasink_capabilities_reject_unsafe_commit_contracts():
    with pytest.raises(ValueError, match="immediate"):
        DataSinkCapabilities(CommitProtocol.IMMEDIATE, RetryMode.IDEMPOTENT, supports_abort=True)
    with pytest.raises(ValueError, match="operation-scoped abort"):
        DataSinkCapabilities(CommitProtocol.TWO_PHASE, RetryMode.IDEMPOTENT)


def test_datasink_execution_options_reject_invalid_resource_requests():
    assert DataSinkExecutionOptions(batch_size=10, cpus=0.5).map_batches_kwargs() == {
        "batch_size": 10,
        "cpus": 0.5,
    }
    with pytest.raises(ValueError, match="batch_size"):
        DataSinkExecutionOptions(batch_size=0)
    with pytest.raises(ValueError, match="finite"):
        DataSinkExecutionOptions(cpus=float("inf"))


def test_relation_write_sink_local_fast_returns_aggregated_summary(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    relation = vane.sql("SELECT i::BIGINT AS id, ('row-' || i::VARCHAR)::VARCHAR AS text FROM range(257) t(i)")

    summary = relation.write_sink(_AppliedSink(), operation_id="local-fast-operation")

    assert summary.operation_id == "local-fast-operation"
    assert summary.outcome is WriteOutcome.COMMITTED
    assert summary.rows_received == 257
    assert summary.rows_affected == 257
    assert summary.bytes_received > 0
    assert summary.batch_count >= 1
    assert {result.state for result in summary.results} == {WriteState.APPLIED}


def test_relation_write_sink_local_fast_commits_two_phase_results(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    sink = _TwoPhaseSink()

    summary = vane.sql("SELECT i::BIGINT AS id FROM range(17) t(i)").write_sink(
        sink,
        operation_id="local-fast-two-phase",
    )

    assert summary.outcome is WriteOutcome.COMMITTED
    assert summary.rows_received == 17
    assert summary.metadata == {"committed_tokens": summary.batch_count, "operation_id": "local-fast-two-phase"}
    assert sink.bound.commits == [summary.results]


def test_datasink_two_phase_commit_abort_and_reconciliation():
    from vane.datasink import _finalize_results, _raise_execution_error

    context = WriteContext("two-phase-operation")
    results = (_prepared_result(2), _prepared_result(5))
    committed_sink = _TwoPhaseBoundSink()
    summary = _finalize_results(
        committed_sink,
        committed_sink.capabilities,
        context,
        results,
        ("internal fragment cleanup failed",),
    )
    assert summary.outcome is WriteOutcome.COMMITTED
    assert summary.rows_received == 7
    assert summary.metadata == {"committed_tokens": 2, "operation_id": "two-phase-operation"}
    assert summary.warnings == ("internal fragment cleanup failed",)
    assert committed_sink.commits == [results]

    aborted_sink = _TwoPhaseBoundSink()
    with pytest.raises(DataSinkWriteError) as exc_info:
        _raise_execution_error(
            aborted_sink,
            aborted_sink.capabilities,
            context,
            results,
            RuntimeError("worker failed"),
        )
    assert exc_info.value.outcome is WriteOutcome.ABORTED
    assert exc_info.value.safe_to_retry is True
    assert aborted_sink.aborts == [results]

    reconciled_sink = _TwoPhaseBoundSink(fail_commit=True, reconciled_outcome=WriteOutcome.COMMITTED)
    reconciled = _finalize_results(reconciled_sink, reconciled_sink.capabilities, context, results)
    assert reconciled.outcome is WriteOutcome.COMMITTED
    assert reconciled.metadata == {"reconciled": True}


def test_datasink_invalid_commit_metadata_does_not_make_commit_unknown():
    from vane.datasink import _finalize_results

    class _InvalidCommitMetadataSink(_TwoPhaseBoundSink):
        def commit(self, context: WriteContext, results: tuple[WriteResult, ...]) -> dict[str, Any]:
            del context
            self.commits.append(results)
            return {"not_json": object()}

    context = WriteContext("committed-with-invalid-metadata")
    results = (_prepared_result(),)
    sink = _InvalidCommitMetadataSink()

    summary = _finalize_results(sink, sink.capabilities, context, results)

    assert summary.outcome is WriteOutcome.COMMITTED
    assert summary.metadata == {}
    assert "commit succeeded" in summary.warnings[0]
    assert sink.commits == [results]

    restored_summary = pickle.loads(pickle.dumps(summary))
    assert restored_summary == summary
    error = DataSinkWriteError(summary, "committed warning", safe_to_retry=False)
    restored_error = pickle.loads(pickle.dumps(error))
    assert restored_error.summary == summary
    assert restored_error.detail == "committed warning"
    assert restored_error.safe_to_retry is False
    assert str(restored_error) == str(error)


def test_datasink_broken_commit_metadata_does_not_make_commit_unknown():
    from collections.abc import Iterator, Mapping

    from vane.datasink import _finalize_results

    class _UnprintableMetadataError(RuntimeError):
        def __str__(self) -> str:
            raise RuntimeError("planned metadata error __str__ failure")

    class _BrokenMetadata(Mapping[str, Any]):
        def __getitem__(self, key: str) -> Any:
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            raise _UnprintableMetadataError()

        def __len__(self) -> int:
            return 1

    class _BrokenCommitMetadataSink(_TwoPhaseBoundSink):
        def commit(self, context: WriteContext, results: tuple[WriteResult, ...]) -> Mapping[str, Any]:
            del context
            self.commits.append(results)
            return _BrokenMetadata()

    context = WriteContext("committed-with-broken-metadata")
    results = (_prepared_result(),)
    sink = _BrokenCommitMetadataSink()

    summary = _finalize_results(sink, sink.capabilities, context, results)

    assert summary.outcome is WriteOutcome.COMMITTED
    assert summary.metadata == {}
    assert "_UnprintableMetadataError: <error message unavailable>" in summary.warnings[0]
    assert sink.commits == [results]


def test_datasink_unprintable_execution_error_preserves_unknown_outcome():
    from vane.datasink import _raise_execution_error

    class _UnprintableError(RuntimeError):
        def __str__(self) -> str:
            raise RuntimeError("planned __str__ failure")

    sink = _AppliedBoundSink()
    with pytest.raises(DataSinkWriteError) as exc_info:
        _raise_execution_error(
            sink,
            sink.capabilities,
            WriteContext("unprintable-error"),
            (),
            _UnprintableError(),
        )

    assert exc_info.value.outcome is WriteOutcome.UNKNOWN
    assert "_UnprintableError: <error message unavailable>" in str(exc_info.value)


def test_native_datasink_cleanup_warnings_are_bounded():
    from vane.datasink import _cleanup_warnings_from_native

    warnings = _cleanup_warnings_from_native(
        {"data_sink_cleanup_warnings": [" first warning ", "x" * 5000] + [f"warning-{i}" for i in range(20)]}
    )

    assert warnings[0] == "first warning"
    assert len(warnings[1].encode("utf-8")) <= 4096
    assert len(warnings) == 16
    assert warnings[-1] == "additional DataSink cleanup warnings omitted"
    with pytest.raises(TypeError, match="must be a list"):
        _cleanup_warnings_from_native({"data_sink_cleanup_warnings": "not-a-list"})

    malformed_unicode = _cleanup_warnings_from_native(
        {"data_sink_cleanup_warnings": ["cleanup contains a lone surrogate: \ud800"]}
    )
    assert len(malformed_unicode) == 1
    assert malformed_unicode[0].encode("utf-8")


def test_local_fast_rejects_oversized_metadata_before_json_parsing():
    from vane.datasink import _results_from_arrow

    table = pa.table(
        {
            "operation_id": ["bounded-local-result"],
            "state": ["applied"],
            "rows_received": pa.array([1], type=pa.uint64()),
            "rows_affected": pa.array([1], type=pa.uint64()),
            "bytes_received": pa.array([8], type=pa.uint64()),
            "commit_token": pa.array([None], type=pa.binary()),
            "metadata_json": ["x" * (64 * 1024 + 1)],
            "warnings_json": ["[]"],
        }
    )

    with pytest.raises(ValueError, match="metadata exceeds 64 KiB"):
        _results_from_arrow("bounded-local-result", table)


def test_local_fte_datasink_collects_bounded_worker_results(tmp_path, monkeypatch):
    source = tmp_path / "datasink-input.parquet"
    pq.write_table(pa.table({"id": range(1024)}), source, row_group_size=128)
    vane.teardown_runner()
    monkeypatch.setenv("VANE_RUNNER", "local")
    try:
        vane.set_runner_local(num_workers=2, max_running_tasks=2)
        summary = (
            vane.read_parquet(str(source)).repartition(2).write_sink(_AppliedSink(), operation_id="local-fte-operation")
        )
    finally:
        vane.teardown_runner()

    assert summary.outcome is WriteOutcome.COMMITTED
    assert summary.rows_received == 1024
    assert summary.rows_affected == 1024
    assert summary.batch_count > 1


def test_local_fte_datasink_reports_post_write_cleanup_warning(tmp_path, monkeypatch):
    import vane.runners.local.runner as local_runner_module

    source = tmp_path / "datasink-cleanup-warning.parquet"
    pq.write_table(pa.table({"id": range(32)}), source, row_group_size=16)
    original_shutdown = local_runner_module._shutdown_local_write_resources

    def shutdown_with_warning(*args: Any, **kwargs: Any) -> list[BaseException]:
        errors = original_shutdown(*args, **kwargs)
        errors.append(RuntimeError("injected post-write cleanup warning"))
        return errors

    monkeypatch.setattr(local_runner_module, "_shutdown_local_write_resources", shutdown_with_warning)
    vane.teardown_runner()
    monkeypatch.setenv("VANE_RUNNER", "local")
    try:
        vane.set_runner_local(num_workers=1, max_running_tasks=1)
        summary = vane.read_parquet(str(source)).write_sink(_AppliedSink(), operation_id="cleanup-warning")
    finally:
        vane.teardown_runner()

    assert summary.outcome is WriteOutcome.COMMITTED
    assert summary.rows_received == 32
    assert len(summary.warnings) == 1
    assert "injected post-write cleanup warning" in summary.warnings[0]


def test_local_fte_datasink_preserves_success_when_worker_close_fails(tmp_path, monkeypatch):
    source = tmp_path / "datasink-worker-close-warning.parquet"
    pq.write_table(pa.table({"id": range(32)}), source, row_group_size=16)
    vane.teardown_runner()
    monkeypatch.setenv("VANE_RUNNER", "local")
    try:
        vane.set_runner_local(num_workers=1, max_running_tasks=1)
        summary = vane.read_parquet(str(source)).write_sink(_CloseWarningSink(), operation_id="worker-close-warning")
    finally:
        vane.teardown_runner()

    assert summary.outcome is WriteOutcome.COMMITTED
    assert summary.rows_received == 32
    assert len(summary.warnings) >= 1
    assert all("injected worker close failure" in warning for warning in summary.warnings)


def test_local_fte_datasink_reports_progress_update_warning(tmp_path, monkeypatch):
    import vane.runners.local.runner as local_runner_module

    class _FailingProgressRenderer:
        interval_s = 0.1

        def __init__(self, _snapshot_getter: Any) -> None:
            pass

        @staticmethod
        def update(*, force: bool = False) -> None:
            del force
            raise RuntimeError("injected progress update failure")

        @staticmethod
        def finish(*, final_state: str | None) -> None:
            del final_state

    source = tmp_path / "datasink-progress-warning.parquet"
    pq.write_table(pa.table({"id": range(32)}), source, row_group_size=16)
    monkeypatch.setattr(local_runner_module, "ProgressRenderer", _FailingProgressRenderer)
    vane.teardown_runner()
    monkeypatch.setenv("VANE_RUNNER", "local")
    try:
        vane.set_runner_local(num_workers=1, max_running_tasks=1)
        summary = vane.read_parquet(str(source)).write_sink(_AppliedSink(), operation_id="progress-warning")
    finally:
        vane.teardown_runner()

    assert summary.outcome is WriteOutcome.COMMITTED
    assert summary.rows_received == 32
    assert len(summary.warnings) == 1
    assert "injected progress update failure" in summary.warnings[0]


def test_local_fte_datasink_propagates_nested_shuffle_failure(tmp_path, monkeypatch):
    source = tmp_path / "datasink-failure-input.parquet"
    pq.write_table(pa.table({"id": range(128)}), source, row_group_size=64)
    vane.teardown_runner()
    monkeypatch.setenv("VANE_RUNNER", "local")
    try:
        vane.set_runner_local(num_workers=2, max_running_tasks=2)
        with pytest.raises(DataSinkWriteError) as exc_info:
            (
                vane.read_parquet(str(source))
                .map_batches(_fail_upstream_batch, schema={"id": vane.sqltypes.BIGINT})
                .repartition(2)
                .write_sink(_AppliedSink(), operation_id="local-fte-failure")
            )
    finally:
        vane.teardown_runner()

    assert exc_info.value.outcome is WriteOutcome.UNKNOWN
    assert exc_info.value.safe_to_retry is True
    assert exc_info.value.summary.results == ()


@pytest.mark.usefixtures("ray_local")
def test_ray_datasink_round_trips_native_write_results(tmp_path, monkeypatch):
    source = tmp_path / "datasink-ray-input.parquet"
    pq.write_table(pa.table({"id": range(512)}), source, row_group_size=64)
    vane.teardown_runner()
    monkeypatch.setenv("VANE_RUNNER", "ray")
    try:
        runners.set_runner_ray(noop_if_initialized=True)
        summary = vane.read_parquet(str(source)).write_sink(_AppliedSink(), operation_id="ray-operation")
    finally:
        vane.teardown_runner()

    assert summary.outcome is WriteOutcome.COMMITTED
    assert summary.rows_received == 512
    assert summary.rows_affected == 512
    assert summary.batch_count > 1


@pytest.mark.usefixtures("ray_local")
def test_ray_turbopuffer_rejects_duplicates_before_external_requests(tmp_path, monkeypatch):
    source = tmp_path / "turbopuffer-ray-duplicate-ids.parquet"
    pq.write_table(
        pa.table({"id": pa.array([1, 1], type=pa.uint64()), "value": ["first", "conflict"]}),
        source,
        row_group_size=1,
    )
    vane.teardown_runner()
    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.delenv("TURBOPUFFER_API_KEY", raising=False)
    try:
        runners.set_runner_ray(noop_if_initialized=True)
        with pytest.raises(DataSinkWriteError) as exc_info:
            vane.read_parquet(str(source)).write_sink(
                TurbopufferSink(namespace="ray-global-uniqueness", batch_size=1),
                operation_id="ray-cross-batch-duplicates",
            )
    finally:
        vane.teardown_runner()

    assert exc_info.value.outcome is WriteOutcome.UNKNOWN
    assert "globally unique" in str(exc_info.value)
    assert "API key" not in str(exc_info.value)


class _FakeTurbopufferNamespace:
    def __init__(self, client: _FakeTurbopufferClient, name: str) -> None:
        self.client = client
        self.name = name

    def write(self, **request: Any) -> Any:
        self.client.requests.append((self.name, request))
        return SimpleNamespace(
            rows_affected=len(request["upsert_columns"]["id"]),
            rows_upserted=len(request["upsert_columns"]["id"]),
            billing=SimpleNamespace(billable_logical_bytes_written=123),
            performance=SimpleNamespace(server_total_ms=7),
        )


class _FakeTurbopufferClient:
    instances: list[_FakeTurbopufferClient] = []

    def __init__(self, **options: Any) -> None:
        self.options = options
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.closed = False
        self.instances.append(self)

    def namespace(self, name: str) -> _FakeTurbopufferNamespace:
        return _FakeTurbopufferNamespace(self, name)

    def close(self) -> None:
        self.closed = True


def _install_fake_turbopuffer(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    import vane.datasink.turbopuffer as implementation

    module = types.ModuleType("turbopuffer")
    module.Turbopuffer = _FakeTurbopufferClient
    lib_module = types.ModuleType("turbopuffer.lib")
    json_module = types.ModuleType("turbopuffer.lib.json")
    json_module.dumps = lambda value: json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    lib_module.json = json_module
    module.lib = lib_module
    _FakeTurbopufferClient.instances.clear()
    monkeypatch.setitem(sys.modules, "turbopuffer", module)
    monkeypatch.setitem(sys.modules, "turbopuffer.lib", lib_module)
    monkeypatch.setitem(sys.modules, "turbopuffer.lib.json", json_module)
    monkeypatch.setattr(implementation, "_CLIENTS", threading.local())
    return module


def _turbopuffer_worker_table(table: pa.Table, *, id_counts: list[int] | None = None) -> pa.Table:
    from vane.datasink.turbopuffer import _ID_COUNT_COLUMN

    counts = [1] * table.num_rows if id_counts is None else id_counts
    return table.append_column(_ID_COUNT_COLUMN, pa.array(counts, type=pa.uint64()))


def test_turbopuffer_sink_uses_columnar_idempotent_upsert_without_serializing_secret(monkeypatch):
    _install_fake_turbopuffer(monkeypatch)
    monkeypatch.setenv("TP_TEST_API_KEY", "top-secret-api-key")
    sink = TurbopufferSink(
        namespace="documents-v1",
        id_column="document_id",
        distance_metric="cosine_distance",
        schema={"text": {"type": "string", "filterable": True}},
        disable_backpressure=True,
        api_key_env="TP_TEST_API_KEY",
        batch_size=64,
    )
    table = pa.table(
        {
            "document_id": pa.array([11, 12], type=pa.uint64()),
            "vector": pa.array([[0.1, 0.2], [0.3, 0.4]], type=pa.list_(pa.float32(), 2)),
            "text": ["first", "second"],
        }
    )

    bound = sink.bind(table.schema)
    serialized = cloudpickle.dumps(bound)
    assert b"top-secret-api-key" not in serialized
    result = bound.open_writer(WriteContext("tp-operation")).write(_turbopuffer_worker_table(table))

    client = _FakeTurbopufferClient.instances[0]
    assert client.options == {
        "api_key": "top-secret-api-key",
        "region": "gcp-us-central1",
        "max_retries": 2,
        "timeout": 60.0,
        "compression": False,
    }
    namespace, request = client.requests[0]
    assert namespace == "documents-v1"
    assert request["upsert_columns"]["id"] == [11, 12]
    assert struct.unpack("<2f", b64decode(request["upsert_columns"]["vector"][0])) == pytest.approx((0.1, 0.2))
    assert struct.unpack("<2f", b64decode(request["upsert_columns"]["vector"][1])) == pytest.approx((0.3, 0.4))
    assert request["upsert_columns"]["text"] == ["first", "second"]
    assert request["distance_metric"] == "cosine_distance"
    assert request["schema"] == {
        "id": "uint",
        "text": {"type": "string", "filterable": True},
        "vector": {"ann": True, "type": "[2]f32"},
    }
    assert request["disable_backpressure"] is True
    assert result.rows_received == 2
    assert result.rows_affected == 2
    assert result.metadata == {
        "billable_logical_bytes_written": 123,
        "namespace": "documents-v1",
        "provider": "turbopuffer",
        "request_bytes": result.metadata["request_bytes"],
        "rows_upserted": 2,
        "server_total_ms": 7,
    }
    assert result.metadata["request_bytes"] > 0


def test_turbopuffer_sink_rejects_invalid_schema_ids_and_missing_worker_secret(monkeypatch):
    sink = TurbopufferSink(namespace="valid", api_key_env="TP_MISSING_KEY")
    with pytest.raises(ValueError, match="requires ID column"):
        sink.bind(pa.schema([("value", pa.string())]))
    with pytest.raises(TypeError, match="must be an integer, string, or UUID"):
        sink.bind(pa.schema([("id", pa.float64())]))

    bound = sink.bind(pa.schema([("id", pa.uint64()), ("value", pa.string())]))
    monkeypatch.delenv("TP_MISSING_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TP_MISSING_KEY"):
        bound.open_writer(WriteContext("missing-secret")).write(
            _turbopuffer_worker_table(pa.table({"id": [1], "value": ["x"]}))
        )

    _install_fake_turbopuffer(monkeypatch)
    monkeypatch.setenv("TP_MISSING_KEY", "secret")
    writer = bound.open_writer(WriteContext("invalid-id"))
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        writer.write(_turbopuffer_worker_table(pa.table({"id": [-1], "value": ["x"]})))


def test_turbopuffer_sink_normalizes_uuid_and_temporal_values(monkeypatch):
    _install_fake_turbopuffer(monkeypatch)
    monkeypatch.setenv("TP_TYPED_KEY", "secret")
    document_id = uuid.uuid4()
    owner_id = uuid.uuid4()
    table = pa.table(
        {
            "id": pa.array([document_id], type=pa.uuid()),
            "owner_id": pa.array([owner_id], type=pa.uuid()),
            "created_at": pa.array([datetime(2026, 8, 10, 12, 30, tzinfo=timezone.utc)]),
            "published_on": pa.array([date(2026, 8, 10)], type=pa.date32()),
        }
    )
    sink = TurbopufferSink(namespace="typed-values", api_key_env="TP_TYPED_KEY")

    result = sink.bind(table.schema).open_writer(WriteContext("typed-values")).write(_turbopuffer_worker_table(table))

    _, request = _FakeTurbopufferClient.instances[0].requests[0]
    assert request["upsert_columns"] == {
        "id": [str(document_id)],
        "owner_id": [str(owner_id)],
        "created_at": ["2026-08-10T12:30:00+00:00"],
        "published_on": ["2026-08-10"],
    }
    assert request["schema"] == {
        "created_at": "datetime",
        "id": "uuid",
        "owner_id": "uuid",
        "published_on": "datetime",
    }
    assert result.rows_affected == 1


def test_turbopuffer_sink_rejects_duplicate_ids_and_non_finite_values():
    sink = TurbopufferSink(namespace="validated-values")
    duplicate_bound = sink.bind(pa.schema([("id", pa.uint64()), ("value", pa.string())]))
    with pytest.raises(ValueError, match="globally unique"):
        duplicate_bound.open_writer(WriteContext("duplicate-ids")).write(
            _turbopuffer_worker_table(
                pa.table({"id": pa.array([1, 1], type=pa.uint64()), "value": ["first", "second"]}),
                id_counts=[2, 2],
            )
        )

    float_bound = sink.bind(pa.schema([("id", pa.uint64()), ("score", pa.float64())]))
    with pytest.raises(ValueError, match="non-finite"):
        float_bound.open_writer(WriteContext("non-finite")).write(
            _turbopuffer_worker_table(pa.table({"id": pa.array([1], type=pa.uint64()), "score": [float("nan")]}))
        )


def test_turbopuffer_sink_rejects_cross_batch_duplicate_ids_before_upsert(monkeypatch):
    _install_fake_turbopuffer(monkeypatch)
    monkeypatch.setenv("TP_UNIQUE_KEY", "secret")
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    sink = TurbopufferSink(
        namespace="global-uniqueness",
        api_key_env="TP_UNIQUE_KEY",
        batch_size=1,
    )
    relation = vane.from_arrow(
        pa.table(
            {
                "id": pa.array([1, 2, 1], type=pa.uint64()),
                "value": ["first", "unique", "conflict"],
            }
        )
    )

    with pytest.raises(DataSinkWriteError, match="globally unique"):
        relation.write_sink(sink, operation_id="cross-batch-duplicates")

    written_ids = [
        item
        for client in _FakeTurbopufferClient.instances
        for _, request in client.requests
        for item in request["upsert_columns"]["id"]
    ]
    assert 1 not in written_ids


def test_turbopuffer_sink_requires_and_validates_deterministic_vector_dimensions(monkeypatch):
    _install_fake_turbopuffer(monkeypatch)
    monkeypatch.setenv("TP_VECTOR_KEY", "secret")
    variable_schema = pa.schema([("id", pa.uint64()), ("vector", pa.list_(pa.float32()))])
    with pytest.raises(ValueError, match="require an explicit schema mapping"):
        TurbopufferSink(namespace="variable-vector").bind(variable_schema)

    sink = TurbopufferSink(
        namespace="explicit-vector",
        api_key_env="TP_VECTOR_KEY",
        distance_metric="cosine_distance",
        schema={"vector": {"type": "[2]f32", "ann": True}},
    )
    bound = sink.bind(variable_schema)
    writer = bound.open_writer(WriteContext("explicit-vector"))
    writer.write(
        _turbopuffer_worker_table(
            pa.table(
                {
                    "id": pa.array([1], type=pa.uint64()),
                    "vector": pa.array([[0.1, 0.2]], type=pa.list_(pa.float32())),
                }
            )
        )
    )
    with pytest.raises(ValueError, match="exactly 2 values"):
        writer.write(
            _turbopuffer_worker_table(
                pa.table(
                    {
                        "id": pa.array([2], type=pa.uint64()),
                        "vector": pa.array([[0.1, 0.2, 0.3]], type=pa.list_(pa.float32())),
                    }
                )
            )
        )

    encoded_bound = sink.bind(pa.schema([("id", pa.uint64()), ("vector", pa.string())]))
    with pytest.raises(ValueError, match="not valid base64"):
        encoded_bound.open_writer(WriteContext("invalid-base64")).write(
            _turbopuffer_worker_table(pa.table({"id": pa.array([3], type=pa.uint64()), "vector": ["!!!!!!!!!!!!"]}))
        )

    multi_vector_sink = TurbopufferSink(
        namespace="multiple-vectors",
        api_key_env="TP_VECTOR_KEY",
        distance_metric="cosine_distance",
        schema={
            "vector": {"type": "[2]f32", "ann": True},
            "image_vector": {"type": "[2]f16", "ann": True},
        },
    )
    multi_vector_table = pa.table(
        {
            "id": pa.array([4], type=pa.uint64()),
            "vector": pa.array([[0.1, 0.2]], type=pa.list_(pa.float32())),
            "image_vector": pa.array([[0.3, 0.4]], type=pa.list_(pa.float32())),
        }
    )
    multi_vector_sink.bind(multi_vector_table.schema).open_writer(WriteContext("multiple-vectors")).write(
        _turbopuffer_worker_table(multi_vector_table)
    )
    _, multi_vector_request = _FakeTurbopufferClient.instances[0].requests[-1]
    assert struct.unpack("<2f", b64decode(multi_vector_request["upsert_columns"]["image_vector"][0])) == pytest.approx(
        (0.3, 0.4)
    )

    int8_table = pa.table(
        {
            "id": pa.array([5], type=pa.uint64()),
            "vector": pa.array([[1, -2]], type=pa.list_(pa.int8(), 2)),
        }
    )
    int8_sink = TurbopufferSink(
        namespace="int8-vector",
        api_key_env="TP_VECTOR_KEY",
        distance_metric="cosine_distance",
        schema={"vector": {"type": "[2]i8", "ann": True}},
    )
    int8_sink.bind(int8_table.schema).open_writer(WriteContext("int8-vector")).write(
        _turbopuffer_worker_table(int8_table)
    )
    _, int8_request = _FakeTurbopufferClient.instances[0].requests[-1]
    assert struct.unpack("<2f", b64decode(int8_request["upsert_columns"]["vector"][0])) == pytest.approx((1.0, -2.0))

    int8_writer = int8_sink.bind(pa.schema([("id", pa.uint64()), ("vector", pa.list_(pa.float32(), 2))])).open_writer(
        WriteContext("invalid-int8-vector")
    )
    for invalid_vector in ([128.0, 0.0], [1.5, 0.0]):
        with pytest.raises(ValueError, match="integers from -128 through 127"):
            int8_writer.write(
                _turbopuffer_worker_table(
                    pa.table(
                        {
                            "id": pa.array([6], type=pa.uint64()),
                            "vector": pa.array([invalid_vector], type=pa.list_(pa.float32(), 2)),
                        }
                    )
                )
            )

    float16_sink = TurbopufferSink(
        namespace="float16-vector",
        api_key_env="TP_VECTOR_KEY",
        distance_metric="cosine_distance",
        schema={"vector": {"type": "[2]f16", "ann": True}},
    )
    with pytest.raises(ValueError, match="f16 vector .* out-of-range"):
        float16_sink.bind(pa.schema([("id", pa.uint64()), ("vector", pa.list_(pa.float64(), 2))])).open_writer(
            WriteContext("invalid-float16-vector")
        ).write(
            _turbopuffer_worker_table(
                pa.table(
                    {
                        "id": pa.array([7], type=pa.uint64()),
                        "vector": pa.array([[70_000.0, 0.0]], type=pa.list_(pa.float64(), 2)),
                    }
                )
            )
        )


def test_turbopuffer_sink_enforces_serialized_request_size(monkeypatch):
    _install_fake_turbopuffer(monkeypatch)
    monkeypatch.setenv("TP_SIZE_KEY", "secret")
    sink = TurbopufferSink(namespace="request-size", api_key_env="TP_SIZE_KEY", max_batch_bytes=1024)
    table = pa.table({"id": pa.array([1], type=pa.uint64()), "text": ["\x00" * 300]})
    assert table.nbytes < 1024

    with pytest.raises(ValueError, match="serialized request"):
        sink.bind(table.schema).open_writer(WriteContext("request-size")).write(_turbopuffer_worker_table(table))
    assert _FakeTurbopufferClient.instances[0].requests == []


def test_turbopuffer_sink_preserves_success_when_response_statistics_are_malformed(monkeypatch):
    _install_fake_turbopuffer(monkeypatch)
    monkeypatch.setenv("TP_RESPONSE_KEY", "secret")

    def write_with_malformed_statistics(namespace: _FakeTurbopufferNamespace, **request: Any) -> Any:
        namespace.client.requests.append((namespace.name, request))
        return SimpleNamespace(
            rows_upserted="not-an-integer",
            rows_patched="not-an-integer",
            rows_deleted="not-an-integer",
            billing=SimpleNamespace(billable_logical_bytes_written="not-an-integer"),
            performance=SimpleNamespace(server_total_ms="not-an-integer"),
        )

    monkeypatch.setattr(_FakeTurbopufferNamespace, "write", write_with_malformed_statistics)
    table = pa.table({"id": pa.array([1], type=pa.uint64()), "text": ["written"]})
    result = (
        TurbopufferSink(namespace="response-statistics", api_key_env="TP_RESPONSE_KEY")
        .bind(table.schema)
        .open_writer(WriteContext("response-statistics"))
        .write(_turbopuffer_worker_table(table))
    )

    assert len(_FakeTurbopufferClient.instances[0].requests) == 1
    assert result.state is WriteState.APPLIED
    assert result.rows_received == 1
    assert result.rows_affected is None
    assert result.metadata["namespace"] == "response-statistics"
    assert len(result.warnings) == 4
    assert "rows_affected" in result.warnings[0]
    assert "rows_upserted" in result.warnings[1]
    assert result.warnings[2] == "additional worker warnings omitted"


def test_turbopuffer_sink_enforces_request_limit_and_namespace_contract():
    with pytest.raises(ValueError, match="namespace"):
        TurbopufferSink(namespace="contains spaces")
    with pytest.raises(ValueError, match="512 MiB"):
        TurbopufferSink(namespace="valid", max_batch_bytes=512 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="at most 1024 attributes"):
        TurbopufferSink(namespace="valid", schema={f"attribute_{index}": "string" for index in range(1025)})
    with pytest.raises(ValueError, match="schema type for 'id' must be 'uint'"):
        TurbopufferSink(namespace="valid", schema={"id": "int"}).bind(pa.schema([("id", pa.int64())]))
    with pytest.raises(ValueError, match="distance_metric is required"):
        TurbopufferSink(namespace="valid").bind(pa.schema([("id", pa.uint64()), ("vector", pa.list_(pa.float32(), 2))]))
    with pytest.raises(ValueError, match="ann=True"):
        TurbopufferSink(
            namespace="valid",
            distance_metric="cosine_distance",
            schema={"vector": "[2]f32"},
        ).bind(pa.schema([("id", pa.uint64()), ("vector", pa.list_(pa.float32(), 2))]))
    with pytest.raises(ValueError, match="multi-vector attribute .* is not supported"):
        TurbopufferSink(
            namespace="valid",
            distance_metric="cosine_distance",
            schema={"vector": {"type": "[][2]f32", "ann": True}},
        ).bind(pa.schema([("id", pa.uint64()), ("vector", pa.list_(pa.list_(pa.float32(), 2)))]))
    with pytest.raises(ValueError, match="at most two dense vector attributes"):
        TurbopufferSink(
            namespace="valid",
            distance_metric="cosine_distance",
            schema={
                "first": {"type": "[2]f32", "ann": True},
                "second": {"type": "[2]f32", "ann": True},
                "third": {"type": "[2]f32", "ann": True},
            },
        ).bind(pa.schema([("id", pa.uint64())]))
    embedded = TurbopufferSink(
        namespace="valid",
        distance_metric="cosine_distance",
        schema={"text": {"type": "string", "embed": "multilingual-e5-large"}},
        batch_size=10_000,
    ).bind(pa.schema([("id", pa.uint64()), ("text", pa.string())]))
    assert embedded.execution_options.batch_size == 30
    with pytest.raises(TypeError, match="nested arrays"):
        TurbopufferSink(namespace="valid").bind(
            pa.schema([("id", pa.uint64()), ("nested", pa.list_(pa.list_(pa.int64())))])
        )
    with pytest.raises(TypeError, match="numeric arrays"):
        TurbopufferSink(namespace="valid").bind(pa.schema([("id", pa.uint64()), ("vector", pa.list_(pa.string()))]))
