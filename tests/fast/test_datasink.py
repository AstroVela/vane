# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import threading
import time
from pathlib import Path

import cloudpickle
import pyarrow as pa
import pytest

import vane
from vane.datasink import (
    BoundDataSink,
    BoundKeyedUpsertSink,
    DataSink,
    DataSinkExecutionOptions,
    DataSinkWorker,
    DataSinkWriteError,
    EnvironmentSecret,
    WriteContext,
    WriteOutcome,
    WriteResult,
    WriteState,
)


class _Worker(DataSinkWorker):
    def __init__(self, *, fail: bool = False, fail_close: bool = False, reject_open: bool = False) -> None:
        if reject_open:
            raise AssertionError("worker must not open for rejected keyed input")
        self._fail = fail
        self._fail_close = fail_close

    def write(self, table: pa.Table) -> WriteResult:
        if self._fail:
            raise RuntimeError("planned worker failure")
        return WriteResult(
            rows_received=table.num_rows,
            rows_affected=table.num_rows,
            bytes_received=table.nbytes,
            metadata={"columns": table.num_columns},
        )

    def close(self) -> None:
        if self._fail_close:
            raise RuntimeError("planned close failure")


class _Bound(BoundDataSink):
    def __init__(
        self,
        *,
        fail: bool = False,
        fail_close: bool = False,
        options: DataSinkExecutionOptions | None = None,
    ) -> None:
        self._fail = fail
        self._fail_close = fail_close
        self._options = options or DataSinkExecutionOptions(batch_size=2)

    @property
    def execution_options(self) -> DataSinkExecutionOptions:
        return self._options

    def open_worker(self, context: WriteContext) -> DataSinkWorker:
        assert context.operation_id
        return _Worker(fail=self._fail, fail_close=self._fail_close)


class _Sink(DataSink):
    def __init__(self, bound: BoundDataSink | None = None) -> None:
        self._bound = bound or _Bound()

    def bind(self, schema: pa.Schema) -> BoundDataSink:
        assert isinstance(schema, pa.Schema)
        return self._bound


class _KeyedBound(_Bound, BoundKeyedUpsertSink):
    def __init__(self, *, reject_open: bool = False) -> None:
        super().__init__(options=DataSinkExecutionOptions(batch_size=1))
        self._reject_open = reject_open

    @property
    def key_columns(self) -> tuple[str, ...]:
        return ("id",)

    def open_worker(self, context: WriteContext) -> DataSinkWorker:
        return _Worker(reject_open=self._reject_open)


class _SecretBound(_Bound):
    def __init__(self, secret: EnvironmentSecret) -> None:
        super().__init__()
        self.secret = secret


class _UnserializableBound(_Bound):
    def __init__(self) -> None:
        super().__init__()
        self.lock = threading.Lock()


class _ReplayWorker(DataSinkWorker):
    def __init__(self, store: dict[tuple[str, int], int], context: WriteContext) -> None:
        self._store = store
        self._context = context

    def write(self, table: pa.Table) -> WriteResult:
        for row in table.to_pylist():
            self._store[(self._context.operation_id, row["id"])] = row["value"]
        return WriteResult(rows_received=table.num_rows, rows_affected=table.num_rows)


class _ReplayBound(_Bound):
    def __init__(self) -> None:
        super().__init__()
        self.store: dict[tuple[str, int], int] = {}

    def open_worker(self, context: WriteContext) -> DataSinkWorker:
        return _ReplayWorker(self.store, context)


class _TrackingWorker(DataSinkWorker):
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def write(self, table: pa.Table) -> WriteResult:
        self._calls.append("write")
        raise RuntimeError("planned tracking failure")

    def abort(self, error: BaseException) -> None:
        assert isinstance(error, RuntimeError)
        self._calls.append("abort")

    def close(self) -> None:
        self._calls.append("close")


class _TrackingBound(_Bound):
    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self._calls = calls

    def open_worker(self, context: WriteContext) -> DataSinkWorker:
        self._calls.append("open")
        return _TrackingWorker(self._calls)


class _SlowWorker(_Worker):
    def write(self, table: pa.Table) -> WriteResult:
        time.sleep(0.05)
        return super().write(table)


class _SlowBound(_Bound):
    def open_worker(self, context: WriteContext) -> DataSinkWorker:
        return _SlowWorker()


class _TimeoutWorker(DataSinkWorker):
    def write(self, table: pa.Table) -> WriteResult:
        raise TimeoutError("planned provider timeout")


class _TimeoutBound(_Bound):
    def open_worker(self, context: WriteContext) -> DataSinkWorker:
        return _TimeoutWorker()


class _CloseMarkerWorker(_Worker):
    def __init__(self, marker_path: str) -> None:
        super().__init__()
        self._marker_path = marker_path

    def close(self) -> None:
        Path(self._marker_path).write_text("closed", encoding="utf-8")


class _CloseMarkerBound(_Bound):
    def __init__(self, marker_path: Path) -> None:
        super().__init__()
        self._marker_path = str(marker_path)

    def open_worker(self, context: WriteContext) -> DataSinkWorker:
        return _CloseMarkerWorker(self._marker_path)


class _BlockingCloseWorker(_Worker):
    def close(self) -> None:
        time.sleep(60)


class _BlockingCloseBound(_Bound):
    def open_worker(self, context: WriteContext) -> DataSinkWorker:
        return _BlockingCloseWorker()


class _OversizedCloseWorker(_Worker):
    def close(self) -> None:
        raise RuntimeError("cleanup-head:" + "x" * 100_000 + ":cleanup-tail")


class _OversizedCloseBound(_Bound):
    def open_worker(self, context: WriteContext) -> DataSinkWorker:
        return _OversizedCloseWorker()


class _FailingWorkerWithCloseMarker(DataSinkWorker):
    def __init__(self, marker_path: str) -> None:
        self._marker_path = marker_path

    def write(self, table: pa.Table) -> WriteResult:
        raise RuntimeError("planned marked worker failure")

    def abort(self, error: BaseException) -> None:
        Path(self._marker_path).write_text("abort\n", encoding="utf-8")

    def close(self) -> None:
        with Path(self._marker_path).open("a", encoding="utf-8") as marker:
            marker.write("close\n")


class _FailingBoundWithCloseMarker(_Bound):
    def __init__(self, marker_path: Path) -> None:
        super().__init__()
        self._marker_path = str(marker_path)

    def open_worker(self, context: WriteContext) -> DataSinkWorker:
        return _FailingWorkerWithCloseMarker(self._marker_path)


class _AppendThenFailOnceWorker(DataSinkWorker):
    def __init__(self, append_path: str, failure_marker_path: str, context: WriteContext) -> None:
        self._append_path = append_path
        self._failure_marker_path = failure_marker_path
        self._operation_id = context.operation_id

    def write(self, table: pa.Table) -> WriteResult:
        with Path(self._append_path).open("a", encoding="utf-8") as output:
            for row in table.to_pylist():
                output.write(f"{self._operation_id}:{row['id']}\n")
        failure_marker = Path(self._failure_marker_path)
        if not failure_marker.exists():
            failure_marker.write_text("failed", encoding="utf-8")
            raise RuntimeError("planned response loss after append")
        return WriteResult(rows_received=table.num_rows, rows_affected=table.num_rows, bytes_received=table.nbytes)


class _AppendThenFailOnceBound(_Bound):
    def __init__(self, append_path: Path, failure_marker_path: Path, *, max_retries: int) -> None:
        super().__init__(
            options=DataSinkExecutionOptions(
                worker_count=1,
                max_retries=max_retries,
                batch_size=1,
            )
        )
        self._append_path = str(append_path)
        self._failure_marker_path = str(failure_marker_path)

    def open_worker(self, context: WriteContext) -> DataSinkWorker:
        return _AppendThenFailOnceWorker(self._append_path, self._failure_marker_path, context)


class _CountingBound(_Bound):
    def __init__(self, options: DataSinkExecutionOptions) -> None:
        super().__init__(options=options)
        self.opens = 0
        self.writes = 0
        self.closes = 0

    def open_worker(self, context: WriteContext) -> DataSinkWorker:
        self.opens += 1
        owner = self

        class _CountingWorker(_Worker):
            def write(self, table: pa.Table) -> WriteResult:
                owner.writes += 1
                return super().write(table)

            def close(self) -> None:
                owner.closes += 1
                super().close()

        return _CountingWorker()


class _SchemaSink(DataSink):
    def __init__(self) -> None:
        self.schema: pa.Schema | None = None

    def bind(self, schema: pa.Schema) -> BoundDataSink:
        self.schema = schema
        return _Bound()


def _native_result(
    operation_id: str,
    *,
    outcome_unknown: bool = False,
    outcome_aborted: bool = False,
) -> dict[str, object]:
    state = "aborted" if outcome_aborted else "applied"
    return {
        "operation_id": operation_id,
        "outcome_aborted": outcome_aborted,
        "outcome_unknown": outcome_unknown,
        "outcome_error": "planned unknown outcome" if outcome_unknown else "",
        "write_results": [
            {
                "operation_id": operation_id,
                "state": state,
                "rows_received": 2,
                "rows_affected": 0 if outcome_aborted else 2,
                "bytes_received": 0 if outcome_aborted else 16,
                "metadata": {},
                "warnings": [],
            }
        ],
        "data_sink_cleanup_warnings": ["planned cleanup warning"],
    }


def test_local_fast_datasink_applies_aggregates_and_closes_worker(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    relation = vane.sql("SELECT i::INTEGER AS id FROM range(0, 5) t(i)")
    close_marker = tmp_path / "local-fast-worker-closed"

    summary = relation.write_datasink(
        _Sink(_CloseMarkerBound(close_marker)),
        operation_id="local-fast-applied",
    )

    assert summary.operation_id == "local-fast-applied"
    assert summary.outcome is WriteOutcome.APPLIED
    assert summary.rows_received == 5
    assert summary.rows_affected == 5
    assert summary.batch_count >= 1
    assert close_marker.read_text(encoding="utf-8") == "closed"


def test_datasink_rejects_source_explicit_transaction_before_binding():
    connection = vane.connect()
    relation = connection.sql("SELECT 1 AS id")
    sink = _SchemaSink()
    connection.execute("BEGIN")
    try:
        with pytest.raises(vane.InvalidInputException, match="cannot participate in an explicit transaction"):
            relation.write_datasink(sink, operation_id="explicit-transaction-datasink")
    finally:
        connection.execute("ROLLBACK")

    assert sink.schema is None


def test_datasink_terminal_rechecks_transaction_when_it_is_bound():
    connection = vane.connect()
    terminal = connection.sql("SELECT 1 AS id")._mark_datasink("late-explicit-transaction-datasink")
    connection.execute("BEGIN")
    try:
        with pytest.raises(vane.InvalidInputException, match="cannot participate in an explicit transaction"):
            terminal.to_arrow_table()
    finally:
        connection.execute("ROLLBACK")


def test_datasink_rechecks_transaction_on_prepared_relation(monkeypatch):
    from vane import runners

    source = vane.connect().sql("SELECT 1 AS id")
    prepared_connection = vane.connect()
    prepared = prepared_connection.sql("SELECT 2 AS id")

    class _PreparedRelationBound(_Bound):
        def prepare_input(self, relation):
            return prepared

    monkeypatch.setattr(cloudpickle, "dumps", lambda _value: b"serialized")
    monkeypatch.setattr(
        runners,
        "get_or_infer_runner_type",
        lambda: (_ for _ in ()).throw(AssertionError("runner selection must not run")),
    )
    prepared_connection.execute("BEGIN")
    try:
        with pytest.raises(vane.InvalidInputException, match="cannot participate in an explicit transaction"):
            source.write_datasink(
                _Sink(_PreparedRelationBound()),
                operation_id="prepared-explicit-transaction-datasink",
            )
    finally:
        prepared_connection.execute("ROLLBACK")


def test_local_fast_datasink_worker_failure_closes_after_abort(monkeypatch, tmp_path):
    from vane.execution.udf_subprocess import LocalSubprocessActorPool

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    marker = tmp_path / "local-fast-failed-worker-cleanup"
    shutdown_kill_values: list[bool] = []
    original_shutdown = LocalSubprocessActorPool.shutdown

    def tracked_shutdown(self, *, kill=False):
        if not self._closed:
            shutdown_kill_values.append(bool(kill))
        return original_shutdown(self, kill=kill)

    monkeypatch.setattr(LocalSubprocessActorPool, "shutdown", tracked_shutdown)

    with pytest.raises(DataSinkWriteError) as exc_info:
        vane.sql("SELECT 1 AS id").write_datasink(
            _Sink(_FailingBoundWithCloseMarker(marker)),
            operation_id="local-fast-worker-failure",
        )

    assert exc_info.value.outcome is WriteOutcome.UNKNOWN
    assert shutdown_kill_values == [False]
    assert marker.read_text(encoding="utf-8") == "abort\nclose\n"


def test_local_fast_datasink_close_failure_is_cleanup_warning(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")

    summary = vane.sql("SELECT 1 AS id").write_datasink(
        _Sink(_Bound(fail_close=True)),
        operation_id="local-fast-close-failure",
    )

    assert summary.outcome is WriteOutcome.APPLIED
    assert any("planned close failure" in warning for warning in summary.warnings)


def test_local_fast_datasink_close_timeout_is_cleanup_warning(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    monkeypatch.setenv("VANE_UDF_SUBPROCESS_SHUTDOWN_GRACE_S", "0.02")

    summary = vane.sql("SELECT 1 AS id").write_datasink(
        _Sink(_BlockingCloseBound()),
        operation_id="local-fast-close-timeout",
    )

    assert summary.outcome is WriteOutcome.APPLIED
    assert any("graceful shutdown timed out" in warning for warning in summary.warnings)


def test_local_fast_datasink_bounds_native_cleanup_warning(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")

    summary = vane.sql("SELECT 1 AS id").write_datasink(
        _Sink(_OversizedCloseBound()),
        operation_id="local-fast-bounded-close-warning",
    )

    assert summary.outcome is WriteOutcome.APPLIED
    assert len(summary.warnings) == 1
    assert len(summary.warnings[0].encode("utf-8")) <= 4 * 1024
    assert "error text exceeds 4096 bytes and was omitted" in summary.warnings[0]


def test_local_fast_empty_input_does_not_open_a_worker(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    relation = vane.sql("SELECT 1 AS id WHERE false")

    summary = relation.write_datasink(_Sink(_KeyedBound(reject_open=True)), operation_id="empty")

    assert summary.outcome is WriteOutcome.APPLIED
    assert summary.results == ()
    assert summary.rows_received == 0
    assert summary.rows_affected == 0


def test_datasink_terminal_rejects_invalid_empty_wire_schema():
    terminal = vane.sql("SELECT 1 AS invalid_wire_column WHERE false")._mark_datasink("invalid-empty-schema")

    with pytest.raises(vane.InvalidInputException, match="DataSink worker result schema"):
        terminal.to_arrow_table()


def test_worker_close_failure_retains_worker_for_cleanup_retry():
    from vane.datasink import _SinkBatchRuntime

    calls: list[str] = []

    class _RetryCloseWorker(DataSinkWorker):
        def write(self, table: pa.Table) -> WriteResult:
            return WriteResult(rows_received=table.num_rows, rows_affected=table.num_rows)

        def close(self) -> None:
            calls.append("close")
            if len(calls) == 1:
                raise RuntimeError("planned close failure")

    class _RetryCloseBound(_Bound):
        def open_worker(self, context: WriteContext) -> DataSinkWorker:
            return _RetryCloseWorker()

    runtime = _SinkBatchRuntime(_RetryCloseBound(), WriteContext("close-failure"), None)
    runtime(pa.table({"id": [1]}))

    with pytest.raises(RuntimeError, match="planned close failure"):
        runtime.close()
    with pytest.raises(RuntimeError, match="cannot be reused"):
        runtime(pa.table({"id": [2]}))

    runtime.close()
    runtime.close()

    assert calls == ["close", "close"]


def test_worker_abort_diagnostic_cannot_mask_original_failure():
    from vane.datasink import _SinkBatchRuntime

    class _UnnotableWriteError(RuntimeError):
        @property
        def add_note(self):
            raise RuntimeError("planned add_note lookup failure")

    primary_error = _UnnotableWriteError("planned primary write failure")

    class _AbortFailureWorker(DataSinkWorker):
        def write(self, table: pa.Table) -> WriteResult:
            raise primary_error

        def abort(self, error: BaseException) -> None:
            assert error is primary_error
            raise RuntimeError("planned abort failure")

    class _AbortFailureBound(_Bound):
        def open_worker(self, context: WriteContext) -> DataSinkWorker:
            return _AbortFailureWorker()

    runtime = _SinkBatchRuntime(_AbortFailureBound(), WriteContext("abort-note-failure"), None)

    with pytest.raises(_UnnotableWriteError) as exc_info:
        runtime(pa.table({"id": [1]}))

    assert exc_info.value is primary_error


def test_worker_result_publication_failure_aborts_and_poison_actor(monkeypatch):
    from vane import datasink as datasink_module
    from vane.datasink import _SinkBatchRuntime

    calls: list[str] = []

    class _PublicationWorker(DataSinkWorker):
        def write(self, table: pa.Table) -> WriteResult:
            calls.append("write")
            return WriteResult(rows_received=table.num_rows, rows_affected=table.num_rows)

        def abort(self, error: BaseException) -> None:
            assert isinstance(error, MemoryError)
            calls.append("abort")

    class _PublicationBound(_Bound):
        def open_worker(self, context: WriteContext) -> DataSinkWorker:
            calls.append("open")
            return _PublicationWorker()

    runtime = _SinkBatchRuntime(_PublicationBound(), WriteContext("publication-failure"), None)
    monkeypatch.setattr(
        datasink_module,
        "_result_to_wire_table",
        lambda *_args: (_ for _ in ()).throw(MemoryError("planned Arrow publication failure")),
    )

    with pytest.raises(MemoryError, match="planned Arrow publication failure"):
        runtime(pa.table({"id": [1]}))
    with pytest.raises(RuntimeError, match="cannot be reused"):
        runtime(pa.table({"id": [2]}))

    assert calls == ["open", "write", "abort"]


def test_actor_input_boundary_failure_aborts_an_already_open_worker():
    from vane.datasink import _SinkBatchRuntime

    calls: list[str] = []

    class _InputBoundaryWorker(DataSinkWorker):
        def write(self, table: pa.Table) -> WriteResult:
            calls.append("write")
            return WriteResult(rows_received=table.num_rows, rows_affected=table.num_rows)

        def abort(self, error: BaseException) -> None:
            assert isinstance(error, TypeError)
            calls.append("abort")

        def close(self) -> None:
            calls.append("close")

    class _InputBoundaryBound(_Bound):
        def open_worker(self, context: WriteContext) -> DataSinkWorker:
            calls.append("open")
            return _InputBoundaryWorker()

    runtime = _SinkBatchRuntime(_InputBoundaryBound(), WriteContext("input-boundary-failure"), None)
    runtime(pa.table({"id": [1]}))

    with pytest.raises(TypeError, match="expected pyarrow.Table"):
        runtime(object())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="cannot be reused"):
        runtime(pa.table({"id": [2]}))
    runtime.close()

    assert calls == ["open", "write", "abort", "close"]


def test_worker_failure_is_unknown_without_retry_safety_claim(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")

    with pytest.raises(DataSinkWriteError) as exc_info:
        vane.sql("SELECT 1 AS id").write_datasink(_Sink(_Bound(fail=True)), operation_id="worker-failure")

    assert exc_info.value.outcome is WriteOutcome.UNKNOWN
    assert not hasattr(exc_info.value, "safe_to_retry")
    assert exc_info.value.summary.results == ()


def test_non_idempotent_append_is_not_retried_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    append_path = tmp_path / "default-no-retry.log"
    failure_marker = tmp_path / "default-no-retry.failed"

    with pytest.raises(DataSinkWriteError) as exc_info:
        vane.sql("SELECT 7 AS id").write_datasink(
            _Sink(_AppendThenFailOnceBound(append_path, failure_marker, max_retries=0)),
            operation_id="append-default-no-retry",
        )

    assert exc_info.value.outcome is WriteOutcome.UNKNOWN
    assert append_path.read_text(encoding="utf-8").splitlines() == ["append-default-no-retry:7"]


def test_configured_retry_replays_full_non_idempotent_append(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    append_path = tmp_path / "configured-retry.log"
    failure_marker = tmp_path / "configured-retry.failed"

    summary = vane.sql("SELECT 7 AS id").write_datasink(
        _Sink(_AppendThenFailOnceBound(append_path, failure_marker, max_retries=1)),
    )

    assert summary.outcome is WriteOutcome.APPLIED
    assert append_path.read_text(encoding="utf-8").splitlines() == [
        f"{summary.operation_id}:7",
        f"{summary.operation_id}:7",
    ]
    assert summary.warnings[0].startswith("DataSink made 1 framework retry attempt")
    assert "may have applied external writes" in summary.warnings[0]


def test_unserializable_bound_sink_fails_before_execution(monkeypatch):
    from vane import runners

    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: (_ for _ in ()).throw(AssertionError()))
    with pytest.raises(TypeError, match="cloudpickle-serializable"):
        vane.sql("SELECT 1 AS id").write_datasink(_Sink(_UnserializableBound()))


def test_environment_secret_serializes_only_reference(monkeypatch):
    secret_value = "datasink-secret-value-that-must-not-serialize"
    monkeypatch.setenv("TEST_DATASINK_TOKEN", secret_value)
    bound = _SecretBound(EnvironmentSecret("TEST_DATASINK_TOKEN"))

    payload = cloudpickle.dumps(bound)
    restored = cloudpickle.loads(payload)

    assert secret_value.encode() not in payload
    assert restored.secret.resolve() == secret_value
    assert "TEST_DATASINK_TOKEN" in repr(restored.secret)
    assert secret_value not in repr(restored.secret)


def test_datasink_text_values_reject_invalid_utf8():
    with pytest.raises(ValueError, match="valid UTF-8"):
        WriteContext("operation-\ud800")
    with pytest.raises(ValueError, match="valid UTF-8"):
        WriteResult(rows_received=1, warnings=("warning-\ud800",))
    with pytest.raises(ValueError, match="valid UTF-8"):
        EnvironmentSecret("SECRET_\ud800")


def test_datasink_has_no_framework_delivery_capability_api():
    assert not hasattr(vane, "CommitProtocol")
    assert not hasattr(vane, "RetryMode")
    assert not hasattr(vane, "DataSinkCapabilities")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"memory_bytes": -1},
        {"batch_size": 1 << 63},
        {"cpus": float("nan")},
        {"cpus": 1 << 1024},
        {"gpus": -0.1},
    ],
)
def test_execution_options_reject_invalid_resource_requests(kwargs):
    with pytest.raises((TypeError, ValueError)):
        DataSinkExecutionOptions(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"worker_count": 0},
        {"worker_count": True},
        {"worker_count": 1.5},
        {"worker_count": 1 << 63},
        {"max_retries": -1},
        {"max_retries": True},
        {"max_retries": 1.5},
        {"max_retries": 1 << 63},
    ],
)
def test_execution_options_reject_invalid_control_requests(kwargs):
    with pytest.raises((TypeError, ValueError)):
        DataSinkExecutionOptions(**kwargs)


def test_execution_options_map_to_actor_backends():
    options = DataSinkExecutionOptions(worker_count=3, batch_size=20, cpus=1, gpus=0)

    assert options.map_batches_kwargs("ray") == {
        "batch_size": 20,
        "cpus": 1.0,
        "gpus": 0.0,
        "execution_backend": "ray_actor",
        "actor_number": 3,
    }
    assert options.map_batches_kwargs("local") == {
        "batch_size": 20,
        "cpus": 1.0,
        "gpus": 0.0,
        "execution_backend": "subprocess_actor",
        "actor_number": 3,
    }
    assert type(options.cpus) is float
    assert type(options.gpus) is float
    assert DataSinkExecutionOptions().worker_count == 1
    assert DataSinkExecutionOptions().max_retries == 0
    with pytest.raises(ValueError, match="unsupported DataSink runner"):
        options.map_batches_kwargs("unsupported")


def test_execution_options_retry_budget_is_keyword_only():
    options = DataSinkExecutionOptions(3, 20)

    assert options.worker_count == 3
    assert options.batch_size == 20
    assert options.max_retries == 0
    with pytest.raises(TypeError):
        DataSinkExecutionOptions(3, 20, None, None, None, None, None, 1)


def test_datasink_actor_payload_disables_ray_task_replay():
    from vane import datasink as datasink_module

    connection = vane.connect()
    options = DataSinkExecutionOptions()
    actor_type = datasink_module._make_batch_actor(_Bound(options=options), WriteContext("payload-retries"), None)
    assert actor_type._vane_datasink_no_task_retries is True
    mapped = connection.sql("SELECT 1 AS id").map_batches(
        actor_type,
        schema=datasink_module._wire_output_schema(),
        **options.map_batches_kwargs("ray"),
    )
    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(mapped, "datasink-payload-retries")
    physical = logical.to_physical_plan(connection)
    payload = physical.collect_udf_nodes(conn=connection)[0]["payload"]

    assert payload["max_task_retries"] == 0


def test_datasink_actor_reuses_and_closes_worker_after_cloudpickle_round_trip():
    from vane.datasink import _make_batch_actor

    context = WriteContext("worker-lifecycle")
    actor_bound = _CountingBound(DataSinkExecutionOptions(worker_count=1))
    actor_type = _make_batch_actor(actor_bound, context, None)
    assert inspect.isclass(actor_type)
    assert not inspect.signature(actor_type).parameters

    restored_type = cloudpickle.loads(cloudpickle.dumps(actor_type))
    actor = restored_type()
    first = actor(pa.table({"id": [1]}))
    second = actor(pa.table({"id": [2]}))

    assert first.num_rows == second.num_rows == 1
    assert actor._runtime._sink.opens == 1
    assert actor._runtime._sink.writes == 2
    actor._vane_close()
    actor._vane_close()
    assert actor._runtime._sink.closes == 1
    with pytest.raises(RuntimeError, match="closed"):
        actor(pa.table({"id": [3]}))


def test_local_fast_datasink_respects_actor_pool_size(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    options = DataSinkExecutionOptions(worker_count=2, batch_size=1)

    summary = vane.sql("SELECT i::INTEGER AS id FROM range(0, 4) t(i)").write_datasink(
        _Sink(_Bound(options=options)),
        operation_id="local-fast-actor",
    )

    assert summary.outcome is WriteOutcome.APPLIED
    assert summary.rows_received == 4
    assert summary.batch_count == 4


def test_keyed_duplicate_validation_aborts_before_worker_open(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    relation = vane.sql("SELECT * FROM (VALUES (1, 'a'), (1, 'b')) t(id, value)")

    with pytest.raises(DataSinkWriteError) as exc_info:
        relation.write_datasink(_Sink(_KeyedBound(reject_open=True)), operation_id="duplicate-keys")

    assert exc_info.value.outcome is WriteOutcome.ABORTED
    assert exc_info.value.summary.rows_received == 2
    assert {result.state for result in exc_info.value.summary.results} == {WriteState.ABORTED}


def test_keyed_null_validation_aborts_before_worker_open(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    relation = vane.sql("SELECT NULL::INTEGER AS id, 'a' AS value")

    with pytest.raises(DataSinkWriteError) as exc_info:
        relation.write_datasink(_Sink(_KeyedBound(reject_open=True)), operation_id="null-key")

    assert exc_info.value.outcome is WriteOutcome.ABORTED


def test_keyed_unique_input_is_applied(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    relation = vane.sql("SELECT * FROM (VALUES (1, 'a'), (2, 'b')) t(id, value)")

    summary = relation.write_datasink(_Sink(_KeyedBound()), operation_id="unique-keys")

    assert summary.outcome is WriteOutcome.APPLIED
    assert summary.rows_received == 2


def test_keyed_column_names_preserve_significant_whitespace(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")

    class _WhitespaceKeyBound(_KeyedBound):
        @property
        def key_columns(self) -> tuple[str, ...]:
            return (" id ",)

    relation = vane.sql("SELECT 1 AS \" id \", 'value' AS payload")

    summary = relation.write_datasink(
        _Sink(_WhitespaceKeyBound()),
        operation_id="whitespace-key-column",
    )

    assert summary.outcome is WriteOutcome.APPLIED
    assert summary.rows_received == 1


def test_keyed_validation_projects_the_resolved_input_column_name(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")

    class _CasefoldedKeyBound(_KeyedBound):
        @property
        def key_columns(self) -> tuple[str, ...]:
            return ("SS",)

    relation = vane.sql("SELECT 1 AS \"ß\", 'value' AS payload")

    summary = relation.write_datasink(
        _Sink(_CasefoldedKeyBound()),
        operation_id="casefolded-key-column",
    )

    assert summary.outcome is WriteOutcome.APPLIED
    assert summary.rows_received == 1


def test_replayed_and_losing_attempts_converge_for_idempotent_worker():
    from vane.datasink import _results_from_arrow, _SinkBatchRuntime

    bound = _ReplayBound()
    context = WriteContext("replayed-operation")
    runtime = _SinkBatchRuntime(bound, context, None)
    table = pa.table({"id": [1, 2], "value": [10, 20]})

    first = runtime(table)
    losing_replay = runtime(table)
    selected_results = _results_from_arrow(context.operation_id, first)
    runtime.close()

    assert first.num_rows == losing_replay.num_rows == 1
    assert bound.store == {("replayed-operation", 1): 10, ("replayed-operation", 2): 20}
    assert len(selected_results) == 1
    assert selected_results[0].rows_received == 2


def test_worker_failure_aborts_then_closes_during_actor_teardown():
    from vane.datasink import _SinkBatchRuntime

    calls: list[str] = []
    runtime = _SinkBatchRuntime(_TrackingBound(calls), WriteContext("cleanup"), None)

    with pytest.raises(RuntimeError, match="planned tracking failure"):
        runtime(pa.table({"id": [1]}))

    assert calls == ["open", "write", "abort"]
    with pytest.raises(RuntimeError, match="cannot be reused"):
        runtime(pa.table({"id": [2]}))
    runtime.close()
    runtime.close()
    assert calls == ["open", "write", "abort", "close"]


def test_mock_distributed_result_uses_only_selected_results(monkeypatch):
    from vane import runners

    operation_id = "selected-results"

    class FakeRunner:
        def run_datasink(self, relation):
            assert relation.type == "EXTENSION_RELATION"
            return _native_result(operation_id)

    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "ray")
    monkeypatch.setattr(runners, "get_or_create_runner", lambda: FakeRunner())

    summary = vane.sql("SELECT * FROM range(0, 2)").write_datasink(_Sink(), operation_id=operation_id)

    assert summary.outcome is WriteOutcome.APPLIED
    assert summary.batch_count == 1
    assert summary.rows_received == 2
    assert summary.warnings == ("planned cleanup warning",)


@pytest.mark.parametrize("outcome_aborted", [False, True])
def test_malformed_cleanup_warnings_do_not_change_known_outcome(monkeypatch, outcome_aborted):
    from vane import runners

    operation_id = "malformed-cleanup-warnings"
    native_result = _native_result(operation_id, outcome_aborted=outcome_aborted)
    native_result["data_sink_cleanup_warnings"] = object()

    class FakeRunner:
        def run_datasink(self, relation):
            return native_result

    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "ray")
    monkeypatch.setattr(runners, "get_or_create_runner", lambda: FakeRunner())

    if outcome_aborted:
        with pytest.raises(DataSinkWriteError) as exc_info:
            vane.sql("SELECT * FROM range(0, 2)").write_datasink(_Sink(), operation_id=operation_id)
        assert exc_info.value.outcome is WriteOutcome.ABORTED
        summary = exc_info.value.summary
    else:
        summary = vane.sql("SELECT * FROM range(0, 2)").write_datasink(_Sink(), operation_id=operation_id)
        assert summary.outcome is WriteOutcome.APPLIED

    assert summary.warnings == ("DataSink cleanup diagnostics were malformed and ignored",)


@pytest.mark.parametrize(
    ("query", "expected_type"),
    [
        ("SELECT 1.23::DECIMAL(10, 2) AS value", pa.decimal128(10, 2)),
        ("SELECT UUID '00000000-0000-0000-0000-000000000001' AS value", pa.string()),
        ("SELECT TIMETZ '12:34:56+08' AS value", pa.time64("us")),
        ("SELECT 'a'::ENUM('a', 'b') AS value", pa.dictionary(pa.uint8(), pa.string())),
    ],
)
def test_datasink_bind_uses_complete_native_arrow_schema(monkeypatch, query, expected_type):
    from vane import runners

    operation_id = "complete-arrow-schema"

    class FakeRunner:
        def run_datasink(self, relation):
            return _native_result(operation_id)

    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "ray")
    monkeypatch.setattr(runners, "get_or_create_runner", lambda: FakeRunner())
    sink = _SchemaSink()

    vane.sql(query).write_datasink(sink, operation_id=operation_id)

    assert sink.schema is not None
    assert sink.schema.field("value").type == expected_type


def test_mock_distributed_unknown_preserves_partial_results(monkeypatch):
    from vane import runners

    operation_id = "unknown-results"

    class FakeRunner:
        def run_datasink(self, relation):
            return _native_result(operation_id, outcome_unknown=True)

    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "ray")
    monkeypatch.setattr(runners, "get_or_create_runner", lambda: FakeRunner())

    with pytest.raises(DataSinkWriteError) as exc_info:
        vane.sql("SELECT * FROM range(0, 2)").write_datasink(_Sink(), operation_id=operation_id)

    assert exc_info.value.outcome is WriteOutcome.UNKNOWN
    assert exc_info.value.summary.rows_received == 2
    assert exc_info.value.summary.batch_count == 1


def test_mock_distributed_retry_budget_is_exact(monkeypatch):
    from vane import runners

    operation_id = "unknown-retry-budget"

    class FakeRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run_datasink(self, relation):
            self.calls += 1
            return _native_result(operation_id, outcome_unknown=True)

    runner = FakeRunner()
    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "ray")
    monkeypatch.setattr(runners, "get_or_create_runner", lambda: runner)

    with pytest.raises(DataSinkWriteError) as exc_info:
        vane.sql("SELECT * FROM range(0, 2)").write_datasink(
            _Sink(_Bound(options=DataSinkExecutionOptions(max_retries=2))),
            operation_id=operation_id,
        )

    assert runner.calls == 3
    assert exc_info.value.outcome is WriteOutcome.UNKNOWN
    assert exc_info.value.summary.warnings[0].startswith("DataSink made 2 framework retry attempts")


def test_mock_distributed_aborted_outcome_is_not_retried(monkeypatch):
    from vane import runners

    operation_id = "aborted-no-retry"

    class FakeRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run_datasink(self, relation):
            self.calls += 1
            return _native_result(operation_id, outcome_aborted=True)

    runner = FakeRunner()
    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "ray")
    monkeypatch.setattr(runners, "get_or_create_runner", lambda: runner)

    with pytest.raises(DataSinkWriteError) as exc_info:
        vane.sql("SELECT * FROM range(0, 2)").write_datasink(
            _Sink(_Bound(options=DataSinkExecutionOptions(max_retries=3))),
            operation_id=operation_id,
        )

    assert runner.calls == 1
    assert exc_info.value.outcome is WriteOutcome.ABORTED


def test_mock_distributed_aborted_retry_after_unknown_remains_unknown(monkeypatch):
    from vane import runners

    operation_id = "unknown-before-aborted-retry"

    class FakeRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run_datasink(self, relation):
            self.calls += 1
            return _native_result(
                operation_id,
                outcome_unknown=self.calls == 1,
                outcome_aborted=self.calls == 2,
            )

    runner = FakeRunner()
    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "ray")
    monkeypatch.setattr(runners, "get_or_create_runner", lambda: runner)

    with pytest.raises(DataSinkWriteError) as exc_info:
        vane.sql("SELECT * FROM range(0, 2)").write_datasink(
            _Sink(_Bound(options=DataSinkExecutionOptions(max_retries=1))),
            operation_id=operation_id,
        )

    assert runner.calls == 2
    assert exc_info.value.outcome is WriteOutcome.UNKNOWN
    assert "an earlier attempt had an UNKNOWN outcome; final attempt was aborted" in exc_info.value.detail


def test_local_wire_limit_excludes_arrow_container_overhead(monkeypatch):
    from vane import datasink as datasink_module

    context = WriteContext("wire-size")
    result = WriteResult(rows_received=1, rows_affected=1)
    table = datasink_module._result_to_wire_table(context, result)
    wire_bytes = datasink_module._result_wire_bytes(context.operation_id, result)
    assert table.nbytes > wire_bytes
    monkeypatch.setattr(datasink_module, "_MAX_TOTAL_RESULT_BYTES", wire_bytes)

    assert datasink_module._results_from_arrow(context.operation_id, table) == (result,)


@pytest.mark.parametrize(
    "native_result",
    [
        _native_result("outcome-mismatch", outcome_aborted=True) | {"outcome_aborted": False},
        _native_result("outcome-mismatch") | {"outcome_aborted": True},
    ],
)
def test_mock_distributed_known_outcome_rejects_mismatched_worker_states(monkeypatch, native_result):
    from vane import runners

    class FakeRunner:
        def run_datasink(self, relation):
            return native_result

    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "ray")
    monkeypatch.setattr(runners, "get_or_create_runner", lambda: FakeRunner())

    with pytest.raises(DataSinkWriteError) as exc_info:
        vane.sql("SELECT * FROM range(0, 2)").write_datasink(
            _Sink(),
            operation_id="outcome-mismatch",
        )

    assert exc_info.value.outcome is WriteOutcome.UNKNOWN


def test_result_and_error_round_trip_through_cloudpickle():
    result = WriteResult(rows_received=1, rows_affected=1, metadata={"nested": [1, 2]})
    summary = vane.datasink._summary(WriteContext("pickle"), WriteOutcome.UNKNOWN, (result,))
    original = DataSinkWriteError(summary, "unknown")

    restored = cloudpickle.loads(cloudpickle.dumps(original))

    assert restored.outcome is WriteOutcome.UNKNOWN
    assert restored.summary.metadata == summary.metadata
    assert restored.summary.results[0].metadata == result.metadata


def test_result_and_summary_json_boundaries_are_immutable():
    result = WriteResult(rows_received=1, rows_affected=1, metadata={"nested": [1, 2]})
    summary = vane.WriteSummary(
        operation_id="immutable",
        outcome=WriteOutcome.APPLIED,
        results=(result,),
        rows_received=1,
        rows_affected=1,
        bytes_received=0,
        metadata={"provider": {"request_ids": ["one"]}},
    )

    with pytest.raises(TypeError):
        result.metadata["other"] = True
    with pytest.raises(TypeError):
        result.metadata["nested"][0] = 3
    with pytest.raises(TypeError):
        summary.metadata["provider"]["request_ids"][0] = "two"


def test_write_result_rejects_unbounded_metadata_and_warnings():
    with pytest.raises(ValueError, match="64 KiB"):
        WriteResult(rows_received=1, metadata={"value": "x" * (64 * 1024)})
    with pytest.raises(ValueError, match="at most four"):
        WriteResult(rows_received=1, warnings=("a", "b", "c", "d", "e"))
    with pytest.raises(ValueError, match="64 KiB"):
        WriteResult(rows_received=1, warnings=("\x01" * (4 * 1024),) * 4)
    with pytest.raises(ValueError, match="zero rows_affected"):
        WriteResult(rows_received=1, rows_affected=1, state=WriteState.ABORTED)
    with pytest.raises(TypeError, match="sequence of strings"):
        WriteResult(rows_received=1, warnings=(warning for warning in ("never-consumed",)))


def test_write_result_rejects_oversized_metadata_before_encoding():
    class _GuardedLargeText(str):
        def encode(self, *_args, **_kwargs):
            raise AssertionError("oversized metadata must be rejected before encoding")

    with pytest.raises(ValueError, match="64 KiB"):
        WriteResult(rows_received=1, metadata={"value": _GuardedLargeText("x" * 100_000)})


def test_write_result_rejects_oversized_integer_before_encoding():
    from vane import datasink as datasink_module

    oversized_integer = 1 << (datasink_module._MAX_RESULT_METADATA_INTEGER_BITS + 1)

    with pytest.raises(ValueError, match="64 KiB"):
        WriteResult(rows_received=1, metadata={"value": oversized_integer})


def test_summary_aggregates_are_bounded_by_result_count_not_uint64():
    result = WriteResult(
        rows_received=(1 << 64) - 1,
        rows_affected=(1 << 64) - 1,
        bytes_received=(1 << 64) - 1,
    )

    summary = vane.datasink._summary(WriteContext("large-summary"), WriteOutcome.APPLIED, (result, result))

    assert summary.rows_received == 2 * ((1 << 64) - 1)
    with pytest.raises(TypeError, match="sequence of strings"):
        vane.WriteSummary(
            operation_id="bad-warnings",
            outcome=WriteOutcome.APPLIED,
            results=(),
            rows_received=0,
            rows_affected=0,
            bytes_received=0,
            warnings="warning",
        )


def test_summary_enforces_the_aggregate_wire_payload_limit(monkeypatch):
    from vane import datasink as datasink_module

    result = WriteResult(rows_received=1, rows_affected=1, metadata={"value": "payload"})
    wire_bytes = datasink_module._result_wire_bytes("bounded-summary", result)
    monkeypatch.setattr(datasink_module, "_MAX_TOTAL_RESULT_BYTES", wire_bytes - 1)

    with pytest.raises(ValueError, match="64 MiB coordinator payload limit"):
        vane.WriteSummary(
            operation_id="bounded-summary",
            outcome=WriteOutcome.APPLIED,
            results=(result,),
            rows_received=1,
            rows_affected=1,
            bytes_received=0,
        )

    with pytest.raises(TypeError, match="sequence of WriteResult"):
        vane.WriteSummary(
            operation_id="bounded-summary",
            outcome=WriteOutcome.APPLIED,
            results=(item for item in (result,)),
            rows_received=1,
            rows_affected=1,
            bytes_received=0,
        )


@pytest.mark.parametrize(
    "outcome, results, match",
    [
        (
            WriteOutcome.APPLIED,
            (WriteResult(rows_received=1, rows_affected=0, state=WriteState.ABORTED),),
            "applied WriteSummary",
        ),
        (WriteOutcome.ABORTED, (), "aborted WriteSummary"),
        (
            WriteOutcome.ABORTED,
            (WriteResult(rows_received=1, rows_affected=1),),
            "aborted WriteSummary",
        ),
    ],
)
def test_summary_rejects_outcome_and_result_state_mismatches(outcome, results, match):
    with pytest.raises(ValueError, match=match):
        vane.WriteSummary(
            operation_id="summary-state-mismatch",
            outcome=outcome,
            results=results,
            rows_received=sum(result.rows_received for result in results),
            rows_affected=sum(result.rows_affected for result in results),
            bytes_received=sum(result.bytes_received for result in results),
        )


def test_cleanup_warnings_and_outcome_detail_are_bounded():
    from vane.datasink import _cleanup_warnings

    warnings = _cleanup_warnings({"data_sink_cleanup_warnings": ["x" * 10_000] * 20})
    assert len(warnings) == 16
    assert all(len(warning.encode("utf-8")) <= 4 * 1024 for warning in warnings)
    assert warnings[-1] == "additional DataSink warnings omitted"

    literal_sentinel = vane.datasink._summary(
        WriteContext("literal-warning"),
        WriteOutcome.APPLIED,
        (
            WriteResult(
                rows_received=1,
                rows_affected=1,
                warnings=("additional DataSink warnings omitted", "later warning"),
            ),
        ),
    )
    assert literal_sentinel.warnings == ("additional DataSink warnings omitted", "later warning")

    summary = vane.datasink._summary(WriteContext("bounded-detail"), WriteOutcome.UNKNOWN, ())
    error = DataSinkWriteError(summary, "detail-head:" + "x" * 10_000 + ":provider-root-cause")
    assert len(error.detail.encode("utf-8")) <= 4 * 1024
    assert error.detail.startswith("detail-head:")
    assert error.detail.endswith(":provider-root-cause")


def test_cleanup_warning_bounds_input_before_normalization():
    class _GuardedLargeText(str):
        def strip(self, *_args, **_kwargs):
            raise AssertionError("the unbounded input must be sliced before strip")

        def encode(self, *_args, **_kwargs):
            raise AssertionError("the unbounded input must be sliced before encode")

    warning = _GuardedLargeText("warning-head:" + "x" * 100_000 + ":warning-tail")

    bounded = vane.datasink._bounded_warning(warning)

    assert len(bounded.encode("utf-8")) <= 4 * 1024
    assert bounded.startswith("warning-head:")
    assert bounded.endswith(":warning-tail")


def test_diagnostic_text_honors_the_minimum_utf8_byte_limit():
    from vane.execution._diagnostics import bounded_utf8_text

    assert bounded_utf8_text("oversized", 3) == "…"


def test_datasink_error_summary_bounds_oversized_exception_type_name():
    class OversizedTypeNameError(RuntimeError):
        pass

    OversizedTypeNameError.__name__ = "x" * 100_000

    summary = vane.datasink._safe_error_summary(OversizedTypeNameError("planned provider failure"))

    assert summary == "BaseException: planned provider failure"
    assert len(summary.encode("utf-8")) <= 4 * 1024


def test_datasink_error_summary_does_not_invoke_exception_string_conversion():
    class UnprintableError(RuntimeError):
        def __str__(self):
            raise AssertionError("provider exception string conversion must not run")

    summary = vane.datasink._safe_error_summary(UnprintableError("safe provider detail"))

    assert summary == "UnprintableError: safe provider detail"


def test_local_cleanup_warning_batch_bounds_count_and_exception_type_name():
    from vane.runners.local.runner import (
        _DATASINK_CLEANUP_WARNING_LIMIT,
        _DATASINK_CLEANUP_WARNINGS_OMITTED,
        _datasink_cleanup_warning_batch,
    )

    class OversizedTypeNameError(RuntimeError):
        pass

    OversizedTypeNameError.__name__ = "x" * 100_000
    errors = [OversizedTypeNameError(f"cleanup-{index}") for index in range(100)]

    warnings = _datasink_cleanup_warning_batch("DataSink resource shutdown", errors)

    assert len(warnings) == _DATASINK_CLEANUP_WARNING_LIMIT
    assert warnings[0] == "DataSink resource shutdown failed: BaseException: cleanup-0"
    assert warnings[-1] == _DATASINK_CLEANUP_WARNINGS_OMITTED
    assert all(len(warning.encode("utf-8")) <= 4 * 1024 for warning in warnings)


def test_write_result_rejects_oversized_warning_before_normalization():
    class _GuardedLargeText(str):
        def strip(self, *_args, **_kwargs):
            raise AssertionError("oversized warnings must be rejected before strip")

        def encode(self, *_args, **_kwargs):
            raise AssertionError("oversized warnings must be rejected before encode")

    with pytest.raises(ValueError, match="at most 4 KiB"):
        WriteResult(rows_received=0, warnings=(_GuardedLargeText("x" * 100_000),))


def test_result_mapping_rejects_warning_count_before_copying():
    class _GuardedWarnings(list):
        def __iter__(self):
            raise AssertionError("oversized warnings must be rejected before copying")

    payload = {
        "operation_id": "bounded-result-warnings",
        "state": "applied",
        "rows_received": 0,
        "rows_affected": 0,
        "bytes_received": 0,
        "metadata": {},
        "warnings": _GuardedWarnings(["warning"] * 5),
    }

    with pytest.raises(ValueError, match="at most four"):
        vane.datasink._write_result_from_mapping("bounded-result-warnings", payload)


def test_result_mapping_rejects_non_string_state():
    payload = {
        "operation_id": "strict-result-state",
        "state": 1,
        "rows_received": 0,
        "rows_affected": 0,
        "bytes_received": 0,
        "metadata": {},
        "warnings": [],
    }

    with pytest.raises(TypeError, match="state must be a string"):
        vane.datasink._write_result_from_mapping("strict-result-state", payload)


def test_local_fte_datasink(monkeypatch):
    from vane import runners
    from vane.runners.local.runner import LocalRunner

    runner = LocalRunner(num_workers=1)
    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "local")
    monkeypatch.setattr(runners, "get_or_create_runner", lambda: runner)

    summary = vane.sql("SELECT * FROM range(0, 3)").write_datasink(_Sink(), operation_id="local-fte")

    assert summary.outcome is WriteOutcome.APPLIED
    assert summary.rows_received == 3


def test_local_fte_datasink_worker_failure_is_unknown_and_closes_after_abort(monkeypatch, tmp_path):
    from vane import runners
    from vane.runners.local.runner import LocalRunner

    runner = LocalRunner(num_workers=1)
    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "local")
    monkeypatch.setattr(runners, "get_or_create_runner", lambda: runner)
    marker = tmp_path / "local-failed-worker-cleanup"

    with pytest.raises(DataSinkWriteError) as exc_info:
        vane.sql("SELECT 1 AS id").write_datasink(
            _Sink(_FailingBoundWithCloseMarker(marker)),
            operation_id="local-fte-failure",
        )

    assert exc_info.value.outcome is WriteOutcome.UNKNOWN
    assert marker.read_text(encoding="utf-8") == "abort\nclose\n"


def test_local_fte_datasink_cleanup_failure_is_warning(monkeypatch):
    from vane import runners
    from vane.runners.local import runner as local_runner_module
    from vane.runners.local.runner import LocalRunner

    original_shutdown = local_runner_module._shutdown_local_write_resources

    def shutdown_with_warning(*args, **kwargs):
        errors = original_shutdown(*args, **kwargs)
        errors.append(RuntimeError("cleanup-head:" + "x" * 10_000 + ":cleanup-root-cause"))
        return errors

    monkeypatch.setattr(local_runner_module, "_shutdown_local_write_resources", shutdown_with_warning)
    runner = LocalRunner(num_workers=1)
    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "local")
    monkeypatch.setattr(runners, "get_or_create_runner", lambda: runner)

    summary = vane.sql("SELECT 1 AS id").write_datasink(_Sink(), operation_id="local-cleanup-warning")

    assert summary.outcome is WriteOutcome.APPLIED
    assert len(summary.warnings) == 1
    assert len(summary.warnings[0].encode("utf-8")) <= 4 * 1024
    assert "cleanup-head:" in summary.warnings[0]
    assert summary.warnings[0].endswith(":cleanup-root-cause")


def test_local_fte_datasink_close_timeout_is_cleanup_warning(monkeypatch):
    from vane import runners
    from vane.runners.local.runner import LocalRunner

    monkeypatch.setenv("VANE_UDF_SUBPROCESS_SHUTDOWN_GRACE_S", "0.02")
    runner = LocalRunner(num_workers=1)
    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "local")
    monkeypatch.setattr(runners, "get_or_create_runner", lambda: runner)

    summary = vane.sql("SELECT 1 AS id").write_datasink(
        _Sink(_BlockingCloseBound()),
        operation_id="local-close-timeout-warning",
    )

    assert summary.outcome is WriteOutcome.APPLIED
    assert any("graceful shutdown timed out" in warning for warning in summary.warnings)


def test_local_fte_datasink_progress_failure_is_warning(monkeypatch):
    from vane import runners
    from vane.runners.local import runner as local_runner_module
    from vane.runners.local.runner import LocalRunner

    class FailingProgressRenderer:
        interval_s = 0.001

        def __init__(self, snapshot_getter):
            self.snapshot_getter = snapshot_getter

        def update(self, *, force=False):
            raise RuntimeError("planned progress failure")

        def finish(self, *, final_state=None):
            return None

    monkeypatch.setattr(local_runner_module, "progress_enabled", lambda runner: True)
    monkeypatch.setattr(local_runner_module, "ProgressRenderer", FailingProgressRenderer)
    runner = LocalRunner(num_workers=1)
    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "local")
    monkeypatch.setattr(runners, "get_or_create_runner", lambda: runner)

    summary = vane.sql("SELECT 1 AS id").write_datasink(
        _Sink(_SlowBound()),
        operation_id="local-progress-warning",
    )

    assert summary.outcome is WriteOutcome.APPLIED
    assert any("planned progress failure" in warning for warning in summary.warnings)


def test_local_fte_datasink_progress_interrupt_stops_with_unknown_outcome(monkeypatch):
    from vane import runners
    from vane.runners.local import runner as local_runner_module
    from vane.runners.local.runner import LocalRunner

    class InterruptingProgressRenderer:
        interval_s = 0.001

        def __init__(self, snapshot_getter):
            self.snapshot_getter = snapshot_getter

        def update(self, *, force=False):
            raise KeyboardInterrupt("planned progress interrupt")

        def finish(self, *, final_state=None):
            return None

    monkeypatch.setattr(local_runner_module, "progress_enabled", lambda runner: True)
    monkeypatch.setattr(local_runner_module, "ProgressRenderer", InterruptingProgressRenderer)
    runner = LocalRunner(num_workers=1)
    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "local")
    monkeypatch.setattr(runners, "get_or_create_runner", lambda: runner)

    with pytest.raises(DataSinkWriteError) as exc_info:
        vane.sql("SELECT 1 AS id").write_datasink(
            _Sink(_SlowBound()),
            operation_id="local-progress-interrupt",
        )

    assert exc_info.value.outcome is WriteOutcome.UNKNOWN
    assert "planned progress interrupt" in exc_info.value.detail


def test_local_fte_datasink_provider_timeout_is_not_a_progress_wait(monkeypatch):
    from vane import runners
    from vane.runners.local import runner as local_runner_module
    from vane.runners.local.runner import LocalRunner

    class ProgressRenderer:
        interval_s = 0.001

        def __init__(self, snapshot_getter):
            self.snapshot_getter = snapshot_getter

        def update(self, *, force=False):
            return None

        def finish(self, *, final_state=None):
            return None

    monkeypatch.setattr(local_runner_module, "progress_enabled", lambda runner: True)
    monkeypatch.setattr(local_runner_module, "ProgressRenderer", ProgressRenderer)
    runner = LocalRunner(num_workers=1)
    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "local")
    monkeypatch.setattr(runners, "get_or_create_runner", lambda: runner)

    with pytest.raises(DataSinkWriteError) as exc_info:
        vane.sql("SELECT 1 AS id").write_datasink(
            _Sink(_TimeoutBound()),
            operation_id="local-provider-timeout",
        )

    assert exc_info.value.outcome is WriteOutcome.UNKNOWN
    assert "planned provider timeout" in exc_info.value.detail


def test_real_ray_datasink(ray_local, monkeypatch, tmp_path):
    from vane import runners
    from vane.runners.ray.runner import RayRunner

    runner = RayRunner(address=None, max_task_backlog=None)
    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "ray")
    monkeypatch.setattr(runners, "get_or_create_runner", lambda: runner)
    close_marker = tmp_path / "ray-worker-closed"
    try:
        summary = vane.sql("SELECT * FROM range(0, 4)").write_datasink(
            _Sink(_CloseMarkerBound(close_marker)),
            operation_id="ray-datasink",
        )
        assert close_marker.read_text(encoding="utf-8") == "closed"
        close_failure_summary = vane.sql("SELECT 1 AS id").write_datasink(
            _Sink(_Bound(fail_close=True)),
            operation_id="ray-datasink-close-failure",
        )
        empty_summary = vane.sql("SELECT 1 AS id WHERE false").write_datasink(
            _Sink(_KeyedBound(reject_open=True)),
            operation_id="ray-datasink-empty",
        )
        with pytest.raises(DataSinkWriteError) as duplicate_exc_info:
            vane.sql("SELECT * FROM (VALUES (1, 'a'), (1, 'b')) t(id, value)").write_datasink(
                _Sink(_KeyedBound(reject_open=True)),
                operation_id="ray-datasink-duplicate-keys",
            )
        with pytest.raises(DataSinkWriteError) as exc_info:
            vane.sql("SELECT 1 AS id").write_datasink(
                _Sink(_Bound(fail=True)),
                operation_id="ray-datasink-failure",
            )
    finally:
        runner.close()

    assert summary.outcome is WriteOutcome.APPLIED
    assert summary.rows_received == 4
    assert close_failure_summary.outcome is WriteOutcome.APPLIED
    assert any("planned close failure" in warning for warning in close_failure_summary.warnings)
    assert empty_summary.outcome is WriteOutcome.APPLIED
    assert empty_summary.results == ()
    assert duplicate_exc_info.value.outcome is WriteOutcome.ABORTED
    assert {result.state for result in duplicate_exc_info.value.summary.results} == {WriteState.ABORTED}
    assert exc_info.value.outcome is WriteOutcome.UNKNOWN
