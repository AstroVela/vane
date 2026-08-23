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
    CommitProtocol,
    DataSink,
    DataSinkCapabilities,
    DataSinkExecutionOptions,
    DataSinkWorker,
    DataSinkWriteError,
    EnvironmentSecret,
    RetryMode,
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
        protocol: CommitProtocol = CommitProtocol.IMMEDIATE,
        retry: RetryMode = RetryMode.IDEMPOTENT,
        fail: bool = False,
        fail_close: bool = False,
        options: DataSinkExecutionOptions | None = None,
    ) -> None:
        self._capabilities = DataSinkCapabilities(protocol, retry)
        self._fail = fail
        self._fail_close = fail_close
        self._options = options or DataSinkExecutionOptions(batch_size=2)

    @property
    def capabilities(self) -> DataSinkCapabilities:
        return self._capabilities

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


def test_local_fast_empty_input_does_not_open_a_worker(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    relation = vane.sql("SELECT 1 AS id WHERE false")

    summary = relation.write_datasink(_Sink(_KeyedBound(reject_open=True)), operation_id="empty")

    assert summary.outcome is WriteOutcome.APPLIED
    assert summary.results == ()
    assert summary.rows_received == 0
    assert summary.rows_affected == 0


def test_worker_close_failure_is_reported_during_actor_teardown():
    from vane.datasink import _SinkBatchRuntime

    runtime = _SinkBatchRuntime(_Bound(fail_close=True), WriteContext("close-failure"), None)
    runtime(pa.table({"id": [1]}))

    with pytest.raises(RuntimeError, match="planned close failure"):
        runtime.close()


def test_worker_failure_is_unknown_and_safe_to_retry(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")

    with pytest.raises(DataSinkWriteError) as exc_info:
        vane.sql("SELECT 1 AS id").write_datasink(_Sink(_Bound(fail=True)), operation_id="worker-failure")

    assert exc_info.value.outcome is WriteOutcome.UNKNOWN
    assert exc_info.value.safe_to_retry is True
    assert exc_info.value.summary.results == ()


@pytest.mark.parametrize(
    "bound, match",
    [
        (_Bound(protocol=CommitProtocol.TWO_PHASE), "commit_protocol='immediate'"),
        (_Bound(retry=RetryMode.NEVER), "retry_mode='idempotent'"),
    ],
)
def test_unsupported_protocols_fail_before_execution(monkeypatch, bound, match):
    from vane import runners

    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: (_ for _ in ()).throw(AssertionError()))
    with pytest.raises(ValueError, match=match):
        vane.sql("SELECT 1 AS id").write_datasink(_Sink(bound))


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


def test_datasink_protocol_values_reject_invalid_utf8():
    with pytest.raises(ValueError, match="valid UTF-8"):
        WriteContext("operation-\ud800")
    with pytest.raises(ValueError, match="valid UTF-8"):
        WriteResult(rows_received=1, warnings=("warning-\ud800",))
    with pytest.raises(ValueError, match="valid UTF-8"):
        EnvironmentSecret("SECRET_\ud800")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"memory_bytes": -1},
        {"cpus": float("nan")},
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
    ],
)
def test_execution_options_reject_invalid_worker_requests(kwargs):
    with pytest.raises((TypeError, ValueError)):
        DataSinkExecutionOptions(**kwargs)


def test_execution_options_map_to_actor_backends():
    options = DataSinkExecutionOptions(worker_count=3, batch_size=20)

    assert options.map_batches_kwargs("ray") == {
        "batch_size": 20,
        "execution_backend": "ray_actor",
        "actor_number": 3,
    }
    assert options.map_batches_kwargs("local") == {
        "batch_size": 20,
        "execution_backend": "subprocess_actor",
        "actor_number": 3,
    }
    assert DataSinkExecutionOptions().worker_count == 1
    with pytest.raises(ValueError, match="unsupported DataSink runner"):
        options.map_batches_kwargs("unsupported")


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
    original = DataSinkWriteError(summary, "unknown", safe_to_retry=True)

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

    summary = vane.datasink._summary(WriteContext("bounded-detail"), WriteOutcome.UNKNOWN, ())
    error = DataSinkWriteError(summary, "detail-head:" + "x" * 10_000 + ":provider-root-cause", safe_to_retry=True)
    assert len(error.detail.encode("utf-8")) <= 4 * 1024
    assert error.detail.startswith("detail-head:")
    assert error.detail.endswith(":provider-root-cause")


def test_local_fte_datasink(monkeypatch):
    from vane import runners
    from vane.runners.local.runner import LocalRunner

    runner = LocalRunner(num_workers=1)
    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "local")
    monkeypatch.setattr(runners, "get_or_create_runner", lambda: runner)

    summary = vane.sql("SELECT * FROM range(0, 3)").write_datasink(_Sink(), operation_id="local-fte")

    assert summary.outcome is WriteOutcome.APPLIED
    assert summary.rows_received == 3


def test_local_fte_datasink_worker_failure_is_unknown(monkeypatch):
    from vane import runners
    from vane.runners.local.runner import LocalRunner

    runner = LocalRunner(num_workers=1)
    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "local")
    monkeypatch.setattr(runners, "get_or_create_runner", lambda: runner)

    with pytest.raises(DataSinkWriteError) as exc_info:
        vane.sql("SELECT 1 AS id").write_datasink(
            _Sink(_Bound(fail=True)),
            operation_id="local-fte-failure",
        )

    assert exc_info.value.outcome is WriteOutcome.UNKNOWN
    assert exc_info.value.safe_to_retry is True


def test_local_fte_datasink_cleanup_failure_is_warning(monkeypatch):
    from vane import runners
    from vane.runners.local import runner as local_runner_module
    from vane.runners.local.runner import LocalRunner

    original_shutdown = local_runner_module._shutdown_local_write_resources

    def shutdown_with_warning(*args, **kwargs):
        errors = original_shutdown(*args, **kwargs)
        errors.append(RuntimeError("planned local cleanup failure"))
        return errors

    monkeypatch.setattr(local_runner_module, "_shutdown_local_write_resources", shutdown_with_warning)
    runner = LocalRunner(num_workers=1)
    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "local")
    monkeypatch.setattr(runners, "get_or_create_runner", lambda: runner)

    summary = vane.sql("SELECT 1 AS id").write_datasink(_Sink(), operation_id="local-cleanup-warning")

    assert summary.outcome is WriteOutcome.APPLIED
    assert any("planned local cleanup failure" in warning for warning in summary.warnings)


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
    assert empty_summary.outcome is WriteOutcome.APPLIED
    assert empty_summary.results == ()
    assert duplicate_exc_info.value.outcome is WriteOutcome.ABORTED
    assert {result.state for result in duplicate_exc_info.value.summary.results} == {WriteState.ABORTED}
    assert exc_info.value.outcome is WriteOutcome.UNKNOWN
    assert exc_info.value.safe_to_retry is True
