# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
import time
import uuid
import weakref
from collections.abc import Iterator

import pyarrow as pa
import pytest

import vane
from vane import _native
from vane import runners as _runners
from vane.datasource import DataSource, DataSourceTask, read_datasource


@pytest.fixture(autouse=True)
def _vane_shuffle_env(monkeypatch):
    monkeypatch.setenv("VANE_SHUFFLE_ALGORITHM", "flight_shuffle")
    monkeypatch.setenv("VANE_SHUFFLE_LOCAL_DIRS", "/tmp/duckdb_shuffle")
    monkeypatch.setenv("RAY_DEDUP_LOGS", "0")


@pytest.fixture
def duckdb_conn():
    con = vane.connect()
    try:
        yield con
    finally:
        con.close()


@pytest.fixture
def ray_runner(_vane_shuffle_env, request):
    request.getfixturevalue("ray_local")
    try:
        _runners.set_runner_ray(noop_if_initialized=True)
        runner = _runners.get_or_create_runner()
    except Exception:
        pytest.skip("Ray runner not available in this environment")

    if getattr(runner, "name", None) != "ray":
        pytest.skip("Ray runner not active")
    try:
        yield runner
    finally:
        vane_mod = vane
        if vane_mod is not None and hasattr(vane_mod, "teardown_runner"):
            vane_mod.teardown_runner()


def _collect_tables(runner, relation, timeout_s: float = 60.0) -> pa.Table:
    start = time.time()
    parts = list(runner.run_iter_tables(relation))
    elapsed = time.time() - start
    assert elapsed < timeout_s
    assert parts
    return pa.concat_tables(parts)


def _datasource_registry_state() -> dict:
    return dict(_native._datasource_factory_registry_state_for_test())


class RegistryProbeTask(DataSourceTask):
    def __init__(self, value: int) -> None:
        self.value = int(value)

    def execute(self) -> Iterator[pa.RecordBatch]:
        from vane import _native

        state = dict(_native._datasource_factory_registry_state_for_test())
        source_ids = list(state["source_ids"])
        if len(source_ids) != 1:
            raise RuntimeError(f"expected one active datasource source, found {source_ids!r}")
        query_ids = list(state["query_ids"])
        yield pa.record_batch(
            {
                "value": pa.array([self.value], type=pa.int64()),
                "source_id": pa.array([source_ids[0]], type=pa.string()),
                "query_id": pa.array([query_ids[0] if query_ids else ""], type=pa.string()),
                "registry_size": pa.array([state["registry_size"]], type=pa.int64()),
                "owner_count": pa.array([state["owner_count"]], type=pa.int64()),
                "factory_creation_count": pa.array([state["factory_creation_count"]], type=pa.int64()),
            }
        )


class RegistryProbeSource(DataSource):
    def __init__(self, values: list[int]) -> None:
        self.values = [int(value) for value in values]

    @property
    def schema(self) -> dict[str, str]:
        return {
            "value": "BIGINT",
            "source_id": "VARCHAR",
            "query_id": "VARCHAR",
            "registry_size": "BIGINT",
            "owner_count": "BIGINT",
            "factory_creation_count": "BIGINT",
        }

    def get_tasks(self) -> Iterator[DataSourceTask]:
        for value in self.values:
            yield RegistryProbeTask(value)


class StreamingTask(DataSourceTask):
    def execute(self) -> Iterator[pa.RecordBatch]:
        for value in range(3):
            yield pa.record_batch({"value": pa.array([value], type=pa.int64())})


class StreamingSource(DataSource):
    @property
    def schema(self) -> dict[str, str]:
        return {"value": "BIGINT"}

    def get_tasks(self) -> Iterator[DataSourceTask]:
        yield StreamingTask()


class BatchSequenceTask(DataSourceTask):
    def __init__(self, batches: list[list[int]]) -> None:
        self.batches = [list(batch) for batch in batches]

    def execute(self) -> Iterator[pa.RecordBatch]:
        for batch in self.batches:
            yield pa.record_batch({"value": pa.array(batch, type=pa.int64())})


class BatchSequenceSource(DataSource):
    def __init__(self, tasks: list[list[list[int]]]) -> None:
        self.tasks = [[list(batch) for batch in task] for task in tasks]

    @property
    def schema(self) -> dict[str, str]:
        return {"value": "BIGINT"}

    def get_tasks(self) -> Iterator[DataSourceTask]:
        for task in self.tasks:
            yield BatchSequenceTask(task)


class SchemaCallTrackingSource(StreamingSource):
    schema_calls = 0

    @property
    def schema(self) -> dict[str, str]:
        type(self).schema_calls += 1
        return super().schema


class FailingTask(DataSourceTask):
    def execute(self) -> Iterator[pa.RecordBatch]:
        raise RuntimeError("datasource task failed")
        yield  # pragma: no cover


class FailingSource(DataSource):
    @property
    def schema(self) -> dict[str, str]:
        return {"value": "BIGINT"}

    def get_tasks(self) -> Iterator[DataSourceTask]:
        yield FailingTask()


_retained_execution_contexts: list[object] = []
_retained_file_readers: list[object] = []


class RetainingExecutionContextTask(DataSourceTask):
    def __init__(self, outcome: str, path: str | None = None) -> None:
        self.outcome = outcome
        self.path = path

    def execute(self) -> Iterator[pa.RecordBatch]:
        raise RuntimeError("retaining task requires a DataSource execution context")
        yield  # pragma: no cover

    def _execute_with_context(self, execution_context: object) -> Iterator[pa.RecordBatch]:
        _retained_execution_contexts.append(execution_context)
        if self.outcome == "setup_error":
            raise RuntimeError("planned DataSource setup failure")
        if self.path is not None:
            from vane._file import _file_open_in_datasource_context

            reader = _file_open_in_datasource_context(
                vane.File(self.path),
                16,
                execution_context=execution_context,
            )
            _retained_file_readers.append(reader)

        def batches() -> Iterator[pa.RecordBatch]:
            if self.outcome == "stream_error":
                raise RuntimeError("planned DataSource stream failure")
            yield pa.record_batch({"value": pa.array([47], type=pa.int64())})
            if self.outcome == "early_close":
                yield pa.record_batch({"value": pa.array([48], type=pa.int64())})

        return batches()


class RetainingExecutionContextSource(DataSource):
    def __init__(self, outcome: str, path: str | None = None) -> None:
        self.outcome = outcome
        self.path = path

    @property
    def schema(self) -> dict[str, str]:
        return {"value": "BIGINT"}

    def get_tasks(self) -> Iterator[DataSourceTask]:
        yield RetainingExecutionContextTask(self.outcome, self.path)


class SourceKeepaliveProbe(DataSource):
    def __init__(self, path: str) -> None:
        self.path = path

    @property
    def schema(self) -> dict[str, str]:
        return {"value": "BIGINT"}

    def get_tasks(self) -> Iterator[DataSourceTask]:
        class SourceKeepaliveTask(DataSourceTask):
            def __init__(self, path: str) -> None:
                self.path = path

            def execute(self) -> Iterator[pa.RecordBatch]:
                with open(self.path, encoding="utf-8") as source_file:
                    value = int(source_file.read())
                yield pa.record_batch({"value": pa.array([value], type=pa.int64())})

        yield SourceKeepaliveTask(self.path)

    def __del__(self) -> None:
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass


def test_datasource_relation_keeps_source_alive_until_relation_is_released(duckdb_conn, tmp_path):
    source_path = tmp_path / "source-keepalive.txt"
    source_path.write_text("42", encoding="utf-8")
    source = SourceKeepaliveProbe(str(source_path))
    source_ref = weakref.ref(source)

    relation = read_datasource(source, con=duckdb_conn)
    del source
    gc.collect()

    assert source_ref() is not None
    assert source_path.exists()
    assert relation.fetchall() == [(42,)]

    del relation
    gc.collect()
    assert source_ref() is None
    assert not source_path.exists()


@pytest.mark.parametrize("stream_outcome", ["complete", "close", "error"])
def test_ray_runner_plan_retention_does_not_extend_datasource_lifetime(
    duckdb_conn,
    monkeypatch,
    stream_outcome,
    tmp_path,
):
    from vane.runners.ray.runner import RayRunner

    source_path = tmp_path / "runner-plan-retention-source.txt"
    source_path.write_text("43", encoding="utf-8")
    source = SourceKeepaliveProbe(str(source_path))
    source_ref = weakref.ref(source)
    relation = read_datasource(source, con=duckdb_conn, limit=1)
    result = object()

    class _RetainingClient:
        def __init__(self):
            self.plan = None

        def stream_plan(self, plan):
            self.plan = plan
            assert source_ref() is not None
            yield result
            assert source_ref() is not None
            if stream_outcome == "error":
                raise RuntimeError("planned stream failure")

    client = _RetainingClient()
    runner = object.__new__(RayRunner)
    monkeypatch.setattr(RayRunner, "_client_for_session", lambda _self, _session_id: client)

    results = runner.run_iter(relation)
    del source
    del relation
    gc.collect()

    assert source_ref() is not None
    if stream_outcome == "complete":
        assert list(results) == [result]
    elif stream_outcome == "close":
        assert next(results) is result
        results.close()
    else:
        with pytest.raises(RuntimeError, match="planned stream failure"):
            list(results)
    assert client.plan is not None

    gc.collect()
    assert source_ref() is None
    assert not source_path.exists()


def test_ray_runner_keeps_source_alive_until_distributed_scan_finishes(ray_runner, duckdb_conn, tmp_path):
    source_path = tmp_path / "distributed-source-keepalive.txt"
    source_path.write_text("43", encoding="utf-8")
    source = SourceKeepaliveProbe(str(source_path))
    source_ref = weakref.ref(source)

    relation = read_datasource(source, con=duckdb_conn, limit=1)
    del source
    gc.collect()

    assert source_ref() is not None
    assert source_path.exists()
    result = _collect_tables(ray_runner, relation)
    assert result.num_rows == 1
    assert result.column(0).to_pylist() == [43]

    del relation
    gc.collect()
    assert source_ref() is None
    assert not source_path.exists()


def test_datasource_factory_registry_churn_returns_to_baseline(duckdb_conn):
    baseline = _datasource_registry_state()
    assert baseline["registry_size"] == 0
    assert baseline["factory_count"] == 0
    assert baseline["owner_count"] == 0

    source_ids: set[str] = set()
    previous_creation_count = baseline["factory_creation_count"]
    for value in range(64):
        relation = read_datasource(RegistryProbeSource([value]), con=duckdb_conn)
        after_bind = _datasource_registry_state()
        driver_source_id = after_bind["last_created_source_id"]
        assert str(uuid.UUID(driver_source_id)) == driver_source_id
        assert after_bind["registry_size"] == baseline["registry_size"]

        assert relation.fetchall() == [
            (
                value,
                driver_source_id,
                "",
                1,
                1,
                previous_creation_count + 1,
            )
        ]
        source_ids.add(driver_source_id)
        previous_creation_count += 1

        after_query = _datasource_registry_state()
        assert after_query["registry_size"] == baseline["registry_size"]
        assert after_query["factory_count"] == baseline["factory_count"]
        assert after_query["owner_count"] == baseline["owner_count"]
        assert after_query["factory_creation_count"] == previous_creation_count

    assert len(source_ids) == 64


def test_datasource_factory_owner_released_when_stream_finishes(duckdb_conn):
    baseline = _datasource_registry_state()
    relation = read_datasource(StreamingSource(), con=duckdb_conn)

    assert relation.fetchone() == (0,)
    active = _datasource_registry_state()
    assert active["registry_size"] == baseline["registry_size"] + 1
    assert active["factory_count"] == baseline["factory_count"] + 1
    assert active["local_owner_count"] == baseline["local_owner_count"] + 1

    assert relation.fetchall() == [(1,), (2,)]
    finished = _datasource_registry_state()
    assert finished["registry_size"] == baseline["registry_size"]
    assert finished["factory_count"] == baseline["factory_count"]
    assert finished["owner_count"] == baseline["owner_count"]


def test_datasource_scan_reads_entire_large_record_batch(duckdb_conn):
    values = list(range(5000))

    result = read_datasource(BatchSequenceSource([[values]]), con=duckdb_conn).fetchall()

    assert result == [(value,) for value in values]


def test_datasource_scan_continues_after_empty_record_batches(duckdb_conn):
    source = BatchSequenceSource([[[], [10, 20], [], [], [30], []]])

    assert read_datasource(source, con=duckdb_conn).fetchall() == [(10,), (20,), (30,)]


def test_datasource_schema_is_evaluated_once(duckdb_conn):
    SchemaCallTrackingSource.schema_calls = 0

    relation = read_datasource(SchemaCallTrackingSource(), con=duckdb_conn)
    assert SchemaCallTrackingSource.schema_calls == 1
    assert relation.fetchall() == [(0,), (1,), (2,)]
    assert SchemaCallTrackingSource.schema_calls == 1


def test_datasource_factory_owner_released_when_query_fails(duckdb_conn):
    baseline = _datasource_registry_state()
    relation = read_datasource(FailingSource(), con=duckdb_conn)

    with pytest.raises(Exception, match="datasource task failed"):
        relation.fetchall()

    finished = _datasource_registry_state()
    assert finished["registry_size"] == baseline["registry_size"]
    assert finished["factory_count"] == baseline["factory_count"]
    assert finished["owner_count"] == baseline["owner_count"]


@pytest.mark.parametrize("outcome", ["complete", "early_close", "setup_error", "stream_error"])
def test_datasource_execution_context_expires_with_arrow_stream(duckdb_conn, outcome):
    _retained_execution_contexts.clear()
    relation = read_datasource(RetainingExecutionContextSource(outcome), con=duckdb_conn)

    if outcome in {"complete", "early_close"}:
        result = relation.limit(1).fetchall() if outcome == "early_close" else relation.fetchall()
        assert result == [(47,)]
    else:
        with pytest.raises(Exception, match=f"planned DataSource {outcome.removesuffix('_error')} failure"):
            relation.fetchall()

    assert len(_retained_execution_contexts) == 1
    with pytest.raises(vane.InvalidInputException, match="execution context is no longer active"):
        _retained_execution_contexts[0]._check_interrupted()


def test_datasource_reader_cannot_outlive_its_query_context(duckdb_conn, tmp_path):
    path = tmp_path / "retained-reader.bin"
    path.write_bytes(b"retained reader payload")
    _retained_execution_contexts.clear()
    _retained_file_readers.clear()

    relation = read_datasource(RetainingExecutionContextSource("complete", str(path)), con=duckdb_conn)
    assert relation.fetchall() == [(47,)]
    assert len(_retained_execution_contexts) == 1
    assert len(_retained_file_readers) == 1
    execution_context = _retained_execution_contexts[0]
    reader = _retained_file_readers[0]
    try:
        from vane._file import _file_open_in_datasource_context

        with pytest.raises(vane.InvalidInputException, match="execution context is no longer active"):
            _file_open_in_datasource_context(
                vane.File(str(path)),
                16,
                execution_context=execution_context,
            )
        with pytest.raises(vane.InvalidInputException, match="execution context is no longer active"):
            reader.read(1)
    finally:
        reader.close()


def test_datasource_worker_plan_uses_resource_query_owner_when_execution_id_differs(duckdb_conn):
    source_plan_id = "query-datasource-source-plan"
    resource_query_id = "query-datasource-resource-owner"
    execution_query_id = f"{resource_query_id}:stage:retry"
    relation = read_datasource(RegistryProbeSource([41]), con=duckdb_conn)
    logical_plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, source_plan_id)
    source_plan = logical_plan.to_physical_plan(duckdb_conn)
    assert source_plan.idx() == source_plan_id
    assert source_plan.resource_query_id() == source_plan_id
    scan_split_batches = source_plan.scan_split_batch_map()
    assert len(scan_split_batches) == 1
    scan_node_id, split_batches = next(iter(scan_split_batches.items()))
    assert len(split_batches) == 1

    vane.ray_cxx._register_query_python_replay_state(resource_query_id, source_plan)
    worker_connection = vane.connect()
    worker_source_plan = None
    task = None
    worker_plan = None
    result = None
    try:
        worker_source_plan = source_plan.clone(worker_connection)
        task = vane.ray_cxx._make_worker_task_from_plan_for_test(
            worker_source_plan,
            execution_query_id,
            resource_query_id,
        )
        worker_plan = task.plan()
        assert worker_plan.idx() == execution_query_id
        assert worker_plan.resource_query_id() == resource_query_id

        result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(
            worker_connection,
            worker_plan,
            scan_split_batch={str(scan_node_id): bytes(split_batches[0])},
        )
        assert result.completion_status == "ok"
        assert result.partition_payloads[0].column(0).to_pylist() == [41]
        assert result.partition_payloads[0].column(2).to_pylist() == [resource_query_id]

        active = _datasource_registry_state()
        assert active["query_ids"] == [resource_query_id]
        assert execution_query_id not in active["query_ids"]
    finally:
        result = None
        worker_plan = None
        task = None
        worker_source_plan = None
        worker_connection.close()
        _native._release_datasource_factories_for_query(resource_query_id)
        vane.ray_cxx._cleanup_query_python_replay_state(resource_query_id)


def test_datasource_fte_scan_wait_releases_gil_for_queue_seal():
    # Isolate the expected pre-fix deadlock so the parent pytest process can
    # enforce a hard timeout and report both thread stacks.
    code = textwrap.dedent(
        """
        from __future__ import annotations

        import faulthandler
        import sys
        import threading
        import uuid
        from collections.abc import Iterator

        import pyarrow as pa

        import vane
        from vane import _native
        from vane.datasource import DataSource, DataSourceTask, read_datasource


        class SingleTask(DataSourceTask):
            def execute(self) -> Iterator[pa.RecordBatch]:
                yield pa.record_batch({"value": pa.array([41], type=pa.int64())})


        class SingleSource(DataSource):
            @property
            def schema(self) -> dict[str, str]:
                return {"value": "BIGINT"}

            def get_tasks(self) -> Iterator[DataSourceTask]:
                yield SingleTask()


        faulthandler.dump_traceback_later(5.0, repeat=False)
        # After the executor thread starts, only an explicit native GIL release
        # should let this thread resume and seal the queue.
        sys.setswitchinterval(1000.0)

        query_id = f"query-datasource-fte-gil-{uuid.uuid4().hex[:8]}"
        source_connection = vane.connect()
        relation = read_datasource(SingleSource(), con=source_connection)
        source_plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
            relation,
            query_id,
        ).to_physical_plan(source_connection)
        scan_split_batches = source_plan.scan_split_batch_map()
        assert len(scan_split_batches) == 1
        node_id, split_batches = next(iter(scan_split_batches.items()))
        assert len(split_batches) == 1

        vane.ray_cxx._register_query_python_replay_state(query_id, source_plan)
        worker_connection = vane.connect()
        worker_plan = source_plan.clone(worker_connection)
        split_queue = vane.ray_cxx.FteSplitQueue()
        split_queue.add_scan_split(bytes(split_batches[0]))
        started = threading.Event()
        results = []
        errors = []


        def execute_native() -> None:
            started.set()
            try:
                results.append(
                    vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(
                        worker_connection,
                        worker_plan,
                        fte_scan_source_queues={str(node_id): split_queue},
                    )
                )
            except BaseException as exc:
                errors.append(exc)


        thread = threading.Thread(target=execute_native, daemon=True)
        thread.start()
        assert started.wait(timeout=2.0)
        split_queue.no_more_splits()
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        assert errors == []
        assert len(results) == 1
        assert results[0].completion_status == "ok"
        assert results[0].partition_payloads[0].column(0).to_pylist() == [41]
        assert split_queue.consumed_splits() == 1

        faulthandler.cancel_dump_traceback_later()
        results.clear()
        worker_plan = None
        worker_connection.close()
        _native._release_datasource_factories_for_query(query_id)
        vane.ray_cxx._cleanup_query_python_replay_state(query_id)
        source_connection.close()
        print("ok", flush=True)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ok" in proc.stdout


def test_ray_runner_executes_python_datasource_task_on_worker(ray_runner, duckdb_conn):
    driver_pid = os.getpid()

    class ExecutionLocationTask(DataSourceTask):
        def __init__(self, task_id: int) -> None:
            self.task_id = int(task_id)

        def execute(self) -> Iterator[pa.RecordBatch]:
            from vane import _native

            worker_id = os.getenv("VANE_WORKER_ID", "").strip() or os.getenv("VANE_FTE_WORKER_ID", "").strip()
            registry = dict(_native._datasource_factory_registry_state_for_test())
            source_ids = list(registry["source_ids"])
            if len(source_ids) != 1:
                raise RuntimeError(f"expected one active datasource source, found {source_ids!r}")
            query_ids = list(registry["query_ids"])
            if len(query_ids) != 1:
                raise RuntimeError(f"expected one datasource query owner, found {query_ids!r}")
            yield pa.record_batch(
                {
                    "task_id": pa.array([self.task_id], type=pa.int64()),
                    "driver_pid": pa.array([driver_pid], type=pa.int64()),
                    "execute_pid": pa.array([os.getpid()], type=pa.int64()),
                    "worker_id": pa.array([worker_id], type=pa.string()),
                    "source_id": pa.array([source_ids[0]], type=pa.string()),
                    "query_id": pa.array([query_ids[0]], type=pa.string()),
                    "registry_size": pa.array([registry["registry_size"]], type=pa.int64()),
                    "owner_count": pa.array([registry["owner_count"]], type=pa.int64()),
                    "factory_creation_count": pa.array(
                        [registry["factory_creation_count"]],
                        type=pa.int64(),
                    ),
                }
            )

    class ExecutionLocationSource(DataSource):
        @property
        def schema(self) -> dict[str, str]:
            return {
                "task_id": "BIGINT",
                "driver_pid": "BIGINT",
                "execute_pid": "BIGINT",
                "worker_id": "VARCHAR",
                "source_id": "VARCHAR",
                "query_id": "VARCHAR",
                "registry_size": "BIGINT",
                "owner_count": "BIGINT",
                "factory_creation_count": "BIGINT",
            }

        def get_tasks(self) -> Iterator[DataSourceTask]:
            for task_id in range(4):
                yield ExecutionLocationTask(task_id)

    source_ids: set[str] = set()
    query_ids: set[str] = set()
    creation_counts_by_run: list[dict[int, int]] = []
    for _ in range(2):
        relation = read_datasource(ExecutionLocationSource(), con=duckdb_conn)
        driver_state = _datasource_registry_state()
        driver_source_id = driver_state["last_created_source_id"]
        assert str(uuid.UUID(driver_source_id)) == driver_source_id
        assert driver_state["registry_size"] == 0

        table = _collect_tables(ray_runner, relation)
        rows = sorted(zip(*[column.to_pylist() for column in table.columns], strict=True), key=lambda row: row[0])

        assert [row[0] for row in rows] == [0, 1, 2, 3]
        assert all(row[1] == driver_pid for row in rows)
        assert all(row[2] != driver_pid for row in rows)
        assert all(row[3] for row in rows)
        assert {row[4] for row in rows} == {driver_source_id}
        assert len({row[5] for row in rows}) == 1
        assert all(row[6] == 1 for row in rows)
        assert all(row[7] == 1 for row in rows)

        creation_counts_by_pid: dict[int, set[int]] = {}
        for row in rows:
            creation_counts_by_pid.setdefault(row[2], set()).add(row[8])
        assert all(len(counts) == 1 for counts in creation_counts_by_pid.values())

        source_ids.add(driver_source_id)
        query_ids.add(rows[0][5])
        creation_counts_by_run.append({pid: next(iter(counts)) for pid, counts in creation_counts_by_pid.items()})

    assert len(source_ids) == 2
    assert len(query_ids) == 2
    shared_worker_pids = creation_counts_by_run[0].keys() & creation_counts_by_run[1].keys()
    for pid in shared_worker_pids:
        assert creation_counts_by_run[1][pid] == creation_counts_by_run[0][pid] + 1


def test_ray_runner_preserves_large_and_post_empty_datasource_batches(ray_runner, duckdb_conn):
    class RayBatchSequenceTask(DataSourceTask):
        def __init__(self, batches: list[list[int]]) -> None:
            self.batches = [list(batch) for batch in batches]

        def execute(self) -> Iterator[pa.RecordBatch]:
            for batch in self.batches:
                yield pa.record_batch({"value": pa.array(batch, type=pa.int64())})

    class RayBatchSequenceSource(DataSource):
        @property
        def schema(self) -> dict[str, str]:
            return {"value": "BIGINT"}

        def get_tasks(self) -> Iterator[DataSourceTask]:
            yield RayBatchSequenceTask([[], list(range(5000)), [], [5000, 5001]])
            yield RayBatchSequenceTask([[], [6000], []])

    expected = list(range(5002)) + [6000]
    table = _collect_tables(ray_runner, read_datasource(RayBatchSequenceSource(), con=duckdb_conn))

    assert sorted(table.column(0).to_pylist()) == expected
