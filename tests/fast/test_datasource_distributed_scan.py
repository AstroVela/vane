# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import os
import uuid
import weakref
from collections.abc import Iterator

import pyarrow as pa
import pytest

import vane
from vane import _native
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
