# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise public result streaming with a blocked sibling on a real Ray cluster."""

from __future__ import annotations

import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pytest

import vane
from vane.datasource import DataSource, DataSourceTask, read_datasource

pytestmark = [pytest.mark.real_ray, pytest.mark.ray_cluster_owner]


class GatedTask(DataSourceTask):
    def __init__(self, root: str, slow: bool, fail: bool) -> None:
        self.root, self.slow, self.fail = root, slow, fail

    def execute(self) -> Iterator[pa.RecordBatch]:
        yield from self._batches()

    def _execute_with_context(self, execution_context):
        return self._batches(execution_context)

    def _batches(self, execution_context=None):
        root = Path(self.root)
        if self.slow:
            (root / "started").touch()
            deadline = time.monotonic() + 60
            while not (root / "release").exists():
                if execution_context is not None:
                    execution_context._check_interrupted()
                if time.monotonic() >= deadline:
                    raise TimeoutError("test source was not released")
                time.sleep(0.01)
            if self.fail:
                raise RuntimeError("planned late source failure")
            values = [0, 1]
        else:
            (root / "fast").touch()
            values = [2, 3]
        yield pa.record_batch({"value": pa.array(values, type=pa.int64())})


class GatedSource(DataSource):
    def __init__(self, root: Path, *, slow_first: bool = True, fail: bool = False) -> None:
        self.root, self.slow_first, self.fail = str(root), slow_first, fail

    @property
    def schema(self) -> dict[str, str]:
        return {"value": "BIGINT"}

    def get_tasks(self) -> Iterator[DataSourceTask]:
        for slow in (self.slow_first, not self.slow_first):
            yield GatedTask(self.root, slow, self.fail)


class PlusTen:
    def __call__(self, table):
        import pyarrow.compute as pc

        return pa.table({"value": pc.add(table.column(0), 10)})


@pytest.fixture(scope="module")
def result_order_cluster():
    import ray
    from ray_test_profile import ray_test_object_store_options

    vane.teardown_runner()
    assert not ray.is_initialized(), "this test owns its Ray cluster"
    ray.init(address="local", num_cpus=4, include_dashboard=False, **ray_test_object_store_options())
    try:
        yield
    finally:
        vane.teardown_runner()
        ray.shutdown()


@pytest.fixture
def ray_connection(result_order_cluster, monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    con = vane.connect(config={"preserve_insertion_order": False})
    try:
        yield con
    finally:
        con.close()


def _first_batch(relation):
    reader = relation.to_arrow_reader(batch_size=1)
    try:
        return reader, next(reader).column(0).to_pylist()
    except BaseException:
        reader.close()
        raise


def _wait_for(path: Path) -> None:
    deadline = time.monotonic() + 30
    while not path.exists():
        assert time.monotonic() < deadline, f"worker did not reach {path.name}"
        time.sleep(0.01)


@pytest.mark.parametrize("slow_first", [True, False])
@pytest.mark.parametrize("actor_stage", [False, True])
def test_unordered_reader_delivers_before_slow_task_finishes(ray_connection, tmp_path, slow_first, actor_stage):
    source = read_datasource(GatedSource(tmp_path, slow_first=slow_first), con=ray_connection)
    relation = (
        source.map_batches(
            PlusTen, schema={"value": "BIGINT"}, execution_backend="ray_actor", actor_number=1, batch_size=1
        )
        if actor_stage
        else source.project("value + 10")
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_first_batch, relation)
        reader = None
        try:
            reader, first = future.result(timeout=30)
            assert not (tmp_path / "release").exists()
            assert first == [12]
            (tmp_path / "release").touch()
            assert first + reader.read_all().column(0).to_pylist() == [12, 13, 10, 11]
        finally:
            (tmp_path / "release").touch()
            if reader is None and not future.cancel():
                reader, _ = future.result(timeout=30)
            if reader is not None:
                reader.close()


@pytest.mark.parametrize("query", ["ORDER BY value", "ORDER BY value DESC LIMIT 2 OFFSET 1", ""])
def test_ordered_reader_waits_and_honors_sql_order(ray_connection, tmp_path, query):
    if not query:
        ray_connection.execute("SET preserve_insertion_order=true")
    source = read_datasource(GatedSource(tmp_path), con=ray_connection)
    relation = source.query("source", f"SELECT value FROM source {query}")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_first_batch, relation)
        reader = None
        try:
            _wait_for(tmp_path / "fast")
            _wait_for(tmp_path / "started")
            assert not future.done()
            (tmp_path / "release").touch()
            reader, first = future.result(timeout=30)
            values = first + reader.read_all().column(0).to_pylist()
            if query:
                expected = [2, 1] if "DESC" in query else [0, 1, 2, 3]
                assert values == expected
            else:
                # Partition collection order is unspecified without ORDER BY.
                assert sorted(values) == [0, 1, 2, 3]
        finally:
            (tmp_path / "release").touch()
            if reader is None and not future.cancel():
                reader, _ = future.result(timeout=30)
            if reader is not None:
                reader.close()


def test_unordered_reader_propagates_late_failure(ray_connection, tmp_path):
    relation = read_datasource(GatedSource(tmp_path, fail=True), con=ray_connection)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_first_batch, relation)
        reader = None
        try:
            reader, first = future.result(timeout=30)
            assert first == [2]
            (tmp_path / "release").touch()
            with pytest.raises(Exception, match="planned late source failure"):
                reader.read_all()
        finally:
            (tmp_path / "release").touch()
            if reader is not None:
                reader.close()
    assert ray_connection.sql("SELECT 42").fetchall() == [(42,)]


def test_unordered_reader_can_close_before_query_finishes(ray_connection, tmp_path):
    relation = read_datasource(GatedSource(tmp_path), con=ray_connection)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_first_batch, relation)
        reader = None
        try:
            reader, first = future.result(timeout=30)
            assert first == [2]
            # Closing an incrementally published result must use query cleanup,
            # even though another task has not produced its terminal result.
            close = pool.submit(reader.close)
            close.result(timeout=30)
            assert not (tmp_path / "release").exists()
        finally:
            (tmp_path / "release").touch()
            if reader is not None:
                reader.close()
    assert ray_connection.sql("SELECT 42").fetchall() == [(42,)]
