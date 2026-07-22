# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import pyarrow as pa
import pytest

import vane
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
        vane_mod = getattr(vane, "vane_runners_cpp", None)
        if vane_mod is not None and hasattr(vane_mod, "teardown_runner"):
            vane_mod.teardown_runner()


def _collect_tables(runner, relation, timeout_s: float = 60.0) -> pa.Table:
    start = time.time()
    parts = list(runner.run_iter_tables(relation, results_buffer_size=1))
    elapsed = time.time() - start
    assert elapsed < timeout_s
    assert parts
    return pa.concat_tables(parts)


def test_ray_runner_executes_python_datasource_task_on_worker(ray_runner, duckdb_conn):
    driver_pid = os.getpid()

    class ExecutionLocationTask(DataSourceTask):
        def __init__(self, task_id: int) -> None:
            self.task_id = int(task_id)

        def execute(self) -> Iterator[pa.RecordBatch]:
            worker_id = os.getenv("VANE_WORKER_ID", "").strip() or os.getenv("VANE_FTE_WORKER_ID", "").strip()
            yield pa.record_batch(
                {
                    "task_id": pa.array([self.task_id], type=pa.int64()),
                    "driver_pid": pa.array([driver_pid], type=pa.int64()),
                    "execute_pid": pa.array([os.getpid()], type=pa.int64()),
                    "worker_id": pa.array([worker_id], type=pa.string()),
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
            }

        def get_tasks(self) -> Iterator[DataSourceTask]:
            for task_id in range(4):
                yield ExecutionLocationTask(task_id)

    source = ExecutionLocationSource()
    relation = read_datasource(source, con=duckdb_conn)

    table = _collect_tables(ray_runner, relation)
    rows = sorted(zip(*[column.to_pylist() for column in table.columns], strict=True), key=lambda row: row[0])

    assert [row[0] for row in rows] == [0, 1, 2, 3]
    assert all(row[1] == driver_pid for row in rows)
    assert all(row[2] != driver_pid for row in rows)
    assert all(row[3] for row in rows)
