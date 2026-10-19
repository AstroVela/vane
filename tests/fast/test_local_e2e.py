# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys

import pyarrow.parquet as pq
import pytest

import vane
from vane.execution.request_admission import RequestAdmissionLimits
from vane.execution.resources import ResourceVector


@pytest.mark.parametrize("entrypoint", ["root", "runners", "local"])
def test_set_runner_local_selects_native_execution(monkeypatch, entrypoint):
    from vane import runners
    from vane.runners import local

    monkeypatch.setenv("VANE_RUNNER", "ray")
    setter = {"root": vane.set_runner_local, "runners": runners.set_runner_local, "local": local.set_runner_local}[
        entrypoint
    ]
    assert setter() is None
    assert vane.current_config().runner == "local"
    assert vane.get_runner() is None
    with vane.connect() as connection:
        assert connection.execute("SELECT 42").fetchall() == [(42,)]
        assert connection.sql("SELECT 1")._get_runner_type() == "local"


@pytest.mark.parametrize("option", ["num_workers", "max_running_tasks", "execution_mode"])
def test_set_runner_local_rejects_removed_fte_options(option):
    with pytest.raises(TypeError, match=option):
        vane.set_runner_local(**{option: 1})


@pytest.mark.parametrize("runner", ["local", "local-fast"])
def test_local_native_write_and_repartition(monkeypatch, tmp_path, runner):
    monkeypatch.setenv("VANE_RUNNER", runner)
    target = tmp_path / "output.parquet"
    with vane.connect() as connection:
        connection.sql("SELECT i::INTEGER AS x, (i % 5)::INTEGER AS k FROM range(100) t(i)").filter(
            "x >= 10 AND x < 90"
        ).repartition(4).write_parquet(str(target))
        assert connection.read_parquet(str(target)).aggregate("count(*), sum(x), min(k), max(k)").fetchone() == (
            80,
            3960,
            0,
            4,
        )
    assert target.is_file()
    assert pq.read_table(target).num_rows == 80


@pytest.mark.parametrize("runner", ["local", "local-fast"])
def test_public_local_configuration_supports_shared_model_runtime(monkeypatch, runner):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    assert vane.configure(runner=runner).runner == "local"

    @vane.cls(actor_number=1, return_dtype="BIGINT")
    class Add:
        def __init__(self, offset):
            self.offset = offset

        def __call__(self, value):
            return value + self.offset

    with vane.connect() as connection:
        runtime = connection.configure_local_runtime(
            request_limit=RequestAdmissionLimits(1, 1),
            resident_limit=ResourceVector(cpu=1),
        )
        model = runtime.register_model("add", Add(10), version="v1", parameters=["BIGINT"], cpus=1)
        vane.attach_function(model, "add_value", connection=connection)
        with connection.cursor() as cursor:
            assert cursor.execute("SELECT add_value(?)", [2]).fetchall() == [(12,)]
            assert cursor.sql("SELECT 3::BIGINT AS x").project(model(vane.col("x"))).fetchall() == [(13,)]
        assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0
    assert runtime.resource_snapshot()["closed"]


@pytest.mark.parametrize("runner", ["local", "local-fast"])
def test_native_local_execution_does_not_start_ray(tmp_path, runner):
    script = r"""
import sys
import ray
def blocked_init(*args, **kwargs):
    raise AssertionError("native local execution must not start Ray")
ray.init = blocked_init
import vane
with vane.connect() as connection:
    connection.sql("SELECT i AS x FROM range(10) t(i)").write_parquet(sys.argv[1])
    assert connection.read_parquet(sys.argv[1]).aggregate("count(*), sum(x)").fetchone() == (10, 45)
assert vane.get_runner() is None
assert not ray.is_initialized()
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(tmp_path / "without-ray-runtime.parquet")],
        env={**os.environ, "VANE_RUNNER": runner},
        check=True,
        timeout=30,
    )


@pytest.mark.parametrize("sink", ["parquet", "datasink"])
def test_local_subprocess_actors_write_across_repartition(tmp_path, sink):
    script = """
import sys
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import vane
from vane.datasink import BoundDataSink, DataSink, DataSinkWorker, WriteOutcome, WriteResult

class AddOne:
    def __call__(self, batch):
        return pa.table({"x": pc.add(batch.column(0), 1)})

class AddTen:
    def __call__(self, batch):
        return pa.table({"x": pc.add(batch.column(0), 10)})

class Worker(DataSinkWorker):
    def __init__(self, path):
        self.path = path

    def write(self, table):
        pq.write_to_dataset(table, self.path)
        return WriteResult(rows_received=table.num_rows, rows_affected=table.num_rows, bytes_received=table.nbytes)

class Bound(BoundDataSink):
    def __init__(self, path):
        self.path = path

    def open_worker(self, context):
        return Worker(self.path)

class Sink(DataSink):
    def __init__(self, path):
        self.path = path

    def bind(self, schema):
        return Bound(self.path)

vane.set_runner_local()
try:
    with vane.connect() as conn:
        relation = conn.sql("SELECT i::BIGINT AS x FROM range(3) t(i)").map_batches(
            AddOne, schema={"x": "BIGINT"}, execution_backend="subprocess_actor", actor_number=1
        ).repartition(2).map_batches(
            AddTen, schema={"x": "BIGINT"}, execution_backend="subprocess_actor", actor_number=1
        )
        if sys.argv[2] == "parquet":
            relation.write_parquet(sys.argv[1])
        else:
            summary = relation.write_datasink(Sink(sys.argv[1]))
            assert summary.outcome is WriteOutcome.APPLIED and summary.rows_received == 3
    assert sorted(pq.read_table(sys.argv[1]).column(0).to_pylist()) == [11, 12, 13]
finally:
    vane.teardown_runner()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(tmp_path / "actor.parquet"), sink],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
