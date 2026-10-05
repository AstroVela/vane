# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Analytical execution, mixed-mode admission and recovery through public APIs."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

import vane
from tests.fast.test_analytical_exchange import EXPRESSIONS
from tests.fast.test_ray_recovery_runtime import assert_idle, options
from tests.fast.test_ray_recovery_runtime import resources as recovery_resources
from vane.execution.direct_exchange import DirectExchangeLimits

pytestmark = [pytest.mark.real_ray, pytest.mark.usefixtures("ray_local")]


def resources(tmp_path, **changes):
    base = recovery_resources(tmp_path, **changes)
    # Analytical plans retain more partition objects across stage barriers.
    # These small test inputs need a 4 MiB bound per object, including retries.
    return replace(base, exchange_stores=tuple(replace(store, object_bytes=4 << 20) for store in base.exchange_stores))


QUERIES = [
    "select * from (select avg(case when range=0 then 9007199254740993::bigint else 2 end) a from range(3)) "
    "where a > 3002399751580332::double",
    "select range % 7 k, sum(range) s, count(*) n, avg(range) a from range(101) group by k order by k",
    "select count(*), sum(range), avg(range) from range(0)",
    "select range % 4 k, count(distinct range % 7), sum(range) filter(where range % 5 = 0) from range(103) group by k order by k",
    "select a.range from range(101) a join range(3) b on a.range % 7 = b.range order by a.range limit 13 offset 5",
    "select a.range a, b.range b from range(101) a full join range(121) b using(range) order by b.range nulls last, a.range",
    "select range % 13 k, sum(range::decimal(15,2)) s from range(100) group by k order by s desc limit 5",
]


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_public_analytical_sql_and_repeated_execution(tmp_path, mode):
    with (
        vane.connect(backend="local") as local,
        vane.connect(backend="ray", resources=resources(tmp_path)) as connection,
    ):
        for sql in QUERIES + QUERIES[:1]:
            expected = local.execute(sql).to_arrow_table().to_pylist()
            with connection.query(sql, options=options(execution=60) if mode == "fte" else None) as result:
                assert result.collect().to_pylist() == expected
                diagnostic = result.diagnostics()
                assert diagnostic["execution_state"] == "SUCCEEDED"
                assert diagnostic["execution"]["mode"] == mode
                assert diagnostic["cleanup"]["complete"]
            assert_idle(connection)
            assert connection.query_runtime.pool.admission.snapshot()["reservations"] == {}


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_public_analytical_type_roundtrip(tmp_path, mode):
    # Exercise every supported type across stages; IPC and Flight must agree on
    # field names, decimals, timezone, map/list children and null validity.
    columns = ", ".join(f"case when range % 4=0 then null else {expr} end c{i}" for i, expr in enumerate(EXPRESSIONS))
    sql = f"select range, {columns} from range(13) order by range desc"
    limits = replace(resources(tmp_path), exchange=DirectExchangeLimits(32768, 8192, 8, 2))
    with vane.connect(backend="local") as local, vane.connect(backend="ray", resources=limits) as connection:
        expected = local.execute(sql).to_arrow_table().to_pylist()
        with connection.query(sql, options=options(execution=60) if mode == "fte" else None) as result:
            assert result.collect().to_pylist() == expected
        assert_idle(connection)


def test_waiting_pipelined_graph_precedes_next_fte_attempt(tmp_path, monkeypatch):
    from vane.execution.recovery_runtime import RecoveryScheduler

    entered, proceed = threading.Event(), threading.Event()
    original = RecoveryScheduler._release
    held = []

    def hold_first(owner, attempt):
        if not held:
            held.append(attempt)
            entered.set()
            assert proceed.wait(20)
        return original(owner, attempt)

    monkeypatch.setattr(RecoveryScheduler, "_release", hold_first)
    # FTE occupies one context. The graph requires three contexts and
    # must queue atomically; later FTE tasks cannot overtake it.
    limits = resources(tmp_path, worker_count=1, partitions=1, task_contexts_per_worker=3)
    with vane.connect(backend="ray", resources=limits) as connection, connection.cursor() as sibling:
        fte = connection.query("select range from range(19) order by range", options=options(execution=60))
        try:
            assert entered.wait(10)
            with ThreadPoolExecutor() as executor:
                # A join introduces multiple contexts even with one partition.
                query = executor.submit(sibling.query, "select a.range from range(10) a join range(3) b using(range)")
                deadline = time.monotonic() + 10
                while not connection.query_runtime.pool.admission.snapshot()["waiting"]:
                    assert not query.done(), query.exception() if query.done() else None
                    assert time.monotonic() < deadline
                    time.sleep(0.005)
                proceed.set()
                with query.result(20) as result:
                    assert sorted(result.collect().column(0).to_pylist()) == [0, 1, 2]
            assert fte.collect().column(0).to_pylist() == list(range(19))
        finally:
            proceed.set()
            fte.close()
        assert_idle(connection)
        assert connection.query_runtime.pool.admission.snapshot()["waiting"] == []


@pytest.mark.parametrize("capacity_wait", [False, True])
def test_analytical_fte_retries_uncommitted_build_stage(tmp_path, monkeypatch, capacity_wait):
    import ray

    from vane.execution.pipelined_runtime import WorkerPool
    from vane.execution.recovery_runtime import RecoveryScheduler

    dispatch = RecoveryScheduler._dispatch
    replace_worker = WorkerPool.replace
    lost = []

    def replace_and_release(pool, *args):
        try:
            return replace_worker(pool, *args)
        finally:
            pool.admission.release("test-pressure")

    monkeypatch.setattr(WorkerPool, "replace", replace_and_release)

    def lose_once(owner, index, partition, binding, upstream):
        if capacity_wait and not lost and upstream and index == 0:
            demand = {name: 0 for name in owner.pool.admission.capacity}
            demand["contexts"] = owner.resources.task_contexts_per_worker
            assert owner.pool.admission.try_acquire("test-pressure", "pressure", {1: demand})
        admitted = dispatch(owner, index, partition, binding, upstream)
        if admitted and not lost and upstream:
            attempt = owner.active[index]
            lost.append(attempt.reservation.token)
            ray.kill(attempt.worker, no_restart=True)
        return admitted

    monkeypatch.setattr(RecoveryScheduler, "_dispatch", lose_once)
    sql = "select a.range % 7 k, sum(b.range) s from range(200) a join range(100) b using(range) group by k order by k"
    with (
        vane.connect(backend="local") as local,
        vane.connect(backend="ray", resources=resources(tmp_path)) as connection,
    ):
        expected = local.execute(sql).to_arrow_table().to_pylist()
        with connection.query(sql, options=options(execution=90)) as result:
            scheduler = result.context._reader
            assert result.collect().to_pylist() == expected
            assert len(lost) == 1
            attempts = [t for t in scheduler.history if t.task_id == lost[0].task_id]
            assert len(attempts) == 2
            assert attempts[0].input_id == attempts[1].input_id
            assert attempts[0].fence != attempts[1].fence
        assert_idle(connection)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_analytical_parquet_join_preserves_source_contract(tmp_path, mode):
    import pyarrow as pa
    import pyarrow.parquet as pq

    for part in range(3):
        pq.write_table(
            pa.table({"k": [None, 0, 1, 2], "v": [part, part + 10, part + 20, part + 30]}),
            tmp_path / f"part{part}.parquet",
        )
    sql = (
        f"select b.range k, sum(a.v) s, count(*) n from read_parquet('{tmp_path}/part*.parquet') a "
        "join range(3) b on a.k=b.range group by b.range order by s desc"
    )
    with (
        vane.connect(backend="local") as local,
        vane.connect(backend="ray", resources=resources(tmp_path)) as connection,
    ):
        expected = local.execute(sql).to_arrow_table().to_pylist()
        with connection.query(sql, options=options(execution=60) if mode == "fte" else None) as result:
            assert result.collect().to_pylist() == expected
        assert_idle(connection)
