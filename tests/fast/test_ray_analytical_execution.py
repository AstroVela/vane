# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Analytical execution, mixed-mode admission and recovery through public APIs."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

import vane
from tests.fast.test_analytical_exchange import (
    EXPRESSIONS,
    HUGEINT_EXPRESSIONS,
    HUGEINT_VALUE_SQL,
    NESTED_ARRAY_EXPRESSIONS,
    TEMPORAL_VALUES,
    exchange_expected,
)
from tests.fast.test_analytical_fragment_compiler import (
    DECIMAL_SUM_QUERIES,
    HUGEINT_SUM_QUERIES,
    INTERNAL_AGGREGATE_REWRITES,
    ORDERED_AGGREGATES,
    PARQUET_TOP_N_TAILS,
    TOP_N_AGGREGATES,
    WIDE_AGGREGATE_TYPES,
    parquet_top_n_source,
    scaled_wide_value,
    wide_aggregate_queries,
)
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
        expected = exchange_expected(local.execute(sql).to_arrow_table())
        with connection.query(sql, options=options(execution=60) if mode == "fte" else None) as result:
            table = result.collect()
            table.validate(full=True)
            assert table.equals(expected.cast(table.schema))
        assert_idle(connection)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_public_nested_arrays(tmp_path, mode):
    columns = ", ".join(
        f"case when range % 4=0 then null else {expr} end c{i}" for i, expr in enumerate(NESTED_ARRAY_EXPRESSIONS)
    )
    sql = f"select range, {columns} from range(13) order by range desc"
    limits = replace(resources(tmp_path), exchange=DirectExchangeLimits(32768, 8192, 3, 2))
    with vane.connect(backend="local") as local, vane.connect(backend="ray", resources=limits) as connection:
        expected = local.execute(sql).to_arrow_table()
        with connection.query(sql, options=options(execution=60) if mode == "fte" else None) as result:
            table = result.collect()
            table.validate(full=True)
            assert table.equals(expected.cast(table.schema))
        assert_idle(connection)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_public_min_max_overloads(tmp_path, mode):
    with (
        vane.connect(backend="local") as local,
        vane.connect(backend="ray", resources=resources(tmp_path)) as connection,
    ):
        for sql in TOP_N_AGGREGATES:
            expected = local.execute(sql).to_arrow_table().to_pylist()
            with connection.query(sql, options=options(execution=60) if mode == "fte" else None) as result:
                assert result.collect().to_pylist() == expected
            assert_idle(connection)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_public_parquet_top_n_with_default_optimizers(tmp_path, mode):
    with (
        vane.connect(backend="local") as local,
        vane.connect(backend="ray", resources=resources(tmp_path, partitions=2)) as connection,
    ):
        source = parquet_top_n_source(local, tmp_path)
        for tail in PARQUET_TOP_N_TAILS:
            sql = f"select k, v from {source} {tail}"
            expected = local.execute(sql).to_arrow_table()
            with connection.query(sql, options=options(execution=60) if mode == "fte" else None) as result:
                assert result.collect().equals(expected)
            assert_idle(connection)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_public_decimal_sum_keeps_native_precision(tmp_path, mode):
    with (
        vane.connect(backend="local") as local,
        vane.connect(backend="ray", resources=resources(tmp_path, partitions=2)) as connection,
    ):
        for sql in DECIMAL_SUM_QUERIES:
            expected = local.execute(sql).to_arrow_table()
            with connection.query(sql, options=options(execution=60) if mode == "fte" else None) as result:
                table = result.collect()
                table.validate(full=True)
                assert table.equals(expected)
            assert_idle(connection)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
@pytest.mark.parametrize("scale", [0, 5, 38])
def test_public_nested_decimal_aggregation(tmp_path, mode, scale):
    with (
        vane.connect(backend="local", config={"threads": 1}) as local,
        vane.connect(backend="ray", resources=resources(tmp_path, partitions=2)) as connection,
    ):
        for sign in [1, -1]:
            positive = scaled_wide_value(sign * 4 * 10**37, scale)
            negative = scaled_wide_value(-sign * 3 * 10**37, scale)
            groups = (
                "select range%2 k, sum(case when range%2=0 "
                f"then '{positive}'::decimal(38,{scale}) else '{negative}'::decimal(38,{scale}) end) s "
                "from range(6) group by k"
            )
            queries = [
                f"select sum(s) total from ({groups})",
                # Carry the same 39-digit intermediate through another
                # aggregate stage before cancellation produces a legal result.
                f"select sum(s) total from (select k, max(s) s from ({groups}) group by k)",
            ]
            for sql in queries:
                expected = local.execute(sql).to_arrow_table()
                expected.validate(full=True)
                with connection.query(sql, options=options(execution=60) if mode == "fte" else None) as result:
                    table = result.collect()
                    table.validate(full=True)
                    assert table.equals(expected), sql
                assert_idle(connection)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
@pytest.mark.parametrize(
    "queries",
    [ORDERED_AGGREGATES, INTERNAL_AGGREGATE_REWRITES, HUGEINT_SUM_QUERIES],
    ids=["ordered", "rewrites", "hugeint"],
)
def test_public_native_aggregate_semantics(tmp_path, mode, queries):
    with (
        vane.connect(backend="local", config={"threads": 1}) as local,
        vane.connect(backend="ray", resources=resources(tmp_path, partitions=2)) as connection,
    ):
        for sql in queries:
            expected = exchange_expected(local.execute(sql).to_arrow_table())
            with connection.query(sql, options=options(execution=60) if mode == "fte" else None) as result:
                table = result.collect()
                table.validate(full=True)
                assert table.equals(expected.cast(table.schema)), sql
            assert_idle(connection)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_public_wide_numeric_aggregates(tmp_path, mode):
    with (
        vane.connect(backend="local", config={"threads": 1}) as local,
        vane.connect(backend="ray", resources=resources(tmp_path, partitions=2)) as connection,
    ):
        for kind, scale in WIDE_AGGREGATE_TYPES:
            for sql in wide_aggregate_queries(kind, scale):
                expected = local.execute(sql).to_arrow_table()
                with connection.query(sql, options=options(execution=60) if mode == "fte" else None) as result:
                    table = result.collect()
                    table.validate(full=True)
                    assert table.equals(expected.cast(table.schema)), sql
                assert_idle(connection)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_public_hugeint_full_range(tmp_path, mode):
    columns = ", ".join(f"{expr} c{i}" for i, expr in enumerate(HUGEINT_EXPRESSIONS))
    queries = [
        f"select range, {columns} from range(9) order by range desc",
        "select sum(case when range=0 then '100000000000000000000000000000000000000'::hugeint "
        "else 0::hugeint end)::varchar from range(4)",
        f"select {HUGEINT_VALUE_SQL} v, count(*) from range(9) group by v order by v",
        "select null::hugeint v where false",
    ]
    limits = replace(resources(tmp_path), exchange=DirectExchangeLimits(32768, 8192, 8, 2))
    with vane.connect(backend="local") as local, vane.connect(backend="ray", resources=limits) as connection:
        for sql in queries:
            expected = local.execute(sql).to_arrow_table().to_pylist()
            with connection.query(sql, options=options(execution=60) if mode == "fte" else None) as result:
                table = result.collect()
                table.validate(full=True)
                assert table.to_pylist() == expected
            assert_idle(connection)


@pytest.mark.parametrize("cancel", ["close", "interrupt"])
def test_fte_cancel_racing_with_admission_cannot_leave_waiter(tmp_path, monkeypatch, cancel):
    from vane.execution.recovery_runtime import RecoveryScheduler

    entered, enqueue, cancellation_progress = [threading.Event() for _ in range(3)]
    cancel_thread = []
    schedulers = []

    class ObservedLock:
        """Release the race barrier once cancellation blocks or clears waiters."""

        def __init__(self):
            self.lock = threading.RLock()

        def __enter__(self):
            if threading.get_ident() in cancel_thread and not self.lock.acquire(blocking=False):
                cancellation_progress.set()
                self.lock.acquire()
            elif threading.get_ident() not in cancel_thread:
                self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    initialize = RecoveryScheduler.__init__

    def observe_lock(owner, *args):
        initialize(owner, *args)
        owner.lock = ObservedLock()
        schedulers.append(owner)

    monkeypatch.setattr(RecoveryScheduler, "__init__", observe_lock)
    limits = resources(tmp_path, worker_count=1, partitions=1)
    with vane.connect(backend="ray", resources=limits) as connection:
        assert connection.query("select 1").collect().column(0).to_pylist() == [1]
        manager = connection.query_runtime.pool.admission
        pressure = {name: 0 for name in manager.capacity}
        pressure["contexts"] = limits.task_contexts_per_worker
        assert manager.try_acquire("pressure", "pressure", {0: pressure})
        acquire, cancel_waiting = manager.try_acquire, manager.cancel_waiting

        def pause_enqueue(token, *args):
            if token.startswith("fte/"):
                entered.set()
                assert enqueue.wait(10)
            return acquire(token, *args)

        def observe_cleared(query):
            cancel_waiting(query)
            # QueryResult may cancel before scheduler.close(). Wait for the
            # final cancellation immediately before joining the dispatch thread.
            if any(owner.context.query_id == query and owner.cleanup_lock.locked() for owner in schedulers):
                cancellation_progress.set()

        monkeypatch.setattr(manager, "try_acquire", pause_enqueue)
        monkeypatch.setattr(manager, "cancel_waiting", observe_cleared)
        result = connection.query("select range from range(9)", options=options(execution=60))

        def stop():
            cancel_thread.append(threading.get_ident())
            if cancel == "interrupt":
                connection.interrupt()
            result.close()

        try:
            with ThreadPoolExecutor() as executor:
                try:
                    assert entered.wait(10)
                    closing = executor.submit(stop)
                    assert cancellation_progress.wait(5)
                finally:
                    enqueue.set()
                closing.result(15)
            assert manager.snapshot()["waiting"] == []
            manager.release("pressure")
            # Admission includes planning and a fresh result actor's startup.
            # The empty FIFO assertion above detects leaked queue ownership.
            next_options = vane.QueryExecutionOptions(vane.RayExecution("pipelined"), 10, 30, 30)
            with connection.query("select 7", options=next_options) as next_result:
                assert next_result.collect().column(0).to_pylist() == [7]
            assert_idle(connection)
        finally:
            enqueue.set()
            result.close()
            manager.release("pressure")


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_temporal_domain_across_aggregate_stages(tmp_path, mode):
    with (
        vane.connect(backend="local") as local,
        vane.connect(backend="ray", resources=resources(tmp_path)) as connection,
    ):
        for expression, value in TEMPORAL_VALUES:
            sql = f"select min(v)::varchar, max(v)::varchar from (select case when range % 3=0 then null else {expression} end v from range(13))"
            expected = local.execute(sql).to_arrow_table().to_pylist()
            with connection.query(sql, options=options(execution=60) if mode == "fte" else None) as result:
                assert result.collect().to_pylist() == expected
            sql = f"select {expression} v from range(2)"
            with connection.query(sql, options=options(execution=60) if mode == "fte" else None) as result:
                table = result.collect()
                table.validate(full=True)
                assert table.to_pylist() == [dict(v=value)] * 2
            assert_idle(connection)


@pytest.mark.parametrize("outcome", ["admitted", "timeout", "cancelled"])
def test_worker_admission_has_its_own_clock(tmp_path, outcome):
    from vane.execution.request_admission import RequestCancelled, RequestQueueTimeout

    limits = resources(tmp_path, worker_count=1, partitions=1, task_contexts_per_worker=1)
    with vane.connect(backend="ray", resources=limits) as connection, connection.cursor() as sibling:
        connection.query("select 0").collect()
        held = connection.query("select 1")
        manager = connection.query_runtime.pool.admission
        opts = vane.QueryExecutionOptions(vane.RayExecution(), 4 if outcome == "timeout" else 15, 2, 30)
        try:
            with ThreadPoolExecutor() as executor:
                try:
                    pending = executor.submit(sibling.query, "select 2", options=opts)
                    deadline = time.monotonic() + 10
                    while not manager.snapshot()["waiting"]:
                        assert not pending.done(), pending.exception() if pending.done() else None
                        assert time.monotonic() < deadline
                        time.sleep(0.01)
                    query_id = manager.snapshot()["waiting"][0]["query_id"]
                    context = connection.query_runtime._contexts[query_id]
                    assert context.state == "ADMISSION_WAIT"
                    if outcome == "admitted":
                        time.sleep(2.2)
                        assert not pending.done()
                        held.collect()
                        with pending.result(10) as result:
                            assert result.collect().column(0).to_pylist() == [2]
                            timing = result.context._ticket.timing_snapshot()
                            assert timing["queue_wait_seconds"] >= 2.2
                            assert timing["execution_seconds"] < 2
                    else:
                        if outcome == "cancelled":
                            sibling.interrupt()
                        with pytest.raises(RequestCancelled if outcome == "cancelled" else RequestQueueTimeout):
                            pending.result(10)
                        assert context._ticket.timing_snapshot()["execution_seconds"] is None
                    assert manager.snapshot()["waiting"] == []
                finally:
                    held.close()
        finally:
            held.close()
        assert connection.query("select 7").collect().column(0).to_pylist() == [7]
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
