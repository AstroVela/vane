# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Public API differential and repeated lifecycle acceptance on real Ray workers."""

import gc
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace

import pytest

import vane
from tests.execution_acceptance import Case, Evidence, assert_idle, compare, corpus, native_reference
from tests.fast.test_ray_analytical_execution import resources
from tests.fast.test_ray_recovery_runtime import close_result, options
from vane.execution.direct_exchange import DirectExchangeLimits

pytestmark = [pytest.mark.real_ray, pytest.mark.usefixtures("ray_local")]


def query_options(mode):
    return (
        options(execution=90, delivery=90)
        if mode == "fte"
        else vane.QueryExecutionOptions(vane.RayExecution(), 30, 90, 90)
    )


@pytest.mark.parametrize("seed", [0, 970])
@pytest.mark.parametrize("partitions,threads", [(1, 1), (2, 2), (3, 1)])
@pytest.mark.timeout(300)
def test_seeded_sql_and_type_differential(tmp_path, seed, partitions, threads):
    evidence = Evidence(tmp_path / "differential", seed=seed, partitions=partitions, threads=threads)
    cases = corpus(evidence.directory / "inputs", seed)
    limits = replace(
        resources(tmp_path, partitions=partitions, cpus_per_worker=threads, io_concurrency=128),
        exchange=DirectExchangeLimits(32768, 8192, 3, 2),
        staging_buffer_bytes=64 << 20,
    )
    evidence.write("resources-config.json", asdict(limits))
    completed = []
    with vane.connect(config={"threads": 1}) as local, vane.connect(backend="ray", resources=limits) as connection:
        for case in cases:
            expected, schema = native_reference(local, case)
            for mode in ("pipelined", "fte"):
                result = None
                evidence.begin(case, mode=mode)
                try:
                    result = connection.query(case.sql, options=query_options(mode), rows_per_batch=3)
                    actual = result.collect()
                    compare(case, expected, actual, schema)
                    assert result.execution_state == "SUCCEEDED"
                    del actual
                    close_result(result)
                    assert_idle(connection)
                    completed.append({"case": case.name, "mode": mode, "rows": expected.num_rows})
                    evidence.write("completed.json", completed)
                except BaseException as error:
                    evidence.failed(error, connection, result)
                    raise
                finally:
                    if result is not None:
                        close_result(result)
        evidence.write("report.json", {"status": "passed", "checks": completed})


@pytest.mark.parametrize("stop", ["close", "interrupt"])
@pytest.mark.timeout(150)
def test_mixed_queries_repeatedly_return_all_ownership(tmp_path, stop):
    from vane.execution.request_admission import RequestCancelled

    limits = replace(resources(tmp_path), exchange=DirectExchangeLimits(4096, 1024, 4, 2), result_buffer_bytes=1024)
    evidence = Evidence(tmp_path / "lifecycle", stop=stop)
    case = Case("slow_consumer", "select range id from range(100000)")
    with vane.connect(backend="ray", resources=limits) as connection, connection.cursor() as sibling:
        for iteration in range(3):
            result = None
            evidence.begin(case, iteration=iteration)
            try:
                result = connection.query(case.sql, rows_per_batch=4, options=query_options("pipelined"))
                retained = result.read_batch()
                saved = retained.to_pylist()
                assert saved and not result.context.production_done
                # A separate FTE query must make progress while the pipeline is
                # backpressured and its client keeps an exported Arrow view.
                with sibling.query("select sum(range)::bigint s from range(31)", options=query_options("fte")) as short:
                    assert short.collect().column(0).to_pylist() == [465]
                if stop == "interrupt":
                    connection.interrupt()
                    with pytest.raises(RequestCancelled):
                        result.read_batch()
                close_result(result)
                assert retained.to_pylist() == saved
                state = connection.query_runtime.resource_snapshot()
                assert state["result_delivery"]["usage_bytes"] > 0
                evidence.write(f"retained-{iteration}.json", state)
                del retained
                gc.collect()
                assert_idle(connection)
                with sibling.query("select 7", options=query_options("pipelined")) as short:
                    assert short.collect().column(0).to_pylist() == [7]
                assert_idle(connection)
            except BaseException as error:
                evidence.failed(error, connection, result)
                raise
            finally:
                if result is not None:
                    close_result(result)
        evidence.write("report.json", {"status": "passed", "rounds": 3})


@pytest.mark.timeout(180)
@pytest.mark.ray_fault
def test_repeated_worker_loss_replays_fixed_inputs_and_keeps_pool_usable(tmp_path, monkeypatch):
    import ray

    from vane.execution.recovery_runtime import RecoveryScheduler

    evidence = Evidence(tmp_path / "recovery")
    case = corpus(evidence.directory / "inputs", 970)[1]
    dispatch = RecoveryScheduler._dispatch
    killed = []

    def lose_first_uncommitted(owner, index, partition, binding, upstream):
        admitted = dispatch(owner, index, partition, binding, upstream)
        if admitted and upstream and owner.spec.query_id not in {entry[0] for entry in killed}:
            attempt = owner.active[index]
            killed.append((owner.spec.query_id, attempt.reservation.token.task_id))
            ray.kill(attempt.worker, no_restart=True)
        return admitted

    monkeypatch.setattr(RecoveryScheduler, "_dispatch", lose_first_uncommitted)
    with vane.connect() as local, vane.connect(backend="ray", resources=resources(tmp_path)) as connection:
        expected, schema = native_reference(local, case)
        for iteration in range(2):
            result = None
            evidence.begin(case, iteration=iteration, mode="fte")
            try:
                result = connection.query(case.sql, options=query_options("fte"))
                scheduler = result.context._reader
                actual = result.collect()
                compare(case, expected, actual, schema)
                assert len(killed) == iteration + 1
                attempts = [t for t in scheduler.history if t.task_id == killed[-1][1]]
                assert len(attempts) == 2
                assert attempts[0].input_id == attempts[1].input_id
                assert attempts[0].fence != attempts[1].fence
                del actual
                assert_idle(connection)
                with connection.query("select 7", options=query_options("pipelined")) as next_query:
                    assert next_query.collect().column(0).to_pylist() == [7]
                assert_idle(connection)
            except BaseException as error:
                evidence.failed(error, connection, result)
                raise
            finally:
                if result is not None:
                    close_result(result)
        evidence.write("report.json", {"status": "passed", "worker_losses": len(killed)})


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
@pytest.mark.timeout(120)
def test_cancelled_admission_does_not_poison_the_next_query(tmp_path, mode):
    from vane.execution.request_admission import RequestCancelled

    evidence = Evidence(tmp_path / "admission", mode=mode)
    limits = resources(tmp_path, max_active_queries=1)
    with vane.connect(backend="ray", resources=limits) as connection, connection.cursor() as sibling:
        for iteration in range(3):
            held = connection.query("select 1", options=query_options(mode))
            evidence.begin(Case("queued_cancel", "select 2"), iteration=iteration)
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    pending = executor.submit(sibling.query, "select 2", options=query_options(mode))
                    try:
                        deadline = time.monotonic() + 10
                        while connection.query_runtime.resource_snapshot()["request_admission"]["queued_requests"] != 1:
                            assert not pending.done(), pending.exception() if pending.done() else None
                            assert time.monotonic() < deadline
                            time.sleep(0.005)
                        sibling.interrupt()
                        with pytest.raises(RequestCancelled):
                            pending.result(timeout=10)
                    finally:
                        sibling.interrupt()
                        close_result(held)
                assert_idle(connection)
                with sibling.query("select 7", options=query_options(mode)) as result:
                    assert result.collect().column(0).to_pylist() == [7]
                assert_idle(connection)
            except BaseException as error:
                evidence.failed(error, connection, held)
                raise
            finally:
                close_result(held)
        evidence.write("report.json", {"status": "passed", "rounds": 3})


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
@pytest.mark.timeout(120)
def test_unsupported_queries_fail_without_fallback_and_release_admission(tmp_path, mode):
    queries = [
        "select row_number() over () from range(9)",
        "select median(range) from range(9)",
        "select a.range from range(2) a join range(3) b on a.range<b.range",
        "select '00000000-0000-0000-0000-000000000001'::uuid",
    ]
    evidence = Evidence(tmp_path / "unsupported", mode=mode)
    with vane.connect() as local, vane.connect(backend="ray", resources=resources(tmp_path)) as connection:
        for sql in queries:
            assert local.execute(sql).fetchall()
            evidence.begin(Case("unsupported", sql), mode=mode)
            try:
                with pytest.raises(vane.NotImplementedException):
                    connection.query(sql, options=query_options(mode))
                assert_idle(connection)
            except BaseException as error:
                evidence.failed(error, connection)
                raise
        with connection.query("select 7", options=query_options(mode)) as result:
            assert result.collect().column(0).to_pylist() == [7]
        assert_idle(connection)
        evidence.write("report.json", {"status": "passed", "rejected_queries": queries})


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
@pytest.mark.timeout(240)
def test_repeated_long_filter_keeps_the_declared_deadline(tmp_path, mode):
    from vane.execution.request_admission import RequestExecutionTimeout

    evidence = Evidence(tmp_path / "long-filter", mode=mode)
    sql = "select range from range(30000) where hash(upper('" + "ß" * 16000 + "' || range::varchar)) = 0"
    limits = resources(tmp_path, worker_count=1, partitions=1, max_active_queries=1)
    with vane.connect(backend="ray", resources=limits) as connection:
        # Include warmup and a successful query after every cancellation. Keep
        # native computation and the real status/Flight watchers enabled.
        connection.query("select 1", options=query_options(mode)).collect()
        for iteration, deadline in enumerate((60, 2, 60, 2)):
            result = None
            evidence.begin(Case("long-filter", sql), iteration=iteration, execution_timeout=deadline)
            try:
                opts = replace(query_options(mode), execution_timeout=deadline)
                result = connection.query(sql, options=opts)
                if deadline == 2:
                    with pytest.raises(RequestExecutionTimeout):
                        result.collect()
                else:
                    assert result.collect().num_rows == 0
                    assert result.execution_state == "SUCCEEDED"
                close_result(result)
                assert_idle(connection)
                connection.query("select 7", options=query_options(mode)).collect()
                assert_idle(connection)
            except BaseException as error:
                evidence.failed(error, connection, result)
                raise
            finally:
                if result is not None:
                    close_result(result)
        evidence.write("report.json", {"status": "passed", "rounds": 4})
