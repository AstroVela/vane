# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real Ray recovery on the shared native plan, task and result protocols."""

import time
from dataclasses import replace

import pytest

import vane
from vane.execution.direct_exchange import DirectExchangeLimits

pytestmark = [pytest.mark.real_ray, pytest.mark.usefixtures("ray_local")]


def resources(tmp_path, **changes):
    store = vane.ExchangeStore(
        "shared",
        str(tmp_path / "store"),
        capacity_bytes=1 << 30,
        query_bytes=256 << 20,
        source_bytes=32 << 20,
        object_bytes=32 << 20,
    )
    return replace(
        vane.RayResources(), exchange=DirectExchangeLimits(4096, 1024, 64, 2), exchange_stores=(store,), **changes
    )


def options(*, attempts=3, backoff=0, execution=30, delivery=30):
    return vane.QueryExecutionOptions(
        vane.RayExecution("fte", vane.FteOptions("shared", attempts, backoff)), 30, execution, delivery
    )


def assert_idle(connection, *, allow_dead_workers=False):
    import ray

    runtime = connection.query_runtime
    assert runtime.resource_snapshot()["queries"] == {}
    assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0
    assert runtime.resource_snapshot()["result_service"]["active_contexts"] == 0
    for worker in runtime.pool.workers:
        try:
            assert ray.get(worker.resources_snapshot.remote())["reservations"] == {}
        except ray.exceptions.RayActorError:
            if not allow_dead_workers:
                raise
    for store in runtime.stores.values():
        assert store.snapshot() == {"queries": 0, "reserved_bytes": 0}


def close_result(result):
    from vane.execution.result_delivery import _ResultCleanupPending

    deadline = time.monotonic() + 10
    while True:
        try:
            result.close()
            return
        except _ResultCleanupPending:
            # A delivery timer may own cleanup concurrently with the caller.
            # Keep the public close retry contract, with a finite test deadline.
            assert time.monotonic() < deadline
            time.sleep(0.01)


def test_public_fte_query_and_empty_schema(tmp_path):
    with vane.Runtime(resources(tmp_path)) as application, application.connect(execution="fte") as connection:
        for count in (50, 0):
            with connection.query(f"select range as value from range({count})") as result:
                scheduler = result.context._reader
                assert result.schema.names == ["value"]
                table = result.collect()
                assert sorted(table.column("value").to_pylist()) == list(range(count))
                assert scheduler.result_manifest is not None
                assert result.execution_state == "SUCCEEDED"
            assert_idle(connection)


def test_fte_options_and_pipelined_share_one_session_pool(tmp_path):
    with vane.Runtime(resources(tmp_path)) as application, application.connect() as connection:
        assert connection.query("select 1 as value").collect().to_pylist() == [{"value": 1}]
        workers = tuple(connection.query_runtime.pool.epochs)
        assert connection.query("select 2 as value", options=options()).collect().to_pylist() == [{"value": 2}]
        assert tuple(connection.query_runtime.pool.epochs) == workers
        assert_idle(connection)


def test_failed_orphan_cleanup_preserves_live_result_and_new_admission(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path
    from threading import Event

    from vane.execution.fte_store import replace_metadata

    with vane.Runtime(resources(tmp_path)) as application, application.connect(execution="fte") as connection:
        result = connection.query("select 42 as value")
        store = result.context._reader.store
        orphan = None
        try:
            deadline = time.monotonic() + 10
            while not result.context.production_done:
                result.context.check()
                assert time.monotonic() < deadline
                time.sleep(0.01)
            orphan = store.reserve("unrelated-orphan")
            (orphan.directory / "object.mat").write_bytes(b"orphan")
            attempted = Event()
            rmtree = shutil.rmtree

            def fail_orphan(path, *args, **kwargs):
                if Path(path) == orphan.directory:
                    attempted.set()
                    raise PermissionError("unrelated orphan deletion denied")
                return rmtree(path, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(shutil, "rmtree", fail_orphan)
                record = store.allocations / f"{orphan.value['namespace']}.json"
                replace_metadata(record, {**orphan.value, "expires": time.time() - 1})
                orphan.guard.close()
                assert attempted.wait(5), "heartbeat did not attempt orphan cleanup"
                assert result.collect().column("value").to_pylist() == [42]
                assert connection.query("select 7 as value").collect().column("value").to_pylist() == [7]
                assert store.snapshot() == {"queries": 1, "reserved_bytes": store.config.query_bytes}
            store.collect_expired()
            assert_idle(connection)
        finally:
            close_result(result)
            if orphan is not None:
                orphan.close(lambda: None)


def test_slow_orphan_cleanup_does_not_stop_heartbeats_or_result_delivery(tmp_path, monkeypatch):
    import json
    import shutil
    from pathlib import Path
    from threading import Event

    from vane.execution.fte_store import replace_metadata
    from vane.execution.materialized_store import StorageCleanupPending

    config = resources(tmp_path)
    config = replace(config, exchange_stores=(replace(config.exchange_stores[0], lease_seconds=2),))
    with vane.Runtime(config) as application, application.connect(execution="fte") as connection:
        result = connection.query("select 42 as value")
        scheduler = result.context._reader
        store = scheduler.store
        orphan = None
        entered, release = Event(), Event()
        rmtree = shutil.rmtree
        try:
            deadline = time.monotonic() + 10
            while not result.context.production_done:
                result.context.check()
                assert time.monotonic() < deadline
                time.sleep(0.01)
            orphan = store.reserve("slow-orphan")

            def slow_delete(path, *args, **kwargs):
                if Path(path) == orphan.directory:
                    entered.set()
                    assert release.wait(20), "test did not release slow orphan deletion"
                return rmtree(path, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(shutil, "rmtree", slow_delete)
                record = store.allocations / f"{orphan.value['namespace']}.json"
                replace_metadata(record, {**orphan.value, "expires": time.time() - 1})
                orphan.guard.close()
                assert entered.wait(5)
                live_record = store.allocations / f"{scheduler.lease.value['namespace']}.json"
                original = json.loads(live_record.read_bytes())["expires"]
                # Keep deletion blocked for more than two lease durations.
                # The heartbeat that requested it must continue renewing.
                end = time.monotonic() + 4.5
                while time.monotonic() < end:
                    result.context.check()
                    time.sleep(0.05)
                assert json.loads(live_record.read_bytes())["expires"] > original + 3
                assert result.collect().column("value").to_pylist() == [42]
                assert connection.query("select 7 as value").collect().column("value").to_pylist() == [7]
                assert store.snapshot() == {"queries": 1, "reserved_bytes": store.config.query_bytes}
                release.set()
                end = time.monotonic() + 5
                while record.exists():
                    assert time.monotonic() < end
                    time.sleep(0.01)
            assert_idle(connection)
        finally:
            release.set()
            close_result(result)
            if orphan is not None:
                end = time.monotonic() + 5
                while True:
                    try:
                        orphan.close(lambda: None)
                        break
                    except StorageCleanupPending:
                        assert time.monotonic() < end
                        time.sleep(0.01)


@pytest.mark.parametrize("predicate", ["", " where value > 10"])
def test_file_snapshot_precedes_optimization_and_survives_original_change(tmp_path, predicate):
    path = tmp_path / "input.parquet"
    with vane.connect(backend="local") as local:
        local.execute(f"copy (select range as value from range(1, 4)) to '{path}'")
    with vane.Runtime(resources(tmp_path)) as application, application.connect() as connection:
        with connection.query(f"select value from read_parquet('{path}')" + predicate, options=options()) as result:
            with vane.connect(backend="local") as local:
                local.execute(f"copy (select 100 as value) to '{path}' (overwrite true)")
            assert sorted(result.collect().column("value").to_pylist()) == ([] if predicate else [1, 2, 3])
        assert_idle(connection)


@pytest.mark.parametrize("partitions", [1, 2])
def test_fte_store_inside_recursive_input_glob_preserves_duplicate_rows(tmp_path, partitions):
    path = tmp_path / "input.parquet"
    sql = f"select value from read_parquet(['{path}', '{tmp_path}/**/*.parquet'])"
    with vane.connect(backend="local") as local:
        local.execute(f"copy (select 42 as value) to '{path}'")
        expected = [row[0] for row in local.execute(sql).fetchall()]
    assert expected == [42, 42]
    config = resources(tmp_path, worker_count=partitions, partitions=partitions)
    config = replace(config, exchange_stores=(replace(config.exchange_stores[0], source_bytes=path.stat().st_size),))
    with vane.Runtime(config) as application, application.connect(execution="fte") as connection:
        for _ in range(2):
            assert connection.query(sql).collect().column("value").to_pylist() == expected
            assert_idle(connection)
            assert not list((tmp_path / "store").rglob("*.parquet"))


@pytest.mark.parametrize(("option", "column"), [("true", "filename"), ("'origin'", "origin")])
def test_generated_filename_rejection_releases_fte_resources_and_preserves_pipelined(tmp_path, option, column):
    path = tmp_path / "input.parquet"
    with vane.connect(backend="local") as local:
        local.execute(f"copy (select 42 as value) to '{path}'")
    scan = f"parquet_scan('{path}', filename={option})"
    queries = (
        (f"select {column} as provenance from {scan}", [{"provenance": str(path)}]),
        (f"select value from {scan} where {column} = '{path}'", [{"value": 42}]),
    )
    with vane.Runtime(resources(tmp_path)) as application, application.connect() as connection:
        for sql, expected in queries:
            with pytest.raises(vane.NotImplementedException, match="generated filename"):
                connection.query(sql, options=options())
            assert_idle(connection)
            assert not list((tmp_path / "store").rglob("*.parquet"))
            assert connection.query(sql).collect().to_pylist() == expected
            assert_idle(connection)
        # The same FTE session can still execute a query that does not request
        # the generated path, after both failed submissions have been cleaned.
        assert connection.query(f"select value from {scan}", options=options()).collect().to_pylist() == [{"value": 42}]
        assert_idle(connection)


def test_native_hash_stages_and_fixed_inputs(tmp_path, monkeypatch):
    from vane.execution import recovery_runtime
    from vane.execution.compiler import FragmentCompileOptions

    stage = recovery_runtime.stage_ray_query

    def hash_plan(*args, **kwargs):
        kwargs["compile_options"] = FragmentCompileOptions(2, (0,))
        return stage(*args, **kwargs)

    monkeypatch.setattr(recovery_runtime, "stage_ray_query", hash_plan)
    with vane.Runtime(resources(tmp_path)) as application, application.connect() as connection:
        result = connection.query(
            "select range % 7 as k, 'value-' || range::varchar as v from range(80)", options=options()
        )
        scheduler = result.context._reader
        actual = result.collect().to_pylist()
        assert sorted((r["k"], r["v"]) for r in actual) == sorted((i % 7, f"value-{i}") for i in range(80))
        assert scheduler.coordinator.snapshot()["sealed_stages"] == 3
        assert len(scheduler.history) == 5
        assert_idle(connection)


@pytest.mark.parametrize("stage_id", ["fragment0", "fragment1"])
def test_worker_loss_replays_only_the_uncommitted_task(tmp_path, monkeypatch, stage_id):
    import ray

    from vane.execution.recovery_runtime import RecoveryScheduler

    dispatch = RecoveryScheduler._dispatch
    killed = []

    def lose_worker(owner, index, partition, binding, upstream):
        admitted = dispatch(owner, index, partition, binding, upstream)
        if not admitted:
            return False
        if not killed and binding.task.stage_id == stage_id:
            attempt = owner.active[index]
            killed.append(attempt.reservation.token)
            ray.kill(attempt.worker, no_restart=True)
        return True

    monkeypatch.setattr(RecoveryScheduler, "_dispatch", lose_worker)
    with vane.Runtime(resources(tmp_path)) as application, application.connect() as connection:
        result = connection.query("select range as value from range(100)", options=options())
        scheduler = result.context._reader
        assert sorted(result.collect().column("value").to_pylist()) == list(range(100))
        old = killed[0]
        attempts = [token for token in scheduler.history if token.task_id == old.task_id]
        assert len(attempts) == 2
        assert attempts[0].input_id == attempts[1].input_id
        assert attempts[0].worker_epoch != attempts[1].worker_epoch
        assert attempts[0].fence != attempts[1].fence
        assert len(scheduler.history) == 4
        assert_idle(connection)


def test_worker_loss_after_root_commit_does_not_change_result(tmp_path):
    import ray

    with vane.Runtime(resources(tmp_path)) as application, application.connect() as connection:
        result = connection.query("select range as value from range(100)", options=options())
        scheduler = result.context._reader
        deadline = time.monotonic() + 20
        while not result.context.production_done:
            assert time.monotonic() < deadline
            result.context.check()
            time.sleep(0.01)
        for worker in connection.query_runtime.pool.workers:
            ray.kill(worker, no_restart=True)
        assert sorted(result.collect().column("value").to_pylist()) == list(range(100))
        assert scheduler.closed
        assert all(store.snapshot()["reserved_bytes"] == 0 for store in connection.query_runtime.stores.values())


def test_attempt_exhaustion_exposes_no_rows_and_releases_storage(tmp_path, monkeypatch):
    import ray

    from vane.execution.recovery_runtime import RecoveryScheduler

    dispatch = RecoveryScheduler._dispatch
    captured = []

    def lose_every_attempt(owner, index, partition, binding, upstream):
        admitted = dispatch(owner, index, partition, binding, upstream)
        if not admitted:
            return False
        if binding.task.task_id == "fragment0/0":
            captured.append(owner.active[index].reservation.token)
            ray.kill(owner.active[index].worker, no_restart=True)
        return True

    monkeypatch.setattr(RecoveryScheduler, "_dispatch", lose_every_attempt)
    with vane.Runtime(resources(tmp_path)) as application, application.connect() as connection:
        result = connection.query("select range from range(100)", options=options(attempts=2))
        scheduler = result.context._reader
        try:
            with pytest.raises(RuntimeError, match="attempt limit exhausted"):
                result.read_batch()
            assert scheduler.result_manifest is None
            assert len(captured) == 2
            assert captured[0].input_id == captured[1].input_id
        finally:
            close_result(result)
        assert connection.query_runtime.resource_snapshot()["queries"] == {}
        assert all(s.snapshot()["reserved_bytes"] == 0 for s in connection.query_runtime.stores.values())


def test_retry_backoff_uses_the_original_execution_deadline(tmp_path, monkeypatch):
    import ray

    from vane.execution.recovery_runtime import RecoveryScheduler
    from vane.execution.request_admission import RequestExecutionTimeout

    dispatch = RecoveryScheduler._dispatch
    killed = []

    def lose_first_attempt(owner, index, partition, binding, upstream):
        admitted = dispatch(owner, index, partition, binding, upstream)
        if not admitted:
            return False
        if not killed:
            killed.append(owner.active[index].reservation.token)
            ray.kill(owner.active[index].worker, no_restart=True)
        return True

    with vane.Runtime(resources(tmp_path)) as application, application.connect() as connection:
        connection.query("select 1", options=options()).collect()
        monkeypatch.setattr(RecoveryScheduler, "_dispatch", lose_first_attempt)
        start = time.monotonic()
        result = connection.query("select range from range(10)", options=options(execution=3, backoff=10))
        scheduler = result.context._reader
        try:
            with pytest.raises(RequestExecutionTimeout):
                result.read_batch()
            assert time.monotonic() - start < 9
            assert scheduler.result_manifest is None
            assert all(t.attempt == 1 for t in scheduler.history)
        finally:
            close_result(result)
        # The deadline can also expire during replacement. Dead processes own
        # no worker reservations; the next query repairs the pool on admission.
        assert_idle(connection, allow_dead_workers=True)
        assert connection.query("select 42", options=options()).collect().column(0).to_pylist() == [42]
        assert_idle(connection)


@pytest.mark.parametrize("cancel", ["interrupt", "delivery"])
def test_cancellation_and_delivery_deadline_release_every_owner(tmp_path, cancel):
    from vane.execution.request_admission import RequestCancelled
    from vane.execution.result_delivery import ResultDeliveryTimeout

    with vane.Runtime(resources(tmp_path)) as application, application.connect() as connection:
        result = connection.query(
            "select range from range(50000)", options=options(delivery=0.1 if cancel == "delivery" else 30)
        )
        scheduler = result.context._reader
        if cancel == "interrupt":
            connection.interrupt()
            error = RequestCancelled
        else:
            time.sleep(0.2)
            error = ResultDeliveryTimeout
        try:
            with pytest.raises(error):
                result.read_batch()
        finally:
            close_result(result)
        assert scheduler.closed
        assert_idle(connection)
        assert connection.query("select 42", options=options()).collect().column(0).to_pylist() == [42]
        assert_idle(connection)


def test_committed_result_can_outlive_execution_deadline(tmp_path):
    with vane.Runtime(resources(tmp_path)) as application, application.connect() as connection:
        connection.query("select 1", options=options()).collect()
        result = connection.query("select 42 as value", options=options(execution=3))
        deadline = time.monotonic() + 2.5
        while not result.context.production_done:
            result.context.check()
            assert time.monotonic() < deadline
            time.sleep(0.01)
        time.sleep(3.1)
        assert result.collect().to_pylist() == [{"value": 42}]
        assert result.execution_state == "SUCCEEDED"
        assert_idle(connection)


@pytest.mark.parametrize("damage", ["corrupt", "missing"])
def test_committed_input_damage_is_permanent(tmp_path, monkeypatch, damage):
    from vane.execution.recovery_runtime import RecoveryScheduler

    stage = RecoveryScheduler._stage
    damaged = []

    def damage_input(owner, fragment, upstream):
        result = stage(owner, fragment, upstream)
        if not damaged:
            obj = result.attempts[0].objects[0]
            path = owner.store.store.descriptor.path(obj.key)
            if damage == "missing":
                path.unlink()
            else:
                content = path.read_bytes()
                path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
            damaged.append(obj)
        return result

    monkeypatch.setattr(RecoveryScheduler, "_stage", damage_input)
    with vane.Runtime(resources(tmp_path)) as application, application.connect() as connection:
        result = connection.query("select range from range(40)", options=options())
        scheduler = result.context._reader
        try:
            with pytest.raises(Exception, match="checksum|digest|open|file|object"):
                result.read_batch()
            assert scheduler.result_manifest is None
            assert all(t.attempt == 1 for t in scheduler.history)
        finally:
            close_result(result)
        assert_idle(connection)


def test_result_service_loss_after_partial_delivery_fails_without_replay(tmp_path):
    import ray

    with vane.Runtime(resources(tmp_path)) as application, application.connect() as connection:
        result = connection.query("select range from range(20000)", options=options())
        first = result.read_batch()
        expected = first.column(0).to_pylist()
        scheduler = result.context._reader
        attempts = len(scheduler.history)
        ray.kill(scheduler.relay, no_restart=True)
        try:
            with pytest.raises(Exception):
                while result.read_batch() is not None:
                    pass
        finally:
            close_result(result)
        assert result.execution_state == "FAILED"
        assert len(scheduler.history) == attempts
        assert first.column(0).to_pylist() == expected
        del first
        assert_idle(connection)


def test_fte_and_pipelined_share_query_admission(tmp_path):
    from vane.execution.request_admission import RequestQueueTimeout

    with (
        vane.Runtime(resources(tmp_path, max_active_queries=1)) as application,
        application.connect() as connection,
        connection.cursor() as cursor,
    ):
        first = connection.query("select range from range(1000000)")
        batch = first.read_batch()
        del batch
        queued = replace(options(), admission_timeout=0.1)
        with pytest.raises(RequestQueueTimeout):
            cursor.query("select 2", options=queued)
        first.close()
        assert cursor.query("select 2", options=options()).collect().column(0).to_pylist() == [2]
        assert_idle(connection)


def test_pipelined_and_fte_execute_concurrently_on_one_worker_pool(tmp_path):
    import ray

    with (
        vane.Runtime(resources(tmp_path)) as application,
        application.connect() as connection,
        connection.cursor() as cursor,
    ):
        first = connection.query("select range from range(1000000)")
        batch = first.read_batch()
        del batch
        epochs = tuple(connection.query_runtime.pool.epochs)
        assert cursor.query("select range from range(50)", options=options()).collect().num_rows == 50
        assert tuple(connection.query_runtime.pool.epochs) == epochs
        assert any(
            ray.get(w.resources_snapshot.remote())["reservations"] for w in connection.query_runtime.pool.workers
        )
        first.close()
        assert_idle(connection)


@pytest.mark.parametrize("execution_timeout", [60, 2])
def test_fte_status_does_not_wait_for_native_pump(tmp_path, execution_timeout):
    from vane.execution.request_admission import RequestExecutionTimeout

    sql = "select range from range(30000) where hash(upper('" + "ß" * 16000 + "' || range::varchar)) = 0"
    limits = resources(tmp_path, worker_count=1, partitions=1, max_active_queries=1)
    with vane.Runtime(limits) as application, application.connect() as connection:
        connection.query("select 1", options=options()).collect()
        result = connection.query(sql, options=options(execution=execution_timeout, delivery=60))
        try:
            if execution_timeout == 60:
                assert result.collect().num_rows == 0
            else:
                with pytest.raises(RequestExecutionTimeout):
                    result.read_batch()
        finally:
            close_result(result)
        assert_idle(connection)


@pytest.mark.parametrize("publication", ["commit", "seal_stage", "publish_result"])
def test_commit_acknowledgement_loss_does_not_replay_selected_attempt(tmp_path, monkeypatch, publication):
    import errno

    from vane.execution.materialized_store import CommitCoordinator

    publish = getattr(CommitCoordinator, publication)
    lost = []

    def drop_reply(owner, *args):
        value = publish(owner, *args)
        if not lost:
            lost.append(value)
            raise OSError(errno.ETIMEDOUT, "commit acknowledgement lost")
        return value

    monkeypatch.setattr(CommitCoordinator, publication, drop_reply)
    with vane.Runtime(resources(tmp_path)) as application, application.connect() as connection:
        result = connection.query("select range from range(40)", options=options())
        scheduler = result.context._reader
        assert sorted(result.collect().column(0).to_pylist()) == list(range(40))
        assert lost
        assert len(scheduler.history) == 3
        assert all(t.attempt == 1 for t in scheduler.history)
        assert_idle(connection)


def test_expired_coordinator_lease_cancels_real_worker_and_reclaims_orphans(tmp_path):
    import ray

    from vane._native import execution_plan
    from vane.execution.compiler import FragmentCompileOptions
    from vane.execution.fte_plan import bind_task
    from vane.execution.fte_store import StorePool
    from vane.execution.materialized_store import CommitCoordinator
    from vane.execution.pipelined_worker import PipelinedWorker
    from vane.execution.resource_demand import MemoryDemand, ResourceDemand
    from vane.execution.submission import stage_ray_query

    limits = resources(tmp_path, worker_count=1, partitions=1, max_active_queries=1)
    config = replace(limits.exchange_stores[0], lease_seconds=1)
    limits = replace(limits, exchange_stores=(config,))
    worker = ray.remote(num_cpus=1, max_concurrency=4)(PipelinedWorker).remote(limits)
    epoch = ray.get(worker.describe.remote())["epoch"]
    store = StorePool(config)
    lease = store.reserve("orphan-query")
    coordinator = CommitCoordinator(
        store.store,
        "orphan-query",
        execution_plan.engine_identity(),
        (),
        max_bytes=config.query_bytes,
        namespace=lease.value["namespace"],
    )
    try:
        with vane.connect(backend="local") as connection:
            sql = "select range from range(1000000) where hash(upper('" + "ß" * 16000 + "' || range::varchar)) = 0"
            spec, _ = stage_ray_query(
                connection,
                sql,
                query_id="orphan-query",
                options=options(),
                resources=ResourceDemand(1, 8, MemoryDemand(1 << 28, 1 << 20, 1 << 20, 1 << 20), 8),
                source_directory=str(lease.directory / "sources"),
                source_budget=config.source_bytes,
                compile_options=FragmentCompileOptions(1),
            )
        fragment = spec.graph.fragments[0]
        binding = bind_task(spec, fragment, 0, {})
        coordinator.declare_stage((binding.task,))
        reservation = coordinator.begin(binding.task.task_id, epoch, object_bytes=config.object_bytes)
        lease.renew()
        ray.get(
            worker.prepare_materialized.remote(
                epoch, spec.to_dict(), fragment.fragment_id, 0, {}, reservation.to_dict(), lease.to_dict(), 64
            ),
            timeout=10,
        )
        assert ray.get(worker.resources_snapshot.remote())["reservations"]
        # Model lost coordinator ownership; no cancel/release RPC is sent.
        lease.guard.close()
        deadline = time.monotonic() + 10
        while ray.get(worker.resources_snapshot.remote())["reservations"]:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        # The worker drops its native reservation before collecting the
        # orphan. A competing collector can still hold the query lock after
        # deleting the data directory, until it retires the external quota.
        # A collection pass skips that owner; wait for the persisted outcome.
        while True:
            store.collect_expired()
            remaining = store.snapshot()
            if remaining == {"queries": 0, "reserved_bytes": 0}:
                break
            assert time.monotonic() < deadline, remaining
            time.sleep(0.02)
        assert not lease.directory.exists()
        assert not lease.lock_path.exists()
    finally:
        ray.kill(worker, no_restart=True)
        lease.guard.close()
        # A failed assertion must not leave a live query allocation behind.
        deadline = time.monotonic() + 10
        while store.snapshot()["queries"] and time.monotonic() < deadline:
            store.collect_expired()
            time.sleep(0.02)
