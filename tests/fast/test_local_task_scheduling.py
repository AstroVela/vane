# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import threading
import time

import pytest

from vane.execution.resources import ResourceVector
from vane.execution.udf_admission import LocalExecutionCapacity, LocalExecutionSlotPool
from vane.execution.udf_lifecycle import ExecutionCancellationScope, ExecutionCancelledError
from vane.execution.udf_local_resources import local_task_capacity


def _pool(capacity, name, *, cpu=1, heap=0, slots=4):
    return LocalExecutionSlotPool(
        max_slots=slots,
        execution_slot_prefix=name,
        execution_capacity=capacity,
        resources=ResourceVector(cpu=cpu, heap_bytes=heap),
    )


def _take(authority):
    authority.request(0)
    assert authority.state()["state"] == "ready"
    return authority.take(0)


def test_fractional_cpu_heap_and_resident_actors_share_one_node_budget():
    capacity = LocalExecutionCapacity(max_slots=None, resource_limit=ResourceVector(cpu=2, heap_bytes=100))
    release_actor = capacity.reserve_resident(ResourceVector(cpu=0.5, heap_bytes=20))
    pool = _pool(capacity, "decode", cpu=0.5, heap=30)
    authorities = [pool.create_authority() for _ in range(3)]
    leases = [_take(authority) for authority in authorities[:2]]
    try:
        authorities[2].request(0)
        assert authorities[2].state()["state"] == "requested"
        assert capacity.resource_snapshot()["usage"] == ResourceVector(cpu=1.5, heap_bytes=80).to_dict()
        leases[0].complete_execution()
        leases.append(authorities[2].take(0))
        assert capacity.resource_snapshot()["usage"] == ResourceVector(cpu=1.5, heap_bytes=80).to_dict()
    finally:
        for lease in leases:
            lease.release()
        pool.close()
        release_actor()
        release_actor()
    assert capacity.resource_snapshot()["usage"] == ResourceVector().to_dict()
    assert capacity.reserved_slots == 0


def test_closing_an_unsubmitted_grant_returns_its_process_resources():
    capacity = LocalExecutionCapacity(max_slots=None, resource_limit=ResourceVector(cpu=1, heap_bytes=100))
    pool = _pool(capacity, "decode", heap=100)
    first, second = pool.create_authority(), pool.create_authority()
    first.request(0)
    second.request(0)
    assert second.state()["state"] == "requested"
    first.close()
    lease = second.take(0)
    lease.release()
    pool.close()
    assert capacity.resource_snapshot()["usage"] == ResourceVector().to_dict()


def test_transport_wait_yields_cpu_retains_heap_and_reacquires_before_user_execution():
    capacity = LocalExecutionCapacity(max_slots=None, resource_limit=ResourceVector(cpu=1, heap_bytes=100))
    source = _pool(capacity, "source", heap=20)
    downstream = _pool(capacity, "downstream", heap=70)
    source_lease = _take(source.create_authority())
    scope = ExecutionCancellationScope("source", 1)
    waiting, transport_ready, resumed = threading.Event(), threading.Event(), threading.Event()
    errors = []

    def run():
        try:
            with source_lease.suspend_for_wait(scope):
                waiting.set()
                assert transport_ready.wait(5)
            resumed.set()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    downstream_lease = None
    try:
        assert waiting.wait(2)
        assert capacity.resource_snapshot()["usage"] == ResourceVector(heap_bytes=20).to_dict()
        downstream_lease = _take(downstream.create_authority())
        assert capacity.resource_snapshot()["usage"] == ResourceVector(cpu=1, heap_bytes=90).to_dict()
        transport_ready.set()
        assert not resumed.wait(0.1)
        downstream_lease.release()
        assert resumed.wait(2)
        assert capacity.resource_snapshot()["usage"] == ResourceVector(cpu=1, heap_bytes=20).to_dict()
    finally:
        transport_ready.set()
        if downstream_lease is not None:
            downstream_lease.release()
        thread.join(5)
        source_lease.release()
        source.close()
        downstream.close()
    assert not thread.is_alive() and errors == []
    assert capacity.resource_snapshot()["usage"] == ResourceVector().to_dict()


def test_failed_transport_releases_its_owner_without_reacquiring_cpu():
    capacity = LocalExecutionCapacity(max_slots=None, resource_limit=ResourceVector(cpu=1, heap_bytes=100))
    pool = _pool(capacity, "source", heap=20)
    first = _take(pool.create_authority())
    second = None
    try:
        with pytest.raises(ValueError, match="transport failed"):
            with first.suspend_for_wait(ExecutionCancellationScope("failed transport", 1)):
                second = _take(pool.create_authority())
                raise ValueError("transport failed")
        first.release()
        assert capacity.resource_snapshot()["usage"] == ResourceVector(cpu=1, heap_bytes=20).to_dict()
    finally:
        first.release()
        if second is not None:
            second.release()
        pool.close()
    assert capacity.resource_snapshot()["usage"] == ResourceVector().to_dict()


def test_task_pool_capacity_uses_declared_resources():
    limit = ResourceVector(cpu=8, heap_bytes=1000)
    assert local_task_capacity(ResourceVector(cpu=0.5), limit) == 16
    assert local_task_capacity(ResourceVector(cpu=2), limit) == 4
    assert local_task_capacity(ResourceVector(cpu=1, heap_bytes=400), limit) == 2
    with pytest.raises(ValueError, match="exceeds"):
        local_task_capacity(ResourceVector(cpu=9), limit)


def test_cancelled_cpu_resumption_does_not_take_another_tasks_resources():
    capacity = LocalExecutionCapacity(max_slots=None, resource_limit=ResourceVector(cpu=1, heap_bytes=100))
    pool = _pool(capacity, "source", heap=20)
    first = _take(pool.create_authority())
    scope = ExecutionCancellationScope("resume", 1)
    suspended, resume = threading.Event(), threading.Event()
    errors = []

    def run():
        try:
            with first.suspend_for_wait(scope):
                suspended.set()
                assert resume.wait(5)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    second = None
    try:
        assert suspended.wait(2)
        second = _take(pool.create_authority())
        resume.set()
        deadline = time.monotonic() + 2
        while not capacity.resource_snapshot()["resuming_tasks"] and time.monotonic() < deadline:
            time.sleep(0.005)
        assert capacity.resource_snapshot()["resuming_tasks"] == 1
        scope.cancel("test cancellation")
        thread.join(2)
        assert not thread.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], ExecutionCancelledError)
        first.release()
        assert capacity.resource_snapshot()["usage"] == ResourceVector(cpu=1, heap_bytes=20).to_dict()
        assert capacity.resource_snapshot()["resuming_tasks"] == 0
    finally:
        resume.set()
        scope.cancel()
        thread.join(5)
        first.release()
        if second is not None:
            second.release()
        pool.close()


def test_resident_startup_cannot_claim_cpu_yielded_by_a_live_task():
    capacity = LocalExecutionCapacity(max_slots=None, resource_limit=ResourceVector(cpu=1, heap_bytes=100))
    pool = _pool(capacity, "source", heap=20)
    lease = _take(pool.create_authority())
    started, acquired = threading.Event(), threading.Event()
    scope = ExecutionCancellationScope("actor startup", 1)
    releases, errors = [], []

    def reserve():
        started.set()
        try:
            releases.append(capacity.reserve_resident(ResourceVector(cpu=1, heap_bytes=80), scope))
            acquired.set()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=reserve, daemon=True)
    try:
        with lease.suspend_for_wait(ExecutionCancellationScope("source", 1)):
            thread.start()
            assert started.wait(2)
            assert not acquired.wait(0.1)
        assert not acquired.is_set()
        lease.release()
        assert acquired.wait(2)
        thread.join(2)
        assert errors == []
        assert capacity.resource_snapshot()["resident"] == ResourceVector(cpu=1, heap_bytes=80).to_dict()
        with pytest.raises(ValueError, match="exceed"):
            capacity.reserve_resident(ResourceVector(cpu=0.5))
    finally:
        scope.cancel()
        lease.release()
        if thread.ident is not None:
            thread.join(5)
        for release in releases:
            release()
        pool.close()
    assert capacity.resource_snapshot()["usage"] == ResourceVector().to_dict()


@pytest.mark.timeout(60)
@pytest.mark.parametrize("downstream", [None, "subprocess_task", "subprocess_actor"])
def test_one_row_group_runs_multiple_batches_with_one_native_thread(tmp_path, monkeypatch, downstream):
    import pyarrow as pa
    import pyarrow.parquet as pq

    import vane
    import vane.execution.udf_subprocess as subprocess_exec

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    subprocess_exec._shutdown_global_task_runtime()
    runtime = subprocess_exec._GlobalSubprocessTaskRuntime(resource_limit=ResourceVector(cpu=3, heap_bytes=1024**3))
    monkeypatch.setattr(subprocess_exec, "_GLOBAL_TASK_RUNTIME", runtime)
    source_path = tmp_path / "one-row-group.parquet"
    pq.write_table(pa.table({"x": list(range(8192))}), source_path, row_group_size=8192)
    assert pq.ParquetFile(source_path).num_row_groups == 1
    rendezvous = str(tmp_path / "workers")

    def decode(table):
        import os
        import time
        from pathlib import Path

        directory = Path(rendezvous)
        directory.mkdir(exist_ok=True)
        (directory / str(os.getpid())).touch()
        deadline = time.monotonic() + 10
        while len(list(directory.iterdir())) < 2:
            if time.monotonic() > deadline:
                raise RuntimeError("single-row-group UDF did not admit a second batch worker")
            time.sleep(0.01)
        return pa.table({"x": table.column("x"), "pid": [os.getpid()] * table.num_rows})

    def identity(table):
        return table

    class Actor:
        def __call__(self, table):
            return table

    connection = vane.connect(config={"threads": 1})
    schema = {"x": vane.sqltypes.BIGINT, "pid": vane.sqltypes.BIGINT}
    try:
        relation = connection.read_parquet(str(source_path)).map_batches(
            decode,
            schema=schema,
            execution_backend="subprocess_task",
            batch_size=256,
            memory_bytes=64 * 1024**2,
        )
        if downstream is not None:
            options = {"actor_number": 1} if downstream == "subprocess_actor" else {}
            relation = relation.map_batches(
                Actor if downstream == "subprocess_actor" else identity,
                schema=schema,
                execution_backend=downstream,
                batch_size=256,
                **options,
            )
        rows = relation.fetchall()
        assert sorted(row[0] for row in rows) == list(range(8192))
        assert len({row[1] for row in rows}) >= 2
    finally:
        connection.close()
        runtime.close(kill=True)
    assert runtime.execution_capacity.resource_snapshot()["usage"] == ResourceVector().to_dict()


def test_heap_progress_grant_bypasses_blocked_producer_in_the_same_pool():
    capacity = LocalExecutionCapacity(max_slots=None, resource_limit=ResourceVector(cpu=2, heap_bytes=100))
    progress = capacity.reserve_task_progress(
        {"upstream": ResourceVector(cpu=1, heap_bytes=40), "downstream": ResourceVector(cpu=1, heap_bytes=40)}
    )
    pool = _pool(capacity, "shared", heap=40)
    blocker = _pool(capacity, "cpu", heap=0)
    upstream = _take(pool.create_task_authority(progress.bind("upstream")))
    cpu = _take(blocker.create_authority())
    second = pool.create_task_authority(progress.bind("upstream"))
    downstream = pool.create_task_authority(progress.bind("downstream"))
    consumer = None
    try:
        second.request(0)
        downstream.request(0)
        assert second.state()["state"] == downstream.state()["state"] == "requested"
        cpu.release()
        assert second.state()["state"] == "requested"
        assert downstream.state()["state"] == "ready"
        consumer = downstream.take(0)
        assert capacity.resource_snapshot()["usage"]["heap_bytes"] == 80
        consumer.complete_execution()
        assert second.state()["state"] == "requested"
        upstream.complete_execution()
        assert second.state()["state"] == "ready"
        second.close()
        assert capacity.resource_snapshot()["task_progress"]["protected_heap_bytes"] == 80
    finally:
        if consumer is not None:
            consumer.release()
        cpu.release()
        upstream.release()
        pool.close()
        blocker.close()
        progress.shutdown()
        progress.shutdown()
    assert capacity.resource_snapshot()["usage"] == ResourceVector().to_dict()
    assert capacity.resource_snapshot()["task_progress"]["queries"] == 0


def test_prepared_tasks_protect_cpu_from_later_resident_startup():
    from vane.execution.udf_local_resources import LocalProcessCapacityError

    capacity = LocalExecutionCapacity(max_slots=None, resource_limit=ResourceVector(cpu=1, heap_bytes=100))
    progress = capacity.reserve_task_progress({"task": ResourceVector(cpu=1, heap_bytes=40)})
    with pytest.raises(LocalProcessCapacityError, match="node CPU/heap resource capacity"):
        capacity.reserve_resident(ResourceVector(cpu=1))
    assert capacity.resource_snapshot()["usage"] == ResourceVector().to_dict()
    progress.shutdown()
    release = capacity.reserve_resident(ResourceVector(cpu=1))
    release()


@pytest.mark.timeout(60)
@pytest.mark.parametrize("entrypoint", ["runtime", "native"])
@pytest.mark.parametrize(
    "scenario", ["resident_fits", "resident_exhausts_cpu", "heap_unreserved", "heap_reserved", "heap_impossible"]
)
def test_small_transport_pipeline_completes_or_rejects_before_execution(monkeypatch, entrypoint, scenario):
    import gc
    import uuid

    import pyarrow as pa

    import vane
    from vane.execution import ref_bundle, udf_subprocess
    from vane.execution.request_admission import RequestAdmissionLimits
    from vane.execution.udf_local_model import LocalModelRuntime

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    udf_subprocess._shutdown_global_task_runtime()
    resident = scenario.startswith("resident")
    node_cpus = 1 if scenario == "resident_exhausts_cpu" else 2
    tasks = udf_subprocess._GlobalSubprocessTaskRuntime(
        resource_limit=ResourceVector(cpu=node_cpus, heap_bytes=1024**3)
    )
    monkeypatch.setattr(udf_subprocess, "_GLOBAL_TASK_RUNTIME", tasks)
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)

    def expand(table):
        return pa.table({"blob": [b"x" * 65_536 for _ in range(table.num_rows)]})

    def consume(table):
        return pa.table({"size": [len(value.as_py()) for value in table.column(0)]})

    class Consume:
        def __call__(self, table):
            return consume(table)

    heap = (
        (400 if scenario == "heap_reserved" else 600) * 1024**2
        if scenario in {"heap_reserved", "heap_impossible"}
        else None
    )
    impossible = scenario in {"resident_exhausts_cpu", "heap_impossible"}
    try:
        with vane.connect(config={"threads": 1}) as connection:
            relation = (
                connection.sql("SELECT unnest([0,1,2,3])::BIGINT AS x")
                .map_batches(
                    expand,
                    schema={"blob": vane.sqltypes.BLOB},
                    execution_backend="subprocess_task",
                    batch_size=1,
                    min_task_batch_size=1,
                    task_input_max_bytes=8,
                    memory_bytes=heap,
                )
                .map_batches(
                    Consume if resident else consume,
                    schema={"size": vane.sqltypes.BIGINT},
                    execution_backend="subprocess_actor" if resident else "subprocess_task",
                    actor_number=1 if resident else None,
                    batch_size=1,
                    min_task_batch_size=1,
                    task_input_max_bytes=70_000,
                    memory_bytes=heap,
                )
            )
            if entrypoint == "native":
                if impossible:
                    with pytest.raises(vane.Error, match="actor residency and task progress"):
                        relation.fetchall()
                else:
                    for _ in range(2):
                        assert relation.fetchall() == [(65_536,)] * 4
            else:
                plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(
                    connection
                )
                with LocalModelRuntime(
                    session_id=plan.session_id(),
                    session_config=plan.session_config(),
                    request_limit=RequestAdmissionLimits(1, 1),
                    track_graph=True,
                ) as runtime:
                    if impossible:
                        with pytest.raises(ValueError, match="actor residency and task progress"):
                            runtime.request().execute(plan, {}, conn=connection, execution_timeout=15)
                    else:
                        result = runtime.request().execute(plan, {}, conn=connection, execution_timeout=15)
                        assert [
                            value for table in result.partition_payloads for value in table.column(0).to_pylist()
                        ] == [65_536] * 4
                        del result
            if impossible:
                assert tasks.stats()["total_workers"] == 0
    finally:
        tasks.close(kill=True)
        gc.collect()
        snapshot = tasks.execution_capacity.resource_snapshot()
        assert snapshot["usage"] == ResourceVector().to_dict()
        assert snapshot["task_progress"]["queries"] == 0
        assert manager.snapshot()["usage_bytes"] == 0


def test_model_prewarm_retries_after_another_runtime_returns_node_capacity(monkeypatch):
    from vane import pickle as vane_pickle
    from vane.execution import udf_subprocess
    from vane.execution.udf_local_model import LocalModelRuntime
    from vane.execution.udf_local_resources import LocalProcessCapacityError

    udf_subprocess._shutdown_global_task_runtime()
    tasks = udf_subprocess._GlobalSubprocessTaskRuntime(resource_limit=ResourceVector(cpu=1, heap_bytes=1024**3))
    monkeypatch.setattr(udf_subprocess, "_GLOBAL_TASK_RUNTIME", tasks)

    class Identity:
        def __call__(self, table):
            return table

    payload = {
        "function_pickle": vane_pickle.dumps(Identity),
        "call_mode": "map_batches",
        "execution_backend": "subprocess_actor",
        "actor_number": 1,
        "cpus": 1,
    }
    first = LocalModelRuntime(session_id="first", session_config={})
    second = LocalModelRuntime(session_id="second", session_config={})
    try:
        first_model = first.register("identity", version="1", payload=payload)
        second_model = second.register("identity", version="1", payload=payload)
        first_model.prewarm()
        for _ in range(2):
            with pytest.raises(LocalProcessCapacityError, match="node CPU/heap resource capacity"):
                second_model.prewarm()
        first.close()
        assert tasks.execution_capacity.resource_snapshot()["usage"] == ResourceVector().to_dict()
        second_model.prewarm()
        with second_model.acquire() as borrow:
            assert len(borrow.pool.worker_pids()) == 1
    finally:
        first.close(kill=True)
        second.close(kill=True)
        tasks.close(kill=True)
    assert tasks.execution_capacity.resource_snapshot()["usage"] == ResourceVector().to_dict()


@pytest.mark.timeout(40)
@pytest.mark.parametrize(
    "node_cpus,close_owner,cleanup_failures",
    [
        (1, "executor", 1),
        (1, "executor", 2),
        (1, "runtime", 1),
        (1, "runtime", 2),
        (1, "shared_pool", 2),
        (2, "executor", 0),
    ],
)
def test_overflow_retirement_keeps_cleanup_owner_until_process_exits(
    monkeypatch, node_cpus, close_owner, cleanup_failures
):
    import pyarrow as pa

    from vane import pickle as vane_pickle
    from vane.execution import ref_bundle, udf_subprocess
    from vane.execution.udf_local_model import LocalModelRuntime
    from vane.execution.udf_local_resources import LocalProcessCapacityError

    def wait_until(predicate):
        deadline = time.monotonic() + 10
        while not predicate():
            assert time.monotonic() < deadline, "worker cleanup did not make progress"
            time.sleep(0.01)

    def submit(executor):
        table = pa.table({"x": [1]})
        assert executor.request_task_admission(table.nbytes)
        assert executor.task_admission_state()["available"]
        executor.submit(table)

    def expand(table):
        # Fit the 10,000-byte hard limit but wait for the held 5,000 bytes.
        # An oversized single block is rejected instead of entering a wait.
        return pa.table({"blob": [b"x" * 8192]})

    def identity(table):
        return table

    def payload(function, **options):
        return {
            "function_pickle": vane_pickle.dumps(function),
            "call_mode": "map_batches",
            "execution_backend": "subprocess_task",
            "cpus": 1,
            "memory_bytes": 20,
            **options,
        }

    udf_subprocess._shutdown_global_task_runtime()
    tasks = udf_subprocess._GlobalSubprocessTaskRuntime(resource_limit=ResourceVector(cpu=node_cpus, heap_bytes=100))
    monkeypatch.setattr(udf_subprocess, "_GLOBAL_TASK_RUNTIME", tasks)
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 10_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    # A competing task envelope prevents the consumer's first-output soft
    # overage allowance from bypassing the pressure this test must exercise.
    competing_task = manager.reserve_task_bytes(1, 1)
    manager.acquire_allocation(5000, name="held-output")
    source = udf_subprocess.UDFExecutor(
        payload(expand, produce_ref_bundle_output=True, streaming_output_mode="local_shm_ref_bundle")
    )
    consumer_payload = payload(identity)
    consumer = udf_subprocess.UDFExecutor(consumer_payload)
    other_consumer = udf_subprocess.UDFExecutor(consumer_payload) if close_owner == "shared_pool" else None
    replacement_consumer = None
    prewarm_runtime = None
    prewarm_model = None
    original_close = udf_subprocess._SingleSubprocessExecutor.close
    retained = []
    close_kills = []
    failures_remaining = cleanup_failures
    try:
        submit(source)
        wait_until(lambda: manager.snapshot()["waiting_output_grants"] == 1)
        assert tasks.execution_capacity.resource_snapshot()["usage"]["cpu"] == 0
        pool = consumer._task_pool

        def fail_close(worker, *args, **kwargs):
            nonlocal failures_remaining
            if any(wrapper.worker is worker for wrapper in pool._retiring_workers):
                close_kills.append(kwargs.get("kill", False))
                if failures_remaining:
                    failures_remaining -= 1
                    if worker not in retained:
                        retained.append(worker)
                    raise RuntimeError("injected overflow close failure")
            return original_close(worker, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(udf_subprocess._SingleSubprocessExecutor, "close", fail_close)
            submit(consumer)
            results = []

            def take_result():
                result = consumer.take_ready_result()
                if result is not None:
                    results.append(result)
                return bool(results)

            wait_until(take_result)
            if cleanup_failures:
                assert isinstance(results[0], RuntimeError)
                assert "injected overflow close failure" in str(results[0])
                assert len(retained) == 1
                proc = retained[0]._proc
                assert pool.total == 1
                assert pool.active == 0
                assert pool.idle == []
                assert tasks.stats()["total_workers"] == 2
                assert tasks.stats()["retiring_workers"] == 1
            else:
                assert results[0].to_pydict() == {"x": [1]}
                assert retained == []
                proc = pool.idle[0].worker._proc
                assert tasks.stats()["retiring_workers"] == 0
            assert proc.poll() is None

            if other_consumer is not None:
                assert other_consumer._task_pool is pool
                consumer.close(kill=True)
                assert pool.ref_count == 1
                assert tasks.stats()["retiring_workers"] == 1
                assert proc.poll() is None
            owner = tasks if close_owner == "runtime" else other_consumer or consumer
            if failures_remaining:
                with pytest.raises(RuntimeError, match="injected overflow close failure"):
                    owner.close(kill=True)
                if owner is tasks:
                    assert source._wait_for_pending_futures(10)
                assert proc.poll() is None
                assert pool.total == 1
                assert tasks.stats()["retiring_workers"] == 1
                if owner is tasks:
                    with pytest.raises(LocalProcessCapacityError, match="workers awaiting cleanup"):
                        udf_subprocess._global_task_runtime()
                    prewarm_runtime = LocalModelRuntime(session_id="retry-cleanup", session_config={})

                    class Identity:
                        def __call__(self, table):
                            return table

                    prewarm_model = prewarm_runtime.register(
                        "identity",
                        version="1",
                        payload=payload(Identity, execution_backend="subprocess_actor", actor_number=1),
                    )
                    with pytest.raises(LocalProcessCapacityError, match="workers awaiting cleanup"):
                        prewarm_model.prewarm()
                else:
                    assert owner._task_pool is pool
                    assert owner.cleanup_pending()
                    replacement_consumer = udf_subprocess.UDFExecutor(consumer_payload)
                    assert replacement_consumer._task_pool is not pool
                    assert tasks.pools[pool.key] is replacement_consumer._task_pool
            owner.close(kill=cleanup_failures < 2)
            if owner is tasks:
                assert source._wait_for_pending_futures(10)
            assert proc.poll() is not None
            assert pool.total == 0
            assert tasks.stats()["retiring_workers"] == 0
            if replacement_consumer is not None:
                assert tasks.pools[pool.key] is replacement_consumer._task_pool
                assert replacement_consumer._task_pool.ref_count == 1
            if cleanup_failures:
                assert close_kills == [False, *([True] * cleanup_failures)]
            if prewarm_model is not None:
                prewarm_model.prewarm()
                with prewarm_model.acquire() as borrow:
                    assert len(borrow.pool.worker_pids()) == 1
    finally:
        for worker in retained:
            original_close(worker, kill=True)
        competing_task.release()
        manager.release_allocation(5000, name="held-output")
        source.close(kill=True)
        consumer.close(kill=True)
        if other_consumer is not None:
            other_consumer.close(kill=True)
        if replacement_consumer is not None:
            replacement_consumer.close(kill=True)
        tasks.close(kill=True)
        if prewarm_runtime is not None:
            prewarm_runtime.close(kill=True)
            udf_subprocess._shutdown_global_task_runtime()
    wait_until(lambda: tasks.execution_capacity.resource_snapshot()["usage"] == ResourceVector().to_dict())
    assert tasks.stats()["total_workers"] == 0
    assert tasks.stats()["retiring_workers"] == 0
    assert manager.snapshot()["usage_bytes"] == 0


@pytest.mark.timeout(40)
@pytest.mark.parametrize("cleanup_failures", [0, 1, 2])
def test_shared_task_pool_retries_failed_retirement_without_stranding_grants(monkeypatch, cleanup_failures):
    import pyarrow as pa

    from vane import pickle as vane_pickle
    from vane.execution import udf_subprocess

    def wait_until(predicate):
        deadline = time.monotonic() + 5
        while not predicate():
            assert time.monotonic() < deadline, "shared-pool peer did not finish"
            time.sleep(0.01)

    def submit(executor):
        table = pa.table({"x": [1]})
        assert executor.request_task_admission(table.nbytes)
        assert executor.task_admission_state()["available"]
        executor.submit(table)

    def take(executor):
        results = []

        def poll():
            result = executor.take_ready_result()
            if result is not None:
                results.append(result)
            return bool(results)

        wait_until(poll)
        wait_until(lambda: tasks.execution_capacity.resource_snapshot()["usage"] == ResourceVector().to_dict())
        return results[0]

    def identity(table):
        return table

    def increment(table):
        return pa.table({"x": [value + 1 for value in table.column("x").to_pylist()]})

    def payload(function):
        return {
            "function_pickle": vane_pickle.dumps(function),
            "call_mode": "map_batches",
            "execution_backend": "subprocess_task",
            "cpus": 1,
            "memory_bytes": 20,
        }

    udf_subprocess._shutdown_global_task_runtime()
    tasks = udf_subprocess._GlobalSubprocessTaskRuntime(resource_limit=ResourceVector(cpu=1, heap_bytes=100))
    monkeypatch.setattr(udf_subprocess, "_GLOBAL_TASK_RUNTIME", tasks)
    shared_payload = payload(identity)
    first = udf_subprocess.UDFExecutor(shared_payload)
    peer = udf_subprocess.UDFExecutor(shared_payload)
    other = udf_subprocess.UDFExecutor(payload(increment))
    pool = first._task_pool
    assert peer._task_pool is pool
    assert pool.pool_size == 1
    original_close = udf_subprocess._SingleSubprocessExecutor.close
    failures_remaining = cleanup_failures
    failed_workers = []
    close_kills = []

    def fail_close(worker, *args, **kwargs):
        nonlocal failures_remaining
        if any(wrapper.worker is worker for wrapper in pool._retiring_workers):
            close_kills.append(kwargs.get("kill", False))
            if failures_remaining:
                failures_remaining -= 1
                if worker not in failed_workers:
                    failed_workers.append(worker)
                raise RuntimeError("injected retirement retry failure")
        return original_close(worker, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(udf_subprocess._SingleSubprocessExecutor, "close", fail_close)
            with monkeypatch.context() as retirement_patch:
                retirement_patch.setattr(udf_subprocess, "_worker_is_reusable", lambda worker: False)
                submit(first)
                first_result = take(first)
            if cleanup_failures:
                assert isinstance(first_result, RuntimeError)
                assert "injected retirement retry failure" in str(first_result)
                assert len(failed_workers) == 1
                retired_proc = failed_workers[0]._proc
                assert retired_proc.poll() is None
                assert pool.total == 1 and pool.active == 0 and pool.idle == []
            else:
                assert first_result.to_pydict() == {"x": [1]}
            first.close(kill=True)
            assert pool.ref_count == 1
            submit(peer)
            peer_result = take(peer)
            assert close_kills == ([False, True] if cleanup_failures else [False])
            if cleanup_failures == 2:
                assert isinstance(peer_result, RuntimeError)
                assert "injected retirement retry failure" in str(peer_result)
                assert retired_proc.poll() is None
                assert pool.total == 1 and pool.active == 0
                assert tasks.stats()["retiring_workers"] == 1
            else:
                assert peer_result.to_pydict() == {"x": [1]}
                assert tasks.stats()["retiring_workers"] == 0
                if cleanup_failures:
                    assert retired_proc.poll() is not None
                    assert pool.idle[0].worker._proc.pid != retired_proc.pid
            submit(other)
            assert take(other).to_pydict() == {"x": [2]}
            if cleanup_failures == 2:
                assert retired_proc.poll() is None
                assert pool.total == 1
                assert tasks.stats()["retiring_workers"] == 1
        if cleanup_failures == 2:
            submit(peer)
            assert take(peer).to_pydict() == {"x": [1]}
            assert retired_proc.poll() is not None
            assert tasks.stats()["retiring_workers"] == 0
    finally:
        first.close(kill=True)
        peer.close(kill=True)
        other.close(kill=True)
        tasks.close(kill=True)
        for worker in failed_workers:
            original_close(worker, kill=True)
    assert tasks.stats()["total_workers"] == 0
    assert tasks.stats()["retiring_workers"] == 0
    assert tasks.execution_capacity.resource_snapshot()["usage"] == ResourceVector().to_dict()


def test_global_shutdown_keeps_a_new_runtime_created_as_old_cleanup_returns(monkeypatch):
    from vane.execution import udf_subprocess

    udf_subprocess._shutdown_global_task_runtime()
    old = udf_subprocess._GlobalSubprocessTaskRuntime(resource_limit=ResourceVector(cpu=1, heap_bytes=100))
    monkeypatch.setattr(udf_subprocess, "_GLOBAL_TASK_RUNTIME", old)
    close_finished = threading.Event()
    allow_return = threading.Event()
    errors = []
    original_close = old.close

    def close_before_return(*, kill):
        original_close(kill=kill)
        close_finished.set()
        assert allow_return.wait(5)

    monkeypatch.setattr(old, "close", close_before_return)

    def shutdown():
        try:
            udf_subprocess._shutdown_global_task_runtime()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=shutdown, daemon=True)
    thread.start()
    try:
        assert close_finished.wait(2)
        fresh = udf_subprocess._global_task_runtime()
        assert fresh is not old
        allow_return.set()
        thread.join(2)
        assert not thread.is_alive()
        assert errors == []
        assert udf_subprocess._GLOBAL_TASK_RUNTIME is fresh
    finally:
        allow_return.set()
        thread.join(5)
        udf_subprocess._shutdown_global_task_runtime()
