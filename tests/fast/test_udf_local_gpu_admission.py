# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only GPU admission contracts with real subprocess UDF execution."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pytest

from vane import pickle as vane_pickle
from vane.execution import ref_bundle, udf_subprocess
from vane.execution.local_resource_graph import LocalResourceUnitContext
from vane.execution.request_deadline import RequestExecutionDeadline
from vane.execution.resources import ResourceVector
from vane.execution.udf_data_admission import DataAdmissionLimits, DataAdmissionWaitLimits
from vane.execution.udf_data_lease import RuntimeDataLedger
from vane.execution.udf_lifecycle import ExecutionCancellationScope
from vane.execution.udf_local_gpu import LocalGpuModelAdapter
from vane.execution.udf_model_pool import ModelPoolRegistry
from vane.execution.udf_runtime_admission import RuntimeTaskAdmission, TaskAdmissionLimits

DEVICES = ("GPU-aaaaaaaa-0000-0000-0000-000000000001", "GPU-bbbbbbbb-0000-0000-0000-000000000002")


def _wait(check):
    deadline = time.monotonic() + 15
    while True:
        value = check()
        if value:
            return value
        assert time.monotonic() < deadline, "GPU execution did not reach its checkpoint"
        time.sleep(0.005)


def _result(executor):
    chunks = []
    try:
        while True:
            result = _wait(executor.take_ready_result)
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], bool):
                block, finished = result
                if block is not None:
                    chunks.append(block)
                if finished:
                    assert len(chunks) == 1, "expected one output block before task completion"
                    return chunks.pop()
            else:
                return result
    finally:
        for chunk in chunks:
            udf_subprocess._release_local_ref_bundle_result(chunk)


def _admit(executor):
    assert executor.request_task_admission(0)
    _wait(lambda: executor.task_admission_state()["available"])


def _devices(pool):
    return pool.gpu_execution_snapshot()["devices"]


def _demand(pool):
    return sum(d["execution_resources"]["gpu"] for d in _devices(pool))


def _states(pool):
    return [e["state"] for d in _devices(pool) for e in d["executions"]]


class _Harness:
    def __init__(self, *, running=2):
        self.registry = ModelPoolRegistry(resident_limit=ResourceVector(cpu=2, gpu=2))
        self.adapter = LocalGpuModelAdapter(self.registry, devices=DEVICES)
        self.runtime = RuntimeTaskAdmission(TaskAdmissionLimits(running, 16))
        self.borrows, self.queries, self.executors, self.pools, self.releases = [], [], [], [], []

    def model(self, callback, devices=DEVICES[:1], *, refs=False):
        class Model:
            def __call__(self, table):
                return callback(table)

        payload = {
            "function_pickle": vane_pickle.dumps(Model),
            "call_mode": "map_batches",
            "execution_backend": "subprocess_actor",
            "actor_number": len(devices),
            "cpus": 0.5,
            "gpus": 1,
        }
        if refs:
            payload.update(produce_ref_bundle_output=True, streaming_output_mode="local_shm_ref_bundle")
        identity = self.adapter.register(
            f"model-{len(self.pools)}",
            version="v1",
            session_id="gpu-execution",
            session_config={},
            payload=payload,
            devices=devices,
        )
        borrow = self.registry.acquire(identity)
        self.borrows.append(borrow)
        self.pools.append(borrow.pool)
        return borrow.pool, payload

    def executor(self, model, *, limited=True, **options):
        pool, payload = model
        if limited:
            query = self.runtime.open_query()
            self.queries.append(query)
            options["local_task_admission"] = query
        executor = udf_subprocess.UDFExecutor(payload, {"local_actor_pool": pool, "session_config": {}, **options})
        self.executors.append(executor)
        return executor

    def close(self):
        for release in self.releases:
            release.touch()
        for executor in self.executors:
            executor.close(kill=True)
        for query in self.queries:
            query.shutdown()
        self.runtime.close(timeout=15)
        for borrow in self.borrows:
            borrow.shutdown()
        self.registry.close(timeout=15, kill=True)
        assert self.registry.resource_snapshot()["reserved_resources"]["gpu"] == 0
        for pool in self.pools:
            assert _demand(pool) == 0
            assert not pool.cleanup_pending()


@pytest.fixture
def harness():
    instances = []

    def make(**kwargs):
        h = _Harness(**kwargs)
        instances.append(h)
        return h

    yield make
    for h in instances:
        h.close()


def _gated(h, tmp_path):
    root = str(tmp_path)
    h.releases.extend(tmp_path / f"release-{i}" for i in (1, 2))

    def call(table):
        value = table.column(0)[0].as_py()
        if value in (1, 2):
            Path(root, f"entered-{value}").touch()
            deadline = time.monotonic() + 20
            while not Path(root, f"release-{value}").exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("GPU fixture was not released")
                time.sleep(0.005)
        return pa.table({"device": [os.environ["CUDA_VISIBLE_DEVICES"]], "pid": [os.getpid()], "x": [value]})

    return call


def test_replicas_bind_admission_to_devices_and_run_in_parallel(harness, tmp_path):
    h = harness()
    model = h.model(_gated(h, tmp_path), DEVICES)
    pool, _ = model
    first, second = h.executor(model), h.executor(model)
    # Force a different idle-worker ordering from the admission slot ordering.
    with pool._cond:
        pool._idle_workers.reverse()
    for executor in (first, second):
        _admit(executor)
    assert _demand(pool) == 2
    assert h.registry.resource_snapshot()["reserved_resources"]["gpu"] == 2
    first.submit(pa.table({"x": [1]}))
    second.submit(pa.table({"x": [2]}))
    _wait(lambda: (tmp_path / "entered-1").exists() and (tmp_path / "entered-2").exists())
    assert _states(pool) == ["running", "running"]
    for row, worker in zip(_devices(pool), pool.device_snapshot()):
        (execution,) = row["executions"]
        assert execution["device"] == worker["device"]
        assert execution["generation"] == worker["generation"] == 0
        assert execution["pid"] == worker["pid"]
    (tmp_path / "release-1").touch()
    _wait(lambda: _demand(pool) == 1)
    assert _devices(pool)[0]["retained_slots"] == 0
    assert _result(first).to_pydict()["device"] == [DEVICES[0]]
    (tmp_path / "release-2").touch()
    assert _result(second).to_pydict()["device"] == [DEVICES[1]]
    assert _demand(pool) == 0
    assert h.registry.resource_snapshot()["reserved_resources"]["gpu"] == 2


@pytest.mark.parametrize("limited_first", [False, True])
def test_shared_device_keeps_older_query_ahead_of_newer_work(harness, tmp_path, limited_first):
    h = harness()
    model = h.model(_gated(h, tmp_path))
    pool, _ = model
    first = h.executor(model, limited=limited_first)
    older = h.executor(model, limited=not limited_first)
    newer = h.executor(model, limited=limited_first)
    _admit(first)
    first.submit(pa.table({"x": [1]}))
    _wait(lambda: (tmp_path / "entered-1").exists())
    assert older.request_task_admission(0)
    assert newer.request_task_admission(0)
    assert pool.gpu_execution_snapshot()["admission"]["queued_tasks"] == 1
    assert older.task_admission_state()["available"]
    assert not newer.task_admission_state()["available"]
    older.submit(pa.table({"x": [3]}))
    (tmp_path / "release-1").touch()
    assert _result(first).to_pydict()["x"] == [1]
    assert _result(older).to_pydict()["x"] == [3]
    assert newer.task_admission_state()["available"]
    newer.submit(pa.table({"x": [4]}))
    assert _result(newer).to_pydict()["x"] == [4]


def test_busy_device_without_prefetch_does_not_park_another_devices_task_allowance(harness, tmp_path, monkeypatch):
    monkeypatch.setenv("VANE_UDF_ACTOR_PREFETCH_DEPTH", "1")
    h = harness(running=2)
    first_model = h.model(_gated(h, tmp_path))
    second_model = h.model(lambda table: table, DEVICES[1:])
    first, pending = h.executor(first_model), h.executor(first_model)
    other = h.executor(second_model)
    _admit(first)
    first.submit(pa.table({"x": [1]}))
    _wait(lambda: (tmp_path / "entered-1").exists())
    assert pending.request_task_admission(0)
    assert not pending.task_admission_state()["available"]
    assert h.runtime.snapshot()["ready_tasks"] == 0
    _admit(other)
    other.submit(pa.table({"x": [7]}))
    assert _result(other).to_pydict() == {"x": [7]}
    (tmp_path / "release-1").touch()
    assert _result(first).to_pydict()["x"] == [1]


@pytest.mark.parametrize("phase", ["queued", "ready", "running"])
@pytest.mark.parametrize("timeout", [False, True])
def test_cancel_and_deadline_retire_device_execution_without_releasing_model(harness, tmp_path, phase, timeout):
    h = harness(running=1)
    model = h.model(_gated(h, tmp_path))
    pool, _ = model
    cancellation = ExecutionCancellationScope("request", 1)
    target = h.executor(model, local_request_cancellation=cancellation)
    blocker = h.executor(model) if phase == "queued" else None
    if blocker is not None:
        _admit(blocker)
    assert target.request_task_admission(0)
    old_pid = pool.worker_pids()[0]
    if phase == "running":
        target.submit(pa.table({"x": [1]}))
        _wait(lambda: (tmp_path / "entered-1").exists())
    reason = "execution_timeout" if timeout else "cancelled"
    deadline = RequestExecutionDeadline(time.monotonic(), 0.05, lambda: cancellation.cancel(reason))
    try:
        if timeout:
            deadline.start()
        else:
            cancellation.cancel(reason)
        _wait(lambda: cancellation.is_set() and target._cleanup_finished)
        assert cancellation.cancel_reason == reason
        if blocker is not None:
            blocker.close()
        assert _demand(pool) == 0
        assert h.registry.resource_snapshot()["reserved_resources"]["gpu"] == 1
        assert (pool.worker_pids()[0] != old_pid) == (phase == "running")
        next_query = h.executor(model)
        _admit(next_query)
        next_query.submit(pa.table({"x": [7]}))
        assert _result(next_query).to_pydict()["device"] == [DEVICES[0]]
    finally:
        deadline.close()


@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize("action", ["release", "cancel", "timeout"])
def test_byte_wait_has_no_device_execution_reservation(harness, monkeypatch, limited, action):
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 4096)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    h = harness()
    model = h.model(lambda table: table, refs=True)
    pool, _ = model
    ledger = RuntimeDataLedger(
        DataAdmissionLimits(4096, 512, 2048, wait=DataAdmissionWaitLimits(4, 0.1 if action == "timeout" else 10))
    )
    unit = LocalResourceUnitContext("query", "unit", "node", "subprocess_actor")
    query = ledger.open_query(resource_units=[unit])
    executor = h.executor(model, limited=limited, local_data_scope=query, local_resource_unit=unit)
    occupied = manager.reserve_task_bytes(3500, 1)
    try:
        assert executor.request_task_admission(0)
        expected_states = {"waiting_bytes", "failed"} if action == "timeout" else {"waiting_bytes"}
        assert executor.task_admission_state()["state"] in expected_states
        assert _demand(pool) == 0
        assert h.runtime.snapshot()["ready_tasks"] == 0
        if action == "release":
            occupied.release()
            _wait(lambda: executor.task_admission_state()["available"])
            assert _demand(pool) == 1
            executor.submit(pa.table({"x": [7]}))
            output = _result(executor)
            udf_subprocess._release_local_ref_bundle_result(output)
        elif action == "cancel":
            executor.close(kill=True)
        else:
            _wait(lambda: executor.task_admission_state()["state"] == "failed")
        assert _demand(pool) == 0
    finally:
        occupied.release()
        executor.close(kill=True)
        query.shutdown()
        ledger.close()
    assert manager.snapshot()["usage_bytes"] == 0


def test_output_wait_yields_runtime_allowance_but_keeps_its_device_until_consumer_progress(
    harness, monkeypatch, tmp_path
):
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 4096)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    h = harness(running=1)
    entered, release = str(tmp_path / "output-ready"), str(tmp_path / "allow-output")
    h.releases.append(Path(release))

    def produce(table):
        Path(entered).touch()
        while not Path(release).exists():
            time.sleep(0.005)
        return pa.table({"blob": [b"p" * 1000]})

    producer_model = h.model(produce, refs=True)
    consumer_model = h.model(lambda table: pa.table({"length": [len(table.column(0)[0].as_py())]}), DEVICES[1:])
    producer, consumer = h.executor(producer_model), h.executor(consumer_model)
    held = ref_bundle.make_local_shm_ref_bundle_result(pa.table({"blob": [b"c" * 3000]}))
    output = None
    occupied = None
    try:
        _admit(producer)
        producer.submit(pa.table({"x": [1]}))
        _wait(lambda: Path(entered).exists())
        # Exhaust the bounded consumer escape after input decoding. Otherwise
        # the first output is allowed through the soft limit without waiting.
        occupied = manager.request_output_grant(1000, priority="consumer")
        Path(release).touch()
        _wait(
            lambda: (
                _states(producer_model[0]) == ["shared_memory_output"] and h.runtime.snapshot()["waiting_tasks"] == 1
            )
        )
        assert _demand(producer_model[0]) == 1
        assert h.runtime.snapshot()["waiting_tasks"] == 1
        assert h.runtime.snapshot()["running_tasks"] == 0
        _admit(consumer)
        consumer.submit_ref_bundle_with_id(7, held[1], None, held[2], ["blob"])
        result = _result(consumer)
        assert result[2].to_pydict() == {"length": [3000]}
        output = _result(producer)
        assert output[1][0].to_table().column(0).to_pylist() == [b"p" * 1000]
        assert _demand(producer_model[0]) == _demand(consumer_model[0]) == 0
        assert h.runtime.snapshot()["waiting_tasks"] == 0
    finally:
        if occupied is not None:
            manager.release_output_grant(occupied)
        udf_subprocess._release_local_ref_bundle_result(output)
        udf_subprocess._release_local_ref_bundle_result(held)
    assert manager.snapshot()["usage_bytes"] == 0


def test_failed_worker_cleanup_retains_execution_charge_until_pool_retry(harness, monkeypatch):
    def fail(table):
        raise ValueError("GPU model failed")

    h = harness()
    model = h.model(fail)
    pool, _ = model
    executor = h.executor(model)
    worker = pool._workers[0]
    close = worker.close

    def fail_close(*args, **kwargs):
        raise OSError("GPU worker cleanup failed")

    with monkeypatch.context() as fault:
        fault.setattr(worker, "close", fail_close)
        _admit(executor)
        executor.submit(pa.table({"x": [1]}))
        assert isinstance(_result(executor), BaseException)
        assert _states(pool) == ["cleanup_pending"]
        assert _demand(pool) == 1
        assert pool.cleanup_pending()
        assert h.registry.resource_snapshot()["reserved_resources"]["gpu"] == 1
        with pytest.raises(RuntimeError, match="cleanup"):
            pool.shutdown(kill=True)
        assert _demand(pool) == 1
    pool.shutdown(kill=True)
    assert _demand(pool) == 0
    assert worker._cleanup_finished
    assert worker.close == close


@pytest.mark.parametrize("kill", [False, True])
def test_concurrent_shutdown_keeps_failed_worker_owned_during_execution_completion(harness, monkeypatch, kill):
    h = harness()
    pool, _ = h.model(lambda table: table)
    worker = pool._workers[0]
    process = worker._proc
    run_finished, release_run = threading.Event(), threading.Event()
    handoff_entered, release_handoff = threading.Event(), threading.Event()
    run = pool._run
    handoff = pool._replace_attempted_cleanup_workers

    def finish_run(*args, **kwargs):
        result = run(*args, **kwargs)
        run_finished.set()
        assert release_run.wait(15), "GPU execution completion was not released"
        return result

    def pause_handoff(*args, **kwargs):
        handoff_entered.set()
        assert release_handoff.wait(15), "GPU shutdown handoff was not released"
        return handoff(*args, **kwargs)

    def fail_close(*args, **kwargs):
        raise OSError("GPU worker cleanup failed")

    authority = pool.create_admission_authority()
    authority.request(0)
    lease = authority.take(0)
    scope = ExecutionCancellationScope("shutdown-handoff", 1)
    with monkeypatch.context() as fault:
        fault.setattr(pool, "_run", finish_run)
        fault.setattr(pool, "_replace_attempted_cleanup_workers", pause_handoff)
        fault.setattr(worker, "close", fail_close)
        with ThreadPoolExecutor(max_workers=1) as closer:
            try:
                future = pool.submit(lambda w: w._submit_table(pa.table({"x": [7]})), scope, admission=lease)
                assert run_finished.wait(10)
                shutdown = closer.submit(pool.shutdown, kill=kill)
                assert handoff_entered.wait(10)
                # Complete the invocation while shutdown is about to transfer
                # failed workers from live ownership to cleanup ownership.
                release_run.set()
                assert future.result(timeout=10).to_pydict() == {"x": [7]}
                lease.release()
                assert process.poll() is None
                assert worker._cleanup_finished is False
                assert _demand(pool) == 1
                assert _states(pool) == ["cleanup_pending"]
                (execution,) = _devices(pool)[0]["executions"]
                assert execution["pid"] == process.pid
                assert pool.cleanup_pending()
                release_handoff.set()
                with pytest.raises(RuntimeError, match="cleanup"):
                    shutdown.result(timeout=10)
                assert _devices(pool)[0]["executions"] == [execution]
                assert _demand(pool) == 1
            finally:
                release_run.set()
                release_handoff.set()
                lease.release()
                authority.close()
    pool.shutdown(kill=True)
    assert worker._cleanup_finished is True
    assert process.poll() is not None
    assert _demand(pool) == 0
    assert _states(pool) == []
    assert not pool.cleanup_pending()


def test_device_submission_rejects_missing_foreign_and_reused_leases(harness):
    h = harness()
    first, _ = h.model(lambda table: table)
    second, _ = h.model(lambda table: table, DEVICES[1:])
    authority = first.create_admission_authority()
    authority.request(0)
    assert first.gpu_execution_snapshot()["admission"]["ready_tasks"] == 1
    lease = authority.take(0)
    assert first.gpu_execution_snapshot()["admission"]["ready_tasks"] == 0
    scope = ExecutionCancellationScope("direct", 1)
    try:
        with pytest.raises(RuntimeError, match="live admission lease"):
            first.submit(lambda worker: None, scope)
        with pytest.raises(RuntimeError, match="live admission lease"):
            second.submit(lambda worker: None, scope, admission=lease)
        future = first.submit(lambda worker: worker._submit_table(pa.table({"x": [7]})), scope, admission=lease)
        assert future.result(timeout=15).to_pydict() == {"x": [7]}
        with pytest.raises(RuntimeError, match="live admission lease"):
            first.submit(lambda worker: None, scope, admission=lease)
    finally:
        lease.release()
        authority.close()


def test_callback_completion_keeps_registry_cleanup_owned(harness, monkeypatch):
    h = harness()
    model = h.model(lambda table: table)
    pool, _ = model
    executor = h.executor(model)
    entered, release = threading.Event(), threading.Event()
    complete = executor._complete_task_submit

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(15), "GPU completion callback was not released"
        return complete(*args, **kwargs)

    monkeypatch.setattr(executor, "_complete_task_submit", delayed)
    try:
        _admit(executor)
        executor.submit(pa.table({"x": [7]}))
        assert entered.wait(10)
        assert _states(pool) == ["completing"]
        h.borrows[0].shutdown()
        with pytest.raises(RuntimeError, match="cleanup"):
            h.registry.close(kill=True)
        assert h.registry.resource_snapshot()["reserved_resources"]["gpu"] == 1
        assert pool.cleanup_pending()
    finally:
        release.set()
    assert _result(executor).to_pydict() == {"x": [7]}
    h.registry.close(timeout=15, kill=True)
    assert _demand(pool) == 0


def test_cancelling_a_submitted_future_before_it_runs_retires_device_demand(harness, tmp_path):
    marker = str(tmp_path / "executed")

    def call(table):
        Path(marker).touch()
        return table

    h = harness()
    model = h.model(call)
    pool, _ = model
    executor = h.executor(model)
    entered = [threading.Event() for _ in range(pool._executor._max_workers)]
    release = threading.Event()

    def occupy_thread(index):
        entered[index].set()
        assert release.wait(15), "GPU executor thread was not released"

    blockers = [pool._executor.submit(occupy_thread, index) for index in range(len(entered))]
    try:
        assert all(event.wait(10) for event in entered)
        _admit(executor)
        executor.submit(pa.table({"x": [7]}))
        assert _states(pool) == ["submitted"]
        executor.close(kill=True)
        assert _demand(pool) == 0
        assert h.runtime.snapshot()["running_tasks"] == 0
        assert not Path(marker).exists()
    finally:
        release.set()
        for blocker in blockers:
            blocker.result(timeout=10)


def test_idle_worker_loss_does_not_retain_a_completed_invocations_execution(harness, monkeypatch):
    h = harness()
    model = h.model(lambda table: pa.table({"pid": [os.getpid()]}))
    pool, _ = model
    executor = h.executor(model)
    original = pool._run
    old_pid = pool.worker_pids()[0]

    def die_after_returning_to_idle(*args, **kwargs):
        result = original(*args, **kwargs)
        pool._workers[0]._proc.kill()
        pool._workers[0]._proc.wait(timeout=5)
        return result

    with monkeypatch.context() as fault:
        fault.setattr(pool, "_run", die_after_returning_to_idle)
        _admit(executor)
        executor.submit(pa.table({"x": [7]}))
        assert _result(executor).to_pydict() == {"pid": [old_pid]}
    assert _demand(pool) == 0
    assert h.registry.resource_snapshot()["reserved_resources"]["gpu"] == 1
    _admit(executor)
    executor.submit(pa.table({"x": [7]}))
    assert _result(executor).column(0)[0].as_py() != old_pid
    assert pool.device_snapshot()[0]["generation"] == 1
    assert _demand(pool) == 0


def test_cancel_prefetched_call_does_not_claim_the_active_calls_worker(harness, tmp_path):
    h = harness()
    model = h.model(_gated(h, tmp_path))
    pool, _ = model
    first, cancelled, third = [h.executor(model) for _ in range(3)]
    _admit(first)
    first.submit(pa.table({"x": [1]}))
    _wait(lambda: (tmp_path / "entered-1").exists())
    pid = pool.worker_pids()[0]
    _admit(cancelled)
    cancelled.submit(pa.table({"x": [2]}))
    assert not (tmp_path / "entered-2").exists()
    cancelled.close(kill=True)
    _wait(lambda: not cancelled.cleanup_pending())
    assert pool.worker_pids() == [pid]
    assert _states(pool) == ["running"]
    assert pool.admission_slots.active_lease_count == 1
    _admit(third)
    third.submit(pa.table({"x": [3]}))
    (tmp_path / "release-1").touch()
    assert _result(first).column("x").to_pylist() == [1]
    assert _result(third).column("x").to_pylist() == [3]
    assert pool.worker_pids() == [pid]
    assert not (tmp_path / "entered-2").exists()
