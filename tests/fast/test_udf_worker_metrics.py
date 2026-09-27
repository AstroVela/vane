# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import os
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pytest

import vane
from vane import pickle as vane_pickle
from vane.execution.request_admission import RequestAdmissionLimits
from vane.execution.udf import build_executor
from vane.execution.udf_worker_metrics import WorkerLifecycle, WorkerMetrics, WorkerOutcome


def nonzero(metrics):
    return {key: value for key, value in metrics.snapshot().items() if value}


@pytest.mark.parametrize("first", list(WorkerOutcome))
def test_first_worker_outcome_survives_concurrent_cleanup_and_observers(first):
    metrics = WorkerMetrics()
    worker = WorkerLifecycle(metrics)
    worker.ready()
    with ThreadPoolExecutor(8) as observers:
        list(observers.map(worker.finish, [first] * 40))
        list(observers.map(worker.finish, list(WorkerOutcome) * 20))
    assert nonzero(metrics) == {first.value: 1}
    snapshot = metrics.snapshot()
    snapshot[first.value] = 999
    assert nonzero(metrics) == {first.value: 1}


@pytest.mark.parametrize("first", [WorkerOutcome.WORKER_LOSS, WorkerOutcome.EXECUTION_ERROR])
def test_failure_before_readiness_is_initialization_failure(first):
    metrics = WorkerMetrics()
    worker = WorkerLifecycle(metrics)
    worker.finish(first)
    worker.finish(WorkerOutcome.SHUTDOWN)
    assert nonzero(metrics) == {"initialization_failures": 1}


@pytest.mark.parametrize("first", [WorkerOutcome.CANCELLED, WorkerOutcome.SHUTDOWN])
def test_intentional_startup_termination_is_not_initialization_failure(first):
    metrics = WorkerMetrics()
    worker = WorkerLifecycle(metrics)
    worker.finish(first)
    worker.finish(WorkerOutcome.WORKER_LOSS)
    assert nonzero(metrics) == {first.value: 1}


def test_task_rebinding_does_not_republish_old_events_or_retain_previous_runtime():
    previous, current = WorkerMetrics(), WorkerMetrics()
    reference = weakref.ref(previous)
    worker = WorkerLifecycle(previous)
    worker.ready()
    worker.bind(None)
    del previous
    gc.collect()
    assert reference() is None
    worker.bind(current)
    worker.finish(WorkerOutcome.WORKER_LOSS)
    later = WorkerMetrics()
    worker.bind(later)
    worker.finish(WorkerOutcome.WORKER_LOSS)
    assert nonzero(current) == {"worker_losses": 1}
    assert nonzero(later) == {}
    worker_reference = weakref.ref(worker)
    del worker
    gc.collect()
    assert worker_reference() is None


def _task(table):
    value = table.column(0)[0].as_py()
    if value == -1:
        raise ValueError("planned execution failure")
    if value == -2:
        os._exit(23)
    return pa.table({"pid": [os.getpid()]})


class _Actor:
    def __call__(self, table):
        return _task(table)


def _task_payload():
    return {
        "function_pickle": vane_pickle.dumps(_task),
        "call_mode": "map_batches",
        "execution_backend": "subprocess_task",
        "udf_worker_slots": 1,
    }


def _execute(executor, value):
    table = pa.table({"x": [value]})
    ready = threading.Event()
    executor.register_wakeup(ready.set)
    try:
        assert executor.request_task_admission(table.nbytes)
        deadline = time.monotonic() + 15
        while not executor.task_admission_state()["available"]:
            ready.wait(0.01)
            assert time.monotonic() < deadline, "worker admission did not finish"
        executor.submit(table)
        while True:
            ready.clear()
            result = executor.take_ready_result()
            if result is not None:
                return result
            assert time.monotonic() < deadline, "worker result did not arrive"
            ready.wait(0.01)
    finally:
        executor.register_wakeup(None)


@pytest.mark.parametrize("failure,field", [(-1, "execution_errors"), (-2, "worker_losses")])
def test_cached_task_workers_attribute_failure_to_current_runtime(failure, field):
    first, second = WorkerMetrics(), WorkerMetrics()
    payload = _task_payload()
    executors = [
        build_executor(payload, {"local_worker_metrics": first}),
        build_executor(payload, {"local_worker_metrics": second}),
        build_executor(payload),
    ]
    try:
        assert executors[0]._task_pool is executors[1]._task_pool is executors[2]._task_pool
        original = _execute(executors[0], 1).column(0)[0].as_py()
        assert _execute(executors[1], 1).column(0)[0].as_py() == original
        assert isinstance(_execute(executors[1], failure), BaseException)
        assert nonzero(first) == {}
        assert nonzero(second) == {field: 1}
        replacement = _execute(executors[0], 1).column(0)[0].as_py()
        assert replacement != original
        assert _execute(executors[2], 1).column(0)[0].as_py() == replacement
        assert isinstance(_execute(executors[2], failure), BaseException)
        assert nonzero(first) == {}
        assert nonzero(second) == {field: 1}
        assert _execute(executors[0], 1).column(0)[0].as_py() != replacement
    finally:
        for executor in executors:
            executor.close(kill=True)
    assert nonzero(first) == {}
    assert nonzero(second) == {field: 1}


def test_idle_task_worker_loss_is_attributed_when_next_borrower_discovers_it():
    first, second = WorkerMetrics(), WorkerMetrics()
    payload = _task_payload()
    executors = [
        build_executor(payload, {"local_worker_metrics": first}),
        build_executor(payload, {"local_worker_metrics": second}),
    ]
    try:
        original = _execute(executors[0], 1).column(0)[0].as_py()
        pool = executors[0]._task_pool
        with pool.runtime.cond:
            worker = pool.idle[0].worker
        assert worker._proc.pid == original
        worker._proc.kill()
        worker._proc.wait(timeout=5)
        assert nonzero(first) == nonzero(second) == {}
        assert _execute(executors[1], 1).column(0)[0].as_py() != original
        assert nonzero(first) == {}
        assert nonzero(second) == {"worker_losses": 1}
    finally:
        for executor in executors:
            executor.close(kill=True)
    assert nonzero(first) == {}
    assert nonzero(second) == {"worker_losses": 1}


@pytest.mark.parametrize("backend", ["subprocess_actor", "subprocess_task"])
@pytest.mark.parametrize("failure,field", [(-1, "execution_errors"), (-2, "worker_losses")])
def test_native_unregistered_udfs_report_runtime_failures(backend, failure, field):
    with vane.connect(config={"threads": 2}) as connection:
        runtime = connection.configure_local_runtime(request_limit=RequestAdmissionLimits(2, 4))

        def query(value):
            return connection.sql(f"SELECT {value}::BIGINT AS x").map_batches(
                _Actor if backend == "subprocess_actor" else _task,
                schema={"pid": "BIGINT"},
                execution_backend=backend,
                **({"actor_number": 1} if backend == "subprocess_actor" else {}),
            )

        with pytest.raises(Exception, match="planned execution failure" if failure == -1 else "communication failed"):
            query(failure).fetchall()
        snapshot = runtime.resource_snapshot()
        failures = {key: count for key, count in snapshot["worker_failures"].items() if count}
        # A query-owned actor pool can finish replacing the failed worker before
        # request cleanup closes the replacement; that is an ordinary shutdown.
        failures.pop("shutdown_workers", None)
        assert failures == {field: 1}
        assert snapshot["request_admission"]["active_requests"] == 0
        assert snapshot["request_admission"]["failed_executions"] == 1
        assert query(1).fetchall()[0][0] > 0
        assert runtime.resource_snapshot()["worker_failures"][field] == 1
    assert runtime.resource_snapshot()["closed"]


def test_worker_metrics_options_reject_invalid_or_foreign_backend():
    payload = {"function_pickle": vane_pickle.dumps(_task), "call_mode": "map_batches"}
    with pytest.raises(TypeError, match="local_worker_metrics"):
        build_executor(dict(payload, execution_backend="subprocess_task"), {"local_worker_metrics": object()})
    with pytest.raises(ValueError, match="local worker metrics require"):
        build_executor(dict(payload, execution_backend="ray_task"), {"local_worker_metrics": WorkerMetrics()})


@pytest.mark.parametrize("before_spawn", [False, True])
def test_cancelled_startup_does_not_count_worker_loss_or_initialization_failure(monkeypatch, before_spawn):
    from vane.execution.udf_subprocess import _SingleSubprocessExecutor

    metrics = WorkerMetrics()
    observed = []
    original_receive = _SingleSubprocessExecutor._recv_expected

    def observe(worker):
        worker._worker_lifecycle.bind(metrics)
        observed.append(worker)
        if before_spawn:
            worker._cancel_startup()

    def cancel_before_ready(worker, *args, **kwargs):
        assert worker._proc is not None
        worker._cancel_startup()
        return original_receive(worker, *args, **kwargs)

    monkeypatch.setattr(_SingleSubprocessExecutor, "_recv_expected", cancel_before_ready)
    with pytest.raises(RuntimeError, match="startup.*cancelled"):
        _SingleSubprocessExecutor(_task_payload(), startup_observer=observe)
    assert len(observed) == 1
    worker = observed[0]
    worker.close(kill=True)
    assert worker._cleanup_finished
    assert nonzero(metrics) == {"cancelled_workers": 1}


@pytest.mark.parametrize("broken", [False, True])
def test_cleanup_retries_keep_original_worker_outcome(monkeypatch, broken):
    from vane.execution.udf_subprocess import _SingleSubprocessExecutor

    metrics = WorkerMetrics()
    worker = _SingleSubprocessExecutor(
        _task_payload(),
        startup_observer=lambda worker: worker._worker_lifecycle.bind(metrics),
    )
    close_data = worker._close_data_shm

    def fail_close():
        raise RuntimeError("injected shared-memory cleanup failure")

    monkeypatch.setattr(worker, "_close_data_shm", fail_close)
    try:
        with pytest.raises(RuntimeError, match="injected shared-memory cleanup failure"):
            if broken:
                worker._mark_broken("injected execution-scope cleanup failure")
            else:
                worker.close()
        assert not worker._cleanup_finished
        expected = {"runtime_errors" if broken else "shutdown_workers": 1}
        assert nonzero(metrics) == expected
    finally:
        monkeypatch.setattr(worker, "_close_data_shm", close_data)
        worker.close(kill=True)
    assert worker._cleanup_finished
    assert nonzero(metrics) == expected


@pytest.mark.parametrize("failure", ["input_ack", "serialization", "wakeup"])
def test_parent_side_failures_are_not_worker_losses(monkeypatch, failure):
    import vane.execution.udf_subprocess as subprocess_exec

    metrics = WorkerMetrics()
    worker = subprocess_exec._SingleSubprocessExecutor(
        _task_payload(), startup_observer=lambda worker: worker._worker_lifecycle.bind(metrics)
    )

    def fail(*_args, **_kwargs):
        assert worker._proc.poll() is None
        raise RuntimeError("injected parent-side failure")

    try:
        if failure == "input_ack":
            monkeypatch.setattr(subprocess_exec, "consume_local_shm_input_lease", fail)
            with pytest.raises(RuntimeError, match="injected parent-side failure"):
                worker.submit(pa.table({"x": [1]}))
        elif failure == "serialization":
            with monkeypatch.context() as context:
                context.setattr(subprocess_exec.vane_pickle, "dumps", fail)
                with pytest.raises(RuntimeError, match="injected parent-side failure"):
                    worker._submit_ref_bundle_direct({"estimated_num_rows": 1})
        else:
            worker.register_wakeup(fail)
            with pytest.raises(RuntimeError, match="injected parent-side failure"):
                worker.submit(pa.table({"x": [1]}))
        assert not worker.is_reusable()
        assert nonzero(metrics) == {"runtime_errors": 1}
    finally:
        worker.close(kill=True)
    assert worker._cleanup_finished
    assert nonzero(metrics) == {"runtime_errors": 1}


def test_control_delivery_failure_precedes_failed_grant_cleanup(monkeypatch):
    import vane.execution.udf_subprocess as subprocess_exec

    metrics = WorkerMetrics()
    worker = subprocess_exec._SingleSubprocessExecutor(
        _task_payload(), startup_observer=lambda worker: worker._worker_lifecycle.bind(metrics)
    )
    release_grant = subprocess_exec.release_local_shm_output_grant
    event = vane_pickle.dumps({"request_id": 1, "size_bytes": 296})

    def fail_send(*_args):
        raise OSError("injected control delivery failure")

    def fail_release(*_args, **_kwargs):
        raise RuntimeError("injected grant cleanup failure")

    monkeypatch.setattr(worker, "_recv_expected", lambda _expected: (subprocess_exec._MSG_OUTPUT_GRANT_REQUEST, event))
    monkeypatch.setattr(subprocess_exec, "_send_message", fail_send)
    monkeypatch.setattr(subprocess_exec, "release_local_shm_output_grant", fail_release)
    try:
        with pytest.raises(RuntimeError, match="injected grant cleanup failure"):
            worker._recv_submit_result()
        assert not worker._cleanup_finished
        assert nonzero(metrics) == {"worker_losses": 1}
    finally:
        monkeypatch.setattr(subprocess_exec, "release_local_shm_output_grant", release_grant)
        worker.close(kill=True)
    assert worker._cleanup_finished
    assert nonzero(metrics) == {"worker_losses": 1}
