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
@pytest.mark.parametrize("cancel_at", ["before_decode", "after_decode"])
@pytest.mark.parametrize("foreign_grant", [False, True])
def test_graceful_cancel_preserves_worker_with_late_ref_result(monkeypatch, backend, cancel_at, foreign_grant):
    from vane.execution import ref_bundle
    from vane.execution import udf_subprocess as local

    metrics = WorkerMetrics()
    payload = dict(
        _task_payload(),
        function_pickle=vane_pickle.dumps(_Actor if backend == "subprocess_actor" else _task),
        execution_backend=backend,
        actor_number=1,
        produce_ref_bundle_output=True,
        streaming_output_mode="local_shm_ref_bundle",
    )
    pool = local.LocalSubprocessActorPool(payload, 1, worker_metrics=metrics) if backend == "subprocess_actor" else None
    options = {"local_worker_metrics": metrics}
    if pool is not None:
        options["local_actor_pool"] = pool
    first, second = [build_executor(payload, options) for _ in range(2)]
    received, resume, cancelled = threading.Event(), threading.Event(), threading.Event()
    observed = []
    decode = local._SingleSubprocessExecutor._decode_ref_bundle_result
    wait_for_cleanup = first._wait_for_pending_futures
    other_grant = ref_bundle.request_local_shm_output_grant(64, name="another-query") if foreign_grant else None

    def pause_result(worker, data):
        if observed:
            return decode(worker, data)
        observed.append(worker)
        decoded = decode(worker, data) if cancel_at == "after_decode" else None
        received.set()
        assert resume.wait(10), "late result was not resumed"
        if other_grant is not None:
            if decoded is None:
                data = vane_pickle.dumps(dict(vane_pickle.loads(data), grant_id=other_grant))
            else:
                decoded["grant_id"] = other_grant
        return decode(worker, data) if decoded is None else decoded

    def wait_after_cancel(timeout):
        cancelled.set()
        return wait_for_cleanup(timeout)

    def pid():
        result = _execute(second, 1)
        assert not isinstance(result, BaseException), result
        try:
            return ref_bundle.materialize_ref_bundle(result[1], metadata=result[2]).column(0)[0].as_py()
        finally:
            for ref in result[1]:
                ref.release()

    try:
        original_pid = pid()
        before = ref_bundle.local_shm_ref_budget_snapshot()["allocated_bytes"]
        monkeypatch.setattr(local._SingleSubprocessExecutor, "_decode_ref_bundle_result", pause_result)
        monkeypatch.setattr(first, "_wait_for_pending_futures", wait_after_cancel)
        table = pa.table({"x": [1]})
        assert first.request_task_admission(table.nbytes)
        assert first.task_admission_state()["available"]
        first.submit(table)
        assert received.wait(10), "worker did not return its output descriptor"
        with ThreadPoolExecutor(1) as closer:
            closing = closer.submit(first.close, kill=False)
            try:
                assert cancelled.wait(3), "graceful close did not cancel the task"
                worker = observed[0]
                assert worker._current_execution_scope().is_set()
                assert not worker._active_output_grants
            finally:
                resume.set()
            closing.result(timeout=10)
        assert pid() == original_pid
        assert nonzero(metrics) == {}
        assert ref_bundle.local_shm_ref_budget_snapshot()["allocated_bytes"] == before
        if other_grant is not None:
            assert ref_bundle.local_shm_budget_manager().output_grant_pending(other_grant)
    finally:
        resume.set()
        first.close(kill=True)
        second.close(kill=True)
        if pool is not None:
            pool.shutdown(kill=True)
        if other_grant is not None:
            ref_bundle.release_local_shm_output_grant(other_grant, name="another-query")


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


@pytest.mark.parametrize("failure", ["input_ack", "serialization", "deserialization_oom", "wakeup"])
def test_parent_side_failures_are_not_worker_losses(monkeypatch, failure):
    import vane.execution.udf_subprocess as subprocess_exec

    metrics = WorkerMetrics()
    worker = subprocess_exec._SingleSubprocessExecutor(
        _task_payload(), startup_observer=lambda worker: worker._worker_lifecycle.bind(metrics)
    )

    def fail(*_args, **_kwargs):
        assert worker._proc.poll() is None
        if failure == "deserialization_oom":
            raise MemoryError("injected parent-side failure")
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
        elif failure == "deserialization_oom":
            with monkeypatch.context() as context:
                context.setattr(subprocess_exec.vane_pickle, "loads", fail)
                with pytest.raises(RuntimeError, match="injected parent-side failure"):
                    worker.submit(pa.table({"x": [1]}))
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


@pytest.mark.parametrize("failure,field", [("serialization", "runtime_errors"), ("delivery", "worker_losses")])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_control_response_failure_precedes_grant_cleanup(monkeypatch, failure, field, cleanup_fails):
    import vane.execution.udf_subprocess as subprocess_exec

    metrics = WorkerMetrics()
    worker = subprocess_exec._SingleSubprocessExecutor(
        _task_payload(), startup_observer=lambda worker: worker._worker_lifecycle.bind(metrics)
    )
    release_grant = subprocess_exec.release_local_shm_output_grant
    before = subprocess_exec.local_shm_ref_budget_snapshot()["output_grant_bytes"]
    event = vane_pickle.dumps({"request_id": 1, "size_bytes": 296})

    def fail_response(*_args):
        assert worker._proc.poll() is None
        if failure == "serialization":
            raise MemoryError("injected control serialization failure")
        raise OSError("injected control delivery failure")

    def fail_release(*_args, **_kwargs):
        raise RuntimeError("injected grant cleanup failure")

    monkeypatch.setattr(worker, "_recv_expected", lambda _expected: (subprocess_exec._MSG_OUTPUT_GRANT_REQUEST, event))
    if failure == "serialization":
        monkeypatch.setattr(subprocess_exec.vane_pickle, "dumps", fail_response)
    else:
        monkeypatch.setattr(subprocess_exec, "_send_message", fail_response)
    if cleanup_fails:
        monkeypatch.setattr(subprocess_exec, "release_local_shm_output_grant", fail_release)
    try:
        message = "injected grant cleanup failure" if cleanup_fails else f"injected control {failure} failure"
        with pytest.raises(RuntimeError, match=message):
            worker._recv_submit_result()
        assert worker._cleanup_finished is not cleanup_fails
        assert nonzero(metrics) == {field: 1}
        assert subprocess_exec.local_shm_ref_budget_snapshot()["output_grant_bytes"] == before + (
            296 if cleanup_fails else 0
        )
    finally:
        monkeypatch.setattr(subprocess_exec, "release_local_shm_output_grant", release_grant)
        worker.close(kill=True)
    assert worker._cleanup_finished
    assert subprocess_exec.local_shm_ref_budget_snapshot()["output_grant_bytes"] == before
    assert nonzero(metrics) == {field: 1}


@pytest.mark.parametrize(
    "message,field",
    [
        ("_MSG_INPUT_CONSUMED", "input_lease_id"),
        ("_MSG_INPUT_CONSUME_FAILED", "input_lease_id"),
        ("_MSG_OUTPUT_GRANT_REQUEST", "size_bytes"),
        ("_MSG_OUTPUT_GRANT_RELEASE", "grant_id"),
    ],
)
@pytest.mark.parametrize("malformed", ["pickle", "mapping", "missing", "integer"])
def test_malformed_control_messages_are_worker_losses(monkeypatch, message, field, malformed):
    import vane.execution.udf_subprocess as subprocess_exec

    metrics = WorkerMetrics()
    worker = subprocess_exec._SingleSubprocessExecutor(
        _task_payload(), startup_observer=lambda worker: worker._worker_lifecycle.bind(metrics)
    )
    payloads = {
        "pickle": b"invalid pickle",
        "mapping": vane_pickle.dumps(None),
        "missing": vane_pickle.dumps({}),
        "integer": vane_pickle.dumps({field: "not an integer"}),
    }
    monkeypatch.setattr(
        worker, "_recv_expected", lambda _expected: (getattr(subprocess_exec, message), payloads[malformed])
    )
    close_data = worker._close_data_shm

    def fail_close():
        raise RuntimeError("injected protocol cleanup failure")

    if malformed == "pickle":
        monkeypatch.setattr(worker, "_close_data_shm", fail_close)
    try:
        with pytest.raises(RuntimeError):
            worker._recv_submit_result()
        assert nonzero(metrics) == {"worker_losses": 1}
        assert not worker.is_reusable()
        assert worker._cleanup_finished is (malformed != "pickle")
    finally:
        monkeypatch.setattr(worker, "_close_data_shm", close_data)
        worker.close(kill=True)
    assert worker._cleanup_finished
    assert nonzero(metrics) == {"worker_losses": 1}


@pytest.mark.parametrize("cleanup_failure", [None, "input", "wakeup"])
def test_worker_input_failure_precedes_parent_cleanup(monkeypatch, cleanup_failure):
    import vane.execution.udf_subprocess as subprocess_exec

    metrics = WorkerMetrics()
    worker = subprocess_exec._SingleSubprocessExecutor(
        _task_payload(), startup_observer=lambda worker: worker._worker_lifecycle.bind(metrics)
    )
    _marker, refs, metadata, names = subprocess_exec.make_local_shm_ref_bundle_result(pa.table({"x": [1]}))
    payload, lease_id = subprocess_exec._make_local_ref_bundle_worker_payload_with_lease(
        refs, None, metadata, names, submit_id=None, name="test-input", reserve_output_credit=False
    )
    assert payload is not None
    # Keep the real lease alive, but send an unreadable descriptor so the worker
    # reports INPUT_CONSUME_FAILED before its final ERROR message.
    payload["block_refs"][0]["shm_name"] = refs[0].name + "-missing"
    cancel_input = subprocess_exec.cancel_local_shm_input_lease

    def fail_cleanup(*_args, **_kwargs):
        raise RuntimeError("injected parent cleanup failure")

    if cleanup_failure == "input":
        monkeypatch.setattr(subprocess_exec, "cancel_local_shm_input_lease", fail_cleanup)
    elif cleanup_failure == "wakeup":
        worker.register_wakeup(fail_cleanup)
    try:
        message = "injected parent cleanup failure" if cleanup_failure else "FileNotFoundError"
        with pytest.raises(RuntimeError, match=message):
            worker._submit_ref_bundle_direct(payload)
        assert nonzero(metrics) == {"execution_errors": 1}
        assert not worker.is_reusable()
        if cleanup_failure == "input":
            assert not worker._cleanup_finished
            assert lease_id in worker._active_input_leases
            assert subprocess_exec.local_shm_budget_manager().input_lease_pending(lease_id)
    finally:
        monkeypatch.setattr(subprocess_exec, "cancel_local_shm_input_lease", cancel_input)
        worker.register_wakeup(None)
        worker.close(kill=True)
        for ref in refs:
            ref.release()
    assert worker._cleanup_finished
    assert not subprocess_exec.local_shm_budget_manager().input_lease_pending(lease_id)
    assert nonzero(metrics) == {"execution_errors": 1}


@pytest.mark.parametrize("backend", ["subprocess_actor", "subprocess_task"])
@pytest.mark.parametrize(
    "failure",
    [
        "pickle",
        "mapping",
        "block",
        "block_size",
        "missing_shm",
        "metadata",
        "metadata_entry",
        "rows",
        "row_range",
        "bytes",
        "slice",
        "names",
        "grant",
        "inactive_grant",
        "allocation",
    ],
)
def test_native_final_response_failures_are_accounted_before_cleanup(monkeypatch, backend, failure):
    from vane.execution import ref_bundle
    from vane.execution import udf_subprocess as local

    receive, loads = local._recv_message, vane_pickle.loads
    descriptors = []
    allocation_payload = b"injected final response allocation failure"

    def receive_invalid_result(sock):
        message, payload = receive(sock)
        if message != local._MSG_REF_BUNDLE_RESULT or descriptors:
            return message, payload
        descriptor = loads(payload)
        descriptors.append(descriptor)
        if failure == "pickle":
            return message, b"invalid result pickle"
        if failure == "mapping":
            return message, vane_pickle.dumps(None)
        if failure == "allocation":
            return message, allocation_payload
        changes = {
            "block": {"block_refs": [{"provider": "local_shm", "ipc_size_bytes": 296}]},
            "block_size": {"block_refs": [dict(descriptor["block_refs"][0], ipc_size_bytes=0)]},
            "missing_shm": {
                "block_refs": [
                    dict(descriptor["block_refs"][0], shm_name=descriptor["block_refs"][0]["shm_name"] + "-missing")
                ]
            },
            "metadata": {"metadata": [{}, {}]},
            "metadata_entry": {"metadata": [7]},
            "rows": {"metadata": [dict(descriptor["metadata"][0], num_rows="invalid rows")]},
            "row_range": {"metadata": [dict(descriptor["metadata"][0], num_rows=-1)]},
            "bytes": {"metadata": [dict(descriptor["metadata"][0], size_bytes="invalid bytes")]},
            "slice": {"metadata": [dict(descriptor["metadata"][0], slice_start=1, slice_end=0)]},
            "names": {"names": 7},
            "grant": {"grant_id": "invalid grant"},
            "inactive_grant": {"grant_id": descriptor["grant_id"] + 1_000_000},
        }
        return message, vane_pickle.dumps(dict(descriptor, **changes[failure]))

    def allocate_result(payload):
        if payload == allocation_payload:
            raise MemoryError("injected final response allocation failure")
        return loads(payload)

    monkeypatch.setattr(local, "_recv_message", receive_invalid_result)
    monkeypatch.setattr(vane_pickle, "loads", allocate_result)
    try:
        with vane.connect(config={"threads": 2}) as connection:
            runtime = connection.configure_local_runtime(request_limit=RequestAdmissionLimits(2, 4))

            def query():
                return connection.sql("SELECT 1::BIGINT AS x").map_batches(
                    _Actor if backend == "subprocess_actor" else _task,
                    schema={"pid": "BIGINT"},
                    execution_backend=backend,
                    **({"actor_number": 1} if backend == "subprocess_actor" else {}),
                )

            with pytest.raises(Exception):
                query().fetchall()
            assert len(descriptors) == 1
            snapshot = runtime.resource_snapshot()
            outcomes = {key: value for key, value in snapshot["worker_failures"].items() if value}
            outcomes.pop("shutdown_workers", None)
            field = "runtime_errors" if failure == "allocation" else "worker_losses"
            assert outcomes == {field: 1}
            assert snapshot["request_admission"]["failed_executions"] == 1
            assert snapshot["request_admission"]["active_requests"] == 0
            assert query().fetchall()[0][0] > 0
            assert runtime.resource_snapshot()["worker_failures"][field] == 1
    finally:
        # Corrupting a response can hide names from the parent. The probe owns
        # the original descriptor so its injected protocol damage leaks no shm.
        for descriptor in descriptors:
            ref_bundle.release_local_shm_ref_bundle_descriptor(descriptor)


@pytest.mark.parametrize("failure", ["missing_shm", "allocation"])
@pytest.mark.parametrize("cleanup_failure", [None, "budget", "descriptor", "worker"])
def test_result_adoption_failure_keeps_category_and_retires_worker(monkeypatch, failure, cleanup_failure):
    from vane.execution import ref_bundle
    from vane.execution import udf_subprocess as local

    metrics = WorkerMetrics()
    worker = local._SingleSubprocessExecutor(
        dict(_task_payload(), produce_ref_bundle_output=True, streaming_output_mode="local_shm_ref_bundle"),
        startup_observer=lambda worker: worker._worker_lifecycle.bind(metrics),
    )
    before = ref_bundle.local_shm_ref_budget_snapshot()["allocated_bytes"]
    receive, open_shm = local._recv_message, ref_bundle._open_existing_shm
    release_budget, close_data = ref_bundle._release_local_shm_ref_budget, worker._close_data_shm
    release_descriptor = local.release_local_shm_ref_bundle_descriptor
    descriptors = []
    injected = []

    def missing_result(sock):
        message, payload = receive(sock)
        if message == local._MSG_REF_BUNDLE_RESULT:
            descriptor = vane_pickle.loads(payload)
            descriptors.append(descriptor)
            if failure == "missing_shm":
                changed = dict(
                    descriptor,
                    block_refs=[
                        dict(descriptor["block_refs"][0], shm_name=descriptor["block_refs"][0]["shm_name"] + "-missing")
                    ],
                )
                payload = vane_pickle.dumps(changed)
        return message, payload

    def fail_allocation(name, *, track):
        if descriptors and name == descriptors[0]["block_refs"][0]["shm_name"]:
            raise MemoryError("injected adoption allocation failure")
        return open_shm(name, track=track)

    def fail_cleanup(*_args, **_kwargs):
        injected.append(cleanup_failure)
        raise RuntimeError("injected adoption cleanup failure")

    def fail_budget_callback(size, *, name=""):
        release_budget(size, name=name)
        # Model a failing post-release notification, without leaking a test
        # allocation. The failed shm open must remain the primary exception.
        if descriptors and not injected:
            fail_cleanup()

    monkeypatch.setattr(local, "_recv_message", missing_result)
    if failure == "allocation":
        monkeypatch.setattr(ref_bundle, "_open_existing_shm", fail_allocation)
    if cleanup_failure == "budget":
        monkeypatch.setattr(ref_bundle, "_release_local_shm_ref_budget", fail_budget_callback)
    elif cleanup_failure == "descriptor":
        monkeypatch.setattr(local, "release_local_shm_ref_bundle_descriptor", fail_cleanup)
    elif cleanup_failure == "worker":
        monkeypatch.setattr(worker, "_close_data_shm", fail_cleanup)
    field = "worker_losses" if failure == "missing_shm" else "runtime_errors"
    try:
        with pytest.raises(RuntimeError, match="result decoding failed") as raised:
            worker.submit(pa.table({"x": [1]}))
        assert isinstance(raised.value.__cause__, FileNotFoundError if failure == "missing_shm" else MemoryError)
        assert bool(injected) is (cleanup_failure is not None)
        assert nonzero(metrics) == {field: 1}
        assert not worker.is_reusable()
    finally:
        monkeypatch.setattr(ref_bundle, "_open_existing_shm", open_shm)
        monkeypatch.setattr(ref_bundle, "_release_local_shm_ref_budget", release_budget)
        monkeypatch.setattr(local, "release_local_shm_ref_bundle_descriptor", release_descriptor)
        monkeypatch.setattr(worker, "_close_data_shm", close_data)
        worker.close(kill=True)
        for descriptor in descriptors:
            release_descriptor(descriptor)
    assert worker._cleanup_finished
    assert nonzero(metrics) == {field: 1}
    assert ref_bundle.local_shm_ref_budget_snapshot()["allocated_bytes"] == before


@pytest.mark.parametrize("path", ["recv_header", "recv_payload", "send_submit", "send_grant", "send_finished"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_frame_allocation_failure_is_not_worker_loss(monkeypatch, path, cleanup_fails):
    from vane.execution import udf_subprocess as local

    metrics = WorkerMetrics()
    worker = local._SingleSubprocessExecutor(
        dict(_task_payload(), produce_ref_bundle_output=True, streaming_output_mode="local_shm_ref_bundle"),
        startup_observer=lambda worker: worker._worker_lifecycle.bind(metrics),
    )
    read_exact, header = local._read_exact, local._HEADER
    close_data = worker._close_data_shm
    injected = []

    def allocation_failure():
        assert worker._proc.poll() is None
        injected.append(path)
        raise MemoryError("injected frame allocation failure")

    def read(sock, size):
        if path == "recv_header" or size != header.size:
            allocation_failure()
        return read_exact(sock, size)

    class FrameHeader:
        size = header.size

        def pack(self, message, size):
            target = {
                "send_submit": local._MSG_SUBMIT_REF_BUNDLE,
                "send_grant": local._MSG_OUTPUT_GRANT_GRANTED,
                "send_finished": local._MSG_FINISHED,
            }[path]
            if message == target:
                allocation_failure()
            return header.pack(message, size)

        def unpack(self, data):
            return header.unpack(data)

    def fail_close():
        raise RuntimeError("injected frame cleanup failure")

    if path.startswith("recv_"):
        monkeypatch.setattr(local, "_read_exact", read)
    else:
        monkeypatch.setattr(local, "_HEADER", FrameHeader())
    if cleanup_fails:
        monkeypatch.setattr(worker, "_close_data_shm", fail_close)
    try:
        with pytest.raises(RuntimeError, match="injected frame (allocation|cleanup) failure"):
            if path == "send_finished":
                worker.finished_submitting()
            else:
                worker.submit(pa.table({"x": [1]}))
        assert injected == [path]
        assert nonzero(metrics) == {"runtime_errors": 1}
        assert not worker.is_reusable()
        assert worker._cleanup_finished is not cleanup_fails
    finally:
        monkeypatch.setattr(worker, "_close_data_shm", close_data)
        worker.close(kill=True)
    assert worker._cleanup_finished
    assert nonzero(metrics) == {"runtime_errors": 1}


@pytest.mark.parametrize("failure", ["size", "ipc", "allocation"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_direct_ipc_response_failure_precedes_cleanup(monkeypatch, failure, cleanup_fails):
    from vane.execution import udf_subprocess as local

    metrics = WorkerMetrics()
    worker = local._SingleSubprocessExecutor(
        _task_payload(), startup_observer=lambda worker: worker._worker_lifecycle.bind(metrics)
    )
    receive, close_data = local._recv_message, worker._close_data_shm

    def invalid_size(sock):
        message, payload = receive(sock)
        if message == local._MSG_OK:
            return message, local.struct.pack("<Q", 1)
        return message, payload

    def invalid_ipc(*_args):
        assert worker._proc.poll() is None
        if failure == "allocation":
            raise MemoryError("injected IPC allocation failure")
        return b"invalid Arrow IPC"

    def fail_close():
        raise RuntimeError("injected IPC cleanup failure")

    if failure == "size":
        monkeypatch.setattr(local, "_recv_message", invalid_size)
    else:
        monkeypatch.setattr(local, "_read_ipc_from_shm", invalid_ipc)
    if cleanup_fails:
        monkeypatch.setattr(worker, "_close_data_shm", fail_close)
    try:
        with pytest.raises(Exception):
            worker.submit(pa.table({"x": [1]}))
        field = "runtime_errors" if failure == "allocation" else "worker_losses"
        assert nonzero(metrics) == {field: 1}
        assert not worker.is_reusable()
        assert worker._cleanup_finished is not cleanup_fails
    finally:
        monkeypatch.setattr(worker, "_close_data_shm", close_data)
        worker.close(kill=True)
    assert worker._cleanup_finished
    assert nonzero(metrics) == {field: 1}
