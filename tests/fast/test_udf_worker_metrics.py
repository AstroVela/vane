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
    }


def _corrupt_shm_result(descriptor, failure, *, monkeypatch=None):
    from vane.execution import ref_bundle

    ref = descriptor["block_refs"][0]
    if failure in {"short_descriptor", "oversized_descriptor"}:
        size = 8 if failure == "short_descriptor" else ref["ipc_size_bytes"] + 1
        return dict(descriptor, block_refs=[dict(ref, ipc_size_bytes=size)])
    allocation = ref.get("allocation")
    if allocation is not None and failure in {"empty_mapping", "short_mapping"}:
        from vane.execution.udf_shm_store import LocalShmStore

        assert monkeypatch is not None
        buffer = LocalShmStore.buffer

        def damaged_region(store, slot):
            if slot.descriptor() == allocation:
                # Truncating a shared arena could SIGBUS unrelated views.
                # Inject the damaged view at the pooled buffer boundary.
                return memoryview(bytes(0 if failure == "empty_mapping" else 4))
            return buffer(store, slot)

        monkeypatch.setattr(LocalShmStore, "buffer", damaged_region)
        return descriptor
    name = allocation["shm_name"] if allocation is not None else ref["shm_name"]
    shm = ref_bundle._open_existing_shm(name, track=False)
    try:
        if failure in {"empty_mapping", "short_mapping"}:
            # Truncation is only valid for a standalone, test-owned mapping.
            assert allocation is None, "cannot truncate a shared arena"
            os.ftruncate(shm._fd, 0 if failure == "empty_mapping" else 4)
        else:
            start = allocation["offset"] + ref["allocation_offset"] if allocation is not None else 0
            size = ref["ipc_size_bytes"] if allocation is not None else shm.size
            region = shm.buf[start : start + size]
            try:
                if failure == "oversized_header":
                    region[:8] = (size + 1).to_bytes(8, "little")
                elif failure == "empty_payload":
                    region[:8] = bytes(8)
                elif failure == "invalid_ipc":
                    # Keep the descriptor, mapping and outer length header valid.
                    # Arrow rejects the schema only when the native consumer decodes it.
                    region[8:16] = bytes(8)
                elif failure == "truncated_ipc_body":
                    # A readable schema with a truncated record body exercises read_all
                    # (ArrowIOError/OSError), rather than the initial schema decoder.
                    region[:8] = (int.from_bytes(region[:8], "little") - 12).to_bytes(8, "little")
                else:
                    raise AssertionError(f"unexpected mapping corruption: {failure}")
            finally:
                region.release()
    finally:
        shm.close()
    return descriptor


def _release_corrupted_descriptors(descriptors):
    from vane.execution import ref_bundle

    for descriptor in descriptors:
        # Broken standalone mappings may not be openable for cleanup. Pooled
        # allocations stay owned by the peer/store; never unlink their arena.
        for ref in descriptor["block_refs"]:
            if "allocation" in ref:
                continue
            try:
                ref_bundle.shared_memory._posixshmem.shm_unlink("/" + ref["shm_name"])
            except FileNotFoundError:
                pass


def _take_result(executor, ready=None):
    from vane.execution import udf_subprocess as local

    chunks = []
    deadline = time.monotonic() + 15
    try:
        while True:
            if ready is not None:
                ready.clear()
            result = executor.take_ready_result()
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], bool):
                block, finished = result
                if block is not None:
                    chunks.append(block)
                if finished:
                    assert len(chunks) == 1, "expected one output block before task completion"
                    return chunks.pop()
            elif result is not None:
                return result
            assert time.monotonic() < deadline, "worker result did not arrive"
            if ready is None:
                time.sleep(0.01)
            else:
                ready.wait(0.01)
    finally:
        for chunk in chunks:
            local._release_local_ref_bundle_result(chunk)


def _submit_single(worker, value):
    # Public submit installs the stream receiver and queues the terminal frame.
    worker.submit(pa.table({"x": [value]}))
    return _take_result(worker)


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
        return _take_result(executor, ready)
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
def test_graceful_cancel_retires_worker_with_late_stream_chunk(monkeypatch, backend, cancel_at, foreign_grant):
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
        assert pid() != original_pid
        assert nonzero(metrics) == {"cancelled_workers": 1}
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
        refs, None, metadata, names, submit_id=None, name="test-input"
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
        "short_descriptor",
        "oversized_descriptor",
        "empty_mapping",
        "short_mapping",
        "oversized_header",
        "empty_payload",
        "invalid_ipc",
        "truncated_ipc_body",
        "metadata",
        "metadata_entry",
        "rows",
        "row_range",
        "bytes",
        "ipc_bytes",
        "ipc_bytes_fractional",
        "ipc_bytes_negative",
        "ipc_bytes_overflow",
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
        if message not in (local._MSG_REF_BUNDLE_RESULT, local._MSG_REF_BUNDLE_CHUNK) or descriptors:
            return message, payload
        descriptor = loads(payload)
        descriptors.append(descriptor)
        if failure == "pickle":
            return message, b"invalid result pickle"
        if failure == "mapping":
            return message, vane_pickle.dumps(None)
        if failure == "allocation":
            return message, allocation_payload
        if failure in {
            "short_descriptor",
            "oversized_descriptor",
            "empty_mapping",
            "short_mapping",
            "oversized_header",
            "empty_payload",
            "invalid_ipc",
            "truncated_ipc_body",
        }:
            return message, vane_pickle.dumps(_corrupt_shm_result(descriptor, failure, monkeypatch=monkeypatch))
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
            "ipc_bytes": {"metadata": [dict(descriptor["metadata"][0], ipc_size_bytes="invalid IPC bytes")]},
            "ipc_bytes_fractional": {"metadata": [dict(descriptor["metadata"][0], ipc_size_bytes=296.5)]},
            "ipc_bytes_negative": {"metadata": [dict(descriptor["metadata"][0], ipc_size_bytes=-1)]},
            "ipc_bytes_overflow": {"metadata": [dict(descriptor["metadata"][0], ipc_size_bytes=1 << 64)]},
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
            if failure in {
                "short_descriptor",
                "oversized_descriptor",
                "empty_mapping",
                "short_mapping",
                "oversized_header",
                "empty_payload",
                "invalid_ipc",
                "truncated_ipc_body",
                "ipc_bytes",
                "ipc_bytes_fractional",
                "ipc_bytes_negative",
                "ipc_bytes_overflow",
            }:
                # Other workers may still share the arena; this generation,
                # rather than its physical file, must have been released.
                with pytest.raises(ValueError):
                    ref_bundle.acquire_allocation(descriptors[0]["block_refs"][0]["allocation"])
            assert query().fetchall()[0][0] > 0
            assert runtime.resource_snapshot()["worker_failures"][field] == 1
    finally:
        # Corrupting a response can hide names from the parent. The probe owns
        # the original descriptor so its injected protocol damage leaks no shm.
        _release_corrupted_descriptors(descriptors)


@pytest.mark.parametrize("failure", ["invalid_ipc", "metadata_ipc_size"])
@pytest.mark.parametrize("track_data", [False, True])
def test_native_invalid_result_replaces_registered_model_worker(monkeypatch, failure, track_data):
    from vane.execution import ref_bundle
    from vane.execution import udf_subprocess as local

    @vane.cls(actor_number=1, return_dtype="BIGINT", name="deferred_ipc_pid")
    class Model:
        def __call__(self, value):
            return os.getpid()

    receive = local._recv_message
    descriptors = []

    def corrupt_result(sock):
        message, payload = receive(sock)
        if message in (local._MSG_REF_BUNDLE_RESULT, local._MSG_REF_BUNDLE_CHUNK) and not descriptors:
            descriptor = vane_pickle.loads(payload)
            descriptors.append(descriptor)
            if failure == "metadata_ipc_size":
                descriptor["metadata"][0]["ipc_size_bytes"] = "invalid IPC bytes"
                payload = vane_pickle.dumps(descriptor)
            else:
                _corrupt_shm_result(descriptor, failure, monkeypatch=monkeypatch)
        return message, payload

    before = ref_bundle.local_shm_ref_budget_snapshot()["allocated_bytes"]
    try:
        with vane.connect(config={"threads": 2}) as connection:
            runtime = connection.configure_local_runtime(
                request_limit=RequestAdmissionLimits(2, 4), track_data=track_data
            )
            model = runtime.register_model(
                "pid", Model(), version="v1", parameters=["BIGINT"], cpus=1, memory_bytes=4096
            )
            vane.attach_function(model, connection=connection)
            original = connection.execute("SELECT deferred_ipc_pid(1)").fetchone()[0]
            monkeypatch.setattr(local, "_recv_message", corrupt_result)
            error = "result decoding failed" if failure == "metadata_ipc_size" else "ArrowInvalid"
            with pytest.raises(Exception, match=error):
                connection.execute("SELECT deferred_ipc_pid(1)").fetchall()
            snapshot = runtime.resource_snapshot()
            assert {key: value for key, value in snapshot["worker_failures"].items() if value} == {"worker_losses": 1}
            assert snapshot["request_admission"]["failed_executions"] == 1
            assert snapshot["request_admission"]["active_requests"] == 0
            assert connection.execute("SELECT deferred_ipc_pid(1)").fetchone()[0] != original
            assert runtime.resource_snapshot()["worker_failures"]["worker_losses"] == 1
    finally:
        _release_corrupted_descriptors(descriptors)
    assert ref_bundle.local_shm_ref_budget_snapshot()["allocated_bytes"] == before


@pytest.mark.parametrize("failure", ["invalid_ipc", "truncated_ipc_body"])
@pytest.mark.parametrize("track_data", [False, True])
def test_native_chained_ipc_failure_replaces_registered_producer(monkeypatch, failure, track_data):
    from vane.execution import ref_bundle
    from vane.execution import udf_subprocess as local

    @vane.cls(actor_number=1, return_dtype="BIGINT", name="chained_producer_pid")
    class Producer:
        def __call__(self, value):
            return os.getpid()

    @vane.cls(actor_number=1, return_dtype="BIGINT", name="chained_consumer_id")
    class Consumer:
        def __call__(self, value):
            return value

    receive = local._recv_message
    descriptors = []

    def corrupt_first_result(sock):
        message, payload = receive(sock)
        if message in (local._MSG_REF_BUNDLE_RESULT, local._MSG_REF_BUNDLE_CHUNK) and not descriptors:
            descriptor = vane_pickle.loads(payload)
            descriptors.append(descriptor)
            _corrupt_shm_result(descriptor, failure, monkeypatch=monkeypatch)
        return message, payload

    before = ref_bundle.local_shm_ref_budget_snapshot()["allocated_bytes"]
    try:
        with vane.connect(config={"threads": 2}) as connection:
            runtime = connection.configure_local_runtime(
                request_limit=RequestAdmissionLimits(2, 4), track_data=track_data
            )
            for name, cls in (("producer", Producer), ("consumer", Consumer)):
                model = runtime.register_model(
                    name, cls(), version="v1", parameters=["BIGINT"], cpus=1, memory_bytes=4096
                )
                vane.attach_function(model, connection=connection)
            sql = "SELECT chained_consumer_id(chained_producer_pid(1))"
            original = connection.execute(sql).fetchone()[0]
            monkeypatch.setattr(local, "_recv_message", corrupt_first_result)
            with pytest.raises(Exception):
                connection.execute(sql).fetchall()
            snapshot = runtime.resource_snapshot()
            assert {key: value for key, value in snapshot["worker_failures"].items() if value} == {
                "worker_losses": 1,
                "execution_errors": 1,
            }
            assert snapshot["request_admission"]["failed_executions"] == 1
            assert snapshot["request_admission"]["active_requests"] == 0
            if track_data:
                assert snapshot["data"]["retained_bytes"] == 0
            assert connection.execute(sql).fetchone()[0] != original
            assert runtime.resource_snapshot()["worker_failures"]["worker_losses"] == 1
    finally:
        _release_corrupted_descriptors(descriptors)
    assert ref_bundle.local_shm_ref_budget_snapshot()["allocated_bytes"] == before


@pytest.mark.parametrize("failure", ["invalid_ipc", "truncated_ipc_body"])
@pytest.mark.parametrize("same_worker", [False, True])
@pytest.mark.parametrize("cleanup_failure", [None, "input", "producer"])
def test_chained_ipc_failure_notifies_only_the_bad_blocks_producer(monkeypatch, failure, same_worker, cleanup_failure):
    from vane.execution import ref_bundle
    from vane.execution import udf_subprocess as local

    good_metrics, bad_metrics, consumer_metrics = WorkerMetrics(), WorkerMetrics(), WorkerMetrics()
    payload = dict(_task_payload(), produce_ref_bundle_output=True, streaming_output_mode="local_shm_ref_bundle")

    def worker(metrics):
        return local._SingleSubprocessExecutor(payload, startup_observer=lambda w: w._worker_lifecycle.bind(metrics))

    good, bad = worker(good_metrics), worker(bad_metrics)
    consumer = bad if same_worker else worker(consumer_metrics)
    before = ref_bundle.local_shm_ref_budget_snapshot()["allocated_bytes"]
    cancel_input = local.cancel_local_shm_input_lease
    injected = []
    results = []

    def fail_cleanup():
        assert nonzero(bad_metrics) == {"worker_losses": 1}
        injected.append(cleanup_failure)
        raise RuntimeError("injected chained cleanup failure")

    def cancel(lease_id, *, name=""):
        if name == "udf-input" and not injected:
            fail_cleanup()
        return cancel_input(lease_id, name=name)

    try:
        for producer in (good, bad):
            results.append(_submit_single(producer, 1))
        ref = results[1][1][0]
        _corrupt_shm_result({"block_refs": [ref_bundle._local_shm_descriptor_from_ref(ref)]}, failure)
        if same_worker:
            # The physical task worker now belongs to the consuming runtime.
            bad._worker_lifecycle.bind(consumer_metrics)
        with monkeypatch.context() as patch:
            if cleanup_failure == "input":
                patch.setattr(local, "cancel_local_shm_input_lease", cancel)
            elif cleanup_failure == "producer":
                patch.setattr(bad, "_close_data_shm", fail_cleanup)
            with pytest.raises(RuntimeError, match="ArrowInvalid|OSError"):
                consumer.submit_ref_bundle(
                    results[0][1] + results[1][1], None, results[0][2] + results[1][2], results[0][3]
                )
        assert bool(injected) is (cleanup_failure is not None)
        assert nonzero(good_metrics) == {}
        assert good.is_reusable()
        assert nonzero(bad_metrics) == {"worker_losses": 1}
        assert not bad.is_reusable()
        assert nonzero(consumer_metrics) == ({} if same_worker else {"execution_errors": 1})
    finally:
        for result in results:
            for ref in result[1]:
                ref.release()
        for item in {good, bad, consumer}:
            item.close(kill=True)
    assert ref_bundle.local_shm_ref_budget_snapshot()["allocated_bytes"] == before


@pytest.mark.parametrize("failure", ["allocation", "projection"])
def test_chained_input_processing_failure_does_not_blame_producer(failure):
    from vane.execution import udf_subprocess as local

    class Consumer:
        def __init__(self):
            from vane.execution import ref_bundle

            def fail(*_args, **_kwargs):
                if failure == "allocation":
                    raise pa.ArrowMemoryError("planned downstream allocation failure")
                raise pa.ArrowInvalid("planned downstream projection failure")

            if failure == "allocation":
                pa.ipc.open_stream = fail
            else:
                ref_bundle._apply_ref_bundle_slices = fail

        def __call__(self, table):
            return table

    producer_metrics, consumer_metrics = WorkerMetrics(), WorkerMetrics()
    producer = local._SingleSubprocessExecutor(
        dict(_task_payload(), produce_ref_bundle_output=True, streaming_output_mode="local_shm_ref_bundle"),
        startup_observer=lambda w: w._worker_lifecycle.bind(producer_metrics),
    )
    consumer = local._SingleSubprocessExecutor(
        dict(_task_payload(), function_pickle=vane_pickle.dumps(Consumer), execution_backend="subprocess_actor"),
        startup_observer=lambda w: w._worker_lifecycle.bind(consumer_metrics),
    )
    result = None
    try:
        result = _submit_single(producer, 1)
        with pytest.raises(RuntimeError, match=f"planned downstream {failure} failure"):
            consumer.submit_ref_bundle(result[1], None, result[2], result[3])
        assert nonzero(producer_metrics) == {}
        assert producer.is_reusable()
        assert nonzero(consumer_metrics) == {"execution_errors": 1}
    finally:
        if result is not None:
            for ref in result[1]:
                ref.release()
        producer.close(kill=True)
        consumer.close(kill=True)


@pytest.mark.parametrize("invalid_event", ["block_index", "foreign_lease", "foreign_lease_cancelled"])
def test_chained_decode_failure_cannot_notify_or_release_unowned_inputs(monkeypatch, invalid_event):
    from vane.execution import ref_bundle
    from vane.execution import udf_subprocess as local

    producer_metrics, consumer_metrics = WorkerMetrics(), WorkerMetrics()
    payload = dict(_task_payload(), produce_ref_bundle_output=True, streaming_output_mode="local_shm_ref_bundle")
    producer = local._SingleSubprocessExecutor(
        payload, startup_observer=lambda w: w._worker_lifecycle.bind(producer_metrics)
    )
    consumer = local._SingleSubprocessExecutor(
        payload, startup_observer=lambda w: w._worker_lifecycle.bind(consumer_metrics)
    )
    receive = local._recv_message
    result, foreign_lease = None, None

    def invalid_failure(sock):
        message, data = receive(sock)
        if message == local._MSG_INPUT_CONSUME_FAILED:
            event = vane_pickle.loads(data)
            assert event["invalid_ipc_block"] == 0
            if invalid_event == "block_index":
                event["invalid_ipc_block"] = 1
            else:
                event["input_lease_id"] = foreign_lease
                if invalid_event == "foreign_lease_cancelled":
                    consumer._current_execution_scope().cancel("cancelled before input failure")
            data = vane_pickle.dumps(event)
        return message, data

    try:
        result = _submit_single(producer, 1)
        ref = result[1][0]
        foreign_lease = ref_bundle.create_local_shm_input_lease(result[1])
        _corrupt_shm_result({"block_refs": [ref_bundle._local_shm_descriptor_from_ref(ref)]}, "invalid_ipc")
        monkeypatch.setattr(local, "_recv_message", invalid_failure)
        with pytest.raises(RuntimeError):
            consumer.submit_ref_bundle(result[1], None, result[2], result[3])
        assert nonzero(producer_metrics) == {}
        assert producer.is_reusable()
        field = "cancelled_workers" if invalid_event == "foreign_lease_cancelled" else "worker_losses"
        assert nonzero(consumer_metrics) == {field: 1}
        assert ref_bundle.local_shm_budget_manager().input_lease_pending(foreign_lease)
    finally:
        monkeypatch.setattr(local, "_recv_message", receive)
        if foreign_lease is not None:
            ref_bundle.cancel_local_shm_input_lease(foreign_lease)
        if result is not None:
            for ref in result[1]:
                ref.release()
        producer.close(kill=True)
        consumer.close(kill=True)


@pytest.mark.parametrize("later_borrow", ["idle", "rebound", "cancelled", "replaced"])
@pytest.mark.parametrize("configured_producer", [True, False])
def test_deferred_task_ipc_failure_keeps_producer_attribution(monkeypatch, later_borrow, configured_producer):
    from vane.execution import ref_bundle
    from vane.execution.udf_lifecycle import ExecutionCancellationScope

    producer, borrower = WorkerMetrics(), WorkerMetrics()
    payload = dict(_task_payload(), produce_ref_bundle_output=True, streaming_output_mode="local_shm_ref_bundle")
    first = build_executor(payload, {"local_worker_metrics": producer} if configured_producer else {})
    second = build_executor(payload, {"local_worker_metrics": borrower})
    results = []

    def run(executor):
        result = _execute(executor, 1)
        assert not isinstance(result, BaseException), result
        results.append(result)
        return result

    try:
        result = run(first)
        worker = first._task_pool.idle[0].worker
        original = worker._proc.pid
        _corrupt_shm_result({"block_refs": [ref_bundle._local_shm_descriptor_from_ref(result[1][0])]}, "invalid_ipc")
        if later_borrow == "replaced":
            # A first terminal outcome and a new physical worker already exist.
            worker.close(kill=True)
            replacement_result = run(second)
            replacement = second._task_pool.idle[0].worker
            assert replacement._proc.pid != original
            assert ref_bundle.materialize_ref_bundle(replacement_result[1]).num_rows == 1
        elif later_borrow != "idle":
            # Decode after a different runtime has acquired the cached worker.
            scope = ExecutionCancellationScope("later-borrower", 1)
            wrapper = second._task_pool.acquire_worker(scope, worker_metrics=borrower)
            assert wrapper.worker is worker
            if later_borrow == "cancelled":
                scope.cancel("later query cancellation")
            monkeypatch.setattr(worker, "_active_execution_scope", scope)
        try:
            for _ in range(2):
                with pytest.raises(pa.ArrowInvalid):
                    ref_bundle.materialize_ref_bundle(result[1])
            assert not worker.is_reusable()
            expected = {"worker_losses": 1} if configured_producer and later_borrow != "replaced" else {}
            assert nonzero(producer) == expected
            assert nonzero(borrower) == {}
        finally:
            if later_borrow in {"rebound", "cancelled"}:
                worker._active_execution_scope = None
                second._task_pool.release_worker(wrapper, reusable=worker.is_reusable())
        valid = run(second)
        pid = ref_bundle.materialize_ref_bundle(valid[1]).column(0)[0].as_py()
        assert pid != original
        if later_borrow == "replaced":
            assert pid == replacement._proc.pid
    finally:
        for result in results:
            for ref in result[1]:
                ref.release()
        first.close(kill=True)
        second.close(kill=True)


@pytest.mark.parametrize("failure", [pa.ArrowInvalid, pa.ArrowMemoryError, OSError])
@pytest.mark.parametrize("cleanup_failure", [None, "worker", "mapping"])
def test_standalone_deferred_decode_failure_precedes_cleanup_and_preserves_category(
    monkeypatch, failure, cleanup_failure
):
    from vane.execution import ref_bundle
    from vane.execution import udf_subprocess as local

    metrics = WorkerMetrics()
    worker = local._SingleSubprocessExecutor(
        _task_payload(), startup_observer=lambda worker: worker._worker_lifecycle.bind(metrics)
    )
    before = ref_bundle.local_shm_ref_budget_snapshot()["allocated_bytes"]
    result = None
    close_owner = ref_bundle._LocalShmBufferOwner.close
    cleanup_calls = []

    def fail_decode(*_args, **_kwargs):
        raise failure("injected deferred decoding failure")

    def fail_worker_cleanup():
        cleanup_calls.append("worker")
        raise RuntimeError("injected worker cleanup failure")

    def fail_mapping_cleanup(owner):
        mapping = owner._shm
        is_result = mapping is not None and mapping.name == result[1][0].name
        close_owner(owner)
        if is_result and not cleanup_calls:
            cleanup_calls.append("mapping")
            raise RuntimeError("injected mapping cleanup failure")

    try:
        # Standalone mappings remain the local-input format. Keep their close
        # failure coverage separate from the pooled-buffer tests below.
        descriptor = ref_bundle.make_local_shm_ref_bundle_descriptor(_submit_single(worker, 1))
        result = ref_bundle.make_local_shm_ref_bundle_result_from_descriptor(
            descriptor, on_decode_error=worker._result_decode_error_handler()
        )
        with monkeypatch.context() as patch:
            patch.setattr(pa.ipc, "open_stream", fail_decode)
            if cleanup_failure == "worker":
                patch.setattr(worker, "_close_data_shm", fail_worker_cleanup)
            elif cleanup_failure == "mapping":
                patch.setattr(ref_bundle._LocalShmBufferOwner, "close", fail_mapping_cleanup)
            with pytest.raises(failure, match="injected deferred decoding failure") as raised:
                result[1][0].to_table()
            if cleanup_failure:
                assert isinstance(raised.value.__cause__, RuntimeError)
                assert cleanup_calls == [cleanup_failure]
        field = "runtime_errors" if issubclass(failure, MemoryError) else "worker_losses"
        assert nonzero(metrics) == {field: 1}
        assert not worker.is_reusable()
    finally:
        worker.close(kill=True)
        if result is not None:
            for ref in result[1]:
                ref.release()
    assert worker._cleanup_finished
    assert nonzero(metrics) == {field: 1}
    assert ref_bundle.local_shm_ref_budget_snapshot()["allocated_bytes"] == before


@pytest.mark.parametrize("failure", [pa.ArrowInvalid, pa.ArrowMemoryError, OSError])
@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_pooled_decode_failure_precedes_worker_cleanup_and_preserves_category(monkeypatch, failure, cleanup_failure):
    import traceback

    from vane.execution import ref_bundle
    from vane.execution import udf_subprocess as local

    metrics = WorkerMetrics()
    worker = local._SingleSubprocessExecutor(
        dict(_task_payload(), produce_ref_bundle_output=True, streaming_output_mode="local_shm_ref_bundle"),
        startup_observer=lambda worker: worker._worker_lifecycle.bind(metrics),
    )
    before = ref_bundle.local_shm_ref_budget_snapshot()["allocated_bytes"]
    result = None
    decode_error = failure("injected pooled decoding failure")
    cleanup_error = RuntimeError("injected worker cleanup failure")
    cleanup_calls = []

    def fail_decode(*_args, **_kwargs):
        raise decode_error

    def fail_cleanup():
        cleanup_calls.append("worker")
        raise cleanup_error

    try:
        worker.submit(pa.table({"x": [1]}))
        result, finished = worker.take_ready_result()
        assert not finished
        assert worker.take_ready_result() == (None, True)
        assert result[1][0]._allocation_lease is not None
        with monkeypatch.context() as patch:
            patch.setattr(pa.ipc, "open_stream", fail_decode)
            if cleanup_failure:
                patch.setattr(worker, "_close_data_shm", fail_cleanup)
            with pytest.raises(failure, match="injected pooled decoding failure") as raised:
                result[1][0].to_table()
            assert raised.value is decode_error
            if cleanup_failure:
                cause = raised.value.__cause__
                assert isinstance(cause, RuntimeError)
                assert "UDF subprocess close failed" in str(cause)
                assert cause.__cause__ is cleanup_error
                assert cleanup_calls == ["worker"]
            else:
                assert raised.value.__cause__ is None
                assert cleanup_calls == []
        field = "runtime_errors" if issubclass(failure, MemoryError) else "worker_losses"
        assert nonzero(metrics) == {field: 1}
        assert not worker.is_reusable()
    finally:
        worker.close(kill=True)
        if result is not None:
            for ref in result[1]:
                ref.release()
    assert worker._cleanup_finished
    assert nonzero(metrics) == {field: 1}
    # The injected decoder's traceback still holds its BufferReader argument.
    # Keep charging that live view after worker/descriptor cleanup; only the
    # last physical buffer release may return its transport bytes.
    assert ref_bundle.local_shm_ref_budget_snapshot()["allocated_bytes"] == before + result[1][0].size
    traceback.clear_frames(decode_error.__traceback__)
    if cleanup_failure:
        traceback.clear_frames(cleanup_error.__traceback__)
    assert ref_bundle.local_shm_ref_budget_snapshot()["allocated_bytes"] == before


def test_deferred_result_observer_does_not_retain_worker(monkeypatch):
    from vane.execution import udf_subprocess as local

    metrics = WorkerMetrics()
    worker = local._SingleSubprocessExecutor(
        dict(_task_payload(), produce_ref_bundle_output=True, streaming_output_mode="local_shm_ref_bundle"),
        startup_observer=lambda worker: worker._worker_lifecycle.bind(metrics),
    )
    result = _submit_single(worker, 1)
    worker_ref = weakref.ref(worker)
    try:
        worker.close(kill=True)
        del worker
        gc.collect()
        assert worker_ref() is None

        def fail_decode(*_args, **_kwargs):
            raise pa.ArrowInvalid("late invalid IPC")

        monkeypatch.setattr(pa.ipc, "open_stream", fail_decode)
        with pytest.raises(pa.ArrowInvalid, match="late invalid IPC"):
            result[1][0].to_table()
        assert nonzero(metrics) == {"shutdown_workers": 1}
    finally:
        for ref in result[1]:
            ref.release()


def test_deferred_ipc_failure_retires_worker_during_another_runtime_task(monkeypatch):
    from vane.execution import ref_bundle

    producer, borrower = WorkerMetrics(), WorkerMetrics()
    payload = dict(_task_payload(), produce_ref_bundle_output=True, streaming_output_mode="local_shm_ref_bundle")
    first = build_executor(payload, {"local_worker_metrics": producer})
    second = build_executor(payload, {"local_worker_metrics": borrower})
    entered, resume = threading.Event(), threading.Event()
    results = []
    try:
        result = _execute(first, 1)
        results.append(result)
        worker = first._task_pool.idle[0].worker
        original = worker._proc.pid
        receive = worker._recv_submit_result

        def pause_next_result():
            entered.set()
            assert resume.wait(10), "concurrent worker was not resumed"
            return receive()

        monkeypatch.setattr(worker, "_recv_submit_result", pause_next_result)
        _corrupt_shm_result({"block_refs": [ref_bundle._local_shm_descriptor_from_ref(result[1][0])]}, "invalid_ipc")
        with ThreadPoolExecutor(1) as queries:
            pending = queries.submit(_execute, second, 1)
            try:
                assert entered.wait(10), "next task did not acquire worker"
                with pytest.raises(pa.ArrowInvalid):
                    result[1][0].to_table()
                assert not worker.is_reusable()
            finally:
                resume.set()
            assert isinstance(pending.result(timeout=10), BaseException)
        assert nonzero(producer) == {"worker_losses": 1}
        assert nonzero(borrower) == {}
        replacement = _execute(second, 1)
        results.append(replacement)
        assert ref_bundle.materialize_ref_bundle(replacement[1]).column(0)[0].as_py() != original
        assert nonzero(borrower) == {}
    finally:
        resume.set()
        for result in results:
            for ref in result[1]:
                ref.release()
        first.close(kill=True)
        second.close(kill=True)


@pytest.mark.parametrize("failure", ["missing_shm", "allocation", "short_mapping", "empty_mapping"])
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
    receive, acquire = local._recv_message, ref_bundle.acquire_allocation
    release_budget, close_data = ref_bundle._release_local_shm_ref_budget, worker._close_data_shm
    release_descriptor = local.release_local_shm_ref_bundle_descriptor
    descriptors = []
    injected = []

    def missing_result(sock):
        message, payload = receive(sock)
        if message in (local._MSG_REF_BUNDLE_RESULT, local._MSG_REF_BUNDLE_CHUNK):
            descriptor = vane_pickle.loads(payload)
            descriptors.append(descriptor)
            if failure in {"short_mapping", "empty_mapping"}:
                payload = vane_pickle.dumps(_corrupt_shm_result(descriptor, failure, monkeypatch=monkeypatch))
        return message, payload

    def fail_allocation(allocation):
        if descriptors and allocation == descriptors[0]["block_refs"][0]["allocation"]:
            error = FileNotFoundError if failure == "missing_shm" else MemoryError
            raise error("injected adoption allocation failure")
        return acquire(allocation)

    def fail_cleanup(*_args, **_kwargs):
        injected.append(cleanup_failure)
        raise RuntimeError("injected adoption cleanup failure")

    def fail_budget_callback(size, *, name=""):
        release_budget(size, name=name)
        # Model a failing post-release notification, without leaking a test
        # allocation. The failed region acquisition remains the primary error.
        if descriptors and not injected:
            fail_cleanup()

    monkeypatch.setattr(local, "_recv_message", missing_result)
    if failure in {"missing_shm", "allocation"}:
        monkeypatch.setattr(ref_bundle, "acquire_allocation", fail_allocation)
    if cleanup_failure == "budget":
        monkeypatch.setattr(ref_bundle, "_release_local_shm_ref_budget", fail_budget_callback)
    elif cleanup_failure == "descriptor":
        monkeypatch.setattr(local, "release_local_shm_ref_bundle_descriptor", fail_cleanup)
    elif cleanup_failure == "worker":
        monkeypatch.setattr(worker, "_close_data_shm", fail_cleanup)
    field = "runtime_errors" if failure == "allocation" else "worker_losses"
    try:
        with pytest.raises(RuntimeError, match="result decoding failed") as raised:
            worker.submit(pa.table({"x": [1]}))
        expected_error = {"missing_shm": FileNotFoundError, "allocation": MemoryError}.get(failure, ValueError)
        assert isinstance(raised.value.__cause__, expected_error)
        assert bool(injected) is (cleanup_failure is not None)
        assert nonzero(metrics) == {field: 1}
        assert not worker.is_reusable()
    finally:
        monkeypatch.setattr(ref_bundle, "acquire_allocation", acquire)
        monkeypatch.setattr(ref_bundle, "_release_local_shm_ref_budget", release_budget)
        monkeypatch.setattr(local, "release_local_shm_ref_bundle_descriptor", release_descriptor)
        monkeypatch.setattr(worker, "_close_data_shm", close_data)
        worker.close(kill=True)
        _release_corrupted_descriptors(descriptors)
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
