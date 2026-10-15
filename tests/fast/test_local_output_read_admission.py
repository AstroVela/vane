# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Local reads apply native capacity before exact-byte query admission."""

from __future__ import annotations

import gc
import threading
import time
from concurrent.futures import Future

import pyarrow as pa
import pytest

from tests.local_admission_helpers import linear_metadata
from vane import pickle as vane_pickle
from vane.ai._errors import DiagnosticError, summarize_error
from vane.execution import ref_bundle, udf_subprocess
from vane.execution.resources import ResourceVector

ZERO = {"rows": 0, "bytes": 0, "item_bytes": 0}
ONE = {"rows": 1, "bytes": 1, "item_bytes": 1}


def _wait(predicate):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    pytest.fail("local output admission did not progress")


@pytest.fixture
def prepare(monkeypatch):
    udf_subprocess._shutdown_global_task_runtime()
    runtime = udf_subprocess._GlobalSubprocessTaskRuntime(resource_limit=ResourceVector(cpu=2, heap_bytes=64 * 1024**2))
    monkeypatch.setattr(udf_subprocess, "_GLOBAL_TASK_RUNTIME", runtime)
    monkeypatch.setenv("VANE_LOCAL_SHM_REF_BUDGET_BYTES", str(256 * 1024))
    monkeypatch.setenv("VANE_LOCAL_SHM_STORE_BYTES", str(512 * 1024))
    owners = []

    def create(fn, backend="subprocess_task"):
        class Actor:
            def __call__(self, table):
                return fn(table)

        payload = {
            "function_pickle": vane_pickle.dumps(Actor if backend == "subprocess_actor" else fn),
            "call_mode": "map_batches",
            "execution_backend": backend,
            "actor_number": 1,
            "produce_ref_bundle_output": True,
            "streaming_output_mode": "local_shm_ref_bundle",
            "batch_size": 1,
            "output_batch_size": 1,
            "memory_bytes": 1024**2,
            "udf_task_input_max_bytes": 1024,
            "udf_output_target_max_bytes": 1024,
        }
        nodes = [{"node_id": "source", "payload": payload}]
        resources, options = udf_subprocess.ensure_local_subprocess_actor_pools_for_nodes(
            nodes, resource_graph=linear_metadata(nodes)
        )
        owners.extend(resources)
        executor = udf_subprocess.UDFExecutor(payload, options["source"])
        owners.append(executor)
        binding = options["source"]["local_query_admission"]
        return executor, binding.query, binding.unit_id, runtime

    yield create
    for owner in reversed(owners):
        if isinstance(owner, udf_subprocess.UDFExecutor):
            owner.close(kill=True)
        else:
            owner.shutdown(kill=True)
    runtime.close(kill=True)
    gc.collect()
    assert runtime.execution_capacity.resource_snapshot()["usage"] == ResourceVector().to_dict()
    assert ref_bundle.local_shm_ref_budget_snapshot()["usage_bytes"] == 0


def _request(executor):
    assert executor.request_task_admission(0)
    _wait(lambda: executor.task_admission_state()["available"])


def _submit(executor, submit_id=1):
    _request(executor)
    executor.submit_with_id(submit_id, pa.table({"x": [submit_id]}))


def _next(executor, capacity=ONE):
    return _wait(lambda: executor.take_ready_result(capacity))


def _release(item):
    for ref in item[2][1]:
        ref.release()


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
@pytest.mark.parametrize("end", ["complete", "error"])
def test_worker_completion_is_independent_of_native_and_query_output_capacity(prepare, backend, end):
    def produce(_table):
        yield pa.table({"x": [11]})
        if end == "error":
            raise ValueError("failure after output")

    executor, query, unit_id, runtime = prepare(produce, backend)
    _submit(executor)
    executor.finished_submitting()

    def completed():
        assert executor.take_ready_result(ZERO) is None
        state = query.snapshot()["units"][unit_id]
        assert state["num_task_outputs_generated"] == state["pending_output_count"] == 0
        assert not query.snapshot()["output_leases"]
        return executor._pending_batches == 0

    _wait(completed)
    assert query.snapshot()["units"][unit_id]["num_tasks_finished"] == 0
    if backend == "subprocess_task":
        assert runtime.execution_capacity.resource_snapshot()["usage"]["cpu"] == 0
    item = _next(executor)
    assert item[:2] == (ref_bundle.SUBMIT_RESULT_MARKER, 1)
    assert item[3] is False
    try:
        assert item[2][1][0].to_table().to_pydict() == {"x": [11]}
        assert query.snapshot()["units"][unit_id]["num_task_outputs_generated"] == 1
    finally:
        _release(item)
    terminal = _next(executor, ZERO)
    assert len(terminal) == 3
    if end == "error":
        assert isinstance(terminal[2], RuntimeError)
        assert "failure after output" in str(terminal[2])
    else:
        assert terminal[2] is None
    assert executor.all_tasks_finished()
    assert query.snapshot()["units"][unit_id]["num_tasks_finished"] == 1


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
@pytest.mark.parametrize("cancel", [False, True])
def test_output_grant_wait_is_nonblocking_and_keeps_producer_identity(prepare, monkeypatch, backend, cancel):
    monkeypatch.setenv("VANE_QUERY_OBJECT_STORE_FRACTION", "0.01")

    def produce(_table):
        for _ in range(2):
            yield pa.table({"blob": [b"x" * 32768]})

    executor, query, unit_id, _ = prepare(produce, backend)
    _submit(executor)
    executor.finished_submitting()
    first = _next(executor)
    second = None
    woke = threading.Event()
    executor.register_wakeup(woke.set)
    try:
        _wait(lambda: executor._pending_batches == 0)
        # Hold off the terminal consumer's soft-budget liveness escape so
        # this test exercises an asynchronous grant instead of a direct one.
        query.manager.set_external_consumer_waiting(False)
        second = executor.take_ready_result(ONE)
        assert second is None
        assert query.snapshot()["units"][unit_id]["pending_output_count"] == 1
        assert query.snapshot()["units"][unit_id]["num_tasks_finished"] == 0
        if cancel:
            executor.close(kill=True)
            assert query.snapshot()["units"][unit_id]["pending_output_count"] == 0
            assert query.snapshot()["units"][unit_id]["num_tasks_finished"] == 1
        else:
            woke.clear()
            _release(first)
            query.manager.set_external_consumer_waiting(True)
            assert woke.wait(5)
            second = _next(executor)
            assert second[2][1][0].to_table().to_pydict() == {"blob": [b"x" * 32768]}
            assert _next(executor, ZERO)[2] is None
    finally:
        _release(first)
        if second is not None:
            _release(second)


def test_other_invocation_terminal_bypasses_blocked_data_without_reordering_its_producer(prepare):
    def produce(table):
        if table["x"][0].as_py() == 1:
            yield pa.table({"x": [11]})
            raise ValueError("first producer failed")
        raise ValueError("second producer failed before output")

    executor, query, _, _ = prepare(produce)
    _submit(executor, 1)
    _submit(executor, 2)
    executor.finished_submitting()
    _wait(lambda: executor._pending_batches == 0)
    terminal = _next(executor, ZERO)
    assert terminal[:2] == (ref_bundle.SUBMIT_RESULT_MARKER, 2)
    assert "second producer failed before output" in str(terminal[2])
    assert executor.take_ready_result(ZERO) is None
    assert not query.snapshot()["output_leases"]
    item = _next(executor)
    try:
        assert item[1] == 1 and item[3] is False
    finally:
        _release(item)
    assert "first producer failed" in str(_next(executor, ZERO)[2])


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
def test_unread_outputs_keep_the_two_block_fanout_bound_and_cancel_cleanly(prepare, backend):
    def produce(_table):
        for value in range(10):
            yield pa.table({"x": [value]})

    executor, query, unit_id, _ = prepare(produce, backend)
    _submit(executor)
    _wait(lambda: len(executor._queue) == 2)
    assert executor.take_ready_result(ZERO) is None
    assert len(executor._queue) == 2
    assert executor._pending_batches == 1
    assert query.snapshot()["units"][unit_id]["num_task_outputs_generated"] == 0
    executor.close(kill=True)
    assert not executor.cleanup_pending()
    assert not query._buffered_outputs
    assert not query._live_tasks


def test_retained_view_keeps_query_bytes_after_producer_and_query_close(prepare):
    def produce(_table):
        yield pa.table({"x": [11]})

    executor, query, _, _ = prepare(produce)
    _submit(executor)
    executor.finished_submitting()
    item = _next(executor)
    view = item[2][1][0].to_table()
    size = item[2][1][0].size
    _release(item)
    assert _next(executor, ZERO)[2] is None
    executor.close(kill=True)
    query.shutdown()
    assert query.snapshot()["usage"]["object_store_bytes"] == size
    assert view.to_pydict() == {"x": [11]}
    del view
    gc.collect()
    assert query.snapshot()["usage"]["object_store_bytes"] == 0


def test_failed_unread_output_cleanup_retains_owner_for_close_retry(prepare, monkeypatch):
    def produce(_table):
        yield pa.table({"x": [11]})

    executor, query, _, _ = prepare(produce)
    _submit(executor)
    executor.finished_submitting()
    _wait(lambda: executor._pending_batches == 0)
    release = query.manager.release_task_lease
    failed = []

    def release_once(*args, **kwargs):
        if not failed:
            failed.append(True)
            raise RuntimeError("injected task cleanup failure")
        return release(*args, **kwargs)

    monkeypatch.setattr(query.manager, "release_task_lease", release_once)
    with pytest.raises(RuntimeError, match="injected task cleanup failure"):
        executor.close(kill=True)
    assert executor.cleanup_pending()
    assert executor._output_reads
    executor.close(kill=True)
    assert not executor.cleanup_pending()
    assert not query._live_tasks


def test_partial_buffer_failure_keeps_original_error_and_retry_owner(prepare, monkeypatch):
    executor, query, _, _ = prepare(lambda table: table)
    _request(executor)
    admission = executor._take_task_admission()
    scope = executor._new_execution_scope()
    refs = [ref_bundle.make_local_shm_ref_bundle_result(pa.table({"x": [value]}))[1][0] for value in (1, 2)]
    result = (ref_bundle.REF_BUNDLE_RESULT_MARKER, refs, None, ["x"])
    buffer = udf_subprocess.LocalOutputRead.buffer
    remove_waiter = query.manager.remove_output_waiter
    failed = []

    def buffer_then_fail(self, *args, **kwargs):
        buffer(self, *args, **kwargs)
        raise ValueError("injected buffer failure")

    def remove_once(*args, **kwargs):
        if not failed:
            failed.append(True)
            raise RuntimeError("injected output cleanup failure")
        return remove_waiter(*args, **kwargs)

    monkeypatch.setattr(udf_subprocess.LocalOutputRead, "buffer", buffer_then_fail)
    monkeypatch.setattr(query.manager, "remove_output_waiter", remove_once)
    future = Future()
    executor._pending_batches = 1
    executor._track_task_future(future, 1, 1, time.perf_counter(), admission, scope)
    future.set_result(result)
    assert executor._pending_batches == 0
    assert not executor._task_futures
    assert executor._output_reads
    item = _next(executor, ZERO)
    assert isinstance(item[2], DiagnosticError)
    assert "injected buffer failure" in str(item[2])
    primary, cleanup = summarize_error(item[2]).split("; cleanup: ")
    assert primary.split(" <- ")[0] == "ValueError"
    assert "RuntimeError" in cleanup
    assert "injected" not in summarize_error(item[2])
    executor.close(kill=True)
    assert not executor._output_reads
    assert not query._live_tasks


def test_completion_failure_discards_buffered_data_without_delaying_error(prepare):
    executor, query, _, _ = prepare(lambda table: table)
    _request(executor)
    admission = executor._take_task_admission()
    complete = admission._execution_finished_callback

    def fail_completion():
        complete()
        raise RuntimeError("injected completion failure")

    admission._execution_finished_callback = fail_completion
    scope = executor._new_execution_scope()
    result = ref_bundle.make_local_shm_ref_bundle_result(pa.table({"x": [1]}))
    future = Future()
    executor._pending_batches = 1
    executor._track_task_future(future, 1, 1, time.perf_counter(), admission, scope)
    future.set_result(result)
    item = _next(executor, ZERO)
    assert isinstance(item[2], DiagnosticError)
    assert "injected completion failure" in str(item[2])
    assert summarize_error(item[2]) == "RuntimeError"
    assert not executor._output_reads
    assert not query._buffered_outputs
    assert not query._live_tasks
    assert not query.snapshot()["output_leases"]
