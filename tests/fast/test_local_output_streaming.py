# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Incremental local output ownership, byte admission and native completion."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pytest

from vane import pickle as vane_pickle
from vane.execution import ref_bundle, udf_subprocess
from vane.execution.resources import ResourceVector
from vane.execution.udf_worker_metrics import WorkerMetrics


@pytest.fixture
def runtime(monkeypatch):
    udf_subprocess._shutdown_global_task_runtime()
    runtime = udf_subprocess._GlobalSubprocessTaskRuntime(resource_limit=ResourceVector(cpu=2, heap_bytes=64 * 1024**2))
    monkeypatch.setattr(udf_subprocess, "_GLOBAL_TASK_RUNTIME", runtime)
    monkeypatch.setenv("VANE_LOCAL_SHM_REF_BUDGET_BYTES", str(128 * 1024))
    monkeypatch.setenv("VANE_LOCAL_SHM_STORE_BYTES", str(256 * 1024))
    yield runtime
    runtime.close(kill=True)
    assert runtime.execution_capacity.resource_snapshot()["usage"] == ResourceVector().to_dict()
    snapshot = ref_bundle.local_shm_ref_budget_snapshot()
    assert snapshot["allocated_bytes"] == snapshot["output_grant_bytes"] == 0


def _next(executor):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        result = executor.take_ready_result()
        if result is not None:
            return result
        time.sleep(0.01)
    pytest.fail("local output stream did not progress")


def _payload(fn, **extra):
    return {
        "function_pickle": vane_pickle.dumps(fn),
        "call_mode": "map_batches",
        "execution_backend": "subprocess_task",
        "actor_number": 1,
        "produce_ref_bundle_output": True,
        "streaming_output_mode": "local_shm_ref_bundle",
        "output_batch_size": 1,
        "batch_size": 2,
        "memory_bytes": 1024**2,
        **extra,
    }


def _consume(item):
    result = item[2]
    assert not isinstance(result, BaseException), str(result)
    try:
        table = ref_bundle.materialize_ref_bundle(result[1], None, result[2], result[3])
        return table.to_pydict()
    finally:
        for ref in result[1]:
            ref.release()


@pytest.mark.parametrize("end", ["complete", "error", "cancel"])
@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
def test_block_arrives_before_producer_finishes_and_retains_task_ownership(runtime, tmp_path, end, backend):
    resume = tmp_path / "resume"

    class Producer:
        def __call__(self, _table):
            yield pa.table({"blob": [b"x" * 32768]})
            deadline = time.monotonic() + 10
            while not resume.exists():
                if time.monotonic() > deadline:
                    raise RuntimeError("first block was not consumed")
                time.sleep(0.01)
            if end == "error":
                raise ValueError("failure after first block")
            for _ in range(9):
                yield pa.table({"blob": [b"y" * 32768]})

    def produce(table):
        yield from Producer()(table)

    payload = _payload(Producer if backend == "subprocess_actor" else produce, execution_backend=backend)
    pool = udf_subprocess.LocalSubprocessActorPool(payload, 1) if backend == "subprocess_actor" else None
    executor = udf_subprocess.UDFExecutor(payload, {"local_actor_pool": pool} if pool is not None else None)
    try:
        assert executor.request_task_admission(0)
        executor.submit_with_id(41, pa.table({"x": [1]}))
        first = _next(executor)
        assert first[:2] == (ref_bundle.SUBMIT_RESULT_MARKER, 41)
        assert not isinstance(first[2], BaseException), str(first[2])
        assert first[3] is False
        assert not resume.exists()
        if backend == "subprocess_task":
            assert runtime.execution_capacity.resource_snapshot()["usage"]["heap_bytes"] == 1024**2
        assert _consume(first) == {"blob": [b"x" * 32768]}
        if end == "cancel":
            executor.close(kill=True)
            return
        resume.touch()
        rows = 1
        while True:
            item = _next(executor)
            assert item[:2] == (ref_bundle.SUBMIT_RESULT_MARKER, 41)
            if len(item) == 3:
                if end == "error":
                    assert isinstance(item[2], RuntimeError)
                    assert "failure after first block" in str(item[2])
                else:
                    assert item[2] is None
                    assert rows == 10
                break
            assert item[3] is False
            assert _consume(item) == {"blob": [b"y" * 32768]}
            rows += 1
    finally:
        executor.close(kill=True)
        if pool is not None:
            pool.shutdown(kill=True)


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
@pytest.mark.parametrize("completion_stage", ["before", "after"])
@pytest.mark.parametrize("timeout", [False, True], ids=["wait", "retry"])
def test_close_owns_completion_cleanup_without_stopping_shared_pool_peer(
    runtime, monkeypatch, tmp_path, backend, completion_stage, timeout
):
    def produce(table):
        value = table.column(0)[0].as_py()
        yield pa.table({"x": [value]})
        gate = tmp_path / str(value)
        deadline = time.monotonic() + 20
        while not gate.exists():
            if time.monotonic() > deadline:
                raise RuntimeError("stream was not resumed")
            time.sleep(0.01)
        yield pa.table({"x": [value + 10]})

    class Actor:
        def __call__(self, table):
            return produce(table)

    payload = _payload(
        Actor if backend == "subprocess_actor" else produce,
        execution_backend=backend,
        actor_number=2,
    )
    pool = udf_subprocess.LocalSubprocessActorPool(payload, 2) if backend == "subprocess_actor" else None
    options = {"local_actor_pool": pool} if pool is not None else None
    first, peer = [udf_subprocess.UDFExecutor(payload, options) for _ in range(2)]
    owned_pool = first._actor_pool if pool is not None else first._task_pool
    assert owned_pool is (peer._actor_pool if pool is not None else peer._task_pool)
    entered, release, waiting = (threading.Event() for _ in range(3))
    complete = first._complete_task_submit
    wait = first._wait_for_pending_futures

    def pause():
        entered.set()
        assert release.wait(15), "completion cleanup was not resumed"

    def delayed_completion(*args, **kwargs):
        if completion_stage == "before":
            pause()
        complete(*args, **kwargs)
        if completion_stage == "after":
            pause()

    def observe_wait(timeout_s=None):
        waiting.set()
        return wait(timeout_s)

    monkeypatch.setattr(first, "_complete_task_submit", delayed_completion)
    monkeypatch.setattr(first, "_wait_for_pending_futures", observe_wait)
    monkeypatch.setattr(udf_subprocess, "_subprocess_shutdown_grace_s", lambda: 0.05 if timeout else 10)
    try:
        for executor, value in ((first, 1), (peer, 2)):
            assert executor.request_task_admission(0)
            executor.submit_with_id(value, pa.table({"x": [value]}))
            assert _consume(_next(executor)) == {"x": [value]}

        with ThreadPoolExecutor(1) as closer:
            closing = closer.submit(first.close, kill=True)
            try:
                assert entered.wait(10)
                assert waiting.wait(5), "close skipped completion cleanup"
                if timeout:
                    with pytest.raises(RuntimeError, match="pending UDF cleanup"):
                        closing.result(timeout=5)
                    # Retrying while the callback is blocked keeps the same
                    # owner; it must not claim cleanup succeeded on the retry.
                    with pytest.raises(RuntimeError, match="pending UDF cleanup"):
                        first.close(kill=True)
                else:
                    assert not closing.done()
                assert not first._cleanup_finished
                assert first._task_futures and all(f.done() for f in first._task_futures)
                assert (first._actor_pool if pool is not None else first._task_pool) is owned_pool
                if pool is None:
                    assert owned_pool.ref_count == 2
                    expected_cpu = 2 if completion_stage == "before" else 1
                    assert runtime.execution_capacity.resource_snapshot()["usage"]["cpu"] == expected_cpu
            finally:
                release.set()
            if not timeout:
                closing.result(timeout=10)
        assert wait(10)
        first.close(kill=True)
        assert first._cleanup_finished
        assert not first.cleanup_pending()
        assert not first._task_futures
        assert not first._task_future_meta
        assert not first._execution_scopes
        assert first._actor_pool is first._task_pool is None
        if pool is None:
            assert owned_pool.ref_count == 1
            assert runtime.execution_capacity.resource_snapshot()["usage"]["cpu"] == 1

        # The peer was still executing while the first owner drained. Its
        # second block and terminal response prove close did not abort it.
        (tmp_path / "2").touch()
        assert _consume(_next(peer)) == {"x": [12]}
        assert _next(peer) == (ref_bundle.SUBMIT_RESULT_MARKER, 2, None)
        peer.close(kill=True)
    finally:
        release.set()
        (tmp_path / "1").touch()
        (tmp_path / "2").touch()
        first._wait_for_pending_futures(10)
        first.close(kill=True)
        peer.close(kill=True)
        if pool is not None:
            pool.shutdown(kill=True)


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
@pytest.mark.parametrize("cancel_at", ["descriptor", "adopted"])
@pytest.mark.parametrize("streaming", [False, True])
def test_cancelled_response_never_contaminates_the_next_invocation(
    runtime, monkeypatch, tmp_path, backend, cancel_at, streaming
):
    gate = tmp_path / "resume_cancelled_generator"

    def chunks(table):
        value = table.column(0)[0].as_py()
        if value == 1:
            yield pa.table({"x": [111]})
            deadline = time.monotonic() + 15
            while not gate.exists():
                if time.monotonic() > deadline:
                    raise RuntimeError("cancelled generator was not released")
                time.sleep(0.01)
            yield pa.table({"x": [999]})
        else:
            yield pa.table({"x": [222]})

    def produce(table):
        if streaming:
            return chunks(table)
        return pa.table({"x": [111 if table.column(0)[0].as_py() == 1 else 222]})

    class Actor:
        def __call__(self, table):
            return produce(table)

    metrics = WorkerMetrics()
    payload = _payload(
        Actor if backend == "subprocess_actor" else produce,
        execution_backend=backend,
        call_mode="map_batches" if streaming else "map_batches_rows",
        scalar_arg_count=1,
        batch_size=1,
    )
    pool = (
        udf_subprocess.LocalSubprocessActorPool(payload, 1, worker_metrics=metrics)
        if backend == "subprocess_actor"
        else None
    )
    options = {"local_worker_metrics": metrics, **({"local_actor_pool": pool} if pool is not None else {})}
    first, second = [udf_subprocess.UDFExecutor(payload, options) for _ in range(2)]
    received, resume, cancelled, next_sent = [threading.Event() for _ in range(4)]
    observed, pids = [], []
    decode = udf_subprocess._SingleSubprocessExecutor._decode_ref_bundle_result
    adopt = udf_subprocess.make_local_shm_ref_bundle_result_from_descriptor
    send = udf_subprocess._SingleSubprocessExecutor._send_worker_message
    wait_for_cleanup = first._wait_for_pending_futures

    def pause():
        received.set()
        assert resume.wait(10), "cancelled response was not resumed"

    def decode_result(worker, data):
        result = decode(worker, data)
        if not observed:
            observed.append(worker)
            pids.append(worker._proc.pid)
            if cancel_at == "descriptor":
                pause()
        return result

    def adopt_result(*args, **kwargs):
        result = adopt(*args, **kwargs)
        if cancel_at == "adopted" and not received.is_set():
            pause()
        return result

    def wait_after_cancel(timeout):
        cancelled.set()
        return wait_for_cleanup(timeout)

    def detect_next_input(worker, sock, kind, data=b""):
        result = send(worker, sock, kind, data)
        if kind == udf_subprocess._MSG_SUBMIT_REF_BUNDLE and observed and first._closed:
            pids.append(worker._proc.pid)
            next_sent.set()
        return result

    monkeypatch.setattr(udf_subprocess._SingleSubprocessExecutor, "_decode_ref_bundle_result", decode_result)
    monkeypatch.setattr(udf_subprocess, "make_local_shm_ref_bundle_result_from_descriptor", adopt_result)
    monkeypatch.setattr(udf_subprocess._SingleSubprocessExecutor, "_send_worker_message", detect_next_input)
    monkeypatch.setattr(first, "_wait_for_pending_futures", wait_after_cancel)
    try:
        assert first.request_task_admission(8)
        first.submit_with_id(1, pa.table({"x": [1]}))
        assert received.wait(10)
        with ThreadPoolExecutor(1) as closer:
            closing = closer.submit(first.close, kill=False)
            try:
                assert cancelled.wait(5)
            finally:
                resume.set()
            closing.result(timeout=10)
        assert observed[0].is_reusable() is not streaming
        assert second.request_task_admission(8)
        second.submit_with_id(2, pa.table({"x": [2]}))
        assert next_sent.wait(10), "next invocation waited for the cancelled generator"
        assert (pids[0] != pids[1]) is streaming
        gate.touch()
        rows = []
        while True:
            item = _next(second)
            assert item[:2] == (ref_bundle.SUBMIT_RESULT_MARKER, 2)
            assert not isinstance(item[2], BaseException), item[2]
            if len(item) == 3:
                if item[2] is not None:
                    rows.extend(_consume(item)["x"])
                break
            rows.extend(_consume(item)["x"])
        assert rows == [222]
        assert {key: count for key, count in metrics.snapshot().items() if count} == (
            {"cancelled_workers": 1} if streaming else {}
        )
    finally:
        resume.set()
        gate.touch()
        first.close(kill=True)
        second.close(kill=True)
        if pool is not None:
            pool.shutdown(kill=True)


@pytest.mark.parametrize("streaming", [False, True])
def test_worker_measures_then_writes_each_output_once_with_exact_admission(monkeypatch, streaming, pooled_shm_worker):
    from vane.execution import udf_subprocess_worker as worker

    table = pa.table({"blob": [b"x" * 4096]})
    serialized, grants, sent = [], [], []
    new_stream = pa.ipc.new_stream

    def serialize(sink, *args, **kwargs):
        serialized.append((type(sink), len(grants)))
        return new_stream(sink, *args, **kwargs)

    def forbid_staging(*args, **kwargs):
        raise AssertionError("pooled output must not allocate a staging IPC buffer")

    class Executor:
        _payload = {"call_mode": "map_batches"} if streaming else {}

        def iter_submit(self, _table):
            yield table

        def submit(self, _table):
            pass

        def drain_outputs(self):
            return [] if streaming else [table]

    def grant(_sock, *, size, **kwargs):
        grants.append(size)
        return {"grant_id": 9, "allocation": pooled_shm_worker.reserve_write(9, size)}

    monkeypatch.setattr(pa.ipc, "new_stream", serialize)
    monkeypatch.setattr(pa, "BufferOutputStream", forbid_staging)
    monkeypatch.setattr(worker, "_request_output_grant", grant)
    monkeypatch.setattr(worker, "_send_message", lambda _sock, kind, payload=b"": sent.append((kind, payload)))
    monkeypatch.setattr(worker, "_recv_message", lambda _sock: (worker._MSG_STREAM_CONTINUE, b""))
    _, kind, payload = worker._execute_submit(Executor(), table, None, True, sock=None, submit_count=1)
    chunks = [payload for kind, payload in sent if kind == worker._MSG_REF_BUNDLE_CHUNK]
    descriptor = vane_pickle.loads(chunks[0] if streaming else payload)
    try:
        # Measuring IPC size allocates no payload buffer. Only the second
        # stream writes bytes, after the exact storage grant is acquired.
        assert serialized == [(pa.MockOutputStream, 0), (pa.FixedSizeBufferWriter, 1)]
        assert grants == [descriptor["metadata"][0]["ipc_size_bytes"]]
        assert descriptor["grant_id"] == 9
        if streaming:
            assert len(chunks) == 1
            assert [message for message, _ in sent].count(worker._MSG_STREAM_NEXT) == 2
            assert kind == worker._MSG_OK
    finally:
        ref_bundle.release_local_shm_ref_bundle_descriptor(descriptor)


def test_failed_chunk_send_releases_descriptor_and_grant(monkeypatch, pooled_shm_worker):
    from vane.execution import udf_subprocess_worker as worker

    descriptors, released = [], []
    create = worker.make_pooled_shm_descriptor

    def record(*args, **kwargs):
        descriptor = create(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    def fail(*args):
        raise BrokenPipeError("send failed")

    monkeypatch.setattr(worker, "make_pooled_shm_descriptor", record)
    monkeypatch.setattr(
        worker,
        "_request_output_grant",
        lambda *args, **kwargs: {"grant_id": 9, "allocation": pooled_shm_worker.reserve_write(9, kwargs["size"])},
    )
    monkeypatch.setattr(
        worker,
        "_release_output_grant",
        lambda _sock, grant: (released.append(grant), pooled_shm_worker.finish_write(grant)),
    )
    monkeypatch.setattr(worker, "_send_message", fail)
    with pytest.raises(BrokenPipeError, match="send failed"):
        worker._publish_output_block(pa.table({"x": [1]}), None, submit_count=1, input_lease_id=None)
    assert released == [9]
    assert pooled_shm_worker.store.snapshot()["live_allocations"] == 0
    allocation = descriptors[0]["block_refs"][0]["allocation"]
    reused = pooled_shm_worker.reserve_write(10, allocation["size"])
    assert reused["offset"] == allocation["offset"]
    assert reused["generation"] != allocation["generation"]


@pytest.mark.parametrize("call_mode", ["map_batches", "flat_map"])
@pytest.mark.parametrize("empty", [False, True])
def test_interleaved_submits_preserve_ids_empty_results_and_compute_tail(runtime, call_mode, empty):
    def batches(table):
        return table.slice(0, 0) if empty else table

    def rows(row):
        return [] if empty else [row]

    executor = udf_subprocess.UDFExecutor(
        _payload(
            batches if call_mode == "map_batches" else rows,
            call_mode=call_mode,
            output_schema=[{"name": "x", "type": "BIGINT"}],
        )
    )
    try:
        for submit_id in (41, 42):
            assert executor.request_task_admission(0)
            executor.submit_with_id(submit_id, pa.table({"x": [submit_id] * 3}))
        seen = {41: [], 42: []}
        completed = set()
        while len(completed) < 2:
            item = _next(executor)
            submit_id = item[1]
            assert submit_id not in completed
            if len(item) == 3:
                assert item[2] is None
                completed.add(submit_id)
            else:
                seen[submit_id].extend(_consume(item)["x"])
        assert seen == ({41: [], 42: []} if empty else {41: [41] * 3, 42: [42] * 3})
    finally:
        executor.close(kill=True)


@pytest.mark.parametrize("kill", [False, True])
@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
def test_cancelling_a_full_output_queue_releases_all_block_owners(runtime, monkeypatch, kill, backend):
    def expand(_table):
        for _ in range(100):
            yield pa.table({"blob": [b"x" * 32768]})

    class Expand:
        def __call__(self, table):
            return expand(table)

    full = threading.Event()
    reserve = udf_subprocess.LocalStreamAdapter.reserve

    def observed_reserve(stream, *args, **kwargs):
        with stream._condition:
            if stream._outstanding == 2:
                full.set()
        return reserve(stream, *args, **kwargs)

    monkeypatch.setattr(udf_subprocess.LocalStreamAdapter, "reserve", observed_reserve)
    metrics = WorkerMetrics()
    payload = _payload(Expand if backend == "subprocess_actor" else expand, execution_backend=backend)
    pool = (
        udf_subprocess.LocalSubprocessActorPool(payload, 1, worker_metrics=metrics)
        if backend == "subprocess_actor"
        else None
    )
    executor = udf_subprocess.UDFExecutor(
        payload, {"local_worker_metrics": metrics, **({"local_actor_pool": pool} if pool is not None else {})}
    )
    try:
        assert executor.request_task_admission(0)
        executor.submit_with_id(1, pa.table({"x": [1]}))
        assert full.wait(timeout=15), "producer did not block on its bounded output queue"
        executor.close(kill=kill)
        assert metrics.snapshot()["runtime_errors"] == metrics.snapshot()["execution_errors"] == 0
    finally:
        executor.close(kill=True)
        if pool is not None:
            pool.shutdown(kill=True)


def test_native_pipeline_consumes_blocks_before_a_physical_task_completes(runtime, monkeypatch, tmp_path):
    import vane

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    consumed = tmp_path / "consumed"

    def expand(_table):
        yield pa.table({"blob": [b"x" * 32768]})
        deadline = time.monotonic() + 10
        while not consumed.exists():
            if time.monotonic() > deadline:
                raise RuntimeError("native downstream did not consume the first block")
            time.sleep(0.01)
        for _ in range(9):
            yield pa.table({"blob": [b"x" * 32768]})

    def consume(table):
        consumed.touch()
        return pa.table({"size": [len(value) for value in table.column(0).to_pylist()]})

    con = vane.connect()
    try:
        rel = (
            con.sql("select 1 as x")
            .map_batches(
                expand,
                schema={"blob": vane.sqltypes.BLOB},
                batch_size=1,
                output_batch_size=1,
                execution_backend="subprocess_task",
            )
            .map_batches(
                consume,
                schema={"size": vane.sqltypes.BIGINT},
                batch_size=1,
                execution_backend="subprocess_task",
            )
        )
        assert rel.fetchall() == [(32768,)] * 10
    finally:
        con.close()


def test_physical_store_exhaustion_reports_capacity_failure_and_cleans_grants(runtime, monkeypatch):
    monkeypatch.setenv("VANE_LOCAL_SHM_STORE_BYTES", "64k")

    def produce(_table):
        return pa.table({"blob": [b"x" * 75000]})

    executor = udf_subprocess.UDFExecutor(_payload(produce))
    try:
        assert executor.request_task_admission(0)
        executor.submit_with_id(1, pa.table({"x": [1]}))
        result = _next(executor)[2]
        assert isinstance(result, RuntimeError)
        assert "shared-memory output block exceeds arena capacity" in str(result)
        assert "cancelled" not in str(result)
    finally:
        executor.close(kill=True)
