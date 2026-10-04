# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Incremental local output ownership, byte admission and native completion."""

from __future__ import annotations

import time

import pyarrow as pa
import pytest

from vane import pickle as vane_pickle
from vane.execution import ref_bundle, udf_subprocess
from vane.execution.resources import ResourceVector


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


@pytest.mark.parametrize("streaming", [False, True])
def test_worker_serializes_each_output_once_and_admits_exact_published_size(monkeypatch, streaming, pooled_shm_worker):
    from vane.execution import udf_subprocess_worker as worker

    table = pa.table({"blob": [b"x" * 4096]})
    serialized, grants, sent = [], [], []
    new_stream = pa.ipc.new_stream

    def serialize(*args, **kwargs):
        serialized.append(True)
        return new_stream(*args, **kwargs)

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
    monkeypatch.setattr(worker, "_request_output_grant", grant)
    monkeypatch.setattr(worker, "_send_message", lambda _sock, kind, payload: sent.append((kind, payload)))
    _, kind, payload = worker._execute_submit(Executor(), table, None, True, sock=None, submit_count=1)
    descriptor = vane_pickle.loads(sent[0][1] if streaming else payload)
    try:
        assert serialized == [True]
        assert grants == [descriptor["metadata"][0]["ipc_size_bytes"]]
        assert descriptor["grant_id"] == 9
        if streaming:
            assert sent[0][0] == worker._MSG_REF_BUNDLE_CHUNK
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


def test_cancelling_a_full_output_queue_releases_all_block_owners(runtime):
    def expand(_table):
        for _ in range(100):
            yield pa.table({"blob": [b"x" * 32768]})

    executor = udf_subprocess.UDFExecutor(_payload(expand))
    try:
        assert executor.request_task_admission(0)
        executor.submit_with_id(1, pa.table({"x": [1]}))
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if ref_bundle.local_shm_ref_budget_snapshot()["waiting_output_grants"]:
                break
            time.sleep(0.01)
        else:
            pytest.fail("producer did not block on its bounded output queue")
    finally:
        executor.close(kill=True)


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
