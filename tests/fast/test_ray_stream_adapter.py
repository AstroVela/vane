# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import gc
import weakref
from types import SimpleNamespace

import pytest

from vane.execution.ray_stream_adapter import (
    RayStreamAdapter,
    TaskLeaseObjectRefGenerator,
    validate_ray_stream_contract,
)
from vane.execution.udf_task_admission import TaskAdmission


class _Ref:
    def __init__(self, value, *, nil: bool = False):
        self.value = value
        self._nil = nil
        self.future_result_calls = []

    def is_nil(self):
        return self._nil

    def future(self):
        ref = self

        class _Future:
            def result(self, timeout=None):
                ref.future_result_calls.append(timeout)
                return ref.value

        return _Future()


class _Generator:
    def __init__(self, values):
        self.refs = [_Ref(value) for value in values]
        self.completion = _Ref(None)
        self.read_calls = []
        self.deleted_streams = []
        self.worker = SimpleNamespace(
            core_worker=SimpleNamespace(
                is_object_ref_stream_finished=lambda _ref: not self.refs,
                try_read_next_object_ref_stream=self._read_next,
                async_delete_object_ref_stream=self.deleted_streams.append,
            )
        )

    def completed(self):
        return self.completion

    async def __anext__(self):
        if not self.refs:
            raise StopAsyncIteration
        self.read_calls.append("async")
        return self.refs.pop(0)

    def next_ready(self):
        return bool(self.refs)

    def _read_next(self, _generator_ref):
        self.read_calls.append("core")
        if not self.refs:
            raise AssertionError("adapter read past the non-blocking stream boundary")
        return self.refs.pop(0)

    def is_finished(self):
        raise AssertionError("ObjectRefGenerator.is_finished() performs blocking ray.get")


class _RemoteMethod:
    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.fn(*args, **kwargs)


class _FakeRay:
    __version__ = "2.55.1"
    ObjectRefGenerator = _Generator

    def __init__(self):
        self.get_calls = []
        self.cancel_calls = []

    def get(self, ref):
        self.get_calls.append(ref)
        return ref.value

    def cancel(self, ref, **kwargs):
        self.cancel_calls.append((ref, kwargs))


def _driver(grant):
    return SimpleNamespace(
        acquire_query_task_lease=_RemoteMethod(lambda _request: _Ref(grant)),
        mark_query_task_lease_submitted=_RemoteMethod(lambda *_args: _Ref({"submitted": True})),
        release_query_task_lease=_RemoteMethod(lambda *_args: _Ref({"released": True})),
        cancel_query_task_lease_request=_RemoteMethod(lambda *_args, **_kwargs: _Ref({"cancelled": True})),
    )


def _lease():
    return {
        "lease_id": "lease-1",
        "query_id": "q1",
        "stage_id": "stage:q1:node:1:udf",
        "task_id": "task-1",
        "attempt_id": "attempt-1",
        "resources": {"cpu": 1.0, "gpu": 0.0, "heap_bytes": 1, "object_store_bytes": 0},
        "output_window_bytes": 256,
        "liveness": False,
        "allocation_generation": 1,
    }


def test_contract_accepts_unlisted_ray_version_when_capabilities_are_present():
    ray_module = SimpleNamespace(__version__="9.0.0", ObjectRefGenerator=_Generator)

    validate_ray_stream_contract(ray_module)


def test_contract_rejects_missing_generator_capability_with_version_context():
    class _IncompleteGenerator:
        async def __anext__(self):
            raise StopAsyncIteration

    ray_module = SimpleNamespace(__version__="2.56.0", ObjectRefGenerator=_IncompleteGenerator)

    with pytest.raises(RuntimeError, match="Ray '2.56.0' ObjectRefGenerator contract is missing: completed"):
        validate_ray_stream_contract(ray_module)


def test_stream_blocks_never_exceed_duckdb_vector_size():
    pa = pytest.importorskip("pyarrow")
    from vane.execution.udf_ray_stream_protocol import (
        DUCKDB_STANDARD_VECTOR_SIZE,
        iter_bounded_stream_blocks,
    )

    target_bytes = 1024**3
    payload = {
        "query_id": "q1",
        "stage_id": "stage:q1:node:1:udf",
        "task_lease_id": "lease-1",
        "attempt_id": "attempt-1",
        "udf_output_target_max_bytes": target_bytes,
        "output_window_bytes": target_bytes * 2,
    }
    table = pa.table({"value": range(DUCKDB_STANDARD_VECTOR_SIZE + 37)})

    blocks = list(iter_bounded_stream_blocks(table, payload))

    assert [block.num_rows for block in blocks] == [DUCKDB_STANDARD_VECTOR_SIZE, 37]


def test_adapter_advances_generator_asynchronously_and_never_gets_data_ref():
    fake_ray = _FakeRay()
    block_ref = _Ref("large-block")
    metadata_ref = _Ref({"size_bytes": 10})
    generator = _Generator([])
    generator.refs = [block_ref, metadata_ref]
    adapter = RayStreamAdapter(generator, ray_module=fake_ray)

    async def read_pair():
        return await adapter.read_next_ref_async(), await adapter.read_next_ref_async()

    assert asyncio.run(read_pair()) == (block_ref, metadata_ref)
    assert generator.read_calls == ["async", "async"]
    assert block_ref.future_result_calls == []
    assert metadata_ref.future_result_calls == []

    adapter.mark_drained()
    with pytest.raises(StopAsyncIteration):
        asyncio.run(adapter.read_next_ref_async())


def test_adapter_uses_public_completion_ref_for_stream_lifecycle():
    generator = _Generator([])
    adapter = RayStreamAdapter(generator, ray_module=_FakeRay())

    assert adapter.completion_ref is generator.completion
    assert adapter.stream_finished() is True
    assert adapter.is_terminal_ref(generator.completion) is True

    adapter.retire()

    assert generator.deleted_streams == [generator.completion]


def _admission(driver, request_id="request-1"):
    return TaskAdmission(
        driver=driver,
        request_id=request_id,
        retained_input_bytes=0,
        lease=_lease(),
    )


def test_task_lease_stream_submits_immediately_from_pregranted_admission():
    fake_ray = _FakeRay()
    lease = _lease()
    driver = _driver({"granted": True, "lease": lease})
    submitted = []
    generator = _Generator(["block", "metadata"])
    source = TaskLeaseObjectRefGenerator(
        admission=_admission(driver),
        submitter=lambda granted: submitted.append(granted) or generator,
        ray_module=fake_ray,
    )
    adapter = RayStreamAdapter(source, ray_module=fake_ray)

    assert submitted == [lease]
    assert driver.mark_query_task_lease_submitted.calls[0][0] == ("request-1", "lease-1")

    adapter.release_task()
    adapter.release_task()
    assert len(driver.release_query_task_lease.calls) == 1


def test_task_lease_stream_abandons_pregranted_lease_when_submitter_fails():
    fake_ray = _FakeRay()
    driver = _driver({"granted": True, "lease": _lease()})
    request_id = "request-submit-failed"
    admission = TaskAdmission(
        driver=driver,
        request_id=request_id,
        retained_input_bytes=0,
        lease=_lease(),
        _release_callback=lambda: driver.cancel_query_task_lease_request.remote(
            request_id,
            submitted=False,
        ),
    )

    with pytest.raises(RuntimeError, match="submit failed"):
        TaskLeaseObjectRefGenerator(
            admission=admission,
            submitter=lambda _lease: (_ for _ in ()).throw(RuntimeError("submit failed")),
            ray_module=fake_ray,
        )

    assert driver.cancel_query_task_lease_request.calls == [((request_id,), {"submitted": False})]


def test_successful_submission_releases_captured_input_ownership_immediately():
    fake_ray = _FakeRay()
    lease = _lease()
    driver = _driver({"granted": True, "lease": lease})
    generator = _Generator([])

    class _LargeInput:
        pass

    def make_source():
        large_input = _LargeInput()
        large_input_ref = weakref.ref(large_input)
        source = TaskLeaseObjectRefGenerator(
            admission=_admission(driver, "request-transfer"),
            submitter=lambda _granted: (large_input, generator)[1],
            ray_module=fake_ray,
        )
        return source, large_input_ref

    source, large_input_ref = make_source()
    RayStreamAdapter(source, ray_module=fake_ray)
    gc.collect()

    assert large_input_ref() is None


def test_cancelling_started_stream_cancels_remote_work_and_releases_lease():
    fake_ray = _FakeRay()
    driver = _driver({"granted": True, "lease": _lease()})
    generator = _Generator([])
    source = TaskLeaseObjectRefGenerator(
        admission=_admission(driver, "request-cancel"),
        submitter=lambda _lease: generator,
        ray_module=fake_ray,
    )
    adapter = RayStreamAdapter(source, ray_module=fake_ray)

    adapter.cancel()
    adapter.cancel()

    assert fake_ray.cancel_calls == [(generator, {"force": True, "recursive": True})]
    assert len(driver.cancel_query_task_lease_request.calls) == 1
    assert driver.cancel_query_task_lease_request.calls[0][1] == {"submitted": True}


@pytest.mark.parametrize("terminal_action", ["retire", "cancel"])
def test_terminal_stream_cleanup_disarms_ray_generator_destructor(terminal_action):
    fake_ray = _FakeRay()
    generator = _Generator([])
    adapter = RayStreamAdapter(generator, ray_module=fake_ray)

    getattr(adapter, terminal_action)()
    getattr(adapter, terminal_action)()

    assert generator.deleted_streams == [generator.completion]
    assert generator.worker is None
