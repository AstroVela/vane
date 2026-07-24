# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import gc
import threading
import time
import weakref
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

import duckdb.execution.udf_stream_result_collector as collector_module
from duckdb.execution.ray_stream_adapter import RayStreamAdapter, TaskLeaseObjectRefGenerator
from duckdb.execution.udf_stream_result_collector import (
    AsyncResultCollector,
    _StreamRecord,
)
from duckdb.execution.udf_task_admission import TaskAdmission


class _Ref:
    def __init__(self, value=None, *, ready=True, is_block=False):
        self.value = value
        self._ready = False
        self.is_block = is_block
        self.future_result_calls = []
        self._future = Future()
        self._ready_callbacks = []
        self.ready = ready

    @property
    def ready(self):
        return self._ready

    @ready.setter
    def ready(self, value):
        became_ready = bool(value) and not self._ready
        self._ready = bool(value)
        if not became_ready:
            return
        if not self._future.done():
            self._future.set_result(self.value)
        callbacks, self._ready_callbacks = self._ready_callbacks, []
        for callback in callbacks:
            callback()

    def add_ready_callback(self, callback):
        if self.ready:
            callback()
        else:
            self._ready_callbacks.append(callback)

    def is_nil(self):
        return False

    def future(self):
        if self.is_block:
            raise AssertionError("collector materialized a large block ObjectRef")
        future = self._future
        original_result = future.result

        def tracked_result(timeout=None):
            self.future_result_calls.append(timeout)
            return original_result(timeout=timeout)

        future.result = tracked_result
        return future


class _Generator:
    def __init__(self, refs, *, completed=True):
        self.refs = list(refs)
        self.completion_ref = _Ref(None, ready=completed)
        self.read_count = 0
        self.deleted_streams = []
        self.worker = SimpleNamespace(
            core_worker=SimpleNamespace(
                is_object_ref_stream_finished=lambda _ref: not self.refs and self.completion_ref.ready,
                try_read_next_object_ref_stream=self._read_next,
                async_delete_object_ref_stream=self.deleted_streams.append,
            )
        )

    def completed(self):
        return self.completion_ref

    async def __anext__(self):
        if self.refs:
            ref = self.refs[0]
            if not ref.ready:
                loop = asyncio.get_running_loop()
                ready = loop.create_future()

                def notify_ready():
                    loop.call_soon_threadsafe(lambda: None if ready.done() else ready.set_result(None))

                ref.add_ready_callback(notify_ready)
                await ready
            self.read_count += 1
            return self.refs.pop(0)
        if not self.completion_ref.ready:
            loop = asyncio.get_running_loop()
            completed = loop.create_future()

            def notify_completed():
                loop.call_soon_threadsafe(lambda: None if completed.done() else completed.set_result(None))

            self.completion_ref.add_ready_callback(notify_completed)
            await completed
        raise StopAsyncIteration

    def next_ready(self):
        return bool(self.refs and self.refs[0].ready)

    def _read_next(self, _generator_ref):
        if not self.next_ready():
            raise AssertionError("collector attempted a blocking generator read")
        self.read_count += 1
        return self.refs.pop(0)

    def is_finished(self):
        raise AssertionError("ObjectRefGenerator.is_finished() performs blocking ray.get")


class _FakeRay:
    __version__ = "2.55.1"
    ObjectRefGenerator = _Generator

    def __init__(self):
        self.get_calls = []
        self.cancel_calls = []
        self._cv = threading.Condition()

    def get(self, ref, timeout=None):
        assert isinstance(ref, _Ref)
        if ref.is_block:
            raise AssertionError("collector materialized a large block ObjectRef")
        if not ref.ready:
            raise TimeoutError("control ref is not ready")
        self.get_calls.append((ref, timeout))
        return ref.value

    def wait(self, waitables, *, num_returns, timeout, fetch_local):
        assert fetch_local is False

        def _ready(value):
            if isinstance(value, _Generator):
                return value.next_ready()
            return bool(value.ready)

        deadline = time.monotonic() + float(timeout)
        while True:
            ready = [value for value in waitables if _ready(value)]
            if ready or time.monotonic() >= deadline:
                return ready[:num_returns], [value for value in waitables if value not in ready]
            with self._cv:
                self._cv.wait(timeout=min(0.005, max(0.0, deadline - time.monotonic())))

    def cancel(self, ref, **kwargs):
        self.cancel_calls.append((ref, kwargs))

    def make_ready(self, ref):
        ref.ready = True
        with self._cv:
            self._cv.notify_all()


class _BlockingCancelRay(_FakeRay):
    def __init__(self):
        super().__init__()
        self.cancel_started = threading.Event()
        self.allow_cancel = threading.Event()

    def cancel(self, ref, **kwargs):
        self.cancel_started.set()
        self.allow_cancel.wait(timeout=2.0)
        super().cancel(ref, **kwargs)


class _RemoteMethod:
    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _Ref(self.fn(*args, **kwargs))


class _Driver:
    def __init__(self):
        self.next_task = 0
        self.next_output = 0
        self.acquire_query_task_lease = _RemoteMethod(self._acquire_task)
        self.mark_query_task_lease_submitted = _RemoteMethod(lambda *_args: {"submitted": True})
        self.release_query_task_lease = _RemoteMethod(lambda *_args: {"released": True})
        self.cancel_query_task_lease_request = _RemoteMethod(lambda *_args, **_kwargs: {"cancelled": True})
        self.acquire_query_output_block_lease = _RemoteMethod(self._acquire_output)
        self.handoff_query_output_block_lease = _RemoteMethod(lambda *_args: {"handed_off": True})
        self.release_query_output_block_lease = _RemoteMethod(lambda *_args: {"released": True})
        self.cancel_query_output_block_lease_request = _RemoteMethod(lambda *_args: {"cancelled": True})

    def _acquire_task(self, request):
        self.next_task += 1
        return {
            "granted": True,
            "lease": {
                "lease_id": f"task-lease-{self.next_task}",
                "query_id": request["query_id"],
                "stage_id": request["stage_id"],
                "task_id": request["task_id"],
                "attempt_id": request["attempt_id"],
                "resources": {
                    "cpu": 1.0,
                    "gpu": 0.0,
                    "heap_bytes": 1024,
                    "object_store_bytes": 0,
                },
                "output_window_bytes": 256,
                "liveness": False,
                "allocation_generation": 1,
            },
            "blocked_reason": "",
            "fatal": False,
            "liveness": False,
        }

    def _acquire_output(self, request):
        self.next_output += 1
        return {
            "granted": True,
            "lease": {
                "lease_id": f"output-lease-{self.next_output}",
                "query_id": request["query_id"],
                "producer_stage_id": request["producer_stage_id"],
                "task_lease_id": request["task_lease_id"],
                "attempt_id": request["attempt_id"],
                "block_id": request["block_id"],
                "size_bytes": request["size_bytes"],
                "state": "stage_queue",
                "liveness": False,
                "allocation_generation": 1,
            },
            "blocked_reason": "",
            "fatal": False,
            "liveness": False,
        }


def _metadata(lease, *, index=0, size_bytes=64, rows=1):
    return {
        "protocol_version": 1,
        "query_id": lease["query_id"],
        "producer_stage_id": lease["stage_id"],
        "task_lease_id": lease["lease_id"],
        "attempt_id": lease["attempt_id"],
        "block_id": f"block:{lease['lease_id']}:{index}",
        "size_bytes": size_bytes,
        "num_rows": rows,
        "names": ["value"],
    }


def _source(fake_ray, driver, *, request_id, submitter):
    request = {
        "request_id": request_id,
        "query_id": "q1",
        "stage_id": "stage:q1:node:1:udf",
        "task_id": f"task:{request_id}",
        "attempt_id": f"attempt:{request_id}",
        "retained_input_bytes": 0,
    }
    lease = driver._acquire_task(request)["lease"]
    return TaskLeaseObjectRefGenerator(
        admission=TaskAdmission(
            driver=driver,
            request_id=request_id,
            retained_input_bytes=0,
            lease=lease,
        ),
        submitter=submitter,
        ray_module=fake_ray,
    )


def _drain_until(collector, capacities, predicate=lambda values: bool(values), timeout=3.0):
    deadline = time.monotonic() + timeout
    collected = []
    while time.monotonic() < deadline:
        collected.extend(collector.drain_results(capacities))
        if predicate(collected):
            return collected
        time.sleep(0.005)
    return collected


def test_collector_requires_task_lease_stream_and_has_no_raw_generator_fallback():
    fake_ray = _FakeRay()
    collector = AsyncResultCollector(ray_module=fake_ray)
    try:
        with pytest.raises(TypeError, match="must return TaskLeaseObjectRefGenerator"):
            collector.track_generator_ref(1, 1, _Generator([]))
    finally:
        collector.shutdown()


def test_zero_capacity_does_not_consume_block_or_metadata_objects():
    fake_ray = _FakeRay()
    driver = _Driver()
    holder = {}

    def submitter(lease):
        generator = _Generator(
            [
                _Ref("large", is_block=True),
                _Ref(_metadata(lease)),
            ]
        )
        holder["generator"] = generator
        return generator

    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        1,
        10,
        _source(fake_ray, driver, request_id="zero-capacity", submitter=submitter),
    )
    try:
        time.sleep(0.05)
        assert collector.drain_results({1: {"rows": 0, "bytes": 0, "item_bytes": 0}}) == []
        time.sleep(0.05)
        assert holder["generator"].read_count == 0
        assert driver.acquire_query_output_block_lease.calls == []
    finally:
        collector.shutdown()


def test_direct_block_pair_is_leased_and_large_block_is_never_fetched():
    fake_ray = _FakeRay()
    driver = _Driver()
    block_ref = _Ref("large-block", is_block=True)

    def submitter(lease):
        return _Generator([block_ref, _Ref(_metadata(lease, size_bytes=64))])

    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        2,
        20,
        _source(fake_ray, driver, request_id="direct-pair", submitter=submitter),
    )
    try:
        events = _drain_until(
            collector,
            {2: {"rows": 1, "bytes": 128, "item_bytes": 128}},
            predicate=lambda values: any(item[2] == "complete" for item in values),
        )
        assert [item[2] for item in events] == ["data", "complete"]
        data = events[0]
        assert len(data) == 6
        assert data[3][1] == [block_ref]
        assert data[3][2][0]["output_block_lease_id"] == data[5]
        assert block_ref.future_result_calls == []
        assert len(driver.acquire_query_output_block_lease.calls) == 1

        assert collector.handoff_output_block_lease(data[4], data[5]) is True
        assert collector.handoff_output_block_lease(data[4], data[5]) is False
        assert len(driver.handoff_query_output_block_lease.calls) == 1
        assert driver.release_query_output_block_lease.calls == []

        assert collector.release_output_block_lease(data[4], data[5]) is True
        assert collector.release_output_block_lease(data[4], data[5]) is False
        assert len(driver.release_query_output_block_lease.calls) == 1
        assert len(driver.release_query_task_lease.calls) == 1
    finally:
        collector.shutdown()


def test_metadata_transition_is_atomic_with_concurrent_capacity_refresh(
    monkeypatch,
):
    fake_ray = _FakeRay()
    driver = _Driver()
    metadata_ready = threading.Event()
    allow_transition = threading.Event()

    def pause_ready_metadata(event, _record, **_fields):
        if event == "ready_metadata" and not metadata_ready.is_set():
            metadata_ready.set()
            allow_transition.wait(timeout=2.0)

    monkeypatch.setattr(
        collector_module,
        "_collector_debug_log",
        pause_ready_metadata,
    )

    def submitter(lease):
        return _Generator(
            [
                _Ref("block", is_block=True),
                _Ref(_metadata(lease)),
            ]
        )

    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        2,
        23,
        _source(
            fake_ray,
            driver,
            request_id="atomic-metadata-transition",
            submitter=submitter,
        ),
    )
    capacity = {2: {"rows": 1, "bytes": 128, "item_bytes": 128}}
    try:
        collector.drain_results(capacity)
        assert metadata_ready.wait(timeout=2.0)

        # Capacity updates run on the C++ dispatcher thread. They may race a
        # ready callback on the collector loop, but must not schedule a second
        # wait for the same metadata ObjectRef while its state transition is
        # still being applied.
        assert collector.drain_results(capacity) == []
        allow_transition.set()
        events = _drain_until(
            collector,
            capacity,
            predicate=lambda values: any(item[2] in {"complete", "error"} for item in values),
        )

        assert [item[2] for item in events] == ["data", "complete"]
    finally:
        allow_transition.set()
        collector.shutdown()


def test_inflight_block_keeps_item_capacity_from_read_admission():
    fake_ray = _FakeRay()
    driver = _Driver()
    holder = {}

    def submitter(lease):
        metadata_ref = _Ref(_metadata(lease, size_bytes=128), ready=False)
        generator = _Generator(
            [
                _Ref("admitted-block", is_block=True),
                metadata_ref,
            ]
        )
        holder["metadata_ref"] = metadata_ref
        holder["generator"] = generator
        return generator

    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        2,
        24,
        _source(
            fake_ray,
            driver,
            request_id="stable-item-capacity",
            submitter=submitter,
        ),
    )
    admitted_capacity = {2: {"rows": 1, "bytes": 128, "item_bytes": 128}}
    zero_capacity = {2: {"rows": 0, "bytes": 0, "item_bytes": 0}}
    try:
        collector.drain_results(admitted_capacity)
        deadline = time.monotonic() + 2.0
        while holder["generator"].read_count < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert holder["generator"].read_count == 1

        # The pair is already in flight. A temporary downstream backpressure
        # update may prevent delivery, but it cannot retroactively revoke the
        # item-size admission under which the block was consumed.
        assert collector.drain_results(zero_capacity) == []
        fake_ray.make_ready(holder["metadata_ref"])
        zero_capacity_events = []
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            zero_capacity_events.extend(collector.drain_results(zero_capacity))
            if zero_capacity_events or driver.acquire_query_output_block_lease.calls:
                break
            time.sleep(0.005)

        assert zero_capacity_events == []
        assert len(driver.acquire_query_output_block_lease.calls) == 1
        events = _drain_until(
            collector,
            admitted_capacity,
            predicate=lambda values: any(item[2] in {"complete", "error"} for item in values),
        )
        assert [item[2] for item in events] == ["data", "complete"]
    finally:
        collector.shutdown()


def test_terminal_stream_is_retired_without_waiting_for_collector_shutdown():
    fake_ray = _FakeRay()
    driver = _Driver()
    generator_holder = {}

    def make_source():
        def submitter(lease):
            generator = _Generator(
                [
                    _Ref("large-block", is_block=True),
                    _Ref(_metadata(lease, size_bytes=64)),
                ]
            )
            generator_holder["generator"] = generator
            return generator

        source = _source(
            fake_ray,
            driver,
            request_id="deterministic-retirement",
            submitter=submitter,
        )
        return source, weakref.ref(source)

    source, source_ref = make_source()
    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(2, 22, source)
    del source
    try:
        events = _drain_until(
            collector,
            {2: {"rows": 1, "bytes": 128, "item_bytes": 128}},
            predicate=lambda values: any(item[2] == "complete" for item in values),
        )
        data = next(item for item in events if item[2] == "data")
        assert collector.release_output_block_lease(data[4], data[5]) is True
        del data
        del events

        deadline = time.monotonic() + 1.0
        while source_ref() is not None and time.monotonic() < deadline:
            gc.collect()
            time.sleep(0.005)

        assert collector._records == {}
        assert source_ref() is None
        generator = generator_holder["generator"]
        assert generator.deleted_streams == [generator.completion_ref]
    finally:
        collector.shutdown()


def test_completion_ready_before_final_metadata_does_not_fail_valid_pair():
    fake_ray = _FakeRay()
    driver = _Driver()
    block_ref = _Ref("final-block", is_block=True)
    holder = {}

    def submitter(lease):
        metadata_ref = _Ref(_metadata(lease), ready=False)
        generator = _Generator(
            [block_ref, metadata_ref],
            completed=False,
        )
        holder["metadata_ref"] = metadata_ref
        holder["generator"] = generator
        return generator

    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        2,
        21,
        _source(
            fake_ray,
            driver,
            request_id="completion-before-final-metadata",
            submitter=submitter,
        ),
    )
    capacity = {2: {"rows": 1, "bytes": 128, "item_bytes": 128}}
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            collector.drain_results(capacity)
            generator = holder.get("generator")
            if generator is not None and generator.read_count == 1:
                break
            time.sleep(0.005)
        else:
            pytest.fail("collector did not consume the final block")

        # Ray may publish task completion before the client drains every
        # already-produced stream item.  Make completion and the final
        # metadata ready in the same scheduler turn to exercise that race.
        with fake_ray._cv:
            holder["metadata_ref"].ready = True
            holder["generator"].completion_ref.ready = True
            fake_ray._cv.notify_all()

        events = _drain_until(
            collector,
            capacity,
            predicate=lambda values: any(item[2] in {"complete", "error"} for item in values),
        )

        assert [item[2] for item in events] == ["data", "complete"]
        assert events[0][3][1] == [block_ref]
    finally:
        collector.shutdown()


def test_stale_record_releases_completed_output_lease_with_exact_rpc_contract():
    fake_ray = _FakeRay()
    driver = _Driver()
    source = _source(
        fake_ray,
        driver,
        request_id="stale-record",
        submitter=lambda _lease: _Generator([], completed=False),
    )
    adapter = RayStreamAdapter(source, ray_module=fake_ray)
    lease = adapter.task_lease
    assert lease is not None
    metadata = _metadata(lease)
    output_request = {
        "query_id": metadata["query_id"],
        "producer_stage_id": metadata["producer_stage_id"],
        "task_lease_id": metadata["task_lease_id"],
        "attempt_id": metadata["attempt_id"],
        "block_id": metadata["block_id"],
        "size_bytes": metadata["size_bytes"],
    }
    record = _StreamRecord(
        slot_id=2,
        submit_id=20,
        adapter=adapter,
        sequence=0,
        phase="metadata",
        block_ref=_Ref("large-block", is_block=True),
        metadata=metadata,
        output_request_id=f"output-request:{metadata['block_id']}",
        output_lease_ref=_Ref(driver._acquire_output(output_request)),
    )
    collector = AsyncResultCollector(ray_module=fake_ray)
    original_release = driver.release_query_output_block_lease

    class _LockCheckingRelease:
        @property
        def calls(self):
            return original_release.calls

        def remote(self, *args, **kwargs):
            assert not collector._cv._is_owned()
            return original_release.remote(*args, **kwargs)

    driver.release_query_output_block_lease = _LockCheckingRelease()
    try:
        collector._finish_output_lease(record, record.output_lease_ref.value)

        assert driver.release_query_output_block_lease.calls == [
            ((record.output_request_id, "output-lease-1"), {}),
        ]
    finally:
        collector.shutdown()
        adapter.cancel()


def test_one_slow_stream_cannot_block_a_ready_stream():
    fake_ray = _FakeRay()
    driver = _Driver()
    slow_block = _Ref("slow", ready=False, is_block=True)

    def slow_submitter(lease):
        return _Generator([slow_block, _Ref(_metadata(lease))], completed=False)

    fast_block = _Ref("fast", is_block=True)

    def fast_submitter(lease):
        return _Generator([fast_block, _Ref(_metadata(lease))])

    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        3,
        30,
        _source(fake_ray, driver, request_id="slow", submitter=slow_submitter),
    )
    collector.track_generator_ref(
        3,
        31,
        _source(fake_ray, driver, request_id="fast", submitter=fast_submitter),
    )
    try:
        events = _drain_until(
            collector,
            {3: {"rows": 2, "bytes": 256, "item_bytes": 128}},
            predicate=lambda values: any(item[1] == 31 and item[2] == "data" for item in values),
        )
        fast_data = next(item for item in events if item[1] == 31 and item[2] == "data")
        assert fast_data[3][1] == [fast_block]
        assert all(not (item[1] == 30 and item[2] == "data") for item in events)
    finally:
        collector.cancel_slot(3)
        collector.shutdown()


def test_empty_stream_completion_progresses_with_zero_data_capacity():
    fake_ray = _FakeRay()
    driver = _Driver()
    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        4,
        40,
        _source(
            fake_ray,
            driver,
            request_id="empty",
            submitter=lambda _lease: _Generator([]),
        ),
    )
    try:
        events = _drain_until(
            collector,
            {4: {"rows": 0, "bytes": 0, "item_bytes": 0}},
            predicate=lambda values: bool(values),
        )
        assert events == [(4, 40, "complete", None)]
        assert len(driver.release_query_task_lease.calls) == 1
    finally:
        collector.shutdown()


def test_generator_terminating_mid_pair_fails_without_fetching_block():
    fake_ray = _FakeRay()
    driver = _Driver()
    orphan_block = _Ref("remote-error-or-orphan-block", is_block=True)

    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        9,
        90,
        _source(
            fake_ray,
            driver,
            request_id="incomplete-pair",
            submitter=lambda _lease: _Generator([orphan_block]),
        ),
    )
    try:
        events = _drain_until(
            collector,
            {9: {"rows": 1, "bytes": 128, "item_bytes": 128}},
            predicate=lambda values: any(item[2] == "error" for item in values),
        )
        assert [item[2] for item in events] == ["error"]
        assert "terminated after a block without its metadata" in events[0][3]
        assert orphan_block.future_result_calls == []
        assert driver.acquire_query_output_block_lease.calls == []
    finally:
        collector.shutdown()


def test_failed_completion_retires_without_cancelling_terminal_remote_work():
    fake_ray = _FakeRay()
    driver = _Driver()
    generator = _Generator([], completed=False)
    adapter = RayStreamAdapter(
        _source(
            fake_ray,
            driver,
            request_id="failed-completion",
            submitter=lambda _lease: generator,
        ),
        ray_module=fake_ray,
    )
    record = _StreamRecord(
        slot_id=10,
        submit_id=100,
        adapter=adapter,
        sequence=0,
    )
    completion_future = Future()
    completion_future.set_exception(RuntimeError("planned completion failure"))
    record.completion_future = completion_future
    collector = AsyncResultCollector(ray_module=fake_ray)
    collector._records[(record.slot_id, record.submit_id)] = record

    try:
        collector._complete_producer_wait(record, completion_future)

        assert collector.drain_results({10: {"rows": 1}}) == [
            (
                10,
                100,
                "error",
                "RuntimeError: planned completion failure",
            )
        ]
        assert fake_ray.cancel_calls == []
        assert len(driver.cancel_query_task_lease_request.calls) == 1
        assert generator.deleted_streams == [generator.completion_ref]
    finally:
        collector.shutdown()


def test_explicit_remote_error_pair_preserves_cause_without_output_lease():
    from duckdb.execution.udf_ray_stream_protocol import make_stream_error_pair

    fake_ray = _FakeRay()
    driver = _Driver()
    holder = {}

    def submitter(lease):
        payload = {
            "query_id": lease["query_id"],
            "stage_id": lease["stage_id"],
            "task_lease_id": lease["lease_id"],
            "attempt_id": lease["attempt_id"],
        }
        block, metadata = make_stream_error_pair(
            payload,
            RuntimeError("planned remote failure"),
        )
        block_ref = _Ref(block, is_block=True)
        holder["block_ref"] = block_ref
        generator = _Generator([block_ref, _Ref(metadata)])
        holder["generator"] = generator
        return generator

    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        10,
        100,
        _source(fake_ray, driver, request_id="explicit-error", submitter=submitter),
    )
    try:
        events = _drain_until(
            collector,
            {10: {"rows": 1, "bytes": 128, "item_bytes": 128}},
            predicate=lambda values: any(item[2] == "error" for item in values),
        )
        assert [item[2] for item in events] == ["error"]
        assert "RuntimeError: planned remote failure" in events[0][3]
        assert driver.acquire_query_output_block_lease.calls == []
        assert holder["block_ref"].future_result_calls == []
        assert fake_ray.cancel_calls == []
        assert len(driver.cancel_query_task_lease_request.calls) == 1
        assert holder["generator"].deleted_streams == [holder["generator"].completion_ref]
    finally:
        collector.shutdown()


def test_malformed_metadata_fails_only_its_stream_without_output_admission():
    fake_ray = _FakeRay()
    driver = _Driver()

    def submitter(_lease):
        return _Generator([_Ref("block", is_block=True), _Ref({"size_bytes": 1})])

    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        5,
        50,
        _source(fake_ray, driver, request_id="bad-metadata", submitter=submitter),
    )
    try:
        events = _drain_until(
            collector,
            {5: {"rows": 1, "bytes": 128, "item_bytes": 128}},
            predicate=lambda values: any(item[2] == "error" for item in values),
        )
        assert len(events) == 1
        assert events[0][2] == "error"
        assert "invalid Ray UDF stream metadata" in events[0][3]
        assert driver.acquire_query_output_block_lease.calls == []
        assert len(driver.cancel_query_task_lease_request.calls) == 1
        assert fake_ray.cancel_calls[-1][1] == {"force": True, "recursive": True}
    finally:
        collector.shutdown()


def test_stream_error_wakes_dispatcher_before_generator_cancellation():
    fake_ray = _BlockingCancelRay()
    driver = _Driver()
    wakeup = threading.Event()

    def submitter(_lease):
        return _Generator([_Ref("block", is_block=True), _Ref({"size_bytes": 1})])

    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.set_wakeup_callback(wakeup.set)
    collector.track_generator_ref(
        8,
        80,
        _source(fake_ray, driver, request_id="blocking-cancel", submitter=submitter),
    )
    try:
        deadline = time.monotonic() + 2.0
        while not driver.mark_query_task_lease_submitted.calls and time.monotonic() < deadline:
            time.sleep(0.005)
        assert driver.mark_query_task_lease_submitted.calls
        wakeup.clear()

        collector.drain_results({8: {"rows": 1, "bytes": 128, "item_bytes": 128}})

        assert fake_ray.cancel_started.wait(timeout=2.0)
        assert wakeup.is_set(), "terminal error was not published before ray.cancel"
        fake_ray.allow_cancel.set()
        events = _drain_until(
            collector,
            {8: {"rows": 1, "bytes": 128, "item_bytes": 128}},
            predicate=lambda values: any(item[2] == "error" for item in values),
        )
        assert [item[2] for item in events] == ["error"]
    finally:
        fake_ray.allow_cancel.set()
        collector.shutdown()


def test_block_larger_than_declared_item_capacity_fails_instead_of_stalling_queue():
    fake_ray = _FakeRay()
    driver = _Driver()

    def submitter(lease):
        return _Generator(
            [
                _Ref("oversized-block", is_block=True),
                _Ref(_metadata(lease, size_bytes=129)),
            ]
        )

    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        7,
        70,
        _source(fake_ray, driver, request_id="oversized", submitter=submitter),
    )
    try:
        events = _drain_until(
            collector,
            {7: {"rows": 1, "bytes": 128, "item_bytes": 128}},
            predicate=lambda values: any(item[2] == "error" for item in values),
        )
        assert len(events) == 1
        assert events[0][2] == "error"
        assert "exceeds downstream item capacity" in events[0][3]
        assert driver.acquire_query_output_block_lease.calls == []
        assert len(driver.cancel_query_task_lease_request.calls) == 1
    finally:
        collector.shutdown()


def test_slot_cancellation_recursively_cancels_stream_and_releases_output_lease():
    fake_ray = _FakeRay()
    driver = _Driver()

    def submitter(lease):
        return _Generator(
            [_Ref("block", is_block=True), _Ref(_metadata(lease))],
            completed=False,
        )

    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        6,
        60,
        _source(fake_ray, driver, request_id="cancel-active", submitter=submitter),
    )
    try:
        events = _drain_until(
            collector,
            {6: {"rows": 1, "bytes": 128, "item_bytes": 128}},
            predicate=lambda values: any(item[2] == "data" for item in values),
        )
        assert events[0][2] == "data"
        collector.cancel_slot(6)
        assert fake_ray.cancel_calls
        assert fake_ray.cancel_calls[-1][1] == {"force": True, "recursive": True}
        assert len(driver.release_query_output_block_lease.calls) == 1
        assert collector.slot_has_pending(6) is False
    finally:
        collector.shutdown()


def test_shutdown_clears_callback_and_fully_joins_owned_thread():
    fake_ray = _FakeRay()
    driver = _Driver()
    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.set_wakeup_callback(lambda: None)
    collector.track_generator_ref(
        8,
        80,
        _source(
            fake_ray,
            driver,
            request_id="shutdown-join",
            submitter=lambda _lease: _Generator([], completed=False),
        ),
    )

    collector.shutdown()

    assert collector._wakeup_fn is None
    assert collector._thread.is_alive() is False
