# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

import duckdb.execution.ray_stream_adapter as stream_adapter_module
import duckdb.execution.udf_stream_result_collector as collector_module
from duckdb.execution.ray_stream_adapter import (
    RAY_STREAM_CLEANUP_CONTROL,
    RayStreamAdapter,
    RayStreamCleanupOperation,
    TaskLeaseObjectRefGenerator,
)
from duckdb.execution.udf_stream_result_collector import (
    AsyncResultCollector,
    _OutputLeaseToken,
    _ReadyEvent,
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
        self.wait_calls = []
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
        self.wait_calls.append((list(waitables), num_returns, timeout, time.monotonic()))

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
        self.cancel_started_count = 0
        self.cancel_started_lock = threading.Lock()

    def cancel(self, ref, **kwargs):
        with self.cancel_started_lock:
            self.cancel_started_count += 1
        self.cancel_started.set()
        self.allow_cancel.wait(timeout=2.0)
        super().cancel(ref, **kwargs)


class _BlockingWaitRay(_FakeRay):
    def __init__(self):
        super().__init__()
        self.wait_started = threading.Event()
        self.allow_wait = threading.Event()

    def wait(self, waitables, *, num_returns, timeout, fetch_local):
        self.wait_started.set()
        assert self.allow_wait.wait(timeout=2)
        return super().wait(
            waitables,
            num_returns=num_returns,
            timeout=timeout,
            fetch_local=fetch_local,
        )


class _FailingWaitRay(_FakeRay):
    def wait(self, waitables, *, num_returns, timeout, fetch_local):
        del waitables, num_returns, timeout, fetch_local
        raise RuntimeError("planned generator readiness failure")


class _SelectiveFailingWaitRay(_FakeRay):
    def __init__(self):
        super().__init__()
        self.failed_waitable = None

    def wait(self, waitables, *, num_returns, timeout, fetch_local):
        if len(waitables) > 1:
            raise RuntimeError("planned batch readiness failure")
        if waitables and waitables[0] is self.failed_waitable:
            raise RuntimeError("planned isolated generator readiness failure")
        return super().wait(
            waitables,
            num_returns=num_returns,
            timeout=timeout,
            fetch_local=fetch_local,
        )


class _RemoteMethod:
    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _Ref(self.fn(*args, **kwargs))


class _BlockingRemoteMethod(_RemoteMethod):
    def __init__(self, fn):
        super().__init__(fn)
        self.started = threading.Event()
        self.release = threading.Event()

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.started.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("timed out waiting to release blocked remote call")
        return _Ref(self.fn(*args, **kwargs))


class _DelayedAckRemoteMethod:
    def __init__(self, response_ref):
        self.response_ref = response_ref
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response_ref


class _Driver:
    def __init__(self):
        self.next_task = 0
        self.next_output = 0
        self.acquire_query_task_lease = _RemoteMethod(self._acquire_task)
        self.mark_query_task_lease_submitted = _RemoteMethod(lambda *_args: {"submitted": True})
        self.release_query_task_lease = _RemoteMethod(lambda *_args: {"released": True})
        self.release_query_task_lease_after_completion = _RemoteMethod(lambda *_args: {"scheduled": True})
        self.handoff_query_task_lease_to_teardown = _RemoteMethod(lambda *_args: {"handed_off": True})
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


def _wait_until(predicate, *, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("timed out waiting for collector state transition")


def _capture_thread_error(errors, operation, *args):
    try:
        operation(*args)
    except BaseException as exc:
        errors.append(exc)


def test_collector_requires_task_lease_stream_and_has_no_raw_generator_fallback():
    fake_ray = _FakeRay()
    collector = AsyncResultCollector(ray_module=fake_ray)
    try:
        with pytest.raises(TypeError, match="must return TaskLeaseObjectRefGenerator"):
            collector.track_generator_ref(1, 1, _Generator([]))
    finally:
        collector.shutdown()


def test_event_loop_start_failure_releases_the_registered_stream(monkeypatch):
    fake_ray = _FakeRay()
    driver = _Driver()
    generators = []

    def submitter(_lease):
        generator = _Generator([], completed=False)
        generators.append(generator)
        return generator

    collector = AsyncResultCollector(ray_module=fake_ray)

    def fail_start():
        raise RuntimeError("planned thread start failure")

    monkeypatch.setattr(collector._thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="planned thread start failure"):
        collector.track_generator_ref(
            1,
            2,
            _source(
                fake_ray,
                driver,
                request_id="start-failure",
                submitter=submitter,
            ),
        )

    assert collector._records == {}
    assert len(fake_ray.cancel_calls) == 1
    assert generators[0].deleted_streams == [generators[0].completion_ref]
    with pytest.raises(RuntimeError, match="planned thread start failure"):
        collector.drain_results({})
    collector.shutdown()


def test_scheduler_failure_rejection_cancels_the_unpublished_submitted_stream():
    fake_ray = _FailingWaitRay()
    driver = _Driver()
    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        1,
        1,
        _source(
            fake_ray,
            driver,
            request_id="readiness-failure-owner",
            submitter=lambda _lease: _Generator(
                [_Ref("never-consumed", ready=False, is_block=True)],
                completed=False,
            ),
        ),
    )
    collector.drain_results({1: {"rows": 1, "bytes": 128, "item_bytes": 128}})
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        with collector._cv:
            if collector._thread_error is not None:
                break
        time.sleep(0.005)
    else:
        pytest.fail("readiness scheduler failure was not published")

    rejected_generators = []

    def submit_rejected(_lease):
        generator = _Generator([], completed=False)
        rejected_generators.append(generator)
        return generator

    with pytest.raises(RuntimeError, match="planned generator readiness failure"):
        collector.track_generator_ref(
            1,
            2,
            _source(
                fake_ray,
                driver,
                request_id="rejected-after-readiness-failure",
                submitter=submit_rejected,
            ),
        )

    rejected = rejected_generators[0]
    assert any(ref is rejected for ref, _kwargs in fake_ray.cancel_calls)
    assert rejected.deleted_streams == [rejected.completion_ref]
    assert any(
        args[:3]
        == (
            "rejected-after-readiness-failure",
            "task-lease-2",
            "attempt:rejected-after-readiness-failure",
        )
        and kwargs == {}
        for args, kwargs in driver.release_query_task_lease_after_completion.calls
    )
    collector.shutdown()


def test_registration_is_not_schedulable_until_track_accepts_it(monkeypatch):
    fake_ray = _FakeRay()
    driver = _Driver()
    collector = AsyncResultCollector(ray_module=fake_ray)
    collector._ensure_started()
    collector.drain_results({1: {"rows": 1, "bytes": 128, "item_bytes": 128}})
    registration_paused = threading.Event()
    allow_registration = threading.Event()
    original_ensure_started = collector._ensure_started

    def pause_after_start():
        original_ensure_started()
        registration_paused.set()
        assert allow_registration.wait(timeout=2)

    monkeypatch.setattr(collector, "_ensure_started", pause_after_start)
    generators = []

    def submitter(lease):
        generator = _Generator(
            [
                _Ref("ready-during-registration", is_block=True),
                _Ref(_metadata(lease)),
            ]
        )
        generators.append(generator)
        return generator

    errors = []
    registration = threading.Thread(
        target=lambda: _capture_thread_error(
            errors,
            collector.track_generator_ref,
            1,
            2,
            _source(
                fake_ray,
                driver,
                request_id="registration-fence",
                submitter=submitter,
            ),
        )
    )
    try:
        registration.start()
        assert registration_paused.wait(timeout=1)
        time.sleep(0.05)
        with collector._cv:
            record = collector._records[(1, 2)]
            assert record.registration_accepted is False
        assert generators[0].read_count == 0
        assert collector._ready_by_slot.get(1) is None

        allow_registration.set()
        registration.join(timeout=2)
        assert registration.is_alive() is False
        assert errors == []
        events = _drain_until(
            collector,
            {1: {"rows": 1, "bytes": 128, "item_bytes": 128}},
            predicate=lambda values: any(item[2] == "complete" for item in values),
        )
        assert [item[2] for item in events] == ["data", "complete"]
    finally:
        allow_registration.set()
        registration.join(timeout=2)
        collector.shutdown()


def test_failure_wakeup_can_reenter_shutdown_after_cleanup_handoff():
    fake_ray = _FakeRay()
    driver = _Driver()
    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        1,
        1,
        _source(
            fake_ray,
            driver,
            request_id="reentrant-failure-shutdown",
            submitter=lambda _lease: _Generator([], completed=False),
        ),
    )
    with collector._cv:
        record = collector._records[(1, 1)]
    shutdown_errors = []
    collector.set_wakeup_callback(
        lambda: _capture_thread_error(
            shutdown_errors,
            collector.shutdown,
        )
    )

    collector._fail_record(record, RuntimeError("planned terminal failure"))

    assert shutdown_errors == []
    assert collector._shutdown is True
    assert collector._records == {}
    collector.shutdown()


def test_bounded_cleanup_lane_retries_transient_failure_without_starving_peer(
    monkeypatch,
):
    monkeypatch.setenv("VANE_UDF_STREAM_CONTROL_CLEANUP_WORKERS", "1")
    collector = AsyncResultCollector(ray_module=_FakeRay())
    attempts = 0
    order = []

    def fail_transiently():
        nonlocal attempts
        attempts += 1
        order.append(f"retry-{attempts}")
        if attempts < 6:
            raise RuntimeError("planned transient cleanup failure")

    group = collector._submit_cleanup_operations(
        (
            RayStreamCleanupOperation(
                RAY_STREAM_CLEANUP_CONTROL,
                fail_transiently,
                retry_on_error=True,
            ),
            RayStreamCleanupOperation(
                RAY_STREAM_CLEANUP_CONTROL,
                lambda: order.append("peer"),
            ),
        ),
        store_error=False,
    )
    try:
        assert group is not None
        assert group.wait(timeout=1)
        assert group.errors == ()
        assert order == ["retry-1", "peer", "retry-2", "retry-3", "retry-4", "retry-5", "retry-6"]
    finally:
        collector.shutdown()


def test_scheduler_fence_preserves_incomplete_ownership_retry():
    collector = AsyncResultCollector(ray_module=_FakeRay())
    attempts = 0

    def transfer_owner():
        nonlocal attempts
        attempts += 1
        return attempts >= 2

    operations = collector._fence_cleanup_operations(
        (
            RayStreamCleanupOperation(
                RAY_STREAM_CLEANUP_CONTROL,
                transfer_owner,
                retry_on_incomplete=True,
            ),
        )
    )
    group = collector._submit_cleanup_operations(operations, store_error=False)
    try:
        assert group is not None
        assert group.wait(timeout=1)
        assert group.errors == ()
        assert attempts == 2
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


def test_ready_block_read_reserves_data_capacity_across_streams():
    fake_ray = _FakeRay()
    driver = _Driver()
    generators = []

    def submitter(lease):
        generator = _Generator(
            [
                _Ref(f"block-{lease['lease_id']}", ready=False, is_block=True),
                _Ref(_metadata(lease), ready=False),
            ],
            completed=False,
        )
        generators.append(generator)
        return generator

    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        1,
        10,
        _source(fake_ray, driver, request_id="reserved-read-1", submitter=submitter),
    )
    collector.track_generator_ref(
        1,
        11,
        _source(fake_ray, driver, request_id="reserved-read-2", submitter=submitter),
    )
    capacity = {1: {"rows": 1, "bytes": 128, "item_bytes": 128}}
    try:
        assert collector.drain_results(capacity) == []
        with collector._cv:
            scheduled = [
                record.submit_id
                for record in collector._records.values()
                if record.wait_kind == "data" and record.wait_future is not None
            ]
        assert scheduled == []

        fake_ray.make_ready(generators[0].refs[0])
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            with collector._cv:
                scheduled = [
                    record.submit_id
                    for record in collector._records.values()
                    if record.wait_kind == "data" and record.wait_future is not None
                ]
            if scheduled:
                break
            time.sleep(0.005)
        assert scheduled == [10]

        # A dispatcher capacity refresh must not grant the same downstream
        # credit to the second stream while the first read remains in flight.
        assert collector.drain_results(capacity) == []
        with collector._cv:
            scheduled = [
                record.submit_id
                for record in collector._records.values()
                if record.wait_kind == "data" and record.wait_future is not None
            ]
        assert scheduled == [10]
    finally:
        collector.cancel_slot(1)
        collector.shutdown()


def test_data_capacity_does_not_count_interleaved_control_events():
    fake_ray = _FakeRay()
    collector = AsyncResultCollector(ray_module=fake_ray)
    token = _OutputLeaseToken(
        request_id="strict-consumer-request",
        lease_id="strict-consumer-lease",
        query_id="q1",
        driver=object(),
        slot_id=1,
        submit_id=11,
        size_bytes=64,
    )
    with collector._cv:
        collector._ready_by_slot[1].extend(
            [
                _ReadyEvent(1, 10, "complete", None),
                _ReadyEvent(1, 11, "data", "payload", size_bytes=64, output_token=token),
                _ReadyEvent(1, 11, "complete", None),
                _ReadyEvent(1, 12, "error", "boom"),
            ]
        )

    try:
        events = collector.drain_results({1: {"rows": 1, "bytes": 64, "item_bytes": 64}})

        assert [event[2] for event in events] == ["complete", "data", "complete", "error"]
        assert collector.drain_results({1: {"rows": 0, "bytes": 0, "item_bytes": 0}}) == []
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
        _wait_until(lambda: len(driver.handoff_query_output_block_lease.calls) == 1)
        assert len(driver.handoff_query_output_block_lease.calls) == 1
        assert driver.release_query_output_block_lease.calls == []

        assert collector.release_output_block_lease(data[4], data[5]) is True
        assert collector.release_output_block_lease(data[4], data[5]) is False
        _wait_until(lambda: len(driver.release_query_output_block_lease.calls) == 1)
        assert len(driver.release_query_output_block_lease.calls) == 1
        assert len(driver.release_query_task_lease.calls) == 1
    finally:
        collector.shutdown()


@pytest.mark.parametrize("operation", ["handoff", "release"])
def test_output_lease_control_retries_until_driver_acknowledges_ownership(operation):
    fake_ray = _FakeRay()
    driver = _Driver()
    attempts = 0

    def transient_failure(*_args):
        nonlocal attempts
        attempts += 1
        if attempts <= 5:
            raise RuntimeError(f"planned {operation} submission failure")
        if operation == "handoff":
            return {"handed_off": True}
        return {"released": True}

    if operation == "handoff":
        driver.handoff_query_output_block_lease = _RemoteMethod(transient_failure)
    else:
        driver.release_query_output_block_lease = _RemoteMethod(transient_failure)
    token = _OutputLeaseToken(
        request_id="output-request:retry",
        lease_id="output-lease-retry",
        query_id="q1",
        driver=driver,
        slot_id=2,
        submit_id=20,
        size_bytes=64,
    )
    collector = AsyncResultCollector(ray_module=fake_ray)
    key = (token.request_id, token.lease_id)
    with collector._cv:
        collector._active_output_leases[key] = token
    try:
        callback = (
            collector.handoff_output_block_lease if operation == "handoff" else collector.release_output_block_lease
        )
        assert callback(*key) is True
        if operation == "handoff":
            _wait_until(lambda: token.handed_off)
            assert attempts == 6
            assert collector.release_output_block_lease(*key) is True
        else:
            _wait_until(lambda: key not in collector._active_output_leases)
            assert attempts == 6
    finally:
        collector.shutdown()


def test_output_control_timeout_reuses_the_inflight_driver_response(monkeypatch):
    monkeypatch.setattr(
        stream_adapter_module,
        "_TASK_LEASE_CONTROL_ACK_TIMEOUT_S",
        0.01,
    )
    fake_ray = _FakeRay()
    driver = _Driver()
    response_ref = _Ref({"released": True}, ready=False)
    driver.release_query_output_block_lease = _DelayedAckRemoteMethod(response_ref)
    token = _OutputLeaseToken(
        request_id="output-request:delayed-ack",
        lease_id="output-lease-delayed-ack",
        query_id="q1",
        driver=driver,
        slot_id=2,
        submit_id=21,
        size_bytes=64,
    )
    collector = AsyncResultCollector(ray_module=fake_ray)
    key = (token.request_id, token.lease_id)
    with collector._cv:
        collector._active_output_leases[key] = token
    try:
        assert collector.release_output_block_lease(*key) is True
        _wait_until(lambda: bool(driver.release_query_output_block_lease.calls))
        time.sleep(0.08)

        assert len(driver.release_query_output_block_lease.calls) == 1
        with collector._cv:
            assert collector._active_output_leases[key] is token
            assert token.release_pending is True

        response_ref.ready = True
        _wait_until(lambda: key not in collector._active_output_leases)
        assert len(driver.release_query_output_block_lease.calls) == 1
    finally:
        response_ref.ready = True
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


def test_slot_cancel_fences_output_lease_acquire_before_ref_publication():
    fake_ray = _FakeRay()
    driver = _Driver()
    acquire = _BlockingRemoteMethod(driver._acquire_output)
    driver.acquire_query_output_block_lease = acquire

    def submitter(lease):
        return _Generator(
            [_Ref("block", is_block=True), _Ref(_metadata(lease))],
            completed=False,
        )

    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        2,
        24,
        _source(
            fake_ray,
            driver,
            request_id="cancel-during-output-acquire",
            submitter=submitter,
        ),
    )
    try:
        collector.drain_results({2: {"rows": 1, "bytes": 128, "item_bytes": 128}})
        assert acquire.started.wait(timeout=2.0)

        collector.cancel_slot(2)
        acquire.release.set()

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not driver.cancel_query_output_block_lease_request.calls:
            time.sleep(0.005)

        assert len(driver.cancel_query_output_block_lease_request.calls) == 1
        assert driver.cancel_query_output_block_lease_request.calls[0] == acquire.calls[0]
        assert collector.slot_has_pending(2) is False
    finally:
        acquire.release.set()
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


def test_terminal_stream_is_retired_before_completion_is_published(monkeypatch):
    fake_ray = _FakeRay()
    driver = _Driver()
    generator_holder = {}
    retire_started = threading.Event()
    allow_retire = threading.Event()
    original_retire_operations = RayStreamAdapter.retire_operations

    def blocking_retire_operations(adapter):
        operations = original_retire_operations(adapter)

        def block_retirement():
            retire_started.set()
            allow_retire.wait()

        return (
            RayStreamCleanupOperation(
                RAY_STREAM_CLEANUP_CONTROL,
                block_retirement,
            ),
            *operations,
        )

    monkeypatch.setattr(RayStreamAdapter, "retire_operations", blocking_retire_operations)

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
    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(2, 22, source)
    capacity = {2: {"rows": 1, "bytes": 128, "item_bytes": 128}}
    try:
        events = []
        deadline = time.monotonic() + 2.0
        while not retire_started.is_set() and time.monotonic() < deadline:
            events.extend(collector.drain_results(capacity))
            time.sleep(0.005)
        assert retire_started.is_set()
        events.extend(collector.drain_results(capacity))

        # Completion is the observable retirement boundary. Independent lanes
        # may already delete the local stream while task-accounting cleanup is
        # blocked, but the record cannot complete until the whole plan settles.
        assert [item[2] for item in events] == ["data"]
        assert collector._records
        assert collector._cleanup_groups
        generator = generator_holder["generator"]
        assert source_ref() is None
        assert generator.deleted_streams == [generator.completion_ref]

        data = next(item for item in events if item[2] == "data")
        assert collector.release_output_block_lease(data[4], data[5]) is True
        del data
        del events

        allow_retire.set()
        events = _drain_until(
            collector,
            capacity,
            predicate=lambda values: any(item[2] == "complete" for item in values),
        )

        assert [item[2] for item in events] == ["complete"]
        assert collector._records == {}
        assert source.generator is None
        assert source.lease is None
        assert generator.deleted_streams == [generator.completion_ref]
    finally:
        allow_retire.set()
        collector.shutdown()


def test_blocked_terminal_retirement_does_not_stall_healthy_stream(monkeypatch):
    fake_ray = _FakeRay()
    driver = _Driver()
    retire_started = threading.Event()
    allow_retire = threading.Event()
    healthy_block = _Ref("healthy-after-retire", ready=False, is_block=True)
    original_retire_operations = RayStreamAdapter.retire_operations

    def selectively_blocking_retire_operations(adapter):
        should_block = adapter.task_request_id == "blocked-retire"
        operations = original_retire_operations(adapter)
        if not should_block:
            return operations

        def block_retirement():
            retire_started.set()
            allow_retire.wait()

        return (
            RayStreamCleanupOperation(
                RAY_STREAM_CLEANUP_CONTROL,
                block_retirement,
            ),
            *operations,
        )

    monkeypatch.setattr(
        RayStreamAdapter,
        "retire_operations",
        selectively_blocking_retire_operations,
    )
    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        2,
        25,
        _source(
            fake_ray,
            driver,
            request_id="blocked-retire",
            submitter=lambda _lease: _Generator([]),
        ),
    )
    collector.track_generator_ref(
        3,
        35,
        _source(
            fake_ray,
            driver,
            request_id="healthy-after-retire",
            submitter=lambda lease: _Generator(
                [healthy_block, _Ref(_metadata(lease))],
                completed=False,
            ),
        ),
    )
    try:
        collector.drain_results(
            {
                2: {"rows": 0, "bytes": 0, "item_bytes": 0},
                3: {"rows": 1, "bytes": 128, "item_bytes": 128},
            }
        )
        assert retire_started.wait(timeout=2)

        fake_ray.make_ready(healthy_block)
        healthy_events = _drain_until(
            collector,
            {3: {"rows": 1, "bytes": 128, "item_bytes": 128}},
            predicate=lambda values: any(item[2] == "data" for item in values),
        )

        assert [item[2] for item in healthy_events] == ["data"]
        assert allow_retire.is_set() is False
    finally:
        allow_retire.set()
        collector.cancel_slot(3)
        collector.shutdown()


def test_slot_cancel_waits_for_already_claimed_terminal_retirement(monkeypatch):
    fake_ray = _FakeRay()
    driver = _Driver()
    retire_started = threading.Event()
    allow_retire = threading.Event()
    original_retire_operations = RayStreamAdapter.retire_operations

    def blocking_retire_operations(adapter):
        operations = original_retire_operations(adapter)

        def block_retirement():
            retire_started.set()
            assert allow_retire.wait(timeout=2)

        return (
            RayStreamCleanupOperation(
                RAY_STREAM_CLEANUP_CONTROL,
                block_retirement,
            ),
            *operations,
        )

    monkeypatch.setattr(RayStreamAdapter, "retire_operations", blocking_retire_operations)
    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        4,
        41,
        _source(
            fake_ray,
            driver,
            request_id="cancel-claimed-retirement",
            submitter=lambda _lease: _Generator([]),
        ),
    )
    collector.drain_results({4: {"rows": 0, "bytes": 0, "item_bytes": 0}})
    assert retire_started.wait(timeout=1)

    cancel_errors = []
    cancel_thread = threading.Thread(target=lambda: _capture_thread_error(cancel_errors, collector.cancel_slot, 4))
    try:
        cancel_thread.start()
        time.sleep(0.05)
        assert cancel_thread.is_alive()

        allow_retire.set()
        cancel_thread.join(timeout=2)
        assert cancel_thread.is_alive() is False
        assert cancel_errors == []
        assert collector.slot_has_pending(4) is False
    finally:
        allow_retire.set()
        cancel_thread.join(timeout=2)
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

        deadline = time.monotonic() + 1
        while not driver.release_query_output_block_lease.calls and time.monotonic() < deadline:
            time.sleep(0.005)
        assert driver.release_query_output_block_lease.calls == [
            ((record.output_request_id, "output-lease-1", metadata["query_id"]), {}),
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


def test_capacity_one_selects_ready_stream_without_prefetching_slow_stream():
    fake_ray = _FakeRay()
    driver = _Driver()
    slow_block = _Ref("slow", ready=False, is_block=True)
    generators = {}

    def slow_submitter(lease):
        generator = _Generator(
            [slow_block, _Ref(_metadata(lease))],
            completed=False,
        )
        generators["slow"] = generator
        return generator

    fast_block = _Ref("fast", is_block=True)

    def fast_submitter(lease):
        generator = _Generator([fast_block, _Ref(_metadata(lease))])
        generators["fast"] = generator
        return generator

    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        3,
        32,
        _source(fake_ray, driver, request_id="capacity-one-slow", submitter=slow_submitter),
    )
    collector.track_generator_ref(
        3,
        33,
        _source(fake_ray, driver, request_id="capacity-one-fast", submitter=fast_submitter),
    )
    try:
        events = _drain_until(
            collector,
            {3: {"rows": 1, "bytes": 128, "item_bytes": 128}},
            predicate=lambda values: any(item[1] == 33 and item[2] == "data" for item in values),
        )

        fast_data = next(item for item in events if item[1] == 33 and item[2] == "data")
        assert fast_data[3][1] == [fast_block]
        assert generators["slow"].read_count == 0
        assert generators["fast"].read_count == 2
        assert all(call[2] == 0 for call in fake_ray.wait_calls)
    finally:
        collector.cancel_slot(3)
        collector.shutdown()


def test_capacity_one_schedules_ready_streams_in_round_robin_order():
    fake_ray = _FakeRay()
    driver = _Driver()

    def submitter(lease):
        return _Generator(
            [
                _Ref(f"{lease['task_id']}-0", is_block=True),
                _Ref(_metadata(lease, index=0)),
                _Ref(f"{lease['task_id']}-1", is_block=True),
                _Ref(_metadata(lease, index=1)),
            ]
        )

    collector = AsyncResultCollector(ray_module=fake_ray)
    for submit_id in (37, 38):
        collector.track_generator_ref(
            3,
            submit_id,
            _source(
                fake_ray,
                driver,
                request_id=f"round-robin-{submit_id}",
                submitter=submitter,
            ),
        )
    try:
        events = _drain_until(
            collector,
            {3: {"rows": 1, "bytes": 128, "item_bytes": 128}},
            predicate=lambda values: sum(item[2] == "data" for item in values) == 4,
        )
        data_submit_ids = [item[1] for item in events if item[2] == "data"]

        assert data_submit_ids == [37, 38, 37, 38]
    finally:
        collector.cancel_slot(3)
        collector.shutdown()


def test_ready_generator_without_capacity_is_not_dequeued_or_polled():
    fake_ray = _FakeRay()
    driver = _Driver()
    holder = {}

    def submitter(lease):
        generator = _Generator(
            [_Ref("ready-without-capacity", is_block=True), _Ref(_metadata(lease))],
            completed=False,
        )
        holder["generator"] = generator
        return generator

    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        3,
        34,
        _source(fake_ray, driver, request_id="ready-without-capacity", submitter=submitter),
    )
    try:
        assert collector.drain_results({3: {"rows": 0, "bytes": 0, "item_bytes": 0}}) == []
        time.sleep(0.05)

        assert holder["generator"].read_count == 0
        assert fake_ray.wait_calls == []
    finally:
        collector.cancel_slot(3)
        collector.shutdown()


def test_unready_generator_polling_uses_bounded_idle_backoff():
    fake_ray = _FakeRay()
    driver = _Driver()
    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        3,
        39,
        _source(
            fake_ray,
            driver,
            request_id="idle-poll-backoff",
            submitter=lambda lease: _Generator(
                [_Ref("slow", ready=False, is_block=True), _Ref(_metadata(lease))],
                completed=False,
            ),
        ),
    )
    try:
        assert collector.drain_results({3: {"rows": 1, "bytes": 128, "item_bytes": 128}}) == []
        time.sleep(0.1)
        wait_count = len(fake_ray.wait_calls)

        assert 5 <= wait_count <= 20
        assert all(call[2] == 0 for call in fake_ray.wait_calls)
    finally:
        collector.cancel_slot(3)
        collector.shutdown()


def test_local_eligibility_change_interrupts_idle_readiness_backoff():
    fake_ray = _FakeRay()
    driver = _Driver()
    pending_first_output = _Ref(ready=False)
    output_calls = []

    class _OutputAdmission:
        def remote(self, request):
            output_calls.append(request)
            grant = driver._acquire_output(request)
            if len(output_calls) == 1:
                pending_first_output.value = grant
                return pending_first_output
            return _Ref(grant)

    driver.acquire_query_output_block_lease = _OutputAdmission()
    fast_generators = []

    def slow_submitter(lease):
        return _Generator(
            [_Ref("persistently-slow", ready=False, is_block=True), _Ref(_metadata(lease))],
            completed=False,
        )

    def fast_submitter(lease):
        generator = _Generator(
            [
                _Ref("fast-0", is_block=True),
                _Ref(_metadata(lease, index=0)),
                _Ref("fast-1", is_block=True),
                _Ref(_metadata(lease, index=1)),
            ]
        )
        fast_generators.append(generator)
        return generator

    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        3,
        35,
        _source(fake_ray, driver, request_id="idle-backoff-slow", submitter=slow_submitter),
    )
    collector.track_generator_ref(
        3,
        36,
        _source(fake_ray, driver, request_id="idle-backoff-fast", submitter=fast_submitter),
    )
    try:
        assert collector.drain_results({3: {"rows": 2, "bytes": 256, "item_bytes": 128}}) == []
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and (not output_calls or fast_generators[0].read_count < 2):
            time.sleep(0.005)
        assert len(output_calls) == 1
        assert fast_generators[0].read_count == 2

        # Let the slow-only poll reach its 10 ms idle ceiling. Completing the
        # local output-admission future must wake that sleep immediately.
        time.sleep(0.15)
        started_at = time.monotonic()
        pending_first_output.ready = True
        deadline = started_at + 0.08
        while time.monotonic() < deadline and fast_generators[0].read_count < 4:
            time.sleep(0.002)

        assert fast_generators[0].read_count == 4
        assert time.monotonic() - started_at < 0.08
        assert all(call[2] == 0 for call in fake_ray.wait_calls)
    finally:
        collector.cancel_slot(3)
        collector.shutdown()


def test_slot_cancel_fences_the_active_zero_time_readiness_probe():
    fake_ray = _BlockingWaitRay()
    driver = _Driver()
    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        3,
        40,
        _source(
            fake_ray,
            driver,
            request_id="cancel-readiness-probe",
            submitter=lambda lease: _Generator(
                [_Ref("slow", ready=False, is_block=True), _Ref(_metadata(lease))],
                completed=False,
            ),
        ),
    )
    assert collector.drain_results({3: {"rows": 1, "bytes": 128, "item_bytes": 128}}) == []
    assert fake_ray.wait_started.wait(timeout=1)
    errors = []
    cancel_thread = threading.Thread(
        target=lambda: _capture_thread_error(errors, collector.cancel_slot, 3),
    )
    cancel_thread.start()
    try:
        time.sleep(0.05)
        assert cancel_thread.is_alive()
        assert fake_ray.cancel_calls == []

        fake_ray.allow_wait.set()
        cancel_thread.join(timeout=1)
        assert cancel_thread.is_alive() is False
        assert errors == []
        assert len(fake_ray.cancel_calls) == 1
    finally:
        fake_ray.allow_wait.set()
        cancel_thread.join(timeout=1)
        collector.shutdown()


def test_shutdown_retains_cleanup_after_a_stuck_readiness_probe(monkeypatch):
    monkeypatch.setenv("VANE_UDF_STREAM_SHUTDOWN_TIMEOUT_S", "0.05")
    fake_ray = _BlockingWaitRay()
    driver = _Driver()
    generators = []

    def submitter(lease):
        generator = _Generator(
            [_Ref("slow", ready=False, is_block=True), _Ref(_metadata(lease))],
            completed=False,
        )
        generators.append(generator)
        return generator

    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        3,
        41,
        _source(
            fake_ray,
            driver,
            request_id="shutdown-readiness-probe",
            submitter=submitter,
        ),
    )
    assert collector.drain_results({3: {"rows": 1, "bytes": 128, "item_bytes": 128}}) == []
    assert fake_ray.wait_started.wait(timeout=1)

    started_at = time.monotonic()
    with pytest.raises(RuntimeError, match="multiplexer did not terminate"):
        collector.shutdown()
    assert time.monotonic() - started_at < 0.5
    assert fake_ray.cancel_calls == []

    fake_ray.allow_wait.set()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and (not fake_ray.cancel_calls or not generators[0].deleted_streams):
        time.sleep(0.005)

    assert len(fake_ray.cancel_calls) == 1
    assert generators[0].deleted_streams == [generators[0].completion_ref]
    collector._thread.join(timeout=1)
    assert collector._thread.is_alive() is False


def test_readiness_probe_failure_is_reported_without_dequeuing():
    fake_ray = _FailingWaitRay()
    driver = _Driver()
    generators = []

    def submitter(lease):
        generator = _Generator(
            [_Ref("ready", is_block=True), _Ref(_metadata(lease))],
            completed=False,
        )
        generators.append(generator)
        return generator

    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        3,
        42,
        _source(
            fake_ray,
            driver,
            request_id="readiness-probe-failure",
            submitter=submitter,
        ),
    )
    try:
        collector.drain_results({3: {"rows": 1, "bytes": 128, "item_bytes": 128}})
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            with collector._cv:
                if collector._thread_error is not None:
                    break
            time.sleep(0.005)

        with pytest.raises(RuntimeError, match="planned generator readiness failure"):
            collector.drain_results({3: {"rows": 1, "bytes": 128, "item_bytes": 128}})
        assert generators[0].read_count == 0
    finally:
        collector.shutdown()


def test_batch_wait_failure_isolates_one_bad_generator_from_healthy_slots():
    fake_ray = _SelectiveFailingWaitRay()
    driver = _Driver()
    failed_generator = _Generator(
        [_Ref("failed-block", is_block=True)],
        completed=False,
    )
    healthy_holder = {}

    def healthy_submitter(lease):
        generator = _Generator(
            [
                _Ref("healthy-block", is_block=True),
                _Ref(_metadata(lease)),
            ]
        )
        healthy_holder["generator"] = generator
        return generator

    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.track_generator_ref(
        31,
        310,
        _source(
            fake_ray,
            driver,
            request_id="isolated-readiness-failure",
            submitter=lambda _lease: failed_generator,
        ),
    )
    collector.track_generator_ref(
        32,
        320,
        _source(
            fake_ray,
            driver,
            request_id="healthy-readiness-neighbor",
            submitter=healthy_submitter,
        ),
    )
    fake_ray.failed_waitable = failed_generator
    capacities = {
        31: {"rows": 1, "bytes": 128, "item_bytes": 128},
        32: {"rows": 1, "bytes": 128, "item_bytes": 128},
    }
    events = []
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            events.extend(collector.drain_results(capacities))
            kinds_by_slot = {(event[0], event[2]) for event in events}
            if (31, "error") in kinds_by_slot and (32, "complete") in kinds_by_slot:
                break
            time.sleep(0.005)

        assert any(
            event[0] == 31 and event[2] == "error" and "planned isolated generator readiness failure" in event[3]
            for event in events
        )
        assert [event[2] for event in events if event[0] == 32] == [
            "data",
            "complete",
        ]
        assert failed_generator.read_count == 0
        assert healthy_holder["generator"].read_count == 2
        with collector._cv:
            assert collector._thread_error is None
    finally:
        collector.shutdown()


@pytest.mark.usefixtures("ray_local")
def test_real_ray_generator_wait_is_non_consuming_and_preserves_pair_order():
    import ray

    @ray.remote(num_returns="streaming")
    def yield_pair(metadata):
        yield "real-ray-block"
        yield metadata

    driver = _Driver()

    def submitter(lease):
        metadata = {
            "protocol_version": 1,
            "query_id": lease["query_id"],
            "producer_stage_id": lease["stage_id"],
            "task_lease_id": lease["lease_id"],
            "attempt_id": lease["attempt_id"],
            "block_id": "real-ray:block:0",
            "size_bytes": 64,
            "num_rows": 1,
            "names": ["value"],
        }
        return yield_pair.remote(metadata)

    collector = AsyncResultCollector(ray_module=ray)
    collector.track_generator_ref(
        3,
        43,
        _source(
            ray,
            driver,
            request_id="real-ray-readiness",
            submitter=submitter,
        ),
    )
    try:
        events = _drain_until(
            collector,
            {3: {"rows": 1, "bytes": 128, "item_bytes": 128}},
            predicate=lambda values: any(item[2] == "complete" for item in values),
            timeout=10,
        )

        assert [item[2] for item in events] == ["data", "complete"]
        block_ref = events[0][3][1][0]
        assert ray.get(block_ref) == "real-ray-block"
    finally:
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
        deadline = time.monotonic() + 1
        while not driver.release_query_task_lease_after_completion.calls and time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(driver.release_query_task_lease_after_completion.calls) == 1
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
        assert len(driver.release_query_task_lease_after_completion.calls) == 1
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
        assert len(driver.release_query_task_lease_after_completion.calls) == 1
        assert fake_ray.cancel_calls[-1][1] == {"force": True, "recursive": True}
    finally:
        collector.shutdown()


def test_stream_error_wakes_dispatcher_before_generator_cancellation():
    fake_ray = _BlockingCancelRay()
    driver = _Driver()
    wakeup = threading.Event()
    healthy_block = _Ref("healthy-block", ready=False, is_block=True)

    def submitter(_lease):
        return _Generator([_Ref("block", is_block=True), _Ref({"size_bytes": 1})])

    collector = AsyncResultCollector(ray_module=fake_ray)
    collector.set_wakeup_callback(wakeup.set)
    collector.track_generator_ref(
        8,
        80,
        _source(fake_ray, driver, request_id="blocking-cancel", submitter=submitter),
    )
    collector.track_generator_ref(
        9,
        90,
        _source(
            fake_ray,
            driver,
            request_id="healthy-beside-blocking-cancel",
            submitter=lambda lease: _Generator(
                [healthy_block, _Ref(_metadata(lease))],
                completed=False,
            ),
        ),
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
        fake_ray.make_ready(healthy_block)
        healthy_events = _drain_until(
            collector,
            {9: {"rows": 1, "bytes": 128, "item_bytes": 128}},
            predicate=lambda values: any(item[2] == "data" for item in values),
        )
        assert [item[2] for item in healthy_events] == ["data"]

        fake_ray.allow_cancel.set()
        events = _drain_until(
            collector,
            {8: {"rows": 1, "bytes": 128, "item_bytes": 128}},
            predicate=lambda values: any(item[2] == "error" for item in values),
        )
        assert [item[2] for item in events] == ["error"]
    finally:
        fake_ray.allow_cancel.set()
        collector.cancel_slot(9)
        collector.shutdown()


def test_blocked_cancellations_use_bounded_workers_and_preserve_delete_ordering(
    monkeypatch,
):
    monkeypatch.setenv("VANE_UDF_STREAM_SHUTDOWN_TIMEOUT_S", "0.05")
    monkeypatch.setenv("VANE_UDF_STREAM_CANCELLATION_CLEANUP_WORKERS", "2")
    fake_ray = _BlockingCancelRay()
    driver = _Driver()
    generators = []
    collector = AsyncResultCollector(ray_module=fake_ray)
    for index in range(12):
        generator = _Generator([], completed=False)
        generators.append(generator)
        collector.track_generator_ref(
            18,
            index,
            _source(
                fake_ray,
                driver,
                request_id=f"bounded-cancel-{index}",
                submitter=lambda _lease, generator=generator: generator,
            ),
        )

    try:
        with pytest.raises(RuntimeError, match="cleanup did not terminate"):
            collector.cancel_slot(18)

        assert fake_ray.cancel_started_count == 2
        cancellation_pool = collector._cleanup_pools["cancellation"]
        assert len(cancellation_pool.threads) == 2
        assert driver.cancel_query_task_lease_request.calls == []
        assert len(driver.release_query_task_lease_after_completion.calls) == len(generators)
        assert all(generator.deleted_streams == [] for generator in generators)

        fake_ray.allow_cancel.set()
        deadline = time.monotonic() + 3
        while (
            len(fake_ray.cancel_calls) < len(generators)
            or not all(generator.deleted_streams for generator in generators)
        ) and time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(fake_ray.cancel_calls) == len(generators)
        assert all(generator.deleted_streams == [generator.completion_ref] for generator in generators)
    finally:
        fake_ray.allow_cancel.set()
        collector.shutdown()


def test_blocked_output_controls_do_not_withhold_task_or_stream_cleanup(
    monkeypatch,
):
    monkeypatch.setenv("VANE_UDF_STREAM_OUTPUT_CLEANUP_WORKERS", "2")
    fake_ray = _FakeRay()
    driver = _Driver()
    blocked_release = _BlockingRemoteMethod(lambda *_args: {"released": True})
    driver.release_query_output_block_lease = blocked_release
    collector = AsyncResultCollector(ray_module=fake_ray)
    output_tokens = [
        _OutputLeaseToken(
            request_id=f"output-request:blocked:{index}",
            lease_id=f"output-lease-blocked-{index}",
            query_id="q1",
            driver=driver,
            slot_id=71,
            submit_id=index,
            size_bytes=64,
        )
        for index in range(2)
    ]
    with collector._cv:
        for token in output_tokens:
            collector._active_output_leases[(token.request_id, token.lease_id)] = token
    for token in output_tokens:
        assert collector.release_output_block_lease(token.request_id, token.lease_id)
    _wait_until(lambda: len(blocked_release.calls) == 2)

    generator = _Generator([], completed=False)
    collector.track_generator_ref(
        72,
        720,
        _source(
            fake_ray,
            driver,
            request_id="cleanup-beside-blocked-output",
            submitter=lambda _lease: generator,
        ),
    )
    try:
        collector.cancel_slot(72)

        assert fake_ray.cancel_calls == [
            (generator, {"force": True, "recursive": True}),
        ]
        assert generator.deleted_streams == [generator.completion_ref]
        assert len(driver.release_query_task_lease_after_completion.calls) == 1
        assert all(token.release_pending for token in output_tokens)
    finally:
        blocked_release.release.set()
        _wait_until(
            lambda: all(
                (token.request_id, token.lease_id) not in collector._active_output_leases for token in output_tokens
            )
        )
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
        assert len(driver.release_query_task_lease_after_completion.calls) == 1
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
    assert all(not thread.is_alive() for pool in collector._cleanup_pools.values() for thread in pool.threads)
