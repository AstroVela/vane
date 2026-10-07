# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import Future

import pytest

from vane.runners.fte import FteTaskAttemptId, FteTaskId
from vane.runners.fte.fte_events import TaskStatusChanged
from vane.runners.fte.fte_scheduler import FteEventHandlers, FteQueryScheduler
from vane.runners.ray.fte_status_observation import FteAttemptStatusWatcher, FteStatusObservationRuntime
from vane.runners.ray.safe_get import QueryDeadlineExceeded


class _Worker:
    worker_id = "status-worker"
    worker_incarnation_id = "status-incarnation"
    manager_instance_id = "status-manager"


@pytest.fixture
def observations():
    runtimes = []
    watchers = []

    def create(worker, *, partition=0, runtime=None, handlers=None, scheduler=None):
        if runtime is None:
            runtime = FteStatusObservationRuntime(capacity=2, dispatch_workers=2)
        if runtime not in runtimes:
            runtimes.append(runtime)
        if scheduler is None:
            scheduler = FteQueryScheduler("status-query")
        if handlers is not None:
            scheduler.set_handlers(handlers)
        watcher = FteAttemptStatusWatcher(
            scheduler=scheduler,
            attempt_id=FteTaskAttemptId(FteTaskId("status-query", 0, partition), 0),
            worker=worker,
            runtime=runtime,
        )
        watchers.append(watcher)
        return watcher

    yield create
    for watcher in watchers:
        watcher.stop()
    for watcher in watchers:
        watcher.join(5.0)
        assert not watcher.is_alive()
    for runtime in runtimes:
        runtime.close()


def test_many_attempts_share_threads_and_bound_remote_status_waits(observations):
    entered = threading.Event()
    release = Future()

    class Worker(_Worker):
        active = 0
        peak = 0

        def __init__(self):
            self.calls = []

        async def fte_wait_task_status_async(self, task_id, min_version, timeout_s):
            self.calls.append(task_id["partition_id"])
            self.active += 1
            self.peak = max(self.peak, self.active)
            if self.active == 16:
                entered.set()
            try:
                await asyncio.shield(asyncio.wrap_future(release))
                return {"task_id": task_id, "state": "FINISHED", "version": 1}
            finally:
                self.active -= 1

    before_threads = {t.ident for t in threading.enumerate()}
    runtime = FteStatusObservationRuntime(capacity=16, dispatch_workers=2)
    worker = Worker()
    seen = []
    watchers = [
        observations(
            worker,
            partition=i,
            runtime=runtime,
            handlers=FteEventHandlers(on_task_status_changed=lambda event: seen.append(event.attempt_id)),
        )
        for i in range(512)
    ]
    try:
        for watcher in watchers:
            watcher.start()
        assert entered.wait(2.0)
        assert len(worker.calls) == 16
        created = [t for t in threading.enumerate() if t.ident not in before_threads]
        assert len(created) <= 3
        assert not any(t.name.startswith("fte-status-watcher-") for t in created)
    finally:
        release.set_result(None)
    for watcher in watchers:
        watcher.join(5.0)
        assert not watcher.is_alive()
        assert watcher.completion.result()["state"] == "FINISHED"
    assert worker.peak == 16
    assert sorted(worker.calls) == list(range(512))
    assert len(seen) == 512
    assert len([t for t in threading.enumerate() if t.ident not in before_threads]) <= 3


def test_stop_cancels_capacity_wait_without_submitting_a_remote_request(observations):
    entered = threading.Event()

    class Worker(_Worker):
        def __init__(self):
            self.calls = []

        async def fte_wait_task_status_async(self, task_id, min_version, timeout_s):
            self.calls.append(task_id["partition_id"])
            entered.set()
            await asyncio.Event().wait()

    runtime = FteStatusObservationRuntime(capacity=1, dispatch_workers=1)
    worker = Worker()
    first = observations(worker, runtime=runtime)
    second = observations(worker, partition=1, runtime=runtime)
    first.start()
    assert entered.wait(1.0)
    second.start()
    second.stop()
    second.join(1.0)
    assert not second.is_alive()
    assert worker.calls == [0]
    with pytest.raises(InterruptedError, match="observation stopped"):
        second.completion.result()


def test_query_deadline_expires_while_queued_for_observation_capacity(observations, monkeypatch):
    entered = threading.Event()

    class Worker(_Worker):
        async def fte_wait_task_status_async(self, task_id, min_version, timeout_s):
            assert task_id["partition_id"] == 0
            entered.set()
            await asyncio.Event().wait()

    runtime = FteStatusObservationRuntime(capacity=1, dispatch_workers=1)
    first = observations(Worker(), runtime=runtime)
    first.start()
    assert entered.wait(1.0)
    monkeypatch.setenv("VANE_QUERY_DEADLINE_EPOCH_S", str(time.time() + 0.1))
    failures = []
    second = observations(
        Worker(),
        partition=1,
        runtime=runtime,
        handlers=FteEventHandlers(on_worker_failed=lambda event: failures.append(event.error)),
    )
    second.start()
    second.join(2.0)
    assert not second.is_alive()
    with pytest.raises(QueryDeadlineExceeded):
        second.completion.result()
    assert len(failures) == 1
    assert isinstance(failures[0], QueryDeadlineExceeded)


def test_stop_retains_ownership_until_remote_wait_cleanup_finishes(observations):
    entered = threading.Event()
    cleanup_entered = threading.Event()
    release_cleanup = Future()

    class Worker(_Worker):
        async def fte_wait_task_status_async(self, task_id, min_version, timeout_s):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_entered.set()
                await asyncio.shield(asyncio.wrap_future(release_cleanup))

    watcher = observations(Worker())
    watcher.start()
    assert entered.wait(1.0)
    try:
        watcher.stop()
        assert cleanup_entered.wait(1.0)
        watcher.stop()
        watcher.join(0.05)
        assert watcher.is_alive()
        assert not watcher.completion.done()
        with pytest.raises(RuntimeError, match="live watchers"):
            watcher._runtime.close()
    finally:
        release_cleanup.set_result(None)
    watcher.join(1.0)
    assert not watcher.is_alive()


def test_stop_does_not_abandon_terminal_dispatch_or_shared_completion(observations):
    entered = threading.Event()
    release = threading.Event()

    class Worker(_Worker):
        async def fte_wait_task_status_async(self, task_id, min_version, timeout_s):
            return {"task_id": task_id, "state": "FINISHED", "version": 1, "stats": [42]}

    def terminal(event):
        assert event.status["state"] == "FINISHED"
        entered.set()
        assert release.wait(2.0)

    watcher = observations(Worker(), handlers=FteEventHandlers(on_task_status_changed=terminal))
    watcher.start()
    try:
        assert entered.wait(1.0)
        watcher.stop()
        watcher.join(0.05)
        assert watcher.is_alive()
        assert not watcher.completion.done()
    finally:
        release.set()
    watcher.join(1.0)
    assert watcher.completion.result()["stats"] == [42]


@pytest.mark.parametrize("drainer_fails", [False, True])
def test_external_scheduler_drainer_keeps_status_pending_and_capacity_owned(observations, drainer_fails):
    drainer_entered = threading.Event()
    release_drainer = threading.Event()
    callback_returned = threading.Event()
    failures = []
    drainer_errors = []
    seen = []

    class Worker(_Worker):
        def __init__(self):
            self.calls = []

        async def fte_wait_task_status_async(self, task_id, min_version, timeout_s):
            self.calls.append(task_id["partition_id"])
            return {"task_id": task_id, "state": "FINISHED", "version": 1}

    def status_changed(event):
        if event.attempt_id.partition_id == 99:
            drainer_entered.set()
            assert release_drainer.wait(3.0)
            if drainer_fails:
                raise RuntimeError("external drainer failed")
        else:
            seen.append(event.attempt_id.partition_id)

    scheduler = FteQueryScheduler("status-query")
    scheduler.set_handlers(
        FteEventHandlers(
            on_task_status_changed=status_changed,
            on_worker_failed=lambda event: failures.append(event.error),
        )
    )
    initial = FteTaskAttemptId(FteTaskId("status-query", 0, 99), 0)
    scheduler.enqueue(
        TaskStatusChanged.from_status("status-query", initial, {"task_id": initial.to_dict(), "state": "RUNNING"})
    )

    def drain():
        try:
            scheduler.drain()
        except BaseException as error:
            drainer_errors.append(error)

    runtime = FteStatusObservationRuntime(capacity=1, dispatch_workers=1)
    original_dispatch = runtime.dispatch

    async def dispatch(function, *args):
        result = await original_dispatch(function, *args)
        callback_returned.set()
        return result

    runtime.dispatch = dispatch
    worker = Worker()
    first = observations(worker, runtime=runtime, scheduler=scheduler)
    second = None if drainer_fails else observations(worker, partition=1, runtime=runtime, scheduler=scheduler)
    drainer = threading.Thread(target=drain)
    drainer.start()
    try:
        assert drainer_entered.wait(1.0)
        first.start()
        assert callback_returned.wait(1.0)
        first.stop()
        if second is not None:
            second.start()

        async def checkpoint():
            await asyncio.sleep(0)
            return runtime.slots.locked(), runtime.dispatch_slots.locked()

        assert asyncio.run_coroutine_threadsafe(checkpoint(), runtime.loop).result(timeout=1.0) == (True, False)
        assert first.is_alive()
        assert not first.completion.done()
        assert worker.calls == [0]
        assert seen == []
    finally:
        release_drainer.set()
        drainer.join(2.0)
    assert not drainer.is_alive()
    first.join(2.0)
    assert not first.is_alive()
    if drainer_fails:
        with pytest.raises(RuntimeError, match="external drainer failed"):
            first.completion.result()
        assert len(drainer_errors) == len(failures) == 1
    else:
        assert first.completion.result()["state"] == "FINISHED"
        second.join(2.0)
        assert not second.is_alive()
        assert second.completion.result()["state"] == "FINISHED"
        assert worker.calls == seen == [0, 1]
        assert drainer_errors == failures == []


def test_soft_status_timeout_retries_without_reporting_worker_loss(observations):
    failures = []

    class Worker(_Worker):
        calls = 0

        async def fte_wait_task_status_async(self, task_id, min_version, timeout_s):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary status wait timeout")
            return {"task_id": task_id, "state": "FINISHED", "version": 1}

    worker = Worker()
    watcher = observations(worker, handlers=FteEventHandlers(on_worker_failed=lambda event: failures.append(event)))
    watcher.start()
    watcher.join(1.0)
    assert watcher.completion.result()["state"] == "FINISHED"
    assert worker.calls == 2
    assert failures == []
