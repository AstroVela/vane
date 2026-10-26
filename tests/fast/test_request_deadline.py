# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from vane.execution import request_admission, request_deadline, udf_local_request
from vane.execution.request_admission import (
    RequestAdmissionLimits,
    RequestCancelled,
    RequestExecutionTimeout,
    RequestQueueTimeout,
)
from vane.execution.request_deadline import RequestExecutionDeadline
from vane.execution.udf_local_model import LocalModelRuntime


def runtime(monkeypatch, active=1):
    model_runtime = LocalModelRuntime(
        session_id="deadlines", session_config={}, request_limit=RequestAdmissionLimits(active, 2)
    )
    monkeypatch.setattr(model_runtime, "_prepare", lambda *args, **kwargs: [])
    monkeypatch.setattr(udf_local_request, "_execute_native", lambda conn, plan, **kwargs: plan)
    return model_runtime


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf"), True, "1", object()])
def test_invalid_execution_timeout_does_not_consume_request(monkeypatch, timeout):
    with runtime(monkeypatch) as models:
        request = models.request()
        with pytest.raises(ValueError, match="execution_timeout"):
            request.execute("result", {}, conn=object(), execution_timeout=timeout)
        assert request.state == "ready"
        assert request.execute("result", {}, conn=object()) == "result"


def test_zero_execution_timeout_expires_before_preparation(monkeypatch):
    with runtime(monkeypatch) as models:
        prepared = []
        monkeypatch.setattr(models, "_prepare", lambda *args, **kwargs: prepared.append(True))
        request, queued = models.request(), models.request()
        with pytest.raises(RequestExecutionTimeout):
            request.execute("result", {}, conn=object(), execution_timeout=0)
        assert not prepared
        assert request.state == "execution_timed_out"
        assert request.cancellation_reason == "execution_timeout"
        assert queued.state == "ready"
        state = models.resource_snapshot()["request_admission"]
        assert state["execution_timed_out_requests"] == 1
        assert state["timed_out_requests"] == state["cancelled_requests"] == state["completed_requests"] == 0
        assert not request.cancel()


def test_completion_checks_deadline_when_watcher_is_delayed(monkeypatch):
    now = [10.0]
    clock = SimpleNamespace(monotonic=lambda: now[0])
    monkeypatch.setattr(request_admission, "time", clock)
    monkeypatch.setattr(request_deadline, "time", clock)
    monkeypatch.setattr(RequestExecutionDeadline, "start", lambda self: None)
    with runtime(monkeypatch) as models:
        request = models.request()
        now[0] = 100.0  # Time spent holding an unclaimed ready ticket is excluded.

        def execute(*args, **kwargs):
            assert request._ticket.claimed_at == 100.0
            now[0] = 102.0
            return "overdue result"

        monkeypatch.setattr(udf_local_request, "_execute_native", execute)
        with pytest.raises(RequestExecutionTimeout):
            request.execute("result", {}, conn=object(), execution_timeout=1)
        assert request.state == "execution_timed_out"


@pytest.mark.parametrize("stage", ["start", "prepare"])
def test_delayed_watcher_cannot_start_native_work_after_expiry(monkeypatch, stage):
    now = [10.0]
    clock = SimpleNamespace(monotonic=lambda: now[0])
    monkeypatch.setattr(request_admission, "time", clock)
    monkeypatch.setattr(request_deadline, "time", clock)

    def start(self):
        if stage == "start":
            now[0] = 12.0

    monkeypatch.setattr(RequestExecutionDeadline, "start", start)
    with runtime(monkeypatch) as models:
        native_calls, cleaned = [], []

        def prepare(*args, **kwargs):
            now[0] = 12.0
            return [SimpleNamespace(shutdown=lambda **k: cleaned.append(True), cleanup_pending=lambda: False)]

        monkeypatch.setattr(models, "_prepare", prepare)
        monkeypatch.setattr(udf_local_request, "_execute_native", lambda *a, **k: native_calls.append(True))
        request = models.request()
        with pytest.raises(RequestExecutionTimeout):
            request.execute("result", {}, conn=object(), execution_timeout=1)
        assert not native_calls
        assert cleaned == ([True] if stage == "prepare" else [])


@pytest.mark.parametrize("stage", ["prepare", "execute"])
def test_failure_before_deadline_preserves_primary_error_and_stops_watcher(monkeypatch, stage):
    with runtime(monkeypatch) as models:

        def fail(*args, **kwargs):
            raise ValueError("original execution failure")

        if stage == "prepare":
            monkeypatch.setattr(models, "_prepare", fail)
        else:
            monkeypatch.setattr(udf_local_request, "_execute_native", fail)
        request = models.request()
        with pytest.raises(ValueError, match="original execution failure"):
            request.execute("result", {}, conn=object(), execution_timeout=60)
        assert request.cancellation_reason is None and request.state == "finished"
        assert request._deadline._stopped.is_set()
        assert request._deadline._callback is None


def test_queue_wait_does_not_spend_execution_timeout(monkeypatch):
    with runtime(monkeypatch) as models, ThreadPoolExecutor(max_workers=1) as threads:
        holder = models.request()._ticket.take()
        request = models.request()
        future = threads.submit(request.execute, "result", {}, conn=object(), execution_timeout=0.1)
        try:
            time.sleep(0.15)
            assert request.state == "queued" and request._deadline is None
        finally:
            holder.release()
        assert future.result(timeout=3) == "result"
        assert request.state == "finished"


def test_queue_expiration_does_not_start_execution_deadline(monkeypatch):
    with runtime(monkeypatch) as models:
        holder = models.request()._ticket.take()
        request = models.request(queue_timeout=0.02)
        try:
            with pytest.raises(RequestQueueTimeout):
                request.execute("result", {}, conn=object(), execution_timeout=0.001)
            assert request.state == "timed_out" and request._deadline is None
            state = models.resource_snapshot()["request_admission"]
            assert state["timed_out_requests"] == 1 and state["execution_timed_out_requests"] == 0
        finally:
            holder.release()


def test_slow_cleanup_is_outside_execution_deadline(monkeypatch):
    entered, proceed = threading.Event(), threading.Event()

    class Cleanup:
        def shutdown(self, *, kill=False):
            entered.set()
            assert proceed.wait(5)

        def cleanup_pending(self):
            return not proceed.is_set()

    with runtime(monkeypatch) as models, ThreadPoolExecutor(max_workers=1) as threads:
        monkeypatch.setattr(models, "_prepare", lambda *a, **k: [Cleanup()])
        request, queued = models.request(), models.request()
        future = threads.submit(request.execute, "result", {}, conn=object(), execution_timeout=0.1)
        try:
            assert entered.wait(3)
            time.sleep(0.15)
            assert request.cancellation_reason is None
            assert request._deadline._stopped.is_set()
            assert queued.state == "queued"
        finally:
            proceed.set()
        assert future.result(timeout=3) == "result"
        assert request.state == "finished" and queued.state == "ready"


def test_slow_cancellation_cannot_delay_another_deadline(monkeypatch):
    blocked, release, other_cancelled = (threading.Event() for _ in range(3))
    with runtime(monkeypatch, active=2) as models, ThreadPoolExecutor(max_workers=2) as threads:

        def execute(conn, plan, *, cancellation):
            def cancelled():
                if plan == "slow":
                    blocked.set()
                    assert release.wait(5)
                else:
                    other_cancelled.set()

            cancellation.register_cancel_wakeup(cancelled)
            assert cancellation._event.wait(3)
            return "too late"

        monkeypatch.setattr(udf_local_request, "_execute_native", execute)
        first, second = models.request(), models.request()
        slow = threads.submit(first.execute, "slow", {}, conn=object(), execution_timeout=0.1)
        fast = threads.submit(second.execute, "fast", {}, conn=object(), execution_timeout=0.1)
        try:
            assert blocked.wait(3)
            assert other_cancelled.wait(3)
            with pytest.raises(RequestExecutionTimeout):
                fast.result(timeout=3)
            assert not slow.done()
            assert first.state == "cancelling"
            assert models.resource_snapshot()["request_admission"]["active_requests"] == 1
        finally:
            release.set()
        with pytest.raises(RequestExecutionTimeout):
            slow.result(timeout=3)


def test_copied_deadline_callback_is_fenced_after_completion(monkeypatch):
    callbacks = []
    expired = [False]
    monkeypatch.setattr(RequestExecutionDeadline, "start", lambda self: callbacks.append(self._callback))
    monkeypatch.setattr(RequestExecutionDeadline, "expired", lambda self: expired[0])
    with runtime(monkeypatch) as models:
        request = models.request()
        assert request.execute("first", {}, conn=object(), execution_timeout=1) == "first"
        expired[0] = True
        callbacks[0]()  # The watcher copied this before close removed it.
        assert request.state == "finished" and request.cancellation_reason is None
        assert not request._cancellation.is_set()
        assert models.request().execute("next", {}, conn=object()) == "next"


@pytest.mark.parametrize("timeout_first", [False, True])
def test_first_accepted_cancellation_cause_wins(monkeypatch, timeout_first):
    expired = [False]
    monkeypatch.setattr(RequestExecutionDeadline, "start", lambda self: None)
    monkeypatch.setattr(RequestExecutionDeadline, "expired", lambda self: expired[0])
    with runtime(monkeypatch) as models:
        request = models.request()

        def execute(*args, **kwargs):
            if timeout_first:
                expired[0] = True
                request._expire_deadline()
                assert not request.cancel()
            else:
                assert request.cancel()
                expired[0] = True
                request._expire_deadline()
            return "cancelled result"

        monkeypatch.setattr(udf_local_request, "_execute_native", execute)
        with pytest.raises(RequestExecutionTimeout if timeout_first else RequestCancelled):
            request.execute("result", {}, conn=object(), execution_timeout=1)
        assert request.state == ("execution_timed_out" if timeout_first else "cancelled")


def test_deadline_start_failure_preserves_error_and_releases_request(monkeypatch):
    def fail_start(self):
        raise RuntimeError("watcher startup failure")

    monkeypatch.setattr(RequestExecutionDeadline, "start", fail_start)
    with runtime(monkeypatch) as models:
        request = models.request()
        ref = weakref.ref(request)
        with pytest.raises(RuntimeError, match="watcher startup failure"):
            request.execute("result", {}, conn=object(), execution_timeout=60)
        assert models.resource_snapshot()["request_admission"]["active_requests"] == 0
        del request
        gc.collect()
        assert ref() is None


def test_long_deadline_can_be_closed_without_overflow_or_retaining_callback():
    callback = threading.Event()
    deadline = RequestExecutionDeadline(time.monotonic(), 1e100, callback.set)
    deadline.start()
    deadline.close()
    deadline.start()
    assert deadline._callback is None
    assert not callback.is_set()
