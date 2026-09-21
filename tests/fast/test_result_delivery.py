# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pyarrow as pa
import pytest

from vane.execution import local_result_delivery, request_admission, request_deadline, result_delivery
from vane.execution.local_result_delivery import prepare_local_result
from vane.execution.request_admission import RequestAdmissionLimits, RequestCancelled, RequestQueueTimeout
from vane.execution.request_deadline import MonotonicDeadline
from vane.execution.result_delivery import (
    ResultDeliveryCancelled,
    ResultDeliveryClosed,
    ResultDeliveryFull,
    ResultDeliveryLimits,
    ResultDeliveryTimeout,
    RuntimeResultDelivery,
)
from vane.execution.udf_local_model import LocalModelRuntime


class Payload:
    def __init__(self, result, *, value="value", size=32):
        self.owner = result.own_buffer(size)
        self.value = value
        self.fail_close = False
        self.calls = 0
        result.hold(self)

    def export(self, cancellation):
        cancellation.raise_if_cancelled()
        return self.value

    def close(self):
        self.calls += 1
        if self.fail_close:
            raise OSError("injected payload cleanup failure")
        if self.owner is not None:
            self.owner.release()
            self.owner = None

    def cleanup_pending(self):
        return self.owner is not None


def registry(*, results=2, size=4096):
    return RuntimeResultDelivery(ResultDeliveryLimits(results, size))


def native(*tables):
    return SimpleNamespace(result_schema={"names": ["x"], "types": ["BIGINT"]}, partition_payloads=list(tables))


def clock(monkeypatch):
    now = [10.0]
    timer = SimpleNamespace(monotonic=lambda: now[0])
    monkeypatch.setattr(result_delivery, "time", timer)
    monkeypatch.setattr(request_deadline, "time", timer)
    monkeypatch.setattr(MonotonicDeadline, "start", lambda self: None)
    return now


@pytest.mark.parametrize("outcome", ["delivered", "cancelled", "closed", "delivery_timed_out"])
def test_delivery_timing_includes_pending_cleanup_once_but_not_preparation(monkeypatch, outcome):
    now = clock(monkeypatch)
    runtime = registry()
    result = runtime.begin()
    payload = Payload(result)
    now[0] = 20.0
    result.ready(delivery_timeout=2 if outcome == "delivery_timed_out" else None)
    payload.fail_close = True
    now[0] = 23.0
    with pytest.raises((OSError, RuntimeError, ResultDeliveryTimeout)):
        {
            "delivered": result.take,
            "cancelled": result.cancel,
            "closed": result.close,
            "delivery_timed_out": result.take,
        }[outcome]()
    assert runtime.snapshot()["delivery_samples"] == 0
    assert result.timing_snapshot()["delivery_seconds"] is None
    payload.fail_close = False
    now[0] = 27.0
    result.close()
    result.close()
    runtime.close()
    assert result.timing_snapshot()["delivery_seconds"] == 7
    assert runtime.snapshot()["delivery_samples"] == 1
    assert runtime.snapshot()["delivery_seconds"] == 7


def test_preparation_failure_has_no_delivery_latency_sample(monkeypatch):
    now = clock(monkeypatch)
    runtime = registry()
    result = runtime.begin()
    Payload(result)
    now[0] += 10
    result.abort_preparation()
    result.close()
    assert result.timing_snapshot()["delivery_seconds"] is None
    assert runtime.snapshot()["delivery_samples"] == runtime.snapshot()["delivery_seconds"] == 0


@pytest.mark.parametrize("name", ["max_results", "max_bytes"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "10"])
def test_limits_require_positive_integer_capacity(name, value):
    kwargs = {"max_results": 2, "max_bytes": 4096, name: value}
    with pytest.raises(ValueError, match=name):
        ResultDeliveryLimits(**kwargs)


def test_result_capacity_is_reserved_before_preparation_and_returned_after_cleanup():
    runtime = registry(results=1)
    result = runtime.begin()
    with pytest.raises(ResultDeliveryFull):
        runtime.begin()
    with pytest.raises(RuntimeError, match="in progress"):
        runtime.close()
    assert result.state == "closing"
    assert runtime.snapshot()["active_results"] == 1
    result.abort_preparation()
    runtime.close()
    assert result.state == "closed"
    assert runtime.snapshot()["active_results"] == 0
    with pytest.raises(ResultDeliveryClosed):
        runtime.begin()


def test_empty_result_releases_its_slot_without_a_watcher(monkeypatch):
    def unexpected_start(self):
        pytest.fail("empty output started a deadline watcher")

    monkeypatch.setattr(MonotonicDeadline, "start", unexpected_start)
    runtime = registry()
    result = runtime.begin()
    result.ready(delivery_timeout=0)
    assert list(result) == [] and result.state == "delivered"
    assert runtime.snapshot()["delivered_results"] == 1
    runtime.close()


@pytest.mark.parametrize("operation", ["close", "cancel", "runtime"])
def test_cleanup_failures_retain_owners_and_capacity_for_retry(operation):
    runtime = registry(results=1)
    result = runtime.begin()
    payload = Payload(result)
    result.ready(delivery_timeout=None)
    payload.fail_close = True
    with pytest.raises(RuntimeError, match="cleanup failed"):
        (runtime.close if operation == "runtime" else result.cancel if operation == "cancel" else result.close)()
    snapshot = runtime.snapshot()
    assert snapshot["active_results"] == snapshot["cleanup_pending_results"] == 1
    assert snapshot["usage_bytes"] == 32
    assert result.state == "closing"
    payload.fail_close = False
    runtime.close()
    assert runtime.snapshot()["usage_bytes"] == runtime.snapshot()["active_results"] == 0


@pytest.mark.parametrize("view", ["table", "slice", "array", "buffer", "numpy"])
def test_exported_arrow_views_retain_capacity_after_handle_and_runtime_close(view):
    runtime = registry()
    result = runtime.begin()
    prepare_local_result(result, native(pa.table({"x": [1, 2, 3]})))
    result.ready(delivery_timeout=None)
    before = runtime.snapshot()["usage_bytes"]
    assert before > 24
    table = result.take()
    exported = {
        "table": lambda value: value,
        "slice": lambda value: value.slice(1),
        "array": lambda value: value.column(0).chunk(0),
        "buffer": lambda value: value.column(0).chunk(0).buffers()[1],
        "numpy": lambda value: value.column(0).chunk(0).to_numpy(zero_copy_only=True),
    }[view](table)
    del table
    result.close()
    runtime.close()
    assert result.state == "delivered"
    assert runtime.snapshot()["active_results"] == 0
    assert runtime.snapshot()["usage_bytes"] == runtime.snapshot()["exported_bytes"] == before
    if view == "numpy":
        assert exported.tolist() == [1, 2, 3]
    del exported
    gc.collect()
    assert runtime.snapshot()["usage_bytes"] == 0


def test_result_buffers_are_checked_before_allocation(monkeypatch):
    runtime = registry(size=1)
    result = runtime.begin()
    monkeypatch.setattr(
        local_result_delivery.pa, "allocate_buffer", lambda *a, **k: pytest.fail("over-budget allocation")
    )
    with pytest.raises(ResultDeliveryFull):
        prepare_local_result(result, native(pa.table({"x": [1]})))
    result.abort_preparation()
    assert runtime.snapshot()["usage_bytes"] == runtime.snapshot()["active_results"] == 0


def test_partial_preparation_failure_releases_already_encoded_partitions():
    runtime = registry(size=400)
    result = runtime.begin()
    with pytest.raises(ResultDeliveryFull):
        prepare_local_result(result, native(pa.table({"x": [1]}), pa.table({"x": [2]})))
    assert runtime.snapshot()["usage_bytes"] > 0
    result.abort_preparation()
    assert runtime.snapshot()["usage_bytes"] == 0


def test_consumed_views_block_new_buffers_without_holding_result_slots():
    runtime = registry(results=1, size=400)
    result = runtime.begin()
    prepare_local_result(result, native(pa.table({"x": [1]})))
    result.ready(delivery_timeout=None)
    table = result.take()
    second = runtime.begin()
    with pytest.raises(ResultDeliveryFull):
        prepare_local_result(second, native(pa.table({"x": [2]})))
    second.abort_preparation()
    del table
    third = runtime.begin()
    prepare_local_result(third, native(pa.table({"x": [3]})))
    third.ready(delivery_timeout=None)
    assert third.take().column(0).to_pylist() == [3]
    runtime.close()


def test_synchronous_expiry_fences_a_delayed_watcher_and_releases_output(monkeypatch):
    now = clock(monkeypatch)
    runtime = registry()
    result = runtime.begin()
    Payload(result)
    now[0] = 100  # Preparation is outside the ready-result delivery budget.
    result.ready(delivery_timeout=1)
    now[0] = 102
    with pytest.raises(ResultDeliveryTimeout):
        result.take()
    assert result.state == "delivery_timed_out"
    assert runtime.snapshot()["usage_bytes"] == runtime.snapshot()["active_results"] == 0
    assert runtime.snapshot()["timed_out_results"] == 1


def test_zero_delivery_timeout_discards_ready_result():
    runtime = registry()
    result = runtime.begin()
    Payload(result)
    with pytest.raises(ResultDeliveryTimeout):
        result.ready(delivery_timeout=0)
    result.abort_preparation()
    assert runtime.snapshot()["usage_bytes"] == runtime.snapshot()["active_results"] == 0


def test_abandoned_result_expires_without_a_consumer_and_releases_handle():
    runtime = registry()
    result = runtime.begin()
    Payload(result)
    reference = weakref.ref(result)
    result.ready(delivery_timeout=0.02)
    del result
    deadline = time.monotonic() + 3
    while runtime.snapshot()["active_results"]:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    gc.collect()
    assert reference() is None
    assert runtime.snapshot()["usage_bytes"] == 0


@pytest.mark.parametrize("operation", ["cancel", "close", "timeout"])
def test_cancellation_during_export_fences_delivery_and_retains_busy_owner(monkeypatch, operation):
    now = clock(monkeypatch)
    runtime = registry()
    result = runtime.begin()
    payload = Payload(result)
    entered, proceed = threading.Event(), threading.Event()

    def export(cancellation):
        entered.set()
        assert proceed.wait(5)
        return "must not escape"

    monkeypatch.setattr(payload, "export", export)
    result.ready(delivery_timeout=1)
    with ThreadPoolExecutor(max_workers=1) as threads:
        future = threads.submit(result.take)
        try:
            assert entered.wait(3)
            if operation == "timeout":
                now[0] = 12
                result._expire()
            else:
                with pytest.raises(RuntimeError, match="in progress"):
                    (result.cancel if operation == "cancel" else runtime.close)()
            assert result.state == "closing"
            assert runtime.snapshot()["active_results"] == 1
            assert runtime.snapshot()["usage_bytes"] == 32
        finally:
            proceed.set()
        error = {"cancel": ResultDeliveryCancelled, "close": ResultDeliveryClosed, "timeout": ResultDeliveryTimeout}[
            operation
        ]
        with pytest.raises(error):
            future.result(timeout=3)
    runtime.close()
    assert runtime.snapshot()["active_results"] == runtime.snapshot()["usage_bytes"] == 0


def test_concurrent_consumer_rejection_does_not_release_the_current_consumer(monkeypatch):
    runtime = registry()
    result = runtime.begin()
    payload = Payload(result)
    entered, proceed = threading.Event(), threading.Event()

    def export(cancellation):
        entered.set()
        assert proceed.wait(5)
        return "value"

    monkeypatch.setattr(payload, "export", export)
    result.ready(delivery_timeout=None)
    with ThreadPoolExecutor(max_workers=1) as threads:
        future = threads.submit(result.take)
        try:
            assert entered.wait(3)
            with pytest.raises(RuntimeError, match="concurrent"):
                result.take()
            assert runtime.snapshot()["usage_bytes"] == 32
            assert result._taking
        finally:
            proceed.set()
        assert future.result(timeout=3) == "value"
    runtime.close()


def test_copied_deadline_callback_cannot_cancel_a_delivered_result(monkeypatch):
    now = clock(monkeypatch)
    runtime = registry()
    result = runtime.begin()
    Payload(result)
    result.ready(delivery_timeout=1)
    callback = result._deadline._callback
    assert result.take() == "value"
    now[0] = 12
    callback()
    assert result.state == "delivered" and not result._cancellation.is_set()
    assert runtime.snapshot()["delivered_results"] == 1
    assert runtime.snapshot()["timed_out_results"] == 0


@pytest.mark.parametrize("timeout", [False, True])
def test_recorded_cancellation_fences_export_before_scope_dispatch(monkeypatch, timeout):
    now = clock(monkeypatch)
    runtime = registry()
    result = runtime.begin()
    payload = Payload(result)
    result.ready(delivery_timeout=1)
    recorded, dispatch = threading.Event(), threading.Event()
    original = result._cancellation.cancel

    def delayed_cancel(reason):
        recorded.set()
        assert dispatch.wait(5)
        return original(reason)

    monkeypatch.setattr(result._cancellation, "cancel", delayed_cancel)
    monkeypatch.setattr(payload, "export", lambda *a: pytest.fail("cancelled result was exported"))
    with ThreadPoolExecutor(max_workers=1) as threads:
        if timeout:
            now[0] = 12
        cancelling = threads.submit(result._expire if timeout else result.cancel)
        try:
            assert recorded.wait(3)
            assert not result._cancellation.is_set()
            with pytest.raises(ResultDeliveryTimeout if timeout else ResultDeliveryCancelled):
                result.take()
            assert runtime.snapshot()["usage_bytes"] == 32
            assert runtime.snapshot()["active_results"] == 1
        finally:
            dispatch.set()
        cancelling.result(timeout=3)
    assert runtime.snapshot()["usage_bytes"] == runtime.snapshot()["active_results"] == 0


def test_delayed_watcher_cannot_deliver_a_result_that_expires_during_export(monkeypatch):
    now = clock(monkeypatch)
    runtime = registry()
    result = runtime.begin()
    payload = Payload(result)

    def export(cancellation):
        now[0] = 12
        return "late value"

    monkeypatch.setattr(payload, "export", export)
    result.ready(delivery_timeout=1)
    with pytest.raises(ResultDeliveryTimeout):
        result.take()
    assert runtime.snapshot()["usage_bytes"] == runtime.snapshot()["active_results"] == 0


def test_slow_expiry_cleanup_does_not_delay_another_result_deadline(monkeypatch):
    runtime = registry()
    first, second = runtime.begin(), runtime.begin()
    slow, fast = Payload(first), Payload(second)
    entered, proceed = threading.Event(), threading.Event()
    original = slow.close

    def close():
        entered.set()
        assert proceed.wait(5)
        original()

    monkeypatch.setattr(slow, "close", close)
    first.ready(delivery_timeout=0.1)
    second.ready(delivery_timeout=0.1)
    try:
        assert entered.wait(3)
        deadline = time.monotonic() + 3
        while second.state != "delivery_timed_out":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert not fast.cleanup_pending() and slow.cleanup_pending()
        assert runtime.snapshot()["active_results"] == 1
    finally:
        proceed.set()
    deadline = time.monotonic() + 3
    while runtime.snapshot()["active_results"]:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    runtime.close()


def test_expiry_cleanup_failure_remains_retryable_without_retaining_exception(monkeypatch):
    now = clock(monkeypatch)
    runtime = registry()
    result = runtime.begin()
    payload = Payload(result)
    result.ready(delivery_timeout=1)
    payload.fail_close = True
    now[0] = 12
    result._expire()
    assert runtime.snapshot()["cleanup_pending_results"] == 1
    with pytest.raises(RuntimeError, match="cleanup failed"):
        runtime.close()
    payload.fail_close = False
    runtime.close()
    assert result.state == "delivery_timed_out"
    assert runtime.snapshot()["timed_out_results"] == 1


def test_shutdown_fences_every_result_before_first_cleanup_callback(monkeypatch):
    runtime = registry()
    first, second = runtime.begin(), runtime.begin()
    payload = Payload(first)
    Payload(second)
    first.ready(delivery_timeout=None)
    second.ready(delivery_timeout=None)
    entered, proceed = threading.Event(), threading.Event()
    original = payload.close

    def close():
        entered.set()
        assert proceed.wait(5)
        original()

    monkeypatch.setattr(payload, "close", close)
    with ThreadPoolExecutor(max_workers=1) as threads:
        closing = threads.submit(runtime.close)
        try:
            assert entered.wait(3)
            assert not second._cancellation.is_set()
            with pytest.raises(ResultDeliveryClosed):
                second.take()
            assert runtime.snapshot()["active_results"] == 2
        finally:
            proceed.set()
        closing.result(timeout=3)
    assert runtime.snapshot()["active_results"] == 0


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "1"])
@pytest.mark.parametrize("name", ["execution_timeout", "delivery_timeout"])
def test_invalid_timeouts_do_not_consume_a_request_or_result_slot(name, value):
    with LocalModelRuntime(
        session_id="validation",
        session_config={},
        request_limit=RequestAdmissionLimits(1, 1),
        result_limit=ResultDeliveryLimits(1, 4096),
    ) as models:
        request = models.request()
        with pytest.raises(ValueError, match=name):
            request.execute_result(None, {}, conn=None, **{name: value})
        assert request.state == "ready" and not request._used
        assert models.resource_snapshot()["result_delivery"]["active_results"] == 0


def test_managed_result_configuration_is_explicit():
    with pytest.raises(ValueError, match="request_limit"):
        LocalModelRuntime(session_id="validation", session_config={}, result_limit=ResultDeliveryLimits(1, 4096))
    with LocalModelRuntime(
        session_id="validation", session_config={}, request_limit=RequestAdmissionLimits(1, 1)
    ) as models:
        request = models.request()
        with pytest.raises(RuntimeError, match="result_limit"):
            request.execute_result(None, {}, conn=None)
        assert request.state == "ready" and not request._used


def test_failed_buffer_build_keeps_its_owner_until_cleanup(monkeypatch):
    def fail_build(self, table, size):
        self._buffer = pa.allocate_buffer(size)
        raise OSError("injected encoding failure")

    monkeypatch.setattr(local_result_delivery._ArrowResultPayload, "build", fail_build)
    runtime = registry()
    result = runtime.begin()
    with pytest.raises(OSError, match="encoding failure"):
        prepare_local_result(result, native(pa.table({"x": [1]})))
    assert runtime.snapshot()["usage_bytes"] > 0
    result.abort_preparation()
    assert runtime.snapshot()["usage_bytes"] == runtime.snapshot()["active_results"] == 0


def test_deadline_watcher_start_failure_can_discard_prepared_output(monkeypatch):
    def fail_start(self):
        raise OSError("watcher startup failed")

    monkeypatch.setattr(MonotonicDeadline, "start", fail_start)
    runtime = registry()
    result = runtime.begin()
    Payload(result)
    with pytest.raises(OSError, match="watcher startup failed"):
        result.ready(delivery_timeout=1)
    result.abort_preparation()
    assert runtime.snapshot()["usage_bytes"] == runtime.snapshot()["active_results"] == 0


def test_cleanup_error_after_confirmed_release_does_not_retain_the_slot(monkeypatch):
    runtime = registry()
    result = runtime.begin()
    payload = Payload(result)
    result.ready(delivery_timeout=None)
    original = payload.close

    def close():
        original()
        raise OSError("diagnostic failure after cleanup")

    monkeypatch.setattr(payload, "close", close)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        result.close()
    assert runtime.snapshot()["usage_bytes"] == runtime.snapshot()["active_results"] == 0
    assert result.state == "closed"


def test_cleanup_handoff_retries_when_consumer_and_cancellation_both_saw_busy(monkeypatch):
    runtime = registry()
    result = runtime.begin()
    payload = Payload(result)
    result.ready(delivery_timeout=None)
    exporting, recorded, finish_export, finish_cancel, dispatcher_done = (threading.Event() for _ in range(5))
    original_cancel, original_cleanup = result._cancellation.cancel, result._cleanup

    def export(cancellation):
        exporting.set()
        assert finish_export.wait(5)
        return "cancelled output"

    def cancel(reason):
        recorded.set()
        assert finish_cancel.wait(5)
        return original_cancel(reason)

    def cleanup(*, consumer=False):
        if consumer:
            try:
                return original_cleanup(consumer=True)
            except RuntimeError:
                finish_cancel.set()
                assert dispatcher_done.wait(5)
                raise
        try:
            return original_cleanup()
        finally:
            dispatcher_done.set()

    monkeypatch.setattr(payload, "export", export)
    monkeypatch.setattr(result._cancellation, "cancel", cancel)
    monkeypatch.setattr(result, "_cleanup", cleanup)
    with ThreadPoolExecutor(max_workers=2) as threads:
        taking = threads.submit(result.take)
        try:
            assert exporting.wait(3)
            cancelling = threads.submit(result.cancel)
            assert recorded.wait(3)
            finish_export.set()
            with pytest.raises(ResultDeliveryCancelled):
                taking.result(timeout=5)
            with pytest.raises(RuntimeError, match="in progress"):
                cancelling.result(timeout=5)
        finally:
            finish_export.set()
            finish_cancel.set()
    assert runtime.snapshot()["active_results"] == runtime.snapshot()["usage_bytes"] == 0
    assert result.state == "cancelled"


def test_native_completion_metadata_is_preserved():
    runtime = registry()
    result = runtime.begin()
    source = native(pa.table({"x": [1]}))
    source.completion_status = "status-from-native"
    source.stats = [10, 20]
    source.task_stats = {"task": "details"}
    prepare_local_result(result, source)
    result.ready(delivery_timeout=None)
    assert result.completion_status == source.completion_status
    assert result.stats == source.stats and result.task_stats == source.task_stats
    runtime.close()


def test_expiry_after_partial_handoff_retries_when_the_consumer_exits(monkeypatch):
    now = clock(monkeypatch)
    runtime = registry()
    result = runtime.begin()
    Payload(result, value="first")
    Payload(result, value="remaining")
    result.ready(delivery_timeout=1)
    handoff, finish = threading.Event(), threading.Event()
    original = result._cleanup

    def cleanup(*, consumer=False):
        original(consumer=consumer)
        if consumer:
            handoff.set()
            assert finish.wait(5)

    monkeypatch.setattr(result, "_cleanup", cleanup)
    with ThreadPoolExecutor(max_workers=1) as threads:
        taking = threads.submit(result.take)
        try:
            assert handoff.wait(3)
            now[0] = 12
            result._expire()
            assert result.state == "closing"
        finally:
            finish.set()
        assert taking.result(timeout=3) == "first"
    assert runtime.snapshot()["active_results"] == runtime.snapshot()["usage_bytes"] == 0
    assert result.state == "delivery_timed_out"


@pytest.mark.parametrize("boundary", ["export", "handoff"])
def test_deadline_crossed_between_poll_and_locked_decision_is_honored(monkeypatch, boundary):
    now = clock(monkeypatch)
    runtime = registry()
    result = runtime.begin()
    payload = Payload(result)
    result.ready(delivery_timeout=1)
    original = result._expire
    polls = []
    exports = []

    def delayed_poll():
        original()
        polls.append(None)
        if len(polls) == (1 if boundary == "export" else 2):
            # The caller is descheduled after polling but before taking the
            # lock that commits export or the final successful transfer.
            now[0] = 12

    def export(cancellation):
        exports.append(None)
        return "value"

    monkeypatch.setattr(result, "_expire", delayed_poll)
    monkeypatch.setattr(payload, "export", export)
    with pytest.raises(ResultDeliveryTimeout):
        result.take()
    assert len(exports) == int(boundary == "handoff")
    assert result.state == "delivery_timed_out"
    assert runtime.snapshot()["active_results"] == runtime.snapshot()["usage_bytes"] == 0


def test_iterator_exhaustion_releases_consumed_buffers_without_cyclic_gc():
    runtime = registry(results=1, size=400)
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(3):
            result = runtime.begin()
            prepare_local_result(result, native(pa.table({"x": [1, 2]})))
            result.ready(delivery_timeout=None)
            assert sum(table.num_rows for table in result) == 2
            assert runtime.snapshot()["usage_bytes"] == runtime.snapshot()["active_results"] == 0
    finally:
        runtime.close()
        if enabled:
            gc.enable()
        gc.collect()


def test_discarded_export_errors_do_not_retain_payloads_until_cyclic_gc():
    class Marker:
        pass

    def failed_export(cancellation):
        raise OSError("injected export failure")

    runtime = registry()
    result = runtime.begin()
    payload = Payload(result, value=Marker())
    reference = weakref.ref(payload.value)
    payload.export = failed_export
    result.ready(delivery_timeout=None)
    enabled = gc.isenabled()
    gc.disable()
    try:
        try:
            result.take()
        except OSError:
            pass
        else:
            pytest.fail("export error was swallowed")
        del payload
        assert reference() is None
        assert runtime.snapshot()["usage_bytes"] == runtime.snapshot()["active_results"] == 0
    finally:
        runtime.close()
        if enabled:
            gc.enable()
        gc.collect()


@pytest.mark.parametrize("operation", ["cancel", "drain", "timeout"])
def test_queued_request_termination_and_duplicate_calls_never_reserve_result_capacity(monkeypatch, operation):
    now = [10.0]
    monkeypatch.setattr(request_admission, "time", SimpleNamespace(monotonic=lambda: now[0]))
    with LocalModelRuntime(
        session_id="queued-delivery",
        session_config={},
        request_limit=RequestAdmissionLimits(1, 1),
        result_limit=ResultDeliveryLimits(1, 4096),
    ) as models:
        ready, queued = models.request(), models.request(queue_timeout=1)
        waiting = threading.Event()
        condition = models._request_admission._condition
        original_wait = condition.wait

        def wait(*args, **kwargs):
            waiting.set()
            return original_wait(*args, **kwargs)

        monkeypatch.setattr(condition, "wait", wait)
        with ThreadPoolExecutor(max_workers=1) as threads:
            future = threads.submit(queued.execute_result, None, {}, conn=None)
            try:
                assert waiting.wait(3)
                assert models.resource_snapshot()["result_delivery"]["active_results"] == 0
                with pytest.raises(RuntimeError, match="only execute once"):
                    queued.execute_result(None, {}, conn=None)
                if operation == "cancel":
                    assert queued.cancel()
                elif operation == "drain":
                    models.drain()
                else:
                    now[0] = 12.0
                    assert queued.state == "timed_out"
                with pytest.raises(RequestQueueTimeout if operation == "timeout" else RequestCancelled):
                    future.result(timeout=3)
                snapshot = models.resource_snapshot()["result_delivery"]
                assert snapshot["active_results"] == snapshot["failed_results"] == snapshot["usage_bytes"] == 0
            finally:
                queued.cancel()
                ready.cancel()
