# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import pickle

import pytest

from vane.execution.request_admission import RequestAdmissionLimits, RequestCancelled
from vane.execution.result_delivery import ResultDeliveryFull, ResultDeliveryLimits, RuntimeResultDelivery
from vane.execution.udf_local_model import LocalModelRuntime


def runtime():
    return LocalModelRuntime(
        session_id="capacity-errors",
        session_config={},
        request_limit=RequestAdmissionLimits(1, 1),
        result_limit=ResultDeliveryLimits(1, 1),
    )


@pytest.mark.parametrize("args", [(), ("legacy refusal",), ("legacy refusal", 7)])
def test_legacy_capacity_errors_preserve_exception_and_serialization_contract(args):
    error = ResultDeliveryFull(*args)
    for restored in (error, copy.copy(error), pickle.loads(pickle.dumps(error))):
        assert isinstance(restored, RuntimeError)
        assert restored.args == args
        assert str(restored) == str(RuntimeError(*args))
        assert restored.reason is restored.requested is restored.used is restored.limit is None
        assert restored.execution_started is None


def test_standalone_capacity_snapshots_do_not_infer_request_execution():
    delivery = RuntimeResultDelivery(ResultDeliveryLimits(1, 10))
    result = delivery.begin()
    owner = result.own_buffer(8)
    try:
        with pytest.raises(ResultDeliveryFull) as slots:
            delivery.begin()
        with pytest.raises(ResultDeliveryFull) as bytes_:
            result.own_buffer(3)
        assert (slots.value.reason, slots.value.requested, slots.value.used, slots.value.limit) == ("slots", 1, 1, 1)
        assert (bytes_.value.reason, bytes_.value.requested, bytes_.value.used, bytes_.value.limit) == (
            "bytes",
            3,
            8,
            10,
        )
        assert slots.value.execution_started is bytes_.value.execution_started is None
    finally:
        owner.release()
        result.abort_preparation()
        delivery.close()
    # These are scalar snapshots of the refusal, not views of current usage.
    assert slots.value.used == 1 and bytes_.value.used == 8


def test_managed_slot_refusal_can_retry_without_running_the_operation():
    with runtime() as models:
        occupied = models._result_delivery.begin()
        request = models.request()
        calls = []
        try:
            with pytest.raises(ResultDeliveryFull) as caught:
                request._run_managed_result(lambda: calls.append(True), lambda result, value: None)
            assert caught.value.reason == "slots"
            assert caught.value.execution_started is False
            assert request.state == "ready" and not calls
        finally:
            occupied.abort_preparation()
        with request._run_managed_result(lambda: calls.append(True), lambda result, value: None):
            pass
        assert calls == [True]
        assert request.state == "finished"
        for restored in (copy.copy(caught.value), pickle.loads(pickle.dumps(caught.value))):
            assert (restored.reason, restored.requested, restored.used, restored.limit) == ("slots", 1, 1, 1)
            assert restored.execution_started is False
            assert restored.args == caught.value.args


def test_unknown_before_claim_failure_has_no_retry_guarantee(monkeypatch):
    with runtime() as models:
        request = models.request()
        original = ResultDeliveryFull("runtime result slots are full")

        def refuse():
            raise original

        monkeypatch.setattr(models._result_delivery, "begin", refuse)
        with pytest.raises(ResultDeliveryFull) as caught:
            request._run_managed_result(lambda: pytest.fail("unadmitted execution"), lambda result, value: None)
        assert caught.value is original
        assert caught.value.reason is caught.value.execution_started is None


@pytest.mark.parametrize("stage", ["operation", "encoding"])
def test_outer_execution_overrides_a_nested_pre_execution_refusal(stage):
    with runtime() as outer, runtime() as inner:
        occupied = inner._result_delivery.begin()
        nested = inner.request()

        def refuse():
            nested._run_managed_result(lambda: pytest.fail("nested execution"), lambda result, value: None)

        try:
            request = outer.request()
            with pytest.raises(ResultDeliveryFull) as caught:
                request._run_managed_result(
                    refuse if stage == "operation" else lambda: None,
                    (lambda result, value: refuse()) if stage == "encoding" else lambda result, value: None,
                )
            assert caught.value.reason == "slots"
            assert caught.value.execution_started is True
            assert request.state == "finished" and nested.state == "ready"
            assert outer.resource_snapshot()["result_delivery"]["active_results"] == 0
        finally:
            occupied.abort_preparation()


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_byte_refusal_preserves_execution_state_and_pending_cleanup(cleanup_fails):
    class Payload:
        def __init__(self, result):
            self.owner = result.own_buffer(1)
            self.fail = cleanup_fails
            result.hold(self)

        def close(self):
            if self.fail:
                raise OSError("planned output cleanup failure")
            self.owner.release()

        def cleanup_pending(self):
            return self.fail

    with runtime() as models:
        request = models.request()
        payloads = []
        calls = []

        def prepare(result, value):
            payloads.append(Payload(result))
            result.own_buffer(1)

        try:
            with pytest.raises(ResultDeliveryFull) as caught:
                request._run_managed_result(lambda: calls.append(True), prepare)
            assert caught.value.execution_started is True
            assert (caught.value.reason, caught.value.requested, caught.value.used, caught.value.limit) == (
                "bytes",
                1,
                1,
                1,
            )
            assert calls == [True] and request.state == "finished"
            snapshot = models.resource_snapshot()
            assert snapshot["request_admission"]["active_requests"] == 0
            assert snapshot["result_delivery"]["usage_bytes"] == int(cleanup_fails)
            assert snapshot["result_delivery"]["active_results"] == int(cleanup_fails)
            if cleanup_fails:
                assert "result cleanup failed" in str(caught.value.__cause__)
            with pytest.raises(RuntimeError, match="only execute once"):
                request._run_managed_result(lambda: calls.append(True), prepare)
            assert calls == [True]
            restored = pickle.loads(pickle.dumps(caught.value))
            assert restored.reason == "bytes" and restored.execution_started is True
        finally:
            for payload in payloads:
                payload.fail = False
        models.close()
        assert models.resource_snapshot()["result_delivery"]["usage_bytes"] == 0


def test_cancellation_keeps_precedence_over_a_capacity_error_after_claim():
    with runtime() as models:
        request = models.request()

        def operation():
            assert request.cancel()
            raise ResultDeliveryFull("user exception mentioning slots")

        with pytest.raises(RequestCancelled):
            request._run_managed_result(operation, lambda result, value: None)
        snapshot = models.resource_snapshot()
        assert snapshot["request_admission"]["cancelled_requests"] == 1
        assert snapshot["result_delivery"]["active_results"] == 0
