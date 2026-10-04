# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Readiness, cancellation and resource bounds at the Python source boundary."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pyarrow as pa
import pytest

import vane
from vane.datasource import DataSource, DataSourceTask, read_datasource
from vane.datasource._iterator import _DataSourceIterator, _DataSourceWait
from vane.datasource._video_admission import _DecodeAdmission, _MemoryAdmission


def _admitted(permit):
    ready = threading.Event()
    assert permit.subscribe(ready.set) or ready.wait(5)
    permit.check_admitted()


def test_decoder_admission_is_bounded_fifo_and_cancellable():
    admission = _DecodeAdmission(1, lambda: True, 0.01)
    first, cancelled, second, third = [admission.request() for _ in range(4)]
    try:
        _admitted(first)
        cancelled_ready, second_ready, third_ready = [threading.Event() for _ in range(3)]
        assert not cancelled.subscribe(cancelled_ready.set)
        assert not second.subscribe(second_ready.set)
        assert not third.subscribe(third_ready.set)
        cancelled.close()
        cancelled.close()
        first.close()
        assert second_ready.wait(5)
        second.check_admitted()
        assert not cancelled_ready.is_set()
        assert not third_ready.is_set()
        # An already-released permit cannot free the next decoder's slot.
        first.close()
        assert not third.subscribe(third_ready.set)
        second.close()
        assert third_ready.wait(5)
        third.check_admitted()
    finally:
        for permit in (first, cancelled, second, third):
            permit.close()


def test_decoder_admission_honors_capacity_greater_than_one():
    admission = _DecodeAdmission(2, lambda: True, 0.01)
    first, second, third = [admission.request() for _ in range(3)]
    try:
        _admitted(first)
        _admitted(second)
        ready = threading.Event()
        assert not third.subscribe(ready.set)
        second.close()
        assert ready.wait(5)
        third.check_admitted()
        first.check_admitted()
    finally:
        for permit in (first, second, third):
            permit.close()


def test_memory_pressure_does_not_block_request_or_subscription():
    sampled, allow_sample = threading.Event(), threading.Event()
    sampling_threads = []

    def memory_ready():
        sampling_threads.append(threading.get_ident())
        sampled.set()
        assert allow_sample.wait(5)
        return True

    admission = _DecodeAdmission(1, memory_ready, 0.01)
    permit = admission.request()
    try:
        assert sampled.wait(5)
        ready = threading.Event()
        assert not permit.subscribe(ready.set)
        # Cancellation must not wait for the memory probe either.
        permit.close()
        allow_sample.set()
        next_permit = admission.request()
        try:
            _admitted(next_permit)
            assert not ready.is_set()
            assert all(ident != threading.get_ident() for ident in sampling_threads)
        finally:
            next_permit.close()
    finally:
        allow_sample.set()
        permit.close()


def test_memory_recovery_wakes_an_already_subscribed_request():
    sampled, recovered = threading.Event(), threading.Event()

    def memory_ready():
        sampled.set()
        return recovered.is_set()

    admission = _DecodeAdmission(1, memory_ready, 0.01)
    permit = admission.request()
    try:
        assert sampled.wait(5)
        ready = threading.Event()
        assert not permit.subscribe(ready.set)
        assert not ready.is_set()
        recovered.set()
        assert ready.wait(5)
        permit.check_admitted()
        assert permit.subscribe(lambda: pytest.fail("ready permits need no callback"))
    finally:
        permit.close()


def test_memory_probe_failure_reaches_reader_without_consuming_a_slot():
    def memory_ready():
        raise OSError("memory probe failed")

    admission = _DecodeAdmission(1, memory_ready, 0.01)
    for _ in range(2):
        permit = admission.request()
        try:
            ready = threading.Event()
            assert permit.subscribe(ready.set) or ready.wait(5)
            with pytest.raises(OSError, match="memory probe failed"):
                permit.check_admitted()
        finally:
            permit.close()


def test_memory_admission_preserves_hysteresis_and_available_memory_override():
    memory = SimpleNamespace(available=0, percent=81.0)
    gate = _MemoryAdmission(lambda: memory, 4096, high=80, low=70)
    assert not gate()
    memory.percent = 75.0
    assert not gate()
    memory.percent = 69.0
    assert gate()
    memory.percent = 75.0
    assert gate()
    memory.percent = 99.0
    memory.available = 4096
    assert gate()
    memory.available = 4095
    assert not gate()


class _ManualWait(_DataSourceWait):
    def __init__(self, *, ready=False, wake_during_subscribe=False):
        self.ready = ready
        self.wake_during_subscribe = wake_during_subscribe
        self.closed = False
        self.wakeup = None

    def subscribe(self, wakeup):
        if self.ready:
            return True
        self.wakeup = wakeup
        if self.wake_during_subscribe:
            self.complete()
        return False

    def complete(self):
        self.ready = True
        wakeup, self.wakeup = self.wakeup, None
        if wakeup is not None:
            wakeup()

    def close(self):
        self.closed = True
        self.wakeup = None


@pytest.mark.parametrize("early_wakeup", [False, True])
def test_source_wait_never_enters_arrow_stream_and_cannot_lose_early_wakeup(early_wakeup):
    wait = _ManualWait(wake_during_subscribe=early_wakeup)
    batches = [pa.record_batch({"id": []}), pa.record_batch({"id": [17, 19]})]

    def source():
        yield wait
        yield from batches

    iterator = _DataSourceIterator(source())
    wakeup = threading.Event()
    try:
        assert not iterator.poll(wakeup.set)
        if not early_wakeup:
            assert not wakeup.is_set()
            # Resubscription replaces the old execution epoch's callback.
            assert not iterator.poll(wakeup.set)
            wait.complete()
        assert wakeup.is_set()
        for batch in batches:
            assert iterator.poll(wakeup.set)
            assert iterator.poll(wakeup.set)  # poll must not consume a staged batch
            assert next(iterator) is batch
        assert iterator.poll(wakeup.set)
        with pytest.raises(StopIteration):
            next(iterator)
    finally:
        iterator.close()


def test_source_wait_ready_before_subscription_needs_no_wakeup():
    wait = _ManualWait(ready=True)
    batch = pa.record_batch({"id": [2]})
    iterator = _DataSourceIterator(iter([wait, batch]))
    try:
        assert iterator.poll(lambda: pytest.fail("already ready"))
        assert next(iterator) is batch
    finally:
        iterator.close()


def test_closing_a_pending_source_cancels_wait_and_closes_generator():
    wait = _ManualWait()
    closed = threading.Event()

    def source():
        try:
            yield wait
            pytest.fail("closed source resumed")
        finally:
            closed.set()

    iterator = _DataSourceIterator(source())
    assert not iterator.poll(lambda: pytest.fail("cancelled source woke up"))
    iterator.close()
    iterator.close()
    assert wait.closed
    assert closed.is_set()
    wait.complete()
    assert iterator.poll(lambda: None)
    with pytest.raises(StopIteration):
        next(iterator)


def test_error_after_source_wait_keeps_original_exception():
    error = ValueError("source failure after admission")

    def source():
        yield _ManualWait(ready=True)
        raise error

    iterator = _DataSourceIterator(source())
    try:
        assert iterator.poll(lambda: None)
        with pytest.raises(ValueError, match="source failure after admission") as caught:
            next(iterator)
        assert caught.value is error
    finally:
        iterator.close()


class _WakeDuringPollTask(DataSourceTask):
    def __init__(self, offset):
        self.offset = offset

    def execute(self):
        raise AssertionError("requires native readiness polling")

    def _execute_with_context(self, execution_context):
        for number in range(16):
            execution_context._check_interrupted()
            yield _ManualWait(wake_during_subscribe=True)
            yield pa.record_batch({"id": [self.offset + number]})


class _WakeDuringPollSource(DataSource):
    @property
    def schema(self):
        return {"id": "BIGINT"}

    def get_tasks(self):
        return [_WakeDuringPollTask(offset) for offset in range(0, 128, 16)]


def test_native_scan_cannot_lose_callback_before_blocking(duckdb_cursor):
    # Every callback runs synchronously before the scan returns BLOCKED.
    rows = read_datasource(_WakeDuringPollSource(), con=duckdb_cursor).fetchall()
    assert sorted(rows) == [(number,) for number in range(128)]


_peer_blocked = threading.Event()
_peer_closed = threading.Event()


class _NeverReadyWait(_DataSourceWait):
    def subscribe(self, wakeup):
        _peer_blocked.set()
        return False

    def close(self):
        _peer_closed.set()


class _EarlyFinishTask(DataSourceTask):
    def __init__(self, emit):
        self.emit = emit

    def execute(self):
        raise AssertionError("requires native readiness polling")

    def _execute_with_context(self, execution_context):
        if self.emit:
            # A bounded test barrier ensures another pipeline task has entered
            # admission before this row makes LIMIT finish the entire pipeline.
            assert _peer_blocked.wait(5)
            yield pa.record_batch({"id": [7]})
        else:
            yield _NeverReadyWait()
            raise AssertionError("an unready source must be closed, never resumed")


class _EarlyFinishSource(DataSource):
    @property
    def schema(self):
        return {"id": "BIGINT"}

    def get_tasks(self):
        return [_EarlyFinishTask(True), _EarlyFinishTask(False)]


def test_native_early_finish_wakes_sources_that_never_become_ready():
    _peer_blocked.clear()
    _peer_closed.clear()
    connection = vane.connect(config={"threads": 2, "preserve_insertion_order": False})
    rows, errors = [], []

    def fetch():
        try:
            rows.extend(read_datasource(_EarlyFinishSource(), con=connection).limit(1).fetchall())
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=fetch)
    try:
        worker.start()
        worker.join(10)
        assert not worker.is_alive(), "early completion failed to wake pending admission"
        assert not errors
        assert rows == [(7,)]
        assert _peer_closed.is_set()
    finally:
        connection.interrupt()
        worker.join(5)
        connection.close()
