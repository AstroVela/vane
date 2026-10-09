# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import pytest

from vane.execution.local_stream_adapter import LocalStreamAdapter
from vane.execution.udf_lifecycle import ExecutionCancellationScope, ExecutionCancelledError
from vane.execution.udf_stream_backpressure import StreamCapacity, StreamReadWindow


@pytest.mark.parametrize("capacity", [dict(rows=0), dict(rows=1, bytes=0), dict(rows=1, item_bytes=0)])
def test_zero_downstream_capacity_stops_reading_even_an_empty_block(capacity):
    window = StreamReadWindow(StreamCapacity.parse(capacity))
    assert not window.may_read()
    assert not window.take(0)


def test_one_oversized_block_makes_progress_without_reading_ahead():
    window = StreamReadWindow(StreamCapacity(rows=4, bytes=100, item_bytes=100))
    assert window.take(150)
    assert not window.may_read()
    assert not window.take(1)
    assert StreamReadWindow(StreamCapacity(rows=1, bytes=100, item_bytes=100)).take(1)


def test_inflight_read_reserves_downstream_event_capacity():
    window = StreamReadWindow(StreamCapacity(rows=1, bytes=100, item_bytes=100))
    assert window.may_read(pending=0)
    assert not window.may_read(pending=1)


def test_local_stream_pauses_iterator_until_consumer_reads():
    stream = LocalStreamAdapter()
    scope = ExecutionCancellationScope("window", 1)
    advanced = []
    blocked = threading.Event()

    def wait_context():
        blocked.set()
        return nullcontext()

    def produce():
        for value in range(3):
            stream.reserve(scope, wait_context)
            advanced.append(value)

    with ThreadPoolExecutor(max_workers=1) as threads:
        producer = threads.submit(produce)
        try:
            assert blocked.wait(timeout=5)
            assert advanced == [0, 1]
            stream.consumed()
            producer.result(timeout=5)
            assert advanced == [0, 1, 2]
        finally:
            scope.cancel()
            stream.close()


def test_cancellation_wakes_full_stream_without_a_consumer():
    stream = LocalStreamAdapter()
    scope = ExecutionCancellationScope("cancel", 1)
    for _ in range(2):
        stream.reserve(scope, nullcontext)
    waiting = threading.Event()

    def wait_context():
        waiting.set()
        return nullcontext()

    with ThreadPoolExecutor(max_workers=1) as threads:
        producer = threads.submit(stream.reserve, scope, wait_context)
        try:
            assert waiting.wait(timeout=5)
            scope.cancel("consumer stopped")
            with pytest.raises(ExecutionCancelledError, match="consumer stopped"):
                producer.result(timeout=5)
        finally:
            stream.close()
