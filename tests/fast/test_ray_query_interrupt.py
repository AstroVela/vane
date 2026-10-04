# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError

import pytest

import vane
from vane._query_interrupt import QueryResultIterator, check_query_interrupted, has_query_interrupt_check
from vane.runners.ray import safe_get


class _Ref:
    def __init__(self, future):
        self._future = future

    def future(self):
        return self._future


def test_result_interrupt_preserves_cleanup_and_restores_calling_context():
    waiting = threading.Event()
    cancelled = threading.Event()
    finished = threading.Event()
    cleanup = []
    errors = []

    class WaitingFuture(Future):
        def result(self, timeout=None):
            waiting.set()
            return super().result(timeout)

    pending = WaitingFuture()
    cleanup_result = Future()
    cleanup_result.set_result("released")

    def check():
        if cancelled.is_set():
            raise vane.InterruptException("query interrupted")

    def results():
        try:
            yield safe_get.resolve_object_refs_blocking(_Ref(pending))
        finally:
            cleanup.append(safe_get.resolve_object_refs_blocking(_Ref(cleanup_result), honor_query_interrupt=False))

    iterator = QueryResultIterator(results(), check)

    def consume():
        try:
            next(iterator)
        except BaseException as error:
            errors.append(error)
        finally:
            cleanup.append(has_query_interrupt_check())
            finished.set()

    worker = threading.Thread(target=consume)
    worker.start()
    try:
        assert waiting.wait(5)
        cancelled.set()
        assert finished.wait(5)
        assert not pending.done()
        assert len(errors) == 1 and isinstance(errors[0], vane.InterruptException)
        assert cleanup == ["released", False]
        assert not has_query_interrupt_check()
    finally:
        pending.set_result(None)
        worker.join(5)
        assert not worker.is_alive()
        iterator.close()


def test_interrupt_polling_preserves_progress_interval(monkeypatch):
    clock = [0.0]
    progress = []

    class PendingFuture:
        def result(self, timeout=None):
            assert timeout is not None and timeout <= 0.1
            clock[0] += timeout
            raise FutureTimeoutError

        def done(self):
            return False

    monkeypatch.setattr(safe_get.time, "monotonic", lambda: clock[0])

    def check():
        if clock[0] >= 0.65:
            raise vane.InterruptException("query interrupted")

    def results():
        yield safe_get.resolve_object_refs_blocking(
            _Ref(PendingFuture()), on_wait=lambda: progress.append(clock[0]), wait_interval_s=0.5
        )

    with pytest.raises(vane.InterruptException):
        next(QueryResultIterator(results(), check))
    assert progress == [pytest.approx(0.5)]
    assert not has_query_interrupt_check()


def test_result_interrupt_context_is_restored_between_yields_and_nested_queries():
    cancelled = False

    def inner_results():
        assert has_query_interrupt_check()
        yield 7
        yield 8

    def check_outer():
        if cancelled:
            raise vane.InterruptException("outer query interrupted")

    def outer_results():
        nonlocal cancelled
        inner = QueryResultIterator(inner_results(), lambda: None)
        try:
            yield next(inner)
            assert next(inner) == 8
            cancelled = True
            check_query_interrupted()
            pytest.fail("nested iteration lost the outer cancellation context")
        finally:
            inner.close()

    outer = QueryResultIterator(outer_results(), check_outer)
    assert next(outer) == 7
    assert not has_query_interrupt_check()
    with pytest.raises(vane.InterruptException, match="outer query interrupted"):
        next(outer)
    assert not has_query_interrupt_check()
