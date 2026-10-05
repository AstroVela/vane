# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Small capacity, FIFO fairness and reservation cleanup across execution modes."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from vane.execution.worker_resources import WorkerResourceManager


def test_graph_waits_atomically_and_cannot_be_overtaken_by_fte_attempts():
    manager = WorkerResourceManager({"contexts": 2}, 2)
    assert manager.try_acquire("fte0", "large", {0: {"contexts": 1}})
    assert manager.try_acquire("fte1", "large", {1: {"contexts": 1}})
    assert not manager.try_acquire("pipe", "small", {0: {"contexts": 2}, 1: {"contexts": 2}})
    manager.release("fte0")
    assert not manager.try_acquire("fte2", "large", {0: {"contexts": 1}})
    # The graph takes no partial reservation while waiting for its second worker.
    assert manager.snapshot()["used"][0]["contexts"] == 0
    manager.release("fte1")
    assert manager.try_acquire("pipe", "small", {0: {"contexts": 2}, 1: {"contexts": 2}})
    manager.release("pipe")
    assert manager.try_acquire("fte2", "large", {0: {"contexts": 1}})
    manager.release("fte2")
    assert manager.snapshot()["reservations"] == {}
    assert manager.snapshot()["waiting"] == []


def test_cancel_waiter_does_not_release_live_owners():
    manager = WorkerResourceManager({"bytes": 10}, 1)
    assert manager.try_acquire("live", "live", {0: {"bytes": 10}})
    assert not manager.try_acquire("wait", "canceled", {0: {"bytes": 10}})
    assert not manager.try_acquire("after", "next", {0: {"bytes": 1}})
    manager.cancel_waiting("canceled")
    assert list(manager.snapshot()["reservations"]) == ["live"]
    manager.release("live")
    assert manager.try_acquire("after", "next", {0: {"bytes": 1}})


def test_capacity_failure_has_no_partial_reservation():
    manager = WorkerResourceManager({"contexts": 2, "bytes": 10}, 2)
    with pytest.raises(ValueError, match="exceeds worker 1 bytes"):
        manager.try_acquire("bad", "query", {0: {"contexts": 1, "bytes": 10}, 1: {"contexts": 1, "bytes": 11}})
    assert manager.snapshot()["reservations"] == {}
    assert manager.snapshot()["waiting"] == []


def test_blocking_wait_is_cancelable_and_releases_queue_position():
    manager = WorkerResourceManager({"contexts": 1}, 1)
    manager.try_acquire("busy", "first", {0: {"contexts": 1}})
    checked, cancel = threading.Event(), threading.Event()

    def check():
        checked.set()
        if cancel.is_set():
            raise RuntimeError("canceled")

    with ThreadPoolExecutor() as pool:
        waiting = pool.submit(manager.acquire, "wait", "second", {0: {"contexts": 1}}, check)
        assert checked.wait(2)
        cancel.set()
        with pytest.raises(RuntimeError, match="canceled"):
            waiting.result(2)
    assert manager.snapshot()["waiting"] == []
    assert list(manager.snapshot()["reservations"]) == ["busy"]


def test_close_wakes_waiters_and_tokens_cannot_change_demand():
    manager = WorkerResourceManager({"contexts": 2}, 1)
    assert manager.try_acquire("a", "a", {0: {"contexts": 1}})
    with pytest.raises(ValueError, match="token reused"):
        manager.try_acquire("a", "a", {0: {"contexts": 2}})
    manager.close()
    with pytest.raises(RuntimeError, match="closed"):
        manager.try_acquire("b", "b", {0: {"contexts": 1}})
