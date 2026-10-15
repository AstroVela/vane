# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""FTE completion must include task registrations concurrent with status reads."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from vane.runners.fte.fte_events import SplitEventsSubmitted
from vane.runners.fte.fte_scheduler import FteEventDrivenTaskSource, FteEventHandlers
from vane.runners.ray.fragment_registry import (
    _FTE_FRAGMENT_EXECUTIONS,
    _FTE_REGISTRY_LOCK,
    _FTE_SCHEDULERS,
)
from vane.runners.ray.fragment_worker_results import fte_query_status


@pytest.fixture
def completion_query():
    query_id = "query-completion-snapshot"
    scheduler = _FTE_SCHEDULERS.get_or_create(query_id)

    def fragment(index):
        return SimpleNamespace(
            task_context_info={"query_idx": 1, "last_node_id": 2, "task_id": index, "node_ids": [2]},
            query_status_snapshot=lambda: {
                "failed": False,
                "no_more_partitions": True,
                "partitions": [
                    {
                        "partition_id": 0,
                        "running": False,
                        "failed": False,
                        "finished": True,
                        "selected_attempt": 0,
                        "task_id": f"{query_id}.{index}.0",
                        "failures": [],
                    }
                ],
            },
        )

    first, second = fragment(0), fragment(1)

    def publish_second(_events):
        with _FTE_REGISTRY_LOCK:
            _FTE_FRAGMENT_EXECUTIONS[(query_id, "second")] = second

    scheduler.set_handlers(FteEventHandlers(on_split_events=publish_second))
    with _FTE_REGISTRY_LOCK:
        _FTE_FRAGMENT_EXECUTIONS[(query_id, "first")] = first
    try:
        yield SimpleNamespace(
            query_id=query_id,
            scheduler=scheduler,
            first=first,
            second=second,
            publish_second=publish_second,
            event=SplitEventsSubmitted.from_events(query_id, [{"fragment_id": "second"}]),
        )
    finally:
        _FTE_SCHEDULERS.drop_query(query_id)
        with _FTE_REGISTRY_LOCK:
            _FTE_FRAGMENT_EXECUTIONS.pop((query_id, "first"), None)
            _FTE_FRAGMENT_EXECUTIONS.pop((query_id, "second"), None)


def _status(query, scoped):
    contexts = [query.first.task_context_info, query.second.task_context_info] if scoped else None
    return fte_query_status(query.query_id, contexts)


def _assert_complete(query, scoped):
    status = _status(query, scoped)
    assert status["finished"] is True
    assert status["registration_pending"] is False
    assert status["selected_attempt_task_ids"] == [f"{query.query_id}.{index}.0.0" for index in (0, 1)]


@pytest.mark.parametrize("scoped", [False, True])
def test_completion_waits_for_queued_registration(completion_query, scoped):
    query = completion_query
    query.scheduler.enqueue(query.event)
    status = _status(query, scoped)
    assert status["matched"] is True
    assert status["finished"] is False
    assert status["registration_pending"] is True
    query.scheduler.drain()
    _assert_complete(query, scoped)


@pytest.mark.parametrize("scoped", [False, True])
def test_completion_waits_for_executing_registration(completion_query, scoped):
    query = completion_query
    entered = threading.Event()
    release = threading.Event()

    def publish_after_release(events):
        entered.set()
        assert release.wait(5), "registration was not released"
        query.publish_second(events)

    query.scheduler.set_handlers(FteEventHandlers(on_split_events=publish_after_release))
    query.scheduler.enqueue(query.event)
    with ThreadPoolExecutor(max_workers=1) as executor:
        draining = executor.submit(query.scheduler.drain)
        try:
            assert entered.wait(5), "registration did not start"
            assert query.scheduler.stats().queued_events == 0
            status = _status(query, scoped)
            assert status["finished"] is False
            assert status["registration_pending"] is True
        finally:
            release.set()
            draining.result(timeout=5)
    _assert_complete(query, scoped)


@pytest.mark.parametrize("scoped", [False, True])
def test_completion_rechecks_registration_during_snapshot(completion_query, scoped):
    query = completion_query
    original_snapshot = query.first.query_status_snapshot

    def snapshot_with_registration():
        query.first.query_status_snapshot = original_snapshot
        query.scheduler.enqueue(query.event)
        query.scheduler.drain()
        return original_snapshot()

    query.first.query_status_snapshot = snapshot_with_registration
    status = _status(query, scoped)
    assert status["finished"] is False
    assert status["registration_pending"] is True
    _assert_complete(query, scoped)


@pytest.mark.parametrize("scoped", [False, True])
def test_completion_waits_for_task_source_to_submit(completion_query, scoped):
    query = completion_query
    source = FteEventDrivenTaskSource(query.scheduler, "later-fragment")
    try:
        status = _status(query, scoped)
        assert status["finished"] is False
        assert status["registration_pending"] is True
        source.submit([query.event])
    finally:
        source.close()
    _assert_complete(query, scoped)


def test_completion_rechecks_scheduler_replacement(completion_query):
    query = completion_query
    original_snapshot = query.first.query_status_snapshot

    def snapshot_with_replacement():
        query.first.query_status_snapshot = original_snapshot
        _FTE_SCHEDULERS.drop_query(query.query_id)
        _FTE_SCHEDULERS.get_or_create(query.query_id)
        query.publish_second([])
        return original_snapshot()

    query.first.query_status_snapshot = snapshot_with_replacement
    status = _status(query, True)
    assert status["finished"] is False
    assert status["registration_pending"] is True
    _assert_complete(query, True)
