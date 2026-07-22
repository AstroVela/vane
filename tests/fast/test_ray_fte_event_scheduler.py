# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time

import pytest

from vane.runners.ray.fte import FteTaskAttemptId, FteTaskId
from vane.runners.ray.fte_events import (
    MemoryPressureDetected,
    QueryAbort,
    ResourceAdmissionChanged,
    RetryDelayExpired,
    SourceInputExhausted,
    SplitEventsSubmitted,
    TaskStatusChanged,
    WorkerFailed,
    WorkerReservationCompleted,
)
from vane.runners.ray.fte_scheduler import (
    FteAttemptStatusWatcher,
    FteEventDrivenTaskSource,
    FteEventHandlers,
    FteQueryScheduler,
    FteSchedulerRegistry,
)
from vane.runners.ray.safe_get import QueryDeadlineExceeded


def test_event_scheduler_dispatches_events_in_order():
    registry = FteSchedulerRegistry()
    scheduler = registry.get_or_create("query-event")
    calls = []

    scheduler.enqueue(
        SplitEventsSubmitted.from_events(
            "query-event",
            [
                {"query_id": "query-event", "fragment_id": "f0", "value": 1},
                {"query_id": "query-event", "fragment_id": "f0", "value": 2},
            ],
        )
    )
    scheduler.enqueue(SourceInputExhausted.from_source_node_ids("query-event", {"7", 8}))
    scheduler.enqueue(
        TaskStatusChanged.from_status(
            "query-event",
            FteTaskAttemptId(FteTaskId("query-event", 0, 3), 1),
            {"state": "FINISHED"},
        )
    )
    scheduler.enqueue(WorkerFailed("query-event", "worker-a", "lost"))
    scheduler.enqueue(MemoryPressureDetected("query-event", 2))
    scheduler.enqueue(ResourceAdmissionChanged("query-event"))
    scheduler.enqueue(WorkerReservationCompleted("query-event", 0, "f0", 3, 1, "worker-b"))

    outputs = scheduler.drain(
        FteEventHandlers(
            on_split_events=lambda events: calls.append(("split", [event["value"] for event in events])) or ["h0"],
            on_source_input_exhausted=lambda source_ids: calls.append(("exhausted", sorted(source_ids))) or ["h1"],
            on_task_status_changed=lambda event: calls.append(("status", str(event.attempt_id))) or ["h2"],
            on_worker_failed=lambda event: calls.append(("worker", event.worker_id, event.error)) or ["h3"],
            on_memory_pressure_detected=lambda event: calls.append(("memory", event.max_count_per_worker)) or ["h4"],
            on_resource_admission_changed=lambda event: calls.append(("resource", event.query_id)) or ["h5"],
            on_worker_reservation_completed=lambda event: (
                calls.append(
                    ("reservation", event.fragment_execution_id, event.fragment_id, event.partition_id, event.worker_id)
                )
                or ["h6"]
            ),
        )
    )

    assert outputs == ["h0", "h1", "h2", "h3", "h4", "h5", "h6"]
    assert calls == [
        ("split", [1, 2]),
        ("exhausted", ["7", "8"]),
        ("status", "query-event.0.3.1"),
        ("worker", "worker-a", "lost"),
        ("memory", 2),
        ("resource", "query-event"),
        ("reservation", 0, "f0", 3, "worker-b"),
    ]
    stats = scheduler.stats().to_dict()
    assert stats["queued_events"] == 0
    assert stats["processed_events"] == 7
    assert stats["event_counts"] == {
        "SplitEventsSubmitted": 1,
        "SourceInputExhausted": 1,
        "TaskStatusChanged": 1,
        "WorkerFailed": 1,
        "MemoryPressureDetected": 1,
        "ResourceAdmissionChanged": 1,
        "WorkerReservationCompleted": 1,
    }


def test_event_scheduler_rejects_wrong_query_id_and_drops_registry_entry():
    registry = FteSchedulerRegistry()
    scheduler = registry.get_or_create("query-a")

    try:
        scheduler.enqueue(SplitEventsSubmitted.from_events("query-b", []))
    except ValueError as exc:
        assert "does not match scheduler" in str(exc)
    else:
        raise AssertionError("expected query id mismatch to fail")

    assert registry.stats()["query-a"]["state"] == "RUNNING"
    registry.drop_query("query-a")
    assert registry.stats() == {}


def test_event_scheduler_abort_clears_pending_events():
    scheduler = FteSchedulerRegistry().get_or_create("query-abort")
    aborted = []
    scheduler.enqueue(SplitEventsSubmitted.from_events("query-abort", [{"query_id": "query-abort"}]))
    scheduler.enqueue(QueryAbort("query-abort", "cancel"))

    scheduler.drain(
        FteEventHandlers(
            on_query_abort=lambda event: aborted.append(event.reason) or [],
        )
    )

    stats = scheduler.stats().to_dict()
    assert aborted == ["cancel"]
    assert stats["state"] == "ABORTED"
    assert stats["queued_events"] == 0
    assert stats["processed_events"] == 2


def test_event_scheduler_uses_bound_handlers_by_default():
    scheduler = FteSchedulerRegistry().get_or_create("query-bound")
    calls = []
    scheduler.set_handlers(
        FteEventHandlers(
            on_split_events=lambda events: calls.append([event["value"] for event in events]) or ["ok"],
        )
    )

    scheduler.enqueue(
        SplitEventsSubmitted.from_events(
            "query-bound",
            [{"query_id": "query-bound", "value": 7}],
        )
    )

    assert scheduler.drain() == ["ok"]
    assert calls == [[7]]


def test_event_scheduler_pauses_and_resumes_registered_task_source():
    scheduler = FteQueryScheduler("query-source-backpressure")
    calls = []
    values = []

    scheduler.register_task_source(
        "scan-7",
        pause=lambda: calls.append("pause"),
        resume=lambda: calls.append("resume"),
        high_watermark=2,
        low_watermark=0,
    )

    scheduler.enqueue(
        SplitEventsSubmitted.from_events(
            "query-source-backpressure",
            [{"query_id": "query-source-backpressure", "value": 1}],
        )
    )
    assert calls == []

    scheduler.enqueue(
        SplitEventsSubmitted.from_events(
            "query-source-backpressure",
            [{"query_id": "query-source-backpressure", "value": 2}],
        )
    )
    scheduler.enqueue(
        SplitEventsSubmitted.from_events(
            "query-source-backpressure",
            [{"query_id": "query-source-backpressure", "value": 3}],
        )
    )

    assert calls == ["pause"]
    assert scheduler.stats().to_dict()["paused_task_source_count"] == 1

    scheduler.drain(
        FteEventHandlers(
            on_split_events=lambda events: values.extend(event["value"] for event in events) or [],
        )
    )

    assert values == [1, 2, 3]
    assert calls == ["pause", "resume"]
    stats = scheduler.stats().to_dict()
    assert stats["registered_task_source_count"] == 1
    assert stats["paused_task_source_count"] == 0


def test_event_driven_task_source_drains_when_paused():
    scheduler = FteQueryScheduler("query-event-source")
    values = []
    source = FteEventDrivenTaskSource(
        scheduler,
        "source-0",
        high_watermark=2,
        low_watermark=0,
    )
    scheduler.set_handlers(
        FteEventHandlers(
            on_split_events=lambda events: values.extend(event["value"] for event in events) or [],
        )
    )

    source.submit(
        SplitEventsSubmitted.from_events(
            "query-event-source",
            [{"query_id": "query-event-source", "value": value}],
        )
        for value in range(5)
    )

    assert values == [0, 1, 2, 3, 4]
    assert source.pause_count >= 1
    assert source.resume_count == source.pause_count
    stats = scheduler.stats().to_dict()
    assert stats["queued_events"] == 0
    assert stats["registered_task_source_count"] == 0
    assert stats["paused_task_source_count"] == 0


def test_event_driven_task_source_keeps_queue_within_high_watermark():
    scheduler = FteQueryScheduler("query-event-source-watermark")
    values = []
    queued_before_drain = []
    source = FteEventDrivenTaskSource(
        scheduler,
        "source-watermark",
        high_watermark=3,
        low_watermark=1,
    )
    scheduler.set_handlers(
        FteEventHandlers(
            on_split_events=lambda events: values.extend(event["value"] for event in events) or [],
        )
    )

    def drain():
        queued_before_drain.append(scheduler.stats().to_dict()["queued_events"])
        return scheduler.drain()

    source.submit(
        (
            SplitEventsSubmitted.from_events(
                "query-event-source-watermark",
                [{"query_id": "query-event-source-watermark", "value": value}],
            )
            for value in range(12)
        ),
        drain=drain,
    )

    assert values == list(range(12))
    assert max(queued_before_drain) <= 3
    assert source.pause_count >= 1
    assert source.resume_count == source.pause_count
    stats = scheduler.stats().to_dict()
    assert stats["queued_events"] == 0
    assert stats["registered_task_source_count"] == 0
    assert stats["paused_task_source_count"] == 0


def test_event_driven_task_source_stress_does_not_accumulate_events():
    scheduler = FteQueryScheduler("query-event-source-stress")
    processed_count = 0
    queued_before_drain = []
    source = FteEventDrivenTaskSource(
        scheduler,
        "source-stress",
        high_watermark=5,
        low_watermark=1,
    )

    def on_split_events(events):
        nonlocal processed_count
        processed_count += len(events)
        return []

    scheduler.set_handlers(FteEventHandlers(on_split_events=on_split_events))

    def drain():
        queued_before_drain.append(scheduler.stats().to_dict()["queued_events"])
        return scheduler.drain()

    source.submit(
        (
            SplitEventsSubmitted.from_events(
                "query-event-source-stress",
                [{"query_id": "query-event-source-stress", "value": value}],
            )
            for value in range(1000)
        ),
        drain=drain,
    )

    stats = scheduler.stats().to_dict()
    assert processed_count == 1000
    assert stats["event_counts"]["SplitEventsSubmitted"] == 1000
    assert stats["queued_events"] == 0
    assert stats["registered_task_source_count"] == 0
    assert stats["paused_task_source_count"] == 0
    assert max(queued_before_drain) <= 5
    assert source.pause_count >= 100
    assert source.resume_count == source.pause_count


def test_event_scheduler_pauses_sources_by_individual_watermarks():
    scheduler = FteQueryScheduler("query-multi-source-watermark")
    calls = []
    scheduler.register_task_source(
        "source-small",
        pause=lambda: calls.append(("small", "pause")),
        resume=lambda: calls.append(("small", "resume")),
        high_watermark=2,
        low_watermark=0,
    )
    scheduler.register_task_source(
        "source-large",
        pause=lambda: calls.append(("large", "pause")),
        resume=lambda: calls.append(("large", "resume")),
        high_watermark=4,
        low_watermark=1,
    )

    for value in range(2):
        scheduler.enqueue(
            SplitEventsSubmitted.from_events(
                "query-multi-source-watermark",
                [{"query_id": "query-multi-source-watermark", "value": value}],
            )
        )

    assert calls == [("small", "pause")]
    assert scheduler.stats().to_dict()["paused_task_source_count"] == 1

    for value in range(2, 4):
        scheduler.enqueue(
            SplitEventsSubmitted.from_events(
                "query-multi-source-watermark",
                [{"query_id": "query-multi-source-watermark", "value": value}],
            )
        )

    assert calls == [("small", "pause"), ("large", "pause")]
    assert scheduler.stats().to_dict()["paused_task_source_count"] == 2

    scheduler.drain()

    assert calls == [
        ("small", "pause"),
        ("large", "pause"),
        ("large", "resume"),
        ("small", "resume"),
    ]
    assert scheduler.stats().to_dict()["paused_task_source_count"] == 0


def test_event_driven_task_source_close_resumes_paused_source_and_isolates_replacement():
    scheduler = FteQueryScheduler("query-source-attempt-boundary")
    first = FteEventDrivenTaskSource(
        scheduler,
        "source-attempt",
        high_watermark=1,
        low_watermark=0,
    )
    scheduler.enqueue(
        SplitEventsSubmitted.from_events(
            "query-source-attempt-boundary",
            [{"query_id": "query-source-attempt-boundary", "value": 1}],
        )
    )

    assert first.paused is True
    assert first.pause_count == 1

    first.close()

    assert first.paused is False
    assert first.resume_count == 1
    assert scheduler.stats().to_dict()["registered_task_source_count"] == 0

    second = FteEventDrivenTaskSource(
        scheduler,
        "source-attempt",
        high_watermark=10,
        low_watermark=0,
    )

    assert second.paused is False
    assert second.pause_count == 0
    assert scheduler.stats().to_dict()["registered_task_source_count"] == 1
    second.close()


def test_event_scheduler_close_resumes_paused_task_source():
    scheduler = FteQueryScheduler("query-close-source")
    calls = []
    scheduler.register_task_source(
        "scan-7",
        pause=lambda: calls.append("pause"),
        resume=lambda: calls.append("resume"),
        high_watermark=1,
        low_watermark=0,
    )

    scheduler.enqueue(
        SplitEventsSubmitted.from_events(
            "query-close-source",
            [{"query_id": "query-close-source", "value": 1}],
        )
    )

    assert calls == ["pause"]

    scheduler.close()

    assert calls == ["pause", "resume"]
    stats = scheduler.stats().to_dict()
    assert stats["registered_task_source_count"] == 0
    assert stats["paused_task_source_count"] == 0


def test_event_scheduler_dispatches_current_retry_delay_generation_only():
    scheduler = FteSchedulerRegistry().get_or_create("query-retry-delay")
    calls = []
    stale_generation = scheduler.arm_retry_delay(0)
    current_generation = scheduler.arm_retry_delay(0)

    scheduler.enqueue(RetryDelayExpired("query-retry-delay", stale_generation))
    scheduler.enqueue(RetryDelayExpired("query-retry-delay", current_generation))

    outputs = scheduler.drain(
        FteEventHandlers(
            on_retry_delay_expired=lambda event: calls.append(event.generation) or ["retry"],
        )
    )

    assert outputs == ["retry"]
    assert calls == [current_generation]


def test_event_scheduler_retry_delay_timer_records_handler_failure():
    scheduler = FteSchedulerRegistry().get_or_create("query-retry-delay-failure")
    generation = scheduler.arm_retry_delay(0)

    def fail_retry_delay(_event):
        raise RuntimeError("retry drain failed")

    scheduler.set_handlers(FteEventHandlers(on_retry_delay_expired=fail_retry_delay))
    scheduler._fire_retry_delay(generation)

    stats = scheduler.stats().to_dict()
    assert stats["state"] == "FAILED"
    assert stats["queued_events"] == 0
    assert "retry drain failed" in stats["failure_reason"]


def test_attempt_status_watcher_enqueues_terminal_status_event():
    class _Worker:
        worker_id = "worker-a"

        def fte_wait_task_status(self, task_id, min_version, timeout_s):
            return {
                "state": "FINISHED",
                "task_id": task_id,
                "version": 3,
            }

    scheduler = FteSchedulerRegistry().get_or_create("query-watch")
    calls = []
    scheduler.set_handlers(
        FteEventHandlers(
            on_task_status_changed=lambda event: calls.append(str(event.attempt_id)) or [],
        )
    )
    attempt_id = FteTaskAttemptId(FteTaskId("query-watch", 0, 1), 0)
    watcher = FteAttemptStatusWatcher(
        scheduler=scheduler,
        attempt_id=attempt_id,
        worker=_Worker(),
        wait_timeout_s=0,
    )

    assert watcher.start() is True
    watcher.join(1.0)

    assert calls == ["query-watch.0.1.0"]
    assert scheduler.stats().to_dict()["event_counts"] == {
        "TaskStatusChanged": 1,
    }


def test_attempt_status_watcher_enqueues_running_progress_before_terminal():
    class _Worker:
        worker_id = "worker-progress"

        def __init__(self):
            self.min_versions = []
            self.statuses = [
                {
                    "state": "RUNNING",
                    "version": 1,
                    "task_stats": {"processed_input_rows": 5},
                },
                {
                    "state": "RUNNING",
                    "version": 2,
                    "task_stats": {"processed_input_rows": 9},
                },
                {
                    "state": "FINISHED",
                    "version": 3,
                    "task_stats": {"processed_input_rows": 10},
                },
            ]

        def fte_wait_task_status(self, task_id, min_version, timeout_s):
            self.min_versions.append(min_version)
            return {"task_id": task_id, **self.statuses.pop(0)}

    scheduler = FteSchedulerRegistry().get_or_create("query-watch-progress")
    statuses = []
    scheduler.set_handlers(
        FteEventHandlers(
            on_task_status_changed=lambda event: statuses.append(event.status) or [],
        )
    )
    attempt_id = FteTaskAttemptId(FteTaskId("query-watch-progress", 0, 1), 0)
    worker = _Worker()
    watcher = FteAttemptStatusWatcher(
        scheduler=scheduler,
        attempt_id=attempt_id,
        worker=worker,
        wait_timeout_s=0,
        poll_interval_s=0.001,
    )

    assert watcher.start() is True
    watcher.join(1.0)

    assert [status["state"] for status in statuses] == [
        "RUNNING",
        "RUNNING",
        "FINISHED",
    ]
    assert [status["task_stats"]["processed_input_rows"] for status in statuses] == [5, 9, 10]
    assert worker.min_versions == [None, 2, 3]


def test_attempt_status_watcher_marks_scheduler_failed_when_handler_fails():
    class _Worker:
        worker_id = "worker-handler-failure"

        def fte_wait_task_status(self, task_id, min_version, timeout_s):
            return {
                "state": "FINISHED",
                "task_id": task_id,
                "version": 1,
            }

    def fail_status_handler(_event):
        raise RuntimeError("terminal handler failed")

    scheduler = FteSchedulerRegistry().get_or_create("query-watch-handler-failure")
    scheduler.set_handlers(FteEventHandlers(on_task_status_changed=fail_status_handler))
    attempt_id = FteTaskAttemptId(FteTaskId("query-watch-handler-failure", 0, 1), 0)
    watcher = FteAttemptStatusWatcher(
        scheduler=scheduler,
        attempt_id=attempt_id,
        worker=_Worker(),
        wait_timeout_s=0,
    )

    try:
        assert watcher.start() is True
        watcher.join(1.0)
    finally:
        watcher.stop()
        watcher.join(1.0)

    stats = scheduler.stats().to_dict()
    assert stats["state"] == "FAILED"
    assert stats["queued_events"] == 0
    assert "terminal handler failed" in stats["failure_reason"]


def test_attempt_status_watcher_requires_status_wait_protocol():
    scheduler = FteSchedulerRegistry().get_or_create("query-watch-required")
    attempt_id = FteTaskAttemptId(FteTaskId("query-watch-required", 0, 1), 0)
    watcher = FteAttemptStatusWatcher(
        scheduler=scheduler,
        attempt_id=attempt_id,
        worker=object(),
        wait_timeout_s=0,
    )

    with pytest.raises(TypeError, match="fte_wait_task_status must be callable"):
        watcher.start()


def test_attempt_status_watcher_reports_malformed_status():
    class _Worker:
        worker_id = "worker-malformed"

        def fte_wait_task_status(self, task_id, min_version, timeout_s):
            return "not-a-status-dict"

    scheduler = FteSchedulerRegistry().get_or_create("query-watch-malformed")
    calls = []
    scheduler.set_handlers(
        FteEventHandlers(
            on_worker_failed=lambda event: calls.append((event.worker_id, str(event.error))) or [],
        )
    )
    attempt_id = FteTaskAttemptId(FteTaskId("query-watch-malformed", 0, 1), 0)
    watcher = FteAttemptStatusWatcher(
        scheduler=scheduler,
        attempt_id=attempt_id,
        worker=_Worker(),
        wait_timeout_s=0,
        poll_interval_s=0.001,
    )

    try:
        assert watcher.start() is True
        watcher.join(1.0)
    finally:
        watcher.stop()
        watcher.join(1.0)

    assert calls == [("worker-malformed", "fte_wait_task_status must return a dict")]


def test_attempt_status_watcher_reports_unknown_task_state_as_worker_failure():
    class _Worker:
        worker_id = "worker-unknown-state"

        def fte_wait_task_status(self, task_id, min_version, timeout_s):
            return {
                "state": "MALFORMED",
                "task_id": task_id,
                "version": 1,
            }

    scheduler = FteSchedulerRegistry().get_or_create("query-watch-unknown-state")
    calls = []
    scheduler.set_handlers(
        FteEventHandlers(
            on_worker_failed=lambda event: calls.append((event.worker_id, str(event.error))) or [],
        )
    )
    attempt_id = FteTaskAttemptId(FteTaskId("query-watch-unknown-state", 0, 1), 0)
    watcher = FteAttemptStatusWatcher(
        scheduler=scheduler,
        attempt_id=attempt_id,
        worker=_Worker(),
        wait_timeout_s=0,
        poll_interval_s=0.001,
    )

    try:
        assert watcher.start() is True
        watcher.join(1.0)
    finally:
        watcher.stop()
        watcher.join(1.0)

    assert watcher.is_alive() is False
    assert len(calls) == 1
    assert calls[0][0] == "worker-unknown-state"
    assert "unknown FTE task state" in calls[0][1]


def test_attempt_status_watcher_treats_query_deadline_as_hard_failure():
    class _Worker:
        worker_id = "worker-query-deadline"

        def fte_wait_task_status(self, task_id, min_version, timeout_s):
            raise QueryDeadlineExceeded("query deadline expired before Ray ObjectRef get")

    scheduler = FteSchedulerRegistry().get_or_create("query-watch-deadline")
    calls = []
    scheduler.set_handlers(
        FteEventHandlers(
            on_worker_failed=lambda event: calls.append((event.worker_id, str(event.error))) or [],
        )
    )
    attempt_id = FteTaskAttemptId(FteTaskId("query-watch-deadline", 0, 1), 0)
    watcher = FteAttemptStatusWatcher(
        scheduler=scheduler,
        attempt_id=attempt_id,
        worker=_Worker(),
        wait_timeout_s=0,
        poll_interval_s=0.001,
    )

    assert watcher.start() is True
    watcher.join(1.0)

    assert watcher.is_alive() is False
    assert calls == [("worker-query-deadline", "query deadline expired before Ray ObjectRef get")]


def test_attempt_status_watcher_join_observes_real_thread_lifecycle():
    entered = threading.Event()

    class _Worker:
        worker_id = "worker-slow-status"

        def fte_wait_task_status(self, task_id, min_version, timeout_s):
            entered.set()
            time.sleep(0.2)
            return {
                "state": "RUNNING",
                "task_id": task_id,
                "version": 1,
            }

    scheduler = FteSchedulerRegistry().get_or_create("query-watch-slow-status")
    attempt_id = FteTaskAttemptId(FteTaskId("query-watch-slow-status", 0, 1), 0)
    watcher = FteAttemptStatusWatcher(
        scheduler=scheduler,
        attempt_id=attempt_id,
        worker=_Worker(),
        wait_timeout_s=1,
        poll_interval_s=0.001,
    )

    assert watcher.start() is True
    assert entered.wait(1.0)
    watcher.stop()
    watcher.join(1.0)

    assert watcher.is_alive() is False
