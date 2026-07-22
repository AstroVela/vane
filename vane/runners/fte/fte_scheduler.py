# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vane.runners.common import QueryDeadlineExceeded
from vane.runners.fte.fte_events import (
    ExchangeSelectorUpdated,
    FteAddSplitsCommand,
    FteCreateTaskCommand,
    FteNoMoreSplitsCommand,
    FteTaskUpdateCommand,
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
from vane.runners.fte.fte_state import FteTaskState
from vane.runners.fte.fte_types import validate_fte_status_identity

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from vane.runners.fte.fte_events import FteEvent, FteWorkerCommand


class FteWorkerCommandExecutor:
    """Execute typed commands through the worker's single ordered control lane."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._command_counts: dict[str, int] = {}

    def _record(self, command: FteWorkerCommand) -> None:
        command_type = command.command_type
        with self._lock:
            self._command_counts[str(command_type)] = self._command_counts.get(str(command_type), 0) + 1

    def command_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._command_counts)

    def create_task(self, command: FteCreateTaskCommand) -> Any:
        self._record(command)
        return command.worker.fte_create_task(command.request)

    @staticmethod
    def _required_worker_method(worker: Any, method_name: str) -> Any:
        method = getattr(worker, method_name, None)
        if method is None:
            raise RuntimeError(f"FTE worker handle must provide {method_name}")
        return method

    def wait_split_queue_has_space(self, command: FteAddSplitsCommand) -> None:
        worker = command.worker
        space = self._required_worker_method(worker, "fte_wait_split_queue_has_space")(
            command.attempt_id.to_dict(),
            command.source_node_id,
            None,
            None,
        )
        if not bool(space.get("has_space", True)):
            raise RuntimeError(f"split queue for task {command.attempt_id} source {command.source_node_id} stayed full")

    def add_splits(self, command: FteAddSplitsCommand) -> Any:
        self._record(command)
        worker = command.worker
        split_payloads = [dict(split) for split in command.splits]
        task_id = command.attempt_id.to_dict()
        return self._required_worker_method(worker, "enqueue_fte_add_splits")(
            task_id,
            command.source_node_id,
            split_payloads,
        )

    def no_more_splits(self, command: FteNoMoreSplitsCommand) -> Any:
        self._record(command)
        worker = command.worker
        task_id = command.attempt_id.to_dict()
        return self._required_worker_method(worker, "enqueue_fte_no_more_splits")(
            task_id,
            command.source_node_id,
        )

    def update_task(self, command: FteTaskUpdateCommand) -> Any:
        self._record(command)
        worker = command.worker
        task_id = command.attempt_id.to_dict()
        update = dict(command.update)
        return self._required_worker_method(worker, "enqueue_fte_update_task")(task_id, update)

    def execute(self, command: Any) -> Any:
        if isinstance(command, FteCreateTaskCommand):
            return self.create_task(command)
        if isinstance(command, FteAddSplitsCommand):
            self.wait_split_queue_has_space(command)
            return self.add_splits(command)
        if isinstance(command, FteNoMoreSplitsCommand):
            return self.no_more_splits(command)
        if isinstance(command, FteTaskUpdateCommand):
            return self.update_task(command)
        raise TypeError(f"unsupported FTE worker command: {type(command).__name__}")


class FteAttemptStatusWatcher:
    def __init__(
        self,
        *,
        scheduler: FteQueryScheduler,
        attempt_id: Any,
        worker: Any,
        poll_interval_s: float = 0.05,
        wait_timeout_s: float = 1.0,
        on_exit: Any = None,
    ) -> None:
        self.scheduler = scheduler
        self.attempt_id = attempt_id
        self.worker = worker
        self.poll_interval_s = max(0.001, float(poll_interval_s))
        self.wait_timeout_s = max(0.0, float(wait_timeout_s))
        self.on_exit = on_exit
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        wait_task_status = getattr(self.worker, "fte_wait_task_status", None)
        if not callable(wait_task_status):
            raise TypeError("FTE worker handle fte_wait_task_status must be callable")
        if self._thread is not None:
            return True
        self._thread = threading.Thread(
            target=self._run_and_notify_exit,
            name=f"fte-status-watcher-{self.attempt_id}",
            daemon=True,
        )
        self._thread.start()
        return True

    def _run_and_notify_exit(self) -> None:
        try:
            self._run()
        finally:
            if callable(self.on_exit):
                self.on_exit(self)

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout_s: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout_s)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def shutdown_timeout_s(self) -> float:
        # Native/test workers may not expose the interruptible Ray wait API.
        # Give one declared server wait plus bounded scheduling slack, then let
        # the registry owner report failure without discarding the live thread.
        return max(5.0, self.wait_timeout_s + 5.0)

    def _wait_task_status(self, min_version: int | None) -> Any:
        interruptible_wait = getattr(
            self.worker,
            "fte_wait_task_status_interruptible",
            None,
        )
        if callable(interruptible_wait):
            return interruptible_wait(
                self.attempt_id.to_dict(),
                min_version,
                self.wait_timeout_s,
                self._stop,
            )
        return self.worker.fte_wait_task_status(
            self.attempt_id.to_dict(),
            min_version,
            self.wait_timeout_s,
        )

    def _enqueue_and_drain(self, event: FteEvent) -> bool:
        try:
            self.scheduler.enqueue(event)
            self.scheduler.drain()
        except Exception as exc:
            self.scheduler.fail(f"FTE status watcher failed while handling {event.event_type}: {exc}")
            return False
        return True

    @staticmethod
    def _is_soft_status_wait_timeout(exc: BaseException) -> bool:
        if isinstance(exc, QueryDeadlineExceeded) or "query deadline expired" in str(exc).lower():
            return False
        if isinstance(exc, TimeoutError):
            return True
        name = exc.__class__.__name__
        if name in {"TimeoutError", "GetTimeoutError"}:
            return True
        message = str(exc)
        return "did not complete within" in message or "timed out" in message.lower()

    def _run(self) -> None:
        min_version = None
        while not self._stop.is_set():
            try:
                status = self._wait_task_status(min_version)
            except Exception as exc:
                if self._stop.is_set():
                    return
                # Client/server wait timeouts are expected under load; retry.
                if self._is_soft_status_wait_timeout(exc):
                    self._stop.wait(self.poll_interval_s)
                    continue
                self._enqueue_and_drain(
                    WorkerFailed(
                        self.scheduler.query_id,
                        str(self.worker.worker_id),
                        exc,
                    )
                )
                return
            if self._stop.is_set():
                return
            if not isinstance(status, dict):
                self._enqueue_and_drain(
                    WorkerFailed(
                        self.scheduler.query_id,
                        str(self.worker.worker_id),
                        TypeError("fte_wait_task_status must return a dict"),
                    )
                )
                return
            try:
                validate_fte_status_identity(status, self.attempt_id)
            except Exception as exc:
                self._enqueue_and_drain(
                    WorkerFailed(
                        self.scheduler.query_id,
                        str(self.worker.worker_id),
                        exc,
                    )
                )
                return
            version = status.get("version")
            if version is not None:
                try:
                    min_version = int(version) + 1
                except (TypeError, ValueError):
                    min_version = None
            raw_state = status.get("state")
            try:
                state = raw_state if isinstance(raw_state, FteTaskState) else FteTaskState(str(raw_state))
            except (TypeError, ValueError):
                self._enqueue_and_drain(
                    WorkerFailed(
                        self.scheduler.query_id,
                        str(self.worker.worker_id),
                        ValueError(f"unknown FTE task state: {raw_state!r}"),
                    )
                )
                return
            terminal = state in {
                FteTaskState.FINISHED,
                FteTaskState.FAILED,
                FteTaskState.CANCELED,
                FteTaskState.ABORTED,
            }
            if not self._enqueue_and_drain(
                TaskStatusChanged.from_status(
                    self.scheduler.query_id,
                    self.attempt_id,
                    status,
                )
            ):
                return
            if terminal:
                return
            self._stop.wait(self.poll_interval_s)


@dataclass
class FteEventHandlers:
    on_split_events: Callable[[list[dict[str, Any]]], list[Any] | None] | None = None
    on_source_input_exhausted: Callable[[set[str]], list[Any] | None] | None = None
    on_task_status_changed: Callable[[TaskStatusChanged], list[Any] | None] | None = None
    on_worker_failed: Callable[[WorkerFailed], list[Any] | None] | None = None
    on_memory_pressure_detected: Callable[[MemoryPressureDetected], list[Any] | None] | None = None
    on_resource_admission_changed: Callable[[ResourceAdmissionChanged], list[Any] | None] | None = None
    on_worker_reservation_completed: Callable[[WorkerReservationCompleted], list[Any] | None] | None = None
    on_retry_delay_expired: Callable[[RetryDelayExpired], list[Any] | None] | None = None
    on_exchange_selector_updated: Callable[[ExchangeSelectorUpdated], list[Any] | None] | None = None
    on_query_abort: Callable[[QueryAbort], list[Any] | None] | None = None


@dataclass(frozen=True)
class FteSchedulerStats:
    query_id: str
    state: str
    queued_events: int
    processed_events: int
    last_event_type: str | None
    last_progress_ms: int | None
    failure_reason: str | None = None
    fragment_state_count: int = 0
    failed_worker_count: int = 0
    command_counts: dict[str, int] = field(default_factory=dict)
    event_counts: dict[str, int] = field(default_factory=dict)
    registered_task_source_count: int = 0
    paused_task_source_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "state": self.state,
            "queued_events": self.queued_events,
            "processed_events": self.processed_events,
            "last_event_type": self.last_event_type,
            "last_progress_ms": self.last_progress_ms,
            "failure_reason": self.failure_reason,
            "fragment_state_count": self.fragment_state_count,
            "failed_worker_count": self.failed_worker_count,
            "command_counts": dict(self.command_counts),
            "event_counts": dict(self.event_counts),
            "registered_task_source_count": self.registered_task_source_count,
            "paused_task_source_count": self.paused_task_source_count,
        }


@dataclass
class FteTaskSourceRegistration:
    source_id: str
    pause: Callable[[], None] | None
    resume: Callable[[], None] | None
    high_watermark: int
    low_watermark: int
    paused: bool = False


class FteEventDrivenTaskSource:
    """Synchronous event source with scheduler-owned pause/resume backpressure."""

    def __init__(
        self,
        scheduler: FteQueryScheduler,
        source_id: str,
        *,
        high_watermark: int = 1024,
        low_watermark: int | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.source_id = str(source_id or "").strip()
        if not self.source_id:
            raise ValueError("source_id must be non-empty")
        self.pause_count = 0
        self.resume_count = 0
        self._paused = False
        self._closed = False
        self._lock = threading.RLock()
        self.scheduler.register_task_source(
            self.source_id,
            pause=self.pause,
            resume=self.resume,
            high_watermark=high_watermark,
            low_watermark=low_watermark,
        )

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def pause(self) -> None:
        with self._lock:
            if self._closed or self._paused:
                return
            self._paused = True
            self.pause_count += 1

    def resume(self) -> None:
        with self._lock:
            if self._closed or not self._paused:
                return
            self._paused = False
            self.resume_count += 1

    def submit(
        self,
        events: Iterable[FteEvent],
        *,
        drain: Callable[[], list[Any]] | None = None,
    ) -> list[Any]:
        drain = self.scheduler.drain if drain is None else drain
        outputs: list[Any] = []
        try:
            for event in events:
                if self.paused:
                    outputs.extend(drain())
                self.scheduler.enqueue(event)
                if self.paused:
                    outputs.extend(drain())
            outputs.extend(drain())
            return outputs
        finally:
            self.close()

    def close(self) -> None:
        should_resume = False
        with self._lock:
            if self._closed:
                return
            should_resume = self._paused
        if should_resume:
            self.resume()
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.scheduler.unregister_task_source(self.source_id, resume=False)


class FteQueryScheduler:
    """Small per-query event loop facade for FTE scheduling.

    The first implementation slice centralizes event ingestion and ordered
    dispatch while existing submit/status handlers still perform the
    concrete mutations. Later phases move those mutations behind event handlers
    owned by this class.
    """

    def __init__(self, query_id: str) -> None:
        query_id = str(query_id or "").strip()
        if not query_id:
            raise ValueError("query_id must be non-empty")
        self.query_id = query_id
        self._events: deque[FteEvent] = deque()
        self._lock = threading.RLock()
        self._state = "RUNNING"
        self._processed_events = 0
        self._last_event_type: str | None = None
        self._last_progress_at: float | None = None
        self._event_counts: dict[str, int] = {}
        self._handlers = FteEventHandlers()
        self._fragment_states: dict[str, Any] = {}
        self.worker_command_executor = FteWorkerCommandExecutor()
        self._failure_reason: str | None = None
        self._retry_delay_generation = 0
        self._retry_delay_timer: threading.Timer | None = None
        self._draining = False
        self._failed_worker_ids: set[str] = set()
        self._task_sources: dict[str, FteTaskSourceRegistration] = {}

    def set_handlers(self, handlers: FteEventHandlers) -> None:
        with self._lock:
            self._handlers = handlers

    def register_task_source(
        self,
        source_id: str,
        *,
        pause: Callable[[], None] | None = None,
        resume: Callable[[], None] | None = None,
        high_watermark: int = 1024,
        low_watermark: int | None = None,
    ) -> None:
        source_id = str(source_id or "").strip()
        if not source_id:
            raise ValueError("source_id must be non-empty")
        high_watermark = max(1, int(high_watermark))
        if low_watermark is None:
            low_watermark = max(0, high_watermark // 2)
        low_watermark = max(0, min(int(low_watermark), high_watermark - 1))
        callbacks: list[Callable[[], None]] = []
        with self._lock:
            old = self._task_sources.get(source_id)
            if old is not None and old.paused and old.resume is not None:
                callbacks.append(old.resume)
            self._task_sources[source_id] = FteTaskSourceRegistration(
                source_id=source_id,
                pause=pause,
                resume=resume,
                high_watermark=high_watermark,
                low_watermark=low_watermark,
            )
            callbacks.extend(self._task_source_pause_callbacks_locked())
        self._run_task_source_callbacks(callbacks)

    def unregister_task_source(self, source_id: str, *, resume: bool = True) -> None:
        callbacks: list[Callable[[], None]] = []
        with self._lock:
            registration = self._task_sources.pop(str(source_id), None)
            if resume and registration is not None and registration.paused and registration.resume is not None:
                callbacks.append(registration.resume)
        self._run_task_source_callbacks(callbacks)

    def enqueue(self, event: FteEvent) -> None:
        if event.query_id != self.query_id:
            raise ValueError(f"event query_id {event.query_id!r} does not match scheduler {self.query_id!r}")
        callbacks: list[Callable[[], None]] = []
        with self._lock:
            if self._state != "RUNNING" and not isinstance(event, QueryAbort):
                return
            self._events.append(event)
            callbacks.extend(self._task_source_pause_callbacks_locked())
        self._run_task_source_callbacks(callbacks)

    def drain(self, handlers: FteEventHandlers | None = None) -> list[Any]:
        outputs: list[Any] = []
        with self._lock:
            already_draining = self._draining
            if not already_draining:
                self._draining = True
        if already_draining:
            return outputs
        try:
            while True:
                with self._lock:
                    if not self._events:
                        break
                    event = self._events.popleft()
                    event_handlers = handlers or self._handlers
                outputs.extend(self._handle_event(event, event_handlers))
                callbacks: list[Callable[[], None]] = []
                with self._lock:
                    callbacks.extend(self._task_source_resume_callbacks_locked())
                self._run_task_source_callbacks(callbacks)
        finally:
            callbacks: list[Callable[[], None]] = []
            with self._lock:
                self._draining = False
                callbacks.extend(self._task_source_resume_callbacks_locked())
            self._run_task_source_callbacks(callbacks)
        return outputs

    def fail(self, reason: Any = None) -> None:
        callbacks: list[Callable[[], None]] = []
        with self._lock:
            if self._state in {"CLOSED", "FAILED"}:
                return
            self._state = "FAILED"
            self._failure_reason = "" if reason is None else str(reason)
            self._events.clear()
            if self._retry_delay_timer is not None:
                self._retry_delay_timer.cancel()
                self._retry_delay_timer = None
            callbacks.extend(self._task_source_resume_callbacks_locked())
        self._run_task_source_callbacks(callbacks)

    def close(self) -> None:
        callbacks: list[Callable[[], None]] = []
        with self._lock:
            for registration in self._task_sources.values():
                if registration.paused and registration.resume is not None:
                    callbacks.append(registration.resume)
            self._state = "CLOSED"
            self._failure_reason = None
            self._events.clear()
            self._fragment_states.clear()
            self._failed_worker_ids.clear()
            self._task_sources.clear()
            if self._retry_delay_timer is not None:
                self._retry_delay_timer.cancel()
                self._retry_delay_timer = None
        self._run_task_source_callbacks(callbacks)

    def stats(self) -> FteSchedulerStats:
        with self._lock:
            last_progress_ms = None
            if self._last_progress_at is not None:
                last_progress_ms = int((time.monotonic() - self._last_progress_at) * 1000)
            return FteSchedulerStats(
                query_id=self.query_id,
                state=self._state,
                queued_events=len(self._events),
                processed_events=self._processed_events,
                last_event_type=self._last_event_type,
                last_progress_ms=last_progress_ms,
                failure_reason=self._failure_reason,
                fragment_state_count=len(self._fragment_states),
                failed_worker_count=len(self._failed_worker_ids),
                command_counts=self.worker_command_executor.command_counts(),
                event_counts=dict(self._event_counts),
                registered_task_source_count=len(self._task_sources),
                paused_task_source_count=sum(1 for source in self._task_sources.values() if source.paused),
            )

    def fragment_state(self, fragment_id: str) -> Any | None:
        with self._lock:
            return self._fragment_states.get(str(fragment_id))

    def put_fragment_state(self, fragment_id: str, state: Any) -> Any:
        with self._lock:
            self._fragment_states[str(fragment_id)] = state
        return state

    def fragment_state_count(self) -> int:
        with self._lock:
            return len(self._fragment_states)

    def is_draining(self) -> bool:
        with self._lock:
            return self._draining

    def record_worker_failure(
        self, worker_ids: set[str] | frozenset[str] | list[str] | tuple[str, ...]
    ) -> frozenset[str]:
        normalized = frozenset(str(worker_id) for worker_id in worker_ids if str(worker_id))
        if not normalized:
            return frozenset()
        with self._lock:
            new_worker_ids = frozenset(
                worker_id for worker_id in normalized if worker_id not in self._failed_worker_ids
            )
            self._failed_worker_ids.update(new_worker_ids)
            return new_worker_ids

    def retry_delay_generation(self) -> int:
        with self._lock:
            return self._retry_delay_generation

    def arm_retry_delay(self, delay_s: float) -> int:
        delay_s = max(0.0, float(delay_s))
        with self._lock:
            self._retry_delay_generation += 1
            generation = self._retry_delay_generation
            if self._retry_delay_timer is not None:
                self._retry_delay_timer.cancel()
                self._retry_delay_timer = None
            if delay_s <= 0 or self._state != "RUNNING":
                return generation
            timer = threading.Timer(delay_s, self._fire_retry_delay, args=(generation,))
            timer.daemon = True
            self._retry_delay_timer = timer
            timer.start()
            return generation

    def _task_source_pause_callbacks_locked(self) -> list[Callable[[], None]]:
        queue_size = len(self._events)
        callbacks: list[Callable[[], None]] = []
        for registration in self._task_sources.values():
            if registration.paused or queue_size < registration.high_watermark:
                continue
            registration.paused = True
            if registration.pause is not None:
                callbacks.append(registration.pause)
        return callbacks

    def _task_source_resume_callbacks_locked(self) -> list[Callable[[], None]]:
        queue_size = len(self._events)
        callbacks: list[Callable[[], None]] = []
        for registration in self._task_sources.values():
            if not registration.paused or queue_size > registration.low_watermark:
                continue
            registration.paused = False
            if registration.resume is not None:
                callbacks.append(registration.resume)
        return callbacks

    @staticmethod
    def _run_task_source_callbacks(callbacks: list[Callable[[], None]]) -> None:
        for callback in callbacks:
            callback()

    def _fire_retry_delay(self, generation: int) -> None:
        try:
            self.enqueue(RetryDelayExpired(self.query_id, generation))
            self.drain()
        except Exception as exc:
            self.fail(f"FTE retry delay handler failed: {exc}")

    def _handle_event(
        self,
        event: FteEvent,
        handlers: FteEventHandlers,
    ) -> list[Any]:
        event_type = event.event_type
        with self._lock:
            self._processed_events += 1
            self._last_event_type = event_type
            self._last_progress_at = time.monotonic()
            self._event_counts[event_type] = self._event_counts.get(event_type, 0) + 1

        if isinstance(event, SplitEventsSubmitted):
            if handlers.on_split_events is None:
                return []
            return list(handlers.on_split_events([dict(item) for item in event.events]) or [])
        if isinstance(event, SourceInputExhausted):
            if handlers.on_source_input_exhausted is None:
                return []
            return list(handlers.on_source_input_exhausted(set(event.source_node_ids)) or [])
        if isinstance(event, TaskStatusChanged):
            if handlers.on_task_status_changed is None:
                return []
            return list(handlers.on_task_status_changed(event) or [])
        if isinstance(event, WorkerFailed):
            if handlers.on_worker_failed is None:
                return []
            return list(handlers.on_worker_failed(event) or [])
        if isinstance(event, MemoryPressureDetected):
            if handlers.on_memory_pressure_detected is None:
                return []
            return list(handlers.on_memory_pressure_detected(event) or [])
        if isinstance(event, ResourceAdmissionChanged):
            if handlers.on_resource_admission_changed is None:
                return []
            return list(handlers.on_resource_admission_changed(event) or [])
        if isinstance(event, WorkerReservationCompleted):
            if handlers.on_worker_reservation_completed is None:
                return []
            return list(handlers.on_worker_reservation_completed(event) or [])
        if isinstance(event, RetryDelayExpired):
            with self._lock:
                if event.generation != self._retry_delay_generation:
                    return []
                self._retry_delay_timer = None
            if handlers.on_retry_delay_expired is None:
                return []
            return list(handlers.on_retry_delay_expired(event) or [])
        if isinstance(event, ExchangeSelectorUpdated):
            if handlers.on_exchange_selector_updated is None:
                return []
            return list(handlers.on_exchange_selector_updated(event) or [])
        if isinstance(event, QueryAbort):
            with self._lock:
                self._state = "ABORTED"
                self._events.clear()
            if handlers.on_query_abort is None:
                return []
            return list(handlers.on_query_abort(event) or [])
        return []


class FteSchedulerRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._schedulers: dict[str, FteQueryScheduler] = {}
        self._pending_drain_cursor_query_id: str | None = None

    def get_or_create(self, query_id: str) -> FteQueryScheduler:
        query_id = str(query_id or "").strip()
        if not query_id:
            raise ValueError("query_id must be non-empty")
        with self._lock:
            scheduler = self._schedulers.get(query_id)
            if scheduler is None:
                scheduler = FteQueryScheduler(query_id)
                self._schedulers[query_id] = scheduler
            return scheduler

    def get(self, query_id: str) -> FteQueryScheduler | None:
        with self._lock:
            return self._schedulers.get(str(query_id))

    def query_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._schedulers)

    def drop_query(self, query_id: str) -> None:
        query_id = str(query_id)
        with self._lock:
            scheduler = self._schedulers.pop(query_id, None)
            if self._pending_drain_cursor_query_id == query_id:
                self._pending_drain_cursor_query_id = None
        if scheduler is not None:
            scheduler.close()

    def clear(self) -> None:
        with self._lock:
            schedulers = list(self._schedulers.values())
            self._schedulers.clear()
            self._pending_drain_cursor_query_id = None
        for scheduler in schedulers:
            scheduler.close()

    def record_pending_drain_progress(self, query_id: str) -> None:
        query_id = str(query_id or "").strip()
        if not query_id:
            return
        with self._lock:
            self._pending_drain_cursor_query_id = query_id

    def ordered_pending_drain_query_ids(self, query_ids: list[str] | tuple[str, ...] | set[str]) -> list[str]:
        ordered = sorted({str(query_id) for query_id in query_ids if str(query_id)})
        if not ordered:
            return []
        with self._lock:
            cursor = self._pending_drain_cursor_query_id
        if cursor not in ordered:
            return ordered
        start_index = (ordered.index(cursor) + 1) % len(ordered)
        return ordered[start_index:] + ordered[:start_index]

    def stats(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            schedulers = list(self._schedulers.values())
        return {scheduler.query_id: scheduler.stats().to_dict() for scheduler in schedulers}
