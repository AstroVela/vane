# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from duckdb.runners.common import QueryDeadlineExceeded
from duckdb.runners.fte import FteWorkerControlFailure, fte_status_wait_timeout_s
from duckdb.runners.fte.fte_events import FteCreateTaskCommand, TaskStatusChanged
from duckdb.runners.fte.fte_scheduler import (
    FteAttemptStatusWatcher,
    FteSplitQueueTerminal,
    FteSplitSubmissionCancelled,
)
from duckdb.runners.ray.fragment_registry import (
    _FTE_CLOSING_QUERIES,
    _FTE_REGISTRY_LOCK,
    _FTE_SCHEDULERS,
    _FTE_STATUS_WATCHERS,
)
from duckdb.runners.ray.fragment_worker_failures import quarantine_fte_worker
from duckdb.runners.ray.fte_fragment_scheduler import (
    _store_fte_result_handles,
    _sync_write_sink_unit_for_fragment,
    begin_fte_registry_operation,
    end_fte_registry_operation,
    fte_partition_task_lease_payload,
)

if TYPE_CHECKING:
    from duckdb.runners.fte import FteFragmentExecution, FteTaskAttemptId


def _fte_command_debug_enabled() -> bool:
    for name in ("VANE_FTE_ADMISSION_DEBUG", "DUCKDB_DISTRIBUTED_DEBUG"):
        value = os.getenv(name, "")
        if value.strip().lower() not in ("", "0", "false", "no", "off"):
            return True
    return False


def _fte_command_debug_log(event: str, **fields: Any) -> None:
    if not _fte_command_debug_enabled():
        return
    parts = [f"event={event}", f"pid={os.getpid()}"]
    for key, value in fields.items():
        text = "None" if value is None else str(value).replace(" ", "_")
        parts.append(f"{key}={text}")
    print("[vane-fte-command] " + " ".join(parts), file=sys.stderr, flush=True)


def _fte_worker_command_debug_fields(command: Any) -> dict[str, str]:
    worker = command.worker
    return {
        "worker_id": str(command.worker_id),
        "worker_incarnation_id": str(command.worker_incarnation_id),
        "manager_instance_id": str(getattr(worker, "manager_instance_id", "") or ""),
        "node_id": str(getattr(worker, "node_id", "") or ""),
        "host": str(getattr(worker, "host", "") or ""),
    }


class _FteQueryClosingEvent:
    """Event-compatible view of one query's registry close state."""

    def __init__(self, query_id: str) -> None:
        self._query_id = str(query_id)

    def is_set(self) -> bool:
        with _FTE_REGISTRY_LOCK:
            return self._query_id in _FTE_CLOSING_QUERIES


@dataclass(frozen=True)
class FteWorkerCommandDispatchResult:
    """Settled ownership of one driver-side worker-command batch."""

    scheduled_attempts: tuple[Any, ...] = ()
    recovery_handles: tuple[Any, ...] = ()
    failures: tuple[FteWorkerControlFailure, ...] = ()
    query_closed: bool = False

    @property
    def failed_worker_incarnations(self) -> frozenset[tuple[str, str]]:
        return frozenset((failure.worker_id, failure.worker_incarnation_id) for failure in self.failures)


class FteWorkerCommandMixin:
    if TYPE_CHECKING:
        # Supplied by the other mixins on the composed Ray worker handle.
        _bind_fte_scheduler_handlers: Any
        _fte_partition_owner: Any
        _fte_task_handle_cls: Any
        _handles_for_fte_worker_control_failure: Any
        manager_instance_id: str

    def _execute_fte_fragment_execution_worker_commands(
        self,
        fragment_execution: FteFragmentExecution,
        worker_commands: list[Any] | tuple[Any, ...],
    ) -> FteWorkerCommandDispatchResult:
        commands = tuple(worker_commands)
        terminal_attempts: set[str] = set()
        successful_create_attempts: dict[str, Any] = {}
        failed_worker_incarnations: set[tuple[str, str]] = set()
        failures: list[FteWorkerControlFailure] = []
        query_closed = False
        abort_error: BaseException | None = None

        for command_index, command in enumerate(commands):
            query_id = str(command.query_id)
            attempt_id = getattr(command, "attempt_id", None)
            if attempt_id is not None and str(attempt_id) in terminal_attempts:
                continue
            worker_id = str(command.worker_id)
            if not worker_id:
                raise ValueError("FTE worker command requires a non-empty worker_id")
            worker_incarnation_id = str(command.worker_incarnation_id)
            if not worker_incarnation_id:
                raise ValueError("FTE worker command requires a non-empty worker_incarnation_id")
            worker_incarnation = (worker_id, worker_incarnation_id)
            worker_debug_fields = _fte_worker_command_debug_fields(command)
            if worker_incarnation in failed_worker_incarnations:
                _fte_command_debug_log(
                    "execute_command_skipped_failed_worker",
                    command_index=command_index,
                    command_count=len(commands),
                    command_type=getattr(command, "command_type", type(command).__name__),
                    query_id=getattr(command, "query_id", ""),
                    fragment_id=getattr(command, "fragment_id", ""),
                    attempt_id=attempt_id,
                    **worker_debug_fields,
                )
                continue
            if not begin_fte_registry_operation(query_id):
                query_closed = True
                break
            try:
                scheduler = _FTE_SCHEDULERS.get_or_create(query_id)
                self._bind_fte_scheduler_handlers(scheduler)
                query_closing = _FteQueryClosingEvent(query_id)
                started_at = time.monotonic()
                command_type = getattr(command, "command_type", type(command).__name__)
                _fte_command_debug_log(
                    "execute_command_start",
                    command_index=command_index,
                    command_count=len(commands),
                    command_type=command_type,
                    query_id=getattr(command, "query_id", ""),
                    fragment_id=getattr(command, "fragment_id", ""),
                    partition_id=getattr(command, "partition_id", ""),
                    attempt_id=attempt_id,
                    **worker_debug_fields,
                )
                try:
                    if isinstance(command, FteCreateTaskCommand):
                        command.request["query_task_lease"] = fte_partition_task_lease_payload(
                            command.query_id,
                            command.fragment_id,
                            command.partition_id,
                            command.attempt_id,
                        )
                    scheduler.worker_command_executor.execute(
                        command,
                        cancel_event=query_closing,
                    )
                except FteSplitSubmissionCancelled as exc:
                    if query_closing.is_set():
                        query_closed = True
                    else:
                        abort_error = exc
                    break
                except FteSplitQueueTerminal as exc:
                    if query_closing.is_set():
                        query_closed = True
                        break
                    terminal_attempts.add(str(exc.attempt_id))
                    scheduler.enqueue(
                        TaskStatusChanged.from_status(
                            query_id,
                            exc.attempt_id,
                            exc.status,
                        ),
                        priority=True,
                    )
                    continue
                except QueryDeadlineExceeded as exc:
                    abort_error = exc
                    break
                except Exception as exc:
                    if query_closing.is_set():
                        query_closed = True
                        break
                    _fte_command_debug_log(
                        "execute_command_error",
                        command_index=command_index,
                        command_count=len(commands),
                        command_type=command_type,
                        query_id=getattr(command, "query_id", ""),
                        fragment_id=getattr(command, "fragment_id", ""),
                        partition_id=getattr(command, "partition_id", ""),
                        attempt_id=attempt_id,
                        elapsed_ms=int((time.monotonic() - started_at) * 1000),
                        error_type=type(exc).__name__,
                        error=exc,
                        **worker_debug_fields,
                    )
                    failure = fragment_execution.worker_control_failure_for_command(command, exc)
                    failed_worker_incarnations.add(worker_incarnation)
                    failures.append(failure)
                    if failure.worker_id:
                        # Fence immediately, before another command or thread can
                        # select the failed worker. The scheduler reconciliation
                        # remains deferred until this batch's healthy tail owns a
                        # terminal outcome.
                        quarantine_fte_worker(
                            failure.worker_id,
                            manager_instance_id=self.manager_instance_id,
                            worker_incarnation_id=failure.worker_incarnation_id,
                        )
                    continue
                try:
                    if isinstance(command, FteCreateTaskCommand):
                        command.worker.record_fte_task_started_from_reservation(
                            command.query_id,
                            command.fragment_id,
                            command.partition_id,
                            command.attempt_id,
                            command.request,
                        )
                    else:
                        fragment_execution.handle_worker_command_success(command)
                except Exception as exc:
                    abort_error = exc
                    break
                if query_closing.is_set():
                    query_closed = True
                    break
                if isinstance(command, FteCreateTaskCommand):
                    successful_create_attempts[str(command.attempt_id)] = command.scheduled_attempt
                _fte_command_debug_log(
                    "execute_command_done",
                    command_index=command_index,
                    command_count=len(commands),
                    command_type=command_type,
                    query_id=getattr(command, "query_id", ""),
                    fragment_id=getattr(command, "fragment_id", ""),
                    partition_id=getattr(command, "partition_id", ""),
                    attempt_id=attempt_id,
                    elapsed_ms=int((time.monotonic() - started_at) * 1000),
                    **worker_debug_fields,
                )
            finally:
                end_fte_registry_operation(query_id)

        recovery_handles: list[Any] = []
        for failure in failures:
            recovery_handles.extend(self._handles_for_fte_worker_control_failure(failure))

        if abort_error is not None:
            raise abort_error

        successful_scheduled_attempts: tuple[Any, ...] = ()
        if not query_closed:
            successful_scheduled_attempts = tuple(
                scheduled_attempt
                for scheduled_attempt in successful_create_attempts.values()
                if (
                    scheduled_attempt.worker_id,
                    scheduled_attempt.worker_incarnation_id,
                )
                not in failed_worker_incarnations
            )

        return FteWorkerCommandDispatchResult(
            scheduled_attempts=successful_scheduled_attempts,
            recovery_handles=tuple(recovery_handles),
            failures=tuple(failures),
            query_closed=query_closed,
        )

    def _execute_fte_fragment_execution_outbox(
        self,
        fragment_execution: FteFragmentExecution,
    ) -> FteWorkerCommandDispatchResult:
        # Worker failure reconciliation mutates fragment state before commands
        # are placed in the outbox.  Publish that state even when the mutation
        # produced no retry command (for example, the final failed sink input).
        _sync_write_sink_unit_for_fragment(fragment_execution)
        return self._execute_fte_fragment_execution_worker_commands(
            fragment_execution,
            fragment_execution.pop_worker_commands(),
        )

    def _execute_fte_fragment_execution_mutation_result(
        self,
        fragment_execution: FteFragmentExecution,
        result: Any,
    ) -> FteWorkerCommandDispatchResult:
        # Assignment can seal the dynamic partition set without scheduling a
        # task.  Synchronize before dispatch so a terminal write-sink unit still
        # advances query-resource allocation in that zero-command case.
        _sync_write_sink_unit_for_fragment(fragment_execution)
        return self._execute_fte_fragment_execution_worker_commands(
            fragment_execution,
            list(result.worker_commands),
        )

    def _handles_for_fte_scheduled_attempts(
        self,
        query_id: str,
        fragment_id: str,
        scheduled_attempts: list[Any],
    ) -> list[Any]:
        if not scheduled_attempts:
            return []
        query_id = str(query_id)
        if not begin_fte_registry_operation(query_id):
            return []
        try:
            fte_handle_cls = self._fte_task_handle_cls()
            handles: list[Any] = []
            watcher_requests: list[tuple[FteTaskAttemptId, Any]] = []
            for scheduled_attempt in scheduled_attempts:
                owner = (
                    self._fte_partition_owner(
                        query_id,
                        fragment_id,
                        scheduled_attempt.attempt_id.partition_id,
                    )
                    or self
                )
                handle = fte_handle_cls(scheduled_attempt.attempt_id, owner)
                task_context_info = dict(scheduled_attempt.descriptor.task_context_info)
                if (
                    "exchange_sink_instance" not in task_context_info
                    and scheduled_attempt.request.get("exchange_sink_instance") is not None
                ):
                    task_context_info["exchange_sink_instance"] = scheduled_attempt.request.get(
                        "exchange_sink_instance"
                    )
                if task_context_info:
                    handle.task_context_info = task_context_info
                query_task_lease = scheduled_attempt.request.get("query_task_lease")
                if not isinstance(query_task_lease, dict):
                    raise RuntimeError(f"scheduled FTE attempt {scheduled_attempt.attempt_id} has no query task lease")
                handle.query_task_lease = dict(query_task_lease)
                handles.append(handle)
                watcher_requests.append((scheduled_attempt.attempt_id, owner))
            # Make results visible before a watcher can publish terminal
            # status; the outer lifecycle token keeps teardown from observing
            # this publication half-complete.
            _store_fte_result_handles(
                query_id,
                handles,
                registry_operation_owned=True,
            )
            for attempt_id, owner in watcher_requests:
                self._start_fte_attempt_status_watcher(query_id, attempt_id, owner)
            with _FTE_REGISTRY_LOCK:
                if query_id in _FTE_CLOSING_QUERIES:
                    # Successful teardown owns registry removal.  Retain the
                    # handles if remote teardown fails and must be retried.
                    return []
            return handles
        finally:
            end_fte_registry_operation(query_id)

    def _start_fte_attempt_status_watcher(
        self,
        query_id: str,
        attempt_id: FteTaskAttemptId,
        worker_handle: Any,
    ) -> None:
        query_id = str(query_id)
        if not begin_fte_registry_operation(query_id):
            return
        try:
            self._start_fte_attempt_status_watcher_while_registry_open(
                query_id,
                attempt_id,
                worker_handle,
            )
        finally:
            end_fte_registry_operation(query_id)

    def _start_fte_attempt_status_watcher_while_registry_open(
        self,
        query_id: str,
        attempt_id: FteTaskAttemptId,
        worker_handle: Any,
    ) -> None:
        query_id = str(query_id)
        with _FTE_REGISTRY_LOCK:
            if query_id in _FTE_CLOSING_QUERIES:
                return
        scheduler = _FTE_SCHEDULERS.get_or_create(query_id)
        self._bind_fte_scheduler_handlers(scheduler)
        watcher = FteAttemptStatusWatcher(
            scheduler=scheduler,
            attempt_id=attempt_id,
            worker=worker_handle,
            wait_timeout_s=fte_status_wait_timeout_s(),
        )
        attempt_key = str(attempt_id)

        def unregister(exited_watcher: FteAttemptStatusWatcher) -> None:
            with _FTE_REGISTRY_LOCK:
                if _FTE_STATUS_WATCHERS.get(attempt_key) is exited_watcher:
                    _FTE_STATUS_WATCHERS.pop(attempt_key, None)

        watcher.on_exit = unregister
        with _FTE_REGISTRY_LOCK:
            previous = _FTE_STATUS_WATCHERS.get(attempt_key)
        if previous is not None:
            previous.stop()
            previous.join(previous.shutdown_timeout_s())
            if previous.is_alive():
                raise RuntimeError(f"previous FTE status watcher did not stop: {attempt_key}")
        with _FTE_REGISTRY_LOCK:
            if query_id in _FTE_CLOSING_QUERIES:
                return
            current = _FTE_STATUS_WATCHERS.get(attempt_key)
            if current is previous:
                _FTE_STATUS_WATCHERS.pop(attempt_key, None)
            elif current is not None:
                raise RuntimeError(f"concurrent FTE status watcher registration: {attempt_key}")
            _FTE_STATUS_WATCHERS[attempt_key] = watcher
            try:
                watcher.start()
            except Exception:
                if _FTE_STATUS_WATCHERS.get(attempt_key) is watcher:
                    _FTE_STATUS_WATCHERS.pop(attempt_key, None)
                raise
