# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING, Any

from vane.runners.fte import fte_status_wait_timeout_s
from vane.runners.fte.fte_events import FteCreateTaskCommand
from vane.runners.fte.fte_scheduler import FteAttemptStatusWatcher
from vane.runners.ray.fragment_registry import (
    _FTE_CLOSING_QUERIES,
    _FTE_REGISTRY_LOCK,
    _FTE_SCHEDULERS,
    _FTE_STATUS_WATCHERS,
)
from vane.runners.ray.fte_fragment_scheduler import (
    _store_fte_result_handles,
    begin_fte_registry_operation,
    end_fte_registry_operation,
    fte_partition_task_lease_payload,
)

if TYPE_CHECKING:
    from vane.runners.fte import FteFragmentExecution, FteTaskAttemptId


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


class FteWorkerCommandMixin:
    def _execute_fte_fragment_execution_worker_commands(
        self,
        fragment_execution: FteFragmentExecution,
        worker_commands: list[Any] | tuple[Any, ...],
    ) -> None:
        for command_index, command in enumerate(worker_commands):
            query_id = str(command.query_id)
            if not begin_fte_registry_operation(query_id):
                return
            try:
                scheduler = _FTE_SCHEDULERS.get_or_create(query_id)
                self._bind_fte_scheduler_handlers(scheduler)
                started_at = time.monotonic()
                command_type = getattr(command, "command_type", type(command).__name__)
                attempt_id = getattr(command, "attempt_id", None)
                _fte_command_debug_log(
                    "execute_command_start",
                    command_index=command_index,
                    command_count=len(worker_commands),
                    command_type=command_type,
                    query_id=getattr(command, "query_id", ""),
                    fragment_id=getattr(command, "fragment_id", ""),
                    partition_id=getattr(command, "partition_id", ""),
                    attempt_id=attempt_id,
                )
                try:
                    if isinstance(command, FteCreateTaskCommand):
                        command.request["query_task_lease"] = fte_partition_task_lease_payload(
                            command.query_id,
                            command.fragment_id,
                            command.partition_id,
                            command.attempt_id,
                        )
                    scheduler.worker_command_executor.execute(command)
                except Exception as exc:
                    _fte_command_debug_log(
                        "execute_command_error",
                        command_index=command_index,
                        command_count=len(worker_commands),
                        command_type=command_type,
                        query_id=getattr(command, "query_id", ""),
                        fragment_id=getattr(command, "fragment_id", ""),
                        partition_id=getattr(command, "partition_id", ""),
                        attempt_id=attempt_id,
                        elapsed_ms=int((time.monotonic() - started_at) * 1000),
                        error_type=type(exc).__name__,
                        error=exc,
                    )
                    raise fragment_execution.worker_control_failure_for_command(command, exc) from exc
                if isinstance(command, FteCreateTaskCommand):
                    self._fte_worker_placement_manager.release_reservation(
                        query_id=command.query_id,
                        fragment_id=command.fragment_id,
                        partition_id=command.partition_id,
                    )
                fragment_execution.handle_worker_command_success(command)
                _fte_command_debug_log(
                    "execute_command_done",
                    command_index=command_index,
                    command_count=len(worker_commands),
                    command_type=command_type,
                    query_id=getattr(command, "query_id", ""),
                    fragment_id=getattr(command, "fragment_id", ""),
                    partition_id=getattr(command, "partition_id", ""),
                    attempt_id=attempt_id,
                    elapsed_ms=int((time.monotonic() - started_at) * 1000),
                )
            finally:
                end_fte_registry_operation(query_id)

    def _execute_fte_fragment_execution_outbox(self, fragment_execution: FteFragmentExecution) -> None:
        self._execute_fte_fragment_execution_worker_commands(
            fragment_execution, fragment_execution.pop_worker_commands()
        )

    def _execute_fte_fragment_execution_mutation_result(
        self,
        fragment_execution: FteFragmentExecution,
        result: Any,
    ) -> list[Any]:
        self._execute_fte_fragment_execution_worker_commands(fragment_execution, list(result.worker_commands))
        return list(result)

    def _handles_for_fte_scheduled_attempts(
        self,
        query_id: str,
        fragment_id: str,
        scheduled_attempts: list[Any],
    ) -> list[Any]:
        with _FTE_REGISTRY_LOCK:
            if str(query_id) in _FTE_CLOSING_QUERIES:
                return []
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
                task_context_info["exchange_sink_instance"] = scheduled_attempt.request.get("exchange_sink_instance")
            if task_context_info:
                handle.task_context_info = task_context_info
            query_task_lease = scheduled_attempt.request.get("query_task_lease")
            if not isinstance(query_task_lease, dict):
                raise RuntimeError(f"scheduled FTE attempt {scheduled_attempt.attempt_id} has no query task lease")
            handle.query_task_lease = dict(query_task_lease)
            handles.append(handle)
            watcher_requests.append((scheduled_attempt.attempt_id, owner))
        _store_fte_result_handles(query_id, handles)
        for attempt_id, owner in watcher_requests:
            self._start_fte_attempt_status_watcher(query_id, attempt_id, owner)
        return handles

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
