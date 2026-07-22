# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vane.runners.fte import (
    FteTaskState,
    FteWorkerControlFailure,
    FteWorkerReservationUnavailable,
)
from vane.runners.fte.fte_events import WorkerFailed
from vane.runners.fte.fte_scheduler import FteEventHandlers
from vane.runners.ray.fragment_registry import (
    _FTE_CLOSING_QUERIES,
    _FTE_FRAGMENT_EXECUTIONS,
    _FTE_FRAGMENT_STATES,
    _FTE_REGISTRY_LOCK,
    _FTE_SCHEDULERS,
    _FTE_STATUS_WATCHERS,
)
from vane.runners.ray.fragment_worker_exchange import apply_exchange_selector_update
from vane.runners.ray.fragment_worker_failures import mark_fte_worker_failed_for_event
from vane.runners.ray.fragment_worker_ordering import fragment_execution_key_for_fte_attempt
from vane.runners.ray.fragment_worker_reservations import (
    fte_worker_reservation_event_state,
    remove_pending_fte_worker_reservation_if_current,
)
from vane.runners.ray.fte_fragment_scheduler import FteWorkerPlacementManager

if TYPE_CHECKING:
    from vane.runners.fte.fte_events import ExchangeSelectorUpdated, TaskStatusChanged, WorkerReservationCompleted


class FteWorkerEventHandlingMixin:
    def _handles_for_fte_worker_control_failure(
        self,
        failure: FteWorkerControlFailure,
    ) -> list[Any]:
        query_id = failure.attempt_id.task_id.query_id
        with _FTE_REGISTRY_LOCK:
            if query_id in _FTE_CLOSING_QUERIES:
                return []
        scheduler = _FTE_SCHEDULERS.get(query_id)
        if scheduler is None:
            return []
        self._bind_fte_scheduler_handlers(scheduler)
        scheduler.enqueue(
            WorkerFailed(
                query_id,
                failure.worker_id,
                failure,
            )
        )
        scheduler.drain()
        return []

    def _handles_for_marked_fte_worker_failed(
        self,
        scheduled_by_stage: list[tuple[str, str, list[Any], list[Any]]],
    ) -> list[Any]:
        handles: list[Any] = []
        for query_id, fragment_id, scheduled_attempts, _ in scheduled_by_stage:
            with _FTE_REGISTRY_LOCK:
                if query_id in _FTE_CLOSING_QUERIES:
                    continue
                fragment_execution = _FTE_FRAGMENT_EXECUTIONS.get((query_id, fragment_id))
            if fragment_execution is not None:
                try:
                    self._execute_fte_fragment_execution_outbox(fragment_execution)
                except FteWorkerControlFailure as exc:
                    handles.extend(self._handles_for_fte_worker_control_failure(exc))
                    continue
            handles.extend(
                self._handles_for_fte_scheduled_attempts(
                    query_id,
                    fragment_id,
                    scheduled_attempts,
                )
            )
        return handles

    def _handles_for_worker_failed_event(self, event: WorkerFailed) -> list[Any]:
        with _FTE_REGISTRY_LOCK:
            if str(event.query_id) in _FTE_CLOSING_QUERIES:
                return []
        scheduled_by_stage = mark_fte_worker_failed_for_event(event)
        if not scheduled_by_stage:
            return []
        return self._handles_for_marked_fte_worker_failed(scheduled_by_stage)

    def _handles_for_worker_reservation_completed_event(self, event: WorkerReservationCompleted) -> list[Any]:
        with _FTE_REGISTRY_LOCK:
            if str(event.query_id) in _FTE_CLOSING_QUERIES:
                return []
        key, future, fragment_execution = fte_worker_reservation_event_state(event)
        if future is None or fragment_execution is None:
            return []
        if event.error is not None:
            remove_pending_fte_worker_reservation_if_current(key, future)
            raise RuntimeError(f"FTE worker reservation failed: {event.error}")
        try:
            future.result()
        except Exception as exc:
            remove_pending_fte_worker_reservation_if_current(key, future)
            raise RuntimeError(f"FTE worker reservation failed: {exc}") from exc
        partition = fragment_execution.partitions.get(int(event.partition_id))
        if partition is None or partition.running_attempt is not None or partition.finished or partition.failed:
            remove_pending_fte_worker_reservation_if_current(key, future)
            FteWorkerPlacementManager.release_owner(
                query_id=key[0],
                fragment_id=key[1],
                partition_id=key[2],
            )
            return []
        remove_pending_fte_worker_reservation_if_current(key, future)
        try:
            scheduled = fragment_execution.start_attempt_with_worker(partition)
            self._execute_fte_fragment_execution_outbox(fragment_execution)
        except FteWorkerControlFailure as exc:
            return self._handles_for_fte_worker_control_failure(exc)
        except FteWorkerReservationUnavailable:
            return self._drain_fte_pending_tasks(query_id_filter=event.query_id)
        handles = self._handles_for_fte_scheduled_attempts(
            event.query_id,
            str(event.fragment_id),
            [scheduled],
        )
        handles.extend(self._drain_fte_pending_tasks())
        return handles

    def _handles_for_task_status_changed_event(self, event: TaskStatusChanged) -> list[Any]:
        raw_state = event.status.get("state")
        state = raw_state if isinstance(raw_state, FteTaskState) else FteTaskState(str(raw_state))
        if state in {
            FteTaskState.FINISHED,
            FteTaskState.FAILED,
            FteTaskState.CANCELED,
            FteTaskState.ABORTED,
        }:
            with _FTE_REGISTRY_LOCK:
                watcher = _FTE_STATUS_WATCHERS.get(str(event.attempt_id))
            if watcher is not None:
                watcher.stop()
        fragment_execution_key = fragment_execution_key_for_fte_attempt(event.attempt_id)
        if fragment_execution_key is None:
            return []
        query_id, fragment_id = fragment_execution_key
        with _FTE_REGISTRY_LOCK:
            if query_id in _FTE_CLOSING_QUERIES:
                return []
            fragment_execution = _FTE_FRAGMENT_EXECUTIONS.get(fragment_execution_key)
        if fragment_execution is None:
            return []
        try:
            scheduled = fragment_execution.handle_task_status(event.status)
            with _FTE_REGISTRY_LOCK:
                if query_id in _FTE_CLOSING_QUERIES:
                    return []
            if scheduled is not None:
                self._execute_fte_fragment_execution_outbox(fragment_execution)
        except FteWorkerControlFailure as exc:
            return self._handles_for_fte_worker_control_failure(exc)
        if scheduled is None:
            return []
        return self._handles_for_fte_scheduled_attempts(
            query_id,
            fragment_id,
            [scheduled],
        )

    def _handles_for_exchange_selector_updated_event(self, event: ExchangeSelectorUpdated) -> list[Any]:
        query_id = str(event.query_id)
        fragment_id = str(event.consumer_fragment_id)
        source_node_id = str(event.source_node_id)
        with _FTE_REGISTRY_LOCK:
            if query_id in _FTE_CLOSING_QUERIES:
                return []
            fragment_execution = _FTE_FRAGMENT_EXECUTIONS.get((query_id, fragment_id))
            fragment_state = _FTE_FRAGMENT_STATES.get((query_id, fragment_id))
        if fragment_execution is None or fragment_state is None or fragment_state.assigner is None:
            return []

        selector_update = apply_exchange_selector_update(fragment_state, event)
        if selector_update is None:
            return []
        selector_snapshot, splits = selector_update
        result = fragment_state.assigner.assign(
            source_node_id,
            [split.to_dict() for split in splits],
            no_more_inputs=selector_snapshot.final,
        )
        for partition_info in result.partitions_added:
            partition = fragment_execution.add_partition(
                partition_info.partition_id,
                partition_info.node_requirements,
            )
            self._try_reserve_fte_partition_for_node_wait(
                query_id,
                fragment_id,
                partition,
                fragment_execution=fragment_execution,
            )
        try:
            scheduled_result = fragment_execution.apply_assignment_result(result)
            with _FTE_REGISTRY_LOCK:
                if query_id in _FTE_CLOSING_QUERIES:
                    return []
            scheduled = self._execute_fte_fragment_execution_mutation_result(fragment_execution, scheduled_result)
        except FteWorkerControlFailure as exc:
            return self._handles_for_fte_worker_control_failure(exc)

        handles = self._handles_for_fte_scheduled_attempts(
            query_id,
            fragment_id,
            scheduled,
        )
        if selector_snapshot.final:
            with _FTE_REGISTRY_LOCK:
                fragment_state.exhausted_source_node_ids.add(source_node_id)
        return handles

    def _bind_fte_scheduler_handlers(self, scheduler) -> None:
        scheduler.set_handlers(
            FteEventHandlers(
                on_split_events=self._submit_fte_pending_tasks,
                on_source_input_exhausted=lambda source_ids, query_id=scheduler.query_id: (
                    self._task_input_stream_exhausted_direct(
                        source_ids,
                        query_id_filter=query_id,
                    )
                ),
                on_task_status_changed=self._handles_for_task_status_changed_event,
                on_worker_failed=self._handles_for_worker_failed_event,
                on_memory_pressure_detected=lambda event: self._revoke_fte_speculative_tasks_for_memory_pressure_direct(
                    max_count_per_worker=event.max_count_per_worker,
                    query_id_filter=event.query_id,
                ),
                on_resource_admission_changed=lambda event: self._drain_fte_pending_tasks(
                    query_id_filter=event.query_id
                ),
                on_worker_reservation_completed=self._handles_for_worker_reservation_completed_event,
                on_retry_delay_expired=lambda event: self._drain_fte_pending_tasks(query_id_filter=event.query_id),
                on_exchange_selector_updated=self._handles_for_exchange_selector_updated_event,
            )
        )
