# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from vane.runners.fte.fte_attempts import (
    ExecutionClassTransition,
    FinishedAttempt,
    FragmentExecutionMutationResult,
    ReadyTask,
    RevokedAttempt,
    RunningAttempt,
    ScheduledAttempt,
)
from vane.runners.fte.fte_descriptor import (
    FteTaskUpdateRequest,
    TaskDescriptor,
    TaskDescriptorStorage,
)
from vane.runners.fte.fte_events import (
    FteAddSplitsCommand,
    FteCreateTaskCommand,
    FteNoMoreSplitsCommand,
    FteTaskUpdateCommand,
)
from vane.runners.fte.fte_exchange import (
    _sink_instance_payload,
    derive_exchange_sink_instance_for_attempt,
)
from vane.runners.fte.fte_failures import (
    _failure_allows_retry,
    _is_memory_failure,
    _missing_output_stats_failure,
)
from vane.runners.fte.fte_split_assigner import (
    PartitionUpdate,
    _normalize_sources,
)
from vane.runners.fte.fte_state import (
    FtePartitionState,
    FteTaskExecutionClass,
    FteTaskState,
    fte_task_execution_class_from_metadata,
    fte_task_execution_class_metadata_present,
)
from vane.runners.fte.fte_types import FteTaskAttemptId, FteTaskId, _check_non_negative
from vane.runners.fte.fte_update_batch import (
    _estimated_payload_bytes,
    fte_task_update_max_payload_bytes,
    fte_task_update_max_splits,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from vane.runners.fte.fte_events import FteWorkerCommand
    from vane.runners.fte.fte_exchange import ExchangeSinkHandle, FteExchangeTracker
    from vane.runners.fte.fte_split_assigner import AssignmentResult, NodeRequirements
    from vane.runners.fte.fte_types import FteSplit


class FteWorkerControlFailure(RuntimeError):
    def __init__(
        self,
        *,
        worker_id: str | None,
        attempt_id: FteTaskAttemptId,
        method_name: str,
        cause: BaseException,
    ) -> None:
        self.worker_id = str(worker_id or "")
        self.attempt_id = attempt_id
        self.method_name = str(method_name)
        self.cause = cause
        super().__init__(
            f"FTE worker {self.worker_id or '<unknown>'} failed during "
            f"{self.method_name} for {self.attempt_id}: {cause}"
        )


class FteWorkerReservationUnavailable(RuntimeError):
    def __init__(
        self,
        *,
        query_id: str,
        fragment_id: str,
        partition_id: int,
        memory_requirement_bytes: int | None,
        blocked_reason: str = "",
    ) -> None:
        self.query_id = str(query_id)
        self.fragment_id = str(fragment_id)
        self.partition_id = int(partition_id)
        self.memory_requirement_bytes = memory_requirement_bytes
        self.blocked_reason = str(blocked_reason)
        super().__init__(
            "no FTE worker reservation available for "
            f"{self.query_id}/{self.fragment_id}/{self.partition_id} "
            f"memory_requirement_bytes={memory_requirement_bytes} "
            f"blocked_reason={self.blocked_reason or '<none>'}"
        )


class FteTaskPartition:
    def __init__(
        self,
        task_id: FteTaskId | str | Mapping[str, Any],
        descriptor: TaskDescriptor,
        *,
        max_attempts: int = 4,
        sink_handle: ExchangeSinkHandle | None = None,
        node_requirements: NodeRequirements | None = None,
        memory_requirement_bytes: int | None = None,
        execution_class: FteTaskExecutionClass | str | None = None,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.task_id = FteTaskId.coerce(task_id)
        if descriptor.task_id != self.task_id:
            raise ValueError("descriptor task_id does not match partition")
        self.descriptor = descriptor
        self.max_attempts = int(max_attempts)
        self.remaining_attempts = int(max_attempts)
        self.sink_handle = sink_handle
        self.node_requirements = node_requirements
        self.memory_requirement_bytes = (
            None if memory_requirement_bytes is None else max(0, int(memory_requirement_bytes))
        )
        self.execution_class = FteTaskExecutionClass.coerce(execution_class)
        self.sealed = bool(descriptor.sealed)
        self.finished = False
        self.failed = False
        self.failure_observed = False
        self.running_attempts: dict[int, RunningAttempt] = {}
        self.running_task_stats: dict[int, Any] = {}
        self.selected_attempt: int | None = None
        self.finished_attempts: dict[int, FinishedAttempt] = {}
        self.selected_output_stats: Any = None
        self.failures: list[Any] = []
        self.ready_for_scheduling = bool(self.sealed)
        self.execution_ready_deferred = False
        self.node_wait_started_at: float | None = None
        self.no_matching_node_started_at: float | None = None

    @property
    def state(self) -> FtePartitionState:
        if self.finished:
            return FtePartitionState.FINISHED
        if self.failed:
            return FtePartitionState.FAILED
        if self.running_attempts:
            return FtePartitionState.RUNNING
        if self.sealed:
            return FtePartitionState.SEALED
        return FtePartitionState.OPEN

    @property
    def running_attempt(self) -> RunningAttempt | None:
        if not self.running_attempts:
            return None
        if len(self.running_attempts) != 1:
            raise RuntimeError(f"partition {self.task_id} has multiple running attempts")
        return next(iter(self.running_attempts.values()))

    def next_attempt_number(self) -> int:
        if self.remaining_attempts <= 0:
            raise RuntimeError(f"partition {self.task_id} has no attempts remaining")
        return self.max_attempts - self.remaining_attempts

    def start_attempt(
        self,
        *,
        sink_instance: Any = None,
        worker_id: str | None = None,
        remote_handle: Any = None,
    ) -> ScheduledAttempt:
        if self.finished or self.failed:
            raise RuntimeError(f"cannot schedule terminal partition {self.task_id}")
        if self.running_attempts:
            raise RuntimeError(f"partition {self.task_id} already has a running attempt")

        attempt_number = self.next_attempt_number()
        attempt_id = FteTaskAttemptId(self.task_id, attempt_number)
        request = self.descriptor.to_create_task_request(
            attempt_number,
            exchange_sink_instance=sink_instance,
        )
        if self.memory_requirement_bytes is not None:
            request["memory_requirement_bytes"] = self.memory_requirement_bytes
        request["execution_class"] = self.execution_class.value
        self.execution_ready_deferred = False
        self.node_wait_started_at = None
        self.no_matching_node_started_at = None
        self.running_attempts[attempt_number] = RunningAttempt(
            attempt_id,
            worker_id=worker_id,
            remote_handle=remote_handle,
            sink_instance=sink_instance,
        )
        self.running_task_stats.pop(attempt_number, None)
        return ScheduledAttempt(
            attempt_id=attempt_id,
            descriptor=self.descriptor,
            request=request,
            sink_instance=sink_instance,
            worker_id=worker_id,
        )

    def mark_waiting_for_node(self) -> None:
        if self.node_wait_started_at is None:
            self.node_wait_started_at = time.time()

    def mark_no_matching_node(self) -> float:
        if self.no_matching_node_started_at is None:
            self.no_matching_node_started_at = time.time()
        return time.time() - self.no_matching_node_started_at

    def reset_no_matching_node(self) -> None:
        self.no_matching_node_started_at = None

    def mark_ready_for_execution(self) -> None:
        self.ready_for_scheduling = True
        self.execution_ready_deferred = False

    def defer_ready_for_execution(self) -> None:
        if self.finished or self.failed or self.running_attempts:
            return
        self.ready_for_scheduling = False
        self.execution_ready_deferred = True

    def seal(self) -> FteTaskExecutionClass | None:
        if self.finished or self.failed:
            raise RuntimeError(f"cannot seal terminal partition {self.task_id}")
        old_class = self.execution_class
        self.sealed = True
        self.descriptor.sealed = True
        if old_class.is_speculative:
            self.execution_class = FteTaskExecutionClass.STANDARD
            self.mark_ready_for_execution()
            return old_class
        self.mark_ready_for_execution()
        return None

    def set_execution_class(self, execution_class: FteTaskExecutionClass | str | None) -> bool:
        new_class = FteTaskExecutionClass.coerce(execution_class)
        if self.execution_class == new_class:
            return False
        if not self.execution_class.can_transition_to(new_class):
            raise ValueError(
                f"cannot change partition {self.task_id} execution class "
                f"from {self.execution_class.value} to {new_class.value}"
            )
        self.execution_class = new_class
        return True

    def mark_attempt_finished(
        self,
        attempt_id: FteTaskAttemptId | str | Mapping[str, Any],
        output_stats: Any = None,
    ) -> bool:
        attempt = FteTaskAttemptId.coerce(attempt_id)
        self._validate_attempt_task(attempt)
        running = self.running_attempts.pop(attempt.attempt_id, None)
        if running is None:
            return False
        task_stats = self.running_task_stats.pop(attempt.attempt_id, None)
        self.finished = True
        self.failed = False
        self.selected_attempt = attempt.attempt_id
        if isinstance(task_stats, Mapping) and isinstance(output_stats, Mapping):
            selected = dict(task_stats)
            selected.update(output_stats)
            self.selected_output_stats = selected
        elif output_stats is not None:
            self.selected_output_stats = output_stats
        else:
            self.selected_output_stats = task_stats
        self.finished_attempts[attempt.attempt_id] = FinishedAttempt(
            attempt,
            sink_instance=running.sink_instance,
            output_stats=self.selected_output_stats,
        )
        self.running_attempts.clear()
        self.running_task_stats.clear()
        return True

    def mark_attempt_failed(
        self,
        attempt_id: FteTaskAttemptId | str | Mapping[str, Any],
        error: Any = None,
        *,
        retryable: bool = True,
    ) -> ReadyTask | None:
        attempt = FteTaskAttemptId.coerce(attempt_id)
        self._validate_attempt_task(attempt)
        if self.finished:
            return None
        running = self.running_attempts.pop(attempt.attempt_id, None)
        if running is None:
            return None
        self.running_task_stats.pop(attempt.attempt_id, None)

        self.failure_observed = True
        self.failures.append(error)
        self.remaining_attempts = max(0, self.remaining_attempts - 1)
        if retryable and self.remaining_attempts > 0:
            return ReadyTask(self.task_id, "retry")
        self.failed = True
        return None

    def update_running_task_stats(
        self,
        attempt_id: FteTaskAttemptId | str | Mapping[str, Any],
        task_stats: Any,
    ) -> bool:
        if not isinstance(task_stats, Mapping):
            return False
        attempt = FteTaskAttemptId.coerce(attempt_id)
        self._validate_attempt_task(attempt)
        if attempt.attempt_id not in self.running_attempts:
            return False
        self.running_task_stats[attempt.attempt_id] = dict(task_stats)
        return True

    def revoke_attempt(
        self,
        attempt_id: FteTaskAttemptId | str | Mapping[str, Any],
        error: Any = None,
    ) -> RevokedAttempt | None:
        attempt = FteTaskAttemptId.coerce(attempt_id)
        self._validate_attempt_task(attempt)
        if self.finished:
            return None
        running = self.running_attempts.pop(attempt.attempt_id, None)
        if running is None:
            return None

        self.failure_observed = True
        self.failures.append(error)
        self.remaining_attempts = max(0, self.remaining_attempts - 1)
        if self.remaining_attempts <= 0:
            self.failed = True
            self.ready_for_scheduling = False
            retry_ready = False
        else:
            retry_ready = self.sealed
            self.ready_for_scheduling = retry_ready
        return RevokedAttempt(
            attempt_id=attempt,
            worker_id=running.worker_id,
            remote_handle=running.remote_handle,
            retry_ready=retry_ready,
        )

    def _validate_attempt_task(self, attempt: FteTaskAttemptId) -> None:
        if attempt.task_id != self.task_id:
            raise ValueError("attempt task_id does not match partition")


_TASK_STATUS_PROGRESS_KEYS = (
    "submitted_split_count",
    "queued_split_count",
    "consumed_split_count",
    "completed_split_count",
    "submitted_split_count_by_source",
    "queued_split_count_by_source",
    "consumed_split_count_by_source",
    "completed_split_count_by_source",
    "submitted_split_bytes",
    "submitted_split_bytes_by_source",
    "queued_split_bytes",
    "queued_split_bytes_by_source",
    "consumed_split_bytes",
    "consumed_split_bytes_by_source",
    "completed_split_bytes",
    "completed_split_bytes_by_source",
    "queue_wait_ms",
    "queue_wait_ms_by_source",
    "submitted_input_rows",
    "submitted_input_rows_by_source",
    "submitted_input_bytes",
    "submitted_input_bytes_by_source",
    "consumed_input_rows",
    "consumed_input_rows_by_source",
    "consumed_input_bytes",
    "consumed_input_bytes_by_source",
    "completed_input_rows",
    "completed_input_rows_by_source",
    "completed_input_bytes",
    "completed_input_bytes_by_source",
    "udf_running_task_count",
    "udf_queued_task_count",
    "udf_max_running_tasks",
    "udf_completed_rows",
    "udf_completed_bytes",
    "udf_emitted_rows",
    "udf_emitted_bytes",
)


def _task_status_progress_stats(status: Mapping[str, Any]) -> dict[str, Any]:
    return {key: status[key] for key in _TASK_STATUS_PROGRESS_KEYS if key in status and status[key] is not None}


class FteFragmentExecution:
    def __init__(
        self,
        query_id: str,
        fragment_execution_id: int,
        *,
        fragment_id: str,
        worker: Any = None,
        worker_id: str | None = None,
        worker_selector: Callable[[FteTaskPartition], Any] | None = None,
        execution_class_transition_callback: Callable[[list[ExecutionClassTransition]], None] | None = None,
        execution_admission_callback: Callable[[FteTaskPartition], bool] | None = None,
        attempt_admission_callback: Callable[[FteTaskPartition], bool] | None = None,
        worker_reservation_callback: Callable[[FteTaskPartition], bool] | None = None,
        descriptor_storage: TaskDescriptorStorage | None = None,
        max_attempts: int = 4,
        exchange: FteExchangeTracker | None = None,
        context: Mapping[str, Any] | None = None,
        resource_request: Mapping[str, Any] | None = None,
        fragment_plan: Any = None,
        fragment_registration_result: Any = None,
        task_context_info: Mapping[str, Any] | None = None,
        source_node_ids: set[str] | list[str] | tuple[str, ...] | None = None,
        dynamic_scan_source_node_ids: set[str] | list[str] | tuple[str, ...] | None = None,
        dynamic_exchange_source_node_ids: set[str] | list[str] | tuple[str, ...] | None = None,
        task_memory_bytes: int,
    ) -> None:
        self.query_id = str(query_id).strip()
        if not self.query_id:
            raise ValueError("query_id must be non-empty")
        self.fragment_execution_id = _check_non_negative("fragment_execution_id", fragment_execution_id)
        self.fragment_id = str(fragment_id).strip()
        if not self.fragment_id:
            raise ValueError("fragment_id must be non-empty")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.worker = worker
        self.worker_id = worker_id
        self.worker_selector = worker_selector
        self.execution_class_transition_callback = execution_class_transition_callback
        self.execution_admission_callback = execution_admission_callback
        self.attempt_admission_callback = attempt_admission_callback
        self.worker_reservation_callback = worker_reservation_callback
        self.descriptor_storage = descriptor_storage if descriptor_storage is not None else TaskDescriptorStorage()
        self.max_attempts = int(max_attempts)
        self.exchange = exchange
        self.context = dict(context or {})
        self.resource_request = dict(resource_request or {})
        self.fragment_plan = fragment_plan
        self.fragment_registration_result = fragment_registration_result
        self.task_context_info = dict(task_context_info or {})
        explicit_execution_class = fte_task_execution_class_metadata_present(
            self.context,
            self.resource_request,
            self.task_context_info,
        )
        self.execution_class = fte_task_execution_class_from_metadata(
            self.context,
            self.resource_request,
            self.task_context_info,
        )
        self.source_node_ids = _normalize_sources(source_node_ids)
        self.dynamic_scan_source_node_ids = _normalize_sources(dynamic_scan_source_node_ids)
        self.dynamic_exchange_source_node_ids = _normalize_sources(dynamic_exchange_source_node_ids)
        if not explicit_execution_class and self.dynamic_exchange_source_node_ids:
            self.execution_class = FteTaskExecutionClass.SPECULATIVE
        self.partitions: dict[int, FteTaskPartition] = {}
        self.failed = False
        self.no_more_partitions = False
        self._state_lock = threading.RLock()
        self._current_worker_commands: list[Any] | None = None
        self._worker_command_outbox: list[Any] = []
        self.task_memory_bytes = int(task_memory_bytes)
        if self.task_memory_bytes <= 0:
            raise ValueError("FTE task_memory_bytes must be positive")

    def add_partition(
        self,
        partition_id: int,
        node_requirements: NodeRequirements | None = None,
    ) -> FteTaskPartition:
        partition_id = _check_non_negative("partition_id", partition_id)
        existing = self.partitions.get(partition_id)
        if existing is not None:
            if existing.node_requirements is None and node_requirements is not None:
                existing.node_requirements = node_requirements
            return existing
        task_id = FteTaskId(self.query_id, self.fragment_execution_id, partition_id)
        sink_handle = self.exchange.add_sink(partition_id) if self.exchange is not None else None
        descriptor = TaskDescriptor(
            task_id,
            self.fragment_id,
            context=self.context,
            resource_request=self.resource_request,
            fragment_plan=self.fragment_plan,
            fragment_registration_result=self.fragment_registration_result,
            exchange_sink_instance=self._base_sink_instance_for_partition(),
            task_context_info=self.task_context_info,
            source_node_ids=set(self.source_node_ids),
            dynamic_scan_source_node_ids=set(self.dynamic_scan_source_node_ids),
            dynamic_exchange_source_node_ids=set(self.dynamic_exchange_source_node_ids),
        )
        partition = FteTaskPartition(
            task_id,
            descriptor,
            max_attempts=self.max_attempts,
            sink_handle=sink_handle,
            node_requirements=node_requirements,
            memory_requirement_bytes=self.task_memory_bytes,
            execution_class=self.execution_class,
        )
        self.partitions[partition_id] = partition
        self.descriptor_storage.put(task_id, descriptor)
        return partition

    def _base_sink_instance_for_partition(self) -> Any:
        if self.exchange is not None:
            return None
        return _sink_instance_payload(self.task_context_info.get("exchange_sink_instance")) or None

    def apply_assignment_result(self, result: AssignmentResult) -> FragmentExecutionMutationResult:
        with self._state_lock:
            previous_worker_commands = self._current_worker_commands
            worker_commands: list[Any] = []
            self._current_worker_commands = worker_commands
            scheduled: list[ScheduledAttempt] = []
            try:
                for info in result.partitions_added:
                    self.add_partition(info.partition_id, info.node_requirements)
                for update in self._coalesce_partition_updates(result.partition_updates):
                    attempt = self.update_partition(update)
                    if attempt is not None:
                        scheduled.append(attempt)
                for partition_id in result.sealed_partitions:
                    attempt = self.seal_partition(partition_id)
                    if attempt is not None:
                        scheduled.append(attempt)
                if result.no_more_partitions:
                    self.no_more_partitions = True
                return FragmentExecutionMutationResult.from_attempts(scheduled, worker_commands)
            finally:
                self._current_worker_commands = previous_worker_commands

    def apply_task_update(
        self,
        partition_id: int,
        update: FteTaskUpdateRequest | Mapping[str, Any] | None,
    ) -> FragmentExecutionMutationResult:
        with self._state_lock:
            previous_worker_commands = self._current_worker_commands
            worker_commands: list[Any] = []
            self._current_worker_commands = worker_commands
            try:
                partition = self.add_partition(partition_id)
                update_request = FteTaskUpdateRequest.coerce(update)
                if not partition.descriptor.apply_task_update(update_request):
                    return FragmentExecutionMutationResult(worker_commands=tuple(worker_commands))
                self.descriptor_storage.put(partition.task_id, partition.descriptor)
                running = partition.running_attempt
                if running is not None:
                    worker = running.remote_handle
                    self._record_worker_command(
                        FteTaskUpdateCommand(
                            query_id=self.query_id,
                            fragment_id=self.fragment_id,
                            worker_id=running.worker_id,
                            worker=worker,
                            attempt_id=running.attempt_id,
                            update=update_request.to_dict(),
                        )
                    )
                return FragmentExecutionMutationResult(worker_commands=tuple(worker_commands))
            finally:
                self._current_worker_commands = previous_worker_commands

    @staticmethod
    def _partition_ready_to_schedule(partition: FteTaskPartition) -> bool:
        return (
            partition.ready_for_scheduling
            and partition.running_attempt is None
            and not partition.finished
            and not partition.failed
        )

    @staticmethod
    def _partition_matches_execution_class(
        partition: FteTaskPartition,
        execution_class: FteTaskExecutionClass | str | None,
    ) -> bool:
        if execution_class is None:
            return True
        return partition.execution_class == FteTaskExecutionClass.coerce(execution_class)

    @staticmethod
    def _partition_pending_attempt_publishable(partition: FteTaskPartition) -> bool:
        return (
            (partition.ready_for_scheduling or partition.execution_ready_deferred)
            and partition.running_attempt is None
            and not partition.finished
            and not partition.failed
        )

    def _mark_partition_ready_for_execution(self, partition: FteTaskPartition) -> bool:
        if self.execution_admission_callback is not None and not self.execution_admission_callback(partition):
            partition.defer_ready_for_execution()
            return False
        partition.mark_ready_for_execution()
        return True

    def has_open_task_running(self) -> bool:
        with self._state_lock:
            return any(
                partition.running_attempts and not partition.sealed
                for partition in self.partitions.values()
                if not partition.finished and not partition.failed
            )

    def has_retryable_running_attempt_on_worker(
        self,
        worker_id: str,
        retryable_by_partition_id: Mapping[int, bool] | None = None,
    ) -> bool:
        worker_id = str(worker_id)
        with self._state_lock:
            for partition in self.partitions.values():
                if retryable_by_partition_id is not None and not bool(
                    retryable_by_partition_id.get(int(partition.task_id.partition_id), False)
                ):
                    continue
                if any(running.worker_id == worker_id for running in partition.running_attempts.values()):
                    return True
            return False

    def _maybe_create_attempt(self, partition: FteTaskPartition) -> ScheduledAttempt | None:
        if self.attempt_admission_callback is not None and not self.attempt_admission_callback(partition):
            return None
        partition.mark_waiting_for_node()
        if self.worker_reservation_callback is not None:
            self.worker_reservation_callback(partition)
            return None
        try:
            return self.start_attempt_with_worker(partition)
        except FteWorkerReservationUnavailable:
            return None

    @staticmethod
    def _transition_from_partition(partition: FteTaskPartition) -> ExecutionClassTransition:
        return ExecutionClassTransition(
            task_id=partition.task_id,
            new_execution_class=partition.execution_class,
            running_attempts=tuple(partition.running_attempts.values()),
        )

    def _emit_execution_class_transitions(
        self,
        transitions: list[ExecutionClassTransition],
    ) -> None:
        if not transitions:
            return
        if self.execution_class_transition_callback is not None:
            self.execution_class_transition_callback(transitions)
            return
        for transition in transitions:
            for running in transition.running_attempts:
                worker = running.remote_handle
                if worker is None:
                    continue
                try:
                    worker.set_fte_task_execution_class(
                        running.attempt_id,
                        transition.new_execution_class,
                    )
                except Exception as exc:
                    raise self._worker_control_failure(
                        running,
                        "set_fte_task_execution_class",
                        exc,
                    ) from exc

    def _partition_requires_finish_output_stats(self, partition: FteTaskPartition) -> bool:
        if self.exchange is not None or partition.sink_handle is not None:
            return True
        if partition.descriptor.exchange_sink_instance is not None:
            return True
        running = partition.running_attempt
        return running is not None and running.sink_instance is not None

    def set_execution_class(
        self,
        execution_class: FteTaskExecutionClass | str | None,
    ) -> list[ExecutionClassTransition]:
        new_class = FteTaskExecutionClass.coerce(execution_class)
        with self._state_lock:
            if self.execution_class != new_class:
                if not self.execution_class.can_transition_to(new_class):
                    raise ValueError(
                        f"cannot change fragment execution {self.query_id}/{self.fragment_id} execution class "
                        f"from {self.execution_class.value} to {new_class.value}"
                    )
                self.execution_class = new_class
            transitions: list[ExecutionClassTransition] = []
            for partition in self.partitions.values():
                if partition.finished or partition.failed:
                    continue
                if partition.set_execution_class(new_class):
                    transitions.append(self._transition_from_partition(partition))
            return transitions

    def has_pending_partitions(
        self,
        execution_class: FteTaskExecutionClass | str | None = None,
    ) -> bool:
        with self._state_lock:
            return any(
                self._partition_pending_attempt_publishable(partition)
                and self._partition_matches_execution_class(partition, execution_class)
                for partition in self.partitions.values()
            )

    def pending_submission_count(self) -> int:
        """Return descriptor partitions eligible for a future task attempt."""
        with self._state_lock:
            return sum(
                1 for partition in self.partitions.values() if self._partition_pending_attempt_publishable(partition)
            )

    def waiting_for_execution_count(
        self,
        execution_class: FteTaskExecutionClass | str | None = None,
    ) -> int:
        with self._state_lock:
            return sum(
                1
                for partition in self.partitions.values()
                if partition.ready_for_scheduling
                and partition.node_wait_started_at is None
                and partition.running_attempt is None
                and not partition.finished
                and not partition.failed
                and self._partition_matches_execution_class(partition, execution_class)
            )

    def waiting_for_node_count(
        self,
        execution_class: FteTaskExecutionClass | str | None = None,
    ) -> int:
        with self._state_lock:
            return sum(
                1
                for partition in self.partitions.values()
                if partition.node_wait_started_at is not None
                and partition.running_attempt is None
                and not partition.finished
                and not partition.failed
                and self._partition_matches_execution_class(partition, execution_class)
            )

    def release_deferred_execution_partitions(
        self,
        execution_class: FteTaskExecutionClass | str | None = None,
    ) -> list[FteTaskAttemptId]:
        with self._state_lock:
            released: list[FteTaskAttemptId] = []
            for partition in sorted(self.partitions.values(), key=lambda value: value.task_id.partition_id):
                if not partition.execution_ready_deferred:
                    continue
                if partition.running_attempt is not None or partition.finished or partition.failed:
                    partition.execution_ready_deferred = False
                    continue
                if not self._partition_matches_execution_class(partition, execution_class):
                    continue
                if not self._mark_partition_ready_for_execution(partition):
                    continue
                released.append(FteTaskAttemptId(partition.task_id, partition.next_attempt_number()))
            return released

    def schedule_next_pending_partition(
        self,
        execution_class: FteTaskExecutionClass | str | None = None,
    ) -> ScheduledAttempt | None:
        with self._state_lock:
            for partition in sorted(self.partitions.values(), key=lambda value: value.task_id.partition_id):
                if not self._partition_ready_to_schedule(partition):
                    continue
                if not self._partition_matches_execution_class(partition, execution_class):
                    continue
                attempt = self._maybe_create_attempt(partition)
                if attempt is not None:
                    return attempt
            return None

    @staticmethod
    def _coalesce_partition_updates(updates: list[PartitionUpdate]) -> list[PartitionUpdate]:
        coalesced: dict[tuple[int, str], PartitionUpdate] = {}
        for update in updates:
            key = (int(update.partition_id), str(update.source_node_id))
            existing = coalesced.get(key)
            if existing is None:
                coalesced[key] = update
                continue
            if existing.no_more_splits and update.splits:
                raise RuntimeError(
                    f"source {update.source_node_id} for partition {update.partition_id} "
                    "received splits after no_more_splits"
                )
            coalesced[key] = PartitionUpdate(
                partition_id=existing.partition_id,
                source_node_id=existing.source_node_id,
                splits=[*existing.splits, *update.splits],
                no_more_splits=existing.no_more_splits or update.no_more_splits,
                ready_for_scheduling=existing.ready_for_scheduling or update.ready_for_scheduling,
            )
        return list(coalesced.values())

    @staticmethod
    def _split_update_batches(splits: list[FteSplit]) -> list[list[FteSplit]]:
        max_splits = fte_task_update_max_splits()
        max_payload_bytes = fte_task_update_max_payload_bytes()
        batches: list[list[FteSplit]] = []
        current: list[FteSplit] = []
        current_bytes = 0
        for split in splits:
            split_bytes = max(1, _estimated_payload_bytes(split.to_dict()))
            if current and (len(current) >= max_splits or current_bytes + split_bytes > max_payload_bytes):
                batches.append(current)
                current = []
                current_bytes = 0
            current.append(split)
            current_bytes += split_bytes
        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _worker_control_failure(
        running: RunningAttempt,
        method_name: str,
        exc: BaseException,
    ) -> FteWorkerControlFailure:
        return FteWorkerControlFailure(
            worker_id=running.worker_id,
            attempt_id=running.attempt_id,
            method_name=method_name,
            cause=exc,
        )

    def _record_worker_command(self, command: Any) -> None:
        if self._current_worker_commands is not None:
            self._current_worker_commands.append(command)
            return
        self._worker_command_outbox.append(command)

    def pop_worker_commands(self) -> list[Any]:
        commands = list(self._worker_command_outbox)
        self._worker_command_outbox.clear()
        return commands

    def update_partition(self, update: PartitionUpdate) -> ScheduledAttempt | None:
        partition = self.add_partition(update.partition_id)
        running_before_update = partition.running_attempt
        added = partition.descriptor.append_splits(update.source_node_id, update.splits)
        marked_no_more = False
        if update.no_more_splits:
            marked_no_more = partition.descriptor.mark_no_more_splits(update.source_node_id)

        should_start = (
            running_before_update is None
            and not partition.finished
            and not partition.failed
            and (update.ready_for_scheduling or update.no_more_splits or partition.sealed)
        )
        if should_start:
            if not self._mark_partition_ready_for_execution(partition):
                return None
            return self._maybe_create_attempt(partition)

        if running_before_update is not None:
            worker = running_before_update.remote_handle
            for batch in self._split_update_batches(added):
                split_payloads = [split.to_dict() for split in batch]
                command = FteAddSplitsCommand(
                    query_id=self.query_id,
                    fragment_id=self.fragment_id,
                    worker_id=running_before_update.worker_id,
                    worker=worker,
                    attempt_id=running_before_update.attempt_id,
                    source_node_id=update.source_node_id,
                    splits=tuple(split_payloads),
                )
                self._record_worker_command(command)
            if marked_no_more:
                command = FteNoMoreSplitsCommand(
                    query_id=self.query_id,
                    fragment_id=self.fragment_id,
                    worker_id=running_before_update.worker_id,
                    worker=worker,
                    attempt_id=running_before_update.attempt_id,
                    source_node_id=update.source_node_id,
                )
                self._record_worker_command(command)
        return None

    def seal_partition(self, partition_id: int) -> ScheduledAttempt | None:
        partition = self.add_partition(partition_id)
        old_class = partition.seal()
        if old_class is not None:
            self._emit_execution_class_transitions([self._transition_from_partition(partition)])
        if partition.running_attempt is None and not partition.finished and not partition.failed:
            if not self._mark_partition_ready_for_execution(partition):
                return None
            return self._maybe_create_attempt(partition)
        return None

    def task_failed(
        self,
        attempt_id: FteTaskAttemptId | str | Mapping[str, Any],
        error: Any = None,
        *,
        retryable: bool = True,
    ) -> ScheduledAttempt | None:
        with self._state_lock:
            attempt = FteTaskAttemptId.coerce(attempt_id)
            partition = self.partitions[attempt.partition_id]
            running = partition.running_attempts.get(attempt.attempt_id)
            failure_retryable = retryable and _failure_allows_retry(error)
            if self.exchange is not None and partition.sink_handle is not None:
                self.exchange.sink_aborted(partition.sink_handle, attempt.attempt_id)
            ready = partition.mark_attempt_failed(attempt, error, retryable=failure_retryable)
            if ready is not None and _is_memory_failure(error):
                partition.failed = True
                partition.ready_for_scheduling = False
                partition.execution_ready_deferred = False
                partition.running_attempts.clear()
                self.failed = True
                if running is not None:
                    running.remote_handle.record_fte_task_terminal(attempt)
                return None
            if partition.failed:
                self.failed = True
            if ready is None:
                if running is not None:
                    running.remote_handle.record_fte_task_terminal(attempt)
                return None
            if not partition.sealed and partition.execution_class.is_speculative:
                partition.ready_for_scheduling = False
                partition.execution_ready_deferred = False
                if running is not None:
                    running.remote_handle.record_fte_task_terminal(attempt)
                return None
            partition.mark_ready_for_execution()
            if running is not None:
                record_without_drain = getattr(
                    running.remote_handle,
                    "record_fte_task_terminal_without_drain",
                    None,
                )
                if callable(record_without_drain):
                    record_without_drain(attempt)
                else:
                    running.remote_handle.record_fte_task_terminal(attempt)
            return self._maybe_create_attempt(partition)

    def revoke_speculative_attempts(
        self,
        *,
        worker_id: str | None = None,
        max_count: int | None = None,
        reason: Any = None,
    ) -> list[RevokedAttempt]:
        revoked: list[RevokedAttempt] = []
        limit = None if max_count is None else max(0, int(max_count))
        if limit == 0:
            return revoked
        with self._state_lock:
            for partition in self.partitions.values():
                if partition.finished or partition.failed or not partition.execution_class.is_speculative:
                    continue
                running_attempts = sorted(
                    partition.running_attempts.values(), key=lambda running: running.attempt_id.attempt_id
                )
                for running in running_attempts:
                    if worker_id is not None and str(running.worker_id or "") != str(worker_id):
                        continue
                    try:
                        if running.remote_handle is not None:
                            running.remote_handle.fte_cancel_task(running.attempt_id.to_dict())
                    except Exception as exc:
                        raise self._worker_control_failure(
                            running,
                            "fte_cancel_task",
                            exc,
                        ) from exc
                    if self.exchange is not None and partition.sink_handle is not None:
                        self.exchange.sink_aborted(partition.sink_handle, running.attempt_id.attempt_id)
                    revoked_attempt = partition.revoke_attempt(
                        running.attempt_id,
                        reason or "speculative task revoked",
                    )
                    if revoked_attempt is None:
                        continue
                    running.remote_handle.record_fte_task_terminal(running.attempt_id)
                    revoked.append(revoked_attempt)
                    if partition.failed:
                        self.failed = True
                    if limit is not None and len(revoked) >= limit:
                        return revoked
        return revoked

    def mark_worker_failed(
        self,
        worker_id: str,
        error: Any = None,
        *,
        retryable: bool = True,
        retryable_by_partition_id: Mapping[int, bool] | None = None,
    ) -> list[ScheduledAttempt]:
        with self._state_lock:
            worker_id = str(worker_id)
            scheduled: list[ScheduledAttempt] = []
            for partition in self.partitions.values():
                failed_attempts = [
                    running.attempt_id
                    for running in partition.running_attempts.values()
                    if running.worker_id == worker_id
                ]
                for attempt in failed_attempts:
                    partition_retryable = retryable
                    if retryable_by_partition_id is not None:
                        partition_retryable = retryable and bool(
                            retryable_by_partition_id.get(int(partition.task_id.partition_id), False)
                        )
                    retry = self.task_failed(attempt, error, retryable=partition_retryable)
                    if retry is not None:
                        scheduled.append(retry)
            return scheduled

    def task_finished(
        self,
        attempt_id: FteTaskAttemptId | str | Mapping[str, Any],
        output_stats: Any = None,
    ) -> bool:
        with self._state_lock:
            attempt = FteTaskAttemptId.coerce(attempt_id)
            partition = self.partitions[attempt.partition_id]
            running_attempts = list(partition.running_attempts.values())
            running = partition.running_attempts.get(attempt.attempt_id)
            unselected_running = [item for item in running_attempts if item.attempt_id.attempt_id != attempt.attempt_id]
            changed = partition.mark_attempt_finished(attempt, output_stats)
            if changed:
                if running is not None:
                    running.remote_handle.record_fte_task_result_ready(attempt)
                cancel_failure: FteWorkerControlFailure | None = None
                for loser in unselected_running:
                    try:
                        if loser.remote_handle is not None:
                            loser.remote_handle.fte_cancel_task(loser.attempt_id.to_dict())
                    except Exception as exc:
                        if cancel_failure is None:
                            cancel_failure = self._worker_control_failure(
                                loser,
                                "fte_cancel_task",
                                exc,
                            )
                    if self.exchange is not None and partition.sink_handle is not None:
                        self.exchange.sink_aborted(partition.sink_handle, loser.attempt_id.attempt_id)
                    loser.remote_handle.record_fte_task_terminal(loser.attempt_id)
                if self.exchange is not None and partition.sink_handle is not None:
                    self.exchange.sink_finished(partition.sink_handle, attempt.attempt_id)
                self.descriptor_storage.remove(partition.task_id)
                if cancel_failure is not None:
                    raise cancel_failure
            return changed

    def handle_task_status(
        self,
        status: Mapping[str, Any],
        *,
        retryable: bool = True,
    ) -> ScheduledAttempt | None:
        if str(status.get("state", "")).upper() == "UNKNOWN":
            return None
        attempt_id = self._attempt_id_from_status(status)
        state = self._task_state_from_status(status)
        task_stats = _task_status_progress_stats(status)
        raw_task_stats = status.get("task_stats")
        if isinstance(raw_task_stats, Mapping):
            task_stats = {**dict(raw_task_stats), **task_stats}
        if task_stats:
            with self._state_lock:
                partition = self.partitions.get(attempt_id.partition_id)
                if partition is not None:
                    partition.update_running_task_stats(attempt_id, task_stats)
        if state == FteTaskState.FINISHED:
            output_stats = self._output_stats_from_status(status)
            partition = self.partitions[attempt_id.partition_id]
            if output_stats is None and self._partition_requires_finish_output_stats(partition):
                return self.task_failed(
                    attempt_id,
                    _missing_output_stats_failure(attempt_id),
                    retryable=retryable,
                )
            self.task_finished(attempt_id, output_stats)
            return None
        if state in (FteTaskState.FAILED, FteTaskState.CANCELED, FteTaskState.ABORTED):
            status_retryable = retryable and state == FteTaskState.FAILED
            return self.task_failed(
                attempt_id,
                status.get("failure") or dict(status),
                retryable=status_retryable,
            )
        return None

    def query_status_snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            partitions = []
            for partition_id, partition in sorted(self.partitions.items()):
                partitions.append(
                    {
                        "partition_id": int(partition_id),
                        "running": bool(partition.running_attempts),
                        "failed": bool(partition.failed),
                        "finished": bool(partition.finished),
                        "selected_attempt": partition.selected_attempt,
                        "task_id": str(partition.task_id),
                        "failures": list(partition.failures),
                    }
                )
            return {
                "failed": bool(self.failed),
                "partitions": partitions,
            }

    def start_attempt_with_worker(self, partition: FteTaskPartition) -> ScheduledAttempt:
        worker_id, worker = self._select_worker(partition)
        sink_instance = None
        if self.exchange is not None:
            if partition.sink_handle is None:
                partition.sink_handle = self.exchange.add_sink(partition.task_id.partition_id)
            sink_instance = self.exchange.instantiate_sink(
                partition.sink_handle,
                partition.next_attempt_number(),
            )
        elif partition.descriptor.exchange_sink_instance is not None:
            sink_instance = derive_exchange_sink_instance_for_attempt(
                partition.descriptor.exchange_sink_instance,
                partition.next_attempt_number(),
                partition.task_id.partition_id,
            )
        scheduled = partition.start_attempt(
            sink_instance=sink_instance,
            worker_id=worker_id,
            remote_handle=worker,
        )
        registration_result = worker.ensure_fragment_registered(
            self.query_id,
            self.fragment_id,
            scheduled.descriptor.fragment_plan,
        )
        if registration_result is not None:
            scheduled.request["fragment_registration_result"] = registration_result
        if scheduled.request.get("fragment_registration_result") is not None:
            scheduled.request["fragment_plan"] = None
        command = FteCreateTaskCommand(
            query_id=self.query_id,
            fragment_id=self.fragment_id,
            worker_id=worker_id,
            worker=worker,
            attempt_id=scheduled.attempt_id,
            partition_id=partition.task_id.partition_id,
            request=scheduled.request,
        )
        self._record_worker_command(command)
        return scheduled

    def handle_worker_command_success(self, command: Any) -> None:
        if isinstance(command, FteCreateTaskCommand):
            command.worker.record_fte_task_started(command.attempt_id, command.request)
            return
        if isinstance(command, FteAddSplitsCommand):
            command.worker.record_fte_splits_added(command.attempt_id, len(command.splits))
            command.worker.record_fte_split_bytes_added(
                command.attempt_id,
                sum(_estimated_payload_bytes(split) for split in command.splits),
            )

    def worker_control_failure_for_command(
        self,
        command: FteWorkerCommand,
        exc: BaseException,
    ) -> FteWorkerControlFailure:
        if isinstance(command, FteCreateTaskCommand):
            method_name = "fte_create_task"
        elif isinstance(command, FteAddSplitsCommand):
            method_name = "fte_add_splits"
        elif isinstance(command, FteNoMoreSplitsCommand):
            method_name = "fte_no_more_splits"
        elif isinstance(command, FteTaskUpdateCommand):
            method_name = "fte_update_task"
        else:
            raise TypeError(f"unsupported FTE worker command: {type(command).__name__}")
        return FteWorkerControlFailure(
            worker_id=command.worker_id,
            attempt_id=command.attempt_id,
            method_name=method_name,
            cause=exc,
        )

    def _select_worker(self, partition: FteTaskPartition) -> tuple[str | None, Any]:
        if self.worker_selector is not None:
            selected = self.worker_selector(partition)
            if isinstance(selected, tuple) and len(selected) == 2:
                return selected[0], selected[1]
            worker_id = getattr(selected, "worker_id", None)
            return worker_id, selected
        if self.worker is None:
            raise RuntimeError("FteFragmentExecution requires a worker or worker_selector")
        return self.worker_id, self.worker

    @staticmethod
    def _attempt_id_from_status(status: Mapping[str, Any]) -> FteTaskAttemptId:
        task_id = status.get("task_id")
        if task_id is not None:
            return FteTaskAttemptId.coerce(task_id)
        task_id_string = status.get("task_id_string")
        if task_id_string is not None:
            return FteTaskAttemptId.parse(str(task_id_string))
        return FteTaskAttemptId.coerce(status)

    @staticmethod
    def _task_state_from_status(status: Mapping[str, Any]) -> FteTaskState:
        state = status.get("state")
        if isinstance(state, FteTaskState):
            return state
        return FteTaskState(str(state))

    @staticmethod
    def _output_stats_from_status(status: Mapping[str, Any]) -> Any:
        output_stats = None
        if status.get("spooling_output_stats") is not None:
            output_stats = status.get("spooling_output_stats")
        elif status.get("output_stats") is not None:
            output_stats = status.get("output_stats")
        task_stats = status.get("task_stats")
        if isinstance(task_stats, Mapping):
            if isinstance(output_stats, Mapping):
                merged = dict(task_stats)
                merged.update(output_stats)
                return merged
            if output_stats is None:
                return dict(task_stats)
        return output_stats
