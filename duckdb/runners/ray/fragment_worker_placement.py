# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from duckdb.runners.fte import (
    FteWorkerReservationUnavailable,
)
from duckdb.runners.fte.fte_events import WorkerReservationCompleted
from duckdb.runners.ray.fragment_registry import (
    _FTE_CLOSING_QUERIES,
    _FTE_REGISTRY_LOCK,
    _FTE_SCHEDULERS,
)
from duckdb.runners.ray.fragment_submission_window import release_fte_partition_submission
from duckdb.runners.ray.fragment_worker_reservations import (
    cancel_fte_worker_reservation_future,
    fte_partition_owner,
    fte_worker_reservation_future_is_current,
    pending_fte_worker_reservation_partition,
)
from duckdb.runners.ray.fragment_worker_selection import (
    available_fte_workers,
    select_fte_worker,
)
from duckdb.runners.ray.fte_fragment_scheduler import (
    FteWorkerPlacementManager,
    _admit_fte_partition_node_wait,
    _node_requirements_have_candidates,
)
from duckdb.runners.ray.fte_scheduler_config import _fte_allowed_no_matching_node_period_s

if TYPE_CHECKING:
    from collections.abc import Mapping

    from duckdb.runners.fte import FteFragmentExecution, FteTaskExecutionClass, NodeRequirements
    from duckdb.runners.ray.fte_fragment_scheduler import FteWorkerReservationFuture


class FteWorkerPlacementMixin:
    if TYPE_CHECKING:
        # Supplied by the other mixins on the composed Ray worker handle.
        _bind_fte_scheduler_handlers: Any
        _fte_worker_placement_manager: Any
        _handles_for_worker_reservation_completed_event: Any
        worker_id: Any

    def _select_fte_worker(
        self,
        *,
        exclude: set[str] | None = None,
        allowed_node_ids: set[str] | None = None,
        memory_requirement_bytes: Any = None,
        execution_class: FteTaskExecutionClass | str | None = None,
        node_requirements: NodeRequirements | Mapping[str, Any] | None = None,
        node_requirements_wait_started_at: float | None = None,
    ) -> Any | None:
        return select_fte_worker(
            self,
            self.worker_id,
            exclude=exclude,
            allowed_node_ids=allowed_node_ids,
            memory_requirement_bytes=memory_requirement_bytes,
            execution_class=execution_class,
            node_requirements=node_requirements,
            node_requirements_wait_started_at=node_requirements_wait_started_at,
        )

    def _enqueue_fte_worker_reservation_completion(self, future: FteWorkerReservationFuture) -> None:
        if future.cancelled():
            return
        worker_id = None
        error = None
        try:
            reservation = future.result()
            worker_id = reservation.worker_id
        except Exception as exc:
            error = exc
        with _FTE_REGISTRY_LOCK:
            if future.query_id in _FTE_CLOSING_QUERIES:
                cancel_fte_worker_reservation_future(future)
                return
            scheduler = _FTE_SCHEDULERS.get(future.query_id)
        if scheduler is None:
            cancel_fte_worker_reservation_future(future)
            return
        self._bind_fte_scheduler_handlers(scheduler)
        scheduler.enqueue(
            WorkerReservationCompleted(
                future.query_id,
                future.fragment_execution_id,
                future.fragment_id,
                future.partition_id,
                future.reservation_generation,
                worker_id,
                error=error,
            )
        )

    def _handle_fte_worker_reservation_callback_error(
        self,
        future: FteWorkerReservationFuture,
        error: BaseException,
    ) -> None:
        callback_failure = RuntimeError(f"FTE worker reservation completion callback failed: {error}")
        callback_failure.__cause__ = error
        # Dispatch itself may be what failed, so enter the generation-fenced
        # completion handler directly instead of trying to enqueue another event.
        self._handles_for_worker_reservation_completed_event(
            WorkerReservationCompleted(
                future.query_id,
                future.fragment_execution_id,
                future.fragment_id,
                future.partition_id,
                future.reservation_generation,
                error=callback_failure,
            )
        )

    def _record_fte_worker_reservation_unavailable(
        self,
        future: FteWorkerReservationFuture,
        fragment_execution: FteFragmentExecution,
        placement: Any,
        *,
        node_requirements: NodeRequirements | Mapping[str, Any] | None,
        node_requirements_wait_started_at: float | None,
    ) -> bool:
        has_matching_node = _node_requirements_have_candidates(
            available_fte_workers(self, self.worker_id),
            node_requirements,
            node_requirements_wait_started_at=node_requirements_wait_started_at,
        )
        current, no_matching_period = fragment_execution.record_partition_placement_unavailable(
            placement,
            has_matching_node=has_matching_node,
        )
        if not current:
            cancel_fte_worker_reservation_future(future)
            return False
        if not has_matching_node and no_matching_period >= _fte_allowed_no_matching_node_period_s():
            raise RuntimeError(
                f"No nodes available to run query {future.query_id}/{future.fragment_id}/{future.partition_id}"
            )
        return True

    def _try_complete_fte_worker_reservation_future(
        self,
        future: FteWorkerReservationFuture,
        *,
        partition: Any | None = None,
        raise_on_no_matching_timeout: bool = False,
    ) -> bool:
        if future.done():
            return False
        fragment_execution, current_partition = pending_fte_worker_reservation_partition(future)
        partition = partition or current_partition
        if fragment_execution is None or partition is None:
            cancel_fte_worker_reservation_future(future)
            return False
        with fragment_execution.partition_placement(
            future.partition_id,
            expected_partition=partition,
        ) as placement:
            if placement is None:
                cancel_fte_worker_reservation_future(future)
                return False
            memory_requirement_bytes = placement.memory_requirement_bytes
            execution_class = placement.execution_class
            node_requirements = placement.node_requirements
            node_requirements_wait_started_at = (
                placement.node_requirements_wait_started_at or future.node_requirements_wait_started_at
            )
            try:
                if not fte_worker_reservation_future_is_current(future):
                    return False
                reservation = self._fte_worker_placement_manager.acquire(
                    query_id=future.query_id,
                    fragment_id=future.fragment_id,
                    partition_id=future.partition_id,
                    memory_requirement_bytes=memory_requirement_bytes,
                    execution_class=execution_class,
                    node_requirements=node_requirements,
                    node_requirements_wait_started_at=node_requirements_wait_started_at,
                )
            except FteWorkerReservationUnavailable as exc:
                if exc.blocked_reason not in {"", "node_capacity"}:
                    # QRM did not grant this descriptor.  Return it to the passive
                    # execution queue; keeping a reservation future here would
                    # recreate the one-waiter-per-logical-partition failure mode.
                    fragment_execution.defer_partition_after_placement_rejection(
                        placement,
                        defer_execution=True,
                    )
                    cancel_fte_worker_reservation_future(
                        future,
                        allow_next_submission=False,
                    )
                    return False
                try:
                    current = self._record_fte_worker_reservation_unavailable(
                        future,
                        fragment_execution,
                        placement,
                        node_requirements=node_requirements,
                        node_requirements_wait_started_at=node_requirements_wait_started_at,
                    )
                    if not current:
                        return False
                except RuntimeError as exc:
                    if raise_on_no_matching_timeout:
                        cancel_fte_worker_reservation_future(future)
                        raise
                    future.set_exception(exc)
                    return True
                return False
            except Exception as exc:
                if fragment_execution.commit_partition_placement(placement):
                    future.set_exception(exc)
                    return True
                cancel_fte_worker_reservation_future(future)
                return False
            future_is_current = fte_worker_reservation_future_is_current(future)
            placement_is_current = fragment_execution.commit_partition_placement(placement)
            if not future_is_current or not placement_is_current:
                FteWorkerPlacementManager.release_owner(
                    query_id=future.query_id,
                    fragment_id=future.fragment_id,
                    partition_id=future.partition_id,
                    terminal=False,
                )
                if future_is_current:
                    cancel_fte_worker_reservation_future(future)
                return False
            future.set_result(reservation)
            return True

    def _request_fte_worker_reservation_for_partition(
        self,
        query_id: str,
        fragment_id: str,
        fragment_execution: FteFragmentExecution,
        partition: Any,
    ) -> bool:
        key = (
            str(query_id),
            str(fragment_id),
            int(partition.task_id.partition_id),
        )
        with fragment_execution.partition_placement(
            key[2],
            expected_partition=partition,
        ) as placement:
            if placement is None:
                release_fte_partition_submission(*key)
                return False
            placement = fragment_execution.mark_partition_waiting_for_node(placement)
            if placement is None:
                release_fte_partition_submission(*key)
                return False
            try:
                future, created = self._fte_worker_placement_manager.request_async(
                    query_id=key[0],
                    fragment_execution_id=fragment_execution.fragment_execution_id,
                    fragment_id=key[1],
                    partition_id=key[2],
                    memory_requirement_bytes=placement.memory_requirement_bytes,
                    execution_class=placement.execution_class,
                    node_requirements=placement.node_requirements,
                    node_requirements_wait_started_at=placement.node_requirements_wait_started_at,
                    on_done=self._enqueue_fte_worker_reservation_completion,
                    on_done_error=self._handle_fte_worker_reservation_callback_error,
                )
            except BaseException:
                release_fte_partition_submission(*key)
                raise
            if not fragment_execution.commit_partition_placement(placement):
                cancel_fte_worker_reservation_future(future)
                return False
        if not created:
            return True
        return self._try_complete_fte_worker_reservation_future(
            future,
            partition=partition,
            raise_on_no_matching_timeout=True,
        )

    @staticmethod
    def _fte_partition_owner(
        query_id: str,
        fragment_id: str,
        partition_id: int,
    ) -> Any | None:
        return fte_partition_owner(query_id, fragment_id, partition_id)

    def _try_reserve_fte_partition_for_node_wait(
        self,
        query_id: str,
        fragment_id: str,
        partition: Any,
        *,
        fragment_execution: FteFragmentExecution | None = None,
    ) -> None:
        if fragment_execution is None:
            return
        partition_id = int(partition.task_id.partition_id)
        with fragment_execution.partition_placement(
            partition_id,
            expected_partition=partition,
        ) as placement:
            if placement is None:
                return
            if not _admit_fte_partition_node_wait(query_id, placement.partition, fragment_execution):
                return
            placement = fragment_execution.mark_partition_waiting_for_node(placement)
            if placement is None:
                release_fte_partition_submission(
                    query_id,
                    fragment_id,
                    partition_id,
                )
                return
            try:
                self._fte_worker_placement_manager.acquire(
                    query_id=str(query_id),
                    fragment_id=str(fragment_id),
                    partition_id=partition_id,
                    memory_requirement_bytes=placement.memory_requirement_bytes,
                    execution_class=placement.execution_class,
                    node_requirements=placement.node_requirements,
                    node_requirements_wait_started_at=placement.node_requirements_wait_started_at,
                )
            except FteWorkerReservationUnavailable as exc:
                if exc.blocked_reason not in {"", "node_capacity"}:
                    fragment_execution.defer_partition_after_placement_rejection(
                        placement,
                        defer_execution=False,
                    )
                return
            if not fragment_execution.commit_partition_placement(placement):
                FteWorkerPlacementManager.release_owner(
                    query_id=str(query_id),
                    fragment_id=str(fragment_id),
                    partition_id=partition_id,
                    terminal=False,
                )
