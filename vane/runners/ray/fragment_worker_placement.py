# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vane.runners.fte import (
    FteWorkerReservationUnavailable,
)
from vane.runners.fte.fte_events import WorkerReservationCompleted
from vane.runners.ray.fragment_registry import (
    _FTE_CLOSING_QUERIES,
    _FTE_REGISTRY_LOCK,
    _FTE_SCHEDULERS,
)
from vane.runners.ray.fragment_worker_reservations import (
    cancel_fte_worker_reservation_future,
    fte_partition_owner,
    fte_worker_reservation_future_is_current,
    pending_fte_worker_reservation_partition,
)
from vane.runners.ray.fragment_worker_selection import (
    available_fte_workers,
    select_fte_worker,
)
from vane.runners.ray.fte_fragment_scheduler import (
    FteWorkerPlacementManager,
    _admit_fte_partition_node_wait,
    _node_requirements_have_candidates,
)
from vane.runners.ray.fte_scheduler_config import _fte_allowed_no_matching_node_period_s

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vane.runners.fte import FteFragmentExecution, FteTaskExecutionClass, NodeRequirements
    from vane.runners.ray.fte_fragment_scheduler import FteWorkerReservationFuture


class FteWorkerPlacementMixin:
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

    def _record_fte_worker_reservation_unavailable(
        self,
        future: FteWorkerReservationFuture,
        partition: Any | None,
        *,
        node_requirements: NodeRequirements | Mapping[str, Any] | None,
        node_requirements_wait_started_at: float | None,
    ) -> None:
        if partition is None:
            cancel_fte_worker_reservation_future(future)
            return
        has_matching_node = _node_requirements_have_candidates(
            available_fte_workers(self, self.worker_id),
            node_requirements,
            node_requirements_wait_started_at=node_requirements_wait_started_at,
        )
        if not has_matching_node:
            no_matching_period = partition.mark_no_matching_node()
            if no_matching_period > _fte_allowed_no_matching_node_period_s():
                raise RuntimeError(
                    f"No nodes available to run query {future.query_id}/{future.fragment_id}/{future.partition_id}"
                )
        else:
            partition.reset_no_matching_node()

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
        if partition.running_attempt is not None or partition.finished or partition.failed:
            cancel_fte_worker_reservation_future(future)
            return False
        memory_requirement_bytes = partition.memory_requirement_bytes
        execution_class = partition.execution_class
        node_requirements = partition.node_requirements
        node_requirements_wait_started_at = partition.node_wait_started_at or future.node_requirements_wait_started_at
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
                cancel_fte_worker_reservation_future(
                    future,
                    allow_next_submission=False,
                )
                partition.node_wait_started_at = None
                partition.defer_ready_for_execution()
                return False
            try:
                self._record_fte_worker_reservation_unavailable(
                    future,
                    partition,
                    node_requirements=node_requirements,
                    node_requirements_wait_started_at=node_requirements_wait_started_at,
                )
            except RuntimeError as exc:
                if raise_on_no_matching_timeout:
                    cancel_fte_worker_reservation_future(future)
                    raise
                future.set_exception(exc)
                return True
            return False
        except Exception as exc:
            future.set_exception(exc)
            return True
        if not fte_worker_reservation_future_is_current(future):
            FteWorkerPlacementManager.release_owner(
                query_id=future.query_id,
                fragment_id=future.fragment_id,
                partition_id=future.partition_id,
            )
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
        future, created = self._fte_worker_placement_manager.request_async(
            query_id=key[0],
            fragment_execution_id=fragment_execution.fragment_execution_id,
            fragment_id=key[1],
            partition_id=key[2],
            memory_requirement_bytes=partition.memory_requirement_bytes,
            execution_class=partition.execution_class,
            node_requirements=partition.node_requirements,
            node_requirements_wait_started_at=partition.node_wait_started_at,
            on_done=self._enqueue_fte_worker_reservation_completion,
        )
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
        partition,
        *,
        fragment_execution: FteFragmentExecution | None = None,
    ) -> None:
        if not _admit_fte_partition_node_wait(query_id, partition, fragment_execution):
            return
        partition.mark_waiting_for_node()
        try:
            self._fte_worker_placement_manager.acquire(
                query_id=str(query_id),
                fragment_id=str(fragment_id),
                partition_id=int(partition.task_id.partition_id),
                memory_requirement_bytes=partition.memory_requirement_bytes,
                execution_class=partition.execution_class,
                node_requirements=partition.node_requirements,
                node_requirements_wait_started_at=partition.node_wait_started_at,
            )
        except FteWorkerReservationUnavailable as exc:
            if exc.blocked_reason not in {"", "node_capacity"}:
                partition.node_wait_started_at = None
