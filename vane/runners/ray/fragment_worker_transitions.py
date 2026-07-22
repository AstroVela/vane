# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vane.runners.fte import (
    FteWorkerControlFailure,
)
from vane.runners.fte.fte_events import MemoryPressureDetected
from vane.runners.ray.fragment_registry import (
    _FTE_FRAGMENT_EXECUTIONS,
    _FTE_REGISTRY_LOCK,
    _FTE_SCHEDULERS,
    _FTE_WORKER_HANDLES,
)
from vane.runners.ray.fragment_worker_state import (
    fte_fragment_execution_items,
    fte_fragment_execution_query_ids,
)
from vane.runners.ray.fte_fragment_scheduler import (
    _fte_effective_worker_memory_budget_bytes,
    _fte_pressure_total_memory_bytes,
    _ordered_fte_fragment_execution_items_for_pending_drain,
    _required_fte_pressure_stats,
    fte_registry_query_is_closing,
)

if TYPE_CHECKING:
    from vane.runners.fte import (
        ExecutionClassTransition,
        FteFragmentExecution,
        FteTaskExecutionClass,
        RevokedAttempt,
    )


class FteWorkerTransitionMixin:
    def _release_deferred_fte_execution_partitions(
        self,
        fragment_execution_items: list[tuple[tuple[str, str], FteFragmentExecution]],
        execution_class: FteTaskExecutionClass,
    ) -> bool:
        released_any = False
        for _, fragment_execution in _ordered_fte_fragment_execution_items_for_pending_drain(
            fragment_execution_items,
            execution_class=execution_class,
        ):
            released_any = (
                bool(fragment_execution.release_deferred_execution_partitions(execution_class)) or released_any
            )
        return released_any

    def _apply_fte_execution_class_transitions(
        self,
        query_id: str,
        fragment_id: str,
        transitions: list[ExecutionClassTransition],
    ) -> None:
        if not transitions:
            return
        for transition in transitions:
            owner = self._fte_partition_owner(
                query_id,
                fragment_id,
                transition.task_id.partition_id,
            )
            self._fte_worker_placement_manager.update_execution_class(
                query_id=query_id,
                fragment_id=fragment_id,
                partition_id=transition.task_id.partition_id,
                execution_class=transition.new_execution_class,
            )
            for running in transition.running_attempts:
                worker = running.remote_handle or owner
                if worker is not None:
                    worker.set_fte_task_execution_class(
                        running.attempt_id,
                        transition.new_execution_class,
                    )

    def set_fte_fragment_execution_execution_class(
        self,
        query_id: str,
        fragment_id: str,
        execution_class: FteTaskExecutionClass | str | None,
    ) -> list[Any]:
        query_id = str(query_id)
        fragment_id = str(fragment_id)
        with _FTE_REGISTRY_LOCK:
            fragment_execution = _FTE_FRAGMENT_EXECUTIONS.get((query_id, fragment_id))
        if fragment_execution is None:
            raise KeyError(f"FTE fragment execution {query_id}/{fragment_id} does not exist")
        transitions = fragment_execution.set_execution_class(execution_class)
        self._apply_fte_execution_class_transitions(query_id, fragment_id, transitions)
        return self._drain_fte_pending_tasks()

    def _revoke_fte_speculative_tasks_direct(
        self,
        *,
        worker_id: str | None = None,
        max_count: int | None = None,
        reason: Any = None,
        query_id_filter: str | None = None,
    ) -> list[RevokedAttempt]:
        if query_id_filter is not None:
            query_id_filter = str(query_id_filter)
        fragment_execution_items = fte_fragment_execution_items(query_id_filter)
        revoked: list[RevokedAttempt] = []
        remaining = None if max_count is None else max(0, int(max_count))
        if remaining == 0:
            return revoked
        for _, fragment_execution in fragment_execution_items:
            count = remaining
            try:
                fragment_execution_revoked = fragment_execution.revoke_speculative_attempts(
                    worker_id=worker_id,
                    max_count=count,
                    reason=reason,
                )
            except FteWorkerControlFailure as exc:
                self._handles_for_fte_worker_control_failure(exc)
                fragment_execution_revoked = []
            revoked.extend(fragment_execution_revoked)
            if remaining is not None:
                remaining -= len(fragment_execution_revoked)
                if remaining <= 0:
                    break
        return revoked

    def revoke_fte_speculative_tasks_for_memory_pressure(
        self,
        *,
        max_count_per_worker: int | None = None,
    ) -> list[Any]:
        query_ids = sorted(fte_fragment_execution_query_ids())
        handles: list[Any] = []
        for query_id in query_ids:
            if fte_registry_query_is_closing(query_id):
                continue
            scheduler = _FTE_SCHEDULERS.get(query_id)
            if scheduler is None:
                continue
            self._bind_fte_scheduler_handlers(scheduler)
            scheduler.enqueue(MemoryPressureDetected(query_id, max_count_per_worker))
            handles.extend(scheduler.drain())
        return handles

    def _revoke_fte_speculative_tasks_for_memory_pressure_direct(
        self,
        *,
        max_count_per_worker: int | None = None,
        query_id_filter: str | None = None,
    ) -> list[Any]:
        if query_id_filter is not None:
            query_id_filter = str(query_id_filter)
        with _FTE_REGISTRY_LOCK:
            workers = [
                handle
                for _, handle in sorted(_FTE_WORKER_HANDLES.items())
                if handle is not None and handle._fte_healthy
            ]
        revoked_any = False
        for worker in workers:
            budget_bytes = _fte_effective_worker_memory_budget_bytes(worker, None)
            if budget_bytes is None:
                continue
            worker_id = str(worker.worker_id)
            if not worker_id:
                continue
            revoked_for_worker = 0
            while True:
                stats = _required_fte_pressure_stats(worker)
                if _fte_pressure_total_memory_bytes(stats) <= budget_bytes:
                    break
                if (
                    int(stats.get("speculative_memory_bytes", 0)) + int(stats.get("eager_speculative_memory_bytes", 0))
                    <= 0
                ):
                    break
                if max_count_per_worker is not None and revoked_for_worker >= max(0, int(max_count_per_worker)):
                    break
                revoked = self._revoke_fte_speculative_tasks_direct(
                    worker_id=worker_id,
                    max_count=1,
                    reason="speculative task revoked due to worker memory pressure",
                    query_id_filter=query_id_filter,
                )
                if not revoked:
                    break
                revoked_for_worker += len(revoked)
                revoked_any = True
        if revoked_any:
            return self._drain_fte_pending_tasks()
        return []
