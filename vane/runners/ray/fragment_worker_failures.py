# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vane.runners.ray.fragment_registry import (
    _FTE_CLOSING_QUERIES,
    _FTE_REGISTRY_LOCK,
    _FTE_SCHEDULERS,
    _FTE_WORKER_HANDLES,
)
from vane.runners.ray.fte_fragment_scheduler import (
    _fte_retry_remaining_delay_s,
    _mark_fte_worker_failed,
    _query_ids_owned_by_fte_workers,
    _worker_actor_death_confirms_quiescence,
    _worker_failure_payload,
)

if TYPE_CHECKING:
    from vane.runners.fte.fte_scheduler import FteQueryScheduler

_WORKER_FAILURE_RECONCILIATION_LOCK = threading.Lock()


def _worker_failure_reconciliation_target(
    worker_id: str,
    *,
    manager_instance_id: str,
    worker_incarnation_id: str,
) -> Any | None:
    with _FTE_REGISTRY_LOCK:
        handle = _FTE_WORKER_HANDLES.get(str(worker_id))
        if handle is None:
            return None
        if str(handle.manager_instance_id) != str(manager_instance_id):
            return None
        if str(handle.worker_incarnation_id) != str(worker_incarnation_id):
            return None
        return handle


@dataclass(frozen=True)
class _FteWorkerFailureReconciliationResult:
    scheduled_by_fragment: tuple[tuple[str, str, list[Any], list[Any]], ...]
    schedulers: tuple[FteQueryScheduler, ...]


class _FteWorkerFailureReconciliationState:
    """Share local failure fences across one global reconciliation."""

    def __init__(self, failure_key: tuple[str, str, str]) -> None:
        self.failure_key = failure_key
        self.completion: Future[None] = Future()
        self._lock = threading.Lock()
        self._schedulers: dict[int, FteQueryScheduler] = {}

    def fence_scheduler(self, scheduler: FteQueryScheduler) -> bool:
        """Record and claim a scheduler fence atomically for this reconciliation."""

        scheduler_identity = id(scheduler)
        _manager_instance_id, worker_id, worker_incarnation_id = self.failure_key
        with self._lock:
            if scheduler_identity in self._schedulers:
                return True
            if not scheduler.record_worker_failure(
                worker_id,
                worker_incarnation_id=worker_incarnation_id,
            ):
                return False
            self._schedulers[scheduler_identity] = scheduler
            return True

    def schedulers(self) -> tuple[FteQueryScheduler, ...]:
        with self._lock:
            return tuple(self._schedulers.values())


_WORKER_FAILURE_RECONCILIATIONS: dict[
    tuple[str, str, str],
    _FteWorkerFailureReconciliationState,
] = {}


class _FteWorkerFailureReconciliation:
    """Own one worker-incarnation failure through final side-effect publication."""

    def __init__(
        self,
        event: Any,
        *,
        state: _FteWorkerFailureReconciliationState,
        target: Any | None,
        owns_reconciliation: bool,
    ) -> None:
        self.event = event
        self.state = state
        self.failure_key = state.failure_key
        self.completion = state.completion
        self.target = target
        self.owns_reconciliation = owns_reconciliation
        self._completion_lock = threading.Lock()
        self._completed = False

    def _registered_schedulers(
        self,
        schedulers: tuple[FteQueryScheduler, ...] = (),
    ) -> tuple[FteQueryScheduler, ...]:
        registered = (*schedulers, *self.state.schedulers())
        return tuple({id(scheduler): scheduler for scheduler in registered}.values())

    def complete(
        self,
        error: BaseException | None,
        *,
        schedulers: tuple[FteQueryScheduler, ...] = (),
    ) -> None:
        if not self.owns_reconciliation:
            raise RuntimeError("only the reconciliation owner can publish completion")
        with self._completion_lock:
            if self._completed:
                return
            self._completed = True
        schedulers = self._registered_schedulers(schedulers)
        if error is not None:
            scheduler_errors: list[Exception] = []
            for scheduler in schedulers:
                try:
                    scheduler.fail(f"FTE worker failure handling failed: {error}")
                except Exception as scheduler_error:
                    # The scheduler state transition precedes its callbacks;
                    # completion must still unblock every failure observer.
                    scheduler_errors.append(scheduler_error)
            if scheduler_errors:
                original_error = error
                error = RuntimeError(f"{error}; scheduler failure callback also failed: {scheduler_errors[0]}")
                error.__cause__ = original_error
        try:
            if self.target is not None:
                self.target._complete_fte_worker_failure_reconciliation(error)
            if error is None:
                self.completion.set_result(None)
            else:
                self.completion.set_exception(error)
        finally:
            with _WORKER_FAILURE_RECONCILIATION_LOCK:
                if _WORKER_FAILURE_RECONCILIATIONS.get(self.failure_key) is self.state:
                    _WORKER_FAILURE_RECONCILIATIONS.pop(self.failure_key, None)

    def _complete_after_futures(
        self,
        futures: list[Future[None]],
        *,
        schedulers: tuple[FteQueryScheduler, ...],
    ) -> None:
        if not futures:
            self.complete(None)
            return
        lock = threading.Lock()
        remaining = len(futures)
        first_error: BaseException | None = None

        def completed(future: Future[None]) -> None:
            nonlocal remaining, first_error
            error: BaseException | None
            try:
                future.result()
            except BaseException as exc:
                error = exc
            else:
                error = None
            should_complete = False
            with lock:
                if first_error is None and error is not None:
                    first_error = error
                remaining -= 1
                should_complete = remaining == 0
            if should_complete:
                self.complete(first_error, schedulers=schedulers)

        for future in futures:
            future.add_done_callback(completed)

    def complete_after_pending_drain(
        self,
        schedulers: tuple[FteQueryScheduler, ...],
        pending_drain: Future[None],
    ) -> None:
        if not self.owns_reconciliation:
            raise RuntimeError("only the reconciliation owner can publish completion")

        def pending_drain_completed(completion: Future[None]) -> None:
            unique_schedulers = self._registered_schedulers(schedulers)
            try:
                completion.result()
                drain_completions = [scheduler.enqueue_drain_barrier() for scheduler in unique_schedulers]
            except BaseException as exc:
                self.complete(exc, schedulers=unique_schedulers)
                return
            self._complete_after_futures(
                drain_completions,
                schedulers=unique_schedulers,
            )

        pending_drain.add_done_callback(pending_drain_completed)

    def reconcile(self) -> _FteWorkerFailureReconciliationResult:
        manager_instance_id, _worker_id, worker_incarnation_id = self.failure_key
        return _reconcile_fte_worker_failure(
            self.event,
            manager_instance_id=manager_instance_id,
            worker_incarnation_id=worker_incarnation_id,
            event_query_id=str(self.event.query_id),
            reconciliation_state=self.state,
        )


def begin_fte_worker_failure_reconciliation(event: Any) -> _FteWorkerFailureReconciliation | None:
    """Elect one full failure publisher; duplicate observers join its barrier."""

    manager_instance_id = str(event.manager_instance_id)
    worker_id = str(event.worker_id)
    worker_incarnation_id = str(event.worker_incarnation_id)
    event_query_id = str(event.query_id)
    with _FTE_REGISTRY_LOCK:
        event_scheduler = _FTE_SCHEDULERS.get(event_query_id)
    if event_scheduler is not None and not event_scheduler.is_owned_by_manager_instance(manager_instance_id):
        return None

    target = _worker_failure_reconciliation_target(
        worker_id,
        manager_instance_id=manager_instance_id,
        worker_incarnation_id=worker_incarnation_id,
    )
    failure_key = (manager_instance_id, worker_id, worker_incarnation_id)
    with _WORKER_FAILURE_RECONCILIATION_LOCK:
        state = _WORKER_FAILURE_RECONCILIATIONS.get(failure_key)
        owns_reconciliation = state is None
        if state is None:
            state = _FteWorkerFailureReconciliationState(failure_key)
            _WORKER_FAILURE_RECONCILIATIONS[failure_key] = state
        if event_scheduler is not None:
            # The local fence must be visible before a duplicate scheduler is
            # allowed to process a causally following CANCELED/ABORTED event.
            state.fence_scheduler(event_scheduler)
        return _FteWorkerFailureReconciliation(
            event,
            state=state,
            target=target if owns_reconciliation else None,
            owns_reconciliation=owns_reconciliation,
        )


def quarantine_fte_worker(
    worker_id: str,
    *,
    manager_instance_id: str,
    worker_incarnation_id: str,
) -> None:
    """Make a failed worker ineligible before per-query reconciliation."""

    worker_id = str(worker_id)
    if not worker_id:
        raise ValueError("worker_id must be non-empty")
    normalized_manager_instance_id = str(manager_instance_id).strip()
    normalized_worker_incarnation_id = str(worker_incarnation_id)
    if not normalized_worker_incarnation_id:
        raise ValueError("worker_incarnation_id must be non-empty")
    with _FTE_REGISTRY_LOCK:
        handle = _FTE_WORKER_HANDLES.get(worker_id)
        if handle is None:
            return
        if str(handle.manager_instance_id) != normalized_manager_instance_id:
            return
        if str(handle.worker_incarnation_id) != normalized_worker_incarnation_id:
            return
        handle._fte_healthy = False


def retire_fte_worker_for_failure(
    worker_id: str,
    error: Any,
    *,
    manager_instance_id: str,
    worker_incarnation_id: str,
) -> None:
    target = _worker_failure_reconciliation_target(
        worker_id,
        manager_instance_id=manager_instance_id,
        worker_incarnation_id=worker_incarnation_id,
    )
    failure = _worker_failure_payload(
        worker_id,
        error,
        worker_incarnation_id=worker_incarnation_id,
    )
    try:
        _mark_fte_worker_failed(
            worker_id,
            failure,
            manager_instance_id=manager_instance_id,
            worker_incarnation_id=worker_incarnation_id,
            primary_worker_process_terminated=_worker_actor_death_confirms_quiescence(error),
            reconcile_query=False,
        )
    except BaseException as exc:
        if target is not None:
            target._complete_fte_worker_failure_reconciliation(exc)
        raise
    else:
        if target is not None:
            target._complete_fte_worker_failure_reconciliation(None)


def _reconcile_fte_worker_failure(
    event: Any,
    *,
    manager_instance_id: str,
    worker_incarnation_id: str,
    event_query_id: str,
    reconciliation_state: _FteWorkerFailureReconciliationState,
) -> _FteWorkerFailureReconciliationResult:
    failure = _worker_failure_payload(
        event.worker_id,
        event.error,
        worker_incarnation_id=worker_incarnation_id,
    )
    quarantine_fte_worker(
        event.worker_id,
        manager_instance_id=manager_instance_id,
        worker_incarnation_id=worker_incarnation_id,
    )
    affected_query_ids = _query_ids_owned_by_fte_workers(
        {str(event.worker_id)},
        {str(event.worker_id): worker_incarnation_id},
    )
    affected_query_ids.add(event_query_id)
    for query_id in sorted(affected_query_ids):
        with _FTE_REGISTRY_LOCK:
            if query_id in _FTE_CLOSING_QUERIES:
                continue
            scheduler = _FTE_SCHEDULERS.get(query_id)
        if scheduler is None or not scheduler.is_owned_by_manager_instance(manager_instance_id):
            continue
        reconciliation_state.fence_scheduler(scheduler)
    reconciliation_schedulers: dict[str, FteQueryScheduler] = {}
    for registered_scheduler in reconciliation_state.schedulers():
        query_id = str(registered_scheduler.query_id)
        with _FTE_REGISTRY_LOCK:
            current_scheduler = _FTE_SCHEDULERS.get(query_id)
            if (
                query_id in _FTE_CLOSING_QUERIES
                or current_scheduler is None
                or current_scheduler is not registered_scheduler
            ):
                continue
        if current_scheduler.is_owned_by_manager_instance(manager_instance_id):
            reconciliation_schedulers[query_id] = current_scheduler
    reconciliation_query_ids = set(reconciliation_schedulers)
    scheduled_by_fragment = _mark_fte_worker_failed(
        event.worker_id,
        failure,
        query_id_filters=reconciliation_query_ids,
        manager_instance_id=manager_instance_id,
        worker_incarnation_id=worker_incarnation_id,
        primary_worker_process_terminated=_worker_actor_death_confirms_quiescence(event.error),
        reconcile_query=bool(reconciliation_query_ids),
    )
    for query_id, scheduler in reconciliation_schedulers.items():
        delay_s = _fte_retry_remaining_delay_s(query_id)
        if delay_s > 0:
            scheduler.arm_retry_delay(delay_s)
    return _FteWorkerFailureReconciliationResult(
        scheduled_by_fragment=tuple(scheduled_by_fragment),
        schedulers=reconciliation_state.schedulers(),
    )
