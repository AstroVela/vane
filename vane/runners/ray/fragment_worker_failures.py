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

_WORKER_FAILURE_RECONCILIATION_LOCK = threading.RLock()


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
    retry_delay_completions: tuple[Future[None], ...]


class _FteWorkerFailureReconciliationState:
    """Serialize a dynamically growing worker-failure reconciliation."""

    def __init__(
        self,
        failure_key: tuple[str, str, str],
        *,
        target: Any | None,
    ) -> None:
        self.failure_key = failure_key
        self.target = target
        self.completion: Future[None] = Future()
        self._schedulers: dict[int, FteQueryScheduler] = {}
        self._pending_schedulers: dict[int, FteQueryScheduler] = {}
        self._initial_reconciliation_pending = True
        self._runner_active = False
        self._pending_publications = 0
        self._error: BaseException | None = None
        self._closed = False

    def fence_scheduler(self, scheduler: FteQueryScheduler) -> bool:
        """Fence a scheduler and enqueue it for reconciliation exactly once."""

        scheduler_identity = id(scheduler)
        _manager_instance_id, worker_id, worker_incarnation_id = self.failure_key
        with _WORKER_FAILURE_RECONCILIATION_LOCK:
            if self._closed:
                return False
            if scheduler_identity in self._schedulers:
                return True
            if not scheduler.record_worker_failure(
                worker_id,
                worker_incarnation_id=worker_incarnation_id,
            ):
                return False
            self._schedulers[scheduler_identity] = scheduler
            self._pending_schedulers[scheduler_identity] = scheduler
            return True

    def schedulers(self) -> tuple[FteQueryScheduler, ...]:
        with _WORKER_FAILURE_RECONCILIATION_LOCK:
            return tuple(self._schedulers.values())

    def try_claim_runner(self) -> bool:
        with _WORKER_FAILURE_RECONCILIATION_LOCK:
            if self._closed or self._error is not None or self._runner_active:
                return False
            if not self._initial_reconciliation_pending and not self._pending_schedulers:
                return False
            self._runner_active = True
            return True

    def take_reconciliation_batch(self) -> tuple[bool, tuple[FteQueryScheduler, ...]] | None:
        with _WORKER_FAILURE_RECONCILIATION_LOCK:
            if self._closed or not self._runner_active:
                return None
            if self._error is not None:
                raise self._error
            initial_reconciliation = self._initial_reconciliation_pending
            self._initial_reconciliation_pending = False
            schedulers = tuple(self._pending_schedulers.values())
            self._pending_schedulers.clear()
            if not initial_reconciliation and not schedulers:
                return None
            return initial_reconciliation, schedulers

    def add_publication(
        self,
        pending_drain: Future[None],
        schedulers: tuple[FteQueryScheduler, ...],
        retry_delay_completions: tuple[Future[None], ...],
    ) -> None:
        with _WORKER_FAILURE_RECONCILIATION_LOCK:
            if self._closed or self._error is not None:
                return
            self._pending_publications += 1

        publication_lock = threading.Lock()
        publication_finished = False

        def finish_publication(error: BaseException | None) -> None:
            nonlocal publication_finished
            with publication_lock:
                if publication_finished:
                    return
                publication_finished = True
            self._finish_publication(error)

        def prerequisites_completed(error: BaseException | None) -> None:
            if error is not None:
                finish_publication(error)
                return
            try:
                drain_completions = [scheduler.enqueue_drain_barrier() for scheduler in schedulers]
            except BaseException as exc:
                finish_publication(exc)
                return
            if not drain_completions:
                finish_publication(None)
                return

            barrier_lock = threading.Lock()
            remaining = len(drain_completions)
            first_error: BaseException | None = None

            def drain_completed(drain_completion: Future[None]) -> None:
                nonlocal remaining, first_error
                try:
                    drain_completion.result()
                except BaseException as exc:
                    error: BaseException | None = exc
                else:
                    error = None
                with barrier_lock:
                    if first_error is None and error is not None:
                        first_error = error
                    remaining -= 1
                    all_completed = remaining == 0
                if all_completed:
                    finish_publication(first_error)

            for drain_completion in drain_completions:
                drain_completion.add_done_callback(drain_completed)

        prerequisite_lock = threading.Lock()
        remaining_prerequisites = 1 + len(retry_delay_completions)

        def prerequisite_completed(completion: Future[None]) -> None:
            nonlocal remaining_prerequisites
            try:
                completion.result()
            except BaseException as exc:
                error: BaseException | None = exc
            else:
                error = None
            with prerequisite_lock:
                remaining_prerequisites -= 1
                all_completed = remaining_prerequisites == 0
            if error is not None:
                finish_publication(error)
            elif all_completed:
                prerequisites_completed(None)

        for prerequisite in (pending_drain, *retry_delay_completions):
            prerequisite.add_done_callback(prerequisite_completed)

    def _finish_publication(self, error: BaseException | None) -> None:
        with _WORKER_FAILURE_RECONCILIATION_LOCK:
            if self._closed:
                return
            self._pending_publications -= 1
            if self._error is None and error is not None:
                self._error = error
            publication = _close_reconciliation_if_ready_locked(self)
            if publication is not None:
                _publish_reconciliation_completion(*publication)

    def release_runner(self) -> bool:
        """Keep the runner when work arrived, otherwise make completion eligible."""

        publication: _ReconciliationPublication | None
        with _WORKER_FAILURE_RECONCILIATION_LOCK:
            if self._closed:
                return False
            if self._error is not None:
                self._runner_active = False
                publication = _close_reconciliation_locked(self, self._error)
            elif self._initial_reconciliation_pending or self._pending_schedulers:
                return True
            else:
                self._runner_active = False
                publication = _close_reconciliation_if_ready_locked(self)
            if publication is not None:
                _publish_reconciliation_completion(*publication)
        return False

    def fail(self, error: BaseException) -> None:
        with _WORKER_FAILURE_RECONCILIATION_LOCK:
            if self._closed:
                return
            if self._error is None:
                self._error = error
            publication = _close_reconciliation_locked(self, self._error)
            _publish_reconciliation_completion(*publication)


_WORKER_FAILURE_RECONCILIATIONS: dict[
    tuple[str, str, str],
    _FteWorkerFailureReconciliationState,
] = {}


_ReconciliationPublication = tuple[
    _FteWorkerFailureReconciliationState,
    BaseException | None,
    tuple["FteQueryScheduler", ...],
]


def _close_reconciliation_locked(
    state: _FteWorkerFailureReconciliationState,
    error: BaseException | None,
) -> _ReconciliationPublication:
    state._closed = True
    state._runner_active = False
    state._pending_schedulers.clear()
    if _WORKER_FAILURE_RECONCILIATIONS.get(state.failure_key) is state:
        _WORKER_FAILURE_RECONCILIATIONS.pop(state.failure_key, None)
    return state, error, tuple(state._schedulers.values())


def _close_reconciliation_if_ready_locked(
    state: _FteWorkerFailureReconciliationState,
) -> _ReconciliationPublication | None:
    if state._closed or state._runner_active:
        return None
    if state._error is not None:
        return _close_reconciliation_locked(state, state._error)
    if state._initial_reconciliation_pending or state._pending_schedulers or state._pending_publications:
        return None
    return _close_reconciliation_locked(state, None)


def _publish_reconciliation_completion(
    state: _FteWorkerFailureReconciliationState,
    error: BaseException | None,
    schedulers: tuple[FteQueryScheduler, ...],
) -> None:
    # Closing, publishing, and removing the state share the global lock.  An
    # observer therefore joins either this state before its linearization
    # point or a successor after completion, never an unpublished gap.
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
        if state.target is not None:
            state.target._complete_fte_worker_failure_reconciliation(error)
    except BaseException as exc:
        if error is None:
            error = exc
            for scheduler in schedulers:
                try:
                    scheduler.fail(f"FTE worker failure completion publication failed: {exc}")
                except Exception:
                    pass
        else:
            original_error = error
            error = RuntimeError(f"{error}; worker failure completion publication also failed: {exc}")
            error.__cause__ = original_error
    if error is None:
        state.completion.set_result(None)
    else:
        state.completion.set_exception(error)


class _FteWorkerFailureReconciliation:
    """Run one serialized slice of a shared worker-incarnation failure."""

    def __init__(
        self,
        event: Any,
        *,
        state: _FteWorkerFailureReconciliationState,
        owns_runner: bool,
    ) -> None:
        self.event = event
        self.state = state
        self.failure_key = state.failure_key
        self.completion = state.completion
        self.owns_runner = owns_runner

    def complete_after_pending_drain(
        self,
        schedulers: tuple[FteQueryScheduler, ...],
        pending_drain: Future[None],
        retry_delay_completions: tuple[Future[None], ...],
    ) -> None:
        if not self.owns_runner:
            raise RuntimeError("only the active reconciliation runner can publish progress")
        unique_schedulers = tuple({id(scheduler): scheduler for scheduler in schedulers}.values())
        unique_retry_delay_completions = tuple(
            {id(completion): completion for completion in retry_delay_completions}.values()
        )
        self.state.add_publication(
            pending_drain,
            unique_schedulers,
            unique_retry_delay_completions,
        )

    def reconcile(self) -> _FteWorkerFailureReconciliationResult | None:
        if not self.owns_runner:
            raise RuntimeError("only the active reconciliation runner can reconcile")
        manager_instance_id, _worker_id, worker_incarnation_id = self.failure_key
        quarantine_fte_worker(
            self.event.worker_id,
            manager_instance_id=manager_instance_id,
            worker_incarnation_id=worker_incarnation_id,
        )
        affected_query_ids = _query_ids_owned_by_fte_workers(
            {str(self.event.worker_id)},
            {str(self.event.worker_id): worker_incarnation_id},
        )
        affected_query_ids.add(str(self.event.query_id))
        for query_id in sorted(affected_query_ids):
            with _FTE_REGISTRY_LOCK:
                if query_id in _FTE_CLOSING_QUERIES:
                    continue
                scheduler = _FTE_SCHEDULERS.get(query_id)
            if scheduler is None or not scheduler.is_owned_by_manager_instance(manager_instance_id):
                continue
            self.state.fence_scheduler(scheduler)
        reconciliation_batch = self.state.take_reconciliation_batch()
        if reconciliation_batch is None:
            return None
        return _reconcile_fte_worker_failure(
            self.event,
            manager_instance_id=manager_instance_id,
            worker_incarnation_id=worker_incarnation_id,
            reconciliation_batch=reconciliation_batch,
        )

    def release_runner(self) -> bool:
        if not self.owns_runner:
            raise RuntimeError("only the active reconciliation runner can release ownership")
        return self.state.release_runner()

    def fail(self, error: BaseException) -> None:
        if not self.owns_runner:
            raise RuntimeError("only the active reconciliation runner can fail reconciliation")
        self.state.fail(error)


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
        if state is None:
            state = _FteWorkerFailureReconciliationState(failure_key, target=target)
            _WORKER_FAILURE_RECONCILIATIONS[failure_key] = state
        if event_scheduler is not None:
            # The local fence must be visible before a duplicate scheduler is
            # allowed to process a causally following CANCELED/ABORTED event.
            state.fence_scheduler(event_scheduler)
        owns_runner = state.try_claim_runner()
        return _FteWorkerFailureReconciliation(
            event,
            state=state,
            owns_runner=owns_runner,
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
    reconciliation_batch: tuple[bool, tuple[FteQueryScheduler, ...]],
) -> _FteWorkerFailureReconciliationResult:
    failure = _worker_failure_payload(
        event.worker_id,
        event.error,
        worker_incarnation_id=worker_incarnation_id,
    )
    initial_reconciliation, registered_schedulers = reconciliation_batch
    reconciliation_schedulers: dict[str, FteQueryScheduler] = {}
    for registered_scheduler in registered_schedulers:
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
    if initial_reconciliation or reconciliation_query_ids:
        scheduled_by_fragment = _mark_fte_worker_failed(
            event.worker_id,
            failure,
            query_id_filters=reconciliation_query_ids,
            manager_instance_id=manager_instance_id,
            worker_incarnation_id=worker_incarnation_id,
            primary_worker_process_terminated=_worker_actor_death_confirms_quiescence(event.error),
            reconcile_query=bool(reconciliation_query_ids),
        )
    else:
        scheduled_by_fragment = []
    retry_delay_completions: list[Future[None]] = []
    for query_id, scheduler in reconciliation_schedulers.items():
        delay_s = _fte_retry_remaining_delay_s(query_id)
        if delay_s > 0:
            retry_delay_completions.append(scheduler.arm_retry_delay(delay_s))
    return _FteWorkerFailureReconciliationResult(
        scheduled_by_fragment=tuple(scheduled_by_fragment),
        schedulers=tuple(reconciliation_schedulers.values()),
        retry_delay_completions=tuple(retry_delay_completions),
    )
