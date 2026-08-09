# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from concurrent.futures import Future
from typing import Any

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

_WORKER_FAILURE_RECONCILIATION_LOCK = threading.Lock()
_WORKER_FAILURE_RECONCILIATIONS: dict[tuple[str, str, str], Future[None]] = {}


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
    failure = _worker_failure_payload(
        worker_id,
        error,
        worker_incarnation_id=worker_incarnation_id,
    )
    _mark_fte_worker_failed(
        worker_id,
        failure,
        manager_instance_id=manager_instance_id,
        worker_incarnation_id=worker_incarnation_id,
        primary_worker_process_terminated=_worker_actor_death_confirms_quiescence(error),
        reconcile_query=False,
    )


def _reconcile_fte_worker_failure(
    event: Any,
    *,
    manager_instance_id: str,
    worker_incarnation_id: str,
    event_query_id: str,
) -> list[tuple[str, str, list[Any], list[Any]]]:
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
    reconciliation_schedulers = {}
    for query_id in sorted(affected_query_ids):
        with _FTE_REGISTRY_LOCK:
            if query_id in _FTE_CLOSING_QUERIES:
                continue
            scheduler = _FTE_SCHEDULERS.get(query_id)
        if scheduler is None or not scheduler.is_owned_by_manager_instance(manager_instance_id):
            continue
        if scheduler.record_worker_failure(
            event.worker_id,
            worker_incarnation_id=worker_incarnation_id,
        ):
            reconciliation_schedulers[query_id] = scheduler
    reconciliation_query_ids = set(reconciliation_schedulers)
    try:
        scheduled_by_fragment = _mark_fte_worker_failed(
            event.worker_id,
            failure,
            query_id_filters=reconciliation_query_ids,
            manager_instance_id=manager_instance_id,
            worker_incarnation_id=worker_incarnation_id,
            primary_worker_process_terminated=_worker_actor_death_confirms_quiescence(event.error),
            reconcile_query=bool(reconciliation_query_ids),
        )
    except BaseException as exc:
        for scheduler in reconciliation_schedulers.values():
            scheduler.fail(f"FTE worker failure handling failed: {exc}")
        raise
    for query_id, scheduler in reconciliation_schedulers.items():
        delay_s = _fte_retry_remaining_delay_s(query_id)
        if delay_s > 0:
            scheduler.arm_retry_delay(delay_s)
    return scheduled_by_fragment


def mark_fte_worker_failed_for_event(event: Any) -> list[tuple[str, str, list[Any], list[Any]]]:
    manager_instance_id = str(event.manager_instance_id)
    worker_incarnation_id = str(event.worker_incarnation_id)
    event_query_id = str(event.query_id)
    with _FTE_REGISTRY_LOCK:
        event_scheduler = _FTE_SCHEDULERS.get(event_query_id)
    if event_scheduler is not None and not event_scheduler.is_owned_by_manager_instance(manager_instance_id):
        return []

    # Independent status watchers can observe the same actor death. Only the
    # owner mutates fragment state; later observers join its completion.
    failure_key = (manager_instance_id, str(event.worker_id), worker_incarnation_id)
    with _WORKER_FAILURE_RECONCILIATION_LOCK:
        reconciliation = _WORKER_FAILURE_RECONCILIATIONS.get(failure_key)
        owns_reconciliation = reconciliation is None
        if owns_reconciliation:
            reconciliation = Future()
            _WORKER_FAILURE_RECONCILIATIONS[failure_key] = reconciliation
    assert reconciliation is not None
    if not owns_reconciliation:
        reconciliation.result()
        return []

    try:
        scheduled_by_fragment = _reconcile_fte_worker_failure(
            event,
            manager_instance_id=manager_instance_id,
            worker_incarnation_id=worker_incarnation_id,
            event_query_id=event_query_id,
        )
    except BaseException as exc:
        reconciliation.set_exception(exc)
        raise
    else:
        reconciliation.set_result(None)
        return scheduled_by_fragment
    finally:
        with _WORKER_FAILURE_RECONCILIATION_LOCK:
            if _WORKER_FAILURE_RECONCILIATIONS.get(failure_key) is reconciliation:
                _WORKER_FAILURE_RECONCILIATIONS.pop(failure_key, None)
