# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from duckdb.runners.ray.fragment_registry import (
    _FTE_CLOSING_QUERIES,
    _FTE_REGISTRY_LOCK,
    _FTE_SCHEDULERS,
    _FTE_WORKER_HANDLES,
)
from duckdb.runners.ray.fte_fragment_scheduler import (
    _fte_retry_remaining_delay_s,
    _mark_fte_worker_failed,
    _worker_actor_death_confirms_quiescence,
    _worker_failure_payload,
)


def quarantine_fte_worker(
    worker_id: str,
    *,
    manager_instance_id: str | None = None,
    worker_incarnation_id: str | None = None,
) -> frozenset[str]:
    """Make a failed worker ineligible before per-query reconciliation."""

    worker_id = str(worker_id or "")
    failed_worker_ids = frozenset({worker_id}) if worker_id else frozenset()
    # Preserve "" as the explicit legacy scope. Only None means that the
    # caller has no ownership metadata and requires the compatibility path.
    normalized_manager_instance_id = None if manager_instance_id is None else str(manager_instance_id or "").strip()
    with _FTE_REGISTRY_LOCK:
        for failed_worker_id in failed_worker_ids:
            handle = _FTE_WORKER_HANDLES.get(failed_worker_id)
            if handle is not None:
                if (
                    normalized_manager_instance_id is not None
                    and str(getattr(handle, "manager_instance_id", "")) != normalized_manager_instance_id
                ):
                    return frozenset()
                if worker_incarnation_id is not None and str(getattr(handle, "worker_incarnation_id", "")) != str(
                    worker_incarnation_id
                ):
                    continue
                handle._fte_healthy = False
    return failed_worker_ids


def retire_fte_worker_for_failure(
    worker_id: str,
    error: Any,
    *,
    failed_worker_ids: set[str] | frozenset[str] | None = None,
    manager_instance_id: str | None = None,
    worker_incarnation_id: str | None = None,
) -> None:
    failure = _worker_failure_payload(
        worker_id,
        error,
        worker_incarnation_id=worker_incarnation_id,
    )
    _mark_fte_worker_failed(
        worker_id,
        failure,
        failed_worker_ids_override=failed_worker_ids,
        manager_instance_id=manager_instance_id,
        worker_incarnation_id=worker_incarnation_id,
        primary_worker_process_terminated=_worker_actor_death_confirms_quiescence(error),
        reconcile_query=False,
    )


def mark_fte_worker_failed_for_event(event: Any) -> list[tuple[str, str, list[Any], list[Any]]]:
    manager_instance_id = getattr(event, "manager_instance_id", None)
    worker_incarnation_id = getattr(event, "worker_incarnation_id", None)
    failure = _worker_failure_payload(
        event.worker_id,
        event.error,
        worker_incarnation_id=worker_incarnation_id,
    )
    with _FTE_REGISTRY_LOCK:
        query_closing = str(event.query_id) in _FTE_CLOSING_QUERIES
        scheduler = _FTE_SCHEDULERS.get(event.query_id)
    # Do not collapse "" into None: legacy schedulers are explicitly owned.
    if (
        scheduler is not None
        and manager_instance_id is not None
        and not scheduler.is_owned_by_manager_instance(manager_instance_id)
    ):
        return []
    failed_worker_ids = (
        {str(item) for item in event.failed_worker_ids}
        if event.failed_worker_ids is not None
        else set(
            quarantine_fte_worker(
                event.worker_id,
                manager_instance_id=manager_instance_id,
                worker_incarnation_id=worker_incarnation_id,
            )
        )
    )
    new_failed_worker_ids: frozenset[str] = frozenset()
    reconcile_query = scheduler is not None and not query_closing
    if reconcile_query:
        new_failed_worker_ids = scheduler.record_worker_failure(
            failed_worker_ids,
            worker_incarnation_id=worker_incarnation_id,
        )
        reconcile_query = bool(new_failed_worker_ids)
    scheduled_by_stage = _mark_fte_worker_failed(
        event.worker_id,
        failure,
        query_id_filter=event.query_id,
        failed_worker_ids_override=new_failed_worker_ids if reconcile_query else failed_worker_ids,
        manager_instance_id=manager_instance_id,
        worker_incarnation_id=worker_incarnation_id,
        primary_worker_process_terminated=_worker_actor_death_confirms_quiescence(event.error),
        reconcile_query=reconcile_query,
    )
    if not reconcile_query:
        return []
    delay_s = _fte_retry_remaining_delay_s(event.query_id)
    if delay_s > 0:
        scheduler.arm_retry_delay(delay_s)
    return scheduled_by_stage
