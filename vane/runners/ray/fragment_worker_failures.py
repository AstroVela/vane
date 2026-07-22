# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from vane.runners.ray.fragment_registry import (
    _FTE_CLOSING_QUERIES,
    _FTE_REGISTRY_LOCK,
    _FTE_SCHEDULERS,
)
from vane.runners.ray.fte_fragment_scheduler import (
    _expanded_fte_failed_worker_ids,
    _fte_retry_remaining_delay_s,
    _mark_fte_worker_failed,
)


def mark_fte_worker_failed_for_event(event: Any) -> list[tuple[str, str, list[Any], list[Any]]]:
    with _FTE_REGISTRY_LOCK:
        if str(event.query_id) in _FTE_CLOSING_QUERIES:
            return []
        scheduler = _FTE_SCHEDULERS.get(event.query_id)
    if scheduler is None:
        return []
    failed_worker_ids = (
        {str(item) for item in event.failed_worker_ids}
        if event.failed_worker_ids is not None
        else _expanded_fte_failed_worker_ids(event.worker_id)
    )
    new_failed_worker_ids = scheduler.record_worker_failure(failed_worker_ids)
    if not new_failed_worker_ids:
        return []
    scheduled_by_stage = _mark_fte_worker_failed(
        event.worker_id,
        event.error,
        query_id_filter=event.query_id,
        failed_worker_ids_override=new_failed_worker_ids,
    )
    delay_s = _fte_retry_remaining_delay_s(event.query_id)
    if delay_s > 0:
        scheduler.arm_retry_delay(delay_s)
    return scheduled_by_stage
