# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from vane.runners.ray.fragment_registry import (
    _FTE_FRAGMENT_EXECUTIONS,
    _FTE_PARTITION_OWNERS,
    _FTE_PENDING_WORKER_RESERVATIONS,
    _FTE_REGISTRY_LOCK,
    _FTE_WORKER_RESERVATION_GENERATIONS,
)
from vane.runners.ray.fragment_submission_window import (
    release_fte_partition_submission,
)
from vane.runners.ray.fragment_worker_waiters import (
    release_fte_partition_task_waiter,
)


def pending_fte_worker_reservation_partition(future: Any) -> tuple[Any | None, Any | None]:
    with _FTE_REGISTRY_LOCK:
        fragment_execution = _FTE_FRAGMENT_EXECUTIONS.get((future.query_id, future.fragment_id))
    if fragment_execution is None:
        return None, None
    return fragment_execution, fragment_execution.partitions.get(future.partition_id)


def pending_fte_worker_reservation_future(
    query_id: str,
    fragment_id: str,
    partition_id: int,
) -> Any | None:
    key = (str(query_id), str(fragment_id), int(partition_id))
    with _FTE_REGISTRY_LOCK:
        return _FTE_PENDING_WORKER_RESERVATIONS.get(key)


def cancel_fte_worker_reservation_future(
    future: Any,
    *,
    allow_next_submission: bool = True,
) -> None:
    with _FTE_REGISTRY_LOCK:
        if _FTE_PENDING_WORKER_RESERVATIONS.get(future.key) is future:
            _FTE_PENDING_WORKER_RESERVATIONS.pop(future.key, None)
    release_fte_partition_task_waiter(future.key)
    release_fte_partition_submission(
        *future.key,
        allow_next=allow_next_submission,
    )
    future.cancel()


def fte_worker_reservation_future_is_current(future: Any) -> bool:
    with _FTE_REGISTRY_LOCK:
        return _FTE_PENDING_WORKER_RESERVATIONS.get(future.key) is future


def fte_pending_worker_reservation_done_count(query_id_filter: str | None = None) -> int:
    if query_id_filter is not None:
        query_id_filter = str(query_id_filter)
    with _FTE_REGISTRY_LOCK:
        return sum(
            1
            for key, future in _FTE_PENDING_WORKER_RESERVATIONS.items()
            if (query_id_filter is None or key[0] == query_id_filter) and future.done()
        )


def remove_pending_fte_worker_reservation_if_current(key: tuple[str, str, int], future: Any) -> bool:
    removed = False
    with _FTE_REGISTRY_LOCK:
        if _FTE_PENDING_WORKER_RESERVATIONS.get(key) is not future:
            return False
        _FTE_PENDING_WORKER_RESERVATIONS.pop(key, None)
        removed = True
    if removed:
        release_fte_partition_task_waiter(key)
    return removed


def fte_worker_reservation_event_state(event: Any) -> tuple[tuple[str, str, int], Any | None, Any | None]:
    key = (event.query_id, str(event.fragment_id), int(event.partition_id))
    with _FTE_REGISTRY_LOCK:
        future = _FTE_PENDING_WORKER_RESERVATIONS.get(key)
        generation = _FTE_WORKER_RESERVATION_GENERATIONS.get(key)
        fragment_execution = _FTE_FRAGMENT_EXECUTIONS.get((event.query_id, str(event.fragment_id)))
    if (
        future is None
        or generation != int(event.reservation_generation)
        or int(future.reservation_generation) != int(event.reservation_generation)
        or fragment_execution is None
    ):
        return key, None, None
    if int(fragment_execution.fragment_execution_id) != int(event.fragment_execution_id):
        return key, None, None
    return key, future, fragment_execution


def fte_partition_owner(
    query_id: str,
    fragment_id: str,
    partition_id: int,
) -> Any | None:
    with _FTE_REGISTRY_LOCK:
        return _FTE_PARTITION_OWNERS.get((str(query_id), str(fragment_id), int(partition_id)))
