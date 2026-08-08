# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from vane.runners.ray.fragment_registry import (
    _FTE_PARTITION_TASK_WAITERS,
    _FTE_REGISTRY_LOCK,
)


def register_fte_partition_task_waiter(
    key: tuple[str, str, int],
    *,
    resource_query_id: str,
    task_id: str,
    attempt_id: str,
) -> None:
    normalized_key = (str(key[0]), str(key[1]), int(key[2]))
    identity = (
        str(resource_query_id),
        str(task_id),
        str(attempt_id),
    )
    if not identity[0]:
        raise ValueError("FTE partition waiter resource_query_id must be non-empty")
    with _FTE_REGISTRY_LOCK:
        existing = _FTE_PARTITION_TASK_WAITERS.get(normalized_key)
        if existing is not None and existing != identity:
            raise RuntimeError(
                "FTE partition waiter identity changed before the prior waiter was released: "
                f"key={normalized_key} existing={existing} new={identity}"
            )
        _FTE_PARTITION_TASK_WAITERS[normalized_key] = identity


def release_fte_partition_task_waiter(key: tuple[str, str, int]) -> bool:
    normalized_key = (str(key[0]), str(key[1]), int(key[2]))
    with _FTE_REGISTRY_LOCK:
        identity = _FTE_PARTITION_TASK_WAITERS.pop(normalized_key, None)
    if identity is None:
        return False

    from vane.runners.ray.query_resource_runtime import get_query_resource_manager

    try:
        manager = get_query_resource_manager(identity[0])
    except KeyError:
        return False
    return manager.remove_task_waiter(identity[1], identity[2])


def release_fte_query_task_waiters(query_id: str) -> int:
    query_key = str(query_id)
    with _FTE_REGISTRY_LOCK:
        identities = [
            _FTE_PARTITION_TASK_WAITERS.pop(key) for key in list(_FTE_PARTITION_TASK_WAITERS) if key[0] == query_key
        ]
    from vane.runners.ray.query_resource_runtime import get_query_resource_manager

    released = 0
    for resource_query_id, task_id, attempt_id in identities:
        try:
            manager = get_query_resource_manager(resource_query_id)
        except KeyError:
            continue
        released += int(manager.remove_task_waiter(task_id, attempt_id))
    return released


__all__ = [
    "register_fte_partition_task_waiter",
    "release_fte_partition_task_waiter",
    "release_fte_query_task_waiters",
]
