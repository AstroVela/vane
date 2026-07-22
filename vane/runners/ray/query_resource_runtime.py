# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from vane.runners.ray.query_execution_graph import QueryAllocation, QueryExecutionGraph
from vane.runners.ray.query_resource_manager import QueryResourceManager

_LOCK = threading.RLock()
_MANAGERS: dict[str, QueryResourceManager] = {}


def register_query_graph(
    graph: QueryExecutionGraph,
    allocation: QueryAllocation,
    *,
    reservation_ratio: float = 0.5,
    on_change: Callable[[], None] | None = None,
) -> QueryResourceManager:
    """Validate and atomically publish the only resource manager for a query."""

    graph.validate_allocation(allocation)
    manager = QueryResourceManager(
        graph,
        allocation,
        reservation_ratio=reservation_ratio,
        on_change=on_change,
    )
    query_id = graph.query_id
    with _LOCK:
        if query_id in _MANAGERS:
            raise ValueError(f"query graph is already registered: {query_id}")
        _MANAGERS[query_id] = manager
    return manager


def get_query_resource_manager(query_id: str) -> QueryResourceManager:
    query_key = str(query_id or "").strip()
    if not query_key:
        raise ValueError("query_id must be non-empty")
    with _LOCK:
        manager = _MANAGERS.get(query_key)
    if manager is None:
        raise KeyError(f"query graph is not registered: {query_key}")
    return manager


def query_resource_manager_snapshot(query_id: str) -> dict[str, Any]:
    query_key = str(query_id or "").strip()
    if not query_key:
        return {}
    with _LOCK:
        manager = _MANAGERS.get(query_key)
    return {} if manager is None else manager.snapshot()


def release_query_resource_manager(query_id: str, *, reason: str) -> dict[str, Any]:
    query_key = str(query_id or "").strip()
    if not query_key:
        return {"released": False, "task_lease_count": 0, "output_lease_count": 0}
    with _LOCK:
        manager = _MANAGERS.pop(query_key, None)
    if manager is None:
        return {"released": False, "task_lease_count": 0, "output_lease_count": 0}
    released = manager.cancel(reason)
    return {"released": True, **released}


def clear_query_resource_managers() -> None:
    with _LOCK:
        managers = list(_MANAGERS.values())
        _MANAGERS.clear()
    for manager in managers:
        manager.cancel("runtime_registry_cleared")


__all__ = [
    "clear_query_resource_managers",
    "get_query_resource_manager",
    "query_resource_manager_snapshot",
    "register_query_graph",
    "release_query_resource_manager",
]
