# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from vane.runners.ray.query_resource_graph import QueryAllocation, QueryResourceGraph
from vane.runners.ray.query_resource_manager import RayQueryResourceManager

_LOCK = threading.RLock()
_MANAGERS: dict[str, RayQueryResourceManager] = {}


def register_query_resource_graph(
    graph: QueryResourceGraph,
    allocation: QueryAllocation,
    *,
    admission_open: bool = True,
    reservation_ratio: float = 0.5,
    on_change: Callable[[], None] | None = None,
    on_eligible_units_change: Callable[[tuple[str, ...]], None] | None = None,
) -> RayQueryResourceManager:
    """Atomically publish the driver-local resource manager for a query."""

    manager = RayQueryResourceManager(
        graph,
        allocation,
        admission_open=admission_open,
        reservation_ratio=reservation_ratio,
        on_change=on_change,
        on_eligible_units_change=on_eligible_units_change,
    )
    query_id = graph.query_id
    with _LOCK:
        if query_id in _MANAGERS:
            raise ValueError(f"query resource graph is already registered: {query_id}")
        _MANAGERS[query_id] = manager
    return manager


def get_query_resource_manager(query_id: str) -> RayQueryResourceManager:
    query_key = str(query_id or "").strip()
    if not query_key:
        raise ValueError("query_id must be non-empty")
    with _LOCK:
        manager = _MANAGERS.get(query_key)
    if manager is None:
        raise KeyError(f"query resource graph is not registered: {query_key}")
    return manager


def query_resource_manager_snapshot(query_id: str) -> dict[str, Any]:
    query_key = str(query_id or "").strip()
    if not query_key:
        return {}
    with _LOCK:
        manager = _MANAGERS.get(query_key)
    return {} if manager is None else manager.snapshot()


def mark_materialization_barrier_completed(query_id: str, physical_node_id: str) -> bool:
    """Advance one driver-local query phase from a native barrier event."""

    query_key = str(query_id or "").strip()
    node_key = str(physical_node_id or "").strip()
    if not query_key or not node_key:
        raise ValueError("materialization barrier completion requires query_id and physical_node_id")
    return get_query_resource_manager(query_key).mark_materialization_barrier_completed_for_node(node_key)


def release_query_resource_manager(query_id: str, *, reason: str) -> dict[str, Any]:
    query_key = str(query_id or "").strip()
    if not query_key:
        return {"released": False, "task_lease_count": 0, "output_lease_count": 0}
    with _LOCK:
        manager = _MANAGERS.get(query_key)
    if manager is None:
        return {"released": False, "task_lease_count": 0, "output_lease_count": 0}
    released = manager.cancel(reason)
    with _LOCK:
        if _MANAGERS.get(query_key) is manager:
            _MANAGERS.pop(query_key, None)
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
    "mark_materialization_barrier_completed",
    "query_resource_manager_snapshot",
    "register_query_resource_graph",
    "release_query_resource_manager",
]
