# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vane.runners.ray.fragment_registry import _FTE_WORKER_HANDLES
from vane.runners.ray.fte_fragment_scheduler import (
    _filter_workers_for_node_requirements,
    _fte_worker_has_memory_capacity,
    _fte_worker_selection_key,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vane.runners.fte import FteTaskExecutionClass, NodeRequirements


def available_fte_workers(
    current_worker: Any,
    current_worker_id: str | None,
    *,
    exclude: set[str] | None = None,
) -> list[Any]:
    exclude = exclude or set()
    workers = [
        handle
        for worker_id, handle in sorted(_FTE_WORKER_HANDLES.items())
        if handle is not None and str(worker_id) not in exclude and handle._fte_healthy
    ]
    if (
        current_worker not in workers
        and (not current_worker_id or str(current_worker_id) not in exclude)
        and current_worker._fte_healthy
    ):
        workers.append(current_worker)
    return workers


def select_fte_worker(
    current_worker: Any,
    current_worker_id: str | None,
    *,
    exclude: set[str] | None = None,
    allowed_node_ids: set[str] | None = None,
    memory_requirement_bytes: Any = None,
    execution_class: FteTaskExecutionClass | str | None = None,
    node_requirements: NodeRequirements | Mapping[str, Any] | None = None,
    node_requirements_wait_started_at: float | None = None,
) -> Any | None:
    workers = available_fte_workers(current_worker, current_worker_id, exclude=exclude)
    if allowed_node_ids is not None:
        workers = [worker for worker in workers if str(worker.node_id) in allowed_node_ids]
    if not workers:
        return None
    workers = _filter_workers_for_node_requirements(
        workers,
        node_requirements,
        node_requirements_wait_started_at=node_requirements_wait_started_at,
    )
    if not workers:
        return None
    workers_with_capacity = [
        worker
        for worker in workers
        if _fte_worker_has_memory_capacity(
            worker,
            memory_requirement_bytes=memory_requirement_bytes,
            execution_class=execution_class,
        )
    ]
    if not workers_with_capacity:
        return None
    workers = workers_with_capacity
    return min(
        workers,
        key=lambda handle: _fte_worker_selection_key(
            handle,
            memory_requirement_bytes=memory_requirement_bytes,
            execution_class=execution_class,
            node_requirements=node_requirements,
        ),
    )
