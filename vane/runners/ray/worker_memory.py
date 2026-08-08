# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RayNodeMemoryLayout:
    """One immutable ownership split for a Ray node's logical heap memory."""

    node_memory_bytes: int
    worker_duckdb_memory_bytes: int
    driver_duckdb_reserve_bytes: int
    runtime_reserve_bytes: int
    task_heap_capacity_bytes: int


def build_ray_node_memory_layout(node_memory_bytes: int) -> RayNodeMemoryLayout:
    """Partition Ray logical memory into infrastructure and task-owned domains.

    The persistent FTE worker has one shared DuckDB DatabaseInstance per node,
    so its memory limit is node-scoped rather than query-scoped.  The remaining
    task heap is the only heap capacity exposed to query/task leases.
    """

    node_memory = int(node_memory_bytes)
    if node_memory <= 0:
        raise ValueError("node_memory_bytes must be positive")

    worker_duckdb = node_memory // 4
    driver_duckdb = node_memory // 20
    runtime_reserve = node_memory // 10
    task_heap = node_memory - worker_duckdb - driver_duckdb - runtime_reserve
    if min(worker_duckdb, driver_duckdb, runtime_reserve, task_heap) <= 0:
        raise ValueError("node_memory_bytes is too small for the production memory layout")
    return RayNodeMemoryLayout(
        node_memory_bytes=node_memory,
        worker_duckdb_memory_bytes=worker_duckdb,
        driver_duckdb_reserve_bytes=driver_duckdb,
        runtime_reserve_bytes=runtime_reserve,
        task_heap_capacity_bytes=task_heap,
    )


__all__ = ["RayNodeMemoryLayout", "build_ray_node_memory_layout"]
