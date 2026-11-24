# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any

from vane.execution.cluster_resource_policy import NodeCapacity
from vane.execution.resources import ResourceVector
from vane.runners.ray.worker_memory import build_ray_node_memory_layout


def read_ray_node_capacities(
    ray_module: Any,
    *,
    object_store_fraction: float = 0.5,
    heap_reserve_bytes_per_node: int = 0,
) -> tuple[NodeCapacity, ...]:
    """Read live Ray capacity without inventing host-resource fallbacks."""

    fraction = float(object_store_fraction)
    if not math.isfinite(fraction) or fraction <= 0 or fraction > 1:
        raise ValueError("object_store_fraction must be in (0, 1]")
    heap_reserve = int(heap_reserve_bytes_per_node)
    if heap_reserve < 0:
        raise ValueError("heap_reserve_bytes_per_node must be >= 0")

    try:
        raw_nodes = ray_module.nodes()
    except Exception as exc:
        raise RuntimeError(f"failed to read Ray node capacity: {exc}") from exc

    capacities: list[NodeCapacity] = []
    for raw_node in raw_nodes:
        if not bool(raw_node.get("Alive", True)):
            continue
        resources = dict(raw_node.get("Resources") or {})
        cpu = max(0.0, float(resources.get("CPU", 0) or 0))
        gpu = max(0.0, float(resources.get("GPU", 0) or 0))
        if cpu <= 0 and gpu <= 0:
            continue
        node_id = str(raw_node.get("NodeID") or raw_node.get("NodeManagerAddress") or "").strip()
        if not node_id:
            raise ValueError("alive Ray node with schedulable resources is missing NodeID")
        ray_heap = max(0, int(float(resources.get("memory", 0) or 0)))
        ray_store = max(0, int(float(resources.get("object_store_memory", 0) or 0)))
        memory_layout = build_ray_node_memory_layout(ray_heap)
        labels = [str(key) for key, value in resources.items() if str(key).startswith("node:") and float(value) > 0]
        labels.extend(f"{key}={value}" for key, value in sorted(dict(raw_node.get("Labels") or {}).items()))
        capacities.append(
            NodeCapacity(
                node_id=node_id,
                resources=ResourceVector(
                    cpu=cpu,
                    gpu=gpu,
                    heap_bytes=max(0, memory_layout.task_heap_capacity_bytes - heap_reserve),
                    object_store_bytes=math.floor(ray_store * fraction),
                ),
                labels=tuple(labels),
            )
        )
    return tuple(sorted(capacities, key=lambda item: item.node_id))
