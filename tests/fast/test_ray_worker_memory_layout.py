# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

from vane.runners.ray.worker_memory import build_ray_node_memory_layout


def test_ray_node_memory_layout_has_one_exact_owner_per_byte():
    layout = build_ray_node_memory_layout(1_000)

    assert layout.worker_duckdb_memory_bytes == 250
    assert layout.driver_duckdb_reserve_bytes == 50
    assert layout.runtime_reserve_bytes == 100
    assert layout.task_heap_capacity_bytes == 600
    assert (
        layout.worker_duckdb_memory_bytes
        + layout.driver_duckdb_reserve_bytes
        + layout.runtime_reserve_bytes
        + layout.task_heap_capacity_bytes
        == layout.node_memory_bytes
    )


@pytest.mark.parametrize("node_memory_bytes", [0, -1])
def test_ray_node_memory_layout_requires_positive_logical_memory(node_memory_bytes):
    with pytest.raises(ValueError, match="node_memory_bytes must be positive"):
        build_ray_node_memory_layout(node_memory_bytes)
