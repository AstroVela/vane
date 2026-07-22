# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from vane.runners.fte import (
    ArbitrarySplitAssigner,
    HashSplitAssigner,
    HashTaskPartition,
)

if TYPE_CHECKING:
    from vane.runners.fte import SplitAssigner
    from vane.runners.ray.fragment_registry import _FteFragmentState


def _dynamic_scan_max_splits_per_partition() -> int | None:
    raw = os.getenv("VANE_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION")
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("VANE_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION must be a positive integer") from exc
    if value <= 0:
        raise ValueError("VANE_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION must be a positive integer")
    return value


def make_fte_assigner(fragment_state: _FteFragmentState) -> SplitAssigner:
    if fragment_state.dynamic_exchange_source_node_ids:
        replicated_exchange_sources = set(fragment_state.replicated_exchange_source_node_ids)
        dynamic_exchange_sources = set(fragment_state.dynamic_exchange_source_node_ids)
        partitioned_exchange_sources = dynamic_exchange_sources - replicated_exchange_sources
        if not partitioned_exchange_sources:
            return ArbitrarySplitAssigner(
                partitioned_sources=set(fragment_state.source_node_ids) - replicated_exchange_sources,
                replicated_sources=replicated_exchange_sources,
                max_splits_per_partition=_dynamic_scan_max_splits_per_partition(),
            )
        source_partition_ids = sorted(fragment_state.exchange_source_partition_ids)
        source_partition_count = max(
            fragment_state.exchange_source_partition_count,
            (max(source_partition_ids) + 1) if source_partition_ids else 1,
        )
        mapping = make_hash_task_partition_mapping(
            source_partition_count,
            fragment_state.exchange_source_task_count,
        )
        return HashSplitAssigner(
            source_partition_count=source_partition_count,
            partitioned_sources=partitioned_exchange_sources,
            replicated_sources=set(fragment_state.dynamic_scan_source_node_ids) | replicated_exchange_sources,
            source_partition_to_task_partition=mapping,
        )
    return ArbitrarySplitAssigner(
        partitioned_sources=set(fragment_state.source_node_ids),
        max_splits_per_partition=_dynamic_scan_max_splits_per_partition(),
    )


def make_hash_task_partition_mapping(
    source_partition_count: int,
    source_task_count: int,
) -> dict[int, HashTaskPartition]:
    source_partition_count = int(source_partition_count)
    source_task_count = int(source_task_count)
    if source_partition_count <= 0:
        raise ValueError("source_partition_count must be positive")
    if source_task_count <= 0:
        source_task_count = source_partition_count
    source_task_count = min(source_task_count, source_partition_count)
    mapping: dict[int, HashTaskPartition] = {}
    for task_idx in range(source_task_count):
        part_start = task_idx * source_partition_count // source_task_count
        part_end = (task_idx + 1) * source_partition_count // source_task_count
        task_partition = HashTaskPartition()
        for source_partition_id in range(part_start, part_end):
            mapping[source_partition_id] = task_partition
    for source_partition_id in range(source_partition_count):
        mapping.setdefault(source_partition_id, HashTaskPartition())
    return mapping
