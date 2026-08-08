# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vane.runners.fte import FteSplit
from vane.runners.ray.fragment_registry import (
    _FTE_REGISTRY_LOCK,
)
from vane.runners.ray.fte_fragment_scheduler import (
    _apply_exchange_selector_snapshot,
    _exchange_source_split_key,
    _normalize_exchange_selector_snapshot,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vane.runners.ray.fragment_registry import _FteFragmentState


def mark_exchange_source_partitions_seen(
    fragment_state: _FteFragmentState,
    exchange_source_metadata_by_source: Mapping[str, tuple[set[int], int, int]],
) -> tuple[dict[str, set[int]], set[str]]:
    partition_ids_by_source: dict[str, set[int]] = {}
    with _FTE_REGISTRY_LOCK:
        for raw_source_node_id, raw_metadata in exchange_source_metadata_by_source.items():
            source_node_id = str(raw_source_node_id)
            source_partition_ids, source_partition_count, source_task_count = raw_metadata
            seen = fragment_state.exchange_source_partition_ids_by_source.setdefault(
                source_node_id,
                set(),
            )
            normalized_ids = {int(partition_id) for partition_id in source_partition_ids}
            if source_node_id in fragment_state.exhausted_source_node_ids:
                if normalized_ids:
                    raise RuntimeError(f"exchange source {source_node_id} received splits after no_more_splits")
                continue

            if normalized_ids:
                seen.update(normalized_ids)
                partition_ids_by_source[source_node_id] = normalized_ids
            partition_count = max(
                int(fragment_state.exchange_source_partition_count_by_source.get(source_node_id, 0)),
                int(source_partition_count),
            )
            task_count = max(
                int(fragment_state.exchange_source_task_count_by_source.get(source_node_id, 0)),
                int(source_task_count),
            )
            fragment_state.exchange_source_partition_count_by_source[source_node_id] = partition_count
            fragment_state.exchange_source_task_count_by_source[source_node_id] = task_count
    return partition_ids_by_source, set()


def apply_exchange_selector_update(
    fragment_state: _FteFragmentState,
    event: Any,
) -> tuple[Any, list[FteSplit]] | None:
    source_node_id = str(event.source_node_id)
    with _FTE_REGISTRY_LOCK:
        selector_snapshot = _normalize_exchange_selector_snapshot(
            fragment_state,
            source_node_id,
            event.selector,
        )
        if selector_snapshot.source_node_id != source_node_id:
            raise ValueError(
                "exchange selector source_node_id does not match event source_node_id: "
                f"{selector_snapshot.source_node_id!r} != {source_node_id!r}"
            )
        raw_splits = [
            FteSplit.from_dict(source_node_id, entry["split"])
            for entry in selector_snapshot.selected.values()
            if entry.get("split") is not None
        ]
        selector_changed, new_selector_partitions = _apply_exchange_selector_snapshot(
            fragment_state,
            selector_snapshot,
        )
        if not selector_changed:
            return None
        if source_node_id in fragment_state.exhausted_source_node_ids and raw_splits:
            raise RuntimeError(f"exchange source {source_node_id} received splits after no_more_splits")
        seen_split_keys = fragment_state.exchange_source_split_keys_by_source.setdefault(
            source_node_id,
            set(),
        )
        splits: list[FteSplit] = []
        for split in raw_splits:
            if split.source_partition_id not in new_selector_partitions:
                continue
            split_key = _exchange_source_split_key(split)
            if split_key in seen_split_keys:
                continue
            seen_split_keys.add(split_key)
            splits.append(split)
    return selector_snapshot, splits
