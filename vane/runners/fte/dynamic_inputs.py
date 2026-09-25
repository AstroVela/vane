# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from vane.runners.fte.fte_types import FteSplit

SplitExchangeSourceTaskByPartition = Callable[[Any], tuple[list[tuple[int, Any]], int, int, bool]]
SplitScanSplitBatch = Callable[[Any], list[tuple[str, Any, int | None]]]


@dataclass(frozen=True)
class FteDynamicInputPreparation:
    splits: list[FteSplit]
    dynamic_scan_sources: set[str]
    dynamic_exchange_sources: set[str]
    replicated_exchange_sources: set[str]
    exchange_source_partition_ids: set[int]
    exchange_source_partition_count: int
    exchange_source_task_count: int
    exchange_source_metadata_by_source: dict[str, tuple[set[int], int, int]]


def strip_fte_dynamic_context(
    context: Mapping[str, Any] | None,
    dynamic_scan_sources: set[str],
    dynamic_exchange_sources: set[str],
) -> dict[str, Any]:
    sanitized = dict(context or {})

    for source_node_id in dynamic_scan_sources:
        sanitized.pop(f"scan_split_batch:{source_node_id}", None)
    for source_node_id in dynamic_exchange_sources:
        sanitized.pop(f"exchange_source_task:{source_node_id}", None)

    def update_node_list(key: str, removed_sources: set[str]) -> None:
        raw = sanitized.get(key)
        if raw in (None, ""):
            sanitized.pop(key, None)
            return
        nodes = [node.strip() for node in str(raw).split(",") if node.strip() and node.strip() not in removed_sources]
        if nodes:
            sanitized[key] = ",".join(nodes)
        else:
            sanitized.pop(key, None)

    update_node_list("scan_split_batch_nodes", dynamic_scan_sources)
    update_node_list("exchange_source_task_nodes", dynamic_exchange_sources)
    return sanitized


def exchange_source_task_is_replicated(value: Mapping[str, Any]) -> bool:
    distribution = str(value.get("distribution") or value.get("source_distribution") or "").strip().lower()
    return bool(value.get("replicated") or value.get("is_replicated") or distribution == "replicated")


def split_exchange_source_task_by_partition(value: Any) -> tuple[list[tuple[int, Any]], int, int, bool]:
    if isinstance(value, Mapping):
        indices = tuple(int(partition_id) for partition_id in (value.get("partition_indices") or ()))
        partition_count = int(value.get("source_partition_count") or value.get("partition_count") or 0)
        if partition_count <= 0 and indices:
            partition_count = max(indices) + 1
        task_count = int(value.get("source_task_count") or value.get("task_count") or partition_count)
        replicated = exchange_source_task_is_replicated(value)
        items: list[tuple[int, Any]] = []
        for partition_id in indices:
            single = dict(value)
            single["partition_indices"] = [partition_id]
            items.append((partition_id, single))
        return items, partition_count, task_count, replicated

    import vane

    raw_items = vane.ray_cxx.split_exchange_source_task_by_partition(value)
    native_items: list[tuple[int, Any]] = []
    partition_count = 0
    source_task_count = 0
    replicated = False
    for raw_item in raw_items:
        partition_id, split_value, raw_partition_count = raw_item[:3]
        raw_source_task_count = raw_item[3] if len(raw_item) >= 4 else raw_partition_count
        raw_replicated = raw_item[4] if len(raw_item) >= 5 else False
        partition_count = max(partition_count, int(raw_partition_count))
        source_task_count = max(source_task_count, int(raw_source_task_count))
        replicated = replicated or bool(raw_replicated)
        partition_id = int(partition_id)
        native_items.append((partition_id, split_value))
        partition_count = max(partition_count, partition_id + 1)
    if source_task_count <= 0:
        source_task_count = partition_count
    return native_items, partition_count, source_task_count, replicated


def split_scan_split_batch(value: Any) -> list[tuple[str, Any, int | None]]:
    """Return independently schedulable singleton batches from one transport batch."""
    if isinstance(value, Mapping):
        raw_splits = value.get("splits")
        if not isinstance(raw_splits, (list, tuple)) or not raw_splits:
            raise ValueError("scan split batch mapping must contain a non-empty splits list")
        result: list[tuple[str, Any, int | None]] = []
        seen_ids: set[str] = set()
        for raw_split in raw_splits:
            if not isinstance(raw_split, Mapping):
                raise TypeError("scan split batch entries must be mappings")
            raw_split_id = raw_split.get("split_id")
            split_id = "" if raw_split_id is None else str(raw_split_id)
            if not split_id:
                raise ValueError("scan split is missing split_id")
            if split_id in seen_ids:
                raise ValueError(f"duplicate scan split_id in batch: {split_id}")
            seen_ids.add(split_id)
            estimated_bytes = raw_split.get("estimated_bytes")
            if estimated_bytes is not None:
                estimated_bytes = int(estimated_bytes)
                if estimated_bytes < 0:
                    raise ValueError("scan split estimated_bytes must be non-negative")
            singleton = dict(value)
            singleton["splits"] = [dict(raw_split)]
            result.append((split_id, singleton, estimated_bytes))
        return result

    import vane

    if isinstance(value, (bytearray, memoryview)):
        value = bytes(value)
    return [
        (str(split_id), singleton_batch, None if estimated_bytes is None else int(estimated_bytes))
        for split_id, singleton_batch, estimated_bytes in vane.ray_cxx.split_scan_split_batch(value)
    ]


def prepare_fte_dynamic_inputs(
    *,
    context: Mapping[str, Any],
    query_id: str,
    fragment_id: str,
    next_split_sequence: Callable[[str, str, str], int],
    split_scan_split_batch_fn: SplitScanSplitBatch | None = None,
    split_exchange_source_task_by_partition_fn: SplitExchangeSourceTaskByPartition | None = None,
) -> FteDynamicInputPreparation:
    splits: list[FteSplit] = []
    dynamic_scan_sources: set[str] = set()
    dynamic_exchange_sources: set[str] = set()
    replicated_exchange_sources: set[str] = set()
    exchange_source_partition_ids: set[int] = set()
    exchange_source_partition_count = 0
    exchange_source_task_count = 0
    exchange_source_metadata_by_source: dict[str, tuple[set[int], int, int]] = {}

    for key, value in context.items():
        if key.startswith("scan_split_batch:"):
            source_node_id = key.split(":", 1)[1]
            if not source_node_id:
                continue
            dynamic_scan_sources.add(source_node_id)
            scan_split_fn = split_scan_split_batch_fn or split_scan_split_batch
            for split_id, singleton_batch, estimated_bytes in scan_split_fn(value):
                splits.append(
                    FteSplit(
                        source_node_id=source_node_id,
                        sequence_id=next_split_sequence(query_id, fragment_id, source_node_id),
                        kind="scan_split",
                        data=singleton_batch,
                        split_id=split_id,
                        size_bytes=estimated_bytes,
                    )
                )
            continue
        if not key.startswith("exchange_source_task:"):
            continue
        source_node_id = key.split(":", 1)[1]
        if not source_node_id:
            continue
        dynamic_exchange_sources.add(source_node_id)
        exchange_split_fn = split_exchange_source_task_by_partition_fn or split_exchange_source_task_by_partition
        split_items, source_partition_count, source_task_count, replicated = exchange_split_fn(value)
        if replicated:
            replicated_exchange_sources.add(source_node_id)
        exchange_source_partition_count = max(exchange_source_partition_count, int(source_partition_count))
        exchange_source_task_count = max(exchange_source_task_count, int(source_task_count))
        source_partition_ids: set[int] = set()
        for source_partition_id, split_value in split_items:
            exchange_source_partition_ids.add(source_partition_id)
            source_partition_ids.add(source_partition_id)
            splits.append(
                FteSplit(
                    source_node_id=source_node_id,
                    sequence_id=next_split_sequence(query_id, fragment_id, source_node_id),
                    kind="exchange_source_task",
                    data=split_value,
                    source_partition_id=source_partition_id,
                )
            )
        existing = exchange_source_metadata_by_source.get(source_node_id)
        if existing is None:
            exchange_source_metadata_by_source[source_node_id] = (
                source_partition_ids,
                int(source_partition_count),
                int(source_task_count),
            )
        else:
            existing_ids, existing_count, existing_task_count = existing
            existing_ids.update(source_partition_ids)
            exchange_source_metadata_by_source[source_node_id] = (
                existing_ids,
                max(int(existing_count), int(source_partition_count)),
                max(int(existing_task_count), int(source_task_count)),
            )

    if exchange_source_partition_count <= 0 and exchange_source_partition_ids:
        exchange_source_partition_count = max(exchange_source_partition_ids) + 1
    if exchange_source_task_count <= 0:
        exchange_source_task_count = exchange_source_partition_count
    return FteDynamicInputPreparation(
        splits=splits,
        dynamic_scan_sources=dynamic_scan_sources,
        dynamic_exchange_sources=dynamic_exchange_sources,
        replicated_exchange_sources=replicated_exchange_sources,
        exchange_source_partition_ids=exchange_source_partition_ids,
        exchange_source_partition_count=exchange_source_partition_count,
        exchange_source_task_count=exchange_source_task_count,
        exchange_source_metadata_by_source=exchange_source_metadata_by_source,
    )


def splits_from_pending_task(
    item: Mapping[str, Any],
    *,
    next_split_sequence: Callable[[str, str, str], int],
    split_scan_split_batch_fn: SplitScanSplitBatch | None = None,
    split_exchange_source_task_by_partition_fn: SplitExchangeSourceTaskByPartition | None = None,
) -> tuple[
    list[FteSplit],
    set[str],
    set[str],
    set[str],
    set[int],
    int,
    int,
    dict[str, tuple[set[int], int, int]],
]:
    prepared = prepare_fte_dynamic_inputs(
        context=item["context"],
        query_id=str(item["query_id"]),
        fragment_id=str(item["fragment_id"]),
        next_split_sequence=next_split_sequence,
        split_scan_split_batch_fn=split_scan_split_batch_fn,
        split_exchange_source_task_by_partition_fn=split_exchange_source_task_by_partition_fn,
    )
    return (
        prepared.splits,
        prepared.dynamic_scan_sources,
        prepared.dynamic_exchange_sources,
        prepared.replicated_exchange_sources,
        prepared.exchange_source_partition_ids,
        prepared.exchange_source_partition_count,
        prepared.exchange_source_task_count,
        prepared.exchange_source_metadata_by_source,
    )
