# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import heapq
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from vane.runners.fte.fte_split_assigner import (
    AssignmentResult,
    NodeRequirements,
    PartitionInfo,
    PartitionUpdate,
    SplitAssigner,
    _splits_from_inputs,
)
from vane.runners.fte.fte_types import FteSplit


@dataclass
class _ScanTask:
    splits: list[FteSplit] = field(default_factory=list)
    size_bytes: int = 0


class ScanBatchSplitAssigner(SplitAssigner):
    """Plan unordered, single-source scans with LPT within a bounded batch.

    The submitter calls flush() at its batch boundary, independently of source
    exhaustion. Emitted tasks are immutable and sealed; worker admission and
    retries remain the responsibility of FteFragmentExecution.
    """

    def __init__(
        self,
        source_node_id: str,
        *,
        worker_slots: int = 1,
        tasks_per_slot: int = 4,
        min_task_size_bytes: int = 64 * 1024 * 1024,
        max_task_size_bytes: int = 256 * 1024 * 1024,
        standard_split_size_bytes: int = 64 * 1024 * 1024,
        max_task_split_count: int = 2048,
        max_batch_split_count: int = 4096,
    ) -> None:
        for name, value in (
            ("worker_slots", worker_slots),
            ("tasks_per_slot", tasks_per_slot),
            ("min_task_size_bytes", min_task_size_bytes),
            ("standard_split_size_bytes", standard_split_size_bytes),
            ("max_task_split_count", max_task_split_count),
            ("max_batch_split_count", max_batch_split_count),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if max_task_size_bytes < min_task_size_bytes:
            raise ValueError("max_task_size_bytes must be >= min_task_size_bytes")
        self._source_node_id = str(source_node_id)
        self._worker_slots = worker_slots
        self._tasks_per_slot = tasks_per_slot
        self._min_task_size_bytes = min_task_size_bytes
        self._max_task_size_bytes = max_task_size_bytes
        self._standard_split_size_bytes = standard_split_size_bytes
        self._max_task_split_count = max_task_split_count
        self._max_batch_split_count = max_batch_split_count
        self._pending: list[FteSplit] = []
        self._next_sequence = 0
        self._next_partition_id = 0
        self._finished = False

    def assign(
        self,
        source_node_id: str,
        inputs: list[Mapping[str, Any]],
        no_more_inputs: bool = False,
    ) -> AssignmentResult:
        if self._finished:
            raise RuntimeError("cannot assign splits after finish")
        if str(source_node_id) != self._source_node_id:
            raise ValueError("batch scan assigner requires a single scan source")
        splits, next_sequence = _splits_from_inputs(source_node_id, inputs, self._next_sequence)
        for split in splits:
            if split.source_node_id != self._source_node_id or split.kind != "scan_split":
                raise ValueError("batch scan assigner requires splits from its scan source")
            if not split.remotely_accessible and not split.addresses:
                raise ValueError("non-remotely-accessible split must have an address")
        self._next_sequence = next_sequence
        result = AssignmentResult()
        for split in splits:
            self._pending.append(split)
            if len(self._pending) == self._max_batch_split_count:
                self._flush_into(result)
        if no_more_inputs:
            self._flush_into(result)
            if self._next_partition_id == 0:
                self._emit_task(result, NodeRequirements(), [])
            self._finished = True
            result.no_more_partitions = True
        return result

    def flush(self) -> AssignmentResult:
        result = AssignmentResult()
        self._flush_into(result)
        return result

    def finish(self) -> AssignmentResult:
        if self._finished:
            return AssignmentResult()
        return self.assign(self._source_node_id, [], no_more_inputs=True)

    def _cost(self, split: FteSplit) -> int:
        return self._standard_split_size_bytes if split.size_bytes is None else split.size_bytes

    def _flush_into(self, result: AssignmentResult) -> None:
        if not self._pending:
            return
        # A split is an indivisible reader-produced range (or whole file).
        # Stable sorting preserves input order for equal cost estimates.
        splits = sorted(self._pending, key=self._cost, reverse=True)
        # Oversize splits already require standalone tasks. Their bytes must
        # not inflate the target size or the number of bins for regular splits.
        total_bytes = sum(self._cost(split) for split in splits if self._cost(split) <= self._max_task_size_bytes)
        target_count = min(
            self._worker_slots * self._tasks_per_slot,
            max(1, total_bytes // self._min_task_size_bytes),
        )
        target_bytes = max(
            self._min_task_size_bytes,
            min(self._max_task_size_bytes, (total_bytes + target_count - 1) // target_count),
        )
        groups: dict[NodeRequirements, list[FteSplit]] = {}
        planned_tasks: list[tuple[NodeRequirements, _ScanTask]] = []
        host_load: dict[str, int] = {}
        for split in splits:
            host = None
            if split.addresses:
                host = min(split.addresses, key=lambda address: (host_load.get(address, 0), address))
                host_load[host] = host_load.get(host, 0) + self._cost(split)
            requirements = NodeRequirements(split.catalog, host, split.remotely_accessible)
            if self._cost(split) > self._max_task_size_bytes:
                planned_tasks.append((requirements, _ScanTask([split], self._cost(split))))
            else:
                groups.setdefault(requirements, []).append(split)

        for requirements, group in groups.items():
            group_bytes = sum(self._cost(split) for split in group)
            task_count = min(
                len(group),
                max(
                    1,
                    (group_bytes + target_bytes - 1) // target_bytes,
                    (len(group) + self._max_task_split_count - 1) // self._max_task_split_count,
                ),
            )
            tasks = [_ScanTask() for _ in range(task_count)]
            available = [(0, index) for index in range(task_count)]
            heapq.heapify(available)
            for split in group:
                cost = self._cost(split)
                # Count-full tasks have already left the heap. If the lightest
                # task cannot fit this split, no heavier task can fit it either.
                if available and available[0][0] + cost <= self._max_task_size_bytes:
                    _, index = heapq.heappop(available)
                else:
                    index = len(tasks)
                    tasks.append(_ScanTask())
                task = tasks[index]
                task.splits.append(split)
                task.size_bytes += cost
                if len(task.splits) < self._max_task_split_count and task.size_bytes < self._max_task_size_bytes:
                    heapq.heappush(available, (task.size_bytes, index))
            for task in tasks:
                if task.splits:
                    planned_tasks.append((requirements, task))
        # Admission consumes partition IDs in order. Rank completed tasks across
        # all locality groups so the longest tasks can start first.
        for requirements, task in sorted(planned_tasks, key=lambda item: item[1].size_bytes, reverse=True):
            self._emit_task(result, requirements, task.splits)
        self._pending.clear()

    def _emit_task(self, result: AssignmentResult, requirements: NodeRequirements, splits: list[FteSplit]) -> None:
        partition_id = self._next_partition_id
        self._next_partition_id += 1
        result.partitions_added.append(PartitionInfo(partition_id, requirements))
        result.partition_updates.append(
            PartitionUpdate(
                partition_id,
                self._source_node_id,
                splits,
                no_more_splits=True,
                ready_for_scheduling=bool(splits),
            )
        )
        result.sealed_partitions.append(partition_id)
