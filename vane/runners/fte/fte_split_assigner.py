# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from vane.runners.fte.fte_types import FteSplit


@dataclass(frozen=True)
class PartitionInfo:
    partition_id: int
    node_requirements: NodeRequirements | None = None


@dataclass(frozen=True)
class NodeRequirements:
    catalog: str | None = None
    host: str | None = None
    remotely_accessible: bool = True


@dataclass(frozen=True)
class PartitionUpdate:
    partition_id: int
    source_node_id: str
    splits: list[FteSplit] = field(default_factory=list)
    no_more_splits: bool = False
    ready_for_scheduling: bool = False

    def __post_init__(self) -> None:
        if self.ready_for_scheduling and not self.splits:
            raise ValueError("partition update with empty splits marked as ready for scheduling")


@dataclass
class AssignmentResult:
    partitions_added: list[PartitionInfo] = field(default_factory=list)
    partition_updates: list[PartitionUpdate] = field(default_factory=list)
    sealed_partitions: list[int] = field(default_factory=list)
    no_more_partitions: bool = False


class SplitAssigner:
    def assign(
        self,
        source_node_id: str,
        inputs: list[Mapping[str, Any]],
        no_more_inputs: bool = False,
    ) -> AssignmentResult:
        raise NotImplementedError

    def finish(self) -> AssignmentResult:
        return AssignmentResult(no_more_partitions=True)


def _normalize_sources(sources: set[str] | list[str] | tuple[str, ...] | None) -> set[str]:
    return {str(source) for source in (sources or [])}


def _splits_from_inputs(
    source_node_id: str, inputs: list[Mapping[str, Any]], next_sequence: int
) -> tuple[list[FteSplit], int]:
    splits: list[FteSplit] = []
    for item in inputs:
        payload = dict(item)
        payload.setdefault("sequence_id", next_sequence)
        split = FteSplit.from_dict(source_node_id, payload)
        splits.append(split)
        next_sequence = max(next_sequence, split.sequence_id + 1)
    return splits, next_sequence


def _split_size_bytes(split: FteSplit, standard_split_size_bytes: int) -> int:
    if split.size_bytes is not None:
        return split.size_bytes
    return standard_split_size_bytes


def _split_source_partition_id(split: FteSplit) -> int:
    return split.source_partition_id


def _append_update(
    result: AssignmentResult,
    partition_id: int,
    source_node_id: str,
    splits: list[FteSplit] | None = None,
    no_more_splits: bool = False,
    ready_for_scheduling: bool = False,
    emit_empty: bool = False,
) -> None:
    if not emit_empty and not splits and not no_more_splits:
        return
    result.partition_updates.append(
        PartitionUpdate(
            partition_id=partition_id,
            source_node_id=source_node_id,
            splits=list(splits or []),
            no_more_splits=no_more_splits,
            ready_for_scheduling=ready_for_scheduling,
        )
    )


def _append_seal(result: AssignmentResult, partition_id: int) -> None:
    if partition_id not in result.sealed_partitions:
        result.sealed_partitions.append(partition_id)


class SingleSplitAssigner(SplitAssigner):
    def __init__(self, all_sources: set[str] | list[str] | tuple[str, ...] | None = None) -> None:
        self._partition_added = False
        self._source_sequences: dict[str, int] = {}
        self._all_sources = _normalize_sources(all_sources)
        self._completed_sources: set[str] = set()

    def assign(
        self,
        source_node_id: str,
        inputs: list[Mapping[str, Any]],
        no_more_inputs: bool = False,
    ) -> AssignmentResult:
        source_node_id = str(source_node_id)
        result = AssignmentResult()
        if not self._partition_added:
            self._partition_added = True
            result.partitions_added.append(PartitionInfo(0))
            result.no_more_partitions = True

        next_sequence = self._source_sequences.get(source_node_id, 0)
        splits, next_sequence = _splits_from_inputs(source_node_id, inputs, next_sequence)
        self._source_sequences[source_node_id] = next_sequence
        if not self._all_sources:
            self._all_sources.add(source_node_id)
        if splits and source_node_id in self._completed_sources:
            raise RuntimeError(f"source is finished: {source_node_id}")

        _append_update(
            result,
            0,
            source_node_id,
            splits,
            ready_for_scheduling=bool(splits),
        )
        if no_more_inputs:
            _append_update(result, 0, source_node_id, no_more_splits=True)
            self._completed_sources.add(source_node_id)
        if self._all_sources and self._completed_sources.issuperset(self._all_sources):
            result.sealed_partitions.append(0)
        return result

    def finish(self) -> AssignmentResult:
        result = AssignmentResult()
        if not self._partition_added:
            self._partition_added = True
            result.partitions_added.append(PartitionInfo(0))
            result.sealed_partitions.append(0)
            result.no_more_partitions = True
        return result


class ArbitrarySplitAssigner(SplitAssigner):
    @dataclass
    class _PartitionAssignment:
        partition_id: int
        assigned_data_size_bytes: int = 0
        assigned_split_count: int = 0
        full: bool = False

        def assign_split(self, size_bytes: int) -> None:
            self.assigned_data_size_bytes += size_bytes
            self.assigned_split_count += 1

    def __init__(
        self,
        max_splits_per_partition: int | None = None,
        *,
        catalog_requirement: str | None = None,
        partitioned_sources: set[str] | list[str] | tuple[str, ...] | None = None,
        replicated_sources: set[str] | list[str] | tuple[str, ...] | None = None,
        min_target_partition_size_bytes: int | None = None,
        max_target_partition_size_bytes: int | None = None,
        adaptive_growth_period: int = 64,
        adaptive_growth_factor: float = 1.26,
        standard_split_size_bytes: int = 64 * 1024 * 1024,
        max_task_split_count: int | None = None,
    ) -> None:
        if max_splits_per_partition is not None and max_splits_per_partition <= 0:
            raise ValueError("max_splits_per_partition must be positive")
        if max_task_split_count is None:
            max_task_split_count = max_splits_per_partition if max_splits_per_partition is not None else 2048
        if max_task_split_count <= 0:
            raise ValueError("max_task_split_count must be positive")
        if adaptive_growth_period <= 0:
            raise ValueError("adaptive_growth_period must be positive")
        if adaptive_growth_factor < 1:
            raise ValueError("adaptive_growth_factor must be >= 1")
        min_target = min_target_partition_size_bytes
        if min_target is None:
            min_target = max_task_split_count * standard_split_size_bytes
        max_target = max_target_partition_size_bytes if max_target_partition_size_bytes is not None else min_target
        if min_target <= 0:
            raise ValueError("min_target_partition_size_bytes must be positive")
        if max_target < min_target:
            raise ValueError("max_target_partition_size_bytes must be >= min_target_partition_size_bytes")
        self._next_partition_id = 0
        self._source_sequences: dict[str, int] = {}
        self._catalog_requirement = None if catalog_requirement is None else str(catalog_requirement)
        self._partitioned_sources = _normalize_sources(partitioned_sources)
        self._replicated_sources = _normalize_sources(replicated_sources)
        self._all_sources = set(self._partitioned_sources) | set(self._replicated_sources)
        self._completed_sources: set[str] = set()
        self._replicated_splits: dict[str, list[FteSplit]] = {}
        self._no_more_replicated_splits = False
        self._all_assignments: list[ArbitrarySplitAssigner._PartitionAssignment] = []
        self._open_assignments: dict[NodeRequirements, ArbitrarySplitAssigner._PartitionAssignment] = {}
        self._adaptive_growth_period = int(adaptive_growth_period)
        self._adaptive_growth_factor = float(adaptive_growth_factor)
        self._min_target_partition_size_bytes = int(min_target)
        self._max_target_partition_size_bytes = int(max_target)
        self._target_partition_size_bytes = int(min_target)
        self._rounded_target_partition_size_bytes = int(min_target)
        self._adaptive_counter = 0
        self._standard_split_size_bytes = int(standard_split_size_bytes)
        self._max_task_split_count = int(max_task_split_count)
        self._finished = False

    def assign(
        self,
        source_node_id: str,
        inputs: list[Mapping[str, Any]],
        no_more_inputs: bool = False,
    ) -> AssignmentResult:
        if self._finished:
            raise RuntimeError("cannot assign splits after finish")
        source_node_id = str(source_node_id)
        next_sequence = self._source_sequences.get(source_node_id, 0)
        splits, next_sequence = _splits_from_inputs(source_node_id, inputs, next_sequence)
        self._source_sequences[source_node_id] = next_sequence
        if not self._all_sources:
            self._partitioned_sources.add(source_node_id)
            self._all_sources.add(source_node_id)
        if source_node_id in self._replicated_sources:
            return self._assign_replicated_splits(source_node_id, splits, no_more_inputs)
        return self._assign_partitioned_splits(source_node_id, splits, no_more_inputs)

    def finish(self) -> AssignmentResult:
        if not self._all_assignments:
            raise RuntimeError("allAssignments is not expected to be empty")
        return AssignmentResult()

    def _assign_replicated_splits(
        self,
        source_node_id: str,
        splits: list[FteSplit],
        no_more_inputs: bool,
    ) -> AssignmentResult:
        result = AssignmentResult()
        self._replicated_splits.setdefault(source_node_id, []).extend(splits)
        for assignment in self._all_assignments:
            _append_update(
                result,
                assignment.partition_id,
                source_node_id,
                splits,
                no_more_splits=no_more_inputs,
                emit_empty=True,
            )
        if no_more_inputs:
            self._completed_sources.add(source_node_id)
            if self._completed_sources.issuperset(self._replicated_sources):
                self._no_more_replicated_splits = True
        if self._no_more_replicated_splits:
            for assignment in self._all_assignments:
                if assignment.full:
                    _append_seal(result, assignment.partition_id)
        return self._merge_assignment_results(result, self._finish_if_all_sources_completed())

    def _assign_partitioned_splits(
        self,
        source_node_id: str,
        splits: list[FteSplit],
        no_more_inputs: bool,
    ) -> AssignmentResult:
        result = AssignmentResult()
        for split in splits:
            node_requirements = self._node_requirements(split)
            assignment = self._open_assignments.get(node_requirements)
            split_size = _split_size_bytes(split, self._standard_split_size_bytes)
            if assignment is not None and (
                assignment.assigned_data_size_bytes + split_size > self._rounded_target_partition_size_bytes
                or assignment.assigned_split_count + 1 > self._max_task_split_count
            ):
                assignment.full = True
                for partitioned_source_id in self._partitioned_sources:
                    _append_update(
                        result,
                        assignment.partition_id,
                        partitioned_source_id,
                        no_more_splits=True,
                    )
                if self._completed_sources.issuperset(self._replicated_sources):
                    _append_seal(result, assignment.partition_id)
                self._open_assignments.pop(node_requirements, None)
                assignment = None
                self._grow_target_if_needed()

            if assignment is None:
                assignment = self._new_assignment(node_requirements, result)
                for replicated_source_id in self._replicated_sources:
                    _append_update(
                        result,
                        assignment.partition_id,
                        replicated_source_id,
                        self._replicated_splits.get(replicated_source_id, []),
                        no_more_splits=replicated_source_id in self._completed_sources,
                        emit_empty=True,
                    )

            _append_update(
                result,
                assignment.partition_id,
                source_node_id,
                [split],
                ready_for_scheduling=True,
            )
            assignment.assign_split(split_size)

        if no_more_inputs:
            self._completed_sources.add(source_node_id)
        return self._merge_assignment_results(result, self._finish_if_all_sources_completed())

    def _new_assignment(self, node_requirements: NodeRequirements, result: AssignmentResult) -> _PartitionAssignment:
        assignment = self._PartitionAssignment(self._next_partition_id)
        self._next_partition_id += 1
        self._all_assignments.append(assignment)
        self._open_assignments[node_requirements] = assignment
        result.partitions_added.append(PartitionInfo(assignment.partition_id, node_requirements))
        return assignment

    def _node_requirements(self, split: FteSplit) -> NodeRequirements:
        if self._catalog_requirement is not None and split.catalog not in (None, self._catalog_requirement):
            raise ValueError(f"unexpected split catalog requirement: {split.catalog}")
        host = min(split.addresses, key=self._rank) if split.addresses else None
        if host is None and not split.remotely_accessible:
            raise ValueError("split is not remotely accessible but has no host address")
        return NodeRequirements(self._catalog_requirement, host, split.remotely_accessible)

    def _rank(self, address: str) -> int:
        flex = self._open_assignments.get(NodeRequirements(self._catalog_requirement, address, True))
        rigid = self._open_assignments.get(NodeRequirements(self._catalog_requirement, address, False))
        if flex is None and rigid is None:
            return -1
        if flex is None:
            return rigid.assigned_data_size_bytes
        if rigid is None:
            return flex.assigned_data_size_bytes
        return flex.assigned_data_size_bytes + rigid.assigned_data_size_bytes

    def _grow_target_if_needed(self) -> None:
        self._adaptive_counter += 1
        if self._adaptive_counter < self._adaptive_growth_period:
            return
        self._target_partition_size_bytes = min(
            self._max_target_partition_size_bytes,
            int(self._target_partition_size_bytes * self._adaptive_growth_factor + 0.999999),
        )
        self._rounded_target_partition_size_bytes = (
            round(self._target_partition_size_bytes / self._min_target_partition_size_bytes)
            * self._min_target_partition_size_bytes
        )
        self._rounded_target_partition_size_bytes = max(
            self._min_target_partition_size_bytes,
            self._rounded_target_partition_size_bytes,
        )
        self._adaptive_counter = 0

    def _finish_if_all_sources_completed(self) -> AssignmentResult:
        result = AssignmentResult()
        if not self._all_sources or not self._completed_sources.issuperset(self._all_sources):
            return result
        if not self._all_assignments:
            assignment = self._PartitionAssignment(0)
            self._next_partition_id = max(self._next_partition_id, 1)
            self._all_assignments.append(assignment)
            result.partitions_added.append(PartitionInfo(0, NodeRequirements(self._catalog_requirement)))
            for replicated_source_id in self._replicated_sources:
                _append_update(
                    result,
                    0,
                    replicated_source_id,
                    self._replicated_splits.get(replicated_source_id, []),
                    no_more_splits=True,
                )
            for partitioned_source_id in self._partitioned_sources:
                _append_update(result, 0, partitioned_source_id, no_more_splits=True)
            _append_seal(result, 0)
        else:
            for assignment in list(self._open_assignments.values()):
                for partitioned_source_id in self._partitioned_sources:
                    _append_update(
                        result,
                        assignment.partition_id,
                        partitioned_source_id,
                        no_more_splits=True,
                    )
                _append_seal(result, assignment.partition_id)
            self._open_assignments.clear()
        self._replicated_splits.clear()
        result.no_more_partitions = True
        self._finished = True
        return result

    @staticmethod
    def _merge_assignment_results(left: AssignmentResult, right: AssignmentResult) -> AssignmentResult:
        seen_seals = set(left.sealed_partitions)
        sealed = list(left.sealed_partitions)
        for partition_id in right.sealed_partitions:
            if partition_id not in seen_seals:
                sealed.append(partition_id)
                seen_seals.add(partition_id)
        return AssignmentResult(
            partitions_added=[*left.partitions_added, *right.partitions_added],
            partition_updates=[*left.partition_updates, *right.partition_updates],
            sealed_partitions=sealed,
            no_more_partitions=left.no_more_partitions or right.no_more_partitions,
        )


@dataclass
class HashTaskPartition:
    sub_partition_count: int = 1
    split_by_source: str | None = None
    task_partition_ids: list[int] = field(default_factory=list)
    next_sub_partition: int = 0

    def __post_init__(self) -> None:
        self.sub_partition_count = int(self.sub_partition_count)
        if self.sub_partition_count <= 0:
            raise ValueError("sub_partition_count must be positive")
        if self.sub_partition_count > 1 and self.split_by_source is None:
            raise ValueError("split_by_source is required for split hash task partitions")
        if self.split_by_source is not None:
            self.split_by_source = str(self.split_by_source)

    def ensure_ids(self, next_task_partition_id: int) -> int:
        while len(self.task_partition_ids) < self.sub_partition_count:
            self.task_partition_ids.append(next_task_partition_id)
            next_task_partition_id += 1
        return next_task_partition_id

    def ids_for_source(self, source_node_id: str) -> list[int]:
        if self.split_by_source == str(source_node_id):
            partition_id = self.task_partition_ids[self.next_sub_partition]
            self.next_sub_partition = (self.next_sub_partition + 1) % len(self.task_partition_ids)
            return [partition_id]
        return list(self.task_partition_ids)


def _coerce_hash_task_partition(value: Any) -> HashTaskPartition:
    if isinstance(value, HashTaskPartition):
        return value
    if isinstance(value, int):
        return HashTaskPartition(sub_partition_count=value)
    if isinstance(value, Mapping):
        return HashTaskPartition(
            sub_partition_count=int(value.get("sub_partition_count", value.get("sub_partitions", 1))),
            split_by_source=value.get("split_by_source") or value.get("split_by"),
        )
    raise TypeError(f"cannot coerce {type(value).__name__} to HashTaskPartition")


class HashSplitAssigner(SplitAssigner):
    """Hash-distribution split assigner with explicit source-partition metadata."""

    def __init__(
        self,
        *,
        source_partition_count: int,
        partitioned_sources: set[str] | list[str] | tuple[str, ...] | None = None,
        replicated_sources: set[str] | list[str] | tuple[str, ...] | None = None,
        source_partition_to_task_partition: Mapping[int, HashTaskPartition | Mapping[str, Any] | int] | None = None,
        source_partition_node_requirements: Mapping[int, NodeRequirements | Mapping[str, Any]] | None = None,
        catalog_requirement: str | None = None,
    ) -> None:
        if source_partition_count <= 0:
            raise ValueError("source_partition_count must be positive")
        self._source_partition_count = int(source_partition_count)
        self._partitioned_sources = _normalize_sources(partitioned_sources)
        self._replicated_sources = _normalize_sources(replicated_sources)
        self._all_sources = set(self._partitioned_sources) | set(self._replicated_sources)
        self._catalog_requirement = None if catalog_requirement is None else str(catalog_requirement)
        self._source_sequences: dict[str, int] = {}
        self._source_partition_to_task_partition = self._build_source_partition_mapping(
            source_partition_to_task_partition,
        )
        self._source_partition_node_requirements = self._normalize_node_requirements(
            source_partition_node_requirements,
        )
        self._created_task_partitions: set[int] = set()
        self._completed_sources: set[str] = set()
        self._replicated_splits: dict[str, list[FteSplit]] = {}
        self._all_task_partitions_created = False

    @staticmethod
    def one_task_per_source_partition(
        source_partition_count: int,
    ) -> dict[int, HashTaskPartition]:
        return {source_partition_id: HashTaskPartition() for source_partition_id in range(source_partition_count)}

    def assign(
        self,
        source_node_id: str,
        inputs: list[Mapping[str, Any]],
        no_more_inputs: bool = False,
    ) -> AssignmentResult:
        source_node_id = str(source_node_id)
        next_sequence = self._source_sequences.get(source_node_id, 0)
        splits, next_sequence = _splits_from_inputs(source_node_id, inputs, next_sequence)
        self._source_sequences[source_node_id] = next_sequence
        if not self._all_sources:
            self._partitioned_sources.add(source_node_id)
            self._all_sources.add(source_node_id)

        result = AssignmentResult()
        self._create_task_partitions(result)

        if source_node_id in self._replicated_sources:
            self._replicated_splits.setdefault(source_node_id, []).extend(splits)
            for task_partition_id in sorted(self._created_task_partitions):
                _append_update(
                    result,
                    task_partition_id,
                    source_node_id,
                    splits,
                    no_more_splits=no_more_inputs,
                    emit_empty=True,
                )
        else:
            for split in splits:
                source_partition_id = _split_source_partition_id(split)
                task_partition = self._task_partition_for_source_partition(source_partition_id)
                for task_partition_id in task_partition.ids_for_source(source_node_id):
                    _append_update(
                        result,
                        task_partition_id,
                        source_node_id,
                        [split],
                        ready_for_scheduling=True,
                    )

        if no_more_inputs:
            self._completed_sources.add(source_node_id)
            for task_partition_id in sorted(self._created_task_partitions):
                _append_update(result, task_partition_id, source_node_id, no_more_splits=True)
            if self._completed_sources.issuperset(self._all_sources):
                for task_partition_id in sorted(self._created_task_partitions):
                    _append_seal(result, task_partition_id)
                self._replicated_splits.clear()

        return result

    def finish(self) -> AssignmentResult:
        if not self._created_task_partitions:
            raise RuntimeError("createdTaskPartitions is not expected to be empty")
        return AssignmentResult()

    def _build_source_partition_mapping(
        self,
        mapping: Mapping[int, HashTaskPartition | Mapping[str, Any] | int] | None,
    ) -> dict[int, HashTaskPartition]:
        if mapping is None:
            mapping = self.one_task_per_source_partition(self._source_partition_count)
        result: dict[int, HashTaskPartition] = {}
        for source_partition_id in range(self._source_partition_count):
            if source_partition_id not in mapping:
                raise ValueError(f"missing task partition mapping for source partition {source_partition_id}")
            result[source_partition_id] = _coerce_hash_task_partition(mapping[source_partition_id])
        return result

    def _normalize_node_requirements(
        self,
        mapping: Mapping[int, NodeRequirements | Mapping[str, Any]] | None,
    ) -> dict[int, NodeRequirements]:
        result: dict[int, NodeRequirements] = {}
        for source_partition_id in range(self._source_partition_count):
            value = mapping.get(source_partition_id) if mapping else None
            if value is None:
                result[source_partition_id] = NodeRequirements(self._catalog_requirement)
            elif isinstance(value, NodeRequirements):
                result[source_partition_id] = value
            elif isinstance(value, Mapping):
                result[source_partition_id] = NodeRequirements(
                    catalog=value.get("catalog", self._catalog_requirement),
                    host=value.get("host"),
                    remotely_accessible=bool(value.get("remotely_accessible", True)),
                )
            else:
                raise TypeError(f"cannot coerce {type(value).__name__} to NodeRequirements")
        return result

    def _create_task_partitions(self, result: AssignmentResult) -> None:
        if self._all_task_partitions_created:
            return
        next_task_partition_id = 0
        partition_id_to_requirements: dict[int, NodeRequirements] = {}
        for source_partition_id in range(self._source_partition_count):
            task_partition = self._source_partition_to_task_partition[source_partition_id]
            next_task_partition_id = task_partition.ensure_ids(next_task_partition_id)
            for task_partition_id in task_partition.task_partition_ids:
                partition_id_to_requirements.setdefault(
                    task_partition_id,
                    self._source_partition_node_requirements[source_partition_id],
                )
        for task_partition_id in sorted(partition_id_to_requirements):
            result.partitions_added.append(
                PartitionInfo(task_partition_id, partition_id_to_requirements[task_partition_id])
            )
            self._created_task_partitions.add(task_partition_id)
        result.no_more_partitions = True
        self._all_task_partitions_created = True

    def _task_partition_for_source_partition(self, source_partition_id: int) -> HashTaskPartition:
        if source_partition_id < 0 or source_partition_id >= self._source_partition_count:
            raise ValueError(f"unknown source partition id: {source_partition_id}")
        return self._source_partition_to_task_partition[source_partition_id]
