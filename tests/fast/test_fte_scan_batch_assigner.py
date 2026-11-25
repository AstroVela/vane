# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from random import Random
from types import SimpleNamespace

import pytest

from vane.runners.fte import ArbitrarySplitAssigner, HashSplitAssigner, ScanBatchSplitAssigner
from vane.runners.ray.fragment_worker_assignment import make_fte_assigner


def _splits(sizes, **metadata):
    return [
        {"kind": "scan_split", "split_id": str(index), "size_bytes": size, "data": (index, size), **metadata}
        for index, size in enumerate(sizes)
    ]


def _tasks(result):
    return [update.splits for update in result.partition_updates if update.splits]


def _loads(result, unknown=64 * 1024 * 1024):
    return [sum(unknown if split.size_bytes is None else split.size_bytes for split in task) for task in _tasks(result)]


def test_scan_batch_lpt_balances_across_singleton_transport_events():
    assigner = ScanBatchSplitAssigner(
        "scan", worker_slots=2, tasks_per_slot=1, min_task_size_bytes=1, max_task_size_bytes=20
    )
    inputs = _splits([9, 8, 7, 6])
    for split in inputs:
        assert assigner.assign("scan", [split]).partitions_added == []
    result = assigner.flush()

    # Next-Fit with a 20-byte limit produces 17, 13 for this input.
    assert _loads(result) == [15, 15]
    assert [[split.split_id for split in task] for task in _tasks(result)] == [["0", "3"], ["1", "2"]]
    assert result.sealed_partitions == [0, 1]
    assert all(update.no_more_splits and update.ready_for_scheduling for update in result.partition_updates)
    assert not result.no_more_partitions
    assert assigner.finish().no_more_partitions


@pytest.mark.parametrize(("slots", "task_count"), [(1, 4), (2, 8), (8, 16)])
def test_scan_batch_task_count_uses_slots_and_minimum_size(slots, task_count):
    assigner = ScanBatchSplitAssigner("scan", worker_slots=slots, min_task_size_bytes=10, max_task_size_bytes=1000)
    result = assigner.assign("scan", _splits([10] * 16), no_more_inputs=True)
    assert len(_tasks(result)) == task_count


def test_scan_batch_tiny_files_are_coalesced_even_with_many_slots():
    assigner = ScanBatchSplitAssigner("scan", worker_slots=100)
    result = assigner.assign("scan", _splits([1] * 100), no_more_inputs=True)
    assert _loads(result) == [100]


def test_scan_batch_unknown_size_is_conservative_and_zero_is_known():
    assigner = ScanBatchSplitAssigner(
        "scan", min_task_size_bytes=10, max_task_size_bytes=10, standard_split_size_bytes=10
    )
    result = assigner.assign("scan", _splits([None, None, 0, 1]), no_more_inputs=True)
    assert sorted(_loads(result, unknown=10)) == [1, 10, 10]
    assert sum(map(len, _tasks(result))) == 4


def test_scan_batch_capacity_and_indivisible_oversize_split():
    assigner = ScanBatchSplitAssigner(
        "scan", tasks_per_slot=1, min_task_size_bytes=1, max_task_size_bytes=10, max_task_split_count=2
    )
    result = assigner.assign("scan", _splits([30, 6, 6, 4, 4, 1, 1, 1]), no_more_inputs=True)
    assert all(len(task) <= 2 for task in _tasks(result))
    assert all(load <= 10 or (len(task) == 1 and load == 30) for task, load in zip(_tasks(result), _loads(result)))
    assert sum(_loads(result)) == 53


@pytest.mark.parametrize(
    ("sizes_mib", "split_limit", "expected_loads_mib"),
    [
        ([2048] + [1] * 7, 2048, [2048, 7]),
        ([2048] + [1] * 7, 2, [2048, 2, 2, 2, 1]),
        ([2048] + [0] * 7, 2048, [2048, 0]),
        ([512, 2048, 1024], 2048, [2048, 1024, 512]),
        ([2048] + [64] * 16, 2048, [2048] + [128] * 8),
    ],
)
def test_scan_batch_oversize_splits_do_not_inflate_regular_task_count(sizes_mib, split_limit, expected_loads_mib):
    mib = 1024 * 1024
    inputs = _splits([size * mib for size in sizes_mib])
    assigner = ScanBatchSplitAssigner("scan", worker_slots=2, max_task_split_count=split_limit)
    result = assigner.assign("scan", inputs, no_more_inputs=True)

    assert _loads(result) == [size * mib for size in expected_loads_mib]
    assert all(len(task) <= split_limit for task in _tasks(result))
    actual = sorted((split for task in _tasks(result) for split in task), key=lambda split: split.sequence_id)
    assert [(split.split_id, split.data, split.sequence_id) for split in actual] == [
        (split["split_id"], split["data"], index) for index, split in enumerate(inputs)
    ]
    assert result.sealed_partitions == list(range(len(expected_loads_mib)))
    assert result.no_more_partitions


def test_scan_batch_emits_largest_final_task_first_across_locality_groups():
    assigner = ScanBatchSplitAssigner("scan", tasks_per_slot=1, min_task_size_bytes=10, max_task_size_bytes=100)
    inputs = [
        {**split, "catalog": host, "addresses": [host], "remotely_accessible": False}
        for split, host in zip(_splits([60, 50, 40]), ["a", "b", "b"])
    ]
    result = assigner.assign("scan", inputs, no_more_inputs=True)

    # Group a has the largest individual split, but group b has more total work.
    assert _loads(result) == [90, 60]
    assert [part.node_requirements.host for part in result.partitions_added] == ["b", "a"]
    assert [part.node_requirements.catalog for part in result.partitions_added] == ["b", "a"]
    assert all(not part.node_requirements.remotely_accessible for part in result.partitions_added)


def test_scan_batch_respects_catalog_and_local_only_access():
    assigner = ScanBatchSplitAssigner("scan", min_task_size_bytes=1, max_task_size_bytes=100)
    inputs = [
        {**split, **metadata}
        for split, metadata in zip(
            _splits([9, 8, 7, 6, 5]),
            [
                {"catalog": "a", "addresses": ["a", "b"], "remotely_accessible": False},
                {"catalog": "a", "addresses": ["a", "b"], "remotely_accessible": False},
                {"catalog": "b", "addresses": ["a"], "remotely_accessible": False},
                {"catalog": "a", "addresses": ["b"], "remotely_accessible": True},
                {"catalog": "a"},
            ],
        )
    ]
    result = assigner.assign("scan", inputs, no_more_inputs=True)
    requirements = {part.partition_id: part.node_requirements for part in result.partitions_added}
    hosts = {}
    for update in result.partition_updates:
        required = requirements[update.partition_id]
        for split in update.splits:
            assert required.catalog == split.catalog
            assert required.remotely_accessible == split.remotely_accessible
            assert required.host in split.addresses if split.addresses else required.host is None
            hosts[split.split_id] = required.host
    assert hosts["0"] == "a"
    assert hosts["1"] == "b"


def test_scan_batch_rejects_missing_local_address_before_buffering():
    assigner = ScanBatchSplitAssigner("scan")
    with pytest.raises(ValueError, match="address"):
        assigner.assign("scan", _splits([1, 2], remotely_accessible=False))
    assert assigner.flush().partitions_added == []


def test_scan_batch_bounds_buffer_and_preserves_identity_across_batches():
    assigner = ScanBatchSplitAssigner("scan", max_batch_split_count=3)
    inputs = _splits([1] * 8)
    first = assigner.assign("scan", inputs[:7])
    second = assigner.flush()
    third = assigner.assign("scan", inputs[7:], no_more_inputs=True)
    results = [first, second, third]
    assert [len(_tasks(result)) for result in results] == [2, 1, 1]
    assert [part.partition_id for result in results for part in result.partitions_added] == [0, 1, 2, 3]
    actual = [split for result in results for task in _tasks(result) for split in task]
    assert [(split.split_id, split.data, split.sequence_id) for split in actual] == [
        (str(index), (index, 1), index) for index in range(8)
    ]
    assert third.no_more_partitions
    assert assigner.flush().partitions_added == []
    with pytest.raises(RuntimeError, match="finish"):
        assigner.assign("scan", inputs)


def test_scan_batch_empty_scan_emits_one_sealed_task():
    assigner = ScanBatchSplitAssigner("scan")
    assert assigner.flush().partitions_added == []
    result = assigner.finish()
    assert result.sealed_partitions == [0]
    assert result.no_more_partitions
    assert result.partition_updates[0].no_more_splits
    assert not result.partition_updates[0].ready_for_scheduling
    assert assigner.finish().partitions_added == []


@pytest.mark.parametrize("seed", range(5))
def test_scan_batch_random_inputs_preserve_all_splits_and_limits(seed):
    random = Random(seed)
    inputs = _splits([random.choice([None, 0, 1, 9, 40, 100, 200]) for _ in range(300)])
    assigner = ScanBatchSplitAssigner(
        "scan",
        worker_slots=4,
        min_task_size_bytes=10,
        max_task_size_bytes=100,
        standard_split_size_bytes=40,
        max_task_split_count=7,
        max_batch_split_count=43,
    )
    results = [assigner.assign("scan", inputs[offset : offset + 13]) for offset in range(0, len(inputs), 13)]
    results.append(assigner.finish())
    seen = []
    for result in results:
        assert result.sealed_partitions == [part.partition_id for part in result.partitions_added]
        for task in _tasks(result):
            assert len(task) <= 7
            assert len(task) == 1 or sum(40 if split.size_bytes is None else split.size_bytes for split in task) <= 100
            seen.extend(split.split_id for split in task)
    assert Counter(seen) == Counter(split["split_id"] for split in inputs)


def _state(**overrides):
    return SimpleNamespace(
        **{
            "source_node_ids": {"scan"},
            "dynamic_scan_source_node_ids": {"scan"},
            "dynamic_exchange_source_node_ids": set(),
            "replicated_exchange_source_node_ids": set(),
            "preserve_order": False,
            "exchange_source_partition_ids": {0, 1},
            "exchange_source_partition_count": 2,
            "exchange_source_task_count": 2,
            **overrides,
        }
    )


def test_scan_batch_factory_uses_capacity_and_split_limit(monkeypatch):
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "2")
    monkeypatch.setenv("VANE_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION", "1")
    assigner = make_fte_assigner(_state())
    assert isinstance(assigner, ScanBatchSplitAssigner)
    result = assigner.assign("scan", _splits([1, 1]), no_more_inputs=True)
    assert len(_tasks(result)) == 2
    monkeypatch.delenv("VANE_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION")
    result = make_fte_assigner(_state()).assign("scan", _splits([64 * 1024 * 1024] * 8), no_more_inputs=True)
    assert len(_tasks(result)) == 8


@pytest.mark.parametrize("slots", ["", "invalid", "0", "-1"])
def test_scan_batch_factory_falls_back_to_one_slot(monkeypatch, slots):
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", slots)
    result = make_fte_assigner(_state()).assign("scan", _splits([64 * 1024 * 1024] * 8), no_more_inputs=True)
    assert len(_tasks(result)) == 4


@pytest.mark.parametrize(
    "state",
    [
        _state(preserve_order=True),
        _state(source_node_ids={"scan", "other"}, dynamic_scan_source_node_ids={"scan", "other"}),
        _state(
            source_node_ids={"scan", "build"},
            dynamic_exchange_source_node_ids={"build"},
            replicated_exchange_source_node_ids={"build"},
        ),
    ],
)
def test_scan_batch_factory_preserves_ordered_multisource_and_broadcast_paths(state):
    assert isinstance(make_fte_assigner(state), ArbitrarySplitAssigner)


def test_scan_batch_factory_preserves_hash_distribution():
    assert isinstance(make_fte_assigner(_state(dynamic_exchange_source_node_ids={"exchange"})), HashSplitAssigner)
