# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vane.execution.resources import ResourceVector
from vane.execution.udf_admission import LocalExecutionCapacity, LocalExecutionSlotPool
from vane.execution.udf_local_memory import LocalTaskMemory

GiB = 1024**3


@pytest.fixture
def memory_capacity():
    capacity = LocalExecutionCapacity(
        max_slots=None,
        resource_limit=ResourceVector(cpu=8, heap_bytes=20 * GiB),
        task_memory_budget=6 * GiB,
        available_memory=lambda: 20 * GiB,
    )
    yield capacity
    for pool in list(capacity._pools):
        pool.close()
    for progress in list(capacity._progress):
        progress.shutdown()
    assert capacity.reserved_slots == 0
    assert not capacity._memory_stages


def _pool(capacity, profile, name="task"):
    return LocalExecutionSlotPool(
        max_slots=8,
        execution_slot_prefix=name,
        execution_capacity=capacity,
        resources=ResourceVector(cpu=1),
        memory=profile,
    )


def test_cold_task_calibrates_before_growing_and_buffered_output_keeps_no_memory_slot(memory_capacity):
    profile = LocalTaskMemory()
    authority = _pool(memory_capacity, profile).create_authority()
    first = authority.try_acquire(0)
    assert first is not None
    assert authority.try_acquire(0) is None
    memory_capacity.observe_task_memory(profile, GiB)
    first.complete_execution()
    assert profile.inflight == 0
    assert authority.active_lease_count == 1
    second = authority.try_acquire(0)
    third = authority.try_acquire(0)
    assert second is not None and third is not None
    assert authority.try_acquire(0) is None
    first.release()
    first.release()
    assert profile.inflight == 2
    second.release()
    third.release()
    assert profile.inflight == 0


def test_observed_peak_caps_parallelism_and_never_shrinks_with_small_later_batches(memory_capacity):
    profile = LocalTaskMemory(peak_bytes=2 * GiB, completed=100)
    authority = _pool(memory_capacity, profile).create_authority()
    first, second = authority.try_acquire(0), authority.try_acquire(0)
    assert first is not None and second is not None
    assert authority.try_acquire(0) is None
    memory_capacity.observe_task_memory(profile, 1)
    first.release()
    replacement = authority.try_acquire(0)
    assert replacement is not None
    assert authority.try_acquire(0) is None
    assert profile.estimate_bytes == 3 * GiB
    replacement.release()
    second.release()


def test_memory_reserves_downstream_progress_before_its_executor_exists(memory_capacity):
    profile = LocalTaskMemory(peak_bytes=8 * GiB, completed=100)
    resources = ResourceVector(cpu=1)
    progress = memory_capacity.reserve_task_progress(
        {"producer": resources, "consumer": resources},
        memory={"producer": profile, "consumer": profile},
    )
    pool = _pool(memory_capacity, profile)
    producer = pool.create_task_authority(progress.bind("producer"))
    first = producer.try_acquire(0)
    assert first is not None
    assert producer.try_acquire(0) is None
    consumer = pool.create_task_authority(progress.bind("consumer"))
    downstream = consumer.try_acquire(0)
    assert downstream is not None
    assert profile.inflight == 2
    downstream.release()
    first.release()


def test_independent_direct_executors_can_each_start_a_cold_task(memory_capacity):
    profile = LocalTaskMemory()
    pool = _pool(memory_capacity, profile)
    producer, consumer = pool.create_authority(), pool.create_authority()
    first = producer.try_acquire(0)
    assert first is not None
    assert producer.try_acquire(0) is None
    downstream = consumer.try_acquire(0)
    assert downstream is not None
    first.release()
    downstream.release()


def test_pressure_does_not_revoke_downstream_credit_after_producers_started(memory_capacity):
    profile = LocalTaskMemory(peak_bytes=GiB, completed=100)
    resources = ResourceVector(cpu=1)
    progress = memory_capacity.reserve_task_progress(
        {"producer": resources, "consumer": resources},
        memory={"producer": profile, "consumer": profile},
    )
    pool = _pool(memory_capacity, profile)
    producer = pool.create_task_authority(progress.bind("producer"))
    consumer = pool.create_task_authority(progress.bind("consumer"))
    producers = [producer.try_acquire(0) for _ in range(2)]
    assert all(lease is not None for lease in producers)
    memory_capacity._available_memory = lambda: 0
    memory_capacity._memory_sample_time = 0
    assert producer.try_acquire(0) is None
    downstream = consumer.try_acquire(0)
    assert downstream is not None
    downstream.release()
    for lease in producers:
        lease.release()


def test_budget_is_shared_with_other_profiles_and_declared_heap(memory_capacity):
    profiles = [LocalTaskMemory(peak_bytes=GiB, completed=100) for _ in range(2)]
    authorities = [_pool(memory_capacity, profile, str(i)).create_authority() for i, profile in enumerate(profiles)]
    release_resident = memory_capacity.reserve_resident(ResourceVector(heap_bytes=3 * GiB))
    leases = [authority.try_acquire(0) for authority in authorities]
    assert all(lease is not None for lease in leases)
    assert all(authority.try_acquire(0) is None for authority in authorities)
    for lease in leases:
        lease.release()
    release_resident()


def test_heavy_stage_uses_shared_surplus_without_spending_downstream_protection(memory_capacity):
    # Reproduce the audio shape: one heavy stage formerly stopped at one task
    # because its estimate exceeded half its fixed share of a three-way split.
    heavy = LocalTaskMemory(peak_bytes=GiB, completed=100)
    small = LocalTaskMemory(peak_bytes=GiB // 8, completed=100)
    resources = {key: ResourceVector(cpu=1) for key in ("heavy", "decode", "resample")}
    progress = memory_capacity.reserve_task_progress(
        resources, memory={"heavy": heavy, "decode": small, "resample": small}
    )
    producer = _pool(memory_capacity, heavy, "heavy").create_task_authority(progress.bind("heavy"))
    leases = [producer.try_acquire(0) for _ in range(2)]
    assert all(lease is not None for lease in leases)
    assert producer.try_acquire(0) is None
    snapshot = memory_capacity.resource_snapshot()["observed_task_memory"]
    assert snapshot["shared_pool_bytes"] > 0
    assert snapshot["shared_used_bytes"] > 0
    downstream = _pool(memory_capacity, small, "small")
    for node in ("decode", "resample"):
        lease = downstream.create_task_authority(progress.bind(node)).try_acquire(0)
        assert lease is not None
        leases.append(lease)
    for lease in leases:
        lease.release()


def test_competing_profiles_cannot_each_spend_the_same_shared_capacity(memory_capacity):
    profiles = [LocalTaskMemory(peak_bytes=GiB, completed=100) for _ in range(2)]
    left, right = [_pool(memory_capacity, profile, str(i)).create_authority() for i, profile in enumerate(profiles)]
    leases = [left.try_acquire(0), right.try_acquire(0), left.try_acquire(0)]
    assert all(lease is not None for lease in leases)
    assert left.try_acquire(0) is None
    right_extra = right.try_acquire(0)
    assert right_extra is not None
    leases.append(right_extra)
    assert left.try_acquire(0) is None and right.try_acquire(0) is None
    assert memory_capacity.resource_snapshot()["observed_task_memory"]["usage_bytes"] == 6 * GiB
    for lease in leases:
        lease.release()


@pytest.mark.parametrize("close_owner", ["authority", "pool", "query"])
def test_closed_owner_keeps_live_heap_charged_until_execution_finishes(memory_capacity, close_owner):
    heavy = LocalTaskMemory(peak_bytes=2 * GiB, completed=100)
    pool = _pool(memory_capacity, heavy, "closing")
    progress = None
    if close_owner == "query":
        progress = memory_capacity.reserve_task_progress({"task": ResourceVector(cpu=1)}, memory={"task": heavy})
    authority = pool.create_task_authority(None if progress is None else progress.bind("task"))
    live = [authority.try_acquire(0), authority.try_acquire(0)]
    assert all(lease is not None for lease in live)
    if close_owner == "query":
        progress.shutdown()
    else:
        (pool if close_owner == "pool" else authority).close()
    peer = _pool(memory_capacity, LocalTaskMemory(peak_bytes=GiB, completed=100), "peer").create_authority()
    first = peer.try_acquire(0)
    assert first is not None
    snapshot = memory_capacity.resource_snapshot()["observed_task_memory"]
    assert snapshot["usage_bytes"] == 15 * GiB // 2
    assert snapshot["retired_usage_bytes"] == 6 * GiB
    peer.request(0)
    assert not peer.state()["available"]
    live[0].complete_execution()
    assert peer.state()["available"]
    second = peer.take(0)
    for lease in [*live, first, second]:
        lease.release()
    assert heavy.inflight == 0


def test_same_stage_authorities_share_one_calibration_and_budget(memory_capacity):
    profile = LocalTaskMemory()
    progress = memory_capacity.reserve_task_progress({"task": ResourceVector(cpu=1)}, memory={"task": profile})
    pool = _pool(memory_capacity, profile)
    first_owner = pool.create_task_authority(progress.bind("task"))
    second_owner = pool.create_task_authority(progress.bind("task"))
    first = first_owner.try_acquire(0)
    assert first is not None
    assert second_owner.try_acquire(0) is None
    first_owner.close()
    assert second_owner.try_acquire(0) is None
    first.release()
    second = second_owner.try_acquire(0)
    assert second is not None
    second.release()


def test_releasing_an_idle_stage_reassigns_its_share_and_wakes_a_waiter(memory_capacity):
    profiles = [LocalTaskMemory(peak_bytes=GiB, completed=100) for _ in range(2)]
    busy, idle = [_pool(memory_capacity, profile, str(i)).create_authority() for i, profile in enumerate(profiles)]
    leases = [busy.try_acquire(0), busy.try_acquire(0)]
    assert all(lease is not None for lease in leases)
    busy.request(0)
    assert not busy.state()["available"]
    idle.close()
    assert busy.state()["available"]
    leases.append(busy.take(0))
    for lease in leases:
        lease.release()


def test_idle_cache_can_keep_workers_supported_by_shared_capacity(memory_capacity):
    heavy = LocalTaskMemory(peak_bytes=GiB, completed=100)
    small = LocalTaskMemory(peak_bytes=GiB // 8, completed=100)
    _pool(memory_capacity, heavy, "heavy").create_authority()
    _pool(memory_capacity, small, "small").create_authority()
    assert memory_capacity.task_memory_limit(heavy) == 3
    memory_capacity._available_memory = lambda: 0
    memory_capacity._memory_sample_time = 0
    assert memory_capacity.task_memory_limit(heavy) == 1


def test_transport_wait_retains_observed_heap_and_leaves_downstream_capacity(memory_capacity):
    from vane.execution.udf_lifecycle import ExecutionCancellationScope

    memory_capacity.resource_limit = ResourceVector(cpu=1, heap_bytes=20 * GiB)
    memory_capacity._task_memory_budget = 3 * GiB
    profiles = [LocalTaskMemory(peak_bytes=GiB, completed=100) for _ in range(2)]
    producer, consumer = [
        _pool(memory_capacity, profile, str(i)).create_authority() for i, profile in enumerate(profiles)
    ]
    upstream = producer.try_acquire(0)
    assert upstream is not None
    try:
        with upstream.suspend_for_wait(ExecutionCancellationScope("memory transport", 1)):
            assert producer.try_acquire(0) is None
            downstream = consumer.try_acquire(0)
            assert downstream is not None
            try:
                snapshot = memory_capacity.resource_snapshot()
                assert snapshot["usage"]["cpu"] == 1
                assert snapshot["observed_task_memory"]["usage_bytes"] == 3 * GiB
                assert profiles[0].inflight == 1
            finally:
                downstream.release()
    finally:
        upstream.release()


@pytest.mark.parametrize("close_pool", [False, True])
def test_cancelling_ready_grant_returns_observed_memory_once(memory_capacity, close_pool):
    profile = LocalTaskMemory()
    pool = _pool(memory_capacity, profile)
    authority = pool.create_authority()
    authority.request(4)
    assert authority.state()["available"]
    assert profile.inflight == 1
    (pool if close_pool else authority).close()
    authority.close()
    assert profile.inflight == 0


def test_memory_pressure_preserves_progress_and_completion_rechecks_recovery(memory_capacity):
    profile = LocalTaskMemory(peak_bytes=GiB, completed=100)
    authority = _pool(memory_capacity, profile).create_authority()
    headroom = [0]
    memory_capacity._available_memory = lambda: headroom[0]
    first = authority.try_acquire(0)
    assert first is not None
    authority.request(0)
    assert not authority.state()["available"]
    headroom[0] = 20 * GiB
    memory_capacity._memory_sample_time = 0
    first.release()
    assert authority.state()["available"]
    second = authority.take(0)
    third = authority.try_acquire(0)
    assert third is not None
    second.release()
    third.release()


def test_rejected_data_guard_does_not_consume_memory_slot(memory_capacity):
    profile = LocalTaskMemory()
    authority = _pool(memory_capacity, profile).create_authority()
    assert authority.try_acquire_if(0, lambda: False) is None
    assert profile.inflight == 0
    lease = authority.try_acquire(0)
    assert lease is not None
    lease.release()


def test_cgroup_headroom_respects_parent_and_host_limits(monkeypatch, tmp_path):
    import vane.execution.udf_local_memory as memory

    (tmp_path / "proc/self").mkdir(parents=True)
    (tmp_path / "proc/self/cgroup").write_text("0::/parent/child\n")
    root = tmp_path / "sys/fs/cgroup"
    child = root / "parent/child"
    child.mkdir(parents=True)
    for path, limit, used in [(root, "max", 0), (child.parent, 1000, 900), (child, 800, 100)]:
        (path / "memory.max").write_text(str(limit))
        (path / "memory.current").write_text(str(used))
    monkeypatch.setattr(memory, "Path", lambda path: tmp_path / Path(path).relative_to("/"))
    monkeypatch.setattr(memory.sys, "platform", "linux")
    monkeypatch.setattr(memory.psutil, "virtual_memory", lambda: SimpleNamespace(available=500))
    assert memory.available_process_memory() == 100
    (child.parent / "memory.max").write_text("max")
    assert memory.available_process_memory() == 500


def test_linux_peak_reads_high_water_not_current_rss(monkeypatch, tmp_path):
    import vane.execution.udf_local_memory as memory

    status = tmp_path / "status"
    status.write_text("VmHWM:\t2048 kB\nVmRSS:\t64 kB\n")
    monkeypatch.setattr(memory.sys, "platform", "linux")
    monkeypatch.setattr(memory, "Path", lambda _: status)
    assert memory.task_process_peak_bytes(123) == 2048 * 1024
    status.unlink()
    assert memory.task_process_peak_bytes(123) == 0


def test_failed_memory_sample_keeps_worker_cleanup_owner(monkeypatch):
    from vane.execution import udf_subprocess as module

    runtime = module._GlobalSubprocessTaskRuntime(resource_limit=ResourceVector(cpu=2, heap_bytes=8 * GiB))
    pool = runtime.acquire_pool({"execution_backend": "subprocess_task"}, 2)
    closed = []
    worker = SimpleNamespace(close=lambda **kw: closed.append(kw), cancel_output_grants=lambda: None, _closed=False)
    wrapper = module._PooledTaskWorker(worker)
    pool._active_wrappers.add(wrapper)
    pool.active = pool.total = runtime.total_workers = 1

    def fail_sample():
        raise OSError("memory sample failed")

    monkeypatch.setattr(runtime.execution_capacity, "_available_memory", fail_sample)
    try:
        with pytest.raises(OSError, match="memory sample failed"):
            pool.release_worker(wrapper)
        assert wrapper in pool._active_wrappers
        assert pool.active == 1
    finally:
        runtime.close(kill=True)
    assert closed == [{"kill": True}]
