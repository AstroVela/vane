# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from vane.runners.fte.fte_config import FteWorkerAdmissionConfig
from vane.runners.fte.fte_failures import _failure_allows_retry
from vane.runners.ray import fte_fragment_scheduler
from vane.runners.ray.fragment_registry import (
    _FTE_FRAGMENT_EXECUTIONS,
    _FTE_REGISTRY_LOCK,
    _FTE_SCHEDULERS,
    _FteFragmentState,
)
from vane.runners.ray.fragment_worker_assignment import make_fte_assigner
from vane.runners.ray.fragment_worker_results import fte_query_status
from vane.runners.ray.fte import (
    ArbitrarySplitAssigner,
    AssignmentResult,
    FteExchangeSourceOutputSelector,
    FteExchangeTracker,
    FteFragmentExecution,
    FtePartitionState,
    FteTaskAttemptId,
    FteTaskExecution,
    FteTaskId,
    FteTaskState,
    FteTaskUpdateRequest,
    FteWorkerCommandExecutor,
    FteWorkerControlFailure,
    FteWorkerReservationUnavailable,
    FteWorkerTaskManager,
    HashSplitAssigner,
    HashTaskPartition,
    NodeRequirements,
    PartitionInfo,
    PartitionUpdate,
    SingleSplitAssigner,
    SpoolingExchangeManager,
    TaskDescriptor,
    TaskDescriptorStorage,
    collect_spooling_output_stats,
    derive_exchange_sink_instance_for_attempt,
    materialize_task_inputs,
)
from vane.runners.ray.fte_attempts import RunningAttempt
from vane.runners.ray.fte_events import FteAddSplitsCommand
from vane.runners.ray.query_execution_graph import (
    NodeResourceAllocation,
    QueryAllocation,
    QueryExecutionGraph,
    ResourceVector,
    StageResourceSpec,
)
from vane.runners.ray.query_resource_runtime import (
    clear_query_resource_managers,
    register_query_graph,
)


def test_fte_partition_admission_uses_non_persistent_descriptor_arbitration(monkeypatch):
    lease = object()
    descriptor_requests = []
    submission_events = []

    class _Grant:
        granted = True
        fatal = False
        blocked_reason = ""

        def __init__(self):
            self.lease = lease

    class _Manager:
        def try_acquire_task_descriptor(self, request):
            descriptor_requests.append(request)
            return _Grant()

    monkeypatch.setattr(
        fte_fragment_scheduler,
        "_fte_partition_attempt_identity",
        lambda *_args: ("task:0", "task:0/attempt:0"),
    )
    monkeypatch.setattr(
        fte_fragment_scheduler,
        "admit_fte_partition_submission",
        lambda *args: submission_events.append(("admit", args)) or True,
    )
    monkeypatch.setattr(
        fte_fragment_scheduler,
        "resolve_fte_partition_submission",
        lambda *args, **kwargs: submission_events.append(("resolve", args, kwargs)),
    )
    monkeypatch.setattr(
        fte_fragment_scheduler,
        "_fte_fragment_resource_identity",
        lambda query_id, _fragment_id: (
            query_id,
            f"stage:{query_id}:node:scan:fte",
        ),
    )
    monkeypatch.setattr(
        "vane.runners.ray.query_resource_runtime.get_query_resource_manager",
        lambda _query_id: _Manager(),
    )

    result = fte_fragment_scheduler._acquire_fte_partition_task_lease(
        query_id="query-global-order",
        fragment_execution_id=1,
        fragment_id="fragment-1",
        partition_id=0,
        node_id="node-a",
    )

    assert result is lease
    assert len(descriptor_requests) == 1
    assert [event[0] for event in submission_events] == ["admit", "resolve"]
    assert submission_events[-1][2]["granted"] is True


class _PlacementWorker:
    _fte_healthy = True

    def __init__(self, worker_id="worker-1", *, reserve_error=None):
        self.worker_id = worker_id
        self.node_id = "node-a"
        self.reserve_error = reserve_error
        self.reserved: list[tuple[str, str, int]] = []
        self.released: list[tuple[str, str, int]] = []
        self.terminal_attempts: list[str] = []
        self.dropped_queries: list[str] = []

    def reserve_fte_partition(self, query_id, fragment_id, partition_id, **_kwargs):
        if self.reserve_error is not None:
            raise self.reserve_error
        self.reserved.append((str(query_id), str(fragment_id), int(partition_id)))

    def release_fte_partition_reservation(self, query_id, fragment_id, partition_id):
        self.released.append((str(query_id), str(fragment_id), int(partition_id)))

    def record_fte_task_terminal(self, attempt_id):
        self.terminal_attempts.append(str(FteTaskAttemptId.coerce(attempt_id)))

    def ensure_fragment_registered(self, _query_id, _fragment_id, _fragment_plan):
        return {"registered": 1, "existing": 0}

    def fte_pressure_stats(self):
        return {
            "running_attempt_count": 0,
            "reserved_partition_count": 0,
            "assigned_memory_bytes": 0,
            "reserved_memory_bytes": 0,
            "assigned_split_bytes": 0,
            "assigned_split_count": 0,
        }

    def _drop_fte_state_for_query(self, query_id):
        self.dropped_queries.append(str(query_id))


class _PlacementCoordinator:
    _fte_healthy = False

    def __init__(self, worker=None, *, select_error=None):
        self.worker = worker
        self.select_error = select_error

    def _select_fte_worker(self, **_kwargs):
        if self.select_error is not None:
            raise self.select_error
        return self.worker


_GIB = 1024**3
_MIB = 1024**2

_ProductionFteFragmentExecution = FteFragmentExecution
_ProductionFteWorkerTaskManager = FteWorkerTaskManager


def _fte_fragment_execution(*args, memory_bytes=1024, **kwargs):
    return _ProductionFteFragmentExecution(
        *args,
        task_memory_bytes=memory_bytes,
        **kwargs,
    )


def _fte_worker_task_manager(
    execute_fn,
    *,
    max_running_tasks=32,
    admission_config=None,
    **kwargs,
):
    if admission_config is None:
        admission_config = FteWorkerAdmissionConfig(
            max_running_tasks=max_running_tasks,
            mode="test",
            memory_budget_bytes=1 << 60,
            task_memory_bytes=1,
        )
    return _ProductionFteWorkerTaskManager(
        execute_fn,
        admission_config=admission_config,
        **kwargs,
    )


def _register_fte_query(query_id: str, node_id: str, *, partitions: int, task_slots: int):
    stage_id = f"stage:{query_id}:node:{node_id}:fte"
    graph = QueryExecutionGraph(
        query_id=query_id,
        plan_digest=f"sha256:{query_id}",
        stages=(
            StageResourceSpec(
                query_id=query_id,
                stage_id=stage_id,
                physical_node_id=f"node:{node_id}:fte",
                stage_kind="fte",
                backend="ray_worker",
                input_stage_ids=(),
                per_task=ResourceVector(cpu=1, heap_bytes=2 * _GIB),
                target_output_block_bytes=128 * _MIB,
                generator_buffer_blocks=2,
                max_concurrency=partitions,
            ),
        ),
        terminal_stage_ids=(stage_id,),
    )
    allocation_resources = ResourceVector(
        cpu=task_slots,
        heap_bytes=task_slots * 2 * _GIB,
        object_store_bytes=task_slots * 256 * _MIB,
    )
    manager = register_query_graph(
        graph,
        QueryAllocation(
            resources=allocation_resources,
            node_allocations=(
                NodeResourceAllocation(
                    node_id="node-a",
                    resources=allocation_resources,
                ),
            ),
            actor_placements=(),
            generation=1,
        ),
    )
    manager.update_stage_state(stage_id, runnable=True)
    return manager, stage_id


def _install_fte_fragment(query_id: str, node_id: str, *, partitions: int, context=None):
    fragment_id = f"{query_id}:node:{node_id}"
    stage_context = {
        "resource_query_id": query_id,
        "resource_stage_id": f"stage:{query_id}:node:{node_id}:fte",
        **dict(context or {}),
    }
    stage = _fte_fragment_execution(
        query_id,
        7,
        fragment_id=fragment_id,
        context=stage_context,
    )
    for partition_id in range(partitions):
        stage.add_partition(partition_id)
    with _FTE_REGISTRY_LOCK:
        _FTE_FRAGMENT_EXECUTIONS[(query_id, fragment_id)] = stage
    fte_fragment_scheduler.ensure_fte_fragment_progress_topology(
        query_id,
        fragment_id,
        lambda: {
            "schema": "pipeline_topology",
            "pipelines": [
                {
                    "pipeline_id": 1,
                    "operators": ["TABLE_SCAN"],
                    "operator_details": [{}],
                    "stage_ids": [],
                }
            ],
        },
    )
    return stage, fragment_id


def _cleanup_fte_query(query_id: str):
    fte_fragment_scheduler._drop_fte_registry_for_query(query_id)
    clear_query_resource_managers()


def test_fte_failure_retryability_parses_boolean_strings():
    assert _failure_allows_retry({"retryable": "true"}) is True
    assert _failure_allows_retry({"retryable": "false"}) is False


def test_fte_stage_uses_one_global_lease_domain_for_all_36_partitions():
    query_id = "q-fte-stage-leases"
    clear_query_resource_managers()
    manager, stage_id = _register_fte_query(query_id, "scan", partitions=36, task_slots=14)
    _, fragment_id = _install_fte_fragment(query_id, "scan", partitions=36)
    worker = _PlacementWorker()
    placement = fte_fragment_scheduler.FteWorkerPlacementManager(_PlacementCoordinator(worker))
    reservations = []
    try:
        for partition_id in range(36):
            try:
                reservations.append(
                    placement.acquire(
                        query_id=query_id,
                        fragment_id=fragment_id,
                        partition_id=partition_id,
                    )
                )
            except FteWorkerReservationUnavailable:
                pass

        snapshot = manager.snapshot()
        assert len(reservations) == 14
        assert {reservation.stage_id for reservation in reservations} == {stage_id}
        assert len({reservation.task_lease_id for reservation in reservations}) == 14
        assert snapshot["stages"][stage_id]["active_task_count"] == 14
        assert snapshot["liveness"]["task_grants_total"] == 0
        assert len(snapshot["task_leases"]) == 14
    finally:
        _cleanup_fte_query(query_id)


def test_fte_task_lease_uses_attempt_identity_and_releases_once_at_terminal():
    query_id = "q-fte-attempt-lease"
    clear_query_resource_managers()
    manager, stage_id = _register_fte_query(query_id, "scan", partitions=1, task_slots=1)
    stage, fragment_id = _install_fte_fragment(query_id, "scan", partitions=1)
    worker = _PlacementWorker()
    placement = fte_fragment_scheduler.FteWorkerPlacementManager(_PlacementCoordinator(worker))
    try:
        reservation = placement.acquire(query_id=query_id, fragment_id=fragment_id, partition_id=0)
        expected_attempt = FteTaskAttemptId(stage.partitions[0].task_id, 0)

        assert reservation.stage_id == stage_id
        assert reservation.attempt_id == str(expected_attempt)
        lease = manager.snapshot()["task_leases"][reservation.task_lease_id]
        assert lease["task_id"] == str(stage.partitions[0].task_id)
        assert lease["attempt_id"] == str(expected_attempt)

        worker.record_fte_task_terminal(expected_attempt)
        worker.record_fte_task_terminal(expected_attempt)

        assert manager.snapshot()["task_leases"] == {}
        assert worker.terminal_attempts == [str(expected_attempt), str(expected_attempt)]
    finally:
        _cleanup_fte_query(query_id)


def test_internal_fte_query_uses_outer_query_resource_identity():
    resource_query_id = "q-fte-resource-owner"
    execution_query_id = f"{resource_query_id}_orderby_stage"
    fragment_id = f"{execution_query_id}:orderby:2:stage:0"
    clear_query_resource_managers()
    manager, stage_id = _register_fte_query(
        resource_query_id,
        "scan",
        partitions=1,
        task_slots=1,
    )
    stage = _fte_fragment_execution(
        execution_query_id,
        0,
        fragment_id=fragment_id,
        context={
            "resource_query_id": resource_query_id,
            "resource_stage_id": stage_id,
        },
    )
    stage.add_partition(0)
    with _FTE_REGISTRY_LOCK:
        _FTE_FRAGMENT_EXECUTIONS[(execution_query_id, fragment_id)] = stage

    worker = _PlacementWorker()
    placement = fte_fragment_scheduler.FteWorkerPlacementManager(_PlacementCoordinator(worker))
    try:
        reservation = placement.acquire(
            query_id=execution_query_id,
            fragment_id=fragment_id,
            partition_id=0,
        )
        payload = fte_fragment_scheduler.fte_partition_task_lease_payload(
            execution_query_id,
            fragment_id,
            0,
            reservation.attempt_id,
        )

        assert reservation.stage_id == stage_id
        assert manager.snapshot()["task_leases"][reservation.task_lease_id]["query_id"] == resource_query_id
        assert payload["query_id"] == resource_query_id
        assert payload["execution_query_id"] == execution_query_id
        assert fte_fragment_scheduler.fte_execution_query_ids_for_resource(resource_query_id) == (execution_query_id,)
    finally:
        fte_fragment_scheduler._drop_fte_registry_for_query(execution_query_id)
        clear_query_resource_managers()


def test_fte_query_drop_releases_all_task_leases():
    query_id = "q-fte-release-query"
    clear_query_resource_managers()
    manager, _ = _register_fte_query(query_id, "scan", partitions=2, task_slots=2)
    _, fragment_id = _install_fte_fragment(query_id, "scan", partitions=2)
    worker = _PlacementWorker()
    placement = fte_fragment_scheduler.FteWorkerPlacementManager(_PlacementCoordinator(worker))
    try:
        first = placement.acquire(query_id=query_id, fragment_id=fragment_id, partition_id=0)
        second = placement.acquire(query_id=query_id, fragment_id=fragment_id, partition_id=1)
        assert first.task_lease_id != second.task_lease_id

        released = placement.release_query(query_id)

        assert released == 2
        assert manager.snapshot()["task_leases"] == {}
        assert sorted(worker.released) == sorted([(query_id, fragment_id, 0), (query_id, fragment_id, 1)])
    finally:
        _cleanup_fte_query(query_id)


@pytest.mark.parametrize(
    ("worker", "error_match"),
    [
        (None, "no live Ray worker"),
        (_PlacementWorker(reserve_error=RuntimeError("reservation failed")), "reservation failed"),
    ],
)
def test_fte_worker_selection_or_reservation_failure_releases_task_lease(worker, error_match):
    query_id = f"q-fte-acquire-failure-{error_match or 'selection'}"
    clear_query_resource_managers()
    manager, _ = _register_fte_query(query_id, "scan", partitions=1, task_slots=1)
    _, fragment_id = _install_fte_fragment(query_id, "scan", partitions=1)
    placement = fte_fragment_scheduler.FteWorkerPlacementManager(_PlacementCoordinator(worker))
    try:
        with pytest.raises(RuntimeError, match=error_match):
            placement.acquire(query_id=query_id, fragment_id=fragment_id, partition_id=0)

        assert manager.snapshot()["task_leases"] == {}
    finally:
        _cleanup_fte_query(query_id)


def test_fte_write_sink_updates_registered_stage_state_instead_of_registering_an_operator():
    query_id = "q-write-sink-stage"
    clear_query_resource_managers()
    manager, stage_id = _register_fte_query(query_id, "sink", partitions=1, task_slots=1)
    stage, fragment_id = _install_fte_fragment(
        query_id,
        "sink",
        partitions=1,
        context={
            "node_id": "sink",
            "copy_output_base": "",
            "copy_output_run_id": "run-write",
            "copy_output_remote_base": "/tmp/out.parquet",
        },
    )
    try:
        fte_fragment_scheduler._sync_write_sink_stage_for_fragment(stage)
        assert manager.snapshot()["stages"][stage_id]["runnable"] is False

        stage.partitions[0].mark_ready_for_execution()
        fte_fragment_scheduler._sync_write_sink_stage_for_fragment(stage)
        assert manager.snapshot()["stages"][stage_id]["runnable"] is True
        assert fragment_id == f"{query_id}:node:sink"
    finally:
        _cleanup_fte_query(query_id)


@pytest.mark.parametrize("snapshot_kind", ["progress", "diagnostic"])
def test_registry_snapshot_releases_global_lock_before_fragment_copy(snapshot_kind):
    query_id = f"q-{snapshot_kind}-lock-order"
    stage, fragment_id = _install_fte_fragment(query_id, "scan", partitions=1)
    state_copy_started = threading.Event()
    state_copy_release = threading.Event()
    original_state_lock = stage._state_lock

    class _BlockingStateLock:
        def __enter__(self):
            state_copy_started.set()
            state_copy_release.wait(timeout=1.0)
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return False

    stage._state_lock = _BlockingStateLock()
    snapshots = []
    errors = []

    def _collect() -> None:
        try:
            if snapshot_kind == "progress":
                snapshot = fte_fragment_scheduler.fte_progress_registry_snapshot(query_id)
            else:
                snapshot = fte_fragment_scheduler.fte_registry_stats()
            snapshots.append(snapshot)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    collector = threading.Thread(target=_collect, daemon=True)
    collector.start()
    acquired_registry_lock = False
    try:
        assert state_copy_started.wait(timeout=1.0)
        acquired_registry_lock = _FTE_REGISTRY_LOCK.acquire(timeout=0.1)
        assert acquired_registry_lock is True
    finally:
        if acquired_registry_lock:
            _FTE_REGISTRY_LOCK.release()
        state_copy_release.set()
        collector.join(timeout=1.0)
        stage._state_lock = original_state_lock
        _cleanup_fte_query(query_id)

    assert collector.is_alive() is False
    assert errors == []
    assert snapshots[0]["queries"][query_id]["fragment_executions"][fragment_id]


@pytest.mark.parametrize("snapshot_kind", ["progress", "diagnostic"])
def test_registry_snapshot_is_observation_only(monkeypatch, snapshot_kind):
    query_id = f"q-{snapshot_kind}-observation-only"
    _, fragment_id = _install_fte_fragment(query_id, "sink", partitions=1)
    monkeypatch.setattr(
        fte_fragment_scheduler,
        "_sync_write_sink_stage_for_fragment",
        lambda _fragment: (_ for _ in ()).throw(AssertionError("progress collection must not mutate scheduler state")),
    )
    try:
        if snapshot_kind == "progress":
            snapshot = fte_fragment_scheduler.fte_progress_registry_snapshot(query_id)
        else:
            snapshot = fte_fragment_scheduler.fte_registry_stats()
    finally:
        _cleanup_fte_query(query_id)

    assert snapshot["queries"][query_id]["fragment_executions"][fragment_id]


def test_progress_registry_publishes_immutable_native_topology_before_fragment_execution():
    query_id = "q-progress-topology"
    fragment_id = f"{query_id}:node:scan"
    topology = {
        "schema": "pipeline_topology",
        "pipelines": [
            {
                "pipeline_id": 1,
                "operators": ["TABLE_SCAN", "PROJECTION"],
                "operator_details": [{}, {}],
                "stage_ids": [],
            }
        ],
    }
    try:
        build_count = 0

        def build_topology():
            nonlocal build_count
            build_count += 1
            return topology

        assert fte_fragment_scheduler.ensure_fte_fragment_progress_topology(
            query_id,
            fragment_id,
            build_topology,
        )
        topology["pipelines"][0]["operators"].append("FILTER")

        snapshot = fte_fragment_scheduler.fte_progress_registry_snapshot(query_id)
        fragment = snapshot["queries"][query_id]["fragment_executions"][fragment_id]
        published = fragment["progress_topology"]

        assert fragment["partitions"] == {}
        assert fragment["fragment_execution_id"] == 0
        assert fragment["pending_submission_count"] == 0
        assert published["schema"] == "pipeline_topology"
        assert published["pipelines"][0]["operators"] == ["TABLE_SCAN", "PROJECTION"]
        assert build_count == 1

        assert not fte_fragment_scheduler.ensure_fte_fragment_progress_topology(
            query_id,
            fragment_id,
            lambda: (_ for _ in ()).throw(AssertionError("published topology must not be rebuilt")),
        )
        assert build_count == 1
    finally:
        _cleanup_fte_query(query_id)


def test_progress_topology_concurrent_registration_builds_exactly_once():
    query_id = "q-progress-topology-concurrent"
    fragment_id = f"{query_id}:node:scan"
    build_started = threading.Event()
    release_build = threading.Event()
    build_count = 0
    build_count_lock = threading.Lock()
    results: list[bool] = []
    errors: list[BaseException] = []

    def build_topology():
        nonlocal build_count
        with build_count_lock:
            build_count += 1
        build_started.set()
        assert release_build.wait(timeout=2.0)
        return {
            "schema": "pipeline_topology",
            "pipelines": [
                {
                    "pipeline_id": 1,
                    "operators": ["TABLE_SCAN"],
                    "operator_details": [{}],
                    "stage_ids": [],
                }
            ],
        }

    def register_topology():
        try:
            results.append(
                fte_fragment_scheduler.ensure_fte_fragment_progress_topology(
                    query_id,
                    fragment_id,
                    build_topology,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=register_topology) for _ in range(8)]
    try:
        for thread in threads:
            thread.start()
        assert build_started.wait(timeout=2.0)
        release_build.set()
        for thread in threads:
            thread.join(timeout=2.0)

        assert errors == []
        assert all(not thread.is_alive() for thread in threads)
        assert build_count == 1
        assert results.count(True) == 1
        assert results.count(False) == 7
    finally:
        release_build.set()
        _cleanup_fte_query(query_id)


def test_progress_topology_failed_build_releases_ownership_for_retry():
    query_id = "q-progress-topology-retry"
    fragment_id = f"{query_id}:node:scan"

    def fail_build():
        raise RuntimeError("planned topology failure")

    topology = {
        "schema": "pipeline_topology",
        "pipelines": [
            {
                "pipeline_id": 1,
                "operators": ["TABLE_SCAN"],
                "operator_details": [{}],
                "stage_ids": [],
            }
        ],
    }
    try:
        with pytest.raises(RuntimeError, match="planned topology failure"):
            fte_fragment_scheduler.ensure_fte_fragment_progress_topology(
                query_id,
                fragment_id,
                fail_build,
            )

        assert fte_fragment_scheduler.ensure_fte_fragment_progress_topology(
            query_id,
            fragment_id,
            lambda: topology,
        )
        snapshot = fte_fragment_scheduler.fte_progress_registry_snapshot(query_id)
        fragment = snapshot["queries"][query_id]["fragment_executions"][fragment_id]
        assert fragment["progress_topology"] == topology
    finally:
        _cleanup_fte_query(query_id)


def test_progress_registry_does_not_finish_unsealed_fragment():
    query_id = "q-progress-unsealed"
    stage, fragment_id = _install_fte_fragment(query_id, "scan", partitions=1)
    try:
        stage.partitions[0].finished = True
        stage.no_more_partitions = False

        unsealed = fte_fragment_scheduler.fte_progress_registry_snapshot(query_id)
        unsealed_query = unsealed["queries"][query_id]
        unsealed_fragment = unsealed_query["fragment_executions"][fragment_id]
        assert unsealed_query["finished"] is False
        assert unsealed_fragment["finished"] is False
        assert set(unsealed_fragment) == {
            "fragment_id",
            "fragment_execution_id",
            "failed",
            "finished",
            "no_more_partitions",
            "pending_submission_count",
            "partitions",
            "progress_topology",
        }
        assert set(unsealed_fragment["partitions"]["0"]) == {
            "state",
            "running_attempts",
            "selected_output_stats",
        }

        stage.no_more_partitions = True
        sealed = fte_fragment_scheduler.fte_progress_registry_snapshot(query_id)
        sealed_query = sealed["queries"][query_id]
        assert sealed_query["finished"] is True
        assert sealed_query["fragment_executions"][fragment_id]["finished"] is True
    finally:
        _cleanup_fte_query(query_id)


def test_progress_registry_counts_unsubmitted_fte_descriptors_as_pending():
    query_id = "q-progress-pending"
    stage, fragment_id = _install_fte_fragment(query_id, "scan", partitions=3)
    try:
        stage.partitions[0].mark_ready_for_execution()
        stage.partitions[1].defer_ready_for_execution()

        snapshot = fte_fragment_scheduler.fte_progress_registry_snapshot(query_id)
        fragment = snapshot["queries"][query_id]["fragment_executions"][fragment_id]

        assert fragment["pending_submission_count"] == 2
    finally:
        _cleanup_fte_query(query_id)


def test_fte_task_ids_parse_and_format():
    task_id = FteTaskId("query.alpha", 3, 17)
    attempt_id = FteTaskAttemptId(task_id, 2)

    assert str(task_id) == "query.alpha.3.17"
    assert str(attempt_id) == "query.alpha.3.17.2"
    assert FteTaskId.parse(str(task_id)) == task_id
    assert FteTaskAttemptId.parse(str(attempt_id)) == attempt_id
    assert FteTaskAttemptId.from_dict(attempt_id.to_dict()) == attempt_id

    with pytest.raises(ValueError, match="query_id"):
        FteTaskId("", 0, 0)
    with pytest.raises(ValueError, match="attempt_id"):
        FteTaskAttemptId(task_id, -1)


def test_task_descriptor_storage_lifecycle():
    storage = TaskDescriptorStorage()
    task_a = FteTaskId("q1", 1, 0)
    task_b = FteTaskId("q1", 1, 1)
    desc_a = TaskDescriptor(task_a, "q1:node:a", sealed=True)
    desc_b = TaskDescriptor(task_b, "q1:node:b", sealed=True)

    storage.put(task_a, desc_a)
    storage.put(task_b, desc_b)

    assert len(storage) == 2
    assert storage.require("q1.1.0") is desc_a
    assert storage.remove(task_a) is desc_a
    assert storage.get(task_a) is None
    assert storage.destroy_query("q1") == 1
    assert len(storage) == 0

    with pytest.raises(ValueError, match="does not match"):
        storage.put(task_a, desc_b)


def test_fte_task_execution_split_seal_wait_rechecks_after_clear():
    async def _execute(_request):
        return None

    execution = FteTaskExecution(
        {
            "task_id": {
                "query_id": "q",
                "fragment_execution_id": 0,
                "partition_id": 0,
                "attempt_id": 0,
            },
            "worker_runtime": "fte",
            "source_node_ids": ["7"],
        },
        _execute,
        default_task_memory_bytes=1,
    )
    original_clear = execution._split_update_event.clear

    def clear_and_seal():
        execution.no_more_split_sources.add("7")
        execution._split_update_event.set()
        original_clear()

    execution._split_update_event.clear = clear_and_seal

    asyncio.run(asyncio.wait_for(execution._wait_for_fte_splits_sealed(), timeout=0.1))


def test_fte_query_status_uses_fragment_execution_snapshot():
    class _SnapshotOnlyFragmentExecution:
        @property
        def partitions(self):
            raise AssertionError("fte_query_status must use query_status_snapshot")

        def query_status_snapshot(self):
            return {
                "failed": False,
                "partitions": [
                    {
                        "partition_id": 0,
                        "running": False,
                        "failed": False,
                        "finished": True,
                        "selected_attempt": 2,
                        "task_id": "q.0.0",
                        "failures": [],
                    }
                ],
            }

    key = ("q-status-snapshot", "q-status-snapshot:node:scan")
    with _FTE_REGISTRY_LOCK:
        _FTE_FRAGMENT_EXECUTIONS[key] = _SnapshotOnlyFragmentExecution()
    try:
        status = fte_query_status("q-status-snapshot")
    finally:
        with _FTE_REGISTRY_LOCK:
            _FTE_FRAGMENT_EXECUTIONS.pop(key, None)

    assert status["finished"] is True
    assert status["finished_count"] == 1
    assert status["selected_attempt_task_ids"] == ["q.0.0.2"]


def test_fte_query_status_handles_multiple_running_attempts():
    query_id = "q-status-multiple-running"
    fragment_id = f"{query_id}:node:scan"
    stage = _fte_fragment_execution(query_id, 0, fragment_id=fragment_id, max_attempts=2)
    partition = stage.add_partition(0)
    partition.seal()
    partition.start_attempt(worker_id="worker-a")
    loser_attempt = FteTaskAttemptId(partition.task_id, 1)
    partition.running_attempts[loser_attempt.attempt_id] = RunningAttempt(
        loser_attempt,
        worker_id="worker-b",
        remote_handle=None,
    )

    key = (query_id, fragment_id)
    with _FTE_REGISTRY_LOCK:
        _FTE_FRAGMENT_EXECUTIONS[key] = stage
    try:
        status = fte_query_status(query_id)
    finally:
        with _FTE_REGISTRY_LOCK:
            _FTE_FRAGMENT_EXECUTIONS.pop(key, None)

    assert status["running_count"] == 1
    assert status["finished"] is False
    assert status["fragment_executions"][fragment_id]["running_count"] == 1


def test_fte_query_status_reports_scheduler_failure():
    scheduler = _FTE_SCHEDULERS.get_or_create("q-scheduler-failure")
    scheduler.fail("retry timer failed")
    try:
        status = fte_query_status("q-scheduler-failure")
    finally:
        _FTE_SCHEDULERS.drop_query("q-scheduler-failure")

    assert status["failed"] is True
    assert status["finished"] is False
    assert status["scheduler_state"] == "FAILED"
    assert status["scheduler_failure"] == "retry timer failed"


def test_task_descriptor_storage_spills_and_reloads(tmp_path):
    storage = TaskDescriptorStorage(max_in_memory_descriptors=1, spill_dir=tmp_path)
    task_a = FteTaskId("qspill", 1, 0)
    task_b = FteTaskId("qspill", 1, 1)
    desc_a = TaskDescriptor(
        task_a,
        "qspill:node:a",
        initial_splits={"7": [{"sequence_id": 0, "kind": "scan_task", "data": b"a"}]},
    )
    desc_b = TaskDescriptor(task_b, "qspill:node:b")

    storage.put(task_a, desc_a)
    storage.put(task_b, desc_b)

    assert storage.stats() == {
        "in_memory": 1,
        "spilled": 1,
        "total": 2,
        "max_in_memory": 1,
    }
    assert storage.require(task_a).initial_splits["7"][0].data == b"a"
    assert storage.stats()["total"] == 2
    assert storage.destroy_query("qspill") == 2
    assert storage.stats()["total"] == 0
    assert list(tmp_path.iterdir()) == []


def test_task_descriptor_builds_create_task_request():
    task_id = FteTaskId("q", 2, 3)
    descriptor = TaskDescriptor(
        task_id,
        "q:node:scan",
        context={"query_id": "q"},
        initial_splits={
            "7": [{"sequence_id": 0, "kind": "scan_task", "data": b"scan"}],
        },
        no_more_splits={"7"},
        resource_request={"cpus": 1},
    )

    request = descriptor.to_create_task_request(1, exchange_sink_instance={"sink": "i"})

    assert request["task_id"] == {
        "query_id": "q",
        "fragment_execution_id": 2,
        "partition_id": 3,
        "attempt_id": 1,
    }
    assert request["fragment_id"] == "q:node:scan"
    assert request["initial_splits"]["7"][0]["data"] == b"scan"
    assert request["no_more_splits"] == ["7"]
    assert request["exchange_sink_instance"] == {"sink": "i"}


def test_derive_exchange_sink_instance_for_retry_attempt_rewrites_output_location():
    base = {
        "sink_handle": {"task_partition_id": 4, "partition_id": 4},
        "task_partition_id": 4,
        "partition_id": 4,
        "attempt_id": 0,
        "output_partition_count": 8,
        "output_location": "q_shuffle_9__sink_4__attempt_0",
        "attempt_path": "q_shuffle_9__sink_4__attempt_0",
    }

    retry = derive_exchange_sink_instance_for_attempt(base, 2, task_partition_id=9)

    assert retry["sink_handle"]["task_partition_id"] == 9
    assert retry["sink_handle"]["partition_id"] == 9
    assert retry["task_partition_id"] == 9
    assert retry["partition_id"] == 9
    assert retry["attempt_id"] == 2
    assert retry["output_partition_count"] == 8
    assert retry["output_location"] == "q_shuffle_9__sink_9__attempt_2"
    assert retry["attempt_path"] == "q_shuffle_9__sink_9__attempt_2"


def test_task_descriptor_appends_splits_idempotently_and_replays_fte_fields():
    task_id = FteTaskId("q", 2, 3)
    descriptor = TaskDescriptor(
        task_id,
        "q:node:scan",
        source_node_ids={"7"},
        dynamic_scan_source_node_ids={"7"},
    )

    added = descriptor.append_splits(
        "7",
        [
            {"sequence_id": 0, "kind": "scan_task", "data": b"a"},
            {"sequence_id": 0, "kind": "scan_task", "data": b"duplicate"},
            {"sequence_id": 1, "kind": "scan_task", "data": b"b"},
        ],
    )
    assert [split.data for split in added] == [b"a", b"b"]
    assert [split.data for split in descriptor.initial_splits["7"]] == [b"a", b"b"]
    assert descriptor.mark_no_more_splits("7") is True
    assert descriptor.mark_no_more_splits("7") is False

    request = descriptor.to_create_task_request(4)

    assert request["worker_runtime"] == "fte"
    assert request["source_node_ids"] == ["7"]
    assert request["dynamic_scan_source_node_ids"] == ["7"]
    assert request["initial_splits"]["7"][1]["data"] == b"b"
    assert request["no_more_splits"] == ["7"]

    with pytest.raises(RuntimeError, match="already marked no_more_splits"):
        descriptor.append_splits("7", [{"sequence_id": 2, "kind": "scan_task"}])


def test_task_descriptor_applies_task_update_request_subset():
    descriptor = TaskDescriptor(
        FteTaskId("q", 2, 3),
        "q:node:scan",
        context={"query_id": "q"},
        resource_request={"cpus": 1},
        source_node_ids={"7"},
    )

    changed = descriptor.apply_task_update(
        FteTaskUpdateRequest.from_dict(
            {
                "context": {"trace": "t0"},
                "resource_request": {"memory": 32},
                "initial_splits": {
                    "7": [
                        {"sequence_id": 0, "kind": "scan_task", "data": b"a"},
                        {"sequence_id": 0, "kind": "scan_task", "data": b"duplicate"},
                    ],
                },
                "split_assignments": [
                    {
                        "source_node_id": "8",
                        "splits": [{"sequence_id": 0, "kind": "scan_task", "data": b"b"}],
                        "no_more_splits": True,
                    }
                ],
                "output_buffers": {"version": 1, "buffers": ["out-0"]},
                "dynamic_filter_domains": {"df0": {"single_value": 7}},
                "dynamic_scan_source_node_ids": ["8"],
            }
        )
    )

    assert changed is True
    assert descriptor.descriptor_version == 1
    request = descriptor.to_create_task_request(0)
    assert request["context"] == {"query_id": "q", "trace": "t0"}
    assert request["resource_request"] == {"cpus": 1, "memory": 32}
    assert [split["data"] for split in request["initial_splits"]["7"]] == [b"a"]
    assert request["initial_splits"]["8"][0]["data"] == b"b"
    assert request["no_more_splits"] == ["8"]
    assert request["output_buffers"] == {"version": 1, "buffers": ["out-0"]}
    assert request["dynamic_filter_domains"] == {"df0": {"single_value": 7}}
    assert request["dynamic_scan_source_node_ids"] == ["8"]


def test_task_update_request_normalizes_output_ids_aliases():
    update = FteTaskUpdateRequest.from_dict(
        {
            "outputIds": {
                "version": "4",
                "@type": "PIPELINED",
                "buffers": {"out-0": 0},
                "noMoreBufferIds": True,
            }
        }
    )

    assert update.output_buffers == {
        "version": 4,
        "type": "pipelined",
        "buffers": {"out-0": 0},
        "no_more_buffer_ids": True,
    }
    assert update.to_dict()["output_buffers"] == update.output_buffers
    open_update = FteTaskUpdateRequest.from_dict({"output_buffers": {"version": 5, "sealed": "false"}})
    assert open_update.output_buffers == {"version": 5}


def test_task_descriptor_output_buffers_follow_versioned_lifecycle():
    descriptor = TaskDescriptor(
        FteTaskId("q", 2, 4),
        "q:node:scan",
        output_buffers={"version": 1, "buffers": {"out-0": 0}},
    )

    assert descriptor.apply_task_update({"output_buffers": {"version": 0, "buffers": {"old": 0}}}) is False
    assert descriptor.descriptor_version == 0
    assert descriptor.output_buffers == {"version": 1, "buffers": {"out-0": 0}}

    assert descriptor.apply_task_update({"output_buffers": {"version": 2, "buffers": {"out-0": 0, "out-1": 1}}}) is True
    assert descriptor.descriptor_version == 1
    assert descriptor.output_buffers == {
        "version": 2,
        "buffers": {"out-0": 0, "out-1": 1},
    }

    assert (
        descriptor.apply_task_update({"output_buffers": {"version": 2, "buffers": {"out-0": 0, "out-1": 1}}}) is False
    )
    with pytest.raises(ValueError, match="conflicting"):
        descriptor.apply_task_update({"output_buffers": {"version": 2, "buffers": {"out-0": 0, "out-2": 2}}})
    with pytest.raises(ValueError, match="assignment"):
        descriptor.apply_task_update({"output_buffers": {"version": 3, "buffers": {"out-0": 9, "out-1": 1}}})
    with pytest.raises(ValueError, match="remove"):
        descriptor.apply_task_update({"output_buffers": {"version": 3, "buffers": {"out-0": 0}}})

    assert (
        descriptor.apply_task_update(
            {
                "output_buffers": {
                    "version": 3,
                    "buffers": {"out-0": 0, "out-1": 1},
                    "noMoreBufferIds": True,
                }
            }
        )
        is True
    )
    assert descriptor.output_buffers["no_more_buffer_ids"] is True
    with pytest.raises(ValueError, match="sealed"):
        descriptor.apply_task_update(
            {
                "output_buffers": {
                    "version": 4,
                    "buffers": {"out-0": 0, "out-1": 1, "out-2": 2},
                    "noMoreBufferIds": True,
                }
            }
        )


def test_task_descriptor_spooling_output_buffers_keep_partition_count_stable():
    descriptor = TaskDescriptor(
        FteTaskId("q", 2, 5),
        "q:node:scan",
        output_buffers={
            "version": 1,
            "@type": "SPOOLING",
            "outputPartitionCount": 4,
            "exchangeSinkInstanceHandle": {"id": "a"},
        },
    )

    assert descriptor.output_buffers == {
        "version": 1,
        "type": "spooling",
        "output_partition_count": 4,
        "exchange_sink_instance": {"id": "a"},
    }
    assert (
        descriptor.apply_task_update(
            {
                "outputIds": {
                    "version": 2,
                    "type": "spooling",
                    "output_partition_count": 4,
                    "exchange_sink_instance": {"id": "b"},
                }
            }
        )
        is True
    )
    assert descriptor.output_buffers["exchange_sink_instance"] == {"id": "b"}
    with pytest.raises(ValueError, match="output_partition_count"):
        descriptor.apply_task_update(
            {
                "output_buffers": {
                    "version": 3,
                    "type": "spooling",
                    "output_partition_count": 5,
                    "exchange_sink_instance": {"id": "c"},
                }
            }
        )


def test_fte_exchange_tracker_selects_first_successful_attempt_only():
    exchange = FteExchangeTracker("q", "stage-0")
    sink = exchange.add_sink(0)

    instance0 = exchange.instantiate_sink(sink, 0)
    instance1 = exchange.instantiate_sink(sink, 1)
    exchange.sink_finished(sink, instance1.attempt_id)
    exchange.sink_finished(sink, instance0.attempt_id)

    assert exchange.selected_attempt(sink) == 1
    assert exchange.output_selector.selected_attempt(0) == 1
    assert exchange.is_final() is False
    assert [handle.to_dict() for handle in exchange.get_source_handles()] == [
        {
            "sink_handle": {
                "query_id": "q",
                "exchange_id": "stage-0",
                "partition_id": 0,
            },
            "attempt_id": 1,
        }
    ]


def test_fte_exchange_source_output_selector_finalizes_only_after_required_partitions():
    selector = FteExchangeSourceOutputSelector()

    assert selector.record_finished(0, 1) is True
    assert selector.record_finished(0, 0) is False
    selector.record_aborted(1, 0)

    assert selector.selected_attempt(0) == 1
    assert selector.selected_attempt(1) is None
    assert selector.try_mark_final({0, 1}) is False
    assert selector.is_final() is False

    assert selector.record_finished(1, 2) is True
    assert selector.try_mark_final({0, 1}) is True
    assert selector.is_final() is True

    with pytest.raises(RuntimeError, match="selector is final"):
        selector.record_finished(2, 0)


def test_fte_exchange_source_output_selector_ignores_late_success_after_final():
    selector = FteExchangeSourceOutputSelector()

    assert selector.record_finished(0, 1) is True
    assert selector.record_finished(1, 2) is True
    assert selector.try_mark_final({0, 1}) is True

    assert selector.record_finished(0, 7) is False
    assert selector.record_finished(1, 2) is False
    selector.record_aborted(1, 8)
    assert selector.selected_attempt(0) == 1
    assert selector.selected_attempt(1) == 2
    assert selector.try_mark_final({1, 0}) is False

    with pytest.raises(RuntimeError, match="different required partitions"):
        selector.try_mark_final({0, 1, 2})
    with pytest.raises(RuntimeError, match="selector is final"):
        selector.record_finished(2, 0)


def test_fte_exchange_source_output_selector_rejects_subset_final_after_extra_success():
    selector = FteExchangeSourceOutputSelector()

    assert selector.record_finished(0, 1) is True
    assert selector.record_finished(1, 3) is True

    with pytest.raises(RuntimeError, match="unrequired selected partitions"):
        selector.try_mark_final({0})

    assert selector.is_final() is False
    assert selector.selected_attempt(0) == 1
    assert selector.selected_attempt(1) == 3
    assert selector.try_mark_final({0, 1}) is True


def test_fte_exchange_source_output_selector_deduplicates_required_partitions_and_attempts():
    selector = FteExchangeSourceOutputSelector()

    selector.record_aborted(0, 0)
    assert selector.record_finished(0, 2) is True
    assert selector.record_finished(0, 2) is False
    assert selector.record_finished(0, 1) is False
    assert selector.selected_attempt(0) == 2

    assert selector.record_finished(1, 0) is True
    assert selector.try_mark_final([0, 1, 1, 0]) is True
    assert selector.try_mark_final((1, 0)) is False

    assert selector.record_finished(0, 9) is False
    assert selector.record_finished(1, 7) is False
    assert selector.selected_attempt(0) == 2
    assert selector.selected_attempt(1) == 0


def test_fte_exchange_source_output_selector_rejects_empty_final_with_selected_partition():
    selector = FteExchangeSourceOutputSelector()

    assert selector.try_mark_final(set()) is False
    assert selector.record_finished(0, 0) is True

    with pytest.raises(RuntimeError, match="unrequired selected partitions"):
        selector.try_mark_final(set())

    assert selector.is_final() is False
    assert selector.selected_attempt(0) == 0


def test_hash_exchange_selector_final_waits_for_all_sources():
    assigner = HashSplitAssigner(
        source_partition_count=2,
        partitioned_sources={"3", "4"},
        source_partition_to_task_partition=HashSplitAssigner.one_task_per_source_partition(2),
    )

    first_source = assigner.assign(
        "3",
        [
            {
                "sequence_id": 0,
                "kind": "exchange_source_task",
                "source_partition_id": 0,
                "data": "s3-p0",
            }
        ],
        no_more_inputs=True,
    )

    assert sorted(info.partition_id for info in first_source.partitions_added) == [0, 1]
    assert first_source.sealed_partitions == []
    assert {
        (update.partition_id, update.source_node_id)
        for update in first_source.partition_updates
        if update.no_more_splits
    } == {(0, "3"), (1, "3")}

    second_source = assigner.assign(
        "4",
        [
            {
                "sequence_id": 0,
                "kind": "exchange_source_task",
                "source_partition_id": 0,
                "data": "s4-p0",
            },
            {
                "sequence_id": 1,
                "kind": "exchange_source_task",
                "source_partition_id": 1,
                "data": "s4-p1",
            },
        ],
        no_more_inputs=True,
    )

    assert sorted(second_source.sealed_partitions) == [0, 1]
    assert {
        (update.partition_id, update.source_node_id)
        for update in second_source.partition_updates
        if update.no_more_splits
    } == {(0, "4"), (1, "4")}


def test_fte_exchange_tracker_rejects_new_sinks_after_final():
    exchange = FteExchangeTracker("q", "stage-final")
    sink = exchange.add_sink(0)

    exchange.sink_finished(sink, 0)
    assert exchange.finalize() is True
    assert exchange.finalize() is False

    with pytest.raises(RuntimeError, match="final"):
        exchange.add_sink(1)

    handles = exchange.get_source_handles()
    assert len(handles) == 1
    assert handles[0].sink_handle.partition_id == 0
    assert handles[0].attempt_id == 0


def test_fte_exchange_tracker_late_success_after_final_keeps_source_handles_stable():
    exchange = FteExchangeTracker("q", "stage-late")
    sink0 = exchange.add_sink(0)
    sink1 = exchange.add_sink(1)

    exchange.sink_finished(sink0, 2)
    exchange.sink_finished(sink1, 0)
    assert exchange.finalize() is True
    initial_handles = [handle.to_dict() for handle in exchange.get_source_handles()]

    assert exchange.add_sink(0) == sink0
    exchange.sink_finished(sink0, 9)
    exchange.sink_finished(sink1, 8)
    exchange.sink_aborted(sink1, 10)

    assert exchange.selected_attempt(sink0) == 2
    assert exchange.selected_attempt(sink1) == 0
    assert [handle.to_dict() for handle in exchange.get_source_handles()] == initial_handles
    assert exchange.finalize() is False


def test_spooling_exchange_manager_late_success_after_final_does_not_replace_selected_files(tmp_path):
    exchange = SpoolingExchangeManager(tmp_path, "q", "stage-late-spool")
    sink = exchange.add_sink(0)
    selected = exchange.instantiate_sink(sink, 0)
    late = exchange.instantiate_sink(sink, 1)
    selected_path = exchange.record_output_file(selected, 0, "selected", b"selected")
    late_path = exchange.record_output_file(late, 0, "late", b"late")

    exchange.finish_attempt(selected)
    exchange.sink_finished(sink, 0)
    assert exchange.finalize() is True

    exchange.finish_attempt(late)
    exchange.sink_finished(sink, 1)
    handles = exchange.get_source_handles()

    assert len(handles) == 1
    assert handles[0].attempt_id == 0
    assert handles[0].files == (selected_path,)
    assert late_path not in handles[0].files


def test_spooling_exchange_manager_writes_markers_and_selected_source_handles(tmp_path):
    exchange = SpoolingExchangeManager(tmp_path, "q", "stage-0")
    sink = exchange.add_sink(0)

    attempt0 = exchange.instantiate_sink(sink, 0)
    attempt1 = exchange.instantiate_sink(sink, 1)
    path0 = exchange.record_output_file(attempt0, 0, "a", b"losing")
    path1 = exchange.record_output_file(attempt1, 0, "b", b"selected")
    exchange.finish_attempt(attempt0)
    exchange.finish_attempt(attempt1)

    exchange.sink_finished(sink, 1)
    exchange.sink_finished(sink, 0)
    handles = exchange.get_source_handles()

    assert exchange.selected_attempt(sink) == 1
    assert len(handles) == 1
    assert handles[0].attempt_id == 1
    assert handles[0].files == (path1,)
    assert handles[0].attempt_path == attempt1.attempt_path
    assert (tmp_path / "q" / "stage-0" / "sink_0" / "attempt_1" / "committed").exists()
    assert (tmp_path / "q" / "stage-0" / "sink_0" / "attempt_1" / "manifest.json").exists()
    assert path0 != path1


def test_collect_spooling_output_stats_from_attempt_path(tmp_path):
    exchange = SpoolingExchangeManager(tmp_path, "q", "stage-0")
    sink = exchange.add_sink(1)
    attempt = exchange.instantiate_sink(sink, 3)
    path0 = exchange.record_output_file(attempt, 0, "a", b"aaa")
    path1 = exchange.record_output_file(attempt, 2, "b", b"bbbbb")
    exchange.finish_attempt(attempt)

    stats = collect_spooling_output_stats(attempt)

    assert stats is not None
    assert stats["attempt_path"] == attempt.attempt_path
    assert stats["committed"] is True
    assert stats["aborted"] is False
    assert stats["file_count"] == 2
    assert stats["total_bytes"] == 8
    assert stats["partitions"] == {
        "0": {"file_count": 1, "total_bytes": 3},
        "2": {"file_count": 1, "total_bytes": 5},
    }
    assert [entry["path"] for entry in stats["files"]] == [path0, path1]


def test_spooling_exchange_manager_abort_cleanup_and_destroy_query(tmp_path):
    exchange = SpoolingExchangeManager(tmp_path, "q", "stage-0")
    sink = exchange.add_sink(2)
    attempt0 = exchange.instantiate_sink(sink, 0)
    attempt1 = exchange.instantiate_sink(sink, 1)
    exchange.record_output_file(attempt0, 0, "old", b"old")
    exchange.record_output_file(attempt1, 0, "new", b"new")

    exchange.sink_aborted(sink, 0)
    exchange.finish_attempt(attempt1)
    exchange.sink_finished(sink, 1)

    attempt0_dir = tmp_path / "q" / "stage-0" / "sink_2" / "attempt_0"
    attempt1_dir = tmp_path / "q" / "stage-0" / "sink_2" / "attempt_1"
    assert (attempt0_dir / "aborted").exists()
    assert attempt1_dir.exists()

    assert exchange.cleanup_unselected_attempts() == 1
    assert not attempt0_dir.exists()
    assert attempt1_dir.exists()

    exchange.destroy_query()
    assert not (tmp_path / "q").exists()


def test_spooling_exchange_manager_removes_committed_but_unselected_attempt(tmp_path):
    exchange = SpoolingExchangeManager(tmp_path, "q", "stage-0")
    sink = exchange.add_sink(0)
    attempt0 = exchange.instantiate_sink(sink, 0)
    attempt1 = exchange.instantiate_sink(sink, 1)
    exchange.record_output_file(attempt0, 0, "old", b"old")
    selected_path = exchange.record_output_file(attempt1, 0, "new", b"new")
    exchange.finish_attempt(attempt0)
    exchange.finish_attempt(attempt1)

    exchange.sink_finished(sink, 1)
    handles = exchange.get_source_handles()

    assert len(handles) == 1
    assert handles[0].attempt_id == 1
    assert handles[0].files == (selected_path,)
    assert (tmp_path / "q" / "stage-0" / "sink_0" / "attempt_0" / "committed").exists()
    assert exchange.cleanup_unselected_attempts() == 1
    assert not (tmp_path / "q" / "stage-0" / "sink_0" / "attempt_0").exists()
    assert (tmp_path / "q" / "stage-0" / "sink_0" / "attempt_1").exists()


def test_spooling_exchange_manager_duplicate_successful_attempts_select_one(tmp_path):
    exchange = SpoolingExchangeManager(tmp_path, "q", "stage-0")
    sink = exchange.add_sink(0)
    attempt0 = exchange.instantiate_sink(sink, 0)
    attempt1 = exchange.instantiate_sink(sink, 1)
    path0 = exchange.record_output_file(attempt0, 0, "first", b"first")
    path1 = exchange.record_output_file(attempt1, 0, "duplicate", b"duplicate")
    exchange.finish_attempt(attempt0)
    exchange.finish_attempt(attempt1)

    exchange.sink_finished(sink, 0)
    exchange.sink_finished(sink, 1)
    handles = exchange.get_source_handles()

    assert exchange.selected_attempt(sink) == 0
    assert len(handles) == 1
    assert handles[0].attempt_id == 0
    assert handles[0].files == (path0,)
    assert path1 not in handles[0].files


def test_two_stage_query_dag_worker_loss_uses_selected_attempt_for_final_result(tmp_path):
    exchange = SpoolingExchangeManager(tmp_path, "q", "stage-upstream")
    upstream = _fte_fragment_execution(
        "q",
        11,
        fragment_id="q:node:upstream",
        worker_selector=lambda partition: _FakeLiveWorker(
            "worker-a" if partition.next_attempt_number() == 0 else "worker-b"
        ),
        max_attempts=2,
        exchange=exchange,
        source_node_ids={"7"},
        dynamic_scan_source_node_ids={"7"},
    )

    scheduled0 = upstream.apply_assignment_result(
        AssignmentResult(
            partitions_added=[PartitionInfo(0)],
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [{"sequence_id": 0, "kind": "scan_task", "data": b"input"}],
                    ready_for_scheduling=True,
                ),
                PartitionUpdate(0, "7", no_more_splits=True),
            ],
        )
    )[0]
    lost_path = exchange.record_output_file(scheduled0.sink_instance, 0, "lost", b"999\n")
    exchange.finish_attempt(scheduled0.sink_instance)

    retries = upstream.mark_worker_failed("worker-a", "host lost before status was observed")
    assert len(retries) == 1
    retry = retries[0]
    selected_path = exchange.record_output_file(retry.sink_instance, 0, "selected", b"10\n20\n")
    exchange.finish_attempt(retry.sink_instance)
    assert upstream.task_finished(retry.attempt_id, {"rows": 2}) is True

    assert upstream.partitions and all(partition.finished for partition in upstream.partitions.values())
    assert exchange.finalize() is True
    selected_handles = exchange.get_source_handles()
    downstream_payload = {
        "partition_indices": [0],
        "source_handles": [handle.to_dict() for handle in selected_handles],
    }
    downstream_context = materialize_task_inputs(
        {},
        {
            "17": [
                {
                    "sequence_id": 0,
                    "kind": "exchange_source_task",
                    "data": downstream_payload,
                }
            ]
        },
    )

    handles = downstream_context["exchange_source_task:17"]["source_handles"]
    final_values = [
        int(line)
        for handle in handles
        for file_path in handle["files"]
        for line in Path(file_path).read_text().splitlines()
    ]

    assert handles[0]["attempt_id"] == 1
    assert handles[0]["files"] == [selected_path]
    assert lost_path not in handles[0]["files"]
    assert sum(final_values) == 30
    assert 999 not in final_values
    assert Path(lost_path).exists()
    assert exchange.cleanup_unselected_attempts() == 1
    assert not Path(scheduled0.sink_instance.attempt_path).exists()
    assert Path(retry.sink_instance.attempt_path).exists()


class _FakeLiveWorker:
    def __init__(self, worker_id="worker-a"):
        self.worker_id = worker_id
        self.calls = []
        self.statuses = {}

    def fte_create_task(self, request):
        self.calls.append(("create", request))
        status = {"state": "RUNNING", "task_id": request["task_id"]}
        self.statuses[self._key(request["task_id"])] = status
        return status

    def fte_add_splits(self, task_id, source_node_id, splits):
        self.calls.append(("add", task_id, source_node_id, splits))
        return {"state": "RUNNING", "task_id": task_id}

    def fte_no_more_splits(self, task_id, source_node_id):
        self.calls.append(("no_more", task_id, source_node_id))
        return {"state": "RUNNING", "task_id": task_id}

    def fte_update_task(self, task_id, update):
        self.calls.append(("update", task_id, update))
        return {"state": "RUNNING", "task_id": task_id, "version": 5}

    def enqueue_fte_add_splits(self, task_id, source_node_id, splits):
        return self.fte_add_splits(task_id, source_node_id, splits)

    def enqueue_fte_no_more_splits(self, task_id, source_node_id):
        return self.fte_no_more_splits(task_id, source_node_id)

    def enqueue_fte_update_task(self, task_id, update):
        return self.fte_update_task(task_id, update)

    def ensure_fragment_registered(self, _query_id, _fragment_id, _fragment_plan):
        return {"registered": 1, "existing": 0}

    def fte_wait_split_queue_has_space(
        self,
        _task_id,
        _source_node_id=None,
        _max_buffered_splits=None,
        _timeout_s=None,
    ):
        return {"has_space": True, "buffered_splits": 0}

    def fte_cancel_task(self, task_id):
        self.calls.append(("cancel", task_id))
        return {"state": "CANCELED", "task_id": task_id}

    def set_fte_task_execution_class(self, task_id, execution_class):
        self.calls.append(
            (
                "set_class",
                FteTaskAttemptId.coerce(task_id).to_dict(),
                getattr(execution_class, "value", str(execution_class)),
            )
        )
        return True

    def fte_get_task_status(self, task_id):
        self.calls.append(("status", task_id))
        return self.statuses[self._key(task_id)]

    def record_fte_task_started(self, _attempt_id, _request):
        return None

    def record_fte_splits_added(self, _attempt_id, _split_count):
        return None

    def record_fte_split_bytes_added(self, _attempt_id, _split_bytes):
        return None

    def record_fte_task_terminal(self, _attempt_id):
        return None

    def record_fte_task_result_ready(self, attempt_id):
        # Unit workers model immediate result adoption by the driver handle.
        return self.record_fte_task_terminal(attempt_id)

    def set_status(self, task_id, state, **extra):
        status = {"state": state, "task_id": task_id}
        status.update(extra)
        self.statuses[self._key(task_id)] = status

    @staticmethod
    def _key(task_id):
        return str(FteTaskAttemptId.coerce(task_id))


def _execute_stage_commands(stage, result=None, executor=None):
    executor = executor or FteWorkerCommandExecutor()
    commands = list(getattr(result, "worker_commands", ()) or ())
    if not commands:
        commands = stage.pop_worker_commands()
    for command in commands:
        try:
            executor.execute(command)
        except Exception as exc:
            raise stage.worker_control_failure_for_command(command, exc) from exc
        stage.handle_worker_command_success(command)


def _handle_live_worker_statuses(stage, *, retryable=True):
    scheduled = []
    with stage._state_lock:
        running_attempts = [
            running for partition in stage.partitions.values() for running in partition.running_attempts.values()
        ]
    for running in running_attempts:
        status = running.remote_handle.fte_get_task_status(running.attempt_id.to_dict())
        retry = stage.handle_task_status(status, retryable=retryable)
        if retry is not None:
            scheduled.append(retry)
    return scheduled


def test_fte_worker_command_executor_requires_split_queue_wait_protocol():
    attempt_id = FteTaskAttemptId(FteTaskId("q", 0, 1), 0)
    command = FteAddSplitsCommand(
        query_id="q",
        fragment_id="q:node:scan",
        worker_id="worker-a",
        worker=object(),
        attempt_id=attempt_id,
        source_node_id="7",
        splits=({"sequence_id": 1, "kind": "scan_task", "data": b"a"},),
    )

    with pytest.raises(RuntimeError, match="fte_wait_split_queue_has_space"):
        FteWorkerCommandExecutor().execute(command)


def test_fte_worker_command_executor_requires_single_ordered_enqueue_protocol():
    class _Worker:
        def fte_wait_split_queue_has_space(self, *_args):
            return {"has_space": True}

        def fte_add_splits(self, _task_id, _source_node_id, _splits):
            raise AssertionError("ordered control must use enqueue_fte_add_splits")

    attempt_id = FteTaskAttemptId(FteTaskId("q", 0, 1), 0)
    command = FteAddSplitsCommand(
        query_id="q",
        fragment_id="q:node:scan",
        worker_id="worker-a",
        worker=_Worker(),
        attempt_id=attempt_id,
        source_node_id="7",
        splits=({"sequence_id": 1, "kind": "scan_task", "data": b"a"},),
    )

    with pytest.raises(RuntimeError, match="enqueue_fte_add_splits"):
        FteWorkerCommandExecutor().execute(command)


def test_fte_fragment_execution_requires_fragment_registration_protocol():
    class _Worker:
        worker_id = "worker-a"

    stage = _fte_fragment_execution(
        "q",
        31,
        fragment_id="q:node:scan",
        worker=_Worker(),
        worker_id="worker-a",
    )

    with pytest.raises(AttributeError, match="ensure_fragment_registered"):
        stage.apply_assignment_result(
            AssignmentResult(
                partitions_added=[PartitionInfo(0)],
                partition_updates=[
                    PartitionUpdate(
                        0,
                        "7",
                        [{"sequence_id": 0, "kind": "scan_task", "data": b"a"}],
                        ready_for_scheduling=True,
                    )
                ],
            )
        )


def test_fte_fragment_execution_does_not_probe_a_sync_control_fallback():
    class _Worker:
        worker_id = "worker-a"

        def ensure_fragment_registered(self, _query_id, _fragment_id, _fragment_plan):
            return None

    stage = _fte_fragment_execution(
        "q",
        32,
        fragment_id="q:node:scan",
        worker=_Worker(),
        worker_id="worker-a",
        source_node_ids={"7"},
    )
    stage.apply_assignment_result(
        AssignmentResult(
            partitions_added=[PartitionInfo(0)],
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [{"sequence_id": 0, "kind": "scan_task", "data": b"a"}],
                    ready_for_scheduling=True,
                )
            ],
        )
    )

    result = stage.apply_assignment_result(
        AssignmentResult(
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [{"sequence_id": 1, "kind": "scan_task", "data": b"b"}],
                )
            ]
        )
    )
    assert [command.command_type for command in result.worker_commands] == ["FteAddSplitsCommand"]


def test_fte_worker_selection_requires_pressure_stats_protocol():
    class _Worker:
        worker_id = "worker-a"

    with pytest.raises(AttributeError, match="fte_pressure_stats"):
        fte_fragment_scheduler._fte_worker_selection_key(_Worker())


def test_fte_fragment_execution_assignment_creates_task_and_sends_later_updates():
    worker = _FakeLiveWorker()
    stage = _fte_fragment_execution(
        "q",
        3,
        fragment_id="q:node:scan",
        worker=worker,
        worker_id=worker.worker_id,
        source_node_ids={"7"},
        dynamic_scan_source_node_ids={"7"},
    )

    scheduled = stage.apply_assignment_result(AssignmentResult(partitions_added=[PartitionInfo(0)]))
    assert scheduled == []
    assert worker.calls == []

    scheduled = stage.apply_assignment_result(
        AssignmentResult(
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [{"sequence_id": 0, "kind": "scan_task", "data": b"a"}],
                    ready_for_scheduling=True,
                )
            ]
        )
    )

    assert len(scheduled) == 1
    assert [command.command_type for command in scheduled.worker_commands] == ["FteCreateTaskCommand"]
    assert worker.calls == []
    _execute_stage_commands(stage, scheduled)
    create_call = worker.calls[0]
    assert create_call[0] == "create"
    assert create_call[1]["task_id"]["attempt_id"] == 0
    assert create_call[1]["worker_runtime"] == "fte"
    assert create_call[1]["source_node_ids"] == ["7"]
    assert create_call[1]["dynamic_scan_source_node_ids"] == ["7"]
    assert create_call[1]["initial_splits"]["7"][0]["data"] == b"a"

    scheduled = stage.apply_assignment_result(
        AssignmentResult(
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [{"sequence_id": 1, "kind": "scan_task", "data": b"b"}],
                    ready_for_scheduling=True,
                )
            ]
        )
    )

    assert scheduled == []
    assert [command.command_type for command in scheduled.worker_commands] == ["FteAddSplitsCommand"]
    _execute_stage_commands(stage, scheduled)
    assert worker.calls[1][0] == "add"
    assert worker.calls[1][1]["attempt_id"] == 0
    assert worker.calls[1][2] == "7"
    assert worker.calls[1][3][0]["data"] == b"b"
    descriptor = stage.descriptor_storage.require(FteTaskId("q", 3, 0))
    assert [split.data for split in descriptor.initial_splits["7"]] == [b"a", b"b"]


def test_fte_fragment_execution_uses_worker_command_executor_for_create_and_updates():
    class _RecordingExecutor(FteWorkerCommandExecutor):
        def __init__(self):
            self.commands = []

        def create_task(self, command):
            self.commands.append(command)
            return {"state": "RUNNING", "task_id": command.request["task_id"]}

        def wait_split_queue_has_space(self, command):
            self.commands.append(("wait", command))

        def add_splits(self, command):
            self.commands.append(command)
            return {"state": "RUNNING", "task_id": command.attempt_id.to_dict()}

        def no_more_splits(self, command):
            self.commands.append(command)
            return {"state": "RUNNING", "task_id": command.attempt_id.to_dict()}

        def update_task(self, command):
            self.commands.append(command)
            return {"state": "RUNNING", "task_id": command.attempt_id.to_dict()}

    worker = _FakeLiveWorker()
    executor = _RecordingExecutor()
    stage = _fte_fragment_execution(
        "q",
        32,
        fragment_id="q:node:scan",
        worker=worker,
        worker_id=worker.worker_id,
        source_node_ids={"7"},
        dynamic_scan_source_node_ids={"7"},
    )

    scheduled_result = stage.apply_assignment_result(
        AssignmentResult(
            partitions_added=[PartitionInfo(0)],
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [{"sequence_id": 0, "kind": "scan_task", "data": b"a"}],
                    ready_for_scheduling=True,
                )
            ],
        )
    )
    scheduled = scheduled_result

    assert len(scheduled) == 1
    assert worker.calls == []
    assert executor.commands == []
    _execute_stage_commands(stage, scheduled, executor)
    assert [command.command_type for command in executor.commands] == ["FteCreateTaskCommand"]
    create_command = executor.commands[0]
    assert create_command.attempt_id == scheduled[0].attempt_id
    assert create_command.request["initial_splits"]["7"][0]["data"] == b"a"

    executor.commands.clear()
    scheduled = stage.apply_assignment_result(
        AssignmentResult(
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [{"sequence_id": 1, "kind": "scan_task", "data": b"b"}],
                    no_more_splits=True,
                )
            ]
        )
    )

    assert scheduled == []
    assert worker.calls == []
    _execute_stage_commands(stage, scheduled, executor)
    assert [item[0] if isinstance(item, tuple) else item.command_type for item in executor.commands] == [
        "wait",
        "FteAddSplitsCommand",
        "FteNoMoreSplitsCommand",
    ]
    add_command = executor.commands[1]
    no_more_command = executor.commands[2]
    assert add_command.source_node_id == "7"
    assert add_command.splits[0]["data"] == b"b"
    assert no_more_command.source_node_id == "7"


def test_fte_fragment_execution_task_update_before_create_is_replayed_in_create_request():
    worker = _FakeLiveWorker()
    stage = _fte_fragment_execution(
        "q",
        34,
        fragment_id="q:node:scan",
        worker=worker,
        worker_id=worker.worker_id,
        source_node_ids={"7"},
    )

    result = stage.apply_task_update(
        0,
        {
            "output_buffers": {"version": 2, "buffers": ["out-0"]},
            "dynamic_filter_domains": {"df0": {"range": [1, 3]}},
            "context": {"trace_token": "abc"},
            "initial_splits": {
                "7": [{"sequence_id": 0, "kind": "scan_task", "data": b"a"}],
            },
        },
    )

    assert result == []
    assert result.worker_commands == ()
    descriptor = stage.descriptor_storage.require(FteTaskId("q", 34, 0))
    assert descriptor.output_buffers == {"version": 2, "buffers": ["out-0"]}
    assert descriptor.dynamic_filter_domains == {"df0": {"range": [1, 3]}}

    scheduled = stage.seal_partition(0)
    _execute_stage_commands(stage)

    assert scheduled is not None
    create = worker.calls[0][1]
    assert create["context"]["trace_token"] == "abc"
    assert create["initial_splits"]["7"][0]["data"] == b"a"
    assert create["output_buffers"] == {"version": 2, "buffers": ["out-0"]}
    assert create["dynamic_filter_domains"] == {"df0": {"range": [1, 3]}}


def test_fte_fragment_execution_task_update_running_attempt_uses_update_command():
    worker = _FakeLiveWorker()
    stage = _fte_fragment_execution(
        "q",
        35,
        fragment_id="q:node:scan",
        worker=worker,
        worker_id=worker.worker_id,
        source_node_ids={"7"},
        dynamic_scan_source_node_ids={"7"},
    )
    scheduled = stage.apply_assignment_result(
        AssignmentResult(
            partitions_added=[PartitionInfo(0)],
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [{"sequence_id": 0, "kind": "scan_task", "data": b"a"}],
                    ready_for_scheduling=True,
                )
            ],
        )
    )
    _execute_stage_commands(stage, scheduled)

    update_result = stage.apply_task_update(
        0,
        {
            "output_buffers": {"version": 3, "buffers": ["out-1"]},
            "dynamic_filter_domains": {"df1": {"single_value": 9}},
            "initial_splits": {
                "7": [{"sequence_id": 1, "kind": "scan_task", "data": b"b"}],
            },
        },
    )

    assert update_result == []
    assert [command.command_type for command in update_result.worker_commands] == ["FteTaskUpdateCommand"]
    _execute_stage_commands(stage, update_result)

    assert worker.calls[-1][0] == "update"
    assert worker.calls[-1][2]["output_buffers"] == {"version": 3, "buffers": ["out-1"]}
    assert worker.calls[-1][2]["dynamic_filter_domains"] == {"df1": {"single_value": 9}}
    assert worker.calls[-1][2]["initial_splits"]["7"][0]["data"] == b"b"
    descriptor = stage.descriptor_storage.require(FteTaskId("q", 35, 0))
    assert [split.data for split in descriptor.initial_splits["7"]] == [b"a", b"b"]


def test_fte_fragment_execution_ignores_stale_output_buffer_update_for_running_attempt():
    worker = _FakeLiveWorker()
    stage = _fte_fragment_execution(
        "q",
        36,
        fragment_id="q:node:scan",
        worker=worker,
        worker_id=worker.worker_id,
        source_node_ids={"7"},
        dynamic_scan_source_node_ids={"7"},
    )
    scheduled = stage.apply_assignment_result(
        AssignmentResult(
            partitions_added=[PartitionInfo(0)],
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [{"sequence_id": 0, "kind": "scan_task", "data": b"a"}],
                    ready_for_scheduling=True,
                )
            ],
        )
    )
    _execute_stage_commands(stage, scheduled)

    fresh_update = stage.apply_task_update(
        0,
        {"output_buffers": {"version": 2, "buffers": ["out-2"]}},
    )
    _execute_stage_commands(stage, fresh_update)
    call_count = len(worker.calls)

    stale_update = stage.apply_task_update(
        0,
        {"output_buffers": {"version": 1, "buffers": ["out-1"]}},
    )

    assert stale_update.worker_commands == ()
    assert len(worker.calls) == call_count
    descriptor = stage.descriptor_storage.require(FteTaskId("q", 36, 0))
    assert descriptor.output_buffers == {"version": 2, "buffers": ["out-2"]}


def test_fte_fragment_execution_appends_descriptor_before_add_splits_command_failure():
    class _FailingAddExecutor(FteWorkerCommandExecutor):
        def create_task(self, command):
            return {"state": "RUNNING", "task_id": command.request["task_id"]}

        def wait_split_queue_has_space(self, _command):
            return None

        def add_splits(self, _command):
            raise RuntimeError("command failed")

    worker = _FakeLiveWorker()
    stage = _fte_fragment_execution(
        "q",
        33,
        fragment_id="q:node:scan",
        worker=worker,
        worker_id=worker.worker_id,
        source_node_ids={"7"},
        dynamic_scan_source_node_ids={"7"},
    )

    scheduled_result = stage.apply_assignment_result(
        AssignmentResult(
            partitions_added=[PartitionInfo(0)],
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [{"sequence_id": 0, "kind": "scan_task", "data": b"a"}],
                    ready_for_scheduling=True,
                )
            ],
        )
    )
    scheduled = scheduled_result

    assert len(scheduled) == 1
    _execute_stage_commands(stage, scheduled)
    update_result = stage.apply_assignment_result(
        AssignmentResult(
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [{"sequence_id": 1, "kind": "scan_task", "data": b"b"}],
                )
            ]
        )
    )
    with pytest.raises(FteWorkerControlFailure, match="fte_add_splits"):
        _execute_stage_commands(stage, update_result, _FailingAddExecutor())

    descriptor = stage.descriptor_storage.require(FteTaskId("q", 33, 0))
    assert [split.data for split in descriptor.initial_splits["7"]] == [b"a", b"b"]


def test_fte_fragment_execution_appends_descriptor_before_split_queue_full_failure():
    class _FullSplitQueueWorker(_FakeLiveWorker):
        def fte_wait_split_queue_has_space(
            self,
            task_id,
            source_node_id=None,
            max_buffered_splits=None,
            timeout_s=None,
        ):
            self.calls.append(
                (
                    "wait_space",
                    task_id,
                    source_node_id,
                    max_buffered_splits,
                    timeout_s,
                )
            )
            return {"has_space": False, "buffered_splits": 1024}

    worker = _FullSplitQueueWorker()
    stage = _fte_fragment_execution(
        "q",
        37,
        fragment_id="q:node:scan",
        worker=worker,
        worker_id=worker.worker_id,
        source_node_ids={"7"},
        dynamic_scan_source_node_ids={"7"},
    )

    scheduled_result = stage.apply_assignment_result(
        AssignmentResult(
            partitions_added=[PartitionInfo(0)],
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [{"sequence_id": 0, "kind": "scan_task", "data": b"a"}],
                    ready_for_scheduling=True,
                )
            ],
        )
    )
    _execute_stage_commands(stage, scheduled_result)
    update_result = stage.apply_assignment_result(
        AssignmentResult(
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [{"sequence_id": 1, "kind": "scan_task", "data": b"b"}],
                )
            ]
        )
    )

    with pytest.raises(FteWorkerControlFailure, match="fte_add_splits"):
        _execute_stage_commands(stage, update_result)

    assert [call[0] for call in worker.calls] == ["create", "wait_space"]
    descriptor = stage.descriptor_storage.require(FteTaskId("q", 37, 0))
    assert [split.data for split in descriptor.initial_splits["7"]] == [b"a", b"b"]


def test_fte_fragment_execution_revoke_unsealed_speculative_waits_for_seal():
    worker = _FakeLiveWorker()
    stage = _fte_fragment_execution(
        "q",
        31,
        fragment_id="q:node:speculative",
        worker=worker,
        worker_id=worker.worker_id,
        context={"task_execution_class": "SPECULATIVE"},
        source_node_ids={"7"},
        dynamic_scan_source_node_ids={"7"},
    )
    scheduled_result = stage.apply_assignment_result(
        AssignmentResult(
            partitions_added=[PartitionInfo(0)],
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [{"sequence_id": 0, "kind": "scan_task", "data": b"a"}],
                    ready_for_scheduling=True,
                )
            ],
        )
    )
    scheduled = scheduled_result
    _execute_stage_commands(stage, scheduled_result)
    assert len(scheduled) == 1
    assert scheduled[0].request["execution_class"] == "SPECULATIVE"

    revoked = stage.revoke_speculative_attempts(reason="memory pressure")

    assert [str(item.attempt_id) for item in revoked] == ["q.31.0.0"]
    assert worker.calls[-1] == ("cancel", scheduled[0].attempt_id.to_dict())
    assert stage.schedule_next_pending_partition() is None
    assert stage.partitions[0].ready_for_scheduling is False

    scheduled_after_seal = stage.seal_partition(0)

    assert scheduled_after_seal is not None
    assert str(scheduled_after_seal.attempt_id) == "q.31.0.1"
    assert scheduled_after_seal.request["execution_class"] == "STANDARD"


def test_fte_fragment_execution_seal_transitions_running_speculative_to_standard():
    worker = _FakeLiveWorker()
    stage = _fte_fragment_execution(
        "q",
        32,
        fragment_id="q:node:speculative-seal",
        worker=worker,
        worker_id=worker.worker_id,
        context={"task_execution_class": "SPECULATIVE"},
        source_node_ids={"7"},
        dynamic_scan_source_node_ids={"7"},
    )
    scheduled_result = stage.apply_assignment_result(
        AssignmentResult(
            partitions_added=[PartitionInfo(0)],
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [{"sequence_id": 0, "kind": "scan_task", "data": b"a"}],
                    ready_for_scheduling=True,
                )
            ],
        )
    )
    scheduled = scheduled_result[0]
    _execute_stage_commands(stage, scheduled_result)

    assert scheduled.request["execution_class"] == "SPECULATIVE"
    assert stage.seal_partition(0) is None
    assert stage.partitions[0].execution_class.value == "STANDARD"
    assert worker.calls[-1] == (
        "set_class",
        scheduled.attempt_id.to_dict(),
        "STANDARD",
    )


def test_fte_fragment_execution_dynamic_exchange_defaults_to_speculative_until_seal():
    worker = _FakeLiveWorker()
    stage = _fte_fragment_execution(
        "q",
        33,
        fragment_id="q:node:dynamic-exchange",
        worker=worker,
        worker_id=worker.worker_id,
        source_node_ids={"3"},
        dynamic_exchange_source_node_ids={"3"},
    )
    scheduled_result = stage.apply_assignment_result(
        AssignmentResult(
            partitions_added=[PartitionInfo(0)],
            partition_updates=[
                PartitionUpdate(
                    0,
                    "3",
                    [{"sequence_id": 0, "kind": "exchange_source_task", "data": b"p0"}],
                    ready_for_scheduling=True,
                )
            ],
        )
    )
    scheduled = scheduled_result[0]
    _execute_stage_commands(stage, scheduled_result)

    assert scheduled.request["execution_class"] == "SPECULATIVE"
    assert stage.seal_partition(0) is None
    assert stage.partitions[0].execution_class.value == "STANDARD"
    assert worker.calls[-1] == (
        "set_class",
        scheduled.attempt_id.to_dict(),
        "STANDARD",
    )


def test_fte_fragment_execution_dynamic_exchange_keeps_explicit_standard_class():
    worker = _FakeLiveWorker()
    stage = _fte_fragment_execution(
        "q",
        34,
        fragment_id="q:node:dynamic-exchange-standard",
        worker=worker,
        worker_id=worker.worker_id,
        context={"task_execution_class": "STANDARD"},
        source_node_ids={"3"},
        dynamic_exchange_source_node_ids={"3"},
    )
    scheduled_result = stage.apply_assignment_result(
        AssignmentResult(
            partitions_added=[PartitionInfo(0)],
            partition_updates=[
                PartitionUpdate(
                    0,
                    "3",
                    [{"sequence_id": 0, "kind": "exchange_source_task", "data": b"p0"}],
                    ready_for_scheduling=True,
                )
            ],
        )
    )
    scheduled = scheduled_result[0]
    _execute_stage_commands(stage, scheduled_result)

    assert scheduled.request["execution_class"] == "STANDARD"
    assert stage.seal_partition(0) is None
    assert [call[0] for call in worker.calls] == ["create"]


def test_fte_fragment_execution_attempt_admission_defers_speculative_until_allowed():
    worker = _FakeLiveWorker()
    allow_speculative = False

    def admit(partition):
        return allow_speculative or not partition.execution_class.is_speculative

    stage = _fte_fragment_execution(
        "q",
        35,
        fragment_id="q:node:dynamic-exchange-admission",
        worker=worker,
        worker_id=worker.worker_id,
        attempt_admission_callback=admit,
        source_node_ids={"3"},
        dynamic_exchange_source_node_ids={"3"},
    )

    scheduled = stage.apply_assignment_result(
        AssignmentResult(
            partitions_added=[PartitionInfo(0)],
            partition_updates=[
                PartitionUpdate(
                    0,
                    "3",
                    [{"sequence_id": 0, "kind": "exchange_source_task", "data": b"p0"}],
                    ready_for_scheduling=True,
                )
            ],
        )
    )

    assert scheduled == []
    assert worker.calls == []
    assert stage.has_pending_partitions("SPECULATIVE") is True

    allow_speculative = True
    scheduled_attempt = stage.schedule_next_pending_partition()
    assert scheduled_attempt is not None
    scheduled = [scheduled_attempt]
    _execute_stage_commands(stage)

    assert [str(attempt.attempt_id) for attempt in scheduled] == ["q.35.0.0"]
    assert worker.calls[0][0] == "create"
    assert worker.calls[0][1]["execution_class"] == "SPECULATIVE"


def test_fte_fragment_execution_execution_admission_defers_ready_until_released():
    worker = _FakeLiveWorker()
    allow_execution = False

    def admit(_partition):
        return allow_execution

    stage = _fte_fragment_execution(
        "q",
        36,
        fragment_id="q:node:execution-admission",
        worker=worker,
        worker_id=worker.worker_id,
        execution_admission_callback=admit,
        source_node_ids={"7"},
        dynamic_scan_source_node_ids={"7"},
    )

    scheduled = stage.apply_assignment_result(
        AssignmentResult(
            partitions_added=[PartitionInfo(0)],
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [{"sequence_id": 0, "kind": "scan_task", "data": b"p0"}],
                    ready_for_scheduling=True,
                )
            ],
        )
    )

    assert scheduled == []
    assert worker.calls == []
    assert stage.partitions[0].ready_for_scheduling is False
    assert stage.partitions[0].execution_ready_deferred is True
    assert stage.waiting_for_execution_count() == 0
    assert stage.has_pending_partitions() is True

    allow_execution = True
    released = stage.release_deferred_execution_partitions()
    scheduled_attempt = stage.schedule_next_pending_partition()
    assert scheduled_attempt is not None
    scheduled = [scheduled_attempt]
    _execute_stage_commands(stage)

    assert [str(task_id) for task_id in released] == ["q.36.0.0"]
    assert [str(attempt.attempt_id) for attempt in scheduled] == ["q.36.0.0"]
    assert stage.partitions[0].execution_ready_deferred is False
    assert worker.calls[0][0] == "create"


def test_fte_fragment_execution_coalesces_and_chunks_split_updates(monkeypatch):
    monkeypatch.setenv("VANE_FTE_TASK_UPDATE_MAX_SPLITS", "2")
    worker = _FakeLiveWorker()
    stage = _fte_fragment_execution(
        "q",
        13,
        fragment_id="q:node:scan",
        worker=worker,
        source_node_ids={"7"},
        dynamic_scan_source_node_ids={"7"},
    )
    initial = stage.apply_assignment_result(
        AssignmentResult(
            partitions_added=[PartitionInfo(0)],
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [{"sequence_id": 0, "kind": "scan_task", "data": b"a"}],
                    ready_for_scheduling=True,
                )
            ],
        )
    )
    _execute_stage_commands(stage, initial)

    scheduled = stage.apply_assignment_result(
        AssignmentResult(
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [
                        {"sequence_id": 1, "kind": "scan_task", "data": b"b"},
                        {"sequence_id": 2, "kind": "scan_task", "data": b"c"},
                    ],
                    ready_for_scheduling=True,
                ),
                PartitionUpdate(
                    0,
                    "7",
                    [
                        {"sequence_id": 2, "kind": "scan_task", "data": b"dup"},
                        {"sequence_id": 3, "kind": "scan_task", "data": b"d"},
                    ],
                    ready_for_scheduling=True,
                ),
                PartitionUpdate(0, "7", no_more_splits=True),
            ]
        )
    )

    assert scheduled == []
    _execute_stage_commands(stage, scheduled)
    assert [call[0] for call in worker.calls] == ["create", "add", "add", "no_more"]
    assert [split["data"] for split in worker.calls[1][3]] == [b"b", b"c"]
    assert [split["data"] for split in worker.calls[2][3]] == [b"d"]
    descriptor = stage.descriptor_storage.require(FteTaskId("q", 13, 0))
    assert [split.data for split in descriptor.initial_splits["7"]] == [b"a", b"b", b"c", b"d"]
    assert descriptor.no_more_splits == {"7"}


def test_fte_fragment_execution_rejects_splits_after_no_more_in_same_batch():
    stage = _fte_fragment_execution("q", 14, fragment_id="q:node:scan", worker=_FakeLiveWorker())

    with pytest.raises(RuntimeError, match="received splits after no_more_splits"):
        stage.apply_assignment_result(
            AssignmentResult(
                partition_updates=[
                    PartitionUpdate(0, "7", no_more_splits=True),
                    PartitionUpdate(
                        0,
                        "7",
                        [{"sequence_id": 0, "kind": "scan_task", "data": b"late"}],
                        ready_for_scheduling=True,
                    ),
                ]
            )
        )


def test_fte_fragment_execution_no_more_is_recorded_and_sent_once():
    worker = _FakeLiveWorker()
    stage = _fte_fragment_execution(
        "q",
        3,
        fragment_id="q:node:scan",
        worker=worker,
        source_node_ids={"7"},
    )

    initial = stage.apply_assignment_result(
        AssignmentResult(
            partitions_added=[PartitionInfo(0)],
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [{"sequence_id": 0, "kind": "scan_task", "data": b"a"}],
                    ready_for_scheduling=True,
                )
            ],
        )
    )
    _execute_stage_commands(stage, initial)
    no_more = stage.apply_assignment_result(
        AssignmentResult(partition_updates=[PartitionUpdate(0, "7", no_more_splits=True)])
    )
    _execute_stage_commands(stage, no_more)
    duplicate_no_more = stage.apply_assignment_result(
        AssignmentResult(partition_updates=[PartitionUpdate(0, "7", no_more_splits=True)])
    )
    _execute_stage_commands(stage, duplicate_no_more)

    assert [call[0] for call in worker.calls] == ["create", "no_more"]
    assert worker.calls[1][1]["attempt_id"] == 0
    assert worker.calls[1][2] == "7"
    descriptor = stage.descriptor_storage.require(FteTaskId("q", 3, 0))
    assert descriptor.no_more_splits == {"7"}
    with pytest.raises(RuntimeError, match="already marked no_more_splits"):
        stage.apply_assignment_result(
            AssignmentResult(
                partition_updates=[
                    PartitionUpdate(
                        0,
                        "7",
                        [{"sequence_id": 1, "kind": "scan_task", "data": b"late"}],
                    )
                ]
            )
        )


def test_fte_fragment_execution_sealed_empty_partition_creates_task():
    worker = _FakeLiveWorker()
    stage = _fte_fragment_execution("q", 3, fragment_id="q:node:scan", worker=worker)

    scheduled = stage.apply_assignment_result(
        AssignmentResult(partitions_added=[PartitionInfo(0)], sealed_partitions=[0])
    )
    _execute_stage_commands(stage, scheduled)

    assert len(scheduled) == 1
    assert worker.calls[0][0] == "create"
    assert worker.calls[0][1]["initial_splits"] == {}
    assert stage.partitions[0].sealed is True


def test_fte_fragment_execution_retry_replays_accumulated_descriptor():
    worker = _FakeLiveWorker()
    stage = _fte_fragment_execution(
        "q",
        4,
        fragment_id="q:node:scan",
        worker=worker,
        max_attempts=2,
        source_node_ids={"7"},
    )

    initial = stage.apply_assignment_result(
        AssignmentResult(
            partitions_added=[PartitionInfo(0)],
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [{"sequence_id": 0, "kind": "scan_task", "data": b"a"}],
                    ready_for_scheduling=True,
                ),
                PartitionUpdate(0, "7", no_more_splits=True),
            ],
        )
    )
    _execute_stage_commands(stage, initial)
    retry = stage.task_failed(FteTaskAttemptId(FteTaskId("q", 4, 0), 0), "lost")
    _execute_stage_commands(stage)

    assert retry is not None
    assert str(retry.attempt_id) == "q.4.0.1"
    assert [call[0] for call in worker.calls] == ["create", "create"]
    assert worker.calls[0][1]["no_more_splits"] == ["7"]
    retry_request = worker.calls[1][1]
    assert retry_request["task_id"]["attempt_id"] == 1
    assert retry_request["initial_splits"]["7"][0]["data"] == b"a"
    assert retry_request["no_more_splits"] == ["7"]
    assert stage.partitions[0].remaining_attempts == 1


def test_fte_fragment_execution_oom_is_terminal_for_fixed_heap():
    worker = _FakeLiveWorker()
    stage = _fte_fragment_execution(
        "qmem",
        4,
        fragment_id="qmem:node:scan",
        worker=worker,
        max_attempts=3,
        source_node_ids={"7"},
        dynamic_scan_source_node_ids={"7"},
    )

    initial = stage.apply_assignment_result(
        AssignmentResult(
            partitions_added=[PartitionInfo(0)],
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [{"sequence_id": 0, "kind": "scan_task", "data": b"a"}],
                    ready_for_scheduling=True,
                ),
                PartitionUpdate(0, "7", no_more_splits=True),
            ],
        )
    )
    _execute_stage_commands(stage, initial)
    first_request = worker.calls[0][1]
    assert first_request["memory_requirement_bytes"] == 1024

    retry = stage.task_failed(
        FteTaskAttemptId(FteTaskId("qmem", 4, 0), 0),
        {
            "error_code": "EXCEEDED_LOCAL_MEMORY_LIMIT",
            "peak_memory_bytes": 1536,
        },
    )
    _execute_stage_commands(stage)

    assert retry is None
    assert stage.failed is True
    assert stage.partitions[0].state == FtePartitionState.FAILED
    assert [call[0] for call in worker.calls] == ["create"]


def test_fte_fragment_execution_finish_removes_descriptor_and_finalizes_exchange():
    worker = _FakeLiveWorker()
    storage = TaskDescriptorStorage()
    exchange = FteExchangeTracker("q", "stage-5")
    stage = _fte_fragment_execution(
        "q",
        5,
        fragment_id="q:node:sink",
        worker=worker,
        descriptor_storage=storage,
        exchange=exchange,
    )

    scheduled = stage.apply_assignment_result(
        AssignmentResult(partitions_added=[PartitionInfo(0)], sealed_partitions=[0])
    )[0]

    assert scheduled.sink_instance is not None
    assert storage.get(FteTaskId("q", 5, 0)) is not None
    assert stage.task_finished(scheduled.attempt_id, {"rows": 10}) is True
    assert storage.get(FteTaskId("q", 5, 0)) is None
    assert stage.partitions[0].selected_output_stats == {"rows": 10}
    assert stage.partitions and all(partition.finished for partition in stage.partitions.values())
    assert exchange.finalize() is True
    handles = exchange.get_source_handles()
    assert len(handles) == 1
    assert handles[0].attempt_id == 0


def test_fte_fragment_execution_finish_cancels_unselected_running_attempts():
    class _AccountingWorker(_FakeLiveWorker):
        def __init__(self, worker_id):
            super().__init__(worker_id)
            self.terminal_attempts = []

        def record_fte_task_started(self, _attempt_id, _request):
            return None

        def record_fte_task_terminal(self, attempt_id):
            self.terminal_attempts.append(str(FteTaskAttemptId.coerce(attempt_id)))

    winner_worker = _AccountingWorker("worker-winner")
    loser_worker = _AccountingWorker("worker-loser")
    exchange = FteExchangeTracker("q", "stage-selected")
    stage = _fte_fragment_execution(
        "q",
        61,
        fragment_id="q:node:selected",
        worker=winner_worker,
        worker_id=winner_worker.worker_id,
        exchange=exchange,
        max_attempts=2,
    )

    scheduled = stage.apply_assignment_result(
        AssignmentResult(partitions_added=[PartitionInfo(0)], sealed_partitions=[0])
    )[0]
    _execute_stage_commands(stage)
    partition = stage.partitions[0]
    loser_attempt = FteTaskAttemptId(FteTaskId("q", 61, 0), 1)
    loser_sink = exchange.instantiate_sink(partition.sink_handle, loser_attempt.attempt_id)
    partition.running_attempts[loser_attempt.attempt_id] = RunningAttempt(
        loser_attempt,
        worker_id=loser_worker.worker_id,
        remote_handle=loser_worker,
        sink_instance=loser_sink,
    )
    partition.running_task_stats[loser_attempt.attempt_id] = {"rows": 99}

    assert stage.task_finished(scheduled.attempt_id, {"rows": 10}) is True

    assert loser_worker.calls[-1] == ("cancel", loser_attempt.to_dict())
    assert winner_worker.terminal_attempts == ["q.61.0.0"]
    assert loser_worker.terminal_attempts == ["q.61.0.1"]
    assert partition.running_attempts == {}
    assert partition.running_task_stats == {}
    assert exchange.selected_attempt(partition.sink_handle) == 0
    assert exchange._aborted_attempts[0] == {1}


def test_fte_fragment_execution_handle_finished_status_marks_task_finished():
    worker = _FakeLiveWorker()
    storage = TaskDescriptorStorage()
    stage = _fte_fragment_execution(
        "q",
        6,
        fragment_id="q:node:scan",
        worker=worker,
        descriptor_storage=storage,
    )
    scheduled = stage.apply_assignment_result(
        AssignmentResult(partitions_added=[PartitionInfo(0)], sealed_partitions=[0])
    )[0]
    worker.set_status(
        scheduled.attempt_id.to_dict(),
        "FINISHED",
        spooling_output_stats={"rows": 11},
    )

    retries = _handle_live_worker_statuses(stage)

    assert retries == []
    assert stage.partitions[0].state == FtePartitionState.FINISHED
    assert stage.partitions[0].selected_output_stats == {"rows": 11}
    assert storage.get(FteTaskId("q", 6, 0)) is None
    assert worker.calls[-1][0] == "status"


def test_fte_fragment_execution_handle_finished_missing_stats_retries_exchange_attempt():
    worker = _FakeLiveWorker()
    storage = TaskDescriptorStorage()
    exchange = FteExchangeTracker("q", "stage-missing-stats")
    stage = _fte_fragment_execution(
        "q",
        41,
        fragment_id="q:node:sink",
        worker=worker,
        descriptor_storage=storage,
        exchange=exchange,
        max_attempts=2,
    )
    scheduled_result = stage.apply_assignment_result(
        AssignmentResult(partitions_added=[PartitionInfo(0)], sealed_partitions=[0])
    )
    scheduled0 = scheduled_result[0]
    _execute_stage_commands(stage, scheduled_result)
    worker.set_status(scheduled0.attempt_id.to_dict(), "FINISHED")

    retries = _handle_live_worker_statuses(stage)
    _execute_stage_commands(stage)

    assert len(retries) == 1
    assert str(retries[0].attempt_id) == "q.41.0.1"
    assert stage.partitions[0].state == FtePartitionState.RUNNING
    assert storage.get(FteTaskId("q", 41, 0)) is not None
    assert exchange.get_source_handles() == []
    assert [call[0] for call in worker.calls] == ["create", "status", "create"]
    retry_request = worker.calls[-1][1]
    assert retry_request["task_id"]["attempt_id"] == 1
    assert retry_request["exchange_sink_instance"]["attempt_id"] == 1


def test_fte_fragment_execution_handle_finished_missing_stats_allows_no_sink_task():
    worker = _FakeLiveWorker()
    storage = TaskDescriptorStorage()
    stage = _fte_fragment_execution(
        "q",
        42,
        fragment_id="q:node:no-sink",
        worker=worker,
        descriptor_storage=storage,
        max_attempts=2,
    )
    scheduled_result = stage.apply_assignment_result(
        AssignmentResult(partitions_added=[PartitionInfo(0)], sealed_partitions=[0])
    )
    scheduled = scheduled_result[0]
    _execute_stage_commands(stage, scheduled_result)
    worker.set_status(scheduled.attempt_id.to_dict(), "FINISHED")

    retries = _handle_live_worker_statuses(stage)

    assert retries == []
    assert stage.partitions[0].state == FtePartitionState.FINISHED
    assert storage.get(FteTaskId("q", 42, 0)) is None
    assert [call[0] for call in worker.calls] == ["create", "status"]


def test_fte_fragment_execution_handle_failed_status_retries_with_replayed_descriptor():
    worker = _FakeLiveWorker()
    stage = _fte_fragment_execution(
        "q",
        7,
        fragment_id="q:node:scan",
        worker=worker,
        max_attempts=2,
        source_node_ids={"7"},
    )
    scheduled_result = stage.apply_assignment_result(
        AssignmentResult(
            partitions_added=[PartitionInfo(0)],
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [{"sequence_id": 0, "kind": "scan_task", "data": b"a"}],
                    ready_for_scheduling=True,
                ),
            ],
        )
    )
    scheduled0 = scheduled_result[0]
    _execute_stage_commands(stage, scheduled_result)
    worker.set_status(scheduled0.attempt_id.to_dict(), "FAILED", failure={"message": "lost"})

    retries = _handle_live_worker_statuses(stage)
    _execute_stage_commands(stage)

    assert len(retries) == 1
    assert str(retries[0].attempt_id) == "q.7.0.1"
    assert stage.partitions[0].state == FtePartitionState.RUNNING
    assert [call[0] for call in worker.calls] == ["create", "status", "create"]
    retry_request = worker.calls[-1][1]
    assert retry_request["initial_splits"]["7"][0]["data"] == b"a"


def test_fte_fragment_execution_handle_user_error_status_fails_without_retry():
    worker = _FakeLiveWorker()
    stage = _fte_fragment_execution(
        "q",
        37,
        fragment_id="q:node:user-error",
        worker=worker,
        max_attempts=2,
    )
    scheduled_result = stage.apply_assignment_result(
        AssignmentResult(partitions_added=[PartitionInfo(0)], sealed_partitions=[0])
    )
    scheduled = scheduled_result[0]
    _execute_stage_commands(stage, scheduled_result)
    worker.set_status(
        scheduled.attempt_id.to_dict(),
        "FAILED",
        failure={
            "error_type": "USER_ERROR",
            "error_code": "INVALID_FUNCTION_ARGUMENT",
            "message": "bad input",
        },
    )

    retries = _handle_live_worker_statuses(stage)

    assert retries == []
    assert stage.failed is True
    assert stage.partitions[0].state == FtePartitionState.FAILED
    assert [call[0] for call in worker.calls] == ["create", "status"]


def test_fte_fragment_execution_handle_fatal_status_fails_without_retry():
    worker = _FakeLiveWorker()
    stage = _fte_fragment_execution(
        "q",
        38,
        fragment_id="q:node:fatal",
        worker=worker,
        max_attempts=2,
    )
    scheduled_result = stage.apply_assignment_result(
        AssignmentResult(partitions_added=[PartitionInfo(0)], sealed_partitions=[0])
    )
    scheduled = scheduled_result[0]
    _execute_stage_commands(stage, scheduled_result)
    worker.set_status(
        scheduled.attempt_id.to_dict(),
        "FAILED",
        failure={
            "error_type": "INTERNAL_ERROR",
            "fatal": True,
            "message": "coordinator cannot recover",
        },
    )

    retries = _handle_live_worker_statuses(stage)

    assert retries == []
    assert stage.failed is True
    assert stage.partitions[0].state == FtePartitionState.FAILED
    assert [call[0] for call in worker.calls] == ["create", "status"]


def test_fte_fragment_execution_handle_canceled_status_fails_without_retry_by_default():
    worker = _FakeLiveWorker()
    stage = _fte_fragment_execution(
        "q",
        39,
        fragment_id="q:node:canceled",
        worker=worker,
        max_attempts=2,
    )
    scheduled_result = stage.apply_assignment_result(
        AssignmentResult(partitions_added=[PartitionInfo(0)], sealed_partitions=[0])
    )
    scheduled = scheduled_result[0]
    _execute_stage_commands(stage, scheduled_result)
    worker.set_status(
        scheduled.attempt_id.to_dict(),
        "CANCELED",
        failure={"message": "query canceled"},
    )

    retries = _handle_live_worker_statuses(stage)

    assert retries == []
    assert stage.failed is True
    assert stage.partitions[0].state == FtePartitionState.FAILED
    assert [call[0] for call in worker.calls] == ["create", "status"]


def test_fte_fragment_execution_oom_status_fails_without_retry():
    worker = _FakeLiveWorker()
    stage = _fte_fragment_execution(
        "qoom",
        40,
        fragment_id="qoom:node:scan",
        worker=worker,
        max_attempts=3,
    )
    scheduled_result = stage.apply_assignment_result(
        AssignmentResult(partitions_added=[PartitionInfo(0)], sealed_partitions=[0])
    )
    scheduled = scheduled_result[0]
    _execute_stage_commands(stage, scheduled_result)
    worker.set_status(
        scheduled.attempt_id.to_dict(),
        "FAILED",
        failure={
            "error_type": "INTERNAL_ERROR",
            "error_code": "EXCEEDED_LOCAL_MEMORY_LIMIT",
            "peak_memory_bytes": 2048,
        },
    )

    retries = _handle_live_worker_statuses(stage)

    assert retries == []
    assert stage.failed is True
    assert stage.partitions[0].state == FtePartitionState.FAILED
    assert [call[0] for call in worker.calls] == ["create", "status"]


def test_fte_fragment_execution_handle_failed_status_marks_stage_failed_when_attempts_exhausted():
    worker = _FakeLiveWorker()
    stage = _fte_fragment_execution(
        "q",
        8,
        fragment_id="q:node:scan",
        worker=worker,
        max_attempts=1,
    )
    scheduled = stage.apply_assignment_result(
        AssignmentResult(partitions_added=[PartitionInfo(0)], sealed_partitions=[0])
    )[0]
    worker.set_status(scheduled.attempt_id.to_dict(), "FAILED", failure={"message": "fatal"})

    retries = _handle_live_worker_statuses(stage)

    assert retries == []
    assert stage.failed is True
    assert stage.partitions[0].state == FtePartitionState.FAILED
    assert stage.descriptor_storage.get(FteTaskId("q", 8, 0)) is not None


def test_fte_fragment_execution_worker_lost_uses_retry_attempt_as_durable_output(tmp_path):
    worker0 = _FakeLiveWorker("worker-a")
    worker1 = _FakeLiveWorker("worker-b")
    exchange = SpoolingExchangeManager(tmp_path, "q", "stage-fte-lost")

    def select_worker(partition):
        return worker0 if partition.next_attempt_number() == 0 else worker1

    stage = _fte_fragment_execution(
        "q",
        10,
        fragment_id="q:node:shuffle",
        worker_selector=select_worker,
        max_attempts=2,
        exchange=exchange,
        source_node_ids={"7"},
        dynamic_scan_source_node_ids={"7"},
    )

    scheduled_result = stage.apply_assignment_result(
        AssignmentResult(
            partitions_added=[PartitionInfo(0)],
            partition_updates=[
                PartitionUpdate(
                    0,
                    "7",
                    [{"sequence_id": 0, "kind": "scan_task", "data": b"old-worker"}],
                    ready_for_scheduling=True,
                ),
                PartitionUpdate(0, "7", no_more_splits=True),
            ],
        )
    )
    scheduled0 = scheduled_result[0]
    _execute_stage_commands(stage, scheduled_result)
    lost_path = exchange.record_output_file(scheduled0.sink_instance, 0, "lost", b"lost")
    exchange.finish_attempt(scheduled0.sink_instance)

    retries = stage.mark_worker_failed("worker-a", "actor died before status was observed")
    _execute_stage_commands(stage)

    assert len(retries) == 1
    retry = retries[0]
    assert retry.worker_id == "worker-b"
    assert str(retry.attempt_id) == "q.10.0.1"
    assert worker1.calls[0][0] == "create"
    retry_request = worker1.calls[0][1]
    assert retry_request["task_id"]["attempt_id"] == 1
    assert retry_request["initial_splits"]["7"][0]["data"] == b"old-worker"
    assert retry_request["no_more_splits"] == ["7"]
    assert (Path(scheduled0.sink_instance.attempt_path) / "committed").exists()
    assert (Path(scheduled0.sink_instance.attempt_path) / "aborted").exists()

    selected_path = exchange.record_output_file(retry.sink_instance, 0, "selected", b"selected")
    exchange.finish_attempt(retry.sink_instance)
    assert stage.task_finished(retry.attempt_id, {"rows": 1}) is True

    assert stage.partitions and all(partition.finished for partition in stage.partitions.values())
    assert exchange.finalize() is True
    handles = exchange.get_source_handles()

    assert len(handles) == 1
    assert handles[0].attempt_id == 1
    assert handles[0].files == (selected_path,)
    assert lost_path not in handles[0].files
    assert stage.descriptor_storage.get(FteTaskId("q", 10, 0)) is None
    assert exchange.cleanup_unselected_attempts() == 1
    assert not Path(scheduled0.sink_instance.attempt_path).exists()
    assert Path(retry.sink_instance.attempt_path).exists()


def test_fte_fragment_execution_retries_base_remote_exchange_sink_instance():
    worker0 = _FakeLiveWorker("worker-a")
    worker1 = _FakeLiveWorker("worker-b")
    base_sink = {
        "sink_handle": {"task_partition_id": 0, "partition_id": 0},
        "task_partition_id": 0,
        "partition_id": 0,
        "attempt_id": 0,
        "output_partition_count": 4,
        "output_location": "q_shuffle_3__sink_0__attempt_0",
        "attempt_path": "q_shuffle_3__sink_0__attempt_0",
    }

    def select_worker(partition):
        return worker0 if partition.next_attempt_number() == 0 else worker1

    stage = _fte_fragment_execution(
        "q",
        11,
        fragment_id="q:node:shuffle",
        worker_selector=select_worker,
        max_attempts=2,
        task_context_info={"exchange_sink_instance": base_sink},
    )

    scheduled_result = stage.apply_assignment_result(
        AssignmentResult(partitions_added=[PartitionInfo(0)], sealed_partitions=[0])
    )
    scheduled0 = scheduled_result[0]
    _execute_stage_commands(stage, scheduled_result)
    retry = stage.task_failed(scheduled0.attempt_id, "lost worker")
    _execute_stage_commands(stage)

    assert retry is not None
    retry_request = worker1.calls[0][1]
    assert scheduled0.request["exchange_sink_instance"]["attempt_id"] == 0
    assert scheduled0.request["exchange_sink_instance"]["sink_handle"]["task_partition_id"] == 0
    assert retry.request["exchange_sink_instance"]["attempt_id"] == 1
    assert retry.request["exchange_sink_instance"]["sink_handle"]["task_partition_id"] == 0
    assert (
        scheduled0.request["exchange_sink_instance"]["output_location"]
        != retry.request["exchange_sink_instance"]["output_location"]
    )
    assert (
        scheduled0.request["exchange_sink_instance"]["attempt_path"]
        != retry.request["exchange_sink_instance"]["attempt_path"]
    )
    assert retry.request["exchange_sink_instance"]["output_location"].endswith("__attempt_1")
    assert retry.request["exchange_sink_instance"]["attempt_path"].endswith("__attempt_1")
    assert retry.request["exchange_sink_instance"]["output_partition_count"] == 4
    assert retry_request["exchange_sink_instance"] == retry.request["exchange_sink_instance"]


def test_fte_fragment_execution_rewrites_base_sink_partition_for_dynamic_task_partition():
    worker = _FakeLiveWorker("worker-a")
    base_sink = {
        "sink_handle": {"task_partition_id": 0, "partition_id": 0},
        "task_partition_id": 0,
        "partition_id": 0,
        "attempt_id": 0,
        "output_partition_count": 4,
        "output_location": "q_shuffle_3__sink_0__attempt_0",
        "attempt_path": "q_shuffle_3__sink_0__attempt_0",
    }
    stage = _fte_fragment_execution(
        "q",
        12,
        fragment_id="q:node:shuffle",
        worker=worker,
        task_context_info={"exchange_sink_instance": base_sink},
    )

    scheduled_result = stage.apply_assignment_result(
        AssignmentResult(partitions_added=[PartitionInfo(3)], sealed_partitions=[3])
    )
    scheduled = scheduled_result[0]
    _execute_stage_commands(stage, scheduled_result)
    sink_instance = scheduled.request["exchange_sink_instance"]

    assert sink_instance["sink_handle"]["task_partition_id"] == 3
    assert sink_instance["sink_handle"]["partition_id"] == 3
    assert sink_instance["task_partition_id"] == 3
    assert sink_instance["partition_id"] == 3
    assert sink_instance["output_location"] == "q_shuffle_3__sink_3__attempt_0"
    assert worker.calls[0][1]["exchange_sink_instance"] == sink_instance


def test_fte_fragment_execution_handle_task_status_accepts_task_id_string():
    worker = _FakeLiveWorker()
    stage = _fte_fragment_execution(
        "q",
        9,
        fragment_id="q:node:scan",
        worker=worker,
    )
    scheduled = stage.apply_assignment_result(
        AssignmentResult(partitions_added=[PartitionInfo(0)], sealed_partitions=[0])
    )[0]

    retry = stage.handle_task_status(
        {
            "task_id_string": str(scheduled.attempt_id),
            "state": FteTaskState.CANCELED,
            "failure": {"message": "canceled"},
        },
        retryable=False,
    )

    assert retry is None
    assert stage.failed is True
    assert stage.partitions[0].state == FtePartitionState.FAILED


def test_single_split_assigner_waits_for_all_sources_before_sealing():
    assigner = SingleSplitAssigner(all_sources={"11", "12"})

    result = assigner.assign(
        "11",
        [{"kind": "scan_task", "data": b"a"}, {"kind": "scan_task", "data": b"b"}],
        no_more_inputs=True,
    )

    assert [p.partition_id for p in result.partitions_added] == [0]
    assert result.sealed_partitions == []
    assert result.no_more_partitions is True
    data_update, completion_update = result.partition_updates
    assert data_update.partition_id == 0
    assert data_update.no_more_splits is False
    assert data_update.ready_for_scheduling is True
    assert [split.sequence_id for split in data_update.splits] == [0, 1]
    assert completion_update.partition_id == 0
    assert completion_update.no_more_splits is True
    assert completion_update.ready_for_scheduling is False
    assert completion_update.splits == []

    result = assigner.assign("12", [], no_more_inputs=True)

    assert result.partitions_added == []
    assert result.sealed_partitions == [0]
    assert result.no_more_partitions is False
    assert result.partition_updates[0].source_node_id == "12"
    assert result.partition_updates[0].no_more_splits is True


def test_single_split_assigner_finish_creates_empty_partition():
    result = SingleSplitAssigner().finish()

    assert [p.partition_id for p in result.partitions_added] == [0]
    assert result.sealed_partitions == [0]
    assert result.no_more_partitions is True


def test_arbitrary_split_assigner_keeps_partition_open_until_full_or_finished():
    assigner = ArbitrarySplitAssigner(
        partitioned_sources={"9"},
        min_target_partition_size_bytes=200,
        max_task_split_count=2,
        standard_split_size_bytes=100,
    )

    result = assigner.assign(
        "9",
        [
            {"kind": "scan_task", "data": b"a", "size_bytes": 100},
            {"kind": "scan_task", "data": b"b", "size_bytes": 100},
        ],
    )

    assert [p.partition_id for p in result.partitions_added] == [0]
    assert result.sealed_partitions == []
    assert result.no_more_partitions is False
    assert [update.partition_id for update in result.partition_updates] == [0, 0]
    assert all(update.ready_for_scheduling for update in result.partition_updates)

    result = assigner.assign(
        "9",
        [{"kind": "scan_task", "data": b"c", "size_bytes": 100}],
        no_more_inputs=True,
    )

    assert [p.partition_id for p in result.partitions_added] == [1]
    assert result.sealed_partitions == [0, 1]
    assert result.no_more_partitions is True
    assert [update.no_more_splits for update in result.partition_updates].count(True) >= 2

    with pytest.raises(RuntimeError, match="after finish"):
        assigner.assign("9", [{"kind": "scan_task", "data": b"d"}])


def test_arbitrary_split_assigner_replays_replicated_splits_to_new_partitions():
    assigner = ArbitrarySplitAssigner(
        partitioned_sources={"probe"},
        replicated_sources={"build"},
        min_target_partition_size_bytes=100,
        max_task_split_count=1,
        standard_split_size_bytes=100,
    )

    result = assigner.assign(
        "build",
        [{"kind": "exchange_source_task", "data": b"broadcast"}],
        no_more_inputs=True,
    )
    assert result.partitions_added == []
    assert result.sealed_partitions == []

    result = assigner.assign(
        "probe",
        [
            {"kind": "scan_task", "data": b"a", "size_bytes": 100},
            {"kind": "scan_task", "data": b"b", "size_bytes": 100},
        ],
        no_more_inputs=True,
    )

    assert [p.partition_id for p in result.partitions_added] == [0, 1]
    replay_updates = [
        update for update in result.partition_updates if update.source_node_id == "build" and update.splits
    ]
    assert [update.partition_id for update in replay_updates] == [0, 1]
    assert all(update.no_more_splits for update in replay_updates)
    assert result.sealed_partitions == [0, 1]
    assert result.no_more_partitions is True


def test_arbitrary_split_assigner_waits_for_replicated_source_before_sealing_full_partition():
    assigner = ArbitrarySplitAssigner(
        partitioned_sources={"probe"},
        replicated_sources={"build"},
        min_target_partition_size_bytes=100,
        max_task_split_count=1,
        standard_split_size_bytes=100,
    )

    result = assigner.assign(
        "probe",
        [
            {"kind": "scan_task", "data": b"a", "size_bytes": 100},
            {"kind": "scan_task", "data": b"b", "size_bytes": 100},
        ],
    )

    assert [p.partition_id for p in result.partitions_added] == [0, 1]
    assert result.sealed_partitions == []
    assert result.no_more_partitions is False

    result = assigner.assign("build", [], no_more_inputs=True)

    assert result.sealed_partitions == [0]
    assert result.no_more_partitions is False

    result = assigner.assign("probe", [], no_more_inputs=True)

    assert result.sealed_partitions == [1]
    assert result.no_more_partitions is True


def test_arbitrary_split_assigner_empty_input_creates_one_partition():
    assigner = ArbitrarySplitAssigner(partitioned_sources={"scan"})

    result = assigner.assign("scan", [], no_more_inputs=True)

    assert [p.partition_id for p in result.partitions_added] == [0]
    assert result.sealed_partitions == [0]
    assert result.no_more_partitions is True
    assert result.partition_updates[0].source_node_id == "scan"
    assert result.partition_updates[0].no_more_splits is True


def test_arbitrary_split_assigner_groups_by_node_requirements():
    assigner = ArbitrarySplitAssigner(
        partitioned_sources={"scan"},
        min_target_partition_size_bytes=1000,
        standard_split_size_bytes=100,
    )

    result = assigner.assign(
        "scan",
        [
            {"kind": "scan_task", "data": b"a", "addresses": ["host-a"], "size_bytes": 100},
            {"kind": "scan_task", "data": b"b", "addresses": ["host-b"], "size_bytes": 100},
        ],
    )

    assert [p.partition_id for p in result.partitions_added] == [0, 1]
    assert [p.node_requirements for p in result.partitions_added] == [
        NodeRequirements(host="host-a"),
        NodeRequirements(host="host-b"),
    ]


def test_arbitrary_split_assigner_ranks_available_hosts():
    assigner = ArbitrarySplitAssigner(
        partitioned_sources={"scan"},
        min_target_partition_size_bytes=1000,
        standard_split_size_bytes=100,
    )

    result = assigner.assign(
        "scan",
        [
            {"kind": "scan_task", "data": b"a", "addresses": ["host-a"], "size_bytes": 100},
            {"kind": "scan_task", "data": b"b", "addresses": ["host-a", "host-b"], "size_bytes": 100},
        ],
    )

    assert [p.node_requirements for p in result.partitions_added] == [
        NodeRequirements(host="host-a"),
        NodeRequirements(host="host-b"),
    ]


def test_arbitrary_split_assigner_rejects_catalog_mismatch_and_non_remote_split_without_host():
    assigner = ArbitrarySplitAssigner(partitioned_sources={"scan"}, catalog_requirement="tpch")

    with pytest.raises(ValueError, match="unexpected split catalog requirement"):
        assigner.assign("scan", [{"kind": "scan_task", "catalog": "hive"}])

    assigner = ArbitrarySplitAssigner(partitioned_sources={"scan"})

    with pytest.raises(ValueError, match="not remotely accessible"):
        assigner.assign("scan", [{"kind": "scan_task", "remotely_accessible": False}])


def test_arbitrary_split_assigner_adapts_target_size_after_growth_period():
    assigner = ArbitrarySplitAssigner(
        partitioned_sources={"scan"},
        min_target_partition_size_bytes=100,
        max_target_partition_size_bytes=400,
        adaptive_growth_period=1,
        adaptive_growth_factor=2,
        standard_split_size_bytes=100,
        max_task_split_count=100,
    )

    result = assigner.assign(
        "scan",
        [
            {"kind": "scan_task", "data": b"a", "size_bytes": 100},
            {"kind": "scan_task", "data": b"b", "size_bytes": 100},
            {"kind": "scan_task", "data": b"c", "size_bytes": 100},
        ],
        no_more_inputs=True,
    )

    assert [p.partition_id for p in result.partitions_added] == [0, 1]
    assert result.sealed_partitions == [0, 1]
    data_updates = [update for update in result.partition_updates if update.source_node_id == "scan" and update.splits]
    assert [update.partition_id for update in data_updates] == [0, 1, 1]


def test_hash_split_assigner_creates_all_task_partitions_before_updates():
    assigner = HashSplitAssigner(
        source_partition_count=3,
        partitioned_sources={"shuffle"},
    )

    result = assigner.assign(
        "shuffle",
        [
            {"kind": "exchange_source_task", "data": b"p0", "source_partition_id": 0},
            {"kind": "exchange_source_task", "data": b"p2", "source_partition_id": 2},
        ],
        no_more_inputs=True,
    )

    assert [p.partition_id for p in result.partitions_added] == [0, 1, 2]
    assert result.no_more_partitions is True
    data_updates = [update for update in result.partition_updates if update.splits]
    assert [(update.partition_id, update.splits[0].source_partition_id) for update in data_updates] == [
        (0, 0),
        (2, 2),
    ]
    assert result.sealed_partitions == [0, 1, 2]


def test_hash_split_assigner_merges_source_partitions_with_explicit_mapping():
    shared = HashTaskPartition()
    assigner = HashSplitAssigner(
        source_partition_count=3,
        partitioned_sources={"shuffle"},
        source_partition_to_task_partition={
            0: shared,
            1: shared,
            2: HashTaskPartition(),
        },
    )

    result = assigner.assign(
        "shuffle",
        [
            {"kind": "exchange_source_task", "data": b"p0", "source_partition_id": 0},
            {"kind": "exchange_source_task", "data": b"p1", "source_partition_id": 1},
            {"kind": "exchange_source_task", "data": b"p2", "source_partition_id": 2},
        ],
        no_more_inputs=True,
    )

    assert [p.partition_id for p in result.partitions_added] == [0, 1]
    data_updates = [update for update in result.partition_updates if update.splits]
    assert [(update.partition_id, update.splits[0].source_partition_id) for update in data_updates] == [
        (0, 0),
        (0, 1),
        (1, 2),
    ]
    assert result.sealed_partitions == [0, 1]


def test_hash_split_assigner_splits_large_partition_by_one_source():
    assigner = HashSplitAssigner(
        source_partition_count=1,
        partitioned_sources={"large", "small"},
        source_partition_to_task_partition={
            0: HashTaskPartition(sub_partition_count=2, split_by_source="large"),
        },
    )

    result = assigner.assign(
        "large",
        [
            {"kind": "exchange_source_task", "data": b"a", "source_partition_id": 0},
            {"kind": "exchange_source_task", "data": b"b", "source_partition_id": 0},
        ],
    )

    assert [p.partition_id for p in result.partitions_added] == [0, 1]
    data_updates = [update for update in result.partition_updates if update.splits]
    assert [update.partition_id for update in data_updates] == [0, 1]

    result = assigner.assign(
        "small",
        [{"kind": "exchange_source_task", "data": b"c", "source_partition_id": 0}],
    )

    data_updates = [update for update in result.partition_updates if update.splits]
    assert [update.partition_id for update in data_updates] == [0, 1]


def test_hash_split_assigner_broadcasts_replicated_splits_to_all_task_partitions():
    assigner = HashSplitAssigner(
        source_partition_count=2,
        partitioned_sources={"probe"},
        replicated_sources={"build"},
    )

    result = assigner.assign(
        "build",
        [{"kind": "exchange_source_task", "data": b"broadcast"}],
        no_more_inputs=True,
    )

    assert [p.partition_id for p in result.partitions_added] == [0, 1]
    replay_updates = [update for update in result.partition_updates if update.splits]
    assert [update.partition_id for update in replay_updates] == [0, 1]
    assert result.sealed_partitions == []

    result = assigner.assign("probe", [], no_more_inputs=True)

    assert result.sealed_partitions == [0, 1]


def test_fte_assigner_treats_replicated_exchange_source_as_broadcast():
    state = _FteFragmentState()
    state.dynamic_exchange_source_node_ids.update({"probe", "build"})
    state.replicated_exchange_source_node_ids.add("build")
    state.exchange_source_partition_ids.update({0, 1})
    state.exchange_source_partition_count = 2
    state.exchange_source_task_count = 2
    assigner = make_fte_assigner(state)

    result = assigner.assign(
        "probe",
        [
            {"kind": "exchange_source_task", "data": b"p0", "source_partition_id": 0},
            {"kind": "exchange_source_task", "data": b"p1", "source_partition_id": 1},
        ],
    )
    assert [p.partition_id for p in result.partitions_added] == [0, 1]

    result = assigner.assign(
        "build",
        [{"kind": "exchange_source_task", "data": b"broadcast", "source_partition_id": 0}],
        no_more_inputs=True,
    )

    broadcast_updates = [update for update in result.partition_updates if update.source_node_id == "build"]
    assert [update.partition_id for update in broadcast_updates if update.splits] == [0, 1]
    assert all(update.no_more_splits for update in broadcast_updates)

    result = assigner.assign("probe", [], no_more_inputs=True)
    assert result.sealed_partitions == [0, 1]


def test_fte_assigner_uses_arbitrary_distribution_for_replicated_exchange_only_fragment():
    state = _FteFragmentState()
    state.source_node_ids.update({"scan", "build"})
    state.dynamic_scan_source_node_ids.add("scan")
    state.dynamic_exchange_source_node_ids.add("build")
    state.replicated_exchange_source_node_ids.add("build")
    state.exchange_source_partition_ids.add(0)
    state.exchange_source_partition_count = 1
    state.exchange_source_task_count = 1
    assigner = make_fte_assigner(state)

    assert isinstance(assigner, ArbitrarySplitAssigner)

    result = assigner.assign(
        "build",
        [{"kind": "exchange_source_task", "data": b"broadcast", "source_partition_id": 0}],
        no_more_inputs=True,
    )
    assert result.partitions_added == []

    result = assigner.assign(
        "scan",
        [{"kind": "scan_task", "data": b"scan", "size_bytes": 1024}],
        no_more_inputs=True,
    )
    assert [p.partition_id for p in result.partitions_added] == [0]
    replay_updates = [update for update in result.partition_updates if update.source_node_id == "build"]
    assert [update.partition_id for update in replay_updates if update.splits] == [0]
    assert all(update.no_more_splits for update in replay_updates)
    assert result.sealed_partitions == [0]


def test_fte_assigner_uses_dynamic_scan_max_splits_per_partition(monkeypatch):
    monkeypatch.setenv("VANE_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION", "1")
    state = _FteFragmentState()
    state.source_node_ids.add("scan")
    state.dynamic_scan_source_node_ids.add("scan")

    assigner = make_fte_assigner(state)

    result = assigner.assign(
        "scan",
        [
            {"kind": "scan_task", "data": b"a", "size_bytes": 1},
            {"kind": "scan_task", "data": b"b", "size_bytes": 1},
        ],
        no_more_inputs=True,
    )
    assert [partition.partition_id for partition in result.partitions_added] == [0, 1]
    assert result.sealed_partitions == [0, 1]


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_fte_assigner_rejects_invalid_dynamic_scan_max_splits_per_partition(monkeypatch, value):
    monkeypatch.setenv("VANE_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION", value)
    state = _FteFragmentState()
    state.source_node_ids.add("scan")
    state.dynamic_scan_source_node_ids.add("scan")

    with pytest.raises(ValueError, match="VANE_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION"):
        make_fte_assigner(state)


def test_materialize_task_inputs_merges_context():
    merge_calls = []

    def merge_scan(values):
        merge_calls.append(values)
        return b"".join(values)

    context = materialize_task_inputs(
        {"query_id": "q"},
        {
            "1": [
                {"sequence_id": 0, "kind": "scan_task", "data": b"a"},
                {"sequence_id": 1, "kind": "scan_task", "data": b"b"},
            ],
            "2": [
                {"sequence_id": 0, "kind": "exchange_source_task", "data": b"ex"},
            ],
        },
        merge_scan_task_descriptors=merge_scan,
    )

    assert context["query_id"] == "q"
    assert context["scan_task:1"] == b"ab"
    assert context["scan_task_nodes"] == "1"
    assert context["exchange_source_task:2"] == b"ex"
    assert context["exchange_source_task_nodes"] == "2"
    assert merge_calls == [[b"a", b"b"]]


def test_fte_worker_task_manager_create_status_info_cancel_and_drop():
    started = asyncio.Event()
    release = asyncio.Event()

    async def execute_fn(request):
        started.set()
        await release.wait()
        return {"ok": request["task_id"]}

    async def run():
        manager = _fte_worker_task_manager(execute_fn)
        task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 1, "attempt_id": 0}
        status = await manager.create_task(
            {
                "task_id": task_id,
                "fragment_id": "q:node:scan",
                "initial_splits": {
                    "7": [{"sequence_id": 0, "kind": "scan_task", "data": b"a"}],
                },
            }
        )
        assert status["state"] == FteTaskState.RUNNING.value
        await started.wait()

        status = await manager.add_splits(
            task_id,
            "7",
            [
                {"sequence_id": 0, "kind": "scan_task", "data": b"duplicate"},
                {"sequence_id": 1, "kind": "scan_task", "data": b"b"},
            ],
        )
        assert status["duplicate_split_count"] == 1

        info = await manager.get_task_info(task_id)
        assert info["initial_split_counts"] == {"7": 2}

        release.set()
        for _ in range(50):
            status = await manager.get_task_status(task_id)
            if status["state"] == FteTaskState.FINISHED.value:
                break
            await asyncio.sleep(0.01)
        assert status["state"] == FteTaskState.FINISHED.value

        dropped = await manager.drop_query("q")
        assert dropped == {"removed": 1, "canceled": 0}

    asyncio.run(run())


def test_fte_worker_task_manager_unknown_status_is_non_throwing():
    async def execute_fn(request):
        return {"ok": request["task_id"]}

    async def run():
        manager = _fte_worker_task_manager(execute_fn)
        task_id = {"query_id": "q", "fragment_execution_id": 9, "partition_id": 3, "attempt_id": 0}
        status = await manager.get_task_status(task_id)
        assert status["state"] == "UNKNOWN"
        assert status["task_id_string"] == "q.9.3.0"
        with pytest.raises(KeyError, match="unknown FTE task attempt"):
            await manager.add_splits(task_id, "7", [])

    asyncio.run(run())


def test_fte_worker_task_manager_update_task_applies_task_update_subset():
    started = asyncio.Event()
    release = asyncio.Event()

    async def execute_fn(_request):
        started.set()
        await release.wait()
        return {"ok": True}

    async def run():
        manager = _fte_worker_task_manager(execute_fn)
        task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 5, "attempt_id": 0}
        await manager.create_task(
            {
                "task_id": task_id,
                "fragment_id": "q:node:scan",
                "initial_splits": {
                    "7": [{"sequence_id": 0, "kind": "scan_task", "data": b"a"}],
                },
            }
        )
        await started.wait()

        status = await manager.update_task(
            task_id,
            {
                "output_buffers": {"version": 1, "buffers": ["out-0"]},
                "dynamic_filter_domains": {"df0": {"single_value": 11}},
                "initial_splits": {
                    "7": [
                        {"sequence_id": 0, "kind": "scan_task", "data": b"dup"},
                        {"sequence_id": 1, "kind": "scan_task", "data": b"b"},
                    ],
                },
            },
        )
        assert status["duplicate_split_count"] == 1
        assert status["output_buffer_status"] == {
            "version": 1,
            "buffer_count": 1,
            "sealed": False,
        }
        status_version = status["version"]

        info = await manager.get_task_info(task_id)
        assert info["initial_split_counts"] == {"7": 2}
        assert info["output_buffers"] == {"version": 1, "buffers": ["out-0"]}
        assert info["output_buffer_status"] == {
            "version": 1,
            "buffer_count": 1,
            "sealed": False,
        }
        assert info["dynamic_filter_domains"] == {"df0": {"single_value": 11}}

        stale = await manager.update_task(
            task_id,
            {"output_buffers": {"version": 0, "buffers": ["stale"]}},
        )
        assert stale["version"] == status_version
        assert stale["output_buffer_status"]["version"] == 1
        with pytest.raises(ValueError, match="conflicting"):
            await manager.update_task(
                task_id,
                {"output_buffers": {"version": 1, "buffers": ["different"]}},
            )

        with pytest.raises(RuntimeError, match="descriptor fields"):
            await manager.update_task(task_id, {"context": {"too_late": True}})

        release.set()

    asyncio.run(run())


def test_fte_worker_task_manager_fte_update_before_execution_updates_descriptor_fields():
    executed = asyncio.Event()

    async def execute_fn(request):
        executed.set()
        assert request["context"]["trace"] == "fte"
        assert request["output_buffers"] == {"version": 2, "buffers": ["out-1"]}
        assert request["dynamic_filter_domains"] == {"df1": {"range": [3, 5]}}
        assert request["initial_splits"]["7"][0]["data"] == b"late"
        assert request["no_more_splits"] == ["7"]
        return {"ok": True}

    async def run():
        manager = _fte_worker_task_manager(execute_fn)
        task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 6, "attempt_id": 0}
        await manager.create_task(
            {
                "task_id": task_id,
                "fragment_id": "q:node:scan",
                "worker_runtime": "fte",
                "source_node_ids": ["7"],
            }
        )
        await asyncio.sleep(0)
        assert executed.is_set() is False

        await manager.update_task(
            task_id,
            {
                "context": {"trace": "fte"},
                "output_buffers": {"version": 2, "buffers": ["out-1"]},
                "dynamic_filter_domains": {"df1": {"range": [3, 5]}},
                "initial_splits": {
                    "7": [{"sequence_id": 0, "kind": "scan_task", "data": b"late"}],
                },
                "no_more_splits": ["7"],
            },
        )
        for _ in range(50):
            status = await manager.get_task_status(task_id)
            if status["state"] == FteTaskState.FINISHED.value:
                break
            await asyncio.sleep(0.01)
        assert status["state"] == FteTaskState.FINISHED.value
        assert executed.is_set() is True

    asyncio.run(run())


def test_fte_worker_task_manager_returns_spooling_output_stats(tmp_path):
    exchange = SpoolingExchangeManager(tmp_path, "q", "stage-0")
    sink = exchange.add_sink(0)
    attempt = exchange.instantiate_sink(sink, 0)

    async def execute_fn(request):
        assert request["exchange_sink_instance"] == attempt.to_dict()
        exchange.record_output_file(attempt, 0, "data", b"payload")
        exchange.finish_attempt(attempt)
        return {"ok": True}

    async def run():
        manager = _fte_worker_task_manager(execute_fn)
        task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 0, "attempt_id": 0}
        await manager.create_task(
            {
                "task_id": task_id,
                "fragment_id": "q:node:sink",
                "exchange_sink_instance": attempt.to_dict(),
            }
        )

        for _ in range(50):
            status = await manager.get_task_status(task_id)
            if status["state"] == FteTaskState.FINISHED.value:
                break
            await asyncio.sleep(0.01)

        assert status["state"] == FteTaskState.FINISHED.value
        assert status["spooling_output_stats"]["file_count"] == 1
        assert status["spooling_output_stats"]["total_bytes"] == len(b"payload")
        info = await manager.get_task_info(task_id)
        assert info["spooling_output_stats"] == status["spooling_output_stats"]

    asyncio.run(run())


def test_fte_worker_task_manager_fte_runtime_waits_for_no_more_splits():
    started = asyncio.Event()
    executed = asyncio.Event()

    async def execute_fn(request):
        executed.set()
        assert request["initial_splits"]["7"][0]["data"] == b"late"
        return {"ok": True}

    async def run():
        manager = _fte_worker_task_manager(execute_fn)
        task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 2, "attempt_id": 0}
        await manager.create_task(
            {
                "task_id": task_id,
                "fragment_id": "q:node:scan",
                "worker_runtime": "fte",
                "source_node_ids": ["7"],
            }
        )
        started.set()
        await asyncio.sleep(0)
        assert executed.is_set() is False

        await manager.add_splits(
            task_id,
            "7",
            [{"sequence_id": 0, "kind": "scan_task", "data": b"late"}],
        )
        await asyncio.sleep(0)
        assert executed.is_set() is False

        await manager.no_more_splits(task_id, "7")
        for _ in range(50):
            status = await manager.get_task_status(task_id)
            if status["state"] == FteTaskState.FINISHED.value:
                break
            await asyncio.sleep(0.01)
        assert status["state"] == FteTaskState.FINISHED.value
        assert executed.is_set() is True

    asyncio.run(run())


def test_fte_worker_task_manager_wait_task_status_wakes_on_terminal_state():
    executed = asyncio.Event()

    async def execute_fn(_request):
        executed.set()
        return {"ok": True}

    async def run():
        manager = _fte_worker_task_manager(execute_fn)
        task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 8, "attempt_id": 0}
        initial = await manager.create_task(
            {
                "task_id": task_id,
                "fragment_id": "q:node:scan",
            }
        )

        status = await asyncio.wait_for(
            manager.wait_task_status(task_id, initial["version"], timeout_s=1.0),
            timeout=2.0,
        )

        assert executed.is_set() is True
        assert status["state"] == FteTaskState.FINISHED.value
        assert status["version"] > initial["version"]
        assert status["result"] == {"ok": True}

    asyncio.run(run())


def test_fte_worker_task_manager_wait_task_status_times_out_with_current_status():
    async def execute_fn(_request):
        await asyncio.sleep(1.0)
        return {"ok": True}

    async def run():
        manager = _fte_worker_task_manager(execute_fn)
        task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 9, "attempt_id": 0}
        await manager.create_task(
            {
                "task_id": task_id,
                "fragment_id": "q:node:scan",
            }
        )

        status = await manager.wait_task_status(task_id, min_version=999, timeout_s=0.01)

        assert status["state"] == FteTaskState.RUNNING.value
        await manager.cancel_task(task_id)

    asyncio.run(run())


def test_fte_worker_task_manager_admits_tasks_through_worker_executor():
    started: list[str] = []
    release_first = asyncio.Event()

    async def execute_fn(request):
        attempt = FteTaskAttemptId.coerce(request["task_id"])
        started.append(str(attempt))
        if attempt.partition_id == 0:
            await release_first.wait()
        return {"partition": attempt.partition_id}

    async def run():
        manager = _fte_worker_task_manager(execute_fn, max_running_tasks=1)
        task0 = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 0, "attempt_id": 0}
        task1 = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 1, "attempt_id": 0}

        status0 = await manager.create_task({"task_id": task0, "fragment_id": "q:node:scan"})
        status1 = await manager.create_task({"task_id": task1, "fragment_id": "q:node:scan"})

        assert status0["state"] == FteTaskState.RUNNING.value
        assert status1["state"] == FteTaskState.QUEUED.value
        assert status1["executor_running_task_count"] == 1
        assert status1["executor_queued_task_count"] == 1
        assert status1["executor_queue_position"] == 0

        await asyncio.sleep(0)
        assert started == ["q.0.0.0"]

        release_first.set()
        for _ in range(50):
            status1 = await manager.get_task_status(task1)
            if "q.0.1.0" in started and status1["state"] == FteTaskState.FINISHED.value:
                break
            await asyncio.sleep(0.01)

        assert started == ["q.0.0.0", "q.0.1.0"]
        assert status1["state"] == FteTaskState.FINISHED.value
        assert status1["executor_running_task_count"] == 0
        assert status1["executor_queued_task_count"] == 0

    asyncio.run(run())


def test_fte_worker_task_manager_fte_runtime_uses_dynamic_exchange_source_queue():
    executed = asyncio.Event()
    captured = {}

    async def execute_fn(request):
        captured["request"] = request
        executed.set()
        await asyncio.sleep(0.05)
        return {"ok": True}

    async def run():
        manager = _fte_worker_task_manager(execute_fn)
        task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 4, "attempt_id": 0}
        await manager.create_task(
            {
                "task_id": task_id,
                "fragment_id": "q:node:exchange-source",
                "worker_runtime": "fte",
                "dynamic_exchange_source_node_ids": ["9"],
            }
        )

        await asyncio.wait_for(executed.wait(), timeout=1)
        queues = captured["request"]["fte_exchange_source_queues"]
        assert sorted(queues) == ["9"]
        queue = queues["9"]
        assert queue.try_get_next() == {"state": "BLOCKED"}

        await manager.add_splits(
            task_id,
            "9",
            [{"sequence_id": 0, "kind": "exchange_source_task", "data": b"binding"}],
        )
        assert queue.try_get_next() == {
            "state": "SPLIT",
            "kind": "exchange_source_task",
            "data": b"binding",
        }

        await manager.no_more_splits(task_id, "9")
        assert queue.try_get_next() == {"state": "FINISHED"}

    asyncio.run(run())


def test_fte_worker_task_manager_fte_dynamic_initial_split_is_queue_only():
    executed = asyncio.Event()
    captured = {}

    async def execute_fn(request):
        captured["request"] = request
        executed.set()
        await asyncio.sleep(0.05)
        return {"ok": True}

    async def run():
        manager = _fte_worker_task_manager(execute_fn)
        task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 6, "attempt_id": 0}
        await manager.create_task(
            {
                "task_id": task_id,
                "fragment_id": "q:node:exchange-source",
                "worker_runtime": "fte",
                "dynamic_exchange_source_node_ids": ["9"],
                "initial_splits": {
                    "9": [
                        {
                            "sequence_id": 0,
                            "kind": "exchange_source_task",
                            "data": b"initial-binding",
                        }
                    ],
                },
            }
        )

        await asyncio.wait_for(executed.wait(), timeout=1)
        request = captured["request"]
        assert request["initial_splits"] == {}

        queue = request["fte_exchange_source_queues"]["9"]
        assert queue.try_get_next() == {
            "state": "SPLIT",
            "kind": "exchange_source_task",
            "data": b"initial-binding",
        }
        assert queue.try_get_next() == {"state": "BLOCKED"}

    asyncio.run(run())


def test_fte_worker_task_manager_fte_runtime_uses_dynamic_scan_source_queue():
    executed = asyncio.Event()
    captured = {}

    async def execute_fn(request):
        captured["request"] = request
        executed.set()
        await asyncio.sleep(0.05)
        return {"ok": True}

    async def run():
        manager = _fte_worker_task_manager(execute_fn)
        task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 5, "attempt_id": 0}
        await manager.create_task(
            {
                "task_id": task_id,
                "fragment_id": "q:node:scan",
                "worker_runtime": "fte",
                "dynamic_scan_source_node_ids": ["7"],
            }
        )

        await asyncio.wait_for(executed.wait(), timeout=1)
        queues = captured["request"]["fte_scan_source_queues"]
        assert sorted(queues) == ["7"]
        queue = queues["7"]
        assert queue.try_get_next() == {"state": "BLOCKED"}

        await manager.add_splits(
            task_id,
            "7",
            [{"sequence_id": 0, "kind": "scan_task", "data": b"scan-binding"}],
        )
        assert queue.try_get_next() == {
            "state": "SPLIT",
            "kind": "scan_task",
            "data": b"scan-binding",
        }

        await manager.no_more_splits(task_id, "7")
        assert queue.try_get_next() == {"state": "FINISHED"}

    asyncio.run(run())


def test_fte_worker_task_manager_fte_dynamic_scan_with_explicit_source_waits_until_sealed():
    executed = asyncio.Event()
    captured = {}

    async def execute_fn(request):
        captured["request"] = request
        executed.set()
        return {"ok": True}

    async def run():
        manager = _fte_worker_task_manager(execute_fn)
        task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 12, "attempt_id": 0}
        await manager.create_task(
            {
                "task_id": task_id,
                "fragment_id": "q:node:scan",
                "worker_runtime": "fte",
                "source_node_ids": ["7"],
                "dynamic_scan_source_node_ids": ["7"],
                "initial_splits": {
                    "7": [
                        {
                            "sequence_id": 0,
                            "kind": "scan_task",
                            "data": b"initial-scan",
                        }
                    ],
                },
            }
        )
        await asyncio.sleep(0)
        assert executed.is_set() is False

        await manager.add_splits(
            task_id,
            "7",
            [{"sequence_id": 1, "kind": "scan_task", "data": b"late-scan"}],
        )
        await asyncio.sleep(0)
        assert executed.is_set() is False

        await manager.no_more_splits(task_id, "7")
        await asyncio.wait_for(executed.wait(), timeout=1)
        status = await manager.wait_task_status(task_id, 0, timeout_s=1.0)
        assert status["state"] == FteTaskState.FINISHED.value

        request = captured["request"]
        assert request["initial_splits"] == {}
        queue = request["fte_scan_source_queues"]["7"]
        assert queue.try_get_next() == {
            "state": "SPLIT",
            "kind": "scan_task",
            "data": b"initial-scan",
        }
        assert queue.try_get_next() == {
            "state": "SPLIT",
            "kind": "scan_task",
            "data": b"late-scan",
        }
        assert queue.try_get_next() == {"state": "FINISHED"}

    asyncio.run(run())


def test_fte_worker_task_manager_wait_split_queue_has_space_tracks_buffered_splits():
    executed = asyncio.Event()
    captured = {}

    async def execute_fn(request):
        captured["request"] = request
        executed.set()
        await asyncio.sleep(0.2)
        return {"ok": True}

    async def run():
        manager = _fte_worker_task_manager(execute_fn)
        task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 10, "attempt_id": 0}
        await manager.create_task(
            {
                "task_id": task_id,
                "fragment_id": "q:node:scan",
                "dynamic_scan_source_node_ids": ["7"],
                "split_queue_max_buffered_splits": 1,
            }
        )

        await asyncio.wait_for(executed.wait(), timeout=1)
        queue = captured["request"]["fte_scan_source_queues"]["7"]
        await manager.add_splits(
            task_id,
            "7",
            [{"sequence_id": 0, "kind": "scan_task", "data": b"scan-binding"}],
        )
        full = await manager.wait_split_queue_has_space(
            task_id,
            source_node_id="7",
            max_buffered_splits=1,
            timeout_s=0.01,
        )
        assert full["has_space"] is False
        assert full["buffered_splits"] == 1
        assert full["buffered_bytes"] == len(b"scan-binding")
        assert full["status"]["queued_split_count"] == 1
        assert full["status"]["queued_split_bytes"] == len(b"scan-binding")
        assert full["status"]["queued_split_weight"] == 1
        assert full["status"]["split_queue_has_space"] is False

        assert queue.try_get_next()["state"] == "SPLIT"
        space = await manager.wait_split_queue_has_space(
            task_id,
            source_node_id="7",
            max_buffered_splits=1,
            timeout_s=0.1,
        )
        assert space["has_space"] is True
        assert space["buffered_splits"] == 0
        assert space["buffered_bytes"] == 0

        await manager.no_more_splits(task_id, "7")

    asyncio.run(run())


def test_fte_worker_task_manager_fte_runtime_cancel_wakes_blocked_task():
    async def execute_fn(_request):
        raise AssertionError("FTE task should not execute after cancel")

    async def run():
        manager = _fte_worker_task_manager(execute_fn)
        task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 3, "attempt_id": 0}
        await manager.create_task(
            {
                "task_id": task_id,
                "fragment_id": "q:node:scan",
                "worker_runtime": "fte",
                "source_node_ids": ["7"],
            }
        )
        await asyncio.sleep(0)
        status = await manager.cancel_task(task_id)
        assert status["state"] == FteTaskState.CANCELED.value

        dropped = await manager.drop_query("q")
        assert dropped == {"removed": 1, "canceled": 0}

    asyncio.run(run())


def test_fte_worker_task_manager_cancel_running_task():
    async def execute_fn(_request):
        await asyncio.sleep(60)

    async def run():
        manager = _fte_worker_task_manager(execute_fn)
        task_id = "q.0.0.0"
        await manager.create_task({"task_id": task_id, "fragment_id": "q:node:scan"})
        status = await manager.cancel_task(task_id)
        assert status["state"] == FteTaskState.CANCELED.value
        dropped = await manager.drop_query("q")
        assert dropped == {"removed": 1, "canceled": 0}

    asyncio.run(run())


def test_fte_worker_task_manager_cancel_finished_loser_releases_result_immediately():
    result_payload = {"partition_refs": ["loser-object-ref"]}

    async def execute_fn(_request):
        return result_payload

    async def run():
        manager = _fte_worker_task_manager(execute_fn)
        task_id = "q.0.11.1"
        initial = await manager.create_task({"task_id": task_id, "fragment_id": "q:node:scan"})
        finished = await manager.wait_task_status(
            task_id,
            min_version=initial["version"],
            timeout_s=1.0,
        )
        assert finished["state"] == FteTaskState.FINISHED.value
        assert finished["result"] is result_payload

        canceled_loser = await manager.cancel_task(task_id)

        assert canceled_loser["state"] == FteTaskState.FINISHED.value
        assert "result" not in canceled_loser
        info = await manager.get_task_info(task_id)
        assert info["result"] is None
        assert manager.tasks[task_id].result is None
        assert manager.tasks[task_id]._result_release_count == 1

    asyncio.run(run())


def test_fte_worker_task_manager_drop_query_preserves_terminal_status_for_long_poll():
    started = asyncio.Event()
    release = asyncio.Event()

    async def execute_fn(_request):
        started.set()
        await release.wait()
        return {"ok": True}

    async def run():
        manager = _fte_worker_task_manager(execute_fn)
        task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 7, "attempt_id": 0}
        initial = await manager.create_task({"task_id": task_id, "fragment_id": "q:node:scan"})
        assert initial["state"] == FteTaskState.RUNNING.value
        await asyncio.wait_for(started.wait(), timeout=1)

        waiter = asyncio.create_task(manager.wait_task_status(task_id, min_version=initial["version"], timeout_s=10.0))
        await asyncio.sleep(0)

        dropped = await manager.drop_query("q")
        assert dropped == {"removed": 1, "canceled": 1}

        waited = await asyncio.wait_for(waiter, timeout=1)
        assert waited["state"] == FteTaskState.CANCELED.value
        assert waited["task_id_string"] == "q.0.7.0"

        status = await manager.get_task_status(task_id)
        assert status["state"] == FteTaskState.CANCELED.value
        assert status["task_id_string"] == "q.0.7.0"

        late_wait = await manager.wait_task_status(
            task_id,
            min_version=initial["version"],
            timeout_s=0.01,
        )
        assert late_wait["state"] == FteTaskState.CANCELED.value

        late_cancel = await manager.cancel_task(task_id)
        assert late_cancel["state"] == FteTaskState.CANCELED.value

        info = await manager.get_task_info(task_id)
        assert info["status"]["state"] == FteTaskState.CANCELED.value
        assert info["status"]["task_id_string"] == "q.0.7.0"
        assert info["initial_split_counts"] == {}

    asyncio.run(run())


def test_fte_worker_task_manager_drop_query_retains_failed_tasks_for_retry():
    async def execute_fn(_request):
        await asyncio.sleep(60)

    async def run():
        manager = _fte_worker_task_manager(execute_fn)
        task0 = "q.0.0.0"
        task1 = "q.0.1.0"
        await manager.create_task({"task_id": task0, "fragment_id": "q:node:scan"})
        await manager.create_task({"task_id": task1, "fragment_id": "q:node:scan"})

        execution0 = manager.tasks[task0]
        execution1 = manager.tasks[task1]
        original_cancel0 = execution0.cancel
        original_cancel1 = execution1.cancel
        cancel_calls: list[str] = []

        def fail_cancel0():
            cancel_calls.append(task0)
            raise RuntimeError("planned cancel failure")

        def cancel1():
            cancel_calls.append(task1)
            return original_cancel1()

        execution0.cancel = fail_cancel0
        execution1.cancel = cancel1

        with pytest.raises(RuntimeError, match="planned cancel failure"):
            await manager.drop_query("q")

        assert set(cancel_calls) == {task0, task1}
        assert manager.query_tasks == {"q": {task0}}
        assert set(manager.tasks) == {task0}
        assert task1 in manager.dropped_task_statuses

        execution0.cancel = original_cancel0
        assert await manager.drop_query("q") == {"removed": 1, "canceled": 1}
        assert "q" not in manager.query_tasks
        assert manager.tasks == {}

    asyncio.run(run())


def test_fte_worker_task_manager_drop_query_clears_queued_task_tombstones():
    started_first = asyncio.Event()
    started_next_query = asyncio.Event()
    release_next_query = asyncio.Event()
    executed: list[str] = []

    async def execute_fn(request):
        attempt = FteTaskAttemptId.coerce(request["task_id"])
        executed.append(str(attempt))
        if attempt.query_id == "q" and attempt.partition_id == 0:
            started_first.set()
            await asyncio.Event().wait()
        if attempt.query_id == "q" and attempt.partition_id == 1:
            raise AssertionError("dropped queued task should not execute")
        if attempt.query_id == "q2":
            started_next_query.set()
            await release_next_query.wait()
        return {"task": str(attempt)}

    async def run():
        manager = _fte_worker_task_manager(execute_fn, max_running_tasks=1)
        task0 = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 0, "attempt_id": 0}
        task1 = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 1, "attempt_id": 0}
        task2 = {"query_id": "q2", "fragment_execution_id": 0, "partition_id": 0, "attempt_id": 0}

        status0 = await manager.create_task({"task_id": task0, "fragment_id": "q:node:scan"})
        status1 = await manager.create_task({"task_id": task1, "fragment_id": "q:node:scan"})
        assert status0["state"] == FteTaskState.RUNNING.value
        assert status1["state"] == FteTaskState.QUEUED.value
        assert status1["executor_queued_task_count"] == 1
        await asyncio.wait_for(started_first.wait(), timeout=1)

        dropped = await manager.drop_query("q")
        assert dropped == {"removed": 2, "canceled": 2}

        assert (await manager.get_task_status(task0))["state"] == FteTaskState.CANCELED.value
        status1 = await manager.get_task_status(task1)
        assert status1["state"] == FteTaskState.CANCELED.value
        assert status1["executor_queued_task_count"] == 0

        status2 = await manager.create_task({"task_id": task2, "fragment_id": "q2:node:scan"})
        assert status2["state"] == FteTaskState.RUNNING.value
        assert status2["executor_queued_task_count"] == 0
        await asyncio.wait_for(started_next_query.wait(), timeout=1)

        assert executed == ["q.0.0.0", "q2.0.0.0"]
        release_next_query.set()
        final = await asyncio.wait_for(
            manager.wait_task_status(task2, status2["version"], timeout_s=1.0),
            timeout=2.0,
        )
        assert final["state"] == FteTaskState.FINISHED.value

    asyncio.run(run())


def test_fte_worker_task_manager_recreate_task_clears_dropped_status_index():
    async def execute_fn(_request):
        return {"ok": True}

    async def run():
        manager = _fte_worker_task_manager(execute_fn)
        task_id = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 2, "attempt_id": 0}
        status = await manager.create_task({"task_id": task_id, "fragment_id": "q:node:scan"})
        await asyncio.wait_for(manager.wait_task_status(task_id, status["version"], timeout_s=1.0), timeout=2.0)

        assert await manager.drop_query("q") == {"removed": 1, "canceled": 0}
        assert "q.0.2.0" in manager.dropped_task_statuses
        assert "q.0.2.0" in manager.dropped_task_order

        await manager.create_task({"task_id": task_id, "fragment_id": "q:node:scan"})

        assert "q.0.2.0" not in manager.dropped_task_statuses
        assert "q.0.2.0" not in manager.dropped_task_order

    asyncio.run(run())


def test_fte_worker_task_manager_explicit_admission_reports_memory_stats():
    release_first = asyncio.Event()

    async def execute_fn(request):
        attempt = FteTaskAttemptId.coerce(request["task_id"])
        if attempt.partition_id == 0:
            await release_first.wait()
        return {"partition": attempt.partition_id}

    async def run():
        manager = _fte_worker_task_manager(
            execute_fn,
            admission_config=FteWorkerAdmissionConfig(
                max_running_tasks=4,
                mode="test",
                memory_budget_bytes=40 * _GIB,
                task_memory_bytes=10 * _GIB,
            ),
        )
        task0 = {"query_id": "q", "fragment_execution_id": 0, "partition_id": 0, "attempt_id": 0}
        status0 = await manager.create_task({"task_id": task0, "fragment_id": "q:node:scan"})

        assert status0["executor_admission_mode"] == "test"
        assert status0["executor_max_running_tasks"] == 4
        assert status0["executor_memory_budget_bytes"] == 40 * 1024 * 1024 * 1024
        assert status0["executor_task_memory_bytes"] == 10 * 1024 * 1024 * 1024
        assert status0["executor_reserved_memory_bytes"] == 10 * 1024 * 1024 * 1024

        release_first.set()
        await asyncio.wait_for(manager.wait_task_status(task0, status0["version"], timeout_s=1.0), timeout=2.0)

    asyncio.run(run())


def test_fte_worker_task_manager_explicit_admission_uses_task_memory_requirement():
    started: list[str] = []
    release_first = asyncio.Event()

    async def execute_fn(request):
        attempt = FteTaskAttemptId.coerce(request["task_id"])
        started.append(str(attempt))
        if attempt.partition_id == 0:
            await release_first.wait()
        return {"partition": attempt.partition_id}

    async def run():
        manager = _fte_worker_task_manager(
            execute_fn,
            admission_config=FteWorkerAdmissionConfig(
                max_running_tasks=4,
                mode="test",
                memory_budget_bytes=20,
                task_memory_bytes=10,
            ),
        )
        task0 = {"query_id": "qmem", "fragment_execution_id": 0, "partition_id": 0, "attempt_id": 0}
        task1 = {"query_id": "qmem", "fragment_execution_id": 0, "partition_id": 1, "attempt_id": 0}

        status0 = await manager.create_task(
            {
                "task_id": task0,
                "fragment_id": "qmem:node:scan",
                "memory_requirement_bytes": 15,
            }
        )
        status1 = await manager.create_task(
            {
                "task_id": task1,
                "fragment_id": "qmem:node:scan",
                "memory_requirement_bytes": 10,
            }
        )

        assert status0["state"] == FteTaskState.RUNNING.value
        assert status0["executor_reserved_memory_bytes"] == 15
        assert status0["executor_task_memory_requirement_bytes"] == 15
        assert status1["state"] == FteTaskState.QUEUED.value
        assert status1["executor_queue_position"] == 0
        assert status1["executor_reserved_memory_bytes"] == 15
        assert status1["executor_task_memory_requirement_bytes"] == 10
        await asyncio.sleep(0)
        assert started == ["qmem.0.0.0"]

        release_first.set()
        for _ in range(50):
            status1 = await manager.get_task_status(task1)
            if "qmem.0.1.0" in started and status1["state"] == FteTaskState.FINISHED.value:
                break
            await asyncio.sleep(0.01)

        assert started == ["qmem.0.0.0", "qmem.0.1.0"]
        assert status1["state"] == FteTaskState.FINISHED.value
        assert status1["executor_reserved_memory_bytes"] == 0

    asyncio.run(run())


def test_ray_fte_worker_requires_exact_query_lease_heap():
    async def execute_fn(request):
        return {"ok": str(FteTaskAttemptId.coerce(request["task_id"]))}

    async def run():
        manager = _fte_worker_task_manager(
            execute_fn,
            admission_config=FteWorkerAdmissionConfig(
                max_running_tasks=4,
                mode="lease",
                memory_budget_bytes=20,
                task_memory_bytes=None,
            ),
            require_query_task_lease=True,
        )
        task_id = {
            "query_id": "qlease",
            "fragment_execution_id": 0,
            "partition_id": 0,
            "attempt_id": 0,
        }
        with pytest.raises(RuntimeError, match="requires query_task_lease"):
            await manager.create_task({"task_id": task_id, "fragment_id": "qlease:node:scan"})

        request = {
            "task_id": task_id,
            "fragment_id": "qlease:node:scan",
            "memory_requirement_bytes": 9,
            "query_task_lease": {
                "lease_id": "lease-qlease",
                "query_id": "qlease",
                "execution_query_id": "qlease",
                "stage_id": "stage:qlease:node:scan:fte",
                "task_id": "qlease.0.0",
                "attempt_id": "qlease.0.0.0",
                "resources": {
                    "cpu": 1.0,
                    "gpu": 0.0,
                    "heap_bytes": 10,
                    "object_store_bytes": 0,
                },
            },
        }
        with pytest.raises(RuntimeError, match="diverges from query task lease"):
            await manager.create_task(request)

        request["memory_requirement_bytes"] = 10
        status = await manager.create_task(request)
        assert status["executor_admission_mode"] == "lease"
        assert status["executor_memory_budget_bytes"] == 20
        assert status["executor_task_memory_requirement_bytes"] == 10
        final = await manager.wait_task_status(task_id, status["version"], timeout_s=1.0)
        assert final["state"] == FteTaskState.FINISHED.value

        await manager.drop_query("qlease")

    asyncio.run(run())


def test_fte_worker_manager_requires_explicit_capacity():
    async def execute_fn(_request):
        return {"ok": True}

    with pytest.raises(TypeError):
        _ProductionFteWorkerTaskManager(execute_fn)


def test_ray_fte_worker_rejects_lease_larger_than_node_capacity():
    async def execute_fn(_request):
        return {"ok": True}

    async def run():
        manager = _fte_worker_task_manager(
            execute_fn,
            admission_config=FteWorkerAdmissionConfig(
                max_running_tasks=4,
                mode="lease",
                memory_budget_bytes=20,
                task_memory_bytes=None,
            ),
            require_query_task_lease=True,
        )
        task_id = {
            "query_id": "qoversize",
            "fragment_execution_id": 0,
            "partition_id": 0,
            "attempt_id": 0,
        }
        request = {
            "task_id": task_id,
            "fragment_id": "qoversize:node:scan",
            "memory_requirement_bytes": 21,
            "query_task_lease": {
                "lease_id": "lease-qoversize",
                "query_id": "qoversize",
                "execution_query_id": "qoversize",
                "stage_id": "stage:qoversize:node:scan:fte",
                "task_id": "qoversize.0.0",
                "attempt_id": "qoversize.0.0.0",
                "resources": {
                    "cpu": 1.0,
                    "gpu": 0.0,
                    "heap_bytes": 21,
                    "object_store_bytes": 0,
                },
            },
        }

        with pytest.raises(RuntimeError, match="exceeds worker task-heap capacity"):
            await manager.create_task(request)

    asyncio.run(run())
