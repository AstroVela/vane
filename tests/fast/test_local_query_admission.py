# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pyarrow as pa
import pytest

from tests.local_admission_helpers import linear_metadata
from vane.execution.local_query_admission import LocalQueryAdmission, build_local_admission_graph
from vane.execution.local_query_coordinator import LocalQueryCoordinator
from vane.execution.query_resource_policy import OutputBlockRequest, QueryResourceManager, TaskRequest
from vane.execution.query_resource_spec import QueryAllocation, QueryResourceGraph
from vane.execution.resources import ResourceVector
from vane.execution.udf_admission import LocalExecutionCapacity, LocalExecutionSlotPool
from vane.execution.udf_lifecycle import ExecutionCancellationScope
from vane.execution.udf_local_actor_admission import LocalActorExecutionSlotPool
from vane.execution.udf_resource_policy import udf_resource_spec


def _graph(backend="subprocess_task", *, query_id="q", memory=None, actor_count=1):
    nodes = []
    for i in range(2):
        payload = {
            "execution_backend": backend,
            "actor_pool_size": actor_count,
            "udf_task_input_max_bytes": 64,
            "udf_output_target_max_bytes": 64,
        }
        if memory is not None:
            payload["memory_bytes"] = memory
        nodes.append({"node_id": str(i), "payload": payload})
    return build_local_admission_graph(
        linear_metadata(nodes), {n["node_id"]: n["payload"] for n in nodes}, query_id=query_id, env={}
    )


def _allocation(cpu=8, heap=1024, store=4096):
    return QueryAllocation(ResourceVector(cpu=cpu, heap_bytes=heap, object_store_bytes=store), 1)


def _authority(query, unit_id, pool):
    physical = pool.create_authority()
    return query.create_authority(unit_id, physical, physical)


def _take(authority, retained=16):
    ready = threading.Event()
    authority.register_wakeup(ready.set)
    assert authority.request(retained)
    if not authority.state()["available"]:
        assert ready.wait(3), authority.state()
    return authority.take(retained)


@pytest.mark.parametrize("heap", [None, 100])
@pytest.mark.parametrize("actor", [False, True])
@pytest.mark.parametrize("cpu", [3, 4])
def test_default_resource_declarations_and_event_trace_match_ray(heap, actor, cpu):
    graph, bindings = _graph("subprocess_actor" if actor else "subprocess_task", memory=heap)
    units = [unit for unit in graph.units if unit.execution_kind != "native"]
    local_units, ray_units = [], []
    for index, unit in enumerate(units):
        # Compare declarations produced from the original unmodified defaults,
        # not an observed local heap cost copied into a Ray memory request.
        options = {
            "execution_backend": "ray_actor" if actor else "ray_task",
            "actor_pool_size": 1,
            "udf_task_input_max_bytes": 64,
            "udf_output_target_max_bytes": 64,
        }
        if heap is not None:
            options["memory_bytes"] = heap
        ray_unit = udf_resource_spec(
            query_id="q",
            resource_unit_id=unit.resource_unit_id,
            physical_node_id=unit.physical_node_id,
            input_unit_ids=(),
            payload=options,
            env={},
        )
        assert unit.per_task == ray_unit.per_task
        assert unit.resident_per_actor == ray_unit.resident_per_actor
        assert unit.actor_prefetch_depth == ray_unit.actor_prefetch_depth == (2 if actor else 1)
        if heap is None:
            assert unit.per_task.heap_bytes == unit.resident_per_actor.heap_bytes == 0
        inputs = () if index == 0 else (units[index - 1].resource_unit_id,)
        local_units.append(replace(unit, input_unit_ids=inputs))
        ray_units.append(replace(ray_unit, input_unit_ids=inputs))
    managers = []
    for specs in (local_units, ray_units):
        shape = QueryResourceGraph("q", "digest", tuple(specs), (specs[-1].resource_unit_id,))
        manager = QueryResourceManager(shape, _allocation(cpu=cpu, heap=350, store=512))
        for unit in shape.units:
            manager.update_unit_state(unit.resource_unit_id, runnable=True)
            if actor:
                manager.set_submitted_actor_slots(unit.resource_unit_id, {0})
                manager.set_ready_actor_slots(unit.resource_unit_id, {0: "node"})
        manager.set_external_consumer_waiting(True)
        managers.append(manager)
    leases = [[], []]
    outputs = [[], []]
    for sequence, index in enumerate((0, 0, 0, 1, 1, 0, 1)):
        outcomes = []
        for side, manager in enumerate(managers):
            unit_id = units[index].resource_unit_id
            grant = manager.try_acquire_task(
                TaskRequest("q", unit_id, str(sequence), "0", None, retained_input_bytes=32)
            )
            outcomes.append((grant.granted, grant.blocked_reason, grant.liveness))
            if grant.granted:
                leases[side].append(grant.lease)
                result = manager.try_acquire_output_block(
                    OutputBlockRequest(
                        "q",
                        unit_id,
                        grant.lease.lease_id,
                        "0",
                        f"block-{sequence}",
                        size_bytes=80,
                    )
                )
                outcomes.append((result.granted, result.blocked_reason, result.liveness))
                if result.granted:
                    outputs[side].append(result.lease)
        assert outcomes[: len(outcomes) // 2] == outcomes[len(outcomes) // 2 :]
        assert managers[0].snapshot()["usage"] == managers[1].snapshot()["usage"]
    for side, manager in enumerate(managers):
        for lease in leases[side]:
            manager.release_task_lease(lease.lease_id, attempt_id="0")
        for lease in outputs[side]:
            manager.release_output_block(lease.lease_id)
    assert managers[0].snapshot()["usage"] == managers[1].snapshot()["usage"]


def test_undeclared_heap_does_not_calibrate_or_reduce_default_task_concurrency():
    graph, bindings = _graph()
    graph = replace(graph, units=graph.units[:2], terminal_unit_ids=(bindings["0"],))
    query = LocalQueryAdmission(graph, _allocation(cpu=4, heap=1))
    capacity = LocalExecutionCapacity(max_slots=None, resource_limit=ResourceVector(cpu=4, heap_bytes=1))
    pool = LocalExecutionSlotPool(
        max_slots=4, execution_slot_prefix="task", execution_capacity=capacity, resources=ResourceVector(cpu=1)
    )
    authority = _authority(query, bindings["0"], pool)
    leases = []
    try:
        for _ in range(4):
            leases.append(_take(authority))
        assert capacity.reserved_slots == 4
        assert query.snapshot()["usage"]["heap_bytes"] == 0
        leases.pop().complete_execution()
        leases.append(_take(authority))
        assert capacity.reserved_slots == 4
    finally:
        for lease in leases:
            lease.release()
        authority.close()
        query.shutdown()
        pool.close()
    assert capacity.reserved_slots == 0


def test_ready_callback_can_submit_again_without_holding_admission_locks():
    graph, bindings = _graph()
    query = LocalQueryAdmission(graph, _allocation(), coordinator=LocalQueryCoordinator())
    pool = LocalExecutionSlotPool(max_slots=2, execution_slot_prefix="task")
    authority = _authority(query, bindings["0"], pool)
    leases = []
    completed = threading.Event()

    def ready():
        if not authority.state()["available"]:
            return
        leases.append(authority.take(16))
        if len(leases) == 1:
            authority.request(16)
        else:
            completed.set()

    authority.register_wakeup(ready)
    try:
        authority.request(16)
        assert completed.wait(3)
        assert len(leases) == 2
    finally:
        authority.register_wakeup(None)
        for lease in leases:
            lease.release()
        authority.close()
        query.shutdown()
        pool.close()


def test_actor_prefetch_releases_execution_before_buffered_output_and_is_bounded():
    graph, bindings = _graph("subprocess_actor")
    query = LocalQueryAdmission(graph, _allocation())
    pool = LocalActorExecutionSlotPool(1, execution_slot_prefix="actor")
    authorities = [_authority(query, bindings["0"], pool) for _ in range(3)]
    leases = []
    try:
        leases = [_take(authority) for authority in authorities[:2]]
        assert [lease.lease["actor_index"] for lease in leases] == [0, 0]
        assert pool.active_lease_count == 2
        ready = threading.Event()
        authorities[2].register_wakeup(ready.set)
        assert authorities[2].request(16)
        assert not authorities[2].state()["available"]
        leases[0].complete_execution()
        assert ready.wait(3)
        leases.append(authorities[2].take(16))
        assert pool.active_lease_count == 2
        # Releasing the old buffered terminal cannot release the new call.
        leases[0].release()
        assert pool.active_lease_count == 2
    finally:
        for lease in leases:
            lease.release()
        for authority in authorities:
            authority.close()
        query.shutdown()
        pool.close()


def test_shared_actor_pool_uses_idle_replica_across_query_policies():
    pool = LocalActorExecutionSlotPool(2, execution_slot_prefix="shared")
    coordinator = LocalQueryCoordinator()
    queries, authorities, leases = [], [], []
    try:
        for name in ("first", "second"):
            graph, bindings = _graph("subprocess_actor", query_id=name, actor_count=2)
            query = LocalQueryAdmission(graph, _allocation(), coordinator=coordinator)
            queries.append(query)
            authorities.append(_authority(query, bindings["0"], pool))
        start = threading.Barrier(2)

        def take(authority):
            start.wait(3)
            return _take(authority)

        with ThreadPoolExecutor(max_workers=2) as threads:
            leases = list(threads.map(take, authorities))
        assert sorted(lease.lease["actor_index"] for lease in leases) == [0, 1]
    finally:
        for lease in leases:
            lease.release()
        for authority in authorities:
            authority.close()
        for query in queries:
            query.shutdown()
        pool.close()


def test_cancel_queued_output_wakes_and_preserves_other_query_owners():
    graph, bindings = _graph()
    query = LocalQueryAdmission(graph, _allocation(store=1))
    pool = LocalExecutionSlotPool(max_slots=2, execution_slot_prefix="task")
    authority = _authority(query, bindings["0"], pool)
    lease = _take(authority)
    scope = ExecutionCancellationScope("output-test", 1)
    first = query.own_output(lease.lease["query_policy_task"], 100, scope, lease)
    errors = []
    started = threading.Event()

    def wait_output():
        started.set()
        try:
            query.own_output(lease.lease["query_policy_task"], 100, scope, lease)
        except BaseException as error:
            errors.append(error)

    waiter = threading.Thread(target=wait_output)
    waiter.start()
    assert started.wait(3)
    scope.cancel("test cancellation")
    waiter.join(3)
    try:
        assert not waiter.is_alive()
        assert errors and "cancel" in str(errors[0]).lower()
        assert query.snapshot()["usage"]["object_store_bytes"] >= 100
    finally:
        first.release()
        lease.release()
        authority.close()
        query.shutdown()
        pool.close()


def test_query_output_bytes_follow_zero_copy_view_after_task_and_query_close():
    from vane.execution.ref_bundle import make_local_shm_ref_bundle_result

    graph, bindings = _graph()
    query = LocalQueryAdmission(graph, _allocation(store=1024 * 1024))
    pool = LocalExecutionSlotPool(max_slots=1, execution_slot_prefix="task")
    authority = _authority(query, bindings["0"], pool)
    lease = _take(authority)
    result = make_local_shm_ref_bundle_result(pa.table({"x": list(range(2048))}))
    ref = result[1][0]
    size = ref.size
    owner = query.own_output(lease.lease["query_policy_task"], size, ExecutionCancellationScope("view", 1), lease)
    ref.attach_query_output_lease(owner)
    view = ref.to_table()
    ref.release()
    lease.release()
    authority.close()
    query.shutdown()
    pool.close()
    assert query.snapshot()["usage"]["object_store_bytes"] == size
    assert view["x"][17].as_py() == 17
    del view
    gc.collect()
    assert query.snapshot()["usage"]["object_store_bytes"] == 0


def test_concurrent_local_queries_share_and_return_the_ray_query_allocation():
    coordinator = LocalQueryCoordinator()
    graph, _ = _graph(query_id="one")
    first = LocalQueryAdmission(graph, _allocation(), coordinator=coordinator)
    graph, _ = _graph(query_id="two")
    second = LocalQueryAdmission(graph, _allocation(), coordinator=coordinator)
    try:
        assert first.manager.allocation.resources.cpu == second.manager.allocation.resources.cpu == 4
        assert first.manager.allocation.resources.object_store_bytes == 2048
        second.shutdown()
        assert first.manager.allocation.resources.cpu == 8
        assert first.manager.allocation.resources.object_store_bytes == 4096
    finally:
        second.shutdown()
        first.shutdown()


@pytest.mark.parametrize("second_device, expected_capacity", [("gpu-0", 1), ("gpu-1", 2)])
def test_concurrent_gpu_queries_count_distinct_devices(second_device, expected_capacity):
    coordinator = LocalQueryCoordinator()
    queries = []
    try:
        for name, device in (("one", "gpu-0"), ("two", second_device)):
            graph, bindings = _graph("subprocess_actor", query_id=name)
            actor = replace(graph.units[1], resident_per_actor=ResourceVector(cpu=1, gpu=1))
            graph = replace(graph, units=(graph.units[0], actor), terminal_unit_ids=(bindings["0"],))
            allocation = replace(_allocation(), resources=replace(_allocation().resources, gpu=1))
            queries.append(LocalQueryAdmission(graph, allocation, coordinator=coordinator, gpu_devices=(device,)))
        assert coordinator._nodes[0].resources.gpu == expected_capacity
        assert sum(query.manager.allocation.resources.gpu for query in queries) == expected_capacity
        queries.pop().shutdown()
        assert coordinator._nodes[0].resources.gpu == 1
        assert queries[0].manager.allocation.resources.gpu == 1
    finally:
        for query in queries:
            query.shutdown()


def test_local_query_honors_shared_reservation_and_refresh_configuration(monkeypatch):
    monkeypatch.setenv("VANE_QUERY_RESOURCE_RESERVATION_RATIO", "0.25")
    monkeypatch.setenv("VANE_QUERY_RESOURCE_REFRESH_INTERVAL_S", "0.2")
    coordinator = LocalQueryCoordinator()
    graph, _ = _graph()
    query = LocalQueryAdmission(graph, _allocation(), coordinator=coordinator)
    try:
        assert query.manager.reservation_ratio == 0.25
        assert coordinator.refresh_interval_s == 0.2
    finally:
        query.shutdown()


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
def test_native_barrier_frontier_reopens_when_downstream_receives_materialized_input(backend):
    nodes = [{"node_id": str(i), "payload": {"execution_backend": backend, "actor_pool_size": 1}} for i in range(2)]
    metadata = linear_metadata(nodes)
    metadata["nodes"][1]["is_materialization_barrier"] = True
    metadata["nodes"][1]["materialized_input_node_ids"] = ["0"]
    graph, bindings = build_local_admission_graph(
        metadata, {n["node_id"]: n["payload"] for n in nodes}, query_id="barrier", env={}
    )
    query = LocalQueryAdmission(graph, _allocation(), coordinator=LocalQueryCoordinator())
    pool = (
        LocalActorExecutionSlotPool(1, execution_slot_prefix="actor")
        if backend == "subprocess_actor"
        else LocalExecutionSlotPool(max_slots=1, execution_slot_prefix="task")
    )
    authority = _authority(query, bindings["1"], pool)
    try:
        assert bindings["1"] not in query.manager.current_eligible_resource_unit_ids()
        lease = _take(authority)
        assert bindings["1"] in query.manager.current_eligible_resource_unit_ids()
        lease.release()
    finally:
        authority.close()
        query.shutdown()
        pool.close()


@pytest.mark.parametrize("configured", [False, True])
def test_native_local_query_uses_shared_policy_by_default(monkeypatch, configured):
    import vane
    from vane.execution import udf_subprocess

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    captured = []
    original = udf_subprocess.ensure_local_subprocess_actor_pools_for_nodes

    def prepare(*args, **kwargs):
        resources, options = original(*args, **kwargs)
        captured.extend(owner for owner in resources if isinstance(owner, LocalQueryAdmission))
        return resources, options

    monkeypatch.setattr(udf_subprocess, "ensure_local_subprocess_actor_pools_for_nodes", prepare)

    def identity(table):
        return table

    with vane.connect() as connection:
        if configured:
            from vane.execution.request_admission import RequestAdmissionLimits

            connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 0))
        relation = connection.sql("SELECT i AS x FROM range(100) t(i)").map_batches(
            identity,
            schema={"x": vane.sqltypes.BIGINT},
            execution_backend="subprocess_task",
        )
        assert sorted(row[0] for row in relation.fetchall()) == list(range(100))
    assert captured
    assert all(type(query.manager) is QueryResourceManager for query in captured)
    assert all(not query.cleanup_pending() for query in captured)
    assert all(query.snapshot()["usage"]["heap_bytes"] == 0 for query in captured)


def test_production_completion_waits_for_active_execution_but_retains_output_bytes():
    graph, bindings = _graph()
    query = LocalQueryAdmission(graph, _allocation())
    pool = LocalExecutionSlotPool(max_slots=1, execution_slot_prefix="task")
    authority = _authority(query, bindings["0"], pool)
    lease = _take(authority)
    owner = query.own_output(lease.lease["query_policy_task"], 50, ExecutionCancellationScope("finish", 1), lease)
    try:
        authority.finished_submitting()
        assert bindings["0"] not in query._completed_units
        lease.complete_execution()
        assert bindings["0"] in query._completed_units
        assert query.snapshot()["usage"]["object_store_bytes"] == 50
        owner.release()
        assert query.snapshot()["usage"]["object_store_bytes"] == 0
    finally:
        owner.release()
        lease.release()
        query.shutdown()
        pool.close()


def test_query_shutdown_retries_failed_ready_backend_cleanup():
    graph, bindings = _graph()
    query = LocalQueryAdmission(graph, _allocation())
    pool = LocalExecutionSlotPool(max_slots=1, execution_slot_prefix="task")
    physical = pool.create_authority()
    original_close = physical.close
    failures = [True]

    def close():
        if failures:
            failures.pop()
            raise RuntimeError("backend close failed")
        original_close()

    physical.close = close
    authority = query.create_authority(bindings["0"], physical, physical)
    assert authority.request(16)
    assert authority.state()["available"]
    with pytest.raises(RuntimeError, match="backend close failed"):
        query.shutdown()
    assert query.cleanup_pending()
    assert query.snapshot()["usage"]["cpu"] == 1
    query.shutdown()
    assert not query.cleanup_pending()
    assert query.snapshot()["usage"]["cpu"] == 0
    pool.close()


def test_stale_query_demand_cannot_reopen_newer_completion_frontier(monkeypatch):
    from vane.execution import local_query_coordinator as module

    coordinator = LocalQueryCoordinator()
    graph, bindings = _graph()
    query = LocalQueryAdmission(graph, _allocation(), coordinator=coordinator)
    original = module.build_query_demand
    changes = [True]

    def change_frontier(*args, **kwargs):
        result = original(*args, **kwargs)
        if changes:
            changes.pop()
            query.manager.update_unit_state(bindings["0"], runnable=False, completed=True)
        return result

    try:
        # Keep the automatic pump out until the in-flight old demand returns.
        with query._pump_lock, coordinator._lock:
            monkeypatch.setattr(module, "build_query_demand", change_frontier)
            coordinator._refresh()
            assert query.snapshot()["allocation_admission_open"] is False
            coordinator._refresh()
            assert query.snapshot()["allocation_admission_open"] is True
    finally:
        query.shutdown()
