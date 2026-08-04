# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from duckdb.runners.ray import query_resource_manager as manager_module
from duckdb.runners.ray.query_resource_graph import (
    MaterializationBarrierSpec,
    NodeResourceAllocation,
    QueryAllocation,
    QueryResourceGraph,
    ResourceUnitSpec,
    ResourceVector,
)
from duckdb.runners.ray.query_resource_manager import (
    OutputBlockRequest,
    RayQueryResourceManager,
    TaskRequest,
)


def _r(*, cpu=0.0, gpu=0.0, heap=0, store=0):
    return ResourceVector(cpu=cpu, gpu=gpu, heap_bytes=heap, object_store_bytes=store)


def _allocation(resources, *, generation=1, nodes=None):
    node_resources = (("node-a", resources),) if nodes is None else tuple(nodes)
    return QueryAllocation(
        resources=resources,
        node_allocations=tuple(
            NodeResourceAllocation(node_id=node_id, resources=node_capacity)
            for node_id, node_capacity in node_resources
        ),
        generation=generation,
    )


def _unit(
    resource_unit_id,
    *,
    inputs=(),
    resources=None,
    target=10,
    blocks=2,
    concurrency=100,
    backend="ray_task",
    actor_pool_size=0,
    actor_prefetch_depth=1,
    resident=None,
    unit_kind=None,
):
    requested = resources or _r(cpu=1, heap=10)
    if backend == "ray_actor":
        resident_resources = resident or _r(
            cpu=requested.cpu,
            gpu=requested.gpu,
            heap=requested.heap_bytes,
        )
        task_resources = _r(store=requested.object_store_bytes)
    elif backend == "ray_worker":
        resident_resources = _r()
        task_resources = _r(store=requested.object_store_bytes)
    else:
        resident_resources = _r()
        task_resources = requested
    resolved_unit_kind = (
        unit_kind
        or {
            "ray_worker": "native_fragment",
            "ray_task": "ray_task_udf",
            "ray_actor": "ray_actor_pool",
        }[backend]
    )
    return ResourceUnitSpec(
        query_id="q",
        resource_unit_id=resource_unit_id,
        physical_node_id=resource_unit_id.rsplit(":", 1)[-1],
        unit_kind=resolved_unit_kind,
        backend=backend,
        input_unit_ids=tuple(inputs),
        per_task=task_resources,
        target_output_block_bytes=target,
        generator_buffer_blocks=blocks,
        max_concurrency=concurrency if backend == "ray_worker" else None,
        resident_per_actor=resident_resources,
        actor_pool_size=actor_pool_size,
        actor_prefetch_depth=actor_prefetch_depth,
    )


def _manager(
    *units,
    resources=None,
    reservation_ratio=0.5,
    terminals=None,
    nodes=None,
    on_change=None,
    barriers=(),
    on_eligible_units_change=None,
):
    graph = QueryResourceGraph(
        query_id="q",
        plan_digest="sha256:test",
        units=tuple(units),
        terminal_unit_ids=tuple(terminals or (units[-1].resource_unit_id,)),
        materialization_barriers=tuple(barriers),
    )
    allocation_resources = resources or _r(cpu=100, gpu=1, heap=1_000, store=1_000)
    allocation = _allocation(
        allocation_resources,
        nodes=nodes,
    )
    graph.validate_allocation(allocation)
    return RayQueryResourceManager(
        graph,
        allocation,
        reservation_ratio=reservation_ratio,
        on_change=on_change,
        on_eligible_units_change=on_eligible_units_change,
    )


def _barrier(node_id, unit, *, materialized_inputs=None):
    return MaterializationBarrierSpec(
        query_id="q",
        barrier_id=f"barrier:q:node:{node_id}",
        physical_node_id=str(node_id),
        materializer_unit_id=unit.resource_unit_id,
        materialized_input_unit_ids=(
            unit.input_unit_ids if materialized_inputs is None else tuple(materialized_inputs)
        ),
    )


def _ready(manager, *unit_ids, consumer_waiting=False):
    for resource_unit_id in unit_ids:
        unit = manager.graph.unit_by_id(resource_unit_id)
        if unit.backend == "ray_actor":
            actor_indices = set(range(unit.actor_pool_size))
            manager.set_submitted_actor_slots(resource_unit_id, actor_indices)
            manager.set_ready_actor_slots(
                resource_unit_id,
                {actor_index: "node-a" for actor_index in actor_indices},
            )
        manager.update_unit_state(
            resource_unit_id,
            runnable=True,
        )
    manager.set_external_consumer_waiting(consumer_waiting)


def _task(resource_unit_id, partition, attempt="0", retained=None, node_id=None):
    return TaskRequest(
        query_id="q",
        resource_unit_id=resource_unit_id,
        task_id=f"task:{resource_unit_id}:partition:{partition}",
        attempt_id=str(attempt),
        node_id=node_id,
        retained_input_bytes=retained,
    )


def test_task_admission_requires_runnable_registered_unit_and_ready_actor():
    actor = _unit(
        "resource:f:gpu",
        resources=_r(cpu=1, gpu=1, heap=100),
        backend="ray_actor",
        concurrency=1,
        actor_pool_size=1,
    )
    manager = _manager(actor, resources=_r(cpu=2, gpu=1, heap=500, store=500))
    manager.set_submitted_actor_slots(actor.resource_unit_id, {0})

    not_runnable = manager.try_acquire_task(_task(actor.resource_unit_id, 0))
    manager.set_ready_actor_slots(actor.resource_unit_id, {})
    manager.update_unit_state(actor.resource_unit_id, runnable=True)
    not_ready = manager.try_acquire_task(_task(actor.resource_unit_id, 0))
    manager.set_ready_actor_slots(actor.resource_unit_id, {0: "node-a"})
    granted = manager.try_acquire_task(_task(actor.resource_unit_id, 0))

    assert not_runnable.blocked_reason == "unit_not_runnable"
    assert not_ready.blocked_reason == "actor_not_ready"
    assert granted.granted
    assert granted.lease.resources == actor.per_task
    assert granted.lease.output_window_bytes == 20


def test_dynamic_reservations_follow_real_runnable_demand_without_static_scope():
    upstream = _unit("resource:f:upstream", resources=_r(cpu=1, heap=10, store=5), target=5, blocks=1)
    downstream = _unit(
        "resource:f:downstream",
        inputs=(upstream.resource_unit_id,),
        resources=_r(cpu=1, heap=10, store=5),
        target=5,
        blocks=1,
    )
    manager = _manager(upstream, downstream, resources=_r(cpu=2, heap=20, store=20))
    _ready(manager, upstream.resource_unit_id, downstream.resource_unit_id)
    manager.note_task_waiting(_task(upstream.resource_unit_id, "waiting"))
    manager.note_task_waiting(_task(downstream.resource_unit_id, "waiting"))

    snapshot = manager.snapshot()

    assert "reservation_scope" not in snapshot
    assert snapshot["admission"]["reservation_unit_ids"]["cpu"] == [
        upstream.resource_unit_id,
        downstream.resource_unit_id,
    ]
    assert snapshot["admission"]["reservation_unit_ids"]["object_store_bytes"] == [
        upstream.resource_unit_id,
        downstream.resource_unit_id,
    ]


def test_barrier_completion_retires_old_eligible_units_and_opens_next_phase():
    upstream = _unit(
        "resource:f:upstream",
        resources=_r(cpu=1, heap=10),
        target=0,
        blocks=0,
    )
    materializer = _unit(
        "resource:f:materializer",
        inputs=(upstream.resource_unit_id,),
        resources=_r(),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    downstream = _unit(
        "resource:f:downstream",
        inputs=(materializer.resource_unit_id,),
        resources=_r(cpu=1, heap=20),
        target=0,
        blocks=0,
    )
    barrier = _barrier("materializer", materializer)
    transitions = []
    manager = _manager(
        upstream,
        materializer,
        downstream,
        resources=_r(cpu=2, heap=30),
        barriers=(barrier,),
        on_eligible_units_change=transitions.append,
    )
    _ready(
        manager,
        upstream.resource_unit_id,
        materializer.resource_unit_id,
        downstream.resource_unit_id,
    )

    blocked_downstream = manager.try_acquire_task(_task(downstream.resource_unit_id, 0))
    assert blocked_downstream.blocked_reason == "materialization_barrier_pending"

    assert manager.mark_materialization_barrier_completed_for_node("materializer") is True
    assert manager.mark_materialization_barrier_completed_for_node("materializer") is False
    assert transitions == [(materializer.resource_unit_id, downstream.resource_unit_id)]
    assert manager.current_eligible_resource_unit_ids() == (
        materializer.resource_unit_id,
        downstream.resource_unit_id,
    )

    retired_upstream = manager.try_acquire_task(_task(upstream.resource_unit_id, 0))
    opened_downstream = manager.try_acquire_task(_task(downstream.resource_unit_id, 0))
    assert retired_upstream.blocked_reason == "allocation_pending"
    assert opened_downstream.blocked_reason == "allocation_pending"
    manager.update_allocation(
        _allocation(_r(cpu=2, heap=30), generation=2),
        admission_open=True,
    )
    retired_upstream = manager.try_acquire_task(_task(upstream.resource_unit_id, 0))
    final_materializer = manager.try_acquire_task(_task(materializer.resource_unit_id, 0))
    opened_downstream = manager.try_acquire_task(_task(downstream.resource_unit_id, 0))
    assert retired_upstream.blocked_reason == "materialization_barrier_pending"
    assert final_materializer.granted
    assert opened_downstream.granted
    assert manager.snapshot()["execution_phase"] == {
        "frontier_barrier_ids": [],
        "eligible_resource_unit_ids": [
            materializer.resource_unit_id,
            downstream.resource_unit_id,
        ],
        "completed_barrier_ids": [barrier.barrier_id],
        "object_store_unlimited_unit_ids": [],
    }


def test_unit_completion_fences_old_allocation_until_eligible_demand_refreshes():
    finished = _unit("resource:f:finished", target=0, blocks=0)
    remaining = _unit("resource:f:remaining", target=0, blocks=0)
    transitions = []
    manager = _manager(
        finished,
        remaining,
        resources=_r(cpu=2, heap=20),
        terminals=(finished.resource_unit_id, remaining.resource_unit_id),
        on_eligible_units_change=transitions.append,
    )
    _ready(manager, finished.resource_unit_id, remaining.resource_unit_id)

    manager.update_unit_state(
        finished.resource_unit_id,
        runnable=False,
        completed=True,
    )

    assert transitions == [(remaining.resource_unit_id,)]
    assert manager.snapshot()["allocation_admission_open"] is False
    assert manager.try_acquire_task(_task(remaining.resource_unit_id, 0)).blocked_reason == "allocation_pending"

    manager.update_allocation(
        _allocation(_r(cpu=2, heap=20), generation=2),
        admission_open=True,
    )
    assert manager.try_acquire_task(_task(remaining.resource_unit_id, 0)).granted


def test_actor_task_leases_own_distinct_concrete_actor_slots():
    actor = _unit(
        "resource:f:gpu",
        resources=_r(store=40),
        resident=_r(cpu=1, gpu=1, heap=100),
        backend="ray_actor",
        concurrency=None,
        actor_pool_size=2,
    )
    manager = _manager(
        actor,
        resources=_r(cpu=2, gpu=2, heap=200, store=500),
    )
    _ready(manager, actor.resource_unit_id)

    first = manager.try_acquire_task(_task(actor.resource_unit_id, 0, retained=20))
    second = manager.try_acquire_task(_task(actor.resource_unit_id, 1, retained=20))
    blocked = manager.try_acquire_task(_task(actor.resource_unit_id, 2, retained=20))

    assert first.granted and second.granted
    assert {first.lease.actor_index, second.lease.actor_index} == {0, 1}
    assert {
        first.lease.execution_slot_id,
        second.lease.execution_slot_id,
    } == {
        f"ray_actor:{actor.resource_unit_id}:0",
        f"ray_actor:{actor.resource_unit_id}:1",
    }
    assert blocked.granted is False
    assert blocked.blocked_reason == "actor_slot"

    assert manager.release_task_lease(first.lease.lease_id, attempt_id="0")
    replacement = manager.try_acquire_task(_task(actor.resource_unit_id, 2, retained=20))
    assert replacement.granted
    assert replacement.lease.actor_index == first.lease.actor_index

    manager.cancel("test cleanup")
    assert manager.snapshot()["active_actor_slots"] == {}


def test_actor_prefetch_depth_queues_one_call_per_concrete_actor():
    actor = _unit(
        "resource:f:gpu-prefetch",
        resources=_r(store=40),
        resident=_r(cpu=1, gpu=1, heap=100),
        backend="ray_actor",
        actor_pool_size=2,
        actor_prefetch_depth=2,
    )
    manager = _manager(
        actor,
        resources=_r(cpu=2, gpu=2, heap=200, store=1_000),
    )
    _ready(manager, actor.resource_unit_id)

    grants = [manager.try_acquire_task(_task(actor.resource_unit_id, partition, retained=20)) for partition in range(5)]

    assert all(grant.granted for grant in grants[:4])
    assert [grant.lease.actor_index for grant in grants[:4]] == [0, 1, 0, 1]
    assert not grants[4].granted
    assert grants[4].blocked_reason == "actor_slot"
    snapshot = manager.snapshot()
    assert snapshot["active_actor_slots"] == {
        f"{actor.resource_unit_id}:0": grants[0].lease.lease_id,
        f"{actor.resource_unit_id}:1": grants[1].lease.lease_id,
    }
    assert snapshot["queued_actor_slots"] == {
        f"{actor.resource_unit_id}:0": [grants[2].lease.lease_id],
        f"{actor.resource_unit_id}:1": [grants[3].lease.lease_id],
    }

    assert manager.release_task_lease(grants[0].lease.lease_id, attempt_id="0")
    snapshot = manager.snapshot()
    assert snapshot["active_actor_slots"][f"{actor.resource_unit_id}:0"] == grants[2].lease.lease_id
    assert f"{actor.resource_unit_id}:0" not in snapshot["queued_actor_slots"]

    assert manager.release_task_lease(grants[3].lease.lease_id, attempt_id="0")
    assert f"{actor.resource_unit_id}:1" not in manager.snapshot()["queued_actor_slots"]


def test_actor_ready_slot_cannot_disappear_while_it_owns_prefetched_work():
    actor = _unit(
        "resource:f:actor-ready-fence",
        resources=_r(store=10),
        resident=_r(cpu=1, heap=100),
        backend="ray_actor",
        actor_pool_size=1,
        actor_prefetch_depth=2,
    )
    manager = _manager(
        actor,
        resources=_r(cpu=1, heap=100, store=100),
    )
    manager.set_submitted_actor_slots(actor.resource_unit_id, {0})
    manager.set_ready_actor_slots(actor.resource_unit_id, {0: "node-a"})
    manager.update_unit_state(actor.resource_unit_id, runnable=True)
    active = manager.try_acquire_task(_task(actor.resource_unit_id, 0))
    queued = manager.try_acquire_task(_task(actor.resource_unit_id, 1))

    assert active.granted and queued.granted
    with pytest.raises(RuntimeError, match="ready actor slot with live leases"):
        manager.set_ready_actor_slots(actor.resource_unit_id, {})

    assert manager.release_task_lease(
        queued.lease.lease_id,
        attempt_id=queued.lease.attempt_id,
    )
    assert manager.release_task_lease(
        active.lease.lease_id,
        attempt_id=active.lease.attempt_id,
    )
    manager.set_ready_actor_slots(actor.resource_unit_id, {})


def test_actor_pool_retirement_is_phase_fenced_and_charged_until_shutdown():
    actor = _unit(
        "resource:f:actor-before-barrier",
        resources=_r(),
        resident=_r(cpu=1, heap=100),
        backend="ray_actor",
        actor_pool_size=1,
    )
    materializer = _unit(
        "resource:f:materializer",
        inputs=(actor.resource_unit_id,),
        resources=_r(),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    downstream = _unit(
        "resource:f:after-barrier",
        inputs=(materializer.resource_unit_id,),
        resources=_r(cpu=1),
        target=0,
        blocks=0,
    )
    manager = _manager(
        actor,
        materializer,
        downstream,
        resources=_r(cpu=2, heap=100, store=100),
        barriers=(_barrier("materializer", materializer),),
    )
    manager.set_submitted_actor_slots(actor.resource_unit_id, {0})
    manager.set_ready_actor_slots(actor.resource_unit_id, {0: "node-a"})

    assert manager.begin_actor_pool_retirement(actor.resource_unit_id) is False
    assert manager.mark_materialization_barrier_completed_for_node("materializer")
    assert manager.begin_actor_pool_retirement(actor.resource_unit_id) is True
    retiring = manager.snapshot()
    assert retiring["retiring_actor_unit_ids"] == [actor.resource_unit_id]
    assert retiring["ready_actor_slots"] == {}
    assert retiring["actor_process_usage"] == _r(cpu=1, heap=100).to_dict()
    with pytest.raises(RuntimeError, match="submitted slots for a retiring actor pool"):
        manager.set_submitted_actor_slots(actor.resource_unit_id, set())

    assert manager.complete_actor_pool_retirement(actor.resource_unit_id) is True
    retired = manager.snapshot()
    assert retired["retiring_actor_unit_ids"] == []
    assert retired["submitted_actor_slots"] == []
    assert retired["actor_process_usage"] == _r().to_dict()


def test_pending_actor_retirement_publishes_both_lifecycle_edges():
    actor = _unit(
        "resource:f:pending-actor",
        resources=_r(),
        resident=_r(cpu=1),
        backend="ray_actor",
        actor_pool_size=1,
    )
    materializer = _unit(
        "resource:f:pending-materializer",
        inputs=(actor.resource_unit_id,),
        resources=_r(),
        target=0,
        blocks=0,
        backend="ray_worker",
    )
    downstream = _unit(
        "resource:f:pending-downstream",
        inputs=(materializer.resource_unit_id,),
        resources=_r(),
        target=0,
        blocks=0,
        backend="ray_worker",
    )
    manager = _manager(
        actor,
        materializer,
        downstream,
        barriers=(_barrier("pending-materializer", materializer),),
    )
    manager.set_submitted_actor_slots(actor.resource_unit_id, {0})
    assert manager.mark_materialization_barrier_completed_for_node("pending-materializer")

    before_retirement = manager.admission_epoch()
    assert manager.begin_actor_pool_retirement(actor.resource_unit_id) is True
    assert manager.admission_epoch() == before_retirement + 1
    assert manager.begin_actor_pool_retirement(actor.resource_unit_id) is True
    assert manager.admission_epoch() == before_retirement + 1

    assert manager.complete_actor_pool_retirement(actor.resource_unit_id) is True
    assert manager.admission_epoch() == before_retirement + 2


def test_ray_tasks_receive_unique_resource_lease_slots():
    unit = _unit("resource:f:cpu", concurrency=None)
    manager = _manager(unit)
    _ready(manager, unit.resource_unit_id)

    first = manager.try_acquire_task(_task(unit.resource_unit_id, 0))
    second = manager.try_acquire_task(_task(unit.resource_unit_id, 1))

    assert first.granted and second.granted
    assert first.lease.execution_slot_id != second.lease.execution_slot_id
    assert first.lease.execution_slot_id == (f"ray_task:{unit.resource_unit_id}:{first.lease.lease_id}")


def test_idle_actor_process_resources_are_in_soft_allocation_usage():
    actor = _unit(
        "resource:f:gpu",
        resources=_r(store=40),
        resident=_r(cpu=1, gpu=1, heap=100),
        backend="ray_actor",
        concurrency=None,
        actor_pool_size=1,
    )
    manager = _manager(
        actor,
        resources=_r(cpu=1, gpu=1, heap=100, store=200),
    )
    _ready(manager, actor.resource_unit_id)

    snapshot = manager.snapshot()
    assert snapshot["usage"] == _r().to_dict()
    assert snapshot["node_usage"]["node-a"] == _r().to_dict()
    assert snapshot["soft_allocation_usage"] == _r(cpu=1, gpu=1, heap=100).to_dict()
    assert snapshot["actor_process_usage"] == _r(cpu=1, gpu=1, heap=100).to_dict()
    assert (
        snapshot["actor_process_usage_by_ready_node"]["node-a"]
        == _r(
            cpu=1,
            gpu=1,
            heap=100,
        ).to_dict()
    )


def test_persistent_soft_actor_debt_warns_once_after_ray_data_delay(monkeypatch, caplog):
    actor = _unit(
        "resource:f:oversubscribed-actor",
        resources=_r(store=40),
        resident=_r(cpu=1, gpu=1, heap=100),
        backend="ray_actor",
        concurrency=None,
        actor_pool_size=1,
    )
    graph = QueryResourceGraph(
        query_id="q",
        plan_digest="sha256:test",
        units=(actor,),
        terminal_unit_ids=(actor.resource_unit_id,),
    )
    allocation = _allocation(
        _r(cpu=0.5, gpu=0.5, heap=50, store=200),
    )
    manager = RayQueryResourceManager(graph, allocation)
    manager.set_submitted_actor_slots(actor.resource_unit_id, {0})
    clock = [100.0]
    monkeypatch.setattr(manager_module.time, "monotonic", lambda: clock[0])

    with caplog.at_level(logging.WARNING, logger=manager_module.__name__):
        first = manager.snapshot()
        clock[0] += 59.9
        manager.snapshot()
        assert caplog.records == []
        clock[0] += 0.1
        warned = manager.snapshot()
        clock[0] += 60
        manager.snapshot()

    assert first["soft_allocation_debt"] == _r(cpu=0.5, gpu=0.5, heap=50).to_dict()
    assert warned["soft_allocation_debt_duration_s"] == pytest.approx(60.0)
    assert warned["soft_allocation_warning_emitted"] is True
    assert len(caplog.records) == 1
    assert "new work can be backpressured" in caplog.records[0].getMessage()


def test_ready_actor_outside_virtual_node_allocation_accepts_zero_increment_call():
    actor = _unit(
        "resource:f:external-actor",
        resources=_r(store=10),
        resident=_r(cpu=1, heap=100),
        backend="ray_actor",
        actor_pool_size=1,
    )
    graph = QueryResourceGraph(
        query_id="q",
        plan_digest="sha256:test",
        units=(actor,),
        terminal_unit_ids=(actor.resource_unit_id,),
    )
    allocation = _allocation(
        _r(cpu=1, heap=100, store=100),
        nodes=(("virtual-node", _r(cpu=1, heap=100, store=100)),),
    )
    manager = RayQueryResourceManager(graph, allocation)
    manager.set_submitted_actor_slots(actor.resource_unit_id, {0})
    manager.set_ready_actor_slots(actor.resource_unit_id, {0: "actual-node"})
    manager.update_unit_state(actor.resource_unit_id, runnable=True)

    grant = manager.try_acquire_task(_task(actor.resource_unit_id, 0))

    assert grant.granted and not grant.liveness
    assert grant.lease.node_id == "actual-node"
    assert grant.lease.actor_index == 0
    snapshot = manager.snapshot()
    assert snapshot["node_usage"]["actual-node"] == _r(store=10).to_dict()
    assert (
        snapshot["actor_process_usage_by_ready_node"]["actual-node"]
        == _r(
            cpu=1,
            heap=100,
        ).to_dict()
    )
    assert snapshot["node_allocation_debt"]["actual-node"] == _r().to_dict()


def test_actor_process_soft_debt_does_not_recharge_existing_actor_invocations():
    actor = _unit(
        "resource:f:soft-debt-actor",
        resources=_r(),
        resident=_r(cpu=1, gpu=1, heap=100),
        backend="ray_actor",
        actor_pool_size=1,
    )
    graph = QueryResourceGraph(
        query_id="q",
        plan_digest="sha256:test",
        units=(actor,),
        terminal_unit_ids=(actor.resource_unit_id,),
    )
    allocation = _allocation(
        _r(cpu=0.5, gpu=0.5, heap=50, store=100),
    )
    manager = RayQueryResourceManager(graph, allocation)
    manager.set_submitted_actor_slots(actor.resource_unit_id, {0})
    manager.set_ready_actor_slots(actor.resource_unit_id, {0: "node-a"})
    manager.update_unit_state(actor.resource_unit_id, runnable=True)

    first = manager.try_acquire_task(_task(actor.resource_unit_id, 0))
    second = manager.try_acquire_task(_task(actor.resource_unit_id, 1))

    assert first.granted and not first.liveness
    assert first.lease.actor_index == 0
    assert not second.granted
    assert second.blocked_reason == "actor_slot"
    assert manager.release_task_lease(
        first.lease.lease_id,
        attempt_id=first.lease.attempt_id,
    )
    replacement = manager.try_acquire_task(_task(actor.resource_unit_id, 1))
    assert replacement.granted and not replacement.liveness


def test_submitted_actor_debt_blocks_new_ray_process_but_not_actor_invocation():
    actor = _unit(
        "resource:f:oversubscribed-actor",
        resources=_r(),
        resident=_r(cpu=1, heap=100),
        backend="ray_actor",
        actor_pool_size=2,
    )
    ray_task = _unit(
        "resource:f:new-ray-task",
        resources=_r(cpu=1, heap=100),
        target=0,
        blocks=0,
    )
    manager = _manager(
        actor,
        ray_task,
        resources=_r(cpu=1, heap=100, store=100),
        terminals=(actor.resource_unit_id, ray_task.resource_unit_id),
    )
    _ready(manager, actor.resource_unit_id, ray_task.resource_unit_id)

    invocation = manager.try_acquire_task(_task(actor.resource_unit_id, 0))
    new_process = manager.try_acquire_task(_task(ray_task.resource_unit_id, 0))
    snapshot = manager.snapshot()

    assert invocation.granted and not invocation.liveness
    assert not new_process.granted
    assert new_process.blocked_reason == "normal_candidate_available"
    assert snapshot["soft_allocation_usage"] == _r(cpu=2, heap=200).to_dict()
    assert snapshot["allocation_debt"] == _r().to_dict()
    assert snapshot["soft_allocation_debt"] == _r(cpu=1, heap=100).to_dict()


def test_soft_reservation_only_divides_each_dimension_among_units_that_need_it():
    cpu_units = []
    for index in range(10):
        cpu_units.append(
            _unit(
                f"resource:f:cpu-{index}",
                inputs=() if not cpu_units else (cpu_units[-1].resource_unit_id,),
                resources=_r(cpu=1, heap=10),
                target=0,
                blocks=0,
            )
        )
    gpu_unit = _unit(
        "resource:f:gpu",
        inputs=(cpu_units[-1].resource_unit_id,),
        resources=_r(gpu=1, heap=20),
        target=0,
        blocks=0,
        backend="ray_actor",
        actor_pool_size=1,
    )
    manager = _manager(
        *cpu_units,
        gpu_unit,
        resources=_r(cpu=10, gpu=1, heap=1_000, store=100),
    )
    _ready(manager, *(unit.resource_unit_id for unit in (*cpu_units, gpu_unit)))

    grant = manager.try_acquire_task(_task(gpu_unit.resource_unit_id, 0))

    assert grant.granted
    assert grant.lease.resources.gpu == 0
    assert gpu_unit.resident_per_actor.gpu == 1


def test_unit_minimum_commitments_are_protected_before_shared_heap_borrowing():
    cpu_units = []
    for index in range(10):
        cpu_units.append(
            _unit(
                f"resource:f:cpu-{index}",
                inputs=() if not cpu_units else (cpu_units[-1].resource_unit_id,),
                resources=_r(cpu=1, heap=20),
                target=0,
                blocks=0,
            )
        )
    gpu_unit = _unit(
        "resource:f:gpu",
        inputs=(cpu_units[-1].resource_unit_id,),
        resources=_r(gpu=1, heap=40),
        target=0,
        blocks=0,
        backend="ray_actor",
        actor_pool_size=1,
    )
    manager = _manager(
        *cpu_units,
        gpu_unit,
        resources=_r(cpu=100, gpu=1, heap=240, store=100),
    )
    _ready(manager, *(unit.resource_unit_id for unit in (*cpu_units, gpu_unit)))

    first_upstream = manager.try_acquire_task(_task(cpu_units[0].resource_unit_id, 0))
    second_upstream = manager.try_acquire_task(_task(cpu_units[0].resource_unit_id, 1))
    downstream = manager.try_acquire_task(_task(gpu_unit.resource_unit_id, 0))

    assert first_upstream.granted
    assert not second_upstream.granted
    assert downstream.granted


def test_actor_process_heap_consumes_shared_soft_credit_before_new_ray_tasks():
    gpu_actor = _unit(
        "resource:f:gpu-actor",
        resources=_r(gpu=1, heap=4),
        target=0,
        blocks=0,
        concurrency=None,
        backend="ray_actor",
        actor_pool_size=1,
    )
    cpu_unit = _unit(
        "resource:f:cpu",
        inputs=(gpu_actor.resource_unit_id,),
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        concurrency=100,
    )
    manager = _manager(
        gpu_actor,
        cpu_unit,
        resources=_r(cpu=100, gpu=1, heap=20, store=20),
    )
    _ready(manager, gpu_actor.resource_unit_id, cpu_unit.resource_unit_id)

    actor_grant = manager.try_acquire_task(_task(gpu_actor.resource_unit_id, 0))
    cpu_grants = [manager.try_acquire_task(_task(cpu_unit.resource_unit_id, partition)) for partition in range(8)]
    denied = manager.try_acquire_task(_task(cpu_unit.resource_unit_id, 8))

    assert actor_grant.granted
    assert all(grant.granted for grant in cpu_grants)
    assert not denied.granted
    assert denied.blocked_reason == "liveness_not_needed"
    assert (
        manager.snapshot()["soft_allocation_usage"]
        == _r(
            cpu=8,
            gpu=1,
            heap=20,
        ).to_dict()
    )


def test_native_fragment_uses_fte_node_outside_query_allocation_attribution():
    native = _unit(
        "resource:f:native-external-node",
        resources=_r(store=10),
        target=10,
        blocks=2,
        backend="ray_worker",
    )
    manager = _manager(
        native,
        resources=_r(store=100),
        nodes=(("node-a", _r(store=100)),),
    )
    _ready(manager, native.resource_unit_id)

    task_grant = manager.try_acquire_task(
        _task(native.resource_unit_id, 0, node_id="node-b"),
    )

    assert task_grant.granted
    assert task_grant.lease.node_id == "node-b"
    output_grant = manager.try_acquire_output_block(
        OutputBlockRequest(
            query_id="q",
            producer_unit_id=native.resource_unit_id,
            task_lease_id=task_grant.lease.lease_id,
            attempt_id=task_grant.lease.attempt_id,
            block_id="native-external-output",
            size_bytes=30,
        )
    )
    assert output_grant.granted
    snapshot = manager.snapshot()
    assert snapshot["allocation"]["node_allocations"] == [
        {"node_id": "node-a", "resources": _r(store=100).to_dict()},
    ]
    # Native publication keeps its bounded FTE window until completion, and
    # the exposed ObjectRef is an additional exact logical byte count.
    assert snapshot["node_usage"]["node-b"] == _r(store=60).to_dict()
    assert snapshot["node_allocation_debt"]["node-b"] == _r().to_dict()


def test_external_native_node_keeps_remote_continuation_on_allocated_node():
    native = _unit(
        "resource:f:native-external-parent",
        resources=_r(),
        target=0,
        blocks=0,
        backend="ray_worker",
    )
    remote = _unit(
        "resource:f:remote-continuation",
        inputs=(native.resource_unit_id,),
        resources=_r(cpu=1, heap=10),
        target=0,
        blocks=0,
    )
    manager = _manager(
        native,
        remote,
        resources=_r(cpu=1, heap=10),
        nodes=(("node-a", _r(cpu=1, heap=10)),),
    )
    _ready(manager, native.resource_unit_id, remote.resource_unit_id)

    native_grant = manager.try_acquire_task(
        _task(native.resource_unit_id, 0, node_id="node-b"),
    )

    assert native_grant.granted
    assert native_grant.lease.node_id == "node-b"
    credits = manager.snapshot()["continuation_credits"]
    assert len(credits) == 1
    assert next(iter(credits.values()))["node_id"] == "node-a"
    remote_grant = manager.try_acquire_task(_task(remote.resource_unit_id, 0, node_id="node-a"))
    assert remote_grant.granted
    assert remote_grant.lease.node_id == "node-a"


def test_empty_runnable_units_do_not_dilute_live_task_reservation():
    units = []
    for index in range(4):
        units.append(
            _unit(
                f"resource:f:demand-{index}",
                inputs=() if not units else (units[-1].resource_unit_id,),
                resources=_r(cpu=1, heap=2),
                target=0,
                blocks=0,
            )
        )
    live_unit = units[-1]
    manager = _manager(
        *units,
        resources=_r(cpu=8, heap=8, store=8),
    )
    for unit in units:
        manager.update_unit_state(
            unit.resource_unit_id,
            runnable=True,
        )

    requests = [_task(live_unit.resource_unit_id, index) for index in range(3)]
    for request in requests:
        manager.note_task_waiting(request)

    grants = [manager.try_acquire_task(request) for request in requests]

    assert all(grant.granted for grant in grants)
    snapshot = manager.snapshot()
    assert snapshot["units"][live_unit.resource_unit_id]["active_task_count"] == 3
    assert all(
        snapshot["units"][unit.resource_unit_id]["active_task_count"] == 0 for unit in units if unit is not live_unit
    )


def test_queued_admission_prefers_downstream_over_upstream_refill():
    upstream = _unit(
        "resource:f:upstream-refill",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
    )
    downstream = _unit(
        "resource:f:downstream-work",
        inputs=(upstream.resource_unit_id,),
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
    )
    manager = _manager(
        upstream,
        downstream,
        resources=_r(cpu=12, heap=12, store=12),
    )
    manager.update_unit_state(upstream.resource_unit_id, runnable=True)

    held = [manager.try_acquire_task(_task(upstream.resource_unit_id, index)) for index in range(5)]
    assert all(grant.granted for grant in held)

    manager.update_unit_state(downstream.resource_unit_id, runnable=True)
    upstream_waiter = _task(upstream.resource_unit_id, "refill")
    downstream_waiter = _task(downstream.resource_unit_id, "ready")
    manager.note_task_waiting(upstream_waiter)
    manager.note_task_waiting(downstream_waiter)

    before = manager.snapshot()["admission"]
    upstream_result = manager.try_acquire_queued_task(upstream_waiter)
    downstream_result = manager.try_acquire_queued_task(downstream_waiter)

    assert before["preferred_task"] == {
        "task_id": downstream_waiter.task_id,
        "attempt_id": downstream_waiter.attempt_id,
        "resource_unit_id": downstream.resource_unit_id,
        "grant_class": "normal",
    }
    assert before["reservation_unit_ids"]["heap_bytes"] == [
        upstream.resource_unit_id,
        downstream.resource_unit_id,
    ]
    assert [item["task_id"] for item in before["waiting_tasks"]] == [
        upstream_waiter.task_id,
        downstream_waiter.task_id,
    ]
    assert not upstream_result.granted
    assert not upstream_result.fatal
    assert upstream_result.blocked_reason == "admission_turn"
    assert downstream_result.granted
    assert downstream_result.lease.resource_unit_id == downstream.resource_unit_id


def test_driver_subset_yields_when_global_winner_is_fte_owned():
    upstream = _unit(
        "resource:f:driver-upstream",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
    )
    downstream = _unit(
        "resource:f:native-downstream",
        inputs=(upstream.resource_unit_id,),
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    manager = _manager(upstream, downstream, resources=_r(cpu=4, heap=8, store=8))
    _ready(manager, upstream.resource_unit_id, downstream.resource_unit_id)
    driver_request = _task(upstream.resource_unit_id, "driver")
    fte_request = _task(downstream.resource_unit_id, "fte")
    manager.note_task_waiting(driver_request)
    manager.note_task_waiting(fte_request)

    selected, yielded = manager.try_acquire_next_queued_task({(driver_request.task_id, driver_request.attempt_id)})

    assert selected == fte_request
    assert not yielded.granted
    assert yielded.blocked_reason == "admission_turn"
    assert manager.snapshot()["task_leases"] == {}
    assert manager.try_acquire_queued_task(fte_request).granted


def test_fte_request_yields_when_global_winner_is_driver_owned():
    upstream = _unit(
        "resource:f:native-upstream",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
    )
    downstream = _unit(
        "resource:f:driver-downstream",
        inputs=(upstream.resource_unit_id,),
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
    )
    manager = _manager(upstream, downstream, resources=_r(cpu=4, heap=8, store=8))
    _ready(manager, upstream.resource_unit_id, downstream.resource_unit_id)
    fte_request = _task(upstream.resource_unit_id, "fte")
    driver_request = _task(downstream.resource_unit_id, "driver")
    manager.note_task_waiting(fte_request)
    manager.note_task_waiting(driver_request)

    yielded = manager.try_acquire_queued_task(fte_request)
    selected, granted = manager.try_acquire_next_queued_task({(driver_request.task_id, driver_request.attempt_id)})

    assert not yielded.granted
    assert yielded.blocked_reason == "admission_turn"
    assert selected == driver_request
    assert granted.granted
    assert granted.lease.resource_unit_id == downstream.resource_unit_id


def test_descriptor_admission_is_not_persisted_and_retries_after_epoch_change():
    unit = _unit(
        "resource:f:descriptor-window",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        concurrency=1,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    manager = _manager(unit, resources=_r(cpu=2, heap=4, store=4))
    _ready(manager, unit.resource_unit_id)

    first = manager.try_acquire_task_descriptor(_task(unit.resource_unit_id, 0))
    assert first.granted
    epoch_before_denial = manager.admission_epoch()

    second_request = _task(unit.resource_unit_id, 1)
    denied = manager.try_acquire_task_descriptor(second_request)
    snapshot = manager.snapshot()

    assert not denied.granted
    assert denied.blocked_reason == "unit_concurrency"
    assert denied.admission_epoch == epoch_before_denial
    assert manager.admission_epoch() == epoch_before_denial
    assert snapshot["units"][unit.resource_unit_id]["pending_task_count"] == 0
    assert snapshot["admission"]["waiting_tasks"] == []

    manager.release_task_lease(
        first.lease.lease_id,
        attempt_id=first.lease.attempt_id,
    )
    assert manager.admission_epoch() > epoch_before_denial
    replacement = manager.try_acquire_task_descriptor(second_request)
    assert replacement.granted


def test_descriptor_admission_marks_non_root_native_fragment_unit_runnable_without_waiter():
    upstream = _unit(
        "resource:f:descriptor-source",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
    )
    native_fragment_unit = _unit(
        "resource:f:descriptor-non-root",
        inputs=(upstream.resource_unit_id,),
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    manager = _manager(upstream, native_fragment_unit, resources=_r(cpu=4, heap=8, store=8))
    manager.update_unit_state(upstream.resource_unit_id, runnable=True)
    assert manager.snapshot()["units"][native_fragment_unit.resource_unit_id]["runnable"] is False

    grant = manager.try_acquire_task_descriptor(_task(native_fragment_unit.resource_unit_id, 0))
    snapshot = manager.snapshot()

    assert grant.granted
    assert snapshot["units"][native_fragment_unit.resource_unit_id]["runnable"] is True
    assert snapshot["units"][native_fragment_unit.resource_unit_id]["pending_task_count"] == 0
    assert snapshot["admission"]["waiting_tasks"] == []


def test_descriptor_admission_yields_to_higher_rank_persistent_waiter():
    upstream = _unit(
        "resource:f:descriptor-upstream",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    downstream = _unit(
        "resource:f:persistent-downstream",
        inputs=(upstream.resource_unit_id,),
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
    )
    manager = _manager(upstream, downstream, resources=_r(cpu=4, heap=8, store=8))
    _ready(manager, upstream.resource_unit_id, downstream.resource_unit_id)
    downstream_waiter = _task(downstream.resource_unit_id, "driver")
    manager.note_task_waiting(downstream_waiter)

    descriptor = manager.try_acquire_task_descriptor(_task(upstream.resource_unit_id, "fte"))

    assert not descriptor.granted
    assert descriptor.blocked_reason == "admission_turn"
    snapshot = manager.snapshot()
    assert [item["task_id"] for item in snapshot["admission"]["waiting_tasks"]] == [downstream_waiter.task_id]
    assert manager.try_acquire_queued_task(downstream_waiter).granted


def test_reregistering_same_task_waiter_does_not_publish_resource_change():
    unit = _unit(
        "resource:f:idempotent-waiter",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
    )
    changes = []
    manager = _manager(
        unit,
        resources=_r(cpu=2, heap=4, store=4),
        on_change=lambda: changes.append("changed"),
    )
    manager.update_unit_state(unit.resource_unit_id, runnable=True)
    changes.clear()
    request = _task(unit.resource_unit_id, 0)

    manager.note_task_waiting(request)
    manager.note_task_waiting(request)

    assert changes == ["changed"]


def test_reapplying_identical_unit_state_does_not_publish_resource_change():
    unit = _unit(
        "resource:f:idempotent-unit-state",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
    )
    changes = []
    manager = _manager(
        unit,
        resources=_r(cpu=2, heap=4, store=4),
        on_change=lambda: changes.append("changed"),
    )

    manager.update_unit_state(unit.resource_unit_id, runnable=True)
    manager.update_unit_state(unit.resource_unit_id, runnable=True)

    assert changes == ["changed"]


def test_parent_native_fragment_owns_transferable_credit_for_one_nested_udf_task():
    parent = _unit(
        "resource:f:parent-native",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    bridge = _unit(
        "resource:f:bridge-native",
        inputs=(parent.resource_unit_id,),
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    child = _unit(
        "resource:f:child-udf",
        inputs=(bridge.resource_unit_id,),
        resources=_r(cpu=1, heap=8),
        target=0,
        blocks=0,
    )
    manager = _manager(
        parent,
        bridge,
        child,
        resources=_r(cpu=100, heap=30, store=12),
    )
    manager.update_unit_state(parent.resource_unit_id, runnable=True)

    parent_grants = [manager.try_acquire_task(_task(parent.resource_unit_id, partition)) for partition in range(4)]

    assert sum(grant.granted for grant in parent_grants) == 3
    assert parent_grants[3].blocked_reason == "continuation_capacity"
    snapshot = manager.snapshot()
    assert snapshot["usage"]["heap_bytes"] == 24
    assert len(snapshot["continuation_credits"]) == 3
    assert all(credit["borrowed_by_task_lease_id"] is None for credit in snapshot["continuation_credits"].values())
    admission = snapshot["admission"]
    assert admission["live_demand_unit_ids"]["heap_bytes"] == []
    assert admission["continuation_unit_ids"]["heap_bytes"] == []
    assert admission["reservation_unit_ids"]["heap_bytes"] == []

    child_requests = [_task(child.resource_unit_id, partition) for partition in range(3)]
    for request in child_requests:
        manager.note_task_waiting(request)
    child_grants = [manager.try_acquire_queued_task(request) for request in child_requests]

    assert all(grant.granted for grant in child_grants)
    borrowed = manager.snapshot()
    assert borrowed["usage"]["heap_bytes"] == 24
    assert {credit["borrowed_by_task_lease_id"] for credit in borrowed["continuation_credits"].values()} == {
        grant.lease.lease_id for grant in child_grants
    }

    fourth_child = _task(child.resource_unit_id, 3)
    manager.note_task_waiting(fourth_child)
    denied = manager.try_acquire_queued_task(fourth_child)
    assert not denied.granted
    assert denied.blocked_reason == "hard_heap_bytes"

    first_child = child_grants[0].lease
    assert manager.release_task_lease(
        first_child.lease_id,
        attempt_id=first_child.attempt_id,
    )
    replacement = manager.try_acquire_queued_task(fourth_child)
    assert replacement.granted
    assert manager.snapshot()["usage"]["heap_bytes"] == 24


def test_nested_udf_uses_normal_allocation_after_continuation_reserve_is_borrowed():
    parent = _unit(
        "resource:f:parent-native-normal-overflow",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    child = _unit(
        "resource:f:child-udf-normal-overflow",
        inputs=(parent.resource_unit_id,),
        resources=_r(cpu=1, heap=8),
        target=0,
        blocks=0,
    )
    manager = _manager(
        parent,
        child,
        resources=_r(cpu=8, heap=40, store=40),
    )
    manager.update_unit_state(parent.resource_unit_id, runnable=True)
    assert manager.try_acquire_task(_task(parent.resource_unit_id, 0)).granted

    first_request = _task(child.resource_unit_id, 0)
    second_request = _task(child.resource_unit_id, 1)
    manager.note_task_waiting(first_request)
    manager.note_task_waiting(second_request)

    first = manager.try_acquire_queued_task(first_request)
    second = manager.try_acquire_queued_task(second_request)

    assert first.granted
    assert second.granted
    snapshot = manager.snapshot()
    borrowed = [
        credit
        for credit in snapshot["continuation_credits"].values()
        if credit["borrowed_by_task_lease_id"] is not None
    ]
    assert len(borrowed) == 1
    assert borrowed[0]["borrowed_by_task_lease_id"] == first.lease.lease_id
    assert all(
        credit["borrowed_by_task_lease_id"] != second.lease.lease_id
        for credit in snapshot["continuation_credits"].values()
    )
    assert snapshot["usage"] == _r(cpu=2, heap=16, store=0).to_dict()


def test_borrowed_continuation_credit_outlives_parent_until_child_releases():
    parent = _unit(
        "resource:f:parent-native",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    child = _unit(
        "resource:f:child-udf",
        inputs=(parent.resource_unit_id,),
        resources=_r(cpu=1, heap=8),
        target=0,
        blocks=0,
    )
    manager = _manager(parent, child, resources=_r(cpu=10, heap=10, store=20))
    manager.update_unit_state(parent.resource_unit_id, runnable=True)

    parent_grant = manager.try_acquire_task(_task(parent.resource_unit_id, 0))
    child_request = _task(child.resource_unit_id, 0)
    manager.note_task_waiting(child_request)
    child_grant = manager.try_acquire_queued_task(child_request)

    assert parent_grant.granted and child_grant.granted
    assert manager.release_task_lease(
        parent_grant.lease.lease_id,
        attempt_id=parent_grant.lease.attempt_id,
    )
    snapshot = manager.snapshot()
    assert snapshot["usage"]["heap_bytes"] == 8
    assert len(snapshot["continuation_credits"]) == 1
    assert next(iter(snapshot["continuation_credits"].values()))["parent_active"] is False

    assert manager.release_task_lease(
        child_grant.lease.lease_id,
        attempt_id=child_grant.lease.attempt_id,
    )
    snapshot = manager.snapshot()
    assert snapshot["usage"]["heap_bytes"] == 0
    assert snapshot["continuation_credits"] == {}


def test_released_child_returns_credit_to_active_parent():
    parent = _unit(
        "resource:f:parent-native",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    child = _unit(
        "resource:f:child-udf",
        inputs=(parent.resource_unit_id,),
        resources=_r(cpu=1, heap=8),
        target=0,
        blocks=0,
    )
    manager = _manager(parent, child, resources=_r(cpu=10, heap=10, store=10))
    manager.update_unit_state(parent.resource_unit_id, runnable=True)

    parent_grant = manager.try_acquire_task(_task(parent.resource_unit_id, 0))
    first_request = _task(child.resource_unit_id, 0)
    manager.note_task_waiting(first_request)
    first_child = manager.try_acquire_queued_task(first_request)
    assert parent_grant.granted and first_child.granted

    assert manager.release_task_lease(
        first_child.lease.lease_id,
        attempt_id=first_child.lease.attempt_id,
    )
    snapshot = manager.snapshot()
    assert snapshot["usage"]["heap_bytes"] == 8
    assert next(iter(snapshot["continuation_credits"].values()))["borrowed_by_task_lease_id"] is None

    second_request = _task(child.resource_unit_id, 1)
    manager.note_task_waiting(second_request)
    second_child = manager.try_acquire_queued_task(second_request)
    assert second_child.granted
    assert manager.snapshot()["usage"]["heap_bytes"] == 8


def test_continuation_credit_can_reserve_a_different_node_from_parent_native_fragment():
    parent = _unit(
        "resource:f:parent-native",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    child = _unit(
        "resource:f:child-udf",
        inputs=(parent.resource_unit_id,),
        resources=_r(cpu=1, heap=8),
        target=0,
        blocks=0,
    )
    manager = _manager(
        parent,
        child,
        resources=_r(cpu=2, heap=10, store=20),
        nodes=(
            ("node-a", _r(cpu=1, heap=2, store=10)),
            ("node-b", _r(cpu=1, heap=8, store=10)),
        ),
    )
    manager.update_unit_state(parent.resource_unit_id, runnable=True)

    parent_grant = manager.try_acquire_task(_task(parent.resource_unit_id, 0, node_id="node-a"))
    assert parent_grant.granted
    credit = next(iter(manager.snapshot()["continuation_credits"].values()))
    assert credit["node_id"] == "node-b"

    child_request = _task(child.resource_unit_id, 0)
    manager.note_task_waiting(child_request)
    child_grant = manager.try_acquire_queued_task(child_request)
    assert child_grant.granted
    assert child_grant.lease.node_id == "node-b"


def test_native_continuation_reserves_one_real_heterogeneous_task_shape():
    parent = _unit(
        "resource:f:parent-native",
        resources=_r(),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    cpu_child = _unit(
        "resource:f:cpu-child",
        inputs=(parent.resource_unit_id,),
        resources=_r(cpu=4),
        target=0,
        blocks=0,
    )
    gpu_child = _unit(
        "resource:f:gpu-child",
        inputs=(parent.resource_unit_id,),
        resources=_r(cpu=1, gpu=1),
        target=0,
        blocks=0,
    )
    manager = _manager(
        parent,
        cpu_child,
        gpu_child,
        terminals=(cpu_child.resource_unit_id, gpu_child.resource_unit_id),
        resources=_r(cpu=5, gpu=1),
        nodes=(
            ("cpu-node", _r(cpu=4)),
            ("gpu-node", _r(cpu=1, gpu=1)),
        ),
    )
    manager.update_unit_state(parent.resource_unit_id, runnable=True)

    parent_grant = manager.try_acquire_task(_task(parent.resource_unit_id, 0, node_id="cpu-node"))

    assert parent_grant.granted
    credit = next(iter(manager.snapshot()["continuation_credits"].values()))
    assert credit["resources"] == _r(cpu=1, gpu=1).to_dict()
    assert credit["node_id"] == "gpu-node"


def test_child_outputs_keep_credit_borrowed_until_last_output_releases():
    parent = _unit(
        "resource:f:parent-native",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    child = _unit(
        "resource:f:child-udf",
        inputs=(parent.resource_unit_id,),
        resources=_r(cpu=1, heap=8),
        target=10,
        blocks=1,
    )
    manager = _manager(parent, child, resources=_r(cpu=10, heap=10, store=20))
    manager.update_unit_state(parent.resource_unit_id, runnable=True)

    parent_grant = manager.try_acquire_task(_task(parent.resource_unit_id, 0))
    first_request = _task(child.resource_unit_id, 0)
    manager.note_task_waiting(first_request)
    first_child = manager.try_acquire_queued_task(first_request)
    output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            child.resource_unit_id,
            first_child.lease.lease_id,
            first_child.lease.attempt_id,
            "child-output",
            10,
        )
    )
    assert parent_grant.granted and first_child.granted and output.granted

    assert manager.release_task_lease(
        first_child.lease.lease_id,
        attempt_id=first_child.lease.attempt_id,
    )
    snapshot = manager.snapshot()
    assert snapshot["usage"] == _r(cpu=1, heap=8, store=10).to_dict()
    credit = next(iter(snapshot["continuation_credits"].values()))
    assert credit["borrowed_by_task_lease_id"] == first_child.lease.lease_id

    second_request = _task(child.resource_unit_id, 1)
    manager.note_task_waiting(second_request)
    denied = manager.try_acquire_queued_task(second_request)
    assert not denied.granted
    assert denied.blocked_reason == "hard_heap_bytes"

    assert manager.release_output_block(output.lease.lease_id)
    second_child = manager.try_acquire_queued_task(second_request)
    assert second_child.granted


def test_handed_off_child_output_recycles_credit_without_unaccounting_physical_bytes():
    parent = _unit(
        "resource:f:parent-native",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    child = _unit(
        "resource:f:child-udf",
        inputs=(parent.resource_unit_id,),
        resources=_r(cpu=1, heap=8),
        target=10,
        blocks=1,
    )
    manager = _manager(parent, child, resources=_r(cpu=10, heap=10, store=30))
    manager.update_unit_state(parent.resource_unit_id, runnable=True)

    parent_grant = manager.try_acquire_task(_task(parent.resource_unit_id, 0))
    first_request = _task(child.resource_unit_id, 0)
    manager.note_task_waiting(first_request)
    first_child = manager.try_acquire_queued_task(first_request)
    first_output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            child.resource_unit_id,
            first_child.lease.lease_id,
            first_child.lease.attempt_id,
            "child-output-1",
            10,
        )
    )
    assert parent_grant.granted and first_child.granted and first_output.granted
    assert manager.transition_output_block(first_output.lease.lease_id, "unit_queue")
    assert manager.release_task_lease(
        first_child.lease.lease_id,
        attempt_id=first_child.lease.attempt_id,
    )

    second_request = _task(child.resource_unit_id, 1)
    manager.note_task_waiting(second_request)
    before_handoff = manager.try_acquire_queued_task(second_request)
    assert not before_handoff.granted
    assert before_handoff.blocked_reason == "hard_heap_bytes"

    assert manager.transition_output_block(first_output.lease.lease_id, "downstream_input")
    after_handoff = manager.try_acquire_queued_task(second_request)
    assert after_handoff.granted
    snapshot = manager.snapshot()
    assert first_output.lease.lease_id in snapshot["output_leases"]
    # The historical ObjectRef remains exact while the new running task adds
    # one learned generator-buffer estimate.
    assert snapshot["usage"] == _r(cpu=1, heap=8, store=20).to_dict()
    credit = next(iter(snapshot["continuation_credits"].values()))
    assert credit["borrowed_by_task_lease_id"] == after_handoff.lease.lease_id
    assert snapshot["output_leases"][first_output.lease.lease_id]["continuation_credit_id"] == credit["credit_id"]

    second_output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            child.resource_unit_id,
            after_handoff.lease.lease_id,
            after_handoff.lease.attempt_id,
            "child-output-2",
            10,
        )
    )
    assert second_output.granted
    with_both_outputs = manager.snapshot()
    # Recycling compute ownership never recycles bytes: both physical blocks
    # remain charged to the query and node flow-control budgets.
    assert with_both_outputs["usage"] == _r(cpu=1, heap=8, store=30).to_dict()
    assert (
        with_both_outputs["output_leases"][second_output.lease.lease_id]["continuation_credit_id"]
        == credit["credit_id"]
    )

    assert manager.release_output_block(first_output.lease.lease_id)
    released = manager.snapshot()
    assert released["usage"] == _r(cpu=1, heap=8, store=20).to_dict()
    credit = next(iter(released["continuation_credits"].values()))
    assert credit["borrowed_by_task_lease_id"] == after_handoff.lease.lease_id


def test_cross_unit_credit_reuse_checks_historical_outputs_before_grant():
    parent = _unit(
        "resource:f:parent-native",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    first_child = _unit(
        "resource:f:child-a",
        inputs=(parent.resource_unit_id,),
        resources=_r(cpu=1, heap=8),
        target=10,
        blocks=2,
    )
    second_child = _unit(
        "resource:f:child-b",
        inputs=(first_child.resource_unit_id,),
        resources=_r(cpu=1, heap=8, store=10),
        target=10,
        blocks=1,
    )
    manager = _manager(
        parent,
        first_child,
        second_child,
        resources=_r(cpu=10, heap=10, store=20),
    )
    manager.update_unit_state(parent.resource_unit_id, runnable=True)

    parent_grant = manager.try_acquire_task(_task(parent.resource_unit_id, 0))
    first_request = _task(first_child.resource_unit_id, 0)
    manager.note_task_waiting(first_request)
    first_grant = manager.try_acquire_queued_task(first_request)
    first_output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            first_child.resource_unit_id,
            first_grant.lease.lease_id,
            first_grant.lease.attempt_id,
            "child-a-output",
            15,
        )
    )
    assert parent_grant.granted and first_grant.granted and first_output.granted
    assert manager.transition_output_block(first_output.lease.lease_id, "unit_queue")
    assert manager.release_task_lease(
        first_grant.lease.lease_id,
        attempt_id=first_grant.lease.attempt_id,
    )
    assert manager.transition_output_block(first_output.lease.lease_id, "downstream_input")
    assert manager.snapshot()["usage"]["object_store_bytes"] == 15

    second_request = _task(second_child.resource_unit_id, 0, retained=10)
    manager.note_task_waiting(second_request)
    granted = manager.try_acquire_queued_task(second_request)
    assert granted.granted and granted.liveness
    admitted = manager.snapshot()
    assert admitted["usage"]["object_store_bytes"] == 25
    assert admitted["node_usage"]["node-a"]["object_store_bytes"] == 25
    assert admitted["soft_object_store_debt_bytes"] == 5


def test_continuation_borrow_reuses_idle_credit_with_smallest_live_output_delta():
    parent = _unit(
        "resource:f:parent-native",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    child = _unit(
        "resource:f:child-udf",
        inputs=(parent.resource_unit_id,),
        resources=_r(cpu=1, heap=8, store=10),
        target=10,
        blocks=1,
    )
    manager = _manager(
        parent,
        child,
        resources=_r(cpu=20, heap=20, store=45),
    )
    manager.update_unit_state(parent.resource_unit_id, runnable=True)
    parents = [manager.try_acquire_task(_task(parent.resource_unit_id, partition)) for partition in range(2)]
    assert all(grant.granted for grant in parents)

    first_request = _task(child.resource_unit_id, 0, retained=10)
    manager.note_task_waiting(first_request)
    first_child = manager.try_acquire_queued_task(first_request)
    assert first_child.granted
    first_output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            child.resource_unit_id,
            first_child.lease.lease_id,
            first_child.lease.attempt_id,
            "historical-output",
            15,
        )
    )
    assert first_output.granted
    assert manager.transition_output_block(first_output.lease.lease_id, "unit_queue")
    assert manager.transition_output_block(first_output.lease.lease_id, "downstream_input")
    assert manager.release_task_lease(
        first_child.lease.lease_id,
        attempt_id=first_child.lease.attempt_id,
    )
    historical_credit_id = manager.snapshot()["output_leases"][first_output.lease.lease_id]["continuation_credit_id"]

    manager.update_allocation(
        _allocation(_r(cpu=20, heap=20, store=40), generation=2),
        admission_open=True,
    )
    second_request = _task(child.resource_unit_id, 1, retained=10)
    manager.note_task_waiting(second_request)
    second_child = manager.try_acquire_queued_task(second_request)

    assert second_child.granted
    borrowed_credit_ids = {
        credit_id
        for credit_id, credit in manager.snapshot()["continuation_credits"].items()
        if credit["borrowed_by_task_lease_id"] == second_child.lease.lease_id
    }
    assert len(borrowed_credit_ids) == 1
    assert borrowed_credit_ids == {historical_credit_id}


def test_cross_unit_idle_credit_output_excess_is_attributed_exactly_once():
    parent = _unit(
        "resource:f:parent-native",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    first_child = _unit(
        "resource:f:child-a",
        inputs=(parent.resource_unit_id,),
        resources=_r(cpu=1, heap=8),
        target=10,
        blocks=2,
    )
    second_child = _unit(
        "resource:f:child-b",
        inputs=(first_child.resource_unit_id,),
        resources=_r(cpu=1, heap=8),
        target=10,
        blocks=2,
    )
    manager = _manager(
        parent,
        first_child,
        second_child,
        resources=_r(cpu=10, heap=10, store=35),
    )
    manager.update_unit_state(parent.resource_unit_id, runnable=True)
    manager.set_external_consumer_waiting(True)
    parent_grant = manager.try_acquire_task(_task(parent.resource_unit_id, 0))
    assert parent_grant.granted

    for index, unit in enumerate((first_child, second_child)):
        request = _task(unit.resource_unit_id, index)
        manager.note_task_waiting(request)
        grant = manager.try_acquire_queued_task(request)
        assert grant.granted
        output = manager.try_acquire_output_block(
            OutputBlockRequest(
                "q",
                unit.resource_unit_id,
                grant.lease.lease_id,
                grant.lease.attempt_id,
                f"cross-unit-output-{index}",
                15,
            )
        )
        assert output.granted
        assert output.liveness is (index == 1)
        assert manager.transition_output_block(output.lease.lease_id, "unit_queue")
        assert manager.transition_output_block(output.lease.lease_id, "downstream_input")
        assert manager.release_task_lease(
            grant.lease.lease_id,
            attempt_id=grant.lease.attempt_id,
        )

    snapshot = manager.snapshot()
    assert snapshot["usage"]["object_store_bytes"] == 30
    unit_store_usage = {
        resource_unit_id: unit_snapshot["usage"]["object_store_bytes"]
        for resource_unit_id, unit_snapshot in snapshot["units"].items()
    }
    assert sum(unit_store_usage.values()) == 30
    assert unit_store_usage[parent.resource_unit_id] == 0
    assert unit_store_usage[first_child.resource_unit_id] == 15
    assert unit_store_usage[second_child.resource_unit_id] == 15


def test_continuation_unit_usage_matches_query_usage_for_active_and_idle_borrower():
    parent = _unit(
        "resource:f:parent-native",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    child = _unit(
        "resource:f:child-udf",
        inputs=(parent.resource_unit_id,),
        resources=_r(cpu=1, heap=8, store=10),
        target=10,
        blocks=1,
    )
    manager = _manager(
        parent,
        child,
        resources=_r(cpu=10, heap=10, store=45),
    )
    manager.update_unit_state(parent.resource_unit_id, runnable=True)
    parent_grant = manager.try_acquire_task(_task(parent.resource_unit_id, 0))
    request = _task(child.resource_unit_id, 0, retained=10)
    manager.note_task_waiting(request)
    child_grant = manager.try_acquire_queued_task(request)
    output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            child.resource_unit_id,
            child_grant.lease.lease_id,
            child_grant.lease.attempt_id,
            "active-credit-output",
            25,
        )
    )
    assert parent_grant.granted and child_grant.granted and output.granted

    active = manager.snapshot()
    assert active["usage"]["object_store_bytes"] == 45
    assert sum(unit["usage"]["object_store_bytes"] for unit in active["units"].values()) == 45

    assert manager.transition_output_block(output.lease.lease_id, "unit_queue")
    assert manager.transition_output_block(output.lease.lease_id, "downstream_input")
    assert manager.release_task_lease(
        child_grant.lease.lease_id,
        attempt_id=child_grant.lease.attempt_id,
    )
    idle = manager.snapshot()
    assert idle["usage"]["object_store_bytes"] == 25
    assert sum(unit["usage"]["object_store_bytes"] for unit in idle["units"].values()) == 25


def test_live_output_from_deleted_continuation_credit_remains_in_unit_usage():
    parent = _unit(
        "resource:f:parent-native",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    child = _unit(
        "resource:f:child-udf",
        inputs=(parent.resource_unit_id,),
        resources=_r(cpu=1, heap=8),
        target=10,
        blocks=2,
    )
    manager = _manager(
        parent,
        child,
        resources=_r(cpu=10, heap=10, store=35),
    )
    manager.update_unit_state(parent.resource_unit_id, runnable=True)
    parent_grant = manager.try_acquire_task(_task(parent.resource_unit_id, 0))
    request = _task(child.resource_unit_id, 0)
    manager.note_task_waiting(request)
    child_grant = manager.try_acquire_queued_task(request)
    output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            child.resource_unit_id,
            child_grant.lease.lease_id,
            child_grant.lease.attempt_id,
            "orphaned-credit-output",
            15,
        )
    )
    assert parent_grant.granted and child_grant.granted and output.granted
    assert manager.transition_output_block(output.lease.lease_id, "unit_queue")
    assert manager.transition_output_block(output.lease.lease_id, "downstream_input")
    assert manager.release_task_lease(
        child_grant.lease.lease_id,
        attempt_id=child_grant.lease.attempt_id,
    )
    assert manager.release_task_lease(
        parent_grant.lease.lease_id,
        attempt_id=parent_grant.lease.attempt_id,
    )

    snapshot = manager.snapshot()
    assert snapshot["continuation_credits"] == {}
    assert snapshot["usage"]["object_store_bytes"] == 15
    assert sum(unit["usage"]["object_store_bytes"] for unit in snapshot["units"].values()) == 15
    assert snapshot["units"][child.resource_unit_id]["usage"]["object_store_bytes"] == 15


def test_query_cancellation_clears_idle_and_borrowed_continuation_credits():
    parent = _unit(
        "resource:f:parent-native",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    child = _unit(
        "resource:f:child-udf",
        inputs=(parent.resource_unit_id,),
        resources=_r(cpu=1, heap=8),
        target=0,
        blocks=0,
    )
    manager = _manager(parent, child, resources=_r(cpu=20, heap=20, store=20))
    manager.update_unit_state(parent.resource_unit_id, runnable=True)
    parents = [manager.try_acquire_task(_task(parent.resource_unit_id, partition)) for partition in range(2)]
    child_request = _task(child.resource_unit_id, 0)
    manager.note_task_waiting(child_request)
    child_grant = manager.try_acquire_queued_task(child_request)
    assert all(grant.granted for grant in parents)
    assert child_grant.granted

    released = manager.cancel("cancel credits")
    snapshot = manager.snapshot()

    assert released == {"task_lease_count": 3, "output_lease_count": 0}
    assert snapshot["task_leases"] == {}
    assert snapshot["continuation_credits"] == {}
    assert snapshot["usage"] == _r().to_dict()


def test_parent_native_fragment_admission_preserves_largest_nested_udf_commitment():
    parent = _unit(
        "resource:f:safe-parent",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    first_child = _unit(
        "resource:f:large-child-a",
        inputs=(parent.resource_unit_id,),
        resources=_r(cpu=1, heap=8),
        target=0,
        blocks=0,
    )
    second_child = _unit(
        "resource:f:large-child-b",
        inputs=(first_child.resource_unit_id,),
        resources=_r(cpu=1, heap=8),
        target=0,
        blocks=0,
    )
    manager = _manager(
        parent,
        first_child,
        second_child,
        resources=_r(cpu=10, heap=10, store=10),
    )
    manager.update_unit_state(parent.resource_unit_id, runnable=True)

    first_parent = manager.try_acquire_task(_task(parent.resource_unit_id, 0))
    second_parent = manager.try_acquire_task(_task(parent.resource_unit_id, 1))

    assert first_parent.granted
    assert not second_parent.granted
    assert second_parent.blocked_reason == "continuation_capacity"

    child_request = _task(first_child.resource_unit_id, 0)
    manager.note_task_waiting(child_request)
    child_grant = manager.try_acquire_queued_task(child_request)

    assert child_grant.granted


def test_native_fragment_placement_does_not_reserve_pinned_actor_process_capacity():
    parent = _unit(
        "resource:f:placed-parent",
        resources=_r(cpu=1, heap=4),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    child = _unit(
        "resource:f:pinned-child",
        inputs=(parent.resource_unit_id,),
        resources=_r(gpu=1, heap=8),
        target=0,
        blocks=0,
        backend="ray_actor",
        actor_pool_size=1,
    )
    manager = _manager(
        parent,
        child,
        resources=_r(cpu=4, gpu=1, heap=20, store=20),
        nodes=(
            ("node-a", _r(cpu=2, gpu=1, heap=10, store=10)),
            ("node-b", _r(cpu=2, heap=10, store=10)),
        ),
    )
    manager.update_unit_state(parent.resource_unit_id, runnable=True)

    pinned_node_request = manager.try_acquire_task(_task(parent.resource_unit_id, "pinned-node", node_id="node-a"))
    automatic_request = manager.try_acquire_task(_task(parent.resource_unit_id, "auto-node"))

    assert pinned_node_request.granted
    assert pinned_node_request.lease.node_id == "node-a"
    assert automatic_request.granted
    assert automatic_request.lease.node_id == "node-b"


def test_ray_task_placement_keeps_actor_invocation_bytes_out_of_hard_reservations():
    producer = _unit(
        "resource:f:placed-ray-task-producer",
        resources=_r(cpu=1, heap=1),
        target=1,
        blocks=1,
    )
    consumer = _unit(
        "resource:f:placed-native-consumer",
        inputs=(producer.resource_unit_id,),
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    actor = _unit(
        "resource:f:placed-actor-consumer",
        inputs=(consumer.resource_unit_id,),
        resources=_r(),
        resident=_r(gpu=1, heap=10),
        target=2,
        blocks=1,
        backend="ray_actor",
        actor_pool_size=1,
    )
    manager = _manager(
        producer,
        consumer,
        actor,
        resources=_r(cpu=3, gpu=1, heap=15, store=4),
        nodes=(
            ("node-a", _r(cpu=1, gpu=1, heap=12, store=2)),
            ("node-b", _r(cpu=2, heap=3, store=2)),
        ),
    )
    manager.update_unit_state(producer.resource_unit_id, runnable=True)

    pinned = manager.try_acquire_task(_task(producer.resource_unit_id, "pinned", node_id="node-a"))
    automatic = manager.try_acquire_task(_task(producer.resource_unit_id, "automatic"))

    assert pinned.granted
    assert automatic.granted
    assert automatic.lease.node_id == "node-b"


def test_cold_topology_caps_upstream_fanout_until_direct_downstream_starts():
    upstream = _unit(
        "resource:f:upstream",
        resources=_r(cpu=1, heap=20),
        target=0,
        blocks=0,
    )
    downstream = _unit(
        "resource:f:downstream",
        inputs=(upstream.resource_unit_id,),
        resources=_r(cpu=1, heap=40),
        target=0,
        blocks=0,
    )
    terminal = _unit(
        "resource:f:terminal",
        inputs=(downstream.resource_unit_id,),
        resources=_r(cpu=1, heap=60),
        target=0,
        blocks=0,
    )
    manager = _manager(
        upstream,
        downstream,
        terminal,
        resources=_r(cpu=10, heap=100, store=100),
    )
    _ready(manager, upstream.resource_unit_id, downstream.resource_unit_id, terminal.resource_unit_id)

    first_upstream = manager.try_acquire_task(_task(upstream.resource_unit_id, 0))
    normal_reason = manager._normal_task_block_reason_locked(_task(upstream.resource_unit_id, 1))[0]
    second_upstream = manager.try_acquire_task(_task(upstream.resource_unit_id, 1))
    first_downstream = manager.try_acquire_task(_task(downstream.resource_unit_id, 0))

    assert first_upstream.granted
    assert normal_reason == "unit_soft_limit"
    assert not second_upstream.granted
    assert first_downstream.granted


def test_native_continuation_does_not_add_process_reservation_to_ray_task_fanout():
    producer = _unit(
        "resource:f:image-cpu-producer",
        resources=_r(cpu=1, heap=2),
        target=1,
        blocks=1,
    )
    consumer = _unit(
        "resource:f:image-gpu-feeder",
        inputs=(producer.resource_unit_id,),
        resources=_r(cpu=1, heap=2),
        target=1,
        blocks=1,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    gpu_actor = _unit(
        "resource:f:image-gpu-actor",
        inputs=(consumer.resource_unit_id,),
        resources=_r(),
        resident=_r(gpu=1, heap=20),
        target=1,
        blocks=1,
        backend="ray_actor",
        actor_pool_size=1,
    )
    manager = _manager(
        producer,
        consumer,
        gpu_actor,
        resources=_r(cpu=36, gpu=1, heap=41, store=10),
    )
    # The downstream units are deliberately latent. Actor process resources
    # are scheduled by Ray Core, while the native continuation is owned by
    # DuckDB and adds no Vane process reservation.
    manager.update_unit_state(producer.resource_unit_id, runnable=True)

    producer_grants = [manager.try_acquire_task(_task(producer.resource_unit_id, partition)) for partition in range(20)]
    blocked_producer = manager.try_acquire_task(_task(producer.resource_unit_id, 20))

    assert all(grant.granted for grant in producer_grants)
    assert not blocked_producer.granted
    assert blocked_producer.blocked_reason == "hard_heap_bytes"
    assert producer.resource_unit_id not in manager.snapshot()["admission"]["downstream_reservations"]

    manager.update_unit_state(consumer.resource_unit_id, runnable=True)
    consumer_grant = manager.try_acquire_task(_task(consumer.resource_unit_id, 0))
    _ready(manager, gpu_actor.resource_unit_id)
    actor_grant = manager.try_acquire_task(_task(gpu_actor.resource_unit_id, 0, retained=0))

    assert consumer_grant.granted
    assert actor_grant.granted and not actor_grant.liveness
    assert actor_grant.lease.actor_index == 0
    assert (
        manager.snapshot()["usage"]
        == _r(
            cpu=20,
            heap=40,
            store=1,
        ).to_dict()
    )
    assert (
        manager.snapshot()["actor_process_usage"]
        == _r(
            gpu=1,
            heap=20,
        ).to_dict()
    )


def test_upstream_fanout_preserves_remote_task_continuation_without_native_slot():
    upstream = _unit(
        "resource:f:streaming-upstream",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
    )
    actor = _unit(
        "resource:f:streaming-actor",
        inputs=(upstream.resource_unit_id,),
        resources=_r(),
        resident=_r(gpu=1, heap=4),
        target=0,
        blocks=0,
        backend="ray_actor",
        actor_pool_size=1,
        actor_prefetch_depth=3,
    )
    downstream = _unit(
        "resource:f:streaming-downstream",
        inputs=(actor.resource_unit_id,),
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
    )
    terminal_native = _unit(
        "resource:f:streaming-terminal-native",
        inputs=(downstream.resource_unit_id,),
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    manager = _manager(
        upstream,
        actor,
        downstream,
        terminal_native,
        resources=_r(cpu=20, gpu=1, heap=24, store=20),
    )
    manager.update_unit_state(upstream.resource_unit_id, runnable=True)

    upstream_grants = [manager.try_acquire_task(_task(upstream.resource_unit_id, partition)) for partition in range(11)]
    assert all(grant.granted for grant in upstream_grants)
    blocked_upstream = manager.try_acquire_task(_task(upstream.resource_unit_id, 11))
    assert not blocked_upstream.granted
    assert blocked_upstream.blocked_reason == "liveness_not_needed"
    snapshot = manager.snapshot()
    assert snapshot["usage"]["heap_bytes"] == 22
    reservations = snapshot["admission"]["downstream_reservations"][upstream.resource_unit_id]
    assert [reservation["reservation_id"] for reservation in reservations] == [
        f"actor_continuation:{upstream.resource_unit_id}:{downstream.resource_unit_id}",
    ]

    _ready(manager, actor.resource_unit_id)
    actor_grant = manager.try_acquire_task(_task(actor.resource_unit_id, 0, retained=0))
    assert actor_grant.granted

    downstream_request = _task(downstream.resource_unit_id, 0)
    manager.note_task_waiting(downstream_request)
    before = manager.snapshot()
    downstream_grant = manager.try_acquire_queued_task(downstream_request)

    assert before["admission"]["preferred_task"] == {
        "task_id": downstream_request.task_id,
        "attempt_id": downstream_request.attempt_id,
        "resource_unit_id": downstream.resource_unit_id,
        "grant_class": "liveness",
    }
    assert downstream_grant.granted
    assert downstream_grant.liveness
    assert manager.snapshot()["usage"]["heap_bytes"] == 24

    terminal_request = _task(terminal_native.resource_unit_id, 0)
    manager.note_task_waiting(terminal_request)
    terminal_grant = manager.try_acquire_queued_task(terminal_request)

    assert terminal_grant.granted
    assert not terminal_grant.liveness
    assert manager.snapshot()["usage"]["heap_bytes"] == 24


def test_actor_process_bypass_leaves_downstream_ray_task_reservation_soft():
    upstream = _unit(
        "resource:f:minimum-upstream",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
    )
    actor = _unit(
        "resource:f:minimum-actor",
        inputs=(upstream.resource_unit_id,),
        resources=_r(),
        resident=_r(cpu=1, heap=4),
        target=0,
        blocks=0,
        backend="ray_actor",
        actor_pool_size=1,
    )
    downstream = _unit(
        "resource:f:minimum-downstream",
        inputs=(actor.resource_unit_id,),
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
    )
    manager = _manager(
        upstream,
        actor,
        downstream,
        resources=_r(cpu=2, heap=6),
    )
    manager.update_unit_state(upstream.resource_unit_id, runnable=True)

    grant = manager.try_acquire_task(_task(upstream.resource_unit_id, 0))

    assert grant.granted and not grant.liveness
    snapshot = manager.snapshot()
    assert snapshot["usage"]["heap_bytes"] == 2
    reservations = snapshot["admission"]["downstream_reservations"][upstream.resource_unit_id]
    assert [reservation["reservation_id"] for reservation in reservations] == [
        f"actor_continuation:{upstream.resource_unit_id}:{downstream.resource_unit_id}",
    ]


def test_remote_process_usage_excludes_native_parent_and_downstream_fragments():
    parent = _unit(
        "resource:f:image-parent-native",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    producer = _unit(
        "resource:f:image-nested-cpu",
        inputs=(parent.resource_unit_id,),
        resources=_r(cpu=1, heap=4),
        target=1,
        blocks=1,
    )
    consumer = _unit(
        "resource:f:image-downstream-native",
        inputs=(producer.resource_unit_id,),
        resources=_r(cpu=1, heap=3),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    actor = _unit(
        "resource:f:image-downstream-actor",
        inputs=(consumer.resource_unit_id,),
        resources=_r(),
        resident=_r(gpu=1, heap=10),
        target=1,
        blocks=1,
        backend="ray_actor",
        actor_pool_size=1,
    )
    # Only the Ray task is QRM hard-accounted. Actor process resources are
    # reported separately and both native fragments use DuckDB's own budget.
    manager = _manager(
        parent,
        producer,
        consumer,
        actor,
        resources=_r(cpu=3, gpu=1, heap=20, store=2),
    )
    manager.update_unit_state(parent.resource_unit_id, runnable=True)

    parent_grant = manager.try_acquire_task(_task(parent.resource_unit_id, 0))
    manager.update_unit_state(producer.resource_unit_id, runnable=True)
    producer_request = _task(producer.resource_unit_id, 0)
    manager.note_task_waiting(producer_request)
    producer_grant = manager.try_acquire_queued_task(producer_request)
    manager.update_unit_state(consumer.resource_unit_id, runnable=True)
    consumer_grant = manager.try_acquire_task(_task(consumer.resource_unit_id, 0))
    _ready(manager, actor.resource_unit_id)
    actor_grant = manager.try_acquire_task(_task(actor.resource_unit_id, 0, retained=0))

    assert parent_grant.granted
    assert producer_grant.granted
    assert consumer_grant.granted
    assert actor_grant.granted
    assert (
        manager.snapshot()["usage"]
        == _r(
            cpu=1,
            heap=4,
            store=0,
        ).to_dict()
    )
    assert (
        manager.snapshot()["actor_process_usage"]
        == _r(
            gpu=1,
            heap=10,
        ).to_dict()
    )


def test_native_fanout_is_capped_only_by_remote_continuation_credits():
    parent = _unit(
        "resource:f:image-parent-native",
        resources=_r(cpu=1, heap=1),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    direct_fte = _unit(
        "resource:f:image-direct-native",
        inputs=(parent.resource_unit_id,),
        resources=_r(cpu=1, heap=1),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    cpu_udf = _unit(
        "resource:f:image-cpu-udf",
        inputs=(direct_fte.resource_unit_id,),
        resources=_r(cpu=1, heap=4),
        target=0,
        blocks=0,
    )
    downstream_fte = _unit(
        "resource:f:image-downstream-native",
        inputs=(cpu_udf.resource_unit_id,),
        resources=_r(cpu=1, heap=4),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    gpu_actor = _unit(
        "resource:f:image-gpu-actor",
        inputs=(downstream_fte.resource_unit_id,),
        resources=_r(),
        resident=_r(gpu=1, heap=8),
        target=0,
        blocks=0,
        backend="ray_actor",
        actor_pool_size=1,
    )
    manager = _manager(
        parent,
        direct_fte,
        cpu_udf,
        downstream_fte,
        gpu_actor,
        resources=_r(cpu=36, gpu=1, heap=49),
    )
    manager.update_unit_state(parent.resource_unit_id, runnable=True)

    parent_grants = [manager.try_acquire_task(_task(parent.resource_unit_id, partition)) for partition in range(13)]

    assert all(grant.granted for grant in parent_grants[:12])
    assert not parent_grants[12].granted
    assert parent_grants[12].blocked_reason == "continuation_capacity"
    snapshot = manager.snapshot()
    assert snapshot["usage"]["heap_bytes"] == 48
    assert parent.resource_unit_id not in snapshot["admission"]["downstream_reservations"]

    manager.update_unit_state(cpu_udf.resource_unit_id, runnable=True)
    cpu_request = _task(cpu_udf.resource_unit_id, 0)
    manager.note_task_waiting(cpu_request)
    cpu_grant = manager.try_acquire_queued_task(cpu_request)
    manager.update_unit_state(
        downstream_fte.resource_unit_id,
        runnable=True,
    )
    downstream_grant = manager.try_acquire_task(_task(downstream_fte.resource_unit_id, 0))

    assert cpu_grant.granted
    assert downstream_grant.granted
    final_snapshot = manager.snapshot()
    assert cpu_grant.lease.lease_id in {
        credit["borrowed_by_task_lease_id"] for credit in final_snapshot["continuation_credits"].values()
    }
    assert final_snapshot["usage"]["heap_bytes"] == 48


def test_hard_heap_limit_is_never_bypassed_by_object_store_liveness():
    unit = _unit("resource:f:decode", resources=_r(cpu=1, heap=100), target=25, blocks=2)
    manager = _manager(unit, resources=_r(cpu=10, heap=250, store=100))
    _ready(manager, unit.resource_unit_id, consumer_waiting=True)

    first = manager.try_acquire_task(_task(unit.resource_unit_id, 0))
    second = manager.try_acquire_task(_task(unit.resource_unit_id, 1))
    denied = manager.try_acquire_task(_task(unit.resource_unit_id, 2))

    assert first.granted and second.granted
    assert not denied.granted
    assert denied.blocked_reason == "hard_heap_bytes"
    assert denied.liveness is False
    snapshot = manager.snapshot()
    assert snapshot["usage"]["heap_bytes"] == 200
    assert snapshot["usage"]["object_store_bytes"] == 0
    assert snapshot["liveness"]["active_task_lease_id"] is None


def test_soft_budget_does_not_start_extra_same_unit_work_while_task_is_active():
    unit = _unit("resource:f:decode", resources=_r(cpu=1, heap=100), target=30, blocks=2)
    manager = _manager(unit, resources=_r(cpu=10, heap=1_000, store=100))
    _ready(manager, unit.resource_unit_id, consumer_waiting=True)

    first = manager.try_acquire_task(_task(unit.resource_unit_id, 0))
    output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            unit.resource_unit_id,
            first.lease.lease_id,
            first.lease.attempt_id,
            "learned-output",
            30,
        )
    )
    second = manager.try_acquire_task(_task(unit.resource_unit_id, 1))

    assert first.granted and output.granted
    assert not second.granted
    assert second.blocked_reason == "liveness_not_needed"
    assert manager.snapshot()["usage"]["object_store_bytes"] == 90


def test_soft_budget_does_not_start_extra_borrower_while_same_unit_task_is_active():
    parent = _unit(
        "resource:f:parent",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    child = _unit(
        "resource:f:child",
        inputs=(parent.resource_unit_id,),
        resources=_r(cpu=1, heap=8, store=81),
    )
    manager = _manager(
        parent,
        child,
        resources=_r(cpu=4, heap=20, store=150),
    )
    _ready(manager, parent.resource_unit_id, child.resource_unit_id)

    first_parent = manager.try_acquire_task(_task(parent.resource_unit_id, 0))
    first = manager.try_acquire_task(_task(child.resource_unit_id, 0, retained=81))
    second_parent = manager.try_acquire_task(_task(parent.resource_unit_id, 1))
    second = manager.try_acquire_task(_task(child.resource_unit_id, 1, retained=81))

    assert first_parent.granted
    assert first.granted
    assert not first.liveness
    assert second_parent.granted
    assert not second.granted
    assert second.blocked_reason == "liveness_not_needed"
    assert manager.snapshot()["usage"]["object_store_bytes"] == 81


def test_retained_input_uses_exact_dynamic_credit_above_nominal_target():
    unit = _unit("resource:f:decode", resources=_r(cpu=1, heap=100, store=30))
    manager = _manager(unit, resources=_r(cpu=10, heap=1_000, store=1_000))
    _ready(manager, unit.resource_unit_id)

    request = _task(unit.resource_unit_id, 0, retained=31)
    manager.note_task_waiting(request)
    assert manager.snapshot()["units"][unit.resource_unit_id]["queued_input_bytes"] == 31
    granted = manager.try_acquire_queued_task(request)

    assert granted.granted
    assert granted.lease.resources.object_store_bytes == 31
    assert manager.snapshot()["usage"]["object_store_bytes"] == 31


def test_retained_input_larger_than_soft_budget_uses_one_liveness_task():
    unit = _unit("resource:f:decode", resources=_r(cpu=1, heap=100, store=101))
    manager = _manager(unit, resources=_r(cpu=10, heap=1_000, store=100))
    _ready(manager, unit.resource_unit_id)

    request = _task(unit.resource_unit_id, 0, retained=101)
    manager.note_task_waiting(request)
    granted = manager.try_acquire_queued_task(request)

    assert granted.granted
    assert granted.liveness
    assert manager.snapshot()["soft_object_store_debt_bytes"] == 1


@pytest.mark.parametrize("acquire_mode", ["queued", "next_queued"])
def test_liveness_parent_lends_escape_to_one_continuation_path_until_handoff(acquire_mode):
    parent = _unit(
        "resource:f:parent",
        resources=_r(cpu=1, heap=2),
        target=60,
        blocks=2,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    child = _unit(
        "resource:f:child",
        inputs=(parent.resource_unit_id,),
        resources=_r(cpu=1, heap=8),
        target=10,
        blocks=2,
    )
    manager = _manager(
        parent,
        child,
        resources=_r(cpu=4, heap=20, store=100),
    )
    manager.update_unit_state(parent.resource_unit_id, runnable=True)
    manager.set_external_consumer_waiting(True)

    parent_grant = manager.try_acquire_task(_task(parent.resource_unit_id, 0))
    child_request = _task(child.resource_unit_id, 0)
    manager.note_task_waiting(child_request)
    if acquire_mode == "queued":
        child_grant = manager.try_acquire_queued_task(child_request)
    else:
        selected, child_grant = manager.try_acquire_next_queued_task(
            {(child_request.task_id, child_request.attempt_id)}
        )
        assert selected == child_request
        assert child_grant is not None

    assert parent_grant.granted and parent_grant.liveness
    # Before the first output, the Ray UDF has no pending-generator estimate,
    # so it can continue the liveness path through normal admission.
    assert child_grant.granted and not child_grant.liveness
    assert manager.snapshot()["liveness"]["active_task_lease_id"] == parent_grant.lease.lease_id

    output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            child.resource_unit_id,
            child_grant.lease.lease_id,
            child_grant.lease.attempt_id,
            "continuation-output",
            10,
        )
    )
    assert output.granted and output.liveness
    assert manager.transition_output_block(output.lease.lease_id, "unit_queue")
    assert manager.release_task_lease(
        parent_grant.lease.lease_id,
        attempt_id=parent_grant.lease.attempt_id,
    )
    assert manager.release_task_lease(
        child_grant.lease.lease_id,
        attempt_id=child_grant.lease.attempt_id,
    )

    producer_owned = manager.snapshot()
    assert producer_owned["liveness"]["active_task_lease_id"] is None
    assert producer_owned["liveness"]["active_output_lease_id"] == output.lease.lease_id

    assert manager.transition_output_block(output.lease.lease_id, "downstream_input")
    handed_off = manager.snapshot()
    assert handed_off["liveness"]["active_task_lease_id"] is None
    assert handed_off["continuation_credits"] == {}
    assert handed_off["usage"]["object_store_bytes"] == 10


def test_liveness_parent_allows_one_direct_actor_continuation_without_sibling_fanout():
    parent = _unit(
        "resource:f:parent",
        resources=_r(cpu=1, heap=2),
        target=60,
        blocks=2,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    first_actor = _unit(
        "resource:f:actor-a",
        inputs=(parent.resource_unit_id,),
        resources=_r(store=20),
        target=10,
        blocks=2,
        backend="ray_actor",
        actor_pool_size=1,
        resident=_r(cpu=1, heap=8),
    )
    sibling_actor = _unit(
        "resource:f:actor-b",
        inputs=(parent.resource_unit_id,),
        resources=_r(store=20),
        target=10,
        blocks=2,
        backend="ray_actor",
        actor_pool_size=1,
        resident=_r(cpu=1, heap=8),
    )
    sibling_task = _unit(
        "resource:f:zz-task-b",
        inputs=(sibling_actor.resource_unit_id,),
        resources=_r(cpu=1, heap=8, store=20),
        target=10,
        blocks=2,
    )
    path_target = _unit(
        "resource:f:aa-path",
        inputs=(first_actor.resource_unit_id,),
        resources=_r(cpu=1, heap=10, store=20),
        target=10,
        blocks=2,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    manager = _manager(
        parent,
        first_actor,
        sibling_actor,
        sibling_task,
        path_target,
        resources=_r(cpu=8, heap=60, store=100),
        terminals=(
            path_target.resource_unit_id,
            sibling_task.resource_unit_id,
        ),
    )
    manager.update_unit_state(parent.resource_unit_id, runnable=True)
    parent_grant = manager.try_acquire_task(_task(parent.resource_unit_id, 0))
    _ready(manager, first_actor.resource_unit_id, sibling_actor.resource_unit_id)

    first_request = _task(first_actor.resource_unit_id, 0)
    manager.note_task_waiting(first_request)
    selected, first_grant = manager.try_acquire_next_queued_task({(first_request.task_id, first_request.attempt_id)})

    assert selected == first_request
    assert first_grant is not None
    assert parent_grant.granted and parent_grant.liveness
    assert first_grant.granted and first_grant.liveness
    sibling_request = _task(sibling_actor.resource_unit_id, 0)
    manager.note_task_waiting(sibling_request)
    sibling_grant = manager.try_acquire_queued_task(sibling_request)
    assert not sibling_grant.granted
    assert sibling_grant.blocked_reason == "liveness_task_active"
    assert manager.remove_task_waiter(
        sibling_request.task_id,
        sibling_request.attempt_id,
    )
    sibling_task_request = _task(sibling_task.resource_unit_id, 0)
    manager.note_task_waiting(sibling_task_request)
    sibling_task_grant = manager.try_acquire_queued_task(sibling_task_request)
    assert not sibling_task_grant.granted
    assert sibling_task_grant.blocked_reason == "liveness_task_active"
    path_request = _task(path_target.resource_unit_id, 0)
    manager.note_task_waiting(path_request)
    selected, path_grant = manager.try_acquire_next_queued_task(
        {
            (sibling_task_request.task_id, sibling_task_request.attempt_id),
            (path_request.task_id, path_request.attempt_id),
        }
    )
    assert selected == path_request
    assert path_grant is not None
    assert path_grant.granted and path_grant.liveness
    assert manager.release_task_lease(
        path_grant.lease.lease_id,
        attempt_id=path_grant.lease.attempt_id,
    )
    assert manager.remove_task_waiter(
        sibling_task_request.task_id,
        sibling_task_request.attempt_id,
    )

    output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            first_actor.resource_unit_id,
            first_grant.lease.lease_id,
            first_grant.lease.attempt_id,
            "actor-continuation-output",
            10,
        )
    )
    assert output.granted and output.liveness
    assert manager.transition_output_block(output.lease.lease_id, "unit_queue")
    assert manager.release_task_lease(
        parent_grant.lease.lease_id,
        attempt_id=parent_grant.lease.attempt_id,
    )
    assert manager.release_task_lease(
        first_grant.lease.lease_id,
        attempt_id=first_grant.lease.attempt_id,
    )
    assert manager.snapshot()["liveness"]["active_task_lease_id"] == parent_grant.lease.lease_id

    assert manager.transition_output_block(output.lease.lease_id, "downstream_input")
    assert manager.snapshot()["liveness"]["active_task_lease_id"] is None


def test_liveness_actor_path_can_start_soft_blocked_downstream_fte():
    parent = _unit(
        "resource:f:parent",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    actor = _unit(
        "resource:f:actor",
        inputs=(parent.resource_unit_id,),
        resources=_r(store=81),
        target=10,
        blocks=2,
        backend="ray_actor",
        actor_pool_size=1,
        resident=_r(cpu=1, heap=8),
    )
    downstream = _unit(
        "resource:f:downstream",
        inputs=(actor.resource_unit_id,),
        resources=_r(cpu=1, heap=10, store=10),
        target=20,
        blocks=2,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    manager = _manager(
        parent,
        actor,
        downstream,
        resources=_r(cpu=4, heap=30, store=100),
    )
    manager.update_unit_state(parent.resource_unit_id, runnable=True)
    parent_grant = manager.try_acquire_task(_task(parent.resource_unit_id, 0))
    _ready(manager, actor.resource_unit_id)
    actor_request = _task(actor.resource_unit_id, 0)
    manager.note_task_waiting(actor_request)
    selected, actor_grant = manager.try_acquire_next_queued_task({(actor_request.task_id, actor_request.attempt_id)})

    assert selected == actor_request
    assert actor_grant is not None
    assert parent_grant.granted and not parent_grant.liveness
    assert actor_grant.granted and not actor_grant.liveness
    downstream_grant = manager.try_acquire_task_descriptor(_task(downstream.resource_unit_id, 0))
    assert downstream_grant.granted and downstream_grant.liveness
    assert manager.snapshot()["liveness"]["active_task_lease_id"] == downstream_grant.lease.lease_id

    assert manager.release_task_lease(
        downstream_grant.lease.lease_id,
        attempt_id=downstream_grant.lease.attempt_id,
    )
    assert manager.release_task_lease(
        actor_grant.lease.lease_id,
        attempt_id=actor_grant.lease.attempt_id,
    )
    assert manager.snapshot()["liveness"]["active_task_lease_id"] is None


def test_liveness_task_credit_returns_after_outputs_leave_the_producer_queue():
    unit = _unit(
        "resource:f:decode",
        resources=_r(store=101),
        backend="ray_worker",
    )
    manager = _manager(unit, resources=_r(cpu=10, heap=1_000, store=100))
    _ready(manager, unit.resource_unit_id)

    first_request = _task(unit.resource_unit_id, 0, retained=101)
    manager.note_task_waiting(first_request)
    first = manager.try_acquire_queued_task(first_request)
    outputs = manager.finish_task_with_outputs(
        first.lease.lease_id,
        attempt_id=first.lease.attempt_id,
        outputs=(
            OutputBlockRequest(
                query_id="q",
                producer_unit_id=unit.resource_unit_id,
                task_lease_id=first.lease.lease_id,
                attempt_id=first.lease.attempt_id,
                block_id="first-output",
                size_bytes=10,
            ),
        ),
    )

    producer_owned = manager.snapshot()
    assert producer_owned["liveness"]["active_task_lease_id"] == first.lease.lease_id
    assert producer_owned["usage"]["object_store_bytes"] == 10
    second_request = _task(unit.resource_unit_id, 1, retained=101)
    manager.note_task_waiting(second_request)
    denied = manager.try_acquire_queued_task(second_request)
    assert not denied.granted
    assert denied.blocked_reason == "liveness_task_active"

    assert manager.transition_output_block(outputs[0].lease_id, "downstream_input")
    handed_off = manager.snapshot()
    assert handed_off["liveness"]["active_task_lease_id"] is None
    assert handed_off["usage"]["object_store_bytes"] == 10
    second = manager.try_acquire_queued_task(second_request)
    assert second.granted and second.liveness


def test_liveness_is_not_granted_while_another_runnable_unit_has_normal_capacity():
    upstream = _unit("resource:f:upstream", resources=_r(cpu=1, heap=70), target=0, blocks=0)
    blocked = _unit(
        "resource:f:blocked",
        inputs=(upstream.resource_unit_id,),
        resources=_r(cpu=1, heap=30),
        target=0,
        blocks=0,
    )
    runnable = _unit(
        "resource:f:runnable",
        inputs=(blocked.resource_unit_id,),
        resources=_r(cpu=1, heap=10),
        target=0,
        blocks=0,
    )
    manager = _manager(
        upstream,
        blocked,
        runnable,
        resources=_r(cpu=100, heap=100, store=100),
    )
    _ready(
        manager, upstream.resource_unit_id, blocked.resource_unit_id, runnable.resource_unit_id, consumer_waiting=True
    )
    assert manager.try_acquire_task(_task(upstream.resource_unit_id, 0)).granted
    manager.note_task_waiting(_task(runnable.resource_unit_id, 0))

    denied = manager.try_acquire_task(_task(blocked.resource_unit_id, 0))

    assert not denied.granted
    assert denied.blocked_reason == "normal_candidate_available"


def test_task_lease_release_is_attempt_aware_and_idempotent():
    unit = _unit("resource:f:decode")
    manager = _manager(unit)
    _ready(manager, unit.resource_unit_id)
    grant = manager.try_acquire_task(_task(unit.resource_unit_id, 0, attempt="a"))

    assert manager.release_task_lease(grant.lease.lease_id, attempt_id="wrong") is False
    assert manager.release_task_lease(grant.lease.lease_id, attempt_id="a") is True
    assert manager.release_task_lease(grant.lease.lease_id, attempt_id="a") is False
    replay = manager.try_acquire_task(_task(unit.resource_unit_id, 0, attempt="a"))
    retry = manager.try_acquire_task(_task(unit.resource_unit_id, 0, attempt="b"))
    assert replay.blocked_reason == "attempt_terminal"
    assert retry.granted


def test_abandoned_pre_submit_task_lease_can_reacquire_the_same_attempt():
    unit = _unit("resource:f:decode")
    manager = _manager(unit)
    _ready(manager, unit.resource_unit_id)
    first = manager.try_acquire_task(_task(unit.resource_unit_id, 0, attempt="a"))

    assert manager.abandon_task_lease(first.lease.lease_id, attempt_id="a") is True
    assert manager.abandon_task_lease(first.lease.lease_id, attempt_id="a") is False

    replacement = manager.try_acquire_task(_task(unit.resource_unit_id, 0, attempt="a"))
    assert replacement.granted
    assert replacement.lease.lease_id != first.lease.lease_id


def test_output_blocks_add_exact_refs_to_dynamic_pending_generator_estimate():
    unit = _unit("resource:f:decode", resources=_r(cpu=1, heap=100, store=10), target=50, blocks=2)
    manager = _manager(unit, resources=_r(cpu=10, heap=1_000, store=1_000))
    _ready(manager, unit.resource_unit_id)
    task = manager.try_acquire_task(_task(unit.resource_unit_id, 0))
    assert manager.snapshot()["usage"]["object_store_bytes"] == 10

    first = manager.try_acquire_output_block(
        OutputBlockRequest("q", unit.resource_unit_id, task.lease.lease_id, "0", "block-1", 40)
    )
    second = manager.try_acquire_output_block(
        OutputBlockRequest("q", unit.resource_unit_id, task.lease.lease_id, "0", "block-2", 50)
    )

    assert first.granted and second.granted
    assert manager.snapshot()["usage"]["object_store_bytes"] == 190

    third = manager.try_acquire_output_block(
        OutputBlockRequest("q", unit.resource_unit_id, task.lease.lease_id, "0", "block-3", 25)
    )
    assert third.granted
    snapshot = manager.snapshot()
    assert snapshot["usage"]["object_store_bytes"] == 202
    unit_snapshot = snapshot["units"][unit.resource_unit_id]
    assert unit_snapshot["bytes_task_outputs_generated"] == 115
    assert unit_snapshot["num_task_outputs_generated"] == 3
    assert unit_snapshot["average_output_block_bytes"] == pytest.approx(115 / 3)
    assert unit_snapshot["pending_output_estimate_per_active_task_bytes"] == 77


def test_finished_task_output_count_caps_dynamic_generator_estimate():
    unit = _unit(
        "resource:f:learned-output-count",
        resources=_r(cpu=1, heap=10),
        target=50,
        blocks=2,
    )
    manager = _manager(unit, resources=_r(cpu=4, heap=40, store=1_000))
    _ready(manager, unit.resource_unit_id)
    first = manager.try_acquire_task(_task(unit.resource_unit_id, 0))
    output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            unit.resource_unit_id,
            first.lease.lease_id,
            first.lease.attempt_id,
            "one-block-task",
            10,
        )
    )
    assert output.granted
    assert manager.release_output_block(output.lease.lease_id)
    assert manager.release_task_lease(
        first.lease.lease_id,
        attempt_id=first.lease.attempt_id,
    )

    learned = manager.snapshot()["units"][unit.resource_unit_id]
    assert learned["average_output_block_bytes"] == 10
    assert learned["average_output_blocks_per_finished_task"] == 1
    assert learned["pending_output_estimate_per_active_task_bytes"] == 10

    second = manager.try_acquire_task(_task(unit.resource_unit_id, 1))
    assert second.granted
    assert second.lease.output_window_bytes == 100
    assert manager.snapshot()["usage"]["object_store_bytes"] == 10


def test_prefetched_actor_invocation_has_no_pending_generator_estimate():
    actor = _unit(
        "resource:f:actor-dynamic-output",
        resources=_r(),
        resident=_r(cpu=1, heap=10),
        target=50,
        blocks=2,
        backend="ray_actor",
        actor_pool_size=1,
        actor_prefetch_depth=2,
    )
    manager = _manager(
        actor,
        resources=_r(cpu=1, heap=10, store=1_000),
    )
    _ready(manager, actor.resource_unit_id)
    active = manager.try_acquire_task(_task(actor.resource_unit_id, 0, retained=0))
    prefetched = manager.try_acquire_task(_task(actor.resource_unit_id, 1, retained=0))
    output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            actor.resource_unit_id,
            active.lease.lease_id,
            active.lease.attempt_id,
            "actor-first-output",
            10,
        )
    )

    assert active.granted and prefetched.granted and output.granted
    # 10 exact output bytes + 20 estimated bytes for only the active call.
    assert manager.snapshot()["usage"]["object_store_bytes"] == 30
    assert manager.release_task_lease(
        active.lease.lease_id,
        attempt_id=active.lease.attempt_id,
    )
    # The promoted call now uses the learned 10-byte/one-output estimate, while
    # the first call's ObjectRef remains exact.
    assert manager.snapshot()["usage"]["object_store_bytes"] == 20


def test_frontier_materializer_disables_object_store_backpressure_for_input():
    upstream = _unit(
        "resource:f:materializer-input",
        resources=_r(cpu=1),
        target=50,
        blocks=2,
    )
    materializer = _unit(
        "resource:f:blocking-materializer",
        inputs=(upstream.resource_unit_id,),
        resources=_r(),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    downstream = _unit(
        "resource:f:after-materializer",
        inputs=(materializer.resource_unit_id,),
        resources=_r(cpu=1),
        target=0,
        blocks=0,
    )
    manager = _manager(
        upstream,
        materializer,
        downstream,
        resources=_r(cpu=2, store=50),
        barriers=(_barrier("blocking-materializer", materializer),),
    )
    _ready(
        manager,
        upstream.resource_unit_id,
        materializer.resource_unit_id,
        downstream.resource_unit_id,
    )
    task = manager.try_acquire_task(_task(upstream.resource_unit_id, 0))
    output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            upstream.resource_unit_id,
            task.lease.lease_id,
            task.lease.attempt_id,
            "materializer-input-block",
            50,
        )
    )

    assert task.granted
    assert output.granted and not output.liveness
    snapshot = manager.snapshot()
    assert snapshot["soft_object_store_debt_bytes"] == 100
    assert snapshot["units"][upstream.resource_unit_id]["object_store_backpressure_disabled"] is True


def test_asymmetric_frontier_lifts_object_store_cap_only_for_materialized_input():
    build = _unit(
        "resource:f:broadcast-build",
        resources=_r(cpu=1),
        target=50,
        blocks=2,
    )
    probe = _unit(
        "resource:f:broadcast-probe",
        resources=_r(cpu=1),
        target=50,
        blocks=2,
    )
    materializer = _unit(
        "resource:f:broadcast-join",
        inputs=(build.resource_unit_id, probe.resource_unit_id),
        resources=_r(),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    downstream = _unit(
        "resource:f:after-broadcast",
        inputs=(materializer.resource_unit_id,),
        resources=_r(cpu=1),
        target=0,
        blocks=0,
    )
    barrier = _barrier(
        "broadcast-join",
        materializer,
        materialized_inputs=(build.resource_unit_id,),
    )
    manager = _manager(
        build,
        probe,
        materializer,
        downstream,
        resources=_r(cpu=3, store=100),
        barriers=(barrier,),
    )
    _ready(
        manager,
        build.resource_unit_id,
        probe.resource_unit_id,
        materializer.resource_unit_id,
        downstream.resource_unit_id,
    )

    before = manager.snapshot()
    assert before["execution_phase"]["object_store_unlimited_unit_ids"] == [
        build.resource_unit_id,
        materializer.resource_unit_id,
    ]
    assert before["units"][build.resource_unit_id]["object_store_backpressure_disabled"] is True
    assert before["units"][probe.resource_unit_id]["object_store_backpressure_disabled"] is False

    assert manager.mark_materialization_barrier_completed_for_node("broadcast-join")
    after = manager.snapshot()
    assert after["execution_phase"]["object_store_unlimited_unit_ids"] == []
    assert after["execution_phase"]["eligible_resource_unit_ids"] == [
        probe.resource_unit_id,
        materializer.resource_unit_id,
        downstream.resource_unit_id,
    ]


def test_output_lease_transitions_preserve_bytes_and_release_after_task_completion():
    unit = _unit("resource:f:decode", resources=_r(cpu=1, heap=100), target=50, blocks=2)
    manager = _manager(unit)
    _ready(manager, unit.resource_unit_id)
    task = manager.try_acquire_task(_task(unit.resource_unit_id, 0))
    block = manager.try_acquire_output_block(
        OutputBlockRequest("q", unit.resource_unit_id, task.lease.lease_id, "0", "block-1", 80)
    )
    before = manager.snapshot()["usage"]["object_store_bytes"]

    for state in ("unit_queue", "downstream_input", "external_consumer"):
        assert manager.transition_output_block(block.lease.lease_id, state) is True
        assert manager.snapshot()["usage"]["object_store_bytes"] == before

    assert manager.release_task_lease(task.lease.lease_id, attempt_id="0") is True
    assert manager.snapshot()["usage"]["object_store_bytes"] == 80
    assert manager.release_output_block(block.lease.lease_id) is True
    assert manager.release_output_block(block.lease.lease_id) is False
    assert manager.snapshot()["usage"]["object_store_bytes"] == 0


def test_released_output_block_identity_cannot_be_leased_again():
    unit = _unit("resource:f:decode", resources=_r(cpu=1, heap=100), target=50, blocks=2)
    manager = _manager(unit)
    _ready(manager, unit.resource_unit_id)
    task = manager.try_acquire_task(_task(unit.resource_unit_id, 0))
    request = OutputBlockRequest(
        "q",
        unit.resource_unit_id,
        task.lease.lease_id,
        "0",
        "block-terminal",
        80,
    )
    first = manager.try_acquire_output_block(request)

    assert first.granted
    assert manager.release_output_block(first.lease.lease_id) is True
    replay = manager.try_acquire_output_block(request)
    assert replay.granted is False
    assert replay.fatal is True
    assert replay.blocked_reason == "output_block_terminal"
    assert manager.snapshot()["output_leases"] == {}


def test_fte_task_completion_atomically_transfers_window_to_output_leases():
    unit = _unit(
        "resource:f:native",
        resources=_r(cpu=1, heap=100, store=5),
        target=10,
        blocks=2,
        backend="ray_worker",
    )
    manager = _manager(unit)
    _ready(manager, unit.resource_unit_id)
    task = manager.try_acquire_task(_task(unit.resource_unit_id, 0))

    leases = manager.finish_task_with_outputs(
        task.lease.lease_id,
        attempt_id="0",
        outputs=(
            OutputBlockRequest("q", unit.resource_unit_id, task.lease.lease_id, "0", "fte-block-0", 8),
            OutputBlockRequest("q", unit.resource_unit_id, task.lease.lease_id, "0", "fte-block-1", 9),
        ),
    )

    snapshot = manager.snapshot()
    assert [lease.state for lease in leases] == ["unit_queue", "unit_queue"]
    assert snapshot["task_leases"] == {}
    assert set(snapshot["output_leases"]) == {lease.lease_id for lease in leases}
    assert snapshot["usage"] == _r(store=17).to_dict()


def test_atomic_fte_completion_rejects_a_ray_udf_task_lease():
    unit = _unit("resource:f:udf", target=10, blocks=2, backend="ray_task")
    manager = _manager(unit)
    _ready(manager, unit.resource_unit_id)
    task = manager.try_acquire_task(_task(unit.resource_unit_id, 0))

    with pytest.raises(RuntimeError, match="requires a native fragment lease"):
        manager.finish_task_with_outputs(
            task.lease.lease_id,
            attempt_id=task.lease.attempt_id,
            outputs=(),
        )

    assert task.lease.lease_id in manager.snapshot()["task_leases"]


def test_fte_task_completion_replaces_pending_estimate_with_oversized_exact_outputs():
    unit = _unit(
        "resource:f:native",
        resources=_r(cpu=1, heap=100),
        target=10,
        blocks=2,
        backend="ray_worker",
    )
    manager = _manager(unit, resources=_r(cpu=100, heap=1_000, store=20))
    _ready(manager, unit.resource_unit_id)
    task = manager.try_acquire_task(_task(unit.resource_unit_id, 0))

    assert manager.snapshot()["usage"]["object_store_bytes"] == 20
    leases = manager.finish_task_with_outputs(
        task.lease.lease_id,
        attempt_id="0",
        outputs=(
            OutputBlockRequest("q", unit.resource_unit_id, task.lease.lease_id, "0", "fte-block-0", 10),
            OutputBlockRequest("q", unit.resource_unit_id, task.lease.lease_id, "0", "fte-block-1", 11),
        ),
    )

    snapshot = manager.snapshot()
    assert snapshot["task_leases"] == {}
    assert set(snapshot["output_leases"]) == {lease.lease_id for lease in leases}
    assert snapshot["usage"]["object_store_bytes"] == 21
    assert snapshot["soft_object_store_debt_bytes"] == 1


def test_output_transition_rejects_skips_and_attempt_mismatch():
    unit = _unit("resource:f:decode")
    manager = _manager(unit)
    _ready(manager, unit.resource_unit_id)
    task = manager.try_acquire_task(_task(unit.resource_unit_id, 0))
    mismatch = manager.try_acquire_output_block(
        OutputBlockRequest("q", unit.resource_unit_id, task.lease.lease_id, "wrong", "block-wrong", 5)
    )
    block = manager.try_acquire_output_block(
        OutputBlockRequest("q", unit.resource_unit_id, task.lease.lease_id, "0", "block-1", 5)
    )

    assert mismatch.blocked_reason == "task_attempt_mismatch"
    with pytest.raises(ValueError, match="invalid output lease transition"):
        manager.transition_output_block(block.lease.lease_id, "external_consumer")


def test_oversized_output_block_uses_one_bounded_liveness_grant():
    unit = _unit("resource:f:decode", target=50, blocks=2)
    manager = _manager(unit, resources=_r(cpu=10, heap=1_000, store=100))
    _ready(manager, unit.resource_unit_id, consumer_waiting=True)
    task = manager.try_acquire_task(_task(unit.resource_unit_id, 0))

    granted = manager.try_acquire_output_block(
        OutputBlockRequest("q", unit.resource_unit_id, task.lease.lease_id, "0", "huge", 101)
    )

    assert granted.granted
    assert granted.liveness
    second_request = OutputBlockRequest(
        "q",
        unit.resource_unit_id,
        task.lease.lease_id,
        "0",
        "second",
        1,
    )
    second = manager.try_acquire_output_block(second_request)
    assert not second.granted
    assert second.blocked_reason == "liveness_output_active"
    assert manager.transition_output_block(granted.lease.lease_id, "unit_queue")
    assert manager.transition_output_block(granted.lease.lease_id, "downstream_input")
    next_block = manager.try_acquire_output_block(second_request)
    assert next_block.granted
    assert next_block.liveness
    assert manager.snapshot()["soft_object_store_debt_bytes"] == 102


def test_object_store_debt_does_not_block_debt_neutral_downstream_compute():
    producer = _unit("resource:f:producer", resources=_r(cpu=1, heap=100), target=50, blocks=2)
    consumer = _unit(
        "resource:f:consumer",
        inputs=(producer.resource_unit_id,),
        resources=_r(cpu=1, heap=100),
        target=0,
        blocks=0,
    )
    manager = _manager(
        producer,
        consumer,
        resources=_r(cpu=10, heap=1_000, store=100),
    )
    _ready(manager, producer.resource_unit_id, consumer.resource_unit_id, consumer_waiting=True)
    producer_task = manager.try_acquire_task(_task(producer.resource_unit_id, 0))
    oversized = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            producer.resource_unit_id,
            producer_task.lease.lease_id,
            "0",
            "oversized",
            101,
        )
    )

    assert oversized.granted and oversized.liveness
    assert manager.snapshot()["soft_object_store_debt_bytes"] == 101
    downstream = manager.try_acquire_task(_task(consumer.resource_unit_id, 0, retained=0))
    assert downstream.granted
    assert not downstream.liveness


def test_allocation_shrink_creates_debt_without_revoking_existing_leases():
    unit = _unit("resource:f:decode", resources=_r(cpu=1, heap=100), target=0, blocks=0)
    manager = _manager(unit, resources=_r(cpu=4, heap=400, store=100))
    _ready(manager, unit.resource_unit_id)
    leases = [manager.try_acquire_task(_task(unit.resource_unit_id, idx)) for idx in range(3)]
    assert all(grant.granted for grant in leases)

    manager.update_allocation(
        _allocation(_r(cpu=2, heap=200, store=100), generation=2),
        admission_open=True,
    )
    denied = manager.try_acquire_task(_task(unit.resource_unit_id, 4))
    snapshot = manager.snapshot()

    assert denied.blocked_reason == "allocation_debt"
    assert snapshot["allocation_debt"] == _r(cpu=1, heap=100).to_dict()
    assert len(snapshot["task_leases"]) == 3


def test_pending_allocation_preserves_live_node_debt_and_stops_new_admission():
    unit = _unit("resource:f:pending", resources=_r(cpu=1, heap=100))
    manager = _manager(
        unit,
        resources=_r(cpu=2, heap=200, store=20),
    )
    _ready(manager, unit.resource_unit_id)
    active = manager.try_acquire_task(_task(unit.resource_unit_id, 1))
    assert active.granted

    manager.update_allocation(
        _allocation(_r(), generation=2, nodes=()),
        admission_open=False,
    )

    blocked = manager.try_acquire_task(_task(unit.resource_unit_id, 2))
    snapshot = manager.snapshot()
    expected_usage = _r(cpu=1, heap=100).to_dict()
    assert blocked.granted is False
    assert blocked.blocked_reason == "allocation_pending"
    assert blocked.fatal is False
    assert snapshot["allocation_admission_open"] is False
    assert snapshot["allocation_debt"] == _r(cpu=1, heap=100).to_dict()
    assert snapshot["soft_object_store_debt_bytes"] == 0
    assert snapshot["node_usage"]["node-a"] == expected_usage
    assert snapshot["node_allocation_debt"]["node-a"] == _r(cpu=1, heap=100).to_dict()
    assert snapshot["node_soft_object_store_debt_bytes"]["node-a"] == 0

    assert manager.release_task_lease(
        active.lease.lease_id,
        attempt_id=active.lease.attempt_id,
    )
    still_blocked = manager.try_acquire_task(_task(unit.resource_unit_id, 3))
    assert still_blocked.blocked_reason == "allocation_pending"

    restored = _r(cpu=2, heap=200, store=20)
    manager.update_allocation(
        _allocation(restored, generation=3),
        admission_open=True,
    )
    assert manager.try_acquire_task(_task(unit.resource_unit_id, 4)).granted


def test_cancellation_releases_every_task_and_output_lease_idempotently():
    unit = _unit("resource:f:decode")
    manager = _manager(unit)
    _ready(manager, unit.resource_unit_id)
    task = manager.try_acquire_task(_task(unit.resource_unit_id, 0))
    block = manager.try_acquire_output_block(
        OutputBlockRequest("q", unit.resource_unit_id, task.lease.lease_id, "0", "block", 5)
    )

    first = manager.cancel("user_cancelled")
    second = manager.cancel("again")

    assert first == {"task_lease_count": 1, "output_lease_count": 1}
    assert second == {"task_lease_count": 0, "output_lease_count": 0}
    snapshot = manager.snapshot()
    assert snapshot["cancelled"] is True
    assert snapshot["usage"] == _r().to_dict()
    assert snapshot["task_leases"] == {}
    assert snapshot["active_actor_slots"] == {}
    assert snapshot["output_leases"] == {}
    assert manager.release_task_lease(task.lease.lease_id, attempt_id="0") is False
    assert manager.release_output_block(block.lease.lease_id) is False


def test_task_and_materialized_output_ownership_stay_on_one_ray_node():
    unit = _unit(
        "resource:f:per-node",
        resources=_r(cpu=1, heap=10),
        target=10,
        blocks=2,
        concurrency=3,
        backend="ray_worker",
    )
    node_capacity = _r(cpu=1, heap=10, store=20)
    manager = _manager(
        unit,
        resources=_r(cpu=2, heap=20, store=40),
        nodes=(("node-a", node_capacity), ("node-b", node_capacity)),
    )
    _ready(manager, unit.resource_unit_id)

    first = manager.try_acquire_task(_task(unit.resource_unit_id, 1, node_id="node-a"))
    second = manager.try_acquire_task(_task(unit.resource_unit_id, 2, node_id="node-b"))
    assert first.granted and second.granted
    assert {first.lease.node_id, second.lease.node_id} == {"node-a", "node-b"}

    manager.release_task_lease(second.lease.lease_id, attempt_id=second.lease.attempt_id)
    output_leases = manager.finish_task_with_outputs(
        first.lease.lease_id,
        attempt_id=first.lease.attempt_id,
        outputs=(
            OutputBlockRequest(
                query_id="q",
                producer_unit_id=unit.resource_unit_id,
                task_lease_id=first.lease.lease_id,
                attempt_id=first.lease.attempt_id,
                block_id="block:node-owned",
                size_bytes=10,
            ),
        ),
    )
    output = output_leases[0]
    assert output.node_id == first.lease.node_id
    assert manager.snapshot()["node_usage"][output.node_id]["object_store_bytes"] == 10

    granted = manager.try_acquire_task(_task(unit.resource_unit_id, 3, node_id=output.node_id))
    assert granted.granted
    assert granted.lease.node_id == output.node_id
    assert manager.snapshot()["node_soft_object_store_debt_bytes"][output.node_id] == 10
