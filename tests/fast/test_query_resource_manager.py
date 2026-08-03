# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

from duckdb.runners.ray.query_resource_graph import (
    ActorPlacement,
    NodeResourceAllocation,
    QueryAllocation,
    QueryResourceGraph,
    ResourceUnitSpec,
    ResourceVector,
)
from duckdb.runners.ray.query_resource_manager import (
    OutputBlockRequest,
    QueryResourceManager,
    TaskRequest,
)


def _r(*, cpu=0.0, gpu=0.0, heap=0, store=0):
    return ResourceVector(cpu=cpu, gpu=gpu, heap_bytes=heap, object_store_bytes=store)


def _allocation(resources, *, generation=1, placements=(), nodes=None):
    node_resources = (("node-a", resources),) if nodes is None else tuple(nodes)
    return QueryAllocation(
        resources=resources,
        node_allocations=tuple(
            NodeResourceAllocation(node_id=node_id, resources=node_capacity)
            for node_id, node_capacity in node_resources
        ),
        actor_placements=tuple(placements),
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
    is_barrier=False,
):
    requested = resources or _r(cpu=1, heap=10)
    if backend == "ray_actor":
        resident_resources = resident or _r(
            cpu=requested.cpu,
            gpu=requested.gpu,
            heap=requested.heap_bytes,
        )
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
        is_barrier=is_barrier,
    )


def _manager(
    *units,
    resources=None,
    reservation_ratio=0.5,
    terminals=None,
    nodes=None,
    on_change=None,
):
    graph = QueryResourceGraph(
        query_id="q",
        plan_digest="sha256:test",
        units=tuple(units),
        terminal_unit_ids=tuple(terminals or (units[-1].resource_unit_id,)),
    )
    allocation_resources = resources or _r(cpu=100, gpu=1, heap=1_000, store=1_000)
    actor_placements = tuple(
        ActorPlacement(resource_unit_id=unit.resource_unit_id, actor_index=actor_index, node_id="node-a")
        for unit in units
        if unit.backend == "ray_actor"
        for actor_index in range(unit.actor_pool_size)
    )
    allocation = _allocation(
        allocation_resources,
        placements=actor_placements,
        nodes=nodes,
    )
    graph.validate_allocation(allocation)
    return QueryResourceManager(
        graph,
        allocation,
        reservation_ratio=reservation_ratio,
        on_change=on_change,
    )


def _ready(manager, *unit_ids, consumer_waiting=False):
    for resource_unit_id in unit_ids:
        manager.update_unit_state(
            resource_unit_id,
            runnable=True,
            actor_ready=True,
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

    not_runnable = manager.try_acquire_task(_task(actor.resource_unit_id, 0))
    manager.update_unit_state(actor.resource_unit_id, runnable=True, actor_ready=False)
    not_ready = manager.try_acquire_task(_task(actor.resource_unit_id, 0))
    manager.update_unit_state(actor.resource_unit_id, runnable=True, actor_ready=True)
    granted = manager.try_acquire_task(_task(actor.resource_unit_id, 0))

    assert not_runnable.blocked_reason == "unit_not_runnable"
    assert not_ready.blocked_reason == "actor_not_ready"
    assert granted.granted
    assert granted.lease.resources == actor.per_task
    assert granted.lease.output_window_bytes == 20


def test_only_first_pending_reservation_scope_and_its_upstream_receive_reservations_without_gating():
    source = _unit(
        "resource:f:scope-source",
        resources=_r(cpu=1, heap=10),
        target=1,
        blocks=1,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    first_barrier = _unit(
        "resource:f:scope-first-barrier",
        inputs=(source.resource_unit_id,),
        resources=_r(cpu=1, heap=10),
        target=1,
        blocks=1,
        backend="ray_worker",
        unit_kind="native_fragment",
        is_barrier=True,
    )
    between = _unit(
        "resource:f:scope-between",
        inputs=(first_barrier.resource_unit_id,),
        resources=_r(cpu=1, heap=10),
        target=1,
        blocks=1,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    second_barrier = _unit(
        "resource:f:scope-second-barrier",
        inputs=(between.resource_unit_id,),
        resources=_r(cpu=1, heap=10),
        target=1,
        blocks=1,
        backend="ray_worker",
        unit_kind="native_fragment",
        is_barrier=True,
    )
    terminal = _unit(
        "resource:f:scope-terminal",
        inputs=(second_barrier.resource_unit_id,),
        resources=_r(cpu=1, heap=10),
        target=1,
        blocks=1,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    manager = _manager(
        source,
        first_barrier,
        between,
        second_barrier,
        terminal,
        resources=_r(cpu=5, heap=100, store=100),
    )
    _ready(
        manager,
        source.resource_unit_id,
        first_barrier.resource_unit_id,
        between.resource_unit_id,
        second_barrier.resource_unit_id,
        terminal.resource_unit_id,
    )
    for unit in (source, first_barrier, between, second_barrier, terminal):
        manager.note_task_waiting(_task(unit.resource_unit_id, "waiting"))

    first_snapshot = manager.snapshot()
    unreserved_between = manager.try_acquire_task(_task(between.resource_unit_id, 0))

    assert first_snapshot["reservation_scope"] == {
        "barrier_unit_id": first_barrier.resource_unit_id,
        "resource_unit_ids": [source.resource_unit_id, first_barrier.resource_unit_id],
        "object_store_backpressure_disabled": True,
        "object_store_unlimited_unit_ids": [
            source.resource_unit_id,
            first_barrier.resource_unit_id,
        ],
    }
    assert first_snapshot["admission"]["reservation_unit_ids"]["cpu"] == [
        source.resource_unit_id,
        first_barrier.resource_unit_id,
    ]
    assert unreserved_between.granted
    assert unreserved_between.liveness is False
    assert between.resource_unit_id not in first_snapshot["admission"]["reservation_unit_ids"]["cpu"]
    assert manager.release_task_lease(unreserved_between.lease.lease_id, attempt_id="0")

    with pytest.raises(ValueError, match="not a blocking materializer"):
        manager.mark_barrier_unit_completed(source.resource_unit_id)
    assert manager.mark_barrier_unit_completed(first_barrier.resource_unit_id)
    assert not manager.mark_barrier_unit_completed(first_barrier.resource_unit_id)
    second_snapshot = manager.snapshot()

    assert second_snapshot["reservation_scope"]["barrier_unit_id"] == second_barrier.resource_unit_id
    assert second_snapshot["reservation_scope"]["resource_unit_ids"] == [
        source.resource_unit_id,
        first_barrier.resource_unit_id,
        between.resource_unit_id,
        second_barrier.resource_unit_id,
    ]
    assert second_snapshot["units"][source.resource_unit_id]["completed"] is False
    assert second_snapshot["units"][first_barrier.resource_unit_id]["completed"] is False
    assert second_snapshot["units"][first_barrier.resource_unit_id]["materialization_completed"] is True
    assert second_snapshot["units"][source.resource_unit_id]["reservation_scope_eligible"] is True
    assert second_snapshot["units"][first_barrier.resource_unit_id]["reservation_scope_eligible"] is True
    assert second_snapshot["units"][between.resource_unit_id]["reservation_scope_eligible"] is True
    assert second_snapshot["units"][terminal.resource_unit_id]["reservation_scope_eligible"] is False
    assert manager.try_acquire_task(_task(between.resource_unit_id, 1)).granted
    assert manager.try_acquire_task(_task(first_barrier.resource_unit_id, 1)).granted


def test_blocking_materialization_disables_only_corresponding_object_store_soft_limits():
    early = _unit(
        "resource:f:spill-early",
        resources=_r(cpu=1, heap=10),
        target=10,
        blocks=1,
    )
    producer = _unit(
        "resource:f:spill-producer",
        inputs=(early.resource_unit_id,),
        resources=_r(cpu=1, heap=10),
        target=10,
        blocks=1,
    )
    barrier = _unit(
        "resource:f:spill-barrier",
        inputs=(producer.resource_unit_id,),
        resources=_r(cpu=1, heap=10),
        target=10,
        blocks=1,
        is_barrier=True,
    )
    terminal = _unit(
        "resource:f:spill-terminal",
        inputs=(barrier.resource_unit_id,),
        resources=_r(cpu=1, heap=10),
        target=10,
        blocks=1,
    )
    manager = _manager(
        early,
        producer,
        barrier,
        terminal,
        resources=_r(cpu=4, heap=100, store=10),
    )
    _ready(
        manager, early.resource_unit_id, producer.resource_unit_id, barrier.resource_unit_id, terminal.resource_unit_id
    )

    producer_task = manager.try_acquire_task(_task(producer.resource_unit_id, 0, retained=0))
    barrier_task = manager.try_acquire_task(_task(barrier.resource_unit_id, 0, retained=0))
    manager.note_task_waiting(_task(early.resource_unit_id, "waiting", retained=0))
    early_normal_reason = manager._normal_task_block_reason_locked(_task(early.resource_unit_id, 0, retained=0))[0]
    early_task = manager.try_acquire_task(_task(early.resource_unit_id, 0, retained=0))
    output = manager.try_acquire_output_block(
        OutputBlockRequest(
            query_id="q",
            producer_unit_id=producer.resource_unit_id,
            task_lease_id=producer_task.lease.lease_id,
            attempt_id="0",
            block_id="materialized-input",
            size_bytes=25,
        )
    )
    snapshot = manager.snapshot()

    assert producer_task.granted and producer_task.liveness is False
    assert barrier_task.granted and barrier_task.liveness is False
    assert early_normal_reason == "query_soft_object_store_bytes"
    assert not early_task.granted
    assert early_task.blocked_reason == "liveness_not_needed"
    assert output.granted and output.liveness is False
    assert snapshot["usage"]["object_store_bytes"] > snapshot["allocation"]["resources"]["object_store_bytes"]
    assert snapshot["reservation_scope"]["object_store_unlimited_unit_ids"] == [
        producer.resource_unit_id,
        barrier.resource_unit_id,
    ]
    assert snapshot["admission"]["reservation_unit_ids"]["object_store_bytes"] == [
        early.resource_unit_id,
    ]


def test_nested_materializers_activate_spill_exemptions_on_real_demand():
    inner_input = _unit("resource:f:nested-inner-input", resources=_r(cpu=1, heap=10))
    receiver = _unit("resource:f:nested-receiver", resources=_r(cpu=1, heap=10))
    inner_barrier = _unit(
        "resource:f:nested-inner-barrier",
        inputs=(inner_input.resource_unit_id, receiver.resource_unit_id),
        resources=_r(cpu=1, heap=10),
        is_barrier=True,
    )
    middle_input = _unit("resource:f:nested-middle-input", resources=_r(cpu=1, heap=10))
    middle_barrier = _unit(
        "resource:f:nested-middle-barrier",
        inputs=(middle_input.resource_unit_id, inner_barrier.resource_unit_id),
        resources=_r(cpu=1, heap=10),
        is_barrier=True,
    )
    outer_input = _unit("resource:f:nested-outer-input", resources=_r(cpu=1, heap=10))
    outer_barrier = _unit(
        "resource:f:nested-outer-barrier",
        inputs=(outer_input.resource_unit_id, middle_barrier.resource_unit_id),
        resources=_r(cpu=1, heap=10),
        is_barrier=True,
    )
    manager = _manager(
        inner_input,
        receiver,
        inner_barrier,
        middle_input,
        middle_barrier,
        outer_input,
        outer_barrier,
        resources=_r(cpu=7, heap=100, store=10),
    )
    _ready(
        manager,
        inner_input.resource_unit_id,
        receiver.resource_unit_id,
        inner_barrier.resource_unit_id,
        middle_input.resource_unit_id,
        middle_barrier.resource_unit_id,
        outer_input.resource_unit_id,
        outer_barrier.resource_unit_id,
    )

    initial = manager.snapshot()
    outer_task = manager.try_acquire_task(_task(outer_input.resource_unit_id, 0, retained=0))
    after_outer_demand = manager.snapshot()
    middle_task = manager.try_acquire_task(_task(middle_input.resource_unit_id, 0, retained=0))
    after_middle_demand = manager.snapshot()

    assert initial["reservation_scope"]["barrier_unit_id"] == inner_barrier.resource_unit_id
    assert initial["reservation_scope"]["object_store_unlimited_unit_ids"] == [
        inner_input.resource_unit_id,
        receiver.resource_unit_id,
        inner_barrier.resource_unit_id,
    ]
    assert outer_task.granted and outer_task.liveness is False
    assert outer_input.resource_unit_id not in initial["admission"]["reservation_unit_ids"]["object_store_bytes"]
    assert after_outer_demand["units"][outer_input.resource_unit_id]["object_store_backpressure_disabled"] is True
    assert after_outer_demand["units"][middle_input.resource_unit_id]["object_store_backpressure_disabled"] is False
    assert middle_task.granted and middle_task.liveness is False
    assert after_middle_demand["units"][middle_input.resource_unit_id]["object_store_backpressure_disabled"] is True
    assert after_middle_demand["reservation_scope"]["barrier_unit_id"] == inner_barrier.resource_unit_id


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


def test_ray_tasks_receive_unique_resource_lease_slots():
    unit = _unit("resource:f:cpu", concurrency=None)
    manager = _manager(unit)
    _ready(manager, unit.resource_unit_id)

    first = manager.try_acquire_task(_task(unit.resource_unit_id, 0))
    second = manager.try_acquire_task(_task(unit.resource_unit_id, 1))

    assert first.granted and second.granted
    assert first.lease.execution_slot_id != second.lease.execution_slot_id
    assert first.lease.execution_slot_id == (f"ray_task:{unit.resource_unit_id}:{first.lease.lease_id}")


def test_idle_actor_resident_resources_remain_charged_to_query_and_node():
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

    snapshot = manager.snapshot()
    assert snapshot["usage"] == _r(cpu=1, gpu=1, heap=100).to_dict()
    assert snapshot["node_usage"]["node-a"] == _r(cpu=1, gpu=1, heap=100).to_dict()


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


def test_soft_heap_reservation_does_not_exceed_cross_dimension_unit_capacity():
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
    assert denied.blocked_reason == "hard_heap_bytes"


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
            actor_ready=True,
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
    manager.update_unit_state(upstream.resource_unit_id, runnable=True, actor_ready=True)

    held = [manager.try_acquire_task(_task(upstream.resource_unit_id, index)) for index in range(5)]
    assert all(grant.granted for grant in held)

    manager.update_unit_state(downstream.resource_unit_id, runnable=True, actor_ready=True)
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
        "resource:f:fte-downstream",
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
        "resource:f:fte-upstream",
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
    manager.update_unit_state(unit.resource_unit_id, runnable=True, actor_ready=True)
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

    manager.update_unit_state(unit.resource_unit_id, runnable=True, actor_ready=True)
    manager.update_unit_state(unit.resource_unit_id, runnable=True, actor_ready=True)

    assert changes == ["changed"]


def test_parent_fte_owns_transferable_credit_for_one_nested_udf_task():
    parent = _unit(
        "resource:f:parent-fte",
        resources=_r(cpu=1, heap=2),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    bridge = _unit(
        "resource:f:bridge-fte",
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
    manager.update_unit_state(parent.resource_unit_id, runnable=True, actor_ready=True)

    parent_grants = [manager.try_acquire_task(_task(parent.resource_unit_id, partition)) for partition in range(4)]

    assert sum(grant.granted for grant in parent_grants) == 3
    assert parent_grants[3].blocked_reason == "continuation_capacity"
    snapshot = manager.snapshot()
    assert snapshot["usage"]["heap_bytes"] == 30
    assert len(snapshot["continuation_credits"]) == 3
    assert all(credit["borrowed_by_task_lease_id"] is None for credit in snapshot["continuation_credits"].values())
    admission = snapshot["admission"]
    assert admission["live_demand_unit_ids"]["heap_bytes"] == [parent.resource_unit_id]
    assert admission["continuation_unit_ids"]["heap_bytes"] == [child.resource_unit_id]
    assert admission["reservation_unit_ids"]["heap_bytes"] == [
        parent.resource_unit_id,
        child.resource_unit_id,
    ]

    child_requests = [_task(child.resource_unit_id, partition) for partition in range(3)]
    for request in child_requests:
        manager.note_task_waiting(request)
    child_grants = [manager.try_acquire_queued_task(request) for request in child_requests]

    assert all(grant.granted for grant in child_grants)
    borrowed = manager.snapshot()
    assert borrowed["usage"]["heap_bytes"] == 30
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
    assert manager.snapshot()["usage"]["heap_bytes"] == 30


def test_nested_udf_uses_normal_allocation_after_continuation_reserve_is_borrowed():
    parent = _unit(
        "resource:f:parent-fte-normal-overflow",
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
    manager.update_unit_state(parent.resource_unit_id, runnable=True, actor_ready=True)
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
    assert snapshot["usage"] == _r(cpu=3, heap=18, store=0).to_dict()


def test_borrowed_continuation_credit_outlives_parent_until_child_releases():
    parent = _unit(
        "resource:f:parent-fte",
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
    manager.update_unit_state(parent.resource_unit_id, runnable=True, actor_ready=True)

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
        "resource:f:parent-fte",
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
    manager.update_unit_state(parent.resource_unit_id, runnable=True, actor_ready=True)

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
    assert snapshot["usage"]["heap_bytes"] == 10
    assert next(iter(snapshot["continuation_credits"].values()))["borrowed_by_task_lease_id"] is None

    second_request = _task(child.resource_unit_id, 1)
    manager.note_task_waiting(second_request)
    second_child = manager.try_acquire_queued_task(second_request)
    assert second_child.granted
    assert manager.snapshot()["usage"]["heap_bytes"] == 10


def test_continuation_credit_can_reserve_a_different_node_from_parent_fte():
    parent = _unit(
        "resource:f:parent-fte",
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
    manager.update_unit_state(parent.resource_unit_id, runnable=True, actor_ready=True)

    parent_grant = manager.try_acquire_task(_task(parent.resource_unit_id, 0, node_id="node-a"))
    assert parent_grant.granted
    credit = next(iter(manager.snapshot()["continuation_credits"].values()))
    assert credit["node_id"] == "node-b"

    child_request = _task(child.resource_unit_id, 0)
    manager.note_task_waiting(child_request)
    child_grant = manager.try_acquire_queued_task(child_request)
    assert child_grant.granted
    assert child_grant.lease.node_id == "node-b"


def test_child_outputs_keep_credit_borrowed_until_last_output_releases():
    parent = _unit(
        "resource:f:parent-fte",
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
    manager = _manager(parent, child, resources=_r(cpu=10, heap=10, store=10))
    manager.update_unit_state(parent.resource_unit_id, runnable=True, actor_ready=True)

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
    assert snapshot["usage"] == _r(cpu=2, heap=10, store=10).to_dict()
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
        "resource:f:parent-fte",
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
    manager.update_unit_state(parent.resource_unit_id, runnable=True, actor_ready=True)

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
    assert snapshot["usage"] == _r(cpu=2, heap=10, store=10).to_dict()
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
    assert with_both_outputs["usage"] == _r(cpu=2, heap=10, store=20).to_dict()
    assert (
        with_both_outputs["output_leases"][second_output.lease.lease_id]["continuation_credit_id"]
        == credit["credit_id"]
    )

    assert manager.release_output_block(first_output.lease.lease_id)
    released = manager.snapshot()
    assert released["usage"] == _r(cpu=2, heap=10, store=10).to_dict()
    credit = next(iter(released["continuation_credits"].values()))
    assert credit["borrowed_by_task_lease_id"] == after_handoff.lease.lease_id


def test_cross_unit_credit_reuse_checks_historical_outputs_before_grant():
    parent = _unit(
        "resource:f:parent-fte",
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
    manager.update_unit_state(parent.resource_unit_id, runnable=True, actor_ready=True)

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
        "resource:f:parent-fte",
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
    manager.update_unit_state(parent.resource_unit_id, runnable=True, actor_ready=True)
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
        "resource:f:parent-fte",
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
        resources=_r(cpu=10, heap=10, store=30),
    )
    manager.update_unit_state(parent.resource_unit_id, runnable=True, actor_ready=True)
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
        "resource:f:parent-fte",
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
        resources=_r(cpu=10, heap=10, store=35),
    )
    manager.update_unit_state(parent.resource_unit_id, runnable=True, actor_ready=True)
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
    assert active["usage"]["object_store_bytes"] == 35
    assert sum(unit["usage"]["object_store_bytes"] for unit in active["units"].values()) == 35

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
        "resource:f:parent-fte",
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
        resources=_r(cpu=10, heap=10, store=20),
    )
    manager.update_unit_state(parent.resource_unit_id, runnable=True, actor_ready=True)
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
        "resource:f:parent-fte",
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
    manager.update_unit_state(parent.resource_unit_id, runnable=True, actor_ready=True)
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


def test_parent_fte_admission_preserves_largest_nested_udf_commitment():
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
    manager.update_unit_state(parent.resource_unit_id, runnable=True, actor_ready=True)

    first_parent = manager.try_acquire_task(_task(parent.resource_unit_id, 0))
    second_parent = manager.try_acquire_task(_task(parent.resource_unit_id, 1))

    assert first_parent.granted
    assert not second_parent.granted
    assert second_parent.blocked_reason == "continuation_capacity"

    child_request = _task(first_child.resource_unit_id, 0)
    manager.note_task_waiting(child_request)
    child_grant = manager.try_acquire_queued_task(child_request)

    assert child_grant.granted


def test_parent_fte_placement_preserves_pinned_actor_continuation_node():
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
    manager.update_unit_state(parent.resource_unit_id, runnable=True, actor_ready=True)

    pinned_node_request = manager.try_acquire_task(_task(parent.resource_unit_id, "bad-node", node_id="node-a"))
    automatic_request = manager.try_acquire_task(_task(parent.resource_unit_id, "auto-node"))

    assert not pinned_node_request.granted
    assert pinned_node_request.blocked_reason == "node_capacity"
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
        "resource:f:placed-fte-consumer",
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
    manager.update_unit_state(producer.resource_unit_id, runnable=True, actor_ready=True)

    assert manager.task_eligible_node_ids(producer.resource_unit_id) == ("node-a", "node-b")
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


def test_latent_fte_and_actor_reservations_cap_ray_task_fanout():
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
    # The downstream units are deliberately latent. Admission must reserve
    # them before their first input makes them runnable.
    manager.update_unit_state(producer.resource_unit_id, runnable=True, actor_ready=True)

    producer_grants = [manager.try_acquire_task(_task(producer.resource_unit_id, partition)) for partition in range(9)]
    blocked_producer = manager.try_acquire_task(_task(producer.resource_unit_id, 9))

    assert all(grant.granted for grant in producer_grants)
    assert not blocked_producer.granted
    assert blocked_producer.blocked_reason == "continuation_capacity"
    reservations = manager.snapshot()["admission"]["downstream_reservations"][producer.resource_unit_id]
    assert [reservation["reservation_id"] for reservation in reservations] == [
        f"native-fragment:{producer.resource_unit_id}",
    ]
    assert sum(reservation["resources"]["object_store_bytes"] for reservation in reservations) == 0

    manager.update_unit_state(consumer.resource_unit_id, runnable=True, actor_ready=True)
    consumer_grant = manager.try_acquire_task(_task(consumer.resource_unit_id, 0))
    manager.update_unit_state(gpu_actor.resource_unit_id, runnable=True, actor_ready=True)
    actor_grant = manager.try_acquire_task(_task(gpu_actor.resource_unit_id, 0, retained=0))

    assert consumer_grant.granted
    assert actor_grant.granted and actor_grant.liveness
    assert actor_grant.lease.actor_index == 0
    assert (
        manager.snapshot()["usage"]
        == _r(
            cpu=10,
            gpu=1,
            heap=40,
            store=11,
        ).to_dict()
    )


def test_upstream_fanout_preserves_concurrent_ray_task_and_fte_slots():
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
    terminal_fte = _unit(
        "resource:f:streaming-terminal-fte",
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
        terminal_fte,
        resources=_r(cpu=20, gpu=1, heap=24, store=20),
    )
    manager.update_unit_state(upstream.resource_unit_id, runnable=True, actor_ready=True)

    upstream_grants = [manager.try_acquire_task(_task(upstream.resource_unit_id, partition)) for partition in range(8)]
    assert all(grant.granted for grant in upstream_grants)
    blocked_upstream = manager.try_acquire_task(_task(upstream.resource_unit_id, 8))
    assert not blocked_upstream.granted
    assert blocked_upstream.blocked_reason == "continuation_capacity"
    snapshot = manager.snapshot()
    assert snapshot["usage"]["heap_bytes"] == 20
    reservations = snapshot["admission"]["downstream_reservations"][upstream.resource_unit_id]
    assert [reservation["reservation_id"] for reservation in reservations] == [
        f"native-fragment:{upstream.resource_unit_id}",
        f"actor_continuation:{upstream.resource_unit_id}:{downstream.resource_unit_id}",
    ]

    manager.update_unit_state(actor.resource_unit_id, runnable=True, actor_ready=True)
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
        "grant_class": "normal",
    }
    assert downstream_grant.granted
    assert not downstream_grant.liveness
    assert manager.snapshot()["usage"]["heap_bytes"] == 22

    terminal_request = _task(terminal_fte.resource_unit_id, 0)
    manager.note_task_waiting(terminal_request)
    terminal_grant = manager.try_acquire_queued_task(terminal_request)

    assert terminal_grant.granted
    assert not terminal_grant.liveness
    assert manager.snapshot()["usage"]["heap_bytes"] == 24


def test_registered_minimum_supports_parent_task_and_downstream_fte_progress():
    parent = _unit(
        "resource:f:image-parent-fte",
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
        "resource:f:image-downstream-fte",
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
    # actor resident + invocation, one Ray task credit, the invoking FTE task,
    # and one downstream FTE progress slot: this is the registered minimum.
    manager = _manager(
        parent,
        producer,
        consumer,
        actor,
        resources=_r(cpu=3, gpu=1, heap=20, store=2),
    )
    manager.update_unit_state(parent.resource_unit_id, runnable=True, actor_ready=True)

    parent_grant = manager.try_acquire_task(_task(parent.resource_unit_id, 0))
    manager.update_unit_state(producer.resource_unit_id, runnable=True, actor_ready=True)
    producer_request = _task(producer.resource_unit_id, 0)
    manager.note_task_waiting(producer_request)
    producer_grant = manager.try_acquire_queued_task(producer_request)
    manager.update_unit_state(consumer.resource_unit_id, runnable=True, actor_ready=True)
    consumer_grant = manager.try_acquire_task(_task(consumer.resource_unit_id, 0))
    manager.update_unit_state(actor.resource_unit_id, runnable=True, actor_ready=True)
    actor_grant = manager.try_acquire_task(_task(actor.resource_unit_id, 0, retained=0))

    assert parent_grant.granted
    assert producer_grant.granted
    assert consumer_grant.granted
    assert actor_grant.granted
    assert (
        manager.snapshot()["usage"]
        == _r(
            cpu=3,
            gpu=1,
            heap=19,
            store=2,
        ).to_dict()
    )


def test_parent_fte_fanout_preserves_shared_fte_slot_across_nested_ray_task():
    parent = _unit(
        "resource:f:image-parent-fte",
        resources=_r(cpu=1, heap=1),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    direct_fte = _unit(
        "resource:f:image-direct-fte",
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
        "resource:f:image-downstream-fte",
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
    manager.update_unit_state(parent.resource_unit_id, runnable=True, actor_ready=True)

    parent_grants = [manager.try_acquire_task(_task(parent.resource_unit_id, partition)) for partition in range(8)]

    assert all(grant.granted for grant in parent_grants[:7])
    assert not parent_grants[7].granted
    assert parent_grants[7].blocked_reason == "continuation_capacity"
    snapshot = manager.snapshot()
    assert snapshot["usage"]["heap_bytes"] == 43
    reservations = snapshot["admission"]["downstream_reservations"][parent.resource_unit_id]
    assert len(reservations) == 1
    assert reservations[0]["reservation_id"] == f"native-fragment:{parent.resource_unit_id}"
    assert reservations[0]["unit_ids"] == [downstream_fte.resource_unit_id]
    assert reservations[0]["resources"]["heap_bytes"] == 4

    manager.update_unit_state(cpu_udf.resource_unit_id, runnable=True, actor_ready=True)
    cpu_request = _task(cpu_udf.resource_unit_id, 0)
    manager.note_task_waiting(cpu_request)
    cpu_grant = manager.try_acquire_queued_task(cpu_request)
    manager.update_unit_state(
        downstream_fte.resource_unit_id,
        runnable=True,
        actor_ready=True,
    )
    downstream_grant = manager.try_acquire_task(_task(downstream_fte.resource_unit_id, 0))

    assert cpu_grant.granted
    assert downstream_grant.granted
    final_snapshot = manager.snapshot()
    assert cpu_grant.lease.lease_id in {
        credit["borrowed_by_task_lease_id"] for credit in final_snapshot["continuation_credits"].values()
    }
    assert final_snapshot["usage"]["heap_bytes"] == 47


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
    assert snapshot["usage"]["object_store_bytes"] == 100
    assert snapshot["liveness"]["active_task_lease_id"] is None


def test_soft_budget_does_not_start_extra_same_unit_work_while_task_is_active():
    unit = _unit("resource:f:decode", resources=_r(cpu=1, heap=100), target=30, blocks=2)
    manager = _manager(unit, resources=_r(cpu=10, heap=1_000, store=100))
    _ready(manager, unit.resource_unit_id, consumer_waiting=True)

    first = manager.try_acquire_task(_task(unit.resource_unit_id, 0))
    second = manager.try_acquire_task(_task(unit.resource_unit_id, 1))

    assert first.granted
    assert not second.granted
    assert second.blocked_reason == "liveness_not_needed"
    assert manager.snapshot()["usage"]["object_store_bytes"] == 60


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
    assert manager.snapshot()["usage"]["object_store_bytes"] == 101


def test_retained_input_uses_exact_dynamic_credit_above_nominal_target():
    unit = _unit("resource:f:decode", resources=_r(cpu=1, heap=100, store=30))
    manager = _manager(unit, resources=_r(cpu=10, heap=1_000, store=1_000))
    _ready(manager, unit.resource_unit_id)

    granted = manager.try_acquire_task(_task(unit.resource_unit_id, 0, retained=31))

    assert granted.granted
    assert granted.lease.resources.object_store_bytes == 31
    assert manager.snapshot()["usage"]["object_store_bytes"] == 51


def test_retained_input_larger_than_soft_budget_uses_one_liveness_task():
    unit = _unit("resource:f:decode", resources=_r(cpu=1, heap=100, store=81))
    manager = _manager(unit, resources=_r(cpu=10, heap=1_000, store=100))
    _ready(manager, unit.resource_unit_id)

    request = _task(unit.resource_unit_id, 0, retained=81)
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
    manager.update_unit_state(parent.resource_unit_id, runnable=True, actor_ready=True)

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
    assert child_grant.granted and child_grant.liveness
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
    assert output.granted and not output.liveness
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
    assert producer_owned["liveness"]["active_task_lease_id"] == parent_grant.lease.lease_id
    blocked = manager.try_acquire_task(_task(parent.resource_unit_id, 1))
    assert not blocked.granted
    assert blocked.blocked_reason == "liveness_task_active"

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
    manager.update_unit_state(parent.resource_unit_id, runnable=True, actor_ready=True)
    parent_grant = manager.try_acquire_task(_task(parent.resource_unit_id, 0))
    manager.update_unit_state(first_actor.resource_unit_id, runnable=True, actor_ready=True)
    manager.update_unit_state(sibling_actor.resource_unit_id, runnable=True, actor_ready=True)

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
    assert output.granted and not output.liveness
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
    manager.update_unit_state(parent.resource_unit_id, runnable=True, actor_ready=True)
    parent_grant = manager.try_acquire_task(_task(parent.resource_unit_id, 0))
    manager.update_unit_state(actor.resource_unit_id, runnable=True, actor_ready=True)
    actor_request = _task(actor.resource_unit_id, 0)
    manager.note_task_waiting(actor_request)
    selected, actor_grant = manager.try_acquire_next_queued_task({(actor_request.task_id, actor_request.attempt_id)})

    assert selected == actor_request
    assert actor_grant is not None
    assert parent_grant.granted and not parent_grant.liveness
    assert actor_grant.granted and actor_grant.liveness
    downstream_grant = manager.try_acquire_task_descriptor(_task(downstream.resource_unit_id, 0))
    assert downstream_grant.granted and downstream_grant.liveness
    assert manager.snapshot()["liveness"]["active_task_lease_id"] == actor_grant.lease.lease_id

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
    unit = _unit("resource:f:decode", resources=_r(cpu=1, heap=100, store=81))
    manager = _manager(unit, resources=_r(cpu=10, heap=1_000, store=100))
    _ready(manager, unit.resource_unit_id)

    first_request = _task(unit.resource_unit_id, 0, retained=81)
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
    second_request = _task(unit.resource_unit_id, 1, retained=81)
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


def test_output_blocks_replace_unused_task_window_without_double_counting():
    unit = _unit("resource:f:decode", resources=_r(cpu=1, heap=100, store=10), target=50, blocks=2)
    manager = _manager(unit, resources=_r(cpu=10, heap=1_000, store=1_000))
    _ready(manager, unit.resource_unit_id)
    task = manager.try_acquire_task(_task(unit.resource_unit_id, 0))
    assert manager.snapshot()["usage"]["object_store_bytes"] == 110

    first = manager.try_acquire_output_block(
        OutputBlockRequest("q", unit.resource_unit_id, task.lease.lease_id, "0", "block-1", 40)
    )
    second = manager.try_acquire_output_block(
        OutputBlockRequest("q", unit.resource_unit_id, task.lease.lease_id, "0", "block-2", 60)
    )

    assert first.granted and second.granted
    assert manager.snapshot()["usage"]["object_store_bytes"] == 110

    third = manager.try_acquire_output_block(
        OutputBlockRequest("q", unit.resource_unit_id, task.lease.lease_id, "0", "block-3", 25)
    )
    assert third.granted
    assert manager.snapshot()["usage"]["object_store_bytes"] == 135


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
        "resource:f:fte",
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


def test_fte_task_completion_rejects_outputs_above_precommitted_window_without_mutation():
    unit = _unit(
        "resource:f:fte",
        resources=_r(cpu=1, heap=100),
        target=10,
        blocks=2,
        backend="ray_worker",
    )
    manager = _manager(unit)
    _ready(manager, unit.resource_unit_id)
    task = manager.try_acquire_task(_task(unit.resource_unit_id, 0))

    with pytest.raises(RuntimeError, match="output bytes 21 exceed task window 20"):
        manager.finish_task_with_outputs(
            task.lease.lease_id,
            attempt_id="0",
            outputs=(
                OutputBlockRequest("q", unit.resource_unit_id, task.lease.lease_id, "0", "fte-block-0", 10),
                OutputBlockRequest("q", unit.resource_unit_id, task.lease.lease_id, "0", "fte-block-1", 11),
            ),
        )

    snapshot = manager.snapshot()
    assert set(snapshot["task_leases"]) == {task.lease.lease_id}
    assert snapshot["output_leases"] == {}


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
    second = manager.try_acquire_output_block(
        OutputBlockRequest("q", unit.resource_unit_id, task.lease.lease_id, "0", "second", 1)
    )
    assert not second.granted
    assert second.blocked_reason == "liveness_output_active"
    assert manager.transition_output_block(granted.lease.lease_id, "unit_queue")
    assert manager.transition_output_block(granted.lease.lease_id, "downstream_input")
    next_block = manager.try_acquire_output_block(
        OutputBlockRequest("q", unit.resource_unit_id, task.lease.lease_id, "0", "second-after-handoff", 1)
    )
    assert next_block.granted
    assert next_block.liveness
    assert manager.snapshot()["soft_object_store_debt_bytes"] == 2


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
    assert manager.snapshot()["soft_object_store_debt_bytes"] == 1
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
    expected_usage = _r(cpu=1, heap=100, store=20).to_dict()
    assert blocked.granted is False
    assert blocked.blocked_reason == "allocation_pending"
    assert blocked.fatal is False
    assert snapshot["allocation_admission_open"] is False
    assert snapshot["allocation_debt"] == _r(cpu=1, heap=100).to_dict()
    assert snapshot["soft_object_store_debt_bytes"] == 20
    assert snapshot["node_usage"]["node-a"] == expected_usage
    assert snapshot["node_allocation_debt"]["node-a"] == _r(cpu=1, heap=100).to_dict()
    assert snapshot["node_soft_object_store_debt_bytes"]["node-a"] == 20

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
    )
    node_capacity = _r(cpu=1, heap=10, store=20)
    manager = _manager(
        unit,
        resources=_r(cpu=2, heap=20, store=40),
        nodes=(("node-a", node_capacity), ("node-b", node_capacity)),
    )
    _ready(manager, unit.resource_unit_id)

    first = manager.try_acquire_task(_task(unit.resource_unit_id, 1))
    second = manager.try_acquire_task(_task(unit.resource_unit_id, 2))
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
