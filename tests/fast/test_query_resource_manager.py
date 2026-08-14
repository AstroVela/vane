# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import logging
import random

import pytest

from vane.runners.ray import query_resource_manager as manager_module
from vane.runners.ray.query_resource_graph import (
    MaterializationBarrierSpec,
    QueryAllocation,
    QueryResourceGraph,
    ResourceUnitSpec,
    ResourceVector,
)
from vane.runners.ray.query_resource_manager import (
    OutputBlockRequest,
    RayQueryResourceManager,
    TaskRequest,
)


def _r(*, cpu=0.0, gpu=0.0, heap=0, store=0):
    return ResourceVector(cpu=cpu, gpu=gpu, heap_bytes=heap, object_store_bytes=store)


def _allocation(resources, *, generation=1, nodes=None):
    del nodes
    return QueryAllocation(
        resources=resources,
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
    physical_node_id = resource_unit_id.rsplit(":", 1)[-1]
    if backend == "ray_worker":
        physical_node_id = f"node:{physical_node_id}:native-fragment"
    return ResourceUnitSpec(
        query_id="q",
        resource_unit_id=resource_unit_id,
        physical_node_id=physical_node_id,
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


def _assert_object_store_budget_invariants(snapshot):
    state = snapshot["admission"]["object_store"]
    budgets = [unit["object_store_budget"] for unit in snapshot["units"].values()]
    total_reserved = sum(
        budget["task_reserved_bytes"] + budget["output_reserved_bytes"]
        for budget in budgets
        if budget["reservation_eligible"]
    )

    assert state["ineligible_usage_bytes"] == sum(budget["ineligible_usage_bytes"] for budget in budgets)
    assert state["shared_used_bytes"] == sum(budget["shared_used_bytes"] for budget in budgets)
    assert state["shared_pool_bytes"] == max(
        0,
        state["limit_bytes"] - state["ineligible_usage_bytes"] - total_reserved,
    )
    assert state["shared_remaining_bytes"] == max(
        0,
        state["shared_pool_bytes"] - state["shared_used_bytes"],
    )
    assert state["query_usage_bytes"] == sum(
        budget["task_internal_usage_bytes"] + budget["output_usage_bytes"] for budget in budgets
    )
    for budget in budgets:
        if budget["reservation_eligible"]:
            assert budget["ineligible_usage_bytes"] == 0
        else:
            assert budget["task_reserved_bytes"] == 0
            assert budget["output_reserved_bytes"] == 0
            assert budget["shared_used_bytes"] == 0


@pytest.mark.parametrize(
    ("task_reserved", "output_reserved"),
    [(0, 0), (0, 3), (3, 0), (3, 4), (7, 8)],
)
def test_object_store_unit_budget_exhaustively_partitions_protected_and_shared_usage(
    task_reserved,
    output_reserved,
):
    for task_internal_usage in range(10):
        for output_usage in range(10):
            budget = manager_module._ObjectStoreUnitBudget(
                task_reserved_bytes=task_reserved,
                output_reserved_bytes=output_reserved,
                task_internal_usage_bytes=task_internal_usage,
                output_usage_bytes=output_usage,
            )
            output_protected_used = min(output_usage, output_reserved)
            expected_task_budget_usage = task_internal_usage + max(0, output_usage - output_reserved)
            task_protected_used = min(expected_task_budget_usage, task_reserved)

            assert budget.task_budget_usage_bytes == expected_task_budget_usage
            assert budget.task_reserved_remaining_bytes == max(0, task_reserved - expected_task_budget_usage)
            assert budget.output_reserved_remaining_bytes == max(0, output_reserved - output_usage)
            assert budget.shared_used_bytes == max(0, expected_task_budget_usage - task_reserved)
            assert task_internal_usage + output_usage == (
                output_protected_used + task_protected_used + budget.shared_used_bytes
            )


@pytest.mark.parametrize("reservation_ratio", [-0.01, 1.01, float("nan"), float("inf")])
def test_reservation_ratio_rejects_nonfinite_or_out_of_range_values(reservation_ratio):
    with pytest.raises(ValueError, match=r"reservation_ratio must be in \[0, 1\]"):
        _manager(_unit("resource:f:ratio-invalid"), reservation_ratio=reservation_ratio)


@pytest.mark.parametrize(
    ("reservation_ratio", "task_reserved", "output_reserved", "shared_pool"),
    [
        (0.0, 0, 0, 7),
        (0.5, 1, 2, 4),
        (1.0, 3, 4, 0),
    ],
)
def test_object_store_reservation_ratio_endpoints_and_odd_byte_rounding(
    reservation_ratio,
    task_reserved,
    output_reserved,
    shared_pool,
):
    unit = _unit("resource:f:ratio", target=1, blocks=1)
    manager = _manager(
        unit,
        resources=_r(cpu=10, heap=100, store=7),
        reservation_ratio=reservation_ratio,
    )
    snapshot = manager.snapshot()

    _assert_object_store_budget_invariants(snapshot)
    assert snapshot["admission"]["object_store"]["shared_pool_bytes"] == shared_pool
    assert snapshot["units"][unit.resource_unit_id]["object_store_budget"] == {
        "reservation_eligible": True,
        "task_reserved_bytes": task_reserved,
        "output_reserved_bytes": output_reserved,
        "task_internal_usage_bytes": 0,
        "output_usage_bytes": 0,
        "ineligible_usage_bytes": 0,
        "shared_used_bytes": 0,
    }


def test_zero_reservation_ratio_uses_only_shared_credit_then_bounded_liveness():
    unit = _unit("resource:f:zero-ratio", target=1, blocks=1)
    manager = _manager(
        unit,
        resources=_r(cpu=10, heap=100, store=2),
        reservation_ratio=0,
    )
    _ready(manager, unit.resource_unit_id, consumer_waiting=True)

    first = manager.try_acquire_task(_task(unit.resource_unit_id, 0))
    second = manager.try_acquire_task(_task(unit.resource_unit_id, 1))
    third_request = _task(unit.resource_unit_id, 2)
    assert first.granted and not first.liveness
    assert second.granted and not second.liveness
    assert manager._normal_task_block_reason_locked(third_request)[0] == "query_soft_object_store_bytes"
    third = manager.try_acquire_task(third_request)
    assert not third.granted and third.blocked_reason == "liveness_task_active"

    output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            unit.resource_unit_id,
            first.lease.lease_id,
            first.lease.attempt_id,
            "zero-ratio-liveness-output",
            1,
        )
    )
    assert output.granted and output.liveness


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
    final_materializer = manager.try_acquire_task(_task(materializer.resource_unit_id, 0, node_id="node-a"))
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


def test_completed_resource_unit_cannot_be_reopened():
    unit = _unit("resource:f:finished", target=0, blocks=0)
    manager = _manager(unit, resources=_r(cpu=1, heap=10))
    manager.update_unit_state(unit.resource_unit_id, runnable=False, completed=True)

    with pytest.raises(RuntimeError, match="completed resource unit cannot become incomplete"):
        manager.update_unit_state(unit.resource_unit_id, runnable=True)

    state = manager.snapshot()["units"][unit.resource_unit_id]
    assert state["runnable"] is False
    assert state["completed"] is True


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


def test_cancelled_manager_cannot_republish_actor_process_usage():
    actor = _unit(
        "resource:f:cancelled-actor",
        resources=_r(),
        resident=_r(cpu=1, heap=100),
        backend="ray_actor",
        actor_pool_size=1,
    )
    manager = _manager(actor, resources=_r(cpu=1, heap=100))
    manager.set_submitted_actor_slots(actor.resource_unit_id, {0})
    manager.set_ready_actor_slots(actor.resource_unit_id, {0: "node-a"})

    manager.cancel("planned cancellation")

    assert manager.current_eligible_resource_unit_ids() == ()
    with pytest.raises(RuntimeError, match="cancelled query"):
        manager.set_submitted_actor_slots(actor.resource_unit_id, {0})
    with pytest.raises(RuntimeError, match="cancelled query"):
        manager.set_ready_actor_slots(actor.resource_unit_id, {0: "node-a"})
    snapshot = manager.snapshot()
    assert snapshot["submitted_actor_slots"] == []
    assert snapshot["ready_actor_slots"] == {}
    assert snapshot["retiring_actor_unit_ids"] == []
    assert snapshot["actor_process_usage"] == _r().to_dict()


def test_failed_manager_fences_new_work_but_preserves_live_physical_usage():
    actor = _unit(
        "resource:f:failed-actor",
        resources=_r(),
        resident=_r(cpu=1, heap=100),
        backend="ray_actor",
        actor_pool_size=1,
    )
    manager = _manager(actor, resources=_r(cpu=1, heap=100, store=20))
    _ready(manager, actor.resource_unit_id)
    task = manager.try_acquire_task(_task(actor.resource_unit_id, 0))
    assert task.granted
    output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            actor.resource_unit_id,
            task.lease.lease_id,
            task.lease.attempt_id,
            "failed-query-output",
            5,
        )
    )
    assert output.granted

    retained = manager.fail("planned terminal failure")

    assert retained == {"task_lease_count": 1, "output_lease_count": 1}
    snapshot = manager.snapshot()
    assert snapshot["failed"] is True
    assert snapshot["failure_reason"] == "planned terminal failure"
    assert snapshot["cancelled"] is False
    assert snapshot["allocation_admission_open"] is False
    assert snapshot["actor_process_usage"] == _r(cpu=1, heap=100).to_dict()
    assert len(snapshot["task_leases"]) == 1
    assert len(snapshot["output_leases"]) == 1

    blocked = manager.try_acquire_task(_task(actor.resource_unit_id, 1))
    assert blocked.granted is False
    assert blocked.fatal is True
    assert blocked.blocked_reason == "query_failed"
    with pytest.raises(RuntimeError, match="failed query"):
        manager.set_submitted_actor_slots(actor.resource_unit_id, {0})

    manager.update_allocation(
        _allocation(_r(cpu=1, heap=100, store=20), generation=2),
        admission_open=True,
    )
    assert manager.snapshot()["allocation_admission_open"] is False

    manager.cancel("ordered teardown completed")
    released = manager.snapshot()
    assert released["actor_process_usage"] == _r().to_dict()
    assert released["task_leases"] == {}
    assert released["output_leases"] == {}


def test_ray_tasks_receive_unique_resource_lease_slots():
    unit = _unit("resource:f:cpu", concurrency=None)
    manager = _manager(unit)
    _ready(manager, unit.resource_unit_id)

    first = manager.try_acquire_task(_task(unit.resource_unit_id, 0))
    second = manager.try_acquire_task(_task(unit.resource_unit_id, 1))

    assert first.granted and second.granted
    assert first.lease.execution_slot_id != second.lease.execution_slot_id
    assert first.lease.execution_slot_id == (f"ray_task:{unit.resource_unit_id}:{first.lease.lease_id}")


def test_submitted_actor_processes_are_charged_once_to_soft_usage():
    actor = _unit(
        "resource:f:actor",
        resources=_r(store=10),
        resident=_r(cpu=1, gpu=0.5, heap=100),
        backend="ray_actor",
        actor_pool_size=2,
        actor_prefetch_depth=2,
    )
    manager = _manager(
        actor,
        resources=_r(cpu=2, gpu=1, heap=200, store=100),
    )

    manager.set_submitted_actor_slots(actor.resource_unit_id, {0, 1})
    snapshot = manager.snapshot()

    assert snapshot["usage"] == _r().to_dict()
    assert snapshot["actor_process_usage"] == _r(cpu=2, gpu=1, heap=200).to_dict()
    assert snapshot["soft_allocation_usage"] == _r(cpu=2, gpu=1, heap=200).to_dict()
    assert snapshot["pending_actor_slots"] == [
        f"{actor.resource_unit_id}:0",
        f"{actor.resource_unit_id}:1",
    ]

    manager.set_ready_actor_slots(actor.resource_unit_id, {0: "node-a", 1: "node-b"})
    _ready(manager, actor.resource_unit_id)
    first = manager.try_acquire_task(_task(actor.resource_unit_id, 0))
    second = manager.try_acquire_task(_task(actor.resource_unit_id, 1))

    assert first.granted and second.granted
    # Invocations charge only their dynamic input/output bytes. The actor
    # process CPU/GPU/heap is not multiplied by batches.
    assert (
        manager.snapshot()["soft_allocation_usage"]
        == _r(
            cpu=2,
            gpu=1,
            heap=200,
            # Two retained 10-byte inputs plus one declared 20-byte
            # generator window for each active actor slot.
            store=60,
        ).to_dict()
    )


def test_actor_invocation_uses_the_ready_actor_runtime_node():
    actor = _unit(
        "resource:f:external-actor",
        resources=_r(store=10),
        resident=_r(cpu=1, heap=100),
        backend="ray_actor",
        actor_pool_size=1,
    )
    manager = _manager(actor, resources=_r(cpu=1, heap=100, store=100))
    manager.set_submitted_actor_slots(actor.resource_unit_id, {0})
    manager.set_ready_actor_slots(actor.resource_unit_id, {0: "actual-node"})
    manager.update_unit_state(actor.resource_unit_id, runnable=True)

    grant = manager.try_acquire_task(_task(actor.resource_unit_id, 0))

    assert grant.granted and not grant.liveness
    assert grant.lease.node_id == "actual-node"
    assert grant.lease.actor_index == 0
    assert manager.snapshot()["actor_process_usage_by_ready_node"] == {
        "actual-node": _r(cpu=1, heap=100).to_dict(),
    }


def test_ray_task_lease_is_unpinned_and_rejects_a_requested_node():
    task = _unit(
        "resource:f:ray-task",
        resources=_r(cpu=1, gpu=0.25, heap=100),
        target=0,
        blocks=0,
    )
    manager = _manager(task, resources=_r(cpu=2, gpu=1, heap=200))
    _ready(manager, task.resource_unit_id)

    pinned = manager.try_acquire_task(_task(task.resource_unit_id, "pinned", node_id="node-a"))
    unpinned = manager.try_acquire_task(_task(task.resource_unit_id, "unpinned"))

    assert pinned.fatal and pinned.blocked_reason == "ray_task_node_must_be_unset"
    assert unpinned.granted and not unpinned.liveness
    assert unpinned.lease.node_id is None
    assert unpinned.lease.resources == _r(cpu=1, gpu=0.25, heap=100)


def test_native_fragment_lease_keeps_actual_worker_node_but_no_process_charge():
    native = _unit(
        "resource:f:native",
        resources=_r(store=10),
        target=10,
        blocks=2,
        backend="ray_worker",
    )
    manager = _manager(native, resources=_r(store=100))
    _ready(manager, native.resource_unit_id)

    missing = manager.try_acquire_task(_task(native.resource_unit_id, "missing"))
    granted = manager.try_acquire_task(_task(native.resource_unit_id, "actual", node_id="node-b"))

    assert missing.fatal and missing.blocked_reason == "ray_worker_node_required"
    assert granted.granted
    assert granted.lease.node_id == "node-b"
    assert granted.lease.resources == _r(store=10)
    assert manager.snapshot()["usage"] == _r(store=30).to_dict()


def test_zero_soft_budget_submits_one_real_ray_task_for_liveness():
    task = _unit(
        "resource:f:autoscaling-demand",
        resources=_r(cpu=4, gpu=2, heap=1_000),
        target=0,
        blocks=0,
    )
    manager = _manager(task, resources=_r())
    _ready(manager, task.resource_unit_id)

    first = manager.try_acquire_task(_task(task.resource_unit_id, 0))
    second = manager.try_acquire_task(_task(task.resource_unit_id, 1))

    assert first.granted and first.liveness
    assert first.lease.node_id is None
    assert first.lease.resources == _r(cpu=4, gpu=2, heap=1_000)
    assert not second.granted and second.blocked_reason == "liveness_task_active"

    assert manager.release_task_lease(first.lease.lease_id, attempt_id=first.lease.attempt_id)
    third = manager.try_acquire_task(_task(task.resource_unit_id, 2))
    assert third.granted and third.liveness


@pytest.mark.parametrize(
    ("resources", "reason"),
    [
        (_r(cpu=0.5, gpu=1, heap=100), "query_soft_cpu"),
        (_r(cpu=1, gpu=0.5, heap=100), "query_soft_gpu"),
        (_r(cpu=1, gpu=1, heap=99), "query_soft_heap_bytes"),
    ],
)
def test_cpu_gpu_and_declared_heap_are_soft_backpressure_dimensions(resources, reason):
    task = _unit(
        "resource:f:shape",
        resources=_r(cpu=1, gpu=1, heap=100),
        target=0,
        blocks=0,
    )
    manager = _manager(task, resources=resources)
    _ready(manager, task.resource_unit_id)
    request = _task(task.resource_unit_id, 0)

    assert manager._normal_task_block_reason_locked(request)[0] == reason
    grant = manager.try_acquire_task(request)

    assert grant.granted and grant.liveness


def test_fixed_actor_soft_debt_does_not_block_zero_increment_invocations():
    actor = _unit(
        "resource:f:soft-debt-actor",
        resources=_r(),
        resident=_r(cpu=1, gpu=1, heap=100),
        target=0,
        blocks=0,
        backend="ray_actor",
        actor_pool_size=1,
    )
    manager = _manager(actor, resources=_r(cpu=1, gpu=1, heap=100))
    manager.set_submitted_actor_slots(actor.resource_unit_id, {0})
    manager.set_ready_actor_slots(actor.resource_unit_id, {0: "node-a"})
    manager.update_unit_state(actor.resource_unit_id, runnable=True)
    manager.update_allocation(
        _allocation(_r(cpu=0.5, gpu=0.5, heap=50), generation=2),
        admission_open=True,
    )

    invocation = manager.try_acquire_task(_task(actor.resource_unit_id, 0))
    snapshot = manager.snapshot()

    assert invocation.granted and not invocation.liveness
    assert snapshot["soft_allocation_debt"] == _r(cpu=0.5, gpu=0.5, heap=50).to_dict()


def test_actor_debt_still_allows_one_real_ray_task_to_reach_ray_core():
    actor = _unit(
        "resource:f:resident-actor",
        resources=_r(),
        resident=_r(cpu=1, heap=100),
        backend="ray_actor",
        actor_pool_size=1,
    )
    task = _unit(
        "resource:f:new-ray-task",
        resources=_r(cpu=1, heap=100),
        target=0,
        blocks=0,
    )
    manager = _manager(
        actor,
        task,
        resources=_r(cpu=1, heap=100),
        terminals=(actor.resource_unit_id, task.resource_unit_id),
    )
    manager.set_submitted_actor_slots(actor.resource_unit_id, {0})
    _ready(manager, task.resource_unit_id)

    first = manager.try_acquire_task(_task(task.resource_unit_id, 0))
    second = manager.try_acquire_task(_task(task.resource_unit_id, 1))

    assert first.granted and first.liveness
    assert not second.granted and second.blocked_reason == "liveness_task_active"


def test_persistent_soft_actor_debt_warns_once_after_ray_data_delay(monkeypatch, caplog):
    actor = _unit(
        "resource:f:oversubscribed-actor",
        resources=_r(),
        resident=_r(cpu=1, gpu=1, heap=100),
        backend="ray_actor",
        actor_pool_size=1,
    )
    manager = _manager(actor, resources=_r(cpu=1, gpu=1, heap=100))
    manager.set_submitted_actor_slots(actor.resource_unit_id, {0})
    manager.update_allocation(
        _allocation(_r(cpu=0.5, gpu=0.5, heap=50), generation=2),
        admission_open=True,
    )
    clock = iter((0.0, 59.0, 60.0, 61.0))
    monkeypatch.setattr(manager_module.time, "monotonic", lambda: next(clock))

    with caplog.at_level(logging.WARNING):
        first = manager.snapshot()
        manager.snapshot()
        warned = manager.snapshot()
        manager.snapshot()

    assert first["soft_allocation_debt"] == _r(cpu=0.5, gpu=0.5, heap=50).to_dict()
    assert warned["soft_allocation_debt_duration_s"] == pytest.approx(60.0)
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert "soft resource reservation" in messages[0]


def test_phase_reservations_include_all_eligible_resource_owners():
    cpu = _unit(
        "resource:f:cpu",
        resources=_r(cpu=1, heap=10),
        target=0,
        blocks=0,
    )
    actor = _unit(
        "resource:f:actor",
        inputs=(cpu.resource_unit_id,),
        resources=_r(store=5),
        resident=_r(gpu=1, heap=20),
        backend="ray_actor",
        actor_pool_size=1,
    )
    native = _unit(
        "resource:f:native",
        inputs=(actor.resource_unit_id,),
        resources=_r(store=5),
        backend="ray_worker",
    )
    manager = _manager(cpu, actor, native, resources=_r(cpu=4, gpu=1, heap=100, store=100))

    reservations = manager.snapshot()["admission"]["reservation_unit_ids"]

    assert reservations["cpu"] == [cpu.resource_unit_id]
    assert reservations["gpu"] == [actor.resource_unit_id]
    assert reservations["heap_bytes"] == [cpu.resource_unit_id, actor.resource_unit_id]
    assert reservations["object_store_bytes"] == [
        actor.resource_unit_id,
        native.resource_unit_id,
    ]


def test_allocation_shrink_keeps_live_lease_and_uses_liveness_after_drain():
    task = _unit(
        "resource:f:shrink",
        resources=_r(cpu=1, heap=100),
        target=0,
        blocks=0,
    )
    manager = _manager(task, resources=_r(cpu=2, heap=200))
    _ready(manager, task.resource_unit_id)
    live = manager.try_acquire_task(_task(task.resource_unit_id, 0))
    assert live.granted and not live.liveness

    manager.update_allocation(
        _allocation(_r(cpu=0.5, heap=50), generation=2),
        admission_open=True,
    )
    blocked = manager.try_acquire_task(_task(task.resource_unit_id, 1))
    snapshot = manager.snapshot()

    assert live.lease.lease_id in snapshot["task_leases"]
    assert snapshot["soft_allocation_debt"] == _r(cpu=0.5, heap=50).to_dict()
    assert not blocked.granted and blocked.blocked_reason == "liveness_task_active"

    manager.release_task_lease(live.lease.lease_id, attempt_id=live.lease.attempt_id)
    escaped = manager.try_acquire_task(_task(task.resource_unit_id, 2))
    assert escaped.granted and escaped.liveness


def test_soft_reservations_protect_parallel_units_but_remain_bypassable_at_global_idle():
    left = _unit(
        "resource:f:left",
        resources=_r(cpu=1, heap=10),
        target=0,
        blocks=0,
    )
    right = _unit(
        "resource:f:right",
        resources=_r(cpu=1, heap=10),
        target=0,
        blocks=0,
    )
    manager = _manager(
        left,
        right,
        resources=_r(cpu=2, heap=20),
        terminals=(left.resource_unit_id, right.resource_unit_id),
    )
    _ready(manager, left.resource_unit_id, right.resource_unit_id)

    left_grant = manager.try_acquire_task(_task(left.resource_unit_id, 0))
    right_grant = manager.try_acquire_task(_task(right.resource_unit_id, 0))
    extra_left = manager.try_acquire_task(_task(left.resource_unit_id, 1))

    assert left_grant.granted and right_grant.granted
    assert not extra_left.granted
    assert extra_left.blocked_reason == "liveness_task_active"


def test_descriptor_admission_does_not_persist_a_denied_request():
    task = _unit(
        "resource:f:descriptor",
        resources=_r(cpu=1, heap=10),
        target=0,
        blocks=0,
    )
    manager = _manager(task, resources=_r())
    request = _task(task.resource_unit_id, 0)

    grant = manager.try_acquire_task_descriptor(request)

    assert grant.granted and grant.liveness
    assert manager.snapshot()["admission"]["waiting_tasks"] == []


def test_idle_runnable_unit_without_real_work_does_not_suppress_liveness():
    blocked = _unit(
        "resource:f:blocked",
        resources=_r(cpu=2),
        target=0,
        blocks=0,
    )
    idle = _unit(
        "resource:f:idle",
        resources=_r(cpu=0.25),
        target=0,
        blocks=0,
    )
    manager = _manager(
        blocked,
        idle,
        resources=_r(cpu=1),
        terminals=(blocked.resource_unit_id, idle.resource_unit_id),
    )
    _ready(manager, blocked.resource_unit_id, idle.resource_unit_id)

    grant = manager.try_acquire_task(_task(blocked.resource_unit_id, 0))

    assert grant.granted and grant.liveness
    assert manager.snapshot()["admission"]["waiting_tasks"] == []


def test_identical_waiter_and_unit_state_updates_do_not_publish_spurious_edges():
    changes = []
    unit = _unit("resource:f:edge", target=0, blocks=0)
    manager = _manager(unit, on_change=lambda: changes.append("changed"))
    request = _task(unit.resource_unit_id, 0)

    manager.update_unit_state(unit.resource_unit_id, runnable=True)
    after_first_state = len(changes)
    manager.update_unit_state(unit.resource_unit_id, runnable=True)
    assert len(changes) == after_first_state

    manager.note_task_waiting(request)
    after_first_waiter = len(changes)
    manager.note_task_waiting(request)
    assert len(changes) == after_first_waiter


def test_object_store_ledgers_remain_disjoint_through_a_mixed_lifecycle():
    upstream = _unit("resource:f:ledger-upstream", target=1, blocks=1)
    downstream = _unit(
        "resource:f:ledger-downstream",
        inputs=(upstream.resource_unit_id,),
        target=1,
        blocks=1,
    )
    manager = _manager(
        upstream,
        downstream,
        resources=_r(cpu=10, heap=100, store=100),
    )
    _ready(manager, upstream.resource_unit_id, downstream.resource_unit_id)
    _assert_object_store_budget_invariants(manager.snapshot())

    upstream_task = manager.try_acquire_task(_task(upstream.resource_unit_id, 0, retained=20))
    request = OutputBlockRequest(
        "q",
        upstream.resource_unit_id,
        upstream_task.lease.lease_id,
        upstream_task.lease.attempt_id,
        "ledger-output",
        15,
    )
    assert manager.note_output_waiting(request) is None
    _assert_object_store_budget_invariants(manager.snapshot())

    selected, output = manager.try_acquire_next_queued_output_block({request.block_id})
    assert selected == request and output.granted
    assert manager.transition_output_block(output.lease.lease_id, "downstream_input")
    assert manager.release_task_lease(
        upstream_task.lease.lease_id,
        attempt_id=upstream_task.lease.attempt_id,
    )
    manager.update_unit_state(upstream.resource_unit_id, runnable=False, completed=True)
    manager.update_allocation(
        _allocation(_r(cpu=10, heap=100, store=100), generation=2),
        admission_open=True,
    )
    _assert_object_store_budget_invariants(manager.snapshot())

    downstream_task = manager.try_acquire_task(_task(downstream.resource_unit_id, 0, retained=10))
    assert downstream_task.granted
    manager.update_allocation(
        _allocation(_r(cpu=10, heap=100, store=40), generation=3),
        admission_open=True,
    )
    _assert_object_store_budget_invariants(manager.snapshot())

    assert manager.release_output_block(output.lease.lease_id)
    assert manager.release_task_lease(
        downstream_task.lease.lease_id,
        attempt_id=downstream_task.lease.attempt_id,
    )
    _assert_object_store_budget_invariants(manager.snapshot())


def test_object_store_ledgers_hold_across_seeded_lifecycle_traces():
    """Exercise valid lifecycle interleavings while checking an independent ledger."""

    for seed in range(12):
        randomizer = random.Random(seed)
        upstream = _unit(f"resource:f:trace-{seed}-upstream", target=1, blocks=1)
        middle = _unit(
            f"resource:f:trace-{seed}-middle",
            inputs=(upstream.resource_unit_id,),
            target=3,
            blocks=1,
        )
        downstream = _unit(
            f"resource:f:trace-{seed}-downstream",
            inputs=(middle.resource_unit_id,),
            target=5,
            blocks=1,
        )
        units = (upstream, middle, downstream)
        manager = _manager(
            *units,
            resources=_r(cpu=100, heap=1_000, store=(17, 31, 64)[seed % 3]),
            reservation_ratio=(0.0, 0.5, 1.0)[seed % 3],
        )
        _ready(
            manager,
            *(unit.resource_unit_id for unit in units),
            consumer_waiting=True,
        )

        active_tasks = []
        waiting_outputs = {}
        active_outputs = {}
        output_states = {}
        completed_unit_ids = set()
        next_identity = 0
        allocation_generation = 1

        for step in range(120):
            action = randomizer.randrange(8)
            live_unit_ids = [unit.resource_unit_id for unit in units if unit.resource_unit_id not in completed_unit_ids]

            if action == 0 and live_unit_ids:
                resource_unit_id = randomizer.choice(live_unit_ids)
                request = _task(
                    resource_unit_id,
                    f"{seed}-{next_identity}",
                    retained=randomizer.randrange(24),
                )
                next_identity += 1
                grant = manager.try_acquire_task(request)
                if grant.granted:
                    active_tasks.append(grant.lease)
            elif action == 1 and active_tasks:
                task = randomizer.choice(active_tasks)
                request = OutputBlockRequest(
                    "q",
                    task.resource_unit_id,
                    task.lease_id,
                    task.attempt_id,
                    f"trace-{seed}-block-{next_identity}",
                    randomizer.randrange(1, 24),
                )
                next_identity += 1
                assert manager.note_output_waiting(request) is None
                waiting_outputs[request.block_id] = request
            elif action == 2 and waiting_outputs:
                request = randomizer.choice(list(waiting_outputs.values()))
                selected, grant = manager.try_acquire_next_queued_output_block({request.block_id})
                assert selected == request
                assert grant is not None
                if grant.granted:
                    waiting_outputs.pop(request.block_id)
                    active_outputs[grant.lease.lease_id] = grant.lease
                    output_states[grant.lease.lease_id] = grant.lease.state
                elif grant.fatal:
                    assert manager.remove_output_waiter(request.block_id)
                    waiting_outputs.pop(request.block_id)
            elif action == 3 and active_outputs:
                transitionable = [
                    lease_id for lease_id, state in output_states.items() if state in {"unit_queue", "downstream_input"}
                ]
                if transitionable:
                    lease_id = randomizer.choice(transitionable)
                    target = "downstream_input" if output_states[lease_id] == "unit_queue" else "external_consumer"
                    assert manager.transition_output_block(lease_id, target)
                    output_states[lease_id] = target
            elif action == 4 and active_tasks:
                task = randomizer.choice(active_tasks)
                assert manager.release_task_lease(task.lease_id, attempt_id=task.attempt_id)
                active_tasks.remove(task)
            elif action == 5 and active_outputs:
                lease_id = randomizer.choice(list(active_outputs))
                assert manager.release_output_block(lease_id)
                active_outputs.pop(lease_id)
                output_states.pop(lease_id)
            elif action == 6:
                allocation_generation += 1
                manager.update_allocation(
                    _allocation(
                        _r(
                            cpu=100,
                            heap=1_000,
                            store=randomizer.choice((0, 7, 17, 31, 64, 101)),
                        ),
                        generation=allocation_generation,
                    ),
                    admission_open=True,
                )
            elif step >= 30:
                completable = [
                    unit.resource_unit_id for unit in units[:-1] if unit.resource_unit_id not in completed_unit_ids
                ]
                if completable:
                    resource_unit_id = randomizer.choice(completable)
                    manager.update_unit_state(resource_unit_id, runnable=False, completed=True)
                    completed_unit_ids.add(resource_unit_id)
                    allocation_generation += 1
                    manager.update_allocation(
                        _allocation(
                            _r(cpu=100, heap=1_000, store=randomizer.choice((7, 31, 64))),
                            generation=allocation_generation,
                        ),
                        admission_open=True,
                    )

            snapshot = manager.snapshot()
            _assert_object_store_budget_invariants(snapshot)

            expected_output_bytes_by_unit = {unit.resource_unit_id: 0 for unit in units}
            for request in waiting_outputs.values():
                expected_output_bytes_by_unit[request.producer_unit_id] += request.size_bytes
            for output in active_outputs.values():
                expected_output_bytes_by_unit[output.producer_unit_id] += output.size_bytes
            for resource_unit_id, expected in expected_output_bytes_by_unit.items():
                assert snapshot["units"][resource_unit_id]["object_store_budget"]["output_usage_bytes"] == expected

            expected_query_usage = sum(task.resources.object_store_bytes for task in active_tasks)
            expected_query_usage += sum(expected_output_bytes_by_unit.values())
            expected_query_usage += sum(
                snapshot["units"][unit.resource_unit_id]["active_task_count"]
                * snapshot["units"][unit.resource_unit_id]["pending_output_estimate_per_active_task_bytes"]
                for unit in units
            )
            assert snapshot["admission"]["object_store"]["query_usage_bytes"] == expected_query_usage

            expected_task_liveness = {task.resource_unit_id: task.lease_id for task in active_tasks if task.liveness}
            expected_output_liveness = {
                output.producer_unit_id: output.lease_id
                for output in active_outputs.values()
                if output.liveness and output_states[output.lease_id] in {"generator_pending", "unit_queue"}
            }
            assert snapshot["liveness"]["active_task_lease_ids_by_unit"] == expected_task_liveness
            assert snapshot["liveness"]["active_output_lease_ids_by_unit"] == expected_output_liveness


def test_task_admission_cannot_consume_the_output_handoff_reservation():
    unit = _unit("resource:f:task-output-split", target=1, blocks=1)
    manager = _manager(
        unit,
        resources=_r(cpu=10, heap=100, store=100),
        reservation_ratio=1.0,
    )
    _ready(manager, unit.resource_unit_id)

    task = manager.try_acquire_task(_task(unit.resource_unit_id, 0, retained=49))
    assert task.granted and not task.liveness
    next_request = _task(unit.resource_unit_id, 1, retained=0)
    assert manager._normal_task_block_reason_locked(next_request)[0] == "unit_soft_object_store_bytes"

    output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            unit.resource_unit_id,
            task.lease.lease_id,
            task.lease.attempt_id,
            "fills-output-reservation",
            50,
        )
    )
    assert output.granted and not output.liveness
    assert manager.snapshot()["usage"]["object_store_bytes"] == 100


def test_output_handoff_can_use_unused_task_reservation_without_shared_credit():
    unit = _unit("resource:f:output-borrows-task", target=1, blocks=1)
    manager = _manager(
        unit,
        resources=_r(cpu=10, heap=100, store=100),
        reservation_ratio=1.0,
    )
    _ready(manager, unit.resource_unit_id)
    task = manager.try_acquire_task(_task(unit.resource_unit_id, 0, retained=0))

    output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            unit.resource_unit_id,
            task.lease.lease_id,
            task.lease.attempt_id,
            "uses-both-protected-halves",
            99,
        )
    )

    assert output.granted and not output.liveness
    budget = manager.snapshot()["units"][unit.resource_unit_id]["object_store_budget"]
    assert budget == {
        "reservation_eligible": True,
        "task_reserved_bytes": 50,
        "output_reserved_bytes": 50,
        "task_internal_usage_bytes": 1,
        "output_usage_bytes": 99,
        "ineligible_usage_bytes": 0,
        "shared_used_bytes": 0,
    }


def test_query_debt_still_preserves_an_independent_units_protected_progress():
    oversized = _unit(
        "resource:f:oversized-branch",
        resources=_r(cpu=1, heap=10, store=101),
        target=0,
        blocks=0,
    )
    protected = _unit("resource:f:protected-branch", target=1, blocks=1)
    manager = _manager(
        oversized,
        protected,
        terminals=(oversized.resource_unit_id, protected.resource_unit_id),
        resources=_r(cpu=10, heap=100, store=100),
    )
    _ready(manager, oversized.resource_unit_id, protected.resource_unit_id)

    debt = manager.try_acquire_task(_task(oversized.resource_unit_id, 0))
    assert debt.granted and debt.liveness
    protected_task = manager.try_acquire_task(_task(protected.resource_unit_id, 0, retained=11))
    assert protected_task.granted and not protected_task.liveness
    protected_output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            protected.resource_unit_id,
            protected_task.lease.lease_id,
            protected_task.lease.attempt_id,
            "parallel-protected-output",
            13,
        )
    )
    assert protected_output.granted and not protected_output.liveness

    shared_request = _task(protected.resource_unit_id, 1, retained=0)
    assert manager._normal_task_block_reason_locked(shared_request)[0] == "query_soft_object_store_bytes"
    assert manager.snapshot()["usage"]["object_store_bytes"] == 126


@pytest.mark.parametrize("retained_output_bytes", [100, 101])
def test_ineligible_usage_at_or_above_the_limit_zeroes_current_reservations(retained_output_bytes):
    completed = _unit(
        "resource:f:completed-native",
        target=1,
        blocks=1,
        backend="ray_worker",
    )
    current = _unit(
        "resource:f:current",
        inputs=(completed.resource_unit_id,),
        target=1,
        blocks=1,
    )
    manager = _manager(
        completed,
        current,
        resources=_r(cpu=10, heap=100, store=100),
    )
    _ready(manager, completed.resource_unit_id, current.resource_unit_id)
    task = manager.try_acquire_task(_task(completed.resource_unit_id, 0, node_id="node-a"))
    manager.finish_task_with_outputs(
        task.lease.lease_id,
        attempt_id=task.lease.attempt_id,
        outputs=(
            OutputBlockRequest(
                "q",
                completed.resource_unit_id,
                task.lease.lease_id,
                task.lease.attempt_id,
                f"completed-{retained_output_bytes}",
                retained_output_bytes,
            ),
        ),
    )
    manager.update_unit_state(completed.resource_unit_id, runnable=False, completed=True)
    manager.update_allocation(
        _allocation(_r(cpu=10, heap=100, store=100), generation=2),
        admission_open=True,
    )

    snapshot = manager.snapshot()
    _assert_object_store_budget_invariants(snapshot)
    assert snapshot["admission"]["object_store"] == {
        "limit_bytes": 100,
        "ineligible_usage_bytes": retained_output_bytes,
        "shared_pool_bytes": 0,
        "shared_used_bytes": 0,
        "shared_remaining_bytes": 0,
        "query_usage_bytes": retained_output_bytes,
    }
    completed_budget = snapshot["units"][completed.resource_unit_id]["object_store_budget"]
    assert completed_budget["reservation_eligible"] is False
    assert completed_budget["ineligible_usage_bytes"] == retained_output_bytes
    assert completed_budget["shared_used_bytes"] == 0
    assert snapshot["units"][current.resource_unit_id]["object_store_budget"]["task_reserved_bytes"] == 0
    assert snapshot["units"][current.resource_unit_id]["object_store_budget"]["output_reserved_bytes"] == 0
    assert manager._normal_task_block_reason_locked(_task(current.resource_unit_id, 0))[0] == (
        "query_soft_object_store_bytes"
    )


def test_completed_producer_can_handoff_an_already_waiting_output_without_a_reservation():
    unit = _unit("resource:f:completed-output-handoff", target=10, blocks=1)
    manager = _manager(unit, resources=_r(cpu=10, heap=100, store=10))
    _ready(manager, unit.resource_unit_id)
    task = manager.try_acquire_task(_task(unit.resource_unit_id, 0, retained=0))
    request = OutputBlockRequest(
        "q",
        unit.resource_unit_id,
        task.lease.lease_id,
        task.lease.attempt_id,
        "completed-output-handoff",
        7,
    )
    assert manager.note_output_waiting(request) is None

    manager.update_unit_state(unit.resource_unit_id, runnable=False, completed=True)
    manager.update_allocation(
        _allocation(_r(cpu=10, heap=100, store=10), generation=2),
        admission_open=True,
    )
    before = manager.snapshot()
    _assert_object_store_budget_invariants(before)
    assert before["units"][unit.resource_unit_id]["object_store_budget"] == {
        "reservation_eligible": False,
        "task_reserved_bytes": 0,
        "output_reserved_bytes": 0,
        "task_internal_usage_bytes": 7,
        "output_usage_bytes": 7,
        "ineligible_usage_bytes": 14,
        "shared_used_bytes": 0,
    }

    selected, output = manager.try_acquire_next_queued_output_block({request.block_id})

    assert selected == request
    assert output.granted and not output.liveness
    after = manager.snapshot()
    _assert_object_store_budget_invariants(after)
    assert after["usage"]["object_store_bytes"] == before["usage"]["object_store_bytes"]
    assert after["units"][unit.resource_unit_id]["object_store_budget"]["ineligible_usage_bytes"] == 14


def test_allocation_shrink_preserves_output_credit_and_growth_restores_shared_credit():
    unit = _unit("resource:f:object-allocation-resize", target=1, blocks=1)
    manager = _manager(unit, resources=_r(cpu=10, heap=100, store=200))
    _ready(manager, unit.resource_unit_id)
    task = manager.try_acquire_task(_task(unit.resource_unit_id, 0, retained=99))
    assert task.granted and not task.liveness

    manager.update_allocation(
        _allocation(_r(cpu=10, heap=100, store=100), generation=2),
        admission_open=True,
    )
    shrunk = manager.snapshot()
    assert shrunk["units"][unit.resource_unit_id]["object_store_budget"] == {
        "reservation_eligible": True,
        "task_reserved_bytes": 25,
        "output_reserved_bytes": 25,
        "task_internal_usage_bytes": 100,
        "output_usage_bytes": 0,
        "ineligible_usage_bytes": 0,
        "shared_used_bytes": 75,
    }
    assert shrunk["admission"]["object_store"]["shared_remaining_bytes"] == 0

    output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            unit.resource_unit_id,
            task.lease.lease_id,
            task.lease.attempt_id,
            "protected-after-shrink",
            25,
        )
    )
    assert output.granted and not output.liveness
    assert manager.snapshot()["usage"]["object_store_bytes"] == 125

    manager.update_allocation(
        _allocation(_r(cpu=10, heap=100, store=300), generation=3),
        admission_open=True,
    )
    assert manager._normal_task_block_reason_locked(_task(unit.resource_unit_id, 1, retained=0))[0] is None


def test_completing_an_idle_unit_reassigns_its_protected_reservation():
    completed = _unit("resource:f:idle-completed", target=1, blocks=1)
    remaining = _unit("resource:f:remaining", target=1, blocks=1)
    manager = _manager(
        completed,
        remaining,
        terminals=(completed.resource_unit_id, remaining.resource_unit_id),
        resources=_r(cpu=10, heap=100, store=100),
    )
    before = manager.snapshot()
    assert before["units"][remaining.resource_unit_id]["object_store_budget"]["task_reserved_bytes"] == 12
    assert before["units"][remaining.resource_unit_id]["object_store_budget"]["output_reserved_bytes"] == 13

    manager.update_unit_state(completed.resource_unit_id, runnable=False, completed=True)
    manager.update_allocation(
        _allocation(_r(cpu=10, heap=100, store=100), generation=2),
        admission_open=True,
    )
    after = manager.snapshot()
    assert after["units"][completed.resource_unit_id]["object_store_budget"]["task_reserved_bytes"] == 0
    assert after["units"][completed.resource_unit_id]["object_store_budget"]["output_reserved_bytes"] == 0
    assert after["units"][remaining.resource_unit_id]["object_store_budget"]["task_reserved_bytes"] == 25
    assert after["units"][remaining.resource_unit_id]["object_store_budget"]["output_reserved_bytes"] == 25


def test_retained_input_uses_exact_dynamic_credit_above_nominal_target():
    unit = _unit(
        "resource:f:decode",
        resources=_r(cpu=1, heap=100, store=30),
        target=0,
        blocks=0,
    )
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
    unit = _unit(
        "resource:f:decode",
        resources=_r(cpu=1, heap=100, store=101),
        target=0,
        blocks=0,
    )
    manager = _manager(unit, resources=_r(cpu=10, heap=1_000, store=100))
    _ready(manager, unit.resource_unit_id)

    request = _task(unit.resource_unit_id, 0, retained=101)
    manager.note_task_waiting(request)
    granted = manager.try_acquire_queued_task(request)

    assert granted.granted
    assert granted.liveness
    assert manager.snapshot()["soft_object_store_debt_bytes"] == 1


def test_object_store_debt_preserves_downstream_task_and_output_reservations():
    upstream = _unit(
        "resource:f:decode",
        resources=_r(cpu=1, heap=10, store=101),
        target=0,
        blocks=0,
    )
    downstream = _unit(
        "resource:f:model",
        inputs=(upstream.resource_unit_id,),
        resources=_r(cpu=1, heap=10, store=1),
        target=1,
        blocks=1,
    )
    sink = _unit(
        "resource:f:sink",
        inputs=(downstream.resource_unit_id,),
        target=0,
        blocks=0,
    )
    manager = _manager(
        upstream,
        downstream,
        sink,
        resources=_r(cpu=10, heap=100, store=100),
    )
    _ready(
        manager,
        upstream.resource_unit_id,
        downstream.resource_unit_id,
        sink.resource_unit_id,
    )

    upstream_task = manager.try_acquire_task(_task(upstream.resource_unit_id, 0))
    assert upstream_task.granted and upstream_task.liveness
    assert manager.snapshot()["soft_object_store_debt_bytes"] == 1

    downstream_task = manager.try_acquire_task(_task(downstream.resource_unit_id, 0, retained=11))
    assert downstream_task.granted and not downstream_task.liveness

    downstream_output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            downstream.resource_unit_id,
            downstream_task.lease.lease_id,
            downstream_task.lease.attempt_id,
            "model-output",
            1,
        )
    )
    assert downstream_output.granted and not downstream_output.liveness

    shared_borrower = _task(upstream.resource_unit_id, 1, retained=1)
    assert manager._normal_task_block_reason_locked(shared_borrower)[0] == "query_soft_object_store_bytes"
    blocked = manager.try_acquire_task(shared_borrower)
    assert not blocked.granted
    assert blocked.blocked_reason == "liveness_task_active"

    snapshot = manager.snapshot()
    assert snapshot["admission"]["object_store"] == {
        "limit_bytes": 100,
        "ineligible_usage_bytes": 0,
        "shared_pool_bytes": 50,
        "shared_used_bytes": 76,
        "shared_remaining_bytes": 0,
        "query_usage_bytes": 114,
    }
    assert snapshot["units"][downstream.resource_unit_id]["object_store_budget"] == {
        "reservation_eligible": True,
        "task_reserved_bytes": 12,
        "output_reserved_bytes": 13,
        "task_internal_usage_bytes": 12,
        "output_usage_bytes": 1,
        "ineligible_usage_bytes": 0,
        "shared_used_bytes": 0,
    }


def test_declared_generator_window_seeds_admission_before_the_first_output():
    unit = _unit(
        "resource:f:decode",
        resources=_r(cpu=1, heap=10),
        target=50,
        blocks=2,
    )
    manager = _manager(
        unit,
        resources=_r(cpu=10, heap=100, store=300),
    )
    _ready(manager, unit.resource_unit_id)

    first = manager.try_acquire_task(_task(unit.resource_unit_id, 0))
    second = manager.try_acquire_task(_task(unit.resource_unit_id, 1))
    third_request = _task(unit.resource_unit_id, 2)

    assert first.granted and not first.liveness
    assert second.granted and not second.liveness
    assert manager._normal_task_block_reason_locked(third_request)[0] == "unit_soft_object_store_bytes"
    third = manager.try_acquire_task(third_request)
    assert not third.granted
    assert third.blocked_reason == "liveness_task_active"

    snapshot = manager.snapshot()
    assert snapshot["usage"]["object_store_bytes"] == 200
    assert snapshot["units"][unit.resource_unit_id]["pending_output_estimate_per_active_task_bytes"] == 100
    assert snapshot["units"][unit.resource_unit_id]["object_store_budget"] == {
        "reservation_eligible": True,
        "task_reserved_bytes": 75,
        "output_reserved_bytes": 75,
        "task_internal_usage_bytes": 200,
        "output_usage_bytes": 0,
        "ineligible_usage_bytes": 0,
        "shared_used_bytes": 125,
    }


def test_completed_zero_output_task_keeps_cold_start_estimate_until_positive_output():
    unit = _unit(
        "resource:f:filter",
        resources=_r(cpu=1, heap=10),
        target=50,
        blocks=2,
    )
    manager = _manager(
        unit,
        resources=_r(cpu=10, heap=100, store=300),
    )
    _ready(manager, unit.resource_unit_id)

    first = manager.try_acquire_task(_task(unit.resource_unit_id, 0))
    assert manager.snapshot()["usage"]["object_store_bytes"] == 100
    assert manager.release_task_lease(
        first.lease.lease_id,
        attempt_id=first.lease.attempt_id,
    )

    snapshot = manager.snapshot()
    assert snapshot["units"][unit.resource_unit_id]["num_tasks_finished"] == 1
    assert snapshot["units"][unit.resource_unit_id]["pending_output_estimate_per_active_task_bytes"] == 100
    second = manager.try_acquire_task(_task(unit.resource_unit_id, 1))
    assert second.granted and not second.liveness
    assert manager.snapshot()["usage"]["object_store_bytes"] == 100

    output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            unit.resource_unit_id,
            second.lease.lease_id,
            second.lease.attempt_id,
            "positive-output",
            10,
        )
    )
    assert output.granted
    snapshot = manager.snapshot()
    assert snapshot["usage"]["object_store_bytes"] == 30
    assert snapshot["units"][unit.resource_unit_id]["pending_output_estimate_per_active_task_bytes"] == 20

    assert manager.release_output_block(output.lease.lease_id)
    assert manager.release_task_lease(
        second.lease.lease_id,
        attempt_id=second.lease.attempt_id,
    )
    learned = manager.snapshot()["units"][unit.resource_unit_id]
    assert learned["num_tasks_finished"] == 2
    assert learned["average_output_blocks_per_finished_task"] == pytest.approx(0.5)
    assert learned["pending_output_estimate_per_active_task_bytes"] == 5


def test_ineligible_output_usage_reduces_current_phase_reservations():
    upstream = _unit("resource:f:completed-producer", target=10, blocks=1)
    downstream = _unit(
        "resource:f:current-producer",
        inputs=(upstream.resource_unit_id,),
        target=10,
        blocks=1,
    )
    manager = _manager(
        upstream,
        downstream,
        resources=_r(cpu=10, heap=100, store=100),
    )
    _ready(manager, upstream.resource_unit_id, downstream.resource_unit_id)

    upstream_task = manager.try_acquire_task(_task(upstream.resource_unit_id, 0))
    upstream_output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            upstream.resource_unit_id,
            upstream_task.lease.lease_id,
            upstream_task.lease.attempt_id,
            "retained-output",
            80,
        )
    )
    assert upstream_output.granted and upstream_output.liveness
    assert manager.transition_output_block(upstream_output.lease.lease_id, "unit_queue")
    assert manager.transition_output_block(upstream_output.lease.lease_id, "downstream_input")
    assert manager.release_task_lease(
        upstream_task.lease.lease_id,
        attempt_id=upstream_task.lease.attempt_id,
    )
    manager.update_unit_state(
        upstream.resource_unit_id,
        runnable=False,
        completed=True,
    )
    manager.update_allocation(
        _allocation(_r(cpu=10, heap=100, store=100), generation=2),
        admission_open=True,
    )

    budget = manager.snapshot()["admission"]["object_store"]
    assert budget == {
        "limit_bytes": 100,
        "ineligible_usage_bytes": 80,
        "shared_pool_bytes": 10,
        "shared_used_bytes": 0,
        "shared_remaining_bytes": 10,
        "query_usage_bytes": 80,
    }
    downstream_budget = manager.snapshot()["units"][downstream.resource_unit_id]["object_store_budget"]
    assert downstream_budget["task_reserved_bytes"] == 5
    assert downstream_budget["output_reserved_bytes"] == 5

    downstream_task = manager.try_acquire_task(_task(downstream.resource_unit_id, 0))
    assert downstream_task.granted and not downstream_task.liveness
    assert manager.snapshot()["usage"]["object_store_bytes"] == 90


def test_global_idle_liveness_is_bounded_to_one_task_across_units():
    first = _unit(
        "resource:f:first",
        resources=_r(cpu=1, heap=100),
        target=0,
        blocks=0,
    )
    second = _unit(
        "resource:f:second",
        resources=_r(cpu=1, heap=100),
        target=0,
        blocks=0,
    )
    manager = _manager(
        first,
        second,
        resources=_r(),
        terminals=(first.resource_unit_id, second.resource_unit_id),
    )
    _ready(manager, first.resource_unit_id, second.resource_unit_id)

    escaped = manager.try_acquire_task(_task(first.resource_unit_id, 0))
    blocked = manager.try_acquire_task(_task(second.resource_unit_id, 0))

    assert escaped.granted and escaped.liveness
    assert not blocked.granted and blocked.blocked_reason == "liveness_task_active"

    manager.release_task_lease(
        escaped.lease.lease_id,
        attempt_id=escaped.lease.attempt_id,
    )
    next_escape = manager.try_acquire_task(_task(second.resource_unit_id, 1))
    assert next_escape.granted and next_escape.liveness


def test_task_liveness_cannot_move_back_upstream():
    upstream = _unit("resource:f:liveness-upstream", target=0, blocks=0)
    downstream = _unit(
        "resource:f:liveness-downstream",
        inputs=(upstream.resource_unit_id,),
        target=0,
        blocks=0,
    )
    manager = _manager(upstream, downstream, resources=_r())
    _ready(manager, upstream.resource_unit_id, downstream.resource_unit_id)

    downstream_escape = manager.try_acquire_task(_task(downstream.resource_unit_id, 0))
    upstream_request = manager.try_acquire_task(_task(upstream.resource_unit_id, 0))

    assert downstream_escape.granted and downstream_escape.liveness
    assert not upstream_request.granted
    assert upstream_request.blocked_reason == "liveness_task_active"


def test_recovery_descriptor_can_restart_root_behind_live_downstream_lease():
    root = _unit(
        "resource:f:retry-root",
        target=10,
        blocks=1,
        backend="ray_worker",
    )
    downstream = _unit(
        "resource:f:retry-downstream",
        inputs=(root.resource_unit_id,),
        target=10,
        blocks=2,
        backend="ray_worker",
    )
    unrelated = _unit(
        "resource:f:unrelated-live-task",
        target=0,
        blocks=0,
        backend="ray_worker",
    )
    manager = _manager(
        root,
        downstream,
        unrelated,
        terminals=(downstream.resource_unit_id, unrelated.resource_unit_id),
        resources=_r(store=100),
    )
    _ready(
        manager,
        root.resource_unit_id,
        downstream.resource_unit_id,
        unrelated.resource_unit_id,
    )

    downstream_task = manager.try_acquire_task(
        _task(
            downstream.resource_unit_id,
            0,
            retained=0,
            node_id="node-b",
        )
    )
    unrelated_task = manager.try_acquire_task(
        _task(
            unrelated.resource_unit_id,
            0,
            retained=0,
            node_id="node-c",
        )
    )
    assert downstream_task.granted and not downstream_task.liveness
    assert unrelated_task.granted and not unrelated_task.liveness
    manager.update_allocation(
        _allocation(_r(store=20), generation=2),
        admission_open=True,
    )
    retry = _task(
        root.resource_unit_id,
        0,
        attempt="1",
        retained=0,
        node_id="node-a",
    )

    ordinary = manager.try_acquire_task_descriptor(retry)
    recovery = manager.try_acquire_task_descriptor(retry, recovery=True)

    assert not ordinary.granted
    assert ordinary.blocked_reason == "liveness_task_active"
    assert recovery.granted and recovery.liveness


def test_recovery_descriptor_cannot_open_an_unrelated_root():
    required_root = _unit(
        "resource:f:required-root",
        target=10,
        blocks=1,
        backend="ray_worker",
    )
    downstream = _unit(
        "resource:f:active-downstream",
        inputs=(required_root.resource_unit_id,),
        target=10,
        blocks=2,
        backend="ray_worker",
    )
    unrelated_root = _unit(
        "resource:f:unrelated-root",
        target=10,
        blocks=1,
        backend="ray_worker",
    )
    manager = _manager(
        required_root,
        downstream,
        unrelated_root,
        terminals=(downstream.resource_unit_id, unrelated_root.resource_unit_id),
        resources=_r(),
    )
    _ready(
        manager,
        required_root.resource_unit_id,
        downstream.resource_unit_id,
        unrelated_root.resource_unit_id,
    )

    downstream_task = manager.try_acquire_task(
        _task(
            downstream.resource_unit_id,
            0,
            retained=0,
            node_id="node-b",
        )
    )
    assert downstream_task.granted and downstream_task.liveness

    recovery = manager.try_acquire_task_descriptor(
        _task(
            unrelated_root.resource_unit_id,
            0,
            attempt="1",
            retained=0,
            node_id="node-a",
        ),
        recovery=True,
    )

    assert not recovery.granted
    assert recovery.blocked_reason == "liveness_task_active"


def test_recovery_descriptor_cannot_move_a_non_root_back_upstream():
    root = _unit(
        "resource:f:root",
        target=10,
        blocks=1,
        backend="ray_worker",
    )
    middle = _unit(
        "resource:f:middle",
        inputs=(root.resource_unit_id,),
        target=10,
        blocks=1,
        backend="ray_worker",
    )
    downstream = _unit(
        "resource:f:downstream",
        inputs=(middle.resource_unit_id,),
        target=10,
        blocks=1,
        backend="ray_worker",
    )
    manager = _manager(root, middle, downstream, resources=_r())
    _ready(
        manager,
        root.resource_unit_id,
        middle.resource_unit_id,
        downstream.resource_unit_id,
    )

    downstream_task = manager.try_acquire_task(_task(downstream.resource_unit_id, 0, retained=0, node_id="node-b"))
    assert downstream_task.granted and downstream_task.liveness

    recovery = manager.try_acquire_task_descriptor(
        _task(
            middle.resource_unit_id,
            0,
            attempt="1",
            retained=0,
            node_id="node-a",
        ),
        recovery=True,
    )

    assert not recovery.granted
    assert recovery.blocked_reason == "liveness_task_active"


def test_recovery_descriptor_can_extend_an_active_liveness_chain_back_to_root():
    root = _unit(
        "resource:f:retry-root",
        target=10,
        blocks=1,
        backend="ray_worker",
    )
    downstream = _unit(
        "resource:f:active-downstream",
        inputs=(root.resource_unit_id,),
        target=10,
        blocks=2,
        backend="ray_worker",
    )
    manager = _manager(root, downstream, resources=_r())
    _ready(manager, root.resource_unit_id, downstream.resource_unit_id)

    downstream_task = manager.try_acquire_task(
        _task(
            downstream.resource_unit_id,
            0,
            retained=0,
            node_id="node-b",
        )
    )
    assert downstream_task.granted and downstream_task.liveness

    recovery = manager.try_acquire_task_descriptor(
        _task(
            root.resource_unit_id,
            0,
            attempt="1",
            retained=0,
            node_id="node-a",
        ),
        recovery=True,
    )

    assert recovery.granted and recovery.liveness
    active = manager.snapshot()["liveness"]["active_task_lease_ids_by_unit"]
    assert active == {
        downstream.resource_unit_id: downstream_task.lease.lease_id,
        root.resource_unit_id: recovery.lease.lease_id,
    }


def test_recovery_descriptor_keeps_retiring_liveness_tokens_in_the_chain():
    left_root = _unit(
        "resource:f:left-root",
        target=10,
        blocks=1,
        backend="ray_worker",
    )
    right_root = _unit(
        "resource:f:right-root",
        target=10,
        blocks=1,
        backend="ray_worker",
    )
    join = _unit(
        "resource:f:join",
        inputs=(left_root.resource_unit_id, right_root.resource_unit_id),
        target=10,
        blocks=1,
        backend="ray_worker",
    )
    manager = _manager(left_root, right_root, join, resources=_r())
    _ready(
        manager,
        left_root.resource_unit_id,
        right_root.resource_unit_id,
        join.resource_unit_id,
    )

    left_task = manager.try_acquire_task(_task(left_root.resource_unit_id, 0, retained=0, node_id="node-a"))
    join_task = manager.try_acquire_task(_task(join.resource_unit_id, 0, retained=0, node_id="node-b"))
    assert left_task.granted and left_task.liveness
    assert join_task.granted and join_task.liveness

    manager.update_unit_state(
        left_root.resource_unit_id,
        runnable=False,
        completed=True,
    )
    manager.update_allocation(
        _allocation(_r(), generation=2),
        admission_open=True,
    )
    assert left_root.resource_unit_id not in manager.current_eligible_resource_unit_ids()

    recovery = manager.try_acquire_task_descriptor(
        _task(
            right_root.resource_unit_id,
            0,
            attempt="1",
            retained=0,
            node_id="node-c",
        ),
        recovery=True,
    )

    assert not recovery.granted
    assert recovery.blocked_reason == "liveness_task_active"


def test_task_liveness_can_cross_a_diamond_only_after_convergence():
    source = _unit("resource:f:diamond-source", target=0, blocks=0)
    left = _unit(
        "resource:f:diamond-left",
        inputs=(source.resource_unit_id,),
        target=0,
        blocks=0,
    )
    right = _unit(
        "resource:f:diamond-right",
        inputs=(source.resource_unit_id,),
        target=0,
        blocks=0,
    )
    join = _unit(
        "resource:f:diamond-join",
        inputs=(left.resource_unit_id, right.resource_unit_id),
        target=0,
        blocks=0,
    )
    manager = _manager(
        source,
        left,
        right,
        join,
        resources=_r(cpu=2, heap=20),
    )
    _ready(
        manager,
        source.resource_unit_id,
        left.resource_unit_id,
        right.resource_unit_id,
        join.resource_unit_id,
    )

    source_task = manager.try_acquire_task(_task(source.resource_unit_id, 0))
    left_escape = manager.try_acquire_task(_task(left.resource_unit_id, 0))
    right_parallel = manager.try_acquire_task(_task(right.resource_unit_id, 0))
    join_escape = manager.try_acquire_task(_task(join.resource_unit_id, 0))

    assert source_task.granted and not source_task.liveness
    assert left_escape.granted and left_escape.liveness
    assert not right_parallel.granted and right_parallel.blocked_reason == "liveness_task_active"
    assert join_escape.granted and join_escape.liveness
    assert manager.snapshot()["liveness"]["active_task_lease_ids_by_unit"] == {
        left.resource_unit_id: left_escape.lease.lease_id,
        join.resource_unit_id: join_escape.lease.lease_id,
    }


def test_task_liveness_advances_a_bounded_downstream_chain():
    upstream = _unit("resource:f:decode", target=10, blocks=1)
    middle = _unit(
        "resource:f:first-model",
        inputs=(upstream.resource_unit_id,),
        target=10,
        blocks=1,
    )
    downstream = _unit(
        "resource:f:second-model",
        inputs=(middle.resource_unit_id,),
        target=10,
        blocks=1,
    )
    sink = _unit(
        "resource:f:sink",
        inputs=(downstream.resource_unit_id,),
        target=0,
        blocks=0,
    )
    manager = _manager(
        upstream,
        middle,
        downstream,
        sink,
        resources=_r(cpu=10, heap=1_000, store=100),
    )
    _ready(manager, upstream.resource_unit_id)
    upstream_task = manager.try_acquire_task(_task(upstream.resource_unit_id, 0))
    _ready(
        manager,
        middle.resource_unit_id,
        downstream.resource_unit_id,
        sink.resource_unit_id,
    )
    middle_task = manager.try_acquire_task(_task(middle.resource_unit_id, 0, retained=101))
    downstream_task = manager.try_acquire_task(_task(downstream.resource_unit_id, 0, retained=101))

    assert upstream_task.granted and not upstream_task.liveness
    assert middle_task.granted and middle_task.liveness
    assert downstream_task.granted and downstream_task.liveness
    assert manager.snapshot()["liveness"]["active_task_lease_ids_by_unit"] == {
        middle.resource_unit_id: middle_task.lease.lease_id,
        downstream.resource_unit_id: downstream_task.lease.lease_id,
    }

    same_unit = manager.try_acquire_task(_task(downstream.resource_unit_id, 1, retained=101))
    assert not same_unit.granted
    assert same_unit.blocked_reason == "liveness_task_active"


def test_task_liveness_does_not_open_a_parallel_downstream_branch():
    upstream = _unit("resource:f:source", target=10, blocks=1)
    left = _unit(
        "resource:f:left",
        inputs=(upstream.resource_unit_id,),
        target=10,
        blocks=1,
    )
    right = _unit(
        "resource:f:right",
        inputs=(upstream.resource_unit_id,),
        target=10,
        blocks=1,
    )
    manager = _manager(
        upstream,
        left,
        right,
        resources=_r(cpu=10, heap=1_000, store=100),
        terminals=(left.resource_unit_id, right.resource_unit_id),
    )
    _ready(manager, upstream.resource_unit_id)
    upstream_task = manager.try_acquire_task(_task(upstream.resource_unit_id, 0))
    _ready(manager, left.resource_unit_id, right.resource_unit_id)
    left_task = manager.try_acquire_task(_task(left.resource_unit_id, 0, retained=101))
    right_task = manager.try_acquire_task(_task(right.resource_unit_id, 0, retained=101))

    assert upstream_task.granted and not upstream_task.liveness
    assert left_task.granted and left_task.liveness
    assert not right_task.granted
    assert right_task.blocked_reason == "liveness_task_active"


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
    assert manager.snapshot()["usage"]["object_store_bytes"] == 110

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
    assert active.granted and prefetched.granted
    # Only the active call can populate this actor's generator window.
    assert manager.snapshot()["usage"]["object_store_bytes"] == 100
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

    assert output.granted
    # 10 exact output bytes + 20 estimated bytes for only the active call.
    assert manager.snapshot()["usage"]["object_store_bytes"] == 30
    assert manager.release_task_lease(
        active.lease.lease_id,
        attempt_id=active.lease.attempt_id,
    )
    # The promoted call now uses the learned 10-byte/one-output estimate, while
    # the first call's ObjectRef remains exact.
    assert manager.snapshot()["usage"]["object_store_bytes"] == 20


def test_prefetched_actor_admission_does_not_reserve_a_second_generator_window():
    actor = _unit(
        "resource:f:actor-prefetch-admission",
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
        resources=_r(cpu=1, heap=10, store=100),
    )
    _ready(manager, actor.resource_unit_id)

    active = manager.try_acquire_task(_task(actor.resource_unit_id, 0, retained=0))
    prefetched = manager.try_acquire_task(_task(actor.resource_unit_id, 1, retained=0))

    assert active.granted and active.liveness
    assert prefetched.granted and not prefetched.liveness
    assert manager.snapshot()["usage"]["object_store_bytes"] == 100


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


def test_asymmetric_frontier_lifts_object_store_cap_for_materialized_branch_only():
    build_source = _unit(
        "resource:f:broadcast-build-source",
        resources=_r(cpu=1),
        target=50,
        blocks=2,
    )
    build = _unit(
        "resource:f:broadcast-build",
        inputs=(build_source.resource_unit_id,),
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
        build_source,
        build,
        probe,
        materializer,
        downstream,
        resources=_r(cpu=3, store=100),
        barriers=(barrier,),
    )
    _ready(
        manager,
        build_source.resource_unit_id,
        build.resource_unit_id,
        probe.resource_unit_id,
        materializer.resource_unit_id,
        downstream.resource_unit_id,
    )

    before = manager.snapshot()
    assert before["execution_phase"]["object_store_unlimited_unit_ids"] == [
        build_source.resource_unit_id,
        build.resource_unit_id,
        materializer.resource_unit_id,
    ]
    assert before["units"][build_source.resource_unit_id]["object_store_backpressure_disabled"] is True
    assert before["units"][build.resource_unit_id]["object_store_backpressure_disabled"] is True
    assert before["units"][probe.resource_unit_id]["object_store_backpressure_disabled"] is False

    materializing_task = manager.try_acquire_task(_task(materializer.resource_unit_id, 0, node_id="node-a"))
    build_task = manager.try_acquire_task(_task(build.resource_unit_id, 0))
    build_output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            build.resource_unit_id,
            build_task.lease.lease_id,
            build_task.lease.attempt_id,
            "accumulated-build-block",
            100,
        )
    )
    upstream_continuation = manager.try_acquire_task(_task(build_source.resource_unit_id, 0, retained=1))
    deferred_probe = manager.try_acquire_task(_task(probe.resource_unit_id, 0, retained=1))

    assert materializing_task.granted and build_task.granted and build_output.granted
    assert upstream_continuation.granted and not upstream_continuation.liveness
    assert not deferred_probe.granted
    assert deferred_probe.blocked_reason == "materialization_barrier_pending"

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


def test_multiple_waiting_outputs_are_counted_once_when_each_becomes_a_lease():
    unit = _unit("resource:f:multiple-waiters", target=10, blocks=1)
    manager = _manager(unit, resources=_r(cpu=10, heap=100, store=1_000))
    _ready(manager, unit.resource_unit_id)
    first_task = manager.try_acquire_task(_task(unit.resource_unit_id, 0))
    second_task = manager.try_acquire_task(_task(unit.resource_unit_id, 1))
    first_request = OutputBlockRequest(
        "q",
        unit.resource_unit_id,
        first_task.lease.lease_id,
        first_task.lease.attempt_id,
        "first-waiting-output",
        7,
    )
    second_request = OutputBlockRequest(
        "q",
        unit.resource_unit_id,
        second_task.lease.lease_id,
        second_task.lease.attempt_id,
        "second-waiting-output",
        11,
    )
    assert manager.note_output_waiting(first_request) is None
    assert manager.note_output_waiting(second_request) is None
    assert manager.snapshot()["usage"]["object_store_bytes"] == 36

    selected, first_grant = manager.try_acquire_next_queued_output_block({first_request.block_id})
    assert selected == first_request and first_grant.granted
    assert manager.snapshot()["usage"]["object_store_bytes"] == 36
    selected, second_grant = manager.try_acquire_next_queued_output_block({second_request.block_id})
    assert selected == second_request and second_grant.granted
    assert manager.snapshot()["usage"]["object_store_bytes"] == 36

    assert manager.release_task_lease(
        first_task.lease.lease_id,
        attempt_id=first_task.lease.attempt_id,
    )
    assert manager.release_task_lease(
        second_task.lease.lease_id,
        attempt_id=second_task.lease.attempt_id,
    )
    assert manager.snapshot()["usage"]["object_store_bytes"] == 18
    assert manager.release_output_block(first_grant.lease.lease_id)
    assert manager.release_output_block(second_grant.lease.lease_id)
    assert manager.snapshot()["usage"]["object_store_bytes"] == 0


def test_removing_a_waiting_output_drops_exact_bytes_but_keeps_the_learned_window():
    unit = _unit("resource:f:removed-waiter", target=10, blocks=1)
    manager = _manager(unit, resources=_r(cpu=10, heap=100, store=100))
    _ready(manager, unit.resource_unit_id)
    task = manager.try_acquire_task(_task(unit.resource_unit_id, 0))
    request = OutputBlockRequest(
        "q",
        unit.resource_unit_id,
        task.lease.lease_id,
        task.lease.attempt_id,
        "removed-waiter",
        7,
    )
    assert manager.note_output_waiting(request) is None
    assert manager.snapshot()["usage"]["object_store_bytes"] == 14

    assert manager.remove_output_waiter(request.block_id)
    snapshot = manager.snapshot()
    assert snapshot["usage"]["object_store_bytes"] == 7
    assert snapshot["units"][unit.resource_unit_id]["num_task_outputs_generated"] == 1
    assert snapshot["units"][unit.resource_unit_id]["pending_output_estimate_per_active_task_bytes"] == 7


def test_waiting_output_stays_charged_during_task_completion_cleanup_race():
    unit = _unit("resource:f:cleanup-race", target=10, blocks=1)
    manager = _manager(unit, resources=_r(cpu=10, heap=100, store=100))
    _ready(manager, unit.resource_unit_id)
    task = manager.try_acquire_task(_task(unit.resource_unit_id, 0))
    request = OutputBlockRequest(
        "q",
        unit.resource_unit_id,
        task.lease.lease_id,
        task.lease.attempt_id,
        "waiting-during-task-cleanup",
        7,
    )
    assert manager.note_output_waiting(request) is None

    assert manager.release_task_lease(
        task.lease.lease_id,
        attempt_id=task.lease.attempt_id,
    )
    snapshot = manager.snapshot()
    assert snapshot["usage"]["object_store_bytes"] == 7
    assert snapshot["units"][unit.resource_unit_id]["object_store_budget"] == {
        "reservation_eligible": True,
        "task_reserved_bytes": 25,
        "output_reserved_bytes": 25,
        "task_internal_usage_bytes": 0,
        "output_usage_bytes": 7,
        "ineligible_usage_bytes": 0,
        "shared_used_bytes": 0,
    }

    selected, denied = manager.try_acquire_next_queued_output_block({request.block_id})
    assert selected == request
    assert denied is not None and denied.fatal
    assert denied.blocked_reason == "task_lease_not_active"
    assert manager.remove_output_waiter(request.block_id)
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
    task = manager.try_acquire_task(_task(unit.resource_unit_id, 0, node_id="node-a"))

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
    task = manager.try_acquire_task(_task(unit.resource_unit_id, 0, node_id="node-a"))

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


def test_task_and_output_liveness_tokens_have_independent_lifetimes():
    unit = _unit("resource:f:liveness-lifetimes", target=1, blocks=1)
    manager = _manager(unit, resources=_r(cpu=10, heap=100, store=1))
    _ready(manager, unit.resource_unit_id, consumer_waiting=True)

    task = manager.try_acquire_task(_task(unit.resource_unit_id, 0, retained=2))
    assert task.granted and task.liveness
    output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            unit.resource_unit_id,
            task.lease.lease_id,
            task.lease.attempt_id,
            "liveness-lifetime-output",
            2,
        )
    )
    assert output.granted and output.liveness
    assert manager.snapshot()["liveness"] == {
        "active_task_lease_ids_by_unit": {unit.resource_unit_id: task.lease.lease_id},
        "active_output_lease_ids_by_unit": {unit.resource_unit_id: output.lease.lease_id},
        "task_grants_total": 1,
        "output_grants_total": 1,
    }

    assert manager.transition_output_block(output.lease.lease_id, "unit_queue")
    assert manager.snapshot()["liveness"]["active_output_lease_ids_by_unit"] == {
        unit.resource_unit_id: output.lease.lease_id
    }
    assert manager.transition_output_block(output.lease.lease_id, "downstream_input")
    handed_off = manager.snapshot()["liveness"]
    assert handed_off["active_output_lease_ids_by_unit"] == {}
    assert handed_off["active_task_lease_ids_by_unit"] == {unit.resource_unit_id: task.lease.lease_id}

    assert manager.release_task_lease(task.lease.lease_id, attempt_id=task.lease.attempt_id)
    assert manager.snapshot()["liveness"]["active_task_lease_ids_by_unit"] == {}
    assert manager.release_output_block(output.lease.lease_id)
    assert manager.snapshot()["usage"]["object_store_bytes"] == 0


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


def test_output_liveness_advances_a_bounded_downstream_chain():
    upstream = _unit("resource:f:decode", target=10, blocks=1)
    bridge = _unit(
        "resource:f:bridge",
        inputs=(upstream.resource_unit_id,),
        target=0,
        blocks=0,
    )
    downstream = _unit(
        "resource:f:model",
        inputs=(bridge.resource_unit_id,),
        target=10,
        blocks=1,
    )
    sink = _unit(
        "resource:f:sink",
        inputs=(downstream.resource_unit_id,),
        target=0,
        blocks=0,
    )
    manager = _manager(
        upstream,
        bridge,
        downstream,
        sink,
        resources=_r(cpu=10, heap=1_000, store=10),
    )
    _ready(
        manager,
        upstream.resource_unit_id,
        bridge.resource_unit_id,
        downstream.resource_unit_id,
        sink.resource_unit_id,
    )
    upstream_task = manager.try_acquire_task(_task(upstream.resource_unit_id, 0))
    upstream_output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            upstream.resource_unit_id,
            upstream_task.lease.lease_id,
            upstream_task.lease.attempt_id,
            "decode-output",
            11,
        )
    )
    assert upstream_output.granted and upstream_output.liveness
    assert manager.transition_output_block(upstream_output.lease.lease_id, "unit_queue")

    downstream_task = manager.try_acquire_task(_task(downstream.resource_unit_id, 0, retained=0))
    assert downstream_task.granted
    model_output = OutputBlockRequest(
        "q",
        downstream.resource_unit_id,
        downstream_task.lease.lease_id,
        downstream_task.lease.attempt_id,
        "model-output",
        2,
    )
    assert manager.note_output_waiting(model_output) is None

    selected, grant = manager.try_acquire_next_queued_output_block({model_output.block_id})

    assert selected == model_output
    assert grant.granted and grant.liveness
    assert manager.snapshot()["liveness"]["active_output_lease_ids_by_unit"] == {
        upstream.resource_unit_id: upstream_output.lease.lease_id,
        downstream.resource_unit_id: grant.lease.lease_id,
    }

    next_model_output = OutputBlockRequest(
        "q",
        downstream.resource_unit_id,
        downstream_task.lease.lease_id,
        downstream_task.lease.attempt_id,
        "next-model-output",
        1,
    )
    assert manager.note_output_waiting(next_model_output) is None
    blocked_request, blocked = manager.try_acquire_next_queued_output_block({next_model_output.block_id})
    assert blocked_request == next_model_output
    assert not blocked.granted and blocked.blocked_reason == "liveness_output_active"

    assert manager.transition_output_block(grant.lease.lease_id, "downstream_input")
    selected, next_grant = manager.try_acquire_next_queued_output_block({next_model_output.block_id})
    assert selected == next_model_output
    assert next_grant.granted and next_grant.liveness


def test_output_liveness_cannot_move_back_upstream():
    upstream = _unit("resource:f:output-liveness-upstream", target=10, blocks=1)
    downstream = _unit(
        "resource:f:output-liveness-downstream",
        inputs=(upstream.resource_unit_id,),
        target=10,
        blocks=1,
    )
    manager = _manager(
        upstream,
        downstream,
        resources=_r(cpu=10, heap=1_000, store=100),
    )
    _ready(
        manager,
        upstream.resource_unit_id,
        downstream.resource_unit_id,
        consumer_waiting=True,
    )
    upstream_task = manager.try_acquire_task(_task(upstream.resource_unit_id, 0, retained=0))
    downstream_task = manager.try_acquire_task(_task(downstream.resource_unit_id, 0, retained=0))
    manager.update_allocation(
        _allocation(_r(cpu=10, heap=1_000, store=10), generation=2),
        admission_open=True,
    )

    downstream_output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            downstream.resource_unit_id,
            downstream_task.lease.lease_id,
            downstream_task.lease.attempt_id,
            "downstream-liveness-first",
            11,
        )
    )
    upstream_output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            upstream.resource_unit_id,
            upstream_task.lease.lease_id,
            upstream_task.lease.attempt_id,
            "upstream-cannot-follow",
            11,
        )
    )

    assert downstream_output.granted and downstream_output.liveness
    assert not upstream_output.granted
    assert upstream_output.blocked_reason == "liveness_output_active"
    assert manager.snapshot()["liveness"]["active_output_lease_ids_by_unit"] == {
        downstream.resource_unit_id: downstream_output.lease.lease_id
    }


def test_output_liveness_does_not_open_a_parallel_branch():
    left = _unit("resource:f:left", target=10, blocks=1)
    left_sink = _unit(
        "resource:f:left-sink",
        inputs=(left.resource_unit_id,),
        target=0,
        blocks=0,
    )
    right = _unit("resource:f:right", target=10, blocks=1)
    right_sink = _unit(
        "resource:f:right-sink",
        inputs=(right.resource_unit_id,),
        target=0,
        blocks=0,
    )
    manager = _manager(
        left,
        left_sink,
        right,
        right_sink,
        terminals=(left_sink.resource_unit_id, right_sink.resource_unit_id),
        resources=_r(cpu=10, heap=1_000, store=40),
    )
    _ready(
        manager,
        left.resource_unit_id,
        left_sink.resource_unit_id,
        right.resource_unit_id,
        right_sink.resource_unit_id,
    )
    left_task = manager.try_acquire_task(_task(left.resource_unit_id, 0))
    right_task = manager.try_acquire_task(_task(right.resource_unit_id, 0))
    left_output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            left.resource_unit_id,
            left_task.lease.lease_id,
            left_task.lease.attempt_id,
            "left-output",
            41,
        )
    )
    assert left_output.granted and left_output.liveness
    assert manager.transition_output_block(left_output.lease.lease_id, "unit_queue")

    right_output = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            right.resource_unit_id,
            right_task.lease.lease_id,
            right_task.lease.attempt_id,
            "right-output",
            41,
        )
    )

    assert not right_output.granted
    assert right_output.blocked_reason == "liveness_output_active"


def test_queued_output_liveness_skips_higher_ranked_nonstarving_branch():
    starving_producer = _unit("resource:f:a-producer", target=10, blocks=1)
    starving_consumer = _unit(
        "resource:f:a-consumer",
        inputs=(starving_producer.resource_unit_id,),
        target=0,
        blocks=0,
    )
    busy_producer = _unit("resource:f:z-producer", target=10, blocks=1)
    busy_consumer = _unit(
        "resource:f:z-consumer",
        inputs=(busy_producer.resource_unit_id,),
        target=0,
        blocks=0,
    )
    manager = _manager(
        starving_producer,
        starving_consumer,
        busy_producer,
        busy_consumer,
        terminals=(starving_consumer.resource_unit_id, busy_consumer.resource_unit_id),
        resources=_r(cpu=100, heap=1_000, store=40),
    )
    _ready(
        manager,
        starving_producer.resource_unit_id,
        starving_consumer.resource_unit_id,
        busy_producer.resource_unit_id,
        busy_consumer.resource_unit_id,
    )
    starving_task = manager.try_acquire_task(_task(starving_producer.resource_unit_id, 0))
    busy_task = manager.try_acquire_task(_task(busy_producer.resource_unit_id, 0))
    busy_consumer_task = manager.try_acquire_task(_task(busy_consumer.resource_unit_id, 0, retained=0))
    assert starving_task.granted and busy_task.granted and busy_consumer_task.granted

    starving_request = OutputBlockRequest(
        "q",
        starving_producer.resource_unit_id,
        starving_task.lease.lease_id,
        starving_task.lease.attempt_id,
        "starving-output",
        41,
    )
    busy_request = OutputBlockRequest(
        "q",
        busy_producer.resource_unit_id,
        busy_task.lease.lease_id,
        busy_task.lease.attempt_id,
        "busy-output",
        41,
    )
    assert manager.note_output_waiting(starving_request) is None
    assert manager.note_output_waiting(busy_request) is None
    assert (
        manager._reverse_topological_rank[busy_producer.resource_unit_id]
        < manager._reverse_topological_rank[starving_producer.resource_unit_id]
    )

    selected, grant = manager.try_acquire_next_queued_output_block({starving_request.block_id, busy_request.block_id})

    assert selected == starving_request
    assert grant.granted and grant.liveness


def test_output_liveness_ignores_consumers_behind_pending_barrier():
    producer = _unit("resource:f:producer", target=10, blocks=1)
    busy_consumer = _unit(
        "resource:f:busy-consumer",
        inputs=(producer.resource_unit_id,),
        target=0,
        blocks=0,
    )
    materialized_input = _unit(
        "resource:f:materialized-input",
        target=0,
        blocks=0,
    )
    materializer = _unit(
        "resource:f:materializer",
        inputs=(materialized_input.resource_unit_id,),
        resources=_r(),
        target=0,
        blocks=0,
        backend="ray_worker",
        unit_kind="native_fragment",
    )
    deferred_consumer = _unit(
        "resource:f:deferred-consumer",
        inputs=(producer.resource_unit_id, materializer.resource_unit_id),
        target=0,
        blocks=0,
    )
    manager = _manager(
        producer,
        busy_consumer,
        materialized_input,
        materializer,
        deferred_consumer,
        terminals=(busy_consumer.resource_unit_id, deferred_consumer.resource_unit_id),
        resources=_r(cpu=100, heap=1_000, store=10),
        barriers=(_barrier("materializer", materializer),),
    )
    _ready(
        manager,
        producer.resource_unit_id,
        busy_consumer.resource_unit_id,
        materialized_input.resource_unit_id,
        materializer.resource_unit_id,
        deferred_consumer.resource_unit_id,
    )
    producer_task = manager.try_acquire_task(_task(producer.resource_unit_id, 0))
    busy_consumer_task = manager.try_acquire_task(_task(busy_consumer.resource_unit_id, 0, retained=0))

    assert producer_task.granted and busy_consumer_task.granted
    assert deferred_consumer.resource_unit_id not in manager.current_eligible_resource_unit_ids()
    grant = manager.try_acquire_output_block(
        OutputBlockRequest(
            "q",
            producer.resource_unit_id,
            producer_task.lease.lease_id,
            producer_task.lease.attempt_id,
            "over-budget-output",
            11,
        )
    )

    assert not grant.granted
    assert grant.blocked_reason == "output_liveness_not_needed"


def test_object_store_soft_debt_does_not_block_zero_input_downstream_compute():
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


def test_native_task_and_materialized_output_preserve_the_actual_runtime_node():
    unit = _unit(
        "resource:f:per-node",
        resources=_r(cpu=1, heap=10),
        target=10,
        blocks=2,
        concurrency=3,
        backend="ray_worker",
    )
    manager = _manager(
        unit,
        resources=_r(cpu=2, heap=20, store=60),
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
    assert manager.snapshot()["usage"]["object_store_bytes"] == 10

    granted = manager.try_acquire_task(_task(unit.resource_unit_id, 3, node_id=output.node_id))
    assert granted.granted
    assert granted.lease.node_id == output.node_id
    assert manager.snapshot()["ray_core_owns_placement"] is True
