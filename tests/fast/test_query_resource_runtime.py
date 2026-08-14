# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

from vane.runners.ray.query_resource_graph import (
    MaterializationBarrierSpec,
    QueryAllocation,
    QueryResourceGraph,
    ResourceUnitSpec,
    ResourceVector,
)
from vane.runners.ray.query_resource_manager import TaskRequest
from vane.runners.ray.query_resource_runtime import (
    clear_query_resource_managers,
    get_query_resource_manager,
    mark_materialization_barrier_completed,
    query_resource_manager_snapshot,
    register_query_resource_graph,
    release_query_resource_manager,
)


@pytest.fixture(autouse=True)
def _clear_registry():
    clear_query_resource_managers()
    yield
    clear_query_resource_managers()


def _graph(digest="sha256:a"):
    unit = ResourceUnitSpec(
        query_id="q",
        resource_unit_id="resource:f:scan",
        physical_node_id="scan",
        unit_kind="native_fragment",
        backend="ray_worker",
        input_unit_ids=(),
        per_task=ResourceVector(),
        target_output_block_bytes=10,
        generator_buffer_blocks=2,
        max_concurrency=4,
    )
    return QueryResourceGraph("q", digest, (unit,), (unit.resource_unit_id,))


def _allocation(generation=1):
    resources = ResourceVector(cpu=4, heap_bytes=1_000, object_store_bytes=1_000)
    return QueryAllocation(
        resources=resources,
        generation=generation,
    )


def test_runtime_never_lazily_creates_manager_before_graph_registration():
    with pytest.raises(KeyError, match="query resource graph is not registered"):
        get_query_resource_manager("q")
    assert query_resource_manager_snapshot("q") == {}


def test_runtime_registers_graph_atomically_and_rejects_every_duplicate():
    manager = register_query_resource_graph(_graph(), _allocation())

    assert get_query_resource_manager("q") is manager
    assert query_resource_manager_snapshot("q")["graph"]["plan_digest"] == "sha256:a"
    with pytest.raises(ValueError, match="already registered"):
        register_query_resource_graph(_graph(), _allocation())
    with pytest.raises(ValueError, match="already registered"):
        register_query_resource_graph(_graph("sha256:different"), _allocation())


def test_runtime_accepts_a_soft_budget_smaller_than_one_concrete_task():
    unit = ResourceUnitSpec(
        query_id="q",
        resource_unit_id="resource:f:udf",
        physical_node_id="udf",
        unit_kind="ray_task_udf",
        backend="ray_task",
        input_unit_ids=(),
        per_task=ResourceVector(cpu=1, heap_bytes=100),
        target_output_block_bytes=10,
        generator_buffer_blocks=2,
        max_concurrency=None,
    )
    graph = QueryResourceGraph("q", "sha256:remote", (unit,), (unit.resource_unit_id,))
    too_small_resources = ResourceVector(cpu=1, heap_bytes=99, object_store_bytes=1_000)
    too_small = QueryAllocation(
        resources=too_small_resources,
        generation=1,
    )

    manager = register_query_resource_graph(graph, too_small)

    assert manager.allocation.resources == too_small_resources
    assert query_resource_manager_snapshot("q")["ray_core_owns_placement"] is True


def test_runtime_can_publish_pending_query_before_minimum_bundle_is_feasible():
    unit = ResourceUnitSpec(
        query_id="q",
        resource_unit_id="resource:f:udf",
        physical_node_id="udf",
        unit_kind="ray_task_udf",
        backend="ray_task",
        input_unit_ids=(),
        per_task=ResourceVector(cpu=1, heap_bytes=100),
        target_output_block_bytes=10,
        generator_buffer_blocks=2,
        max_concurrency=None,
    )
    graph = QueryResourceGraph("q", "sha256:pending", (unit,), (unit.resource_unit_id,))
    pending = QueryAllocation(
        resources=ResourceVector(),
        generation=1,
    )

    manager = register_query_resource_graph(
        graph,
        pending,
        admission_open=False,
    )

    assert manager.snapshot()["allocation_admission_open"] is False
    manager.update_unit_state(unit.resource_unit_id, runnable=True)
    blocked = manager.try_acquire_task(
        TaskRequest(
            query_id="q",
            resource_unit_id=unit.resource_unit_id,
            task_id="task:pending",
            attempt_id="0",
            node_id=None,
        )
    )
    assert blocked.blocked_reason == "allocation_pending"

    manager.update_allocation(
        _allocation(generation=2),
        reopen_fence_epoch=manager.current_allocation_frontier()[1],
    )
    assert manager.try_acquire_task(
        TaskRequest(
            query_id="q",
            resource_unit_id=unit.resource_unit_id,
            task_id="task:running",
            attempt_id="0",
            node_id=None,
        )
    ).granted


def test_runtime_release_cancels_and_removes_manager_idempotently():
    register_query_resource_graph(_graph(), _allocation())

    first = release_query_resource_manager("q", reason="completed")
    second = release_query_resource_manager("q", reason="completed")

    assert first["released"] is True
    assert first["task_lease_count"] == 0
    assert first["output_lease_count"] == 0
    assert second == {"released": False, "task_lease_count": 0, "output_lease_count": 0}
    assert query_resource_manager_snapshot("q") == {}


def test_native_barrier_event_advances_driver_local_execution_phase_once():
    upstream = ResourceUnitSpec(
        query_id="q",
        resource_unit_id="resource:q:upstream",
        physical_node_id="node:upstream:native-fragment",
        unit_kind="native_fragment",
        backend="ray_worker",
        input_unit_ids=(),
        per_task=ResourceVector(),
        target_output_block_bytes=10,
        generator_buffer_blocks=2,
        max_concurrency=4,
    )
    materializer = ResourceUnitSpec(
        query_id="q",
        resource_unit_id="resource:q:materializer",
        physical_node_id="node:materializer:native-fragment",
        unit_kind="native_fragment",
        backend="ray_worker",
        input_unit_ids=(upstream.resource_unit_id,),
        per_task=ResourceVector(),
        target_output_block_bytes=10,
        generator_buffer_blocks=2,
        max_concurrency=4,
    )
    downstream = ResourceUnitSpec(
        query_id="q",
        resource_unit_id="resource:q:downstream",
        physical_node_id="downstream",
        unit_kind="ray_task_udf",
        backend="ray_task",
        input_unit_ids=(materializer.resource_unit_id,),
        per_task=ResourceVector(cpu=1),
        target_output_block_bytes=10,
        generator_buffer_blocks=2,
        max_concurrency=None,
    )
    graph = QueryResourceGraph(
        "q",
        "sha256:barrier",
        (upstream, materializer, downstream),
        (downstream.resource_unit_id,),
        (
            MaterializationBarrierSpec(
                query_id="q",
                barrier_id="barrier:q:node:materializer",
                physical_node_id="materializer",
                materializer_unit_id=materializer.resource_unit_id,
                materialized_input_unit_ids=(upstream.resource_unit_id,),
            ),
        ),
    )
    manager = register_query_resource_graph(graph, _allocation())

    assert manager.current_eligible_resource_unit_ids() == (
        upstream.resource_unit_id,
        materializer.resource_unit_id,
    )
    assert mark_materialization_barrier_completed("q", "materializer") is True
    assert mark_materialization_barrier_completed("q", "materializer") is False
    assert manager.current_eligible_resource_unit_ids() == (
        materializer.resource_unit_id,
        downstream.resource_unit_id,
    )
    assert manager.snapshot()["allocation_admission_open"] is False
