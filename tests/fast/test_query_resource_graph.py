# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

from duckdb.runners.ray.query_resource_graph import (
    MaterializationBarrierSpec,
    NodeResourceAllocation,
    QueryAllocation,
    QueryResourceGraph,
    ResourceUnitSpec,
    ResourceVector,
)

MIB = 1024 * 1024


def _resources(
    *,
    cpu: float = 1.0,
    gpu: float = 0.0,
    heap_bytes: int = 256 * MIB,
    object_store_bytes: int = 0,
) -> ResourceVector:
    return ResourceVector(
        cpu=cpu,
        gpu=gpu,
        heap_bytes=heap_bytes,
        object_store_bytes=object_store_bytes,
    )


def _allocation(resources: ResourceVector, *, generation: int) -> QueryAllocation:
    return QueryAllocation(
        resources=resources,
        node_allocations=(NodeResourceAllocation(node_id="node-a", resources=resources),),
        generation=generation,
    )


def _unit(
    resource_unit_id: str,
    *,
    inputs: tuple[str, ...] = (),
    physical_node_id: str | None = None,
    backend: str = "ray_task",
    per_task: ResourceVector | None = None,
    resident_per_actor: ResourceVector | None = None,
    target_output_block_bytes: int = 16 * MIB,
    generator_buffer_blocks: int = 2,
    max_concurrency: int | None = None,
    actor_pool_size: int = 0,
    actor_prefetch_depth: int = 1,
    unit_kind: str | None = None,
) -> ResourceUnitSpec:
    requested = per_task if per_task is not None else (ResourceVector() if backend == "ray_worker" else _resources())
    if backend == "ray_actor":
        resident = resident_per_actor or ResourceVector(
            cpu=requested.cpu,
            gpu=requested.gpu,
            heap_bytes=requested.heap_bytes,
        )
        invocation = ResourceVector(object_store_bytes=requested.object_store_bytes)
    else:
        resident = ResourceVector()
        invocation = requested
    resolved_unit_kind = (
        unit_kind
        or {
            "ray_worker": "native_fragment",
            "ray_task": "ray_task_udf",
            "ray_actor": "ray_actor_pool",
        }[backend]
    )
    return ResourceUnitSpec(
        query_id="q1",
        resource_unit_id=resource_unit_id,
        physical_node_id=physical_node_id or resource_unit_id.rsplit(":", 1)[-1],
        unit_kind=resolved_unit_kind,
        backend=backend,
        input_unit_ids=inputs,
        per_task=invocation,
        target_output_block_bytes=target_output_block_bytes,
        generator_buffer_blocks=generator_buffer_blocks,
        max_concurrency=max_concurrency,
        resident_per_actor=resident,
        actor_pool_size=actor_pool_size,
        actor_prefetch_depth=actor_prefetch_depth,
    )


def _graph(
    *units: ResourceUnitSpec,
    terminals: tuple[str, ...],
    barriers: tuple[MaterializationBarrierSpec, ...] = (),
) -> QueryResourceGraph:
    return QueryResourceGraph(
        query_id="q1",
        plan_digest="sha256:abc123",
        units=tuple(units),
        terminal_unit_ids=terminals,
        materialization_barriers=barriers,
    )


def _barrier(
    node_id: str,
    unit: ResourceUnitSpec,
    *,
    materialized_inputs: tuple[str, ...] | None = None,
) -> MaterializationBarrierSpec:
    return MaterializationBarrierSpec(
        query_id="q1",
        barrier_id=f"barrier:q1:node:{node_id}",
        physical_node_id=node_id,
        materializer_unit_id=unit.resource_unit_id,
        materialized_input_unit_ids=(unit.input_unit_ids if materialized_inputs is None else materialized_inputs),
    )


def test_resource_vector_arithmetic_is_component_wise_and_non_mutating():
    left = _resources(cpu=1.5, gpu=0.25, heap_bytes=10, object_store_bytes=20)
    right = _resources(cpu=0.5, gpu=0.10, heap_bytes=2, object_store_bytes=3)

    assert left + right == _resources(
        cpu=2.0,
        gpu=0.35,
        heap_bytes=12,
        object_store_bytes=23,
    )
    assert left - right == _resources(
        cpu=1.0,
        gpu=0.15,
        heap_bytes=8,
        object_store_bytes=17,
    )
    assert left == _resources(cpu=1.5, gpu=0.25, heap_bytes=10, object_store_bytes=20)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cpu", -0.01),
        ("gpu", -0.01),
        ("heap_bytes", -1),
        ("object_store_bytes", -1),
    ],
)
def test_resource_vector_rejects_negative_capacity(field, value):
    values = _resources().to_dict()
    values[field] = value

    with pytest.raises(ValueError, match=field):
        ResourceVector.from_dict(values)


def test_resource_vector_fit_and_dominant_share_include_every_dimension():
    demand = _resources(cpu=2, gpu=1, heap_bytes=50, object_store_bytes=80)
    capacity = _resources(cpu=4, gpu=2, heap_bytes=100, object_store_bytes=100)

    assert demand.fits_within(capacity)
    assert demand.dominant_share(capacity) == pytest.approx(0.8)
    assert not _resources(cpu=5).fits_within(capacity)


def test_graph_orders_units_deterministically_and_preserves_one_unit_identity_for_all_attempts():
    scan = _unit("resource:fragment-1:scan", backend="ray_worker", max_concurrency=36)
    cpu_udf = _unit("resource:fragment-1:cpu-udf", inputs=(scan.resource_unit_id,))
    gpu_udf = _unit(
        "resource:fragment-1:gpu-udf",
        inputs=(cpu_udf.resource_unit_id,),
        backend="ray_actor",
        per_task=_resources(cpu=1, gpu=1, heap_bytes=1024 * MIB),
        max_concurrency=None,
        actor_pool_size=1,
    )
    graph = _graph(gpu_udf, scan, cpu_udf, terminals=(gpu_udf.resource_unit_id,))

    assert graph.topological_unit_ids() == (scan.resource_unit_id, cpu_udf.resource_unit_id, gpu_udf.resource_unit_id)
    assert graph.reverse_topological_unit_ids() == (
        gpu_udf.resource_unit_id,
        cpu_udf.resource_unit_id,
        scan.resource_unit_id,
    )
    assert graph.unit_id_for_physical_node("scan") == scan.resource_unit_id
    assert graph.task_identity(scan.resource_unit_id, partition_id=0, attempt_id="a1") == (
        "task:resource:fragment-1:scan:partition:0:attempt:a1"
    )
    assert graph.task_identity(scan.resource_unit_id, partition_id=35, attempt_id="a2").startswith(
        "task:resource:fragment-1:scan:"
    )


def test_completed_barrier_retires_old_phase_before_next_phase_becomes_eligible():
    scan = _unit("resource:q1:scan", backend="ray_worker", max_concurrency=4)
    first = _unit(
        "resource:q1:first-barrier",
        inputs=(scan.resource_unit_id,),
        backend="ray_worker",
        max_concurrency=4,
    )
    middle = _unit("resource:q1:middle", inputs=(first.resource_unit_id,))
    second = _unit(
        "resource:q1:second-barrier",
        inputs=(middle.resource_unit_id,),
        backend="ray_worker",
        max_concurrency=4,
    )
    sink = _unit("resource:q1:sink", inputs=(second.resource_unit_id,))
    first_barrier = _barrier("first", first)
    second_barrier = _barrier("second", second)
    graph = _graph(
        scan,
        first,
        middle,
        second,
        sink,
        terminals=(sink.resource_unit_id,),
        barriers=(first_barrier, second_barrier),
    )

    assert graph.eligible_resource_unit_ids(set()) == (
        scan.resource_unit_id,
        first.resource_unit_id,
    )
    assert graph.eligible_resource_unit_ids({first_barrier.barrier_id}) == (
        first.resource_unit_id,
        middle.resource_unit_id,
        second.resource_unit_id,
    )
    assert graph.eligible_resource_unit_ids({first_barrier.barrier_id, second_barrier.barrier_id}) == (
        second.resource_unit_id,
        sink.resource_unit_id,
    )


def test_parallel_first_barriers_share_one_execution_phase_union():
    scan = _unit("resource:q1:scan", backend="ray_worker", max_concurrency=4)
    left = _unit(
        "resource:q1:left-barrier",
        inputs=(scan.resource_unit_id,),
        backend="ray_worker",
        max_concurrency=4,
    )
    right = _unit(
        "resource:q1:right-barrier",
        inputs=(scan.resource_unit_id,),
        backend="ray_worker",
        max_concurrency=4,
    )
    join = _unit("resource:q1:join", inputs=(left.resource_unit_id, right.resource_unit_id))
    left_barrier = _barrier("left", left)
    right_barrier = _barrier("right", right)
    graph = _graph(
        scan,
        left,
        right,
        join,
        terminals=(join.resource_unit_id,),
        barriers=(left_barrier, right_barrier),
    )

    assert graph.eligible_resource_unit_ids(set()) == (
        scan.resource_unit_id,
        left.resource_unit_id,
        right.resource_unit_id,
    )
    assert tuple(barrier.barrier_id for barrier in graph.frontier_materialization_barriers(set())) == (
        left_barrier.barrier_id,
        right_barrier.barrier_id,
    )
    assert graph.eligible_resource_unit_ids({left_barrier.barrier_id}) == (
        scan.resource_unit_id,
        left.resource_unit_id,
        right.resource_unit_id,
    )
    assert graph.eligible_resource_unit_ids({left_barrier.barrier_id, right_barrier.barrier_id}) == (
        left.resource_unit_id,
        right.resource_unit_id,
        join.resource_unit_id,
    )


def test_completed_parallel_barrier_opens_its_streaming_branch_before_sibling_barrier():
    left_scan = _unit("resource:q1:left-scan", backend="ray_worker", max_concurrency=4)
    left_barrier_unit = _unit(
        "resource:q1:left-barrier",
        inputs=(left_scan.resource_unit_id,),
        backend="ray_worker",
        max_concurrency=4,
    )
    left_stream = _unit(
        "resource:q1:left-stream",
        inputs=(left_barrier_unit.resource_unit_id,),
    )
    right_scan = _unit("resource:q1:right-scan", backend="ray_worker", max_concurrency=4)
    right_barrier_unit = _unit(
        "resource:q1:right-barrier",
        inputs=(right_scan.resource_unit_id,),
        backend="ray_worker",
        max_concurrency=4,
    )
    join = _unit(
        "resource:q1:join",
        inputs=(left_stream.resource_unit_id, right_barrier_unit.resource_unit_id),
    )
    left_barrier = _barrier("left", left_barrier_unit)
    right_barrier = _barrier("right", right_barrier_unit)
    graph = _graph(
        left_scan,
        left_barrier_unit,
        left_stream,
        right_scan,
        right_barrier_unit,
        join,
        terminals=(join.resource_unit_id,),
        barriers=(left_barrier, right_barrier),
    )

    assert graph.eligible_resource_unit_ids({left_barrier.barrier_id}) == (
        left_barrier_unit.resource_unit_id,
        left_stream.resource_unit_id,
        right_scan.resource_unit_id,
        right_barrier_unit.resource_unit_id,
    )


def test_asymmetric_barrier_switches_from_materialized_to_deferred_input_branch():
    build_scan = _unit("resource:q1:build-scan", backend="ray_worker", max_concurrency=4)
    build_udf = _unit(
        "resource:q1:build-udf",
        inputs=(build_scan.resource_unit_id,),
    )
    probe_scan = _unit("resource:q1:probe-scan", backend="ray_worker", max_concurrency=4)
    probe_udf = _unit(
        "resource:q1:probe-udf",
        inputs=(probe_scan.resource_unit_id,),
    )
    broadcast_join = _unit(
        "resource:q1:broadcast-join",
        inputs=(build_udf.resource_unit_id, probe_udf.resource_unit_id),
        backend="ray_worker",
        max_concurrency=4,
    )
    sink = _unit("resource:q1:sink", inputs=(broadcast_join.resource_unit_id,))
    barrier = _barrier(
        "broadcast-join",
        broadcast_join,
        materialized_inputs=(build_udf.resource_unit_id,),
    )
    graph = _graph(
        build_scan,
        build_udf,
        probe_scan,
        probe_udf,
        broadcast_join,
        sink,
        terminals=(sink.resource_unit_id,),
        barriers=(barrier,),
    )

    assert graph.eligible_resource_unit_ids(set()) == (
        build_scan.resource_unit_id,
        build_udf.resource_unit_id,
        broadcast_join.resource_unit_id,
    )
    assert graph.eligible_resource_unit_ids({barrier.barrier_id}) == (
        probe_scan.resource_unit_id,
        probe_udf.resource_unit_id,
        broadcast_join.resource_unit_id,
        sink.resource_unit_id,
    )


def test_graph_serialization_round_trip_is_strict_and_stable():
    scan = _unit("resource:fragment-1:scan")
    sink = _unit(
        "resource:fragment-2:sink",
        inputs=(scan.resource_unit_id,),
        backend="ray_worker",
        target_output_block_bytes=0,
        generator_buffer_blocks=0,
    )
    barrier = _barrier("sink", sink)
    graph = _graph(
        scan,
        sink,
        terminals=(sink.resource_unit_id,),
        barriers=(barrier,),
    )

    payload = graph.to_dict()

    assert QueryResourceGraph.from_dict(payload) == graph
    assert payload["materialization_barriers"][0]["materialized_input_unit_ids"] == [scan.resource_unit_id]
    assert list(payload) == [
        "query_id",
        "plan_digest",
        "units",
        "materialization_barriers",
        "terminal_unit_ids",
    ]
    with pytest.raises(ValueError, match="unknown fields"):
        QueryResourceGraph.from_dict({**payload, "legacy_operator_specs": []})

    unit_payload = dict(payload["units"][0])
    unit_payload["stage_id"] = unit_payload.pop("resource_unit_id")
    with pytest.raises(ValueError, match="unknown fields: stage_id"):
        ResourceUnitSpec.from_dict(unit_payload)


def test_materialization_barrier_identity_is_canonicalized_before_frontier_traversal():
    scan = _unit("resource:q1:scan", backend="ray_worker", max_concurrency=1)
    sink = _unit(
        "resource:q1:sink",
        inputs=(scan.resource_unit_id,),
        backend="ray_worker",
        target_output_block_bytes=0,
        generator_buffer_blocks=0,
        max_concurrency=1,
    )
    barrier = MaterializationBarrierSpec(
        query_id=" q1 ",
        barrier_id=" barrier:q1:node:sink ",
        physical_node_id=" sink ",
        materializer_unit_id=f" {sink.resource_unit_id} ",
        materialized_input_unit_ids=(f" {scan.resource_unit_id} ",),
    )

    graph = _graph(scan, sink, terminals=(sink.resource_unit_id,), barriers=(barrier,))

    assert graph.materialization_barriers[0] == _barrier("sink", sink)
    assert graph.eligible_resource_unit_ids({barrier.barrier_id}) == (sink.resource_unit_id,)


def test_graph_rejects_barrier_id_for_a_different_physical_node():
    scan = _unit("resource:q1:scan", backend="ray_worker", max_concurrency=1)
    sink = _unit(
        "resource:q1:sink",
        inputs=(scan.resource_unit_id,),
        backend="ray_worker",
        target_output_block_bytes=0,
        generator_buffer_blocks=0,
        max_concurrency=1,
    )
    barrier = MaterializationBarrierSpec(
        query_id="q1",
        barrier_id="barrier:q1:node:other",
        physical_node_id="sink",
        materializer_unit_id=sink.resource_unit_id,
        materialized_input_unit_ids=(scan.resource_unit_id,),
    )

    with pytest.raises(ValueError, match="invalid materialization barrier identity"):
        _graph(scan, sink, terminals=(sink.resource_unit_id,), barriers=(barrier,))


def test_graph_rejects_duplicate_unit_ids():
    first = _unit("resource:fragment-1:scan", physical_node_id="scan-a")
    duplicate = _unit("resource:fragment-1:scan", physical_node_id="scan-b")

    with pytest.raises(ValueError, match="duplicate resource_unit_id"):
        _graph(first, duplicate, terminals=(first.resource_unit_id,))


def test_graph_rejects_missing_dependencies():
    sink = _unit("resource:fragment-1:sink", inputs=("resource:fragment-1:missing",))

    with pytest.raises(ValueError, match="missing input unit"):
        _graph(sink, terminals=(sink.resource_unit_id,))


def test_graph_rejects_cycles():
    left = _unit("resource:fragment-1:left", inputs=("resource:fragment-1:right",))
    right = _unit("resource:fragment-1:right", inputs=(left.resource_unit_id,))

    with pytest.raises(ValueError, match="cycle"):
        _graph(left, right, terminals=(right.resource_unit_id,))


def test_graph_rejects_non_terminal_branch_and_terminal_with_downstream_unit():
    scan = _unit("resource:fragment-1:scan")
    used = _unit("resource:fragment-1:used", inputs=(scan.resource_unit_id,))
    orphan = _unit("resource:fragment-1:orphan")

    with pytest.raises(ValueError, match="does not reach a terminal"):
        _graph(scan, used, orphan, terminals=(used.resource_unit_id,))

    with pytest.raises(ValueError, match="terminal unit.*has downstream"):
        _graph(scan, used, terminals=(scan.resource_unit_id,))


@pytest.mark.parametrize(
    ("materialized_inputs", "message"),
    [
        ((), "at least one input"),
        (("resource:q1:missing",), "is not a direct input"),
        (("resource:q1:scan", "resource:q1:scan"), "duplicate materialized"),
    ],
)
def test_graph_rejects_invalid_materialized_barrier_input_edges(
    materialized_inputs,
    message,
):
    scan = _unit("resource:q1:scan", backend="ray_worker", max_concurrency=4)
    materializer = _unit(
        "resource:q1:materializer",
        inputs=(scan.resource_unit_id,),
        backend="ray_worker",
        max_concurrency=4,
    )

    with pytest.raises(ValueError, match=message):
        _graph(
            scan,
            materializer,
            terminals=(materializer.resource_unit_id,),
            barriers=(
                _barrier(
                    "materializer",
                    materializer,
                    materialized_inputs=materialized_inputs,
                ),
            ),
        )


@pytest.mark.parametrize("backend", ["ray_task", "ray_actor"])
def test_graph_accepts_undeclared_heap_for_ray_python_process(backend):
    unit = _unit(
        "resource:fragment-1:udf",
        backend=backend,
        per_task=_resources(heap_bytes=0),
        actor_pool_size=1 if backend == "ray_actor" else 0,
    )

    graph = _graph(unit, terminals=(unit.resource_unit_id,))
    registered = graph.unit_by_id(unit.resource_unit_id)
    process_resources = registered.resident_per_actor if backend == "ray_actor" else registered.per_task

    assert process_resources.heap_bytes == 0


def test_graph_rejects_native_process_resources_owned_outside_duckdb():
    unit = _unit(
        "resource:fragment-1:native",
        backend="ray_worker",
        per_task=_resources(cpu=1, heap_bytes=256 * MIB, object_store_bytes=10),
    )

    with pytest.raises(ValueError, match="process resources are owned by DuckDB"):
        _graph(unit, terminals=(unit.resource_unit_id,))


def test_graph_rejects_ray_unit_without_cpu_or_gpu_scheduling_resources():
    unit = _unit(
        "resource:fragment-1:udf",
        per_task=_resources(cpu=0, gpu=0),
    )

    with pytest.raises(ValueError, match="CPU or GPU"):
        _graph(unit, terminals=(unit.resource_unit_id,))


@pytest.mark.parametrize(
    ("backend", "unit_kind"),
    [
        ("ray_worker", "ray_task_udf"),
        ("ray_task", "native_fragment"),
        ("ray_actor", "ray_task_udf"),
    ],
)
def test_graph_rejects_resource_unit_kind_that_does_not_match_backend(backend, unit_kind):
    unit = _unit(
        "resource:fragment-1:mismatched-kind",
        backend=backend,
        actor_pool_size=1 if backend == "ray_actor" else 0,
        unit_kind=unit_kind,
    )

    with pytest.raises(ValueError, match="does not match backend"):
        _graph(unit, terminals=(unit.resource_unit_id,))


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"actor_pool_size": 0}, "actor_pool_size"),
        ({"actor_pool_size": 1, "max_concurrency": 1}, "concurrency is owned"),
    ],
)
def test_graph_rejects_invalid_actor_pool(changes, message):
    params = {
        "backend": "ray_actor",
        "max_concurrency": None,
        "actor_pool_size": 1,
        **changes,
    }
    unit = _unit("resource:fragment-1:gpu-udf", **params)

    with pytest.raises(ValueError, match=message):
        _graph(unit, terminals=(unit.resource_unit_id,))


def test_graph_rejects_actor_pool_on_non_actor_unit():
    unit = _unit(
        "resource:fragment-1:udf",
        backend="ray_task",
        actor_pool_size=1,
    )

    with pytest.raises(ValueError, match="only valid for ray_actor"):
        _graph(unit, terminals=(unit.resource_unit_id,))


def test_graph_rejects_invalid_actor_prefetch_depth():
    actor = _unit(
        "resource:fragment-1:gpu-udf",
        backend="ray_actor",
        actor_pool_size=1,
        actor_prefetch_depth=0,
    )
    with pytest.raises(ValueError, match="actor_prefetch_depth"):
        _graph(actor, terminals=(actor.resource_unit_id,))

    task = _unit(
        "resource:fragment-1:cpu-udf",
        backend="ray_task",
        actor_prefetch_depth=2,
    )
    with pytest.raises(ValueError, match="only configurable for ray_actor"):
        _graph(task, terminals=(task.resource_unit_id,))


def test_graph_rejects_inconsistent_output_window_shape():
    no_target = _unit(
        "resource:fragment-1:no-target",
        target_output_block_bytes=0,
        generator_buffer_blocks=2,
    )
    no_window = _unit(
        "resource:fragment-1:no-window",
        target_output_block_bytes=16 * MIB,
        generator_buffer_blocks=0,
    )

    with pytest.raises(ValueError, match="both be zero"):
        _graph(no_target, terminals=(no_target.resource_unit_id,))
    with pytest.raises(ValueError, match="both be positive"):
        _graph(no_window, terminals=(no_window.resource_unit_id,))


def test_legacy_intermediate_resource_dimension_is_rejected():
    payload = _resources().to_dict()
    payload["intermediate_bytes"] = 1

    with pytest.raises(ValueError, match="unknown fields: intermediate_bytes"):
        ResourceVector.from_dict(payload)


def test_allocation_validation_keeps_heap_hard_and_object_windows_soft():
    unit = _unit(
        "resource:fragment-1:decode",
        per_task=_resources(cpu=1, heap_bytes=300, object_store_bytes=50),
        target_output_block_bytes=100,
        generator_buffer_blocks=2,
    )
    graph = _graph(unit, terminals=(unit.resource_unit_id,))
    allocation = _allocation(
        _resources(
            cpu=4,
            heap_bytes=299,
            object_store_bytes=250,
        ),
        generation=7,
    )

    with pytest.raises(ValueError, match="heap_bytes"):
        graph.validate_allocation(allocation)

    graph.validate_allocation(
        _allocation(
            _resources(
                cpu=4,
                heap_bytes=300,
                object_store_bytes=1,
            ),
            generation=7,
        )
    )


def test_allocation_accepts_one_output_window_larger_than_soft_object_store_budget():
    unit = _unit(
        "resource:fragment-1:decode",
        target_output_block_bytes=101,
        generator_buffer_blocks=2,
    )
    graph = _graph(unit, terminals=(unit.resource_unit_id,))
    allocation = _allocation(
        _resources(cpu=4, heap_bytes=1024 * MIB, object_store_bytes=201),
        generation=1,
    )

    graph.validate_allocation(allocation)


def test_allocation_rejects_aggregate_resources_that_do_not_form_a_runnable_node():
    unit = _unit(
        "resource:fragment-1:decode",
        per_task=_resources(cpu=2, heap_bytes=300, object_store_bytes=0),
        target_output_block_bytes=100,
        generator_buffer_blocks=2,
    )
    graph = _graph(unit, terminals=(unit.resource_unit_id,))
    allocation = QueryAllocation(
        resources=_resources(cpu=2, heap_bytes=300, object_store_bytes=200),
        node_allocations=(
            NodeResourceAllocation(
                node_id="cpu-only",
                resources=_resources(cpu=2, heap_bytes=1, object_store_bytes=1),
            ),
            NodeResourceAllocation(
                node_id="memory-only",
                resources=_resources(cpu=0, heap_bytes=299, object_store_bytes=199),
            ),
        ),
        generation=1,
    )

    with pytest.raises(ValueError, match="does not fit any allocated Ray node"):
        graph.validate_allocation(allocation)


def test_allocation_validation_can_skip_tasks_outside_the_current_phase():
    unit = _unit(
        "resource:fragment-1:decode",
        per_task=_resources(cpu=1, heap_bytes=300),
        target_output_block_bytes=100,
        generator_buffer_blocks=2,
    )
    graph = _graph(unit, terminals=(unit.resource_unit_id,))
    pending = QueryAllocation(
        resources=ResourceVector(),
        node_allocations=(),
        generation=2,
    )

    with pytest.raises(ValueError, match="maximum task exceeds query allocation"):
        graph.validate_allocation(pending)

    graph.validate_allocation(pending, eligible_unit_ids=())


def test_current_phase_allocation_does_not_reserve_a_task_behind_a_barrier():
    source = _unit(
        "resource:q1:source",
        backend="ray_worker",
        per_task=ResourceVector(),
        target_output_block_bytes=0,
        generator_buffer_blocks=0,
    )
    materializer = _unit(
        "resource:q1:materializer",
        inputs=(source.resource_unit_id,),
        backend="ray_worker",
        per_task=ResourceVector(),
        target_output_block_bytes=10,
        generator_buffer_blocks=2,
    )
    future_task = _unit(
        "resource:q1:future-task",
        inputs=(materializer.resource_unit_id,),
        per_task=_resources(cpu=2, heap_bytes=300),
        target_output_block_bytes=10,
        generator_buffer_blocks=2,
    )
    graph = _graph(
        source,
        materializer,
        future_task,
        terminals=(future_task.resource_unit_id,),
        barriers=(_barrier("materializer", materializer),),
    )
    current = _allocation(
        ResourceVector(object_store_bytes=20),
        generation=1,
    )

    assert graph.eligible_resource_unit_ids(set()) == (
        source.resource_unit_id,
        materializer.resource_unit_id,
    )
    graph.validate_allocation(
        current,
        eligible_unit_ids=graph.eligible_resource_unit_ids(set()),
    )
    with pytest.raises(ValueError, match="maximum task exceeds query allocation"):
        graph.validate_allocation(current)


def test_runtime_allocation_validation_can_leave_actor_pool_to_ray_core():
    actor = _unit(
        "resource:fragment-1:actor",
        backend="ray_actor",
        per_task=_resources(object_store_bytes=100),
        resident_per_actor=_resources(cpu=1, heap_bytes=300),
        target_output_block_bytes=100,
        generator_buffer_blocks=2,
        actor_pool_size=2,
    )
    graph = _graph(actor, terminals=(actor.resource_unit_id,))
    pending = QueryAllocation(
        resources=ResourceVector(),
        node_allocations=(),
        generation=2,
    )

    graph.validate_allocation(pending)


def test_query_allocation_round_trip_requires_exact_per_node_sum():
    resources = _resources(cpu=3, heap_bytes=300, object_store_bytes=400)
    allocation = QueryAllocation(
        resources=resources,
        node_allocations=(
            NodeResourceAllocation(
                node_id="node-a",
                resources=_resources(cpu=1, heap_bytes=100, object_store_bytes=150),
            ),
            NodeResourceAllocation(
                node_id="node-b",
                resources=_resources(cpu=2, heap_bytes=200, object_store_bytes=250),
            ),
        ),
        generation=9,
    )

    assert QueryAllocation.from_dict(allocation.to_dict()) == allocation
    with pytest.raises(ValueError, match="sum of node_allocations"):
        QueryAllocation(
            resources=resources,
            node_allocations=allocation.node_allocations[:1],
            generation=9,
        )
