# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

from vane.runners.ray.query_execution_graph import (
    ActorPlacement,
    NodeResourceAllocation,
    QueryAllocation,
    QueryExecutionGraph,
    ResourceVector,
    StageResourceSpec,
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
        actor_placements=(),
        generation=generation,
    )


def _stage(
    stage_id: str,
    *,
    inputs: tuple[str, ...] = (),
    physical_node_id: str | None = None,
    backend: str = "ray_task",
    per_task: ResourceVector | None = None,
    resident_per_actor: ResourceVector | None = None,
    target_output_block_bytes: int = 16 * MIB,
    generator_buffer_blocks: int = 2,
    max_concurrency: int | None = None,
    actor_min_size: int = 0,
    actor_max_size: int = 0,
    actor_prefetch_depth: int = 1,
    spill_mode: str = "streaming",
) -> StageResourceSpec:
    requested = per_task or _resources()
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
    return StageResourceSpec(
        query_id="q1",
        stage_id=stage_id,
        physical_node_id=physical_node_id or stage_id.rsplit(":", 1)[-1],
        stage_kind="udf" if "udf" in stage_id else "fte",
        backend=backend,
        input_stage_ids=inputs,
        per_task=invocation,
        target_output_block_bytes=target_output_block_bytes,
        generator_buffer_blocks=generator_buffer_blocks,
        max_concurrency=max_concurrency,
        resident_per_actor=resident,
        actor_min_size=actor_min_size,
        actor_max_size=actor_max_size,
        actor_prefetch_depth=actor_prefetch_depth,
        spill_mode=spill_mode,
    )


def _graph(*stages: StageResourceSpec, terminals: tuple[str, ...]) -> QueryExecutionGraph:
    return QueryExecutionGraph(
        query_id="q1",
        plan_digest="sha256:abc123",
        stages=tuple(stages),
        terminal_stage_ids=terminals,
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


def test_graph_orders_stages_deterministically_and_preserves_one_stage_identity_for_all_attempts():
    scan = _stage("stage:fragment-1:scan", backend="ray_worker", max_concurrency=36)
    cpu_udf = _stage("stage:fragment-1:cpu-udf", inputs=(scan.stage_id,))
    gpu_udf = _stage(
        "stage:fragment-1:gpu-udf",
        inputs=(cpu_udf.stage_id,),
        backend="ray_actor",
        per_task=_resources(cpu=1, gpu=1, heap_bytes=1024 * MIB),
        max_concurrency=None,
        actor_min_size=1,
        actor_max_size=1,
    )
    graph = _graph(gpu_udf, scan, cpu_udf, terminals=(gpu_udf.stage_id,))

    assert graph.topological_stage_ids() == (scan.stage_id, cpu_udf.stage_id, gpu_udf.stage_id)
    assert graph.reverse_topological_stage_ids() == (gpu_udf.stage_id, cpu_udf.stage_id, scan.stage_id)
    assert graph.stage_id_for_physical_node("scan") == scan.stage_id
    assert graph.task_identity(scan.stage_id, partition_id=0, attempt_id="a1") == (
        "task:stage:fragment-1:scan:partition:0:attempt:a1"
    )
    assert graph.task_identity(scan.stage_id, partition_id=35, attempt_id="a2").startswith(
        "task:stage:fragment-1:scan:"
    )


def test_graph_identifies_downstream_fte_slots_after_non_fte_boundaries():
    first_fte = _stage(
        "stage:fragment-1:first-fte",
        backend="ray_worker",
        max_concurrency=36,
    )
    direct_fte = _stage(
        "stage:fragment-1:direct-fte",
        inputs=(first_fte.stage_id,),
        backend="ray_worker",
        max_concurrency=36,
    )
    cpu_udf = _stage(
        "stage:fragment-1:cpu-udf",
        inputs=(direct_fte.stage_id,),
    )
    post_task_fte = _stage(
        "stage:fragment-2:post-task-fte",
        inputs=(cpu_udf.stage_id,),
        backend="ray_worker",
        max_concurrency=36,
    )
    gpu_udf = _stage(
        "stage:fragment-2:gpu-udf",
        inputs=(post_task_fte.stage_id,),
        backend="ray_actor",
        per_task=_resources(gpu=1),
        actor_min_size=1,
        actor_max_size=1,
    )
    post_actor_fte = _stage(
        "stage:fragment-3:post-actor-fte",
        inputs=(gpu_udf.stage_id,),
        backend="ray_worker",
        max_concurrency=36,
        target_output_block_bytes=0,
        generator_buffer_blocks=0,
    )
    graph = _graph(
        post_actor_fte,
        cpu_udf,
        first_fte,
        gpu_udf,
        direct_fte,
        post_task_fte,
        terminals=(post_actor_fte.stage_id,),
    )

    assert graph.downstream_fte_stage_ids_requiring_separate_slot(first_fte.stage_id) == (
        post_task_fte.stage_id,
        post_actor_fte.stage_id,
    )
    assert graph.downstream_fte_stage_ids_requiring_separate_slot(direct_fte.stage_id) == (
        post_task_fte.stage_id,
        post_actor_fte.stage_id,
    )
    assert graph.downstream_fte_stage_ids_requiring_separate_slot(cpu_udf.stage_id) == (
        post_task_fte.stage_id,
        post_actor_fte.stage_id,
    )
    assert graph.downstream_fte_stage_ids_requiring_separate_slot(post_task_fte.stage_id) == (post_actor_fte.stage_id,)
    assert graph.downstream_fte_stage_ids_requiring_separate_slot(gpu_udf.stage_id) == (post_actor_fte.stage_id,)
    assert graph.downstream_fte_stage_ids_requiring_separate_slot(post_actor_fte.stage_id) == ()


def test_graph_serialization_round_trip_is_strict_and_stable():
    scan = _stage("stage:fragment-1:scan")
    sink = _stage(
        "stage:fragment-2:sink",
        inputs=(scan.stage_id,),
        backend="ray_worker",
        target_output_block_bytes=0,
        generator_buffer_blocks=0,
    )
    graph = _graph(scan, sink, terminals=(sink.stage_id,))

    payload = graph.to_dict()

    assert QueryExecutionGraph.from_dict(payload) == graph
    assert list(payload) == ["query_id", "plan_digest", "stages", "terminal_stage_ids"]
    with pytest.raises(ValueError, match="unknown fields"):
        QueryExecutionGraph.from_dict({**payload, "legacy_operator_specs": []})


def test_graph_rejects_duplicate_stage_ids():
    first = _stage("stage:fragment-1:scan", physical_node_id="scan-a")
    duplicate = _stage("stage:fragment-1:scan", physical_node_id="scan-b")

    with pytest.raises(ValueError, match="duplicate stage_id"):
        _graph(first, duplicate, terminals=(first.stage_id,))


def test_graph_rejects_missing_dependencies():
    sink = _stage("stage:fragment-1:sink", inputs=("stage:fragment-1:missing",))

    with pytest.raises(ValueError, match="missing input stage"):
        _graph(sink, terminals=(sink.stage_id,))


def test_graph_rejects_cycles():
    left = _stage("stage:fragment-1:left", inputs=("stage:fragment-1:right",))
    right = _stage("stage:fragment-1:right", inputs=(left.stage_id,))

    with pytest.raises(ValueError, match="cycle"):
        _graph(left, right, terminals=(right.stage_id,))


def test_graph_rejects_non_terminal_branch_and_terminal_with_downstream_stage():
    scan = _stage("stage:fragment-1:scan")
    used = _stage("stage:fragment-1:used", inputs=(scan.stage_id,))
    orphan = _stage("stage:fragment-1:orphan")

    with pytest.raises(ValueError, match="does not reach a terminal"):
        _graph(scan, used, orphan, terminals=(used.stage_id,))

    with pytest.raises(ValueError, match="terminal stage.*has downstream"):
        _graph(scan, used, terminals=(scan.stage_id,))


@pytest.mark.parametrize("backend", ["ray_task", "ray_actor", "ray_worker"])
def test_graph_rejects_zero_heap_for_every_ray_python_process(backend):
    stage = _stage(
        "stage:fragment-1:udf",
        backend=backend,
        per_task=_resources(heap_bytes=0),
        actor_min_size=1 if backend == "ray_actor" else 0,
        actor_max_size=1 if backend == "ray_actor" else 0,
    )

    with pytest.raises(ValueError, match="non-zero heap_bytes"):
        _graph(stage, terminals=(stage.stage_id,))


def test_graph_rejects_ray_stage_without_cpu_or_gpu_scheduling_resources():
    stage = _stage(
        "stage:fragment-1:udf",
        per_task=_resources(cpu=0, gpu=0),
    )

    with pytest.raises(ValueError, match="CPU or GPU"):
        _graph(stage, terminals=(stage.stage_id,))


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"actor_min_size": 0, "actor_max_size": 1}, "actor_min_size"),
        ({"actor_min_size": 2, "actor_max_size": 1}, "actor_max_size"),
        ({"actor_min_size": 1, "actor_max_size": 1, "max_concurrency": 1}, "concurrency is owned"),
    ],
)
def test_graph_rejects_invalid_actor_bounds(changes, message):
    params = {
        "backend": "ray_actor",
        "max_concurrency": None,
        "actor_min_size": 1,
        "actor_max_size": 1,
        **changes,
    }
    stage = _stage("stage:fragment-1:gpu-udf", **params)

    with pytest.raises(ValueError, match=message):
        _graph(stage, terminals=(stage.stage_id,))


def test_graph_rejects_actor_bounds_on_non_actor_stage():
    stage = _stage(
        "stage:fragment-1:udf",
        backend="ray_task",
        actor_min_size=1,
        actor_max_size=1,
    )

    with pytest.raises(ValueError, match="only valid for ray_actor"):
        _graph(stage, terminals=(stage.stage_id,))


def test_graph_rejects_invalid_actor_prefetch_depth():
    actor = _stage(
        "stage:fragment-1:gpu-udf",
        backend="ray_actor",
        actor_min_size=1,
        actor_max_size=1,
        actor_prefetch_depth=0,
    )
    with pytest.raises(ValueError, match="actor_prefetch_depth"):
        _graph(actor, terminals=(actor.stage_id,))

    task = _stage(
        "stage:fragment-1:cpu-udf",
        backend="ray_task",
        actor_prefetch_depth=2,
    )
    with pytest.raises(ValueError, match="only configurable for ray_actor"):
        _graph(task, terminals=(task.stage_id,))


@pytest.mark.parametrize("spill_mode", ["disk", "unbounded", "legacy"])
def test_graph_rejects_unknown_spill_mode(spill_mode):
    stage = _stage("stage:fragment-1:scan", spill_mode=spill_mode)

    with pytest.raises(ValueError, match="spill_mode"):
        _graph(stage, terminals=(stage.stage_id,))


def test_graph_rejects_inconsistent_output_window_shape():
    no_target = _stage(
        "stage:fragment-1:no-target",
        target_output_block_bytes=0,
        generator_buffer_blocks=2,
    )
    no_window = _stage(
        "stage:fragment-1:no-window",
        target_output_block_bytes=16 * MIB,
        generator_buffer_blocks=0,
    )

    with pytest.raises(ValueError, match="both be zero"):
        _graph(no_target, terminals=(no_target.stage_id,))
    with pytest.raises(ValueError, match="both be positive"):
        _graph(no_window, terminals=(no_window.stage_id,))


def test_legacy_intermediate_resource_dimension_is_rejected():
    payload = _resources().to_dict()
    payload["intermediate_bytes"] = 1

    with pytest.raises(ValueError, match="unknown fields: intermediate_bytes"):
        ResourceVector.from_dict(payload)


def test_allocation_validation_includes_task_heap_retained_input_and_output_window():
    stage = _stage(
        "stage:fragment-1:decode",
        per_task=_resources(cpu=1, heap_bytes=300, object_store_bytes=50),
        target_output_block_bytes=100,
        generator_buffer_blocks=2,
    )
    graph = _graph(stage, terminals=(stage.stage_id,))
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
                object_store_bytes=250,
            ),
            generation=7,
        )
    )


def test_allocation_rejects_one_output_window_larger_than_hard_object_store_limit():
    stage = _stage(
        "stage:fragment-1:decode",
        target_output_block_bytes=101,
        generator_buffer_blocks=2,
    )
    graph = _graph(stage, terminals=(stage.stage_id,))
    allocation = _allocation(
        _resources(cpu=4, heap_bytes=1024 * MIB, object_store_bytes=201),
        generation=1,
    )

    with pytest.raises(ValueError, match="output window"):
        graph.validate_allocation(allocation)


def test_allocation_rejects_aggregate_resources_that_do_not_form_a_runnable_node():
    stage = _stage(
        "stage:fragment-1:decode",
        per_task=_resources(cpu=2, heap_bytes=300, object_store_bytes=0),
        target_output_block_bytes=100,
        generator_buffer_blocks=2,
    )
    graph = _graph(stage, terminals=(stage.stage_id,))
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
        actor_placements=(),
        generation=1,
    )

    with pytest.raises(ValueError, match="does not fit any allocated Ray node"):
        graph.validate_allocation(allocation)


def test_runtime_allocation_validation_accepts_pending_zero_capacity():
    stage = _stage(
        "stage:fragment-1:decode",
        per_task=_resources(cpu=1, heap_bytes=300),
        target_output_block_bytes=100,
        generator_buffer_blocks=2,
    )
    graph = _graph(stage, terminals=(stage.stage_id,))
    pending = QueryAllocation(
        resources=ResourceVector(),
        node_allocations=(),
        actor_placements=(),
        generation=2,
    )

    with pytest.raises(ValueError, match="maximum task exceeds query allocation"):
        graph.validate_allocation(pending)

    graph.validate_allocation(pending, require_full_minimum=False)


def test_allocation_rejects_cumulative_actor_placements_on_one_node():
    actor = _stage(
        "stage:fragment-1:gpu",
        backend="ray_actor",
        per_task=_resources(object_store_bytes=10),
        resident_per_actor=_resources(cpu=1, gpu=1, heap_bytes=100),
        target_output_block_bytes=10,
        generator_buffer_blocks=2,
        max_concurrency=None,
        actor_min_size=2,
        actor_max_size=2,
    )
    graph = _graph(actor, terminals=(actor.stage_id,))
    per_node = _resources(cpu=1, gpu=1, heap_bytes=100, object_store_bytes=30)
    allocation = QueryAllocation(
        resources=per_node.scale(2),
        node_allocations=(
            NodeResourceAllocation(node_id="node-a", resources=per_node),
            NodeResourceAllocation(node_id="node-b", resources=per_node),
        ),
        actor_placements=(
            ActorPlacement(stage_id=actor.stage_id, actor_index=0, node_id="node-a"),
            ActorPlacement(stage_id=actor.stage_id, actor_index=1, node_id="node-a"),
        ),
        generation=1,
    )

    with pytest.raises(ValueError, match="cumulative actor placements"):
        graph.validate_allocation(allocation)


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
        actor_placements=(),
        generation=9,
    )

    assert QueryAllocation.from_dict(allocation.to_dict()) == allocation
    with pytest.raises(ValueError, match="sum of node_allocations"):
        QueryAllocation(
            resources=resources,
            node_allocations=allocation.node_allocations[:1],
            actor_placements=(),
            generation=9,
        )
