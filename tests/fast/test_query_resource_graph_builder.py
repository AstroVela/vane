# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

from duckdb.runners.ray.cluster_resource_coordinator import (
    ClusterQueryResourceCoordinator,
    NodeCapacity,
)
from duckdb.runners.ray.query_resource_graph import ResourceVector
from duckdb.runners.ray.query_resource_graph_builder import (
    build_query_demand,
    build_query_resource_graph,
    native_fragment_unit_id_for_fragment,
    native_fragment_unit_id_for_node,
    udf_unit_id_for_node,
)

GIB = 1024**3
MIB = 1024**2


def _single_node_cluster(resources: ResourceVector) -> tuple[NodeCapacity, ...]:
    return (NodeCapacity("node-a", resources),)


def _metadata():
    query_id = "query-7"
    return {
        "query_id": query_id,
        "nodes": [
            {
                "node_id": "1",
                "node_name": "ScanSource",
                "input_node_ids": [],
                "is_sink": False,
                "is_materialization_barrier": False,
                "materialized_input_node_ids": [],
                "num_partitions": 36,
                "udf_payload": None,
            },
            {
                "node_id": "2",
                "node_name": "StreamingUDF",
                "input_node_ids": ["1"],
                "is_sink": False,
                "is_materialization_barrier": False,
                "materialized_input_node_ids": [],
                "num_partitions": 36,
                "udf_payload": {
                    "execution_backend": "ray_task",
                    "resource_unit_id": udf_unit_id_for_node(query_id, "2"),
                    "cpus": 1.0,
                    "gpus": 0.0,
                    "memory_bytes": 1536 * MIB,
                    "udf_output_target_max_bytes": 64 * MIB,
                    "udf_task_input_max_bytes": 128 * MIB,
                },
            },
            {
                "node_id": "3",
                "node_name": "StreamingUDF",
                "input_node_ids": ["2"],
                "is_sink": False,
                "is_materialization_barrier": False,
                "materialized_input_node_ids": [],
                "num_partitions": 1,
                "udf_payload": {
                    "execution_backend": "ray_actor",
                    "resource_unit_id": udf_unit_id_for_node(query_id, "3"),
                    "cpus": 1.0,
                    "gpus": 1.0,
                    "memory_bytes": 3 * GIB,
                    "actor_pool_size": 1,
                    "udf_output_target_max_bytes": 32 * MIB,
                    "udf_task_input_max_bytes": 128 * MIB,
                },
            },
            {
                "node_id": "4",
                "node_name": "CopyFinish",
                "input_node_ids": ["3"],
                "is_sink": True,
                "is_materialization_barrier": False,
                "materialized_input_node_ids": [],
                "num_partitions": 1,
                "udf_payload": None,
            },
        ],
        "terminal_node_ids": ["4"],
    }


def test_builder_registers_complete_pipeline_and_nested_resource_units_before_execution():
    graph = build_query_resource_graph(_metadata(), env={})

    assert graph.query_id == "query-7"
    assert graph.plan_digest.startswith("sha256:")
    assert len(graph.units) == 6
    assert graph.topological_unit_ids() == (
        native_fragment_unit_id_for_node("query-7", "1"),
        native_fragment_unit_id_for_node("query-7", "2"),
        udf_unit_id_for_node("query-7", "2"),
        native_fragment_unit_id_for_node("query-7", "3"),
        udf_unit_id_for_node("query-7", "3"),
        native_fragment_unit_id_for_node("query-7", "4"),
    )
    assert graph.terminal_unit_ids == (native_fragment_unit_id_for_node("query-7", "4"),)


def test_builder_delegates_native_process_resources_and_counts_each_ray_process():
    graph = build_query_resource_graph(_metadata(), env={})
    cpu_udf = graph.unit_by_id(udf_unit_id_for_node("query-7", "2"))
    gpu_udf = graph.unit_by_id(udf_unit_id_for_node("query-7", "3"))
    scan = graph.unit_by_id(native_fragment_unit_id_for_node("query-7", "1"))
    cpu_udf_parent = graph.unit_by_id(native_fragment_unit_id_for_node("query-7", "2"))
    gpu_udf_parent = graph.unit_by_id(native_fragment_unit_id_for_node("query-7", "3"))
    native_sink = graph.unit_by_id(native_fragment_unit_id_for_node("query-7", "4"))

    assert scan.backend == "ray_worker"
    assert scan.per_task == ResourceVector()
    assert cpu_udf_parent.per_task == ResourceVector()
    assert gpu_udf_parent.per_task == ResourceVector()
    assert native_sink.per_task == ResourceVector()
    assert cpu_udf.backend == "ray_task"
    assert cpu_udf.per_task == ResourceVector(cpu=1, heap_bytes=1536 * MIB, object_store_bytes=128 * MIB)
    assert cpu_udf.max_concurrency is None
    assert gpu_udf.backend == "ray_actor"
    assert gpu_udf.resident_per_actor == ResourceVector(cpu=1, gpu=1, heap_bytes=3 * GIB)
    assert gpu_udf.per_task == ResourceVector(object_store_bytes=128 * MIB)
    assert gpu_udf.max_concurrency is None
    assert gpu_udf.actor_pool_size == 1
    assert gpu_udf.actor_prefetch_depth == 2


def test_builder_configures_stateless_actor_prefetch_and_disables_it_for_stateful_udfs():
    configured = build_query_resource_graph(
        _metadata(),
        env={"VANE_RAY_ACTOR_PREFETCH_DEPTH": "3"},
    )
    assert configured.unit_by_id(udf_unit_id_for_node("query-7", "3")).actor_prefetch_depth == 3

    stateful_metadata = _metadata()
    stateful_metadata["nodes"][2]["udf_payload"]["stateful"] = True
    stateful = build_query_resource_graph(
        stateful_metadata,
        env={"VANE_RAY_ACTOR_PREFETCH_DEPTH": "3"},
    )
    assert stateful.unit_by_id(udf_unit_id_for_node("query-7", "3")).actor_prefetch_depth == 1

    with pytest.raises(ValueError, match="VANE_RAY_ACTOR_PREFETCH_DEPTH"):
        build_query_resource_graph(
            _metadata(),
            env={"VANE_RAY_ACTOR_PREFETCH_DEPTH": "0"},
        )


def test_builder_uses_two_logical_output_blocks_for_all_streaming_units():
    graph = build_query_resource_graph(_metadata(), env={})
    cpu_udf = graph.unit_by_id(udf_unit_id_for_node("query-7", "2"))
    gpu_udf = graph.unit_by_id(udf_unit_id_for_node("query-7", "3"))

    assert cpu_udf.target_output_block_bytes == 64 * MIB
    assert cpu_udf.generator_buffer_blocks == 2
    assert cpu_udf.output_window_bytes == 128 * MIB
    assert gpu_udf.target_output_block_bytes == 32 * MIB
    assert gpu_udf.generator_buffer_blocks == 2


def test_builder_keeps_generator_buffer_independent_of_downstream_compute_batch():
    metadata = _metadata()
    producer = metadata["nodes"][1]["udf_payload"]
    consumer = metadata["nodes"][2]["udf_payload"]
    producer["udf_output_target_max_bytes"] = 1024
    consumer["udf_task_input_max_bytes"] = 64 * 1024

    graph = build_query_resource_graph(metadata, env={})
    cpu_udf = graph.unit_by_id(udf_unit_id_for_node("query-7", "2"))

    assert cpu_udf.target_output_block_bytes == 1024
    # The producer only shapes Ray's streaming-generator buffer. Downstream
    # accumulation is represented by exact managed ObjectRefs, not by inflating
    # every upstream task's pending-output estimate.
    assert cpu_udf.generator_buffer_blocks == 2
    assert cpu_udf.output_window_bytes == 2 * 1024


def test_builder_leaves_udf_heap_unreserved_when_memory_is_not_declared():
    metadata = _metadata()
    del metadata["nodes"][1]["udf_payload"]["memory_bytes"]
    del metadata["nodes"][2]["udf_payload"]["memory_bytes"]

    graph = build_query_resource_graph(metadata, env={})

    assert graph.unit_by_id(udf_unit_id_for_node("query-7", "2")).per_task.heap_bytes == 0
    assert graph.unit_by_id(udf_unit_id_for_node("query-7", "3")).resident_per_actor.heap_bytes == 0

    cluster = ResourceVector(cpu=64, gpu=4, heap_bytes=64 * GIB, object_store_bytes=64 * GIB)
    demand = build_query_demand(graph, _single_node_cluster(cluster))
    assert demand.minimum.heap_bytes == 0
    assert demand.desired.heap_bytes == 0


def test_builder_maps_materialized_physical_input_to_its_output_resource_unit():
    metadata = _metadata()
    materializer = metadata["nodes"][3]
    materializer["node_name"] = "OrderBy"
    materializer["is_materialization_barrier"] = True
    materializer["materialized_input_node_ids"] = ["3"]

    graph = build_query_resource_graph(metadata, env={})

    assert len(graph.materialization_barriers) == 1
    barrier = graph.materialization_barriers[0]
    assert barrier.materialized_input_unit_ids == (udf_unit_id_for_node("query-7", "3"),)


def test_builder_requires_declared_heap_to_be_positive():
    with pytest.raises(ValueError, match="memory_bytes"):
        metadata = _metadata()
        metadata["nodes"][1]["udf_payload"]["memory_bytes"] = 0
        build_query_resource_graph(metadata, env={})


def test_builder_rejects_missing_or_mismatched_preannotated_udf_resource_unit_identity():
    missing = _metadata()
    missing["nodes"][1]["udf_payload"].pop("resource_unit_id")
    with pytest.raises(ValueError, match="missing pre-registered resource_unit_id"):
        build_query_resource_graph(missing, env={})

    mismatch = _metadata()
    mismatch["nodes"][1]["udf_payload"]["resource_unit_id"] = "resource:legacy:operator"
    with pytest.raises(ValueError, match="resource_unit_id mismatch"):
        build_query_resource_graph(mismatch, env={})


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda payload: payload.pop("query_id"), "missing required fields"),
        (lambda payload: payload.update({"legacy_operators": []}), "unknown fields"),
        (lambda payload: payload.update({"terminal_node_ids": ["missing"]}), "terminal node"),
        (lambda payload: payload["nodes"][1].update({"input_node_ids": ["missing"]}), "input node"),
        (
            lambda payload: payload["nodes"][1].update({"is_materialization_barrier": True}),
            "if and only if",
        ),
        (
            lambda payload: payload["nodes"][1].update({"materialized_input_node_ids": ["1", "1"]}),
            "duplicate materialized",
        ),
        (
            lambda payload: payload["nodes"][1].update(
                {
                    "is_materialization_barrier": True,
                    "materialized_input_node_ids": ["4"],
                }
            ),
            "not a direct input",
        ),
    ],
)
def test_builder_rejects_incomplete_or_legacy_metadata(mutation, message):
    metadata = _metadata()
    mutation(metadata)

    with pytest.raises(ValueError, match=message):
        build_query_resource_graph(metadata, env={})


def test_plan_digest_is_stable_for_node_order_but_changes_with_resources():
    first = _metadata()
    reordered = _metadata()
    reordered["nodes"] = list(reversed(reordered["nodes"]))
    changed = _metadata()
    changed["nodes"][1]["udf_payload"]["memory_bytes"] += 1

    graph_a = build_query_resource_graph(first, env={})
    graph_b = build_query_resource_graph(reordered, env={})
    graph_c = build_query_resource_graph(changed, env={})

    assert graph_a.plan_digest == graph_b.plan_digest
    assert graph_a.plan_digest != graph_c.plan_digest


def test_query_demand_hard_reserves_task_and_actor_shapes_but_soft_accounts_actor_pool_size():
    graph = build_query_resource_graph(_metadata(), env={})
    cluster = ResourceVector(cpu=64, gpu=4, heap_bytes=64 * GIB, object_store_bytes=64 * GIB)

    demand = build_query_demand(graph, _single_node_cluster(cluster))

    assert demand.query_id == graph.query_id
    assert demand.desired == ResourceVector(
        cpu=64,
        gpu=1,
        heap_bytes=64 * GIB,
        object_store_bytes=64 * GIB,
    )
    assert demand.placement_bundles == (ResourceVector(cpu=1, gpu=1, heap_bytes=3 * GIB),)
    assert demand.minimum == ResourceVector(
        cpu=1,
        gpu=1,
        heap_bytes=3 * GIB,
    )


def test_pure_native_query_demands_only_a_soft_object_store_budget():
    metadata = _metadata()
    for node in metadata["nodes"]:
        node["udf_payload"] = None
    graph = build_query_resource_graph(metadata, env={})
    cluster = ResourceVector(cpu=64, gpu=4, heap_bytes=64 * GIB, object_store_bytes=8 * GIB)

    demand = build_query_demand(graph, _single_node_cluster(cluster))

    assert all(unit.backend == "ray_worker" for unit in graph.units)
    assert all(unit.per_task == ResourceVector() for unit in graph.units)
    assert demand.minimum == ResourceVector()
    assert demand.placement_bundles == ()
    assert demand.desired == ResourceVector(object_store_bytes=8 * GIB)


def test_query_demand_treats_pipeline_windows_as_soft_budget_regression_issue_38():
    metadata = _metadata()
    metadata["nodes"][2]["udf_payload"]["actor_pool_size"] = 2
    graph = build_query_resource_graph(metadata, env={})
    cluster = ResourceVector(
        cpu=64,
        gpu=4,
        heap_bytes=64 * GIB,
        object_store_bytes=512 * MIB,
    )

    demand = build_query_demand(graph, _single_node_cluster(cluster))

    assert demand.minimum.object_store_bytes == 0
    assert demand.desired.object_store_bytes == 512 * MIB
    assert all(bundle.object_store_bytes == 0 for bundle in demand.placement_bundles)


def test_query_demand_reserves_gpu_ray_task_as_an_indivisible_placement_bundle():
    metadata = _metadata()
    metadata["nodes"][1]["udf_payload"]["gpus"] = 1.0
    graph = build_query_resource_graph(metadata, env={})
    cluster = ResourceVector(cpu=64, gpu=4, heap_bytes=64 * GIB, object_store_bytes=64 * GIB)

    demand = build_query_demand(graph, _single_node_cluster(cluster))

    assert demand.minimum.gpu == 1
    assert demand.desired.gpu == 4
    assert demand.placement_bundles[0].gpu == 1


def test_query_demand_uses_real_node_envelopes_for_heterogeneous_ray_tasks():
    metadata = _metadata()
    cpu_payload = metadata["nodes"][1]["udf_payload"]
    cpu_payload["cpus"] = 4.0
    cpu_payload.pop("memory_bytes")
    gpu_payload = metadata["nodes"][2]["udf_payload"]
    gpu_payload["execution_backend"] = "ray_task"
    gpu_payload["cpus"] = 1.0
    gpu_payload["gpus"] = 1.0
    gpu_payload.pop("memory_bytes")
    gpu_payload.pop("actor_pool_size")
    graph = build_query_resource_graph(metadata, env={})
    nodes = (
        NodeCapacity(
            "cpu-node",
            ResourceVector(cpu=4, object_store_bytes=GIB),
        ),
        NodeCapacity(
            "gpu-node",
            ResourceVector(cpu=1, gpu=1, object_store_bytes=GIB),
        ),
    )

    demand = build_query_demand(graph, nodes)

    assert demand.placement_bundles == (
        ResourceVector(cpu=1, gpu=1),
        ResourceVector(cpu=4),
    )
    assert demand.minimum == ResourceVector(cpu=5, gpu=1)
    coordinator = ClusterQueryResourceCoordinator(nodes)
    allocation = coordinator.register_query(demand, now=0)
    assert coordinator.query_state(graph.query_id, allocation.generation) == "RUNNING"
    graph.validate_allocation(allocation)


def test_query_demand_collapses_heterogeneous_task_shapes_that_share_one_node():
    metadata = _metadata()
    cpu_payload = metadata["nodes"][1]["udf_payload"]
    cpu_payload["cpus"] = 4.0
    cpu_payload.pop("memory_bytes")
    gpu_payload = metadata["nodes"][2]["udf_payload"]
    gpu_payload["execution_backend"] = "ray_task"
    gpu_payload["cpus"] = 1.0
    gpu_payload["gpus"] = 1.0
    gpu_payload.pop("memory_bytes")
    gpu_payload.pop("actor_pool_size")
    graph = build_query_resource_graph(metadata, env={})
    nodes = (
        NodeCapacity(
            "combined-node",
            ResourceVector(cpu=4, gpu=1, object_store_bytes=GIB),
        ),
    )

    demand = build_query_demand(graph, nodes)

    assert demand.placement_bundles == (ResourceVector(cpu=4, gpu=1),)
    assert demand.minimum == ResourceVector(cpu=4, gpu=1)


def test_capacity_topology_change_rebuilds_placement_capability_envelopes():
    metadata = _metadata()
    cpu_payload = metadata["nodes"][1]["udf_payload"]
    cpu_payload["cpus"] = 4.0
    cpu_payload.pop("memory_bytes")
    gpu_payload = metadata["nodes"][2]["udf_payload"]
    gpu_payload["execution_backend"] = "ray_task"
    gpu_payload["cpus"] = 1.0
    gpu_payload["gpus"] = 1.0
    gpu_payload.pop("memory_bytes")
    gpu_payload.pop("actor_pool_size")
    graph = build_query_resource_graph(metadata, env={})
    combined = (
        NodeCapacity(
            "combined-node",
            ResourceVector(cpu=4, gpu=1, object_store_bytes=2 * GIB),
        ),
    )
    split = (
        NodeCapacity("cpu-node", ResourceVector(cpu=4, object_store_bytes=GIB)),
        NodeCapacity("gpu-node", ResourceVector(cpu=1, gpu=1, object_store_bytes=GIB)),
    )
    coordinator = ClusterQueryResourceCoordinator(combined)
    allocation = coordinator.register_query(build_query_demand(graph, combined), now=0)

    coordinator.update_node_capacities(split, now=1)
    pending = coordinator.snapshot()["queries"][graph.query_id]
    assert pending["state"] == "PENDING_RESOURCES"

    allocation = coordinator.refresh_queries(
        observed_usage_by_query={graph.query_id: ResourceVector()},
        generations={graph.query_id: pending["allocation"]["generation"]},
        demands_by_query={graph.query_id: build_query_demand(graph, split)},
        now=1,
    )[graph.query_id]

    assert coordinator.query_state(graph.query_id, allocation.generation) == "RUNNING"
    graph.validate_allocation(allocation)


def test_query_demand_does_not_turn_fixed_actor_pool_size_into_hard_admission():
    metadata = _metadata()
    actor_payload = metadata["nodes"][2]["udf_payload"]
    actor_payload["actor_pool_size"] = 3
    actor_payload["gpus"] = 0.25
    graph = build_query_resource_graph(metadata, env={})
    cluster = ResourceVector(cpu=64, gpu=4, heap_bytes=64 * GIB, object_store_bytes=64 * GIB)

    demand = build_query_demand(graph, _single_node_cluster(cluster))
    actor_unit = graph.unit_by_id(udf_unit_id_for_node("query-7", "3"))

    assert actor_unit.actor_pool_size == 3
    assert actor_unit.resident_per_actor.gpu == 0.25
    assert demand.minimum.gpu == 0.25
    assert demand.desired.gpu == 0.75


@pytest.mark.parametrize("gpu_node_count", [1, 2])
def test_actor_shape_stays_pending_until_one_concrete_node_can_run_it(gpu_node_count):
    metadata = _metadata()
    metadata["nodes"][1]["udf_payload"] = None
    actor_payload = metadata["nodes"][2]["udf_payload"]
    actor_payload["cpus"] = 1.0
    actor_payload["gpus"] = 2.0
    actor_payload.pop("memory_bytes")
    graph = build_query_resource_graph(metadata, env={})
    split = tuple(
        NodeCapacity(f"gpu-{index}", ResourceVector(cpu=4, gpu=1, object_store_bytes=GIB))
        for index in range(gpu_node_count)
    )

    demand = build_query_demand(graph, split)

    assert demand.placement_bundles == (ResourceVector(cpu=1, gpu=2),)
    assert demand.minimum == ResourceVector(cpu=1, gpu=2)
    assert demand.desired.gpu == 2
    coordinator = ClusterQueryResourceCoordinator(split)
    allocation = coordinator.register_query(demand, now=0)
    assert coordinator.query_state(graph.query_id, allocation.generation) == "PENDING_RESOURCES"

    coordinator.update_node_capacities(
        (NodeCapacity("gpu-capable", ResourceVector(cpu=4, gpu=2, object_store_bytes=2 * GIB)),),
        now=1,
    )
    snapshot = coordinator.snapshot()["queries"][graph.query_id]
    assert snapshot["state"] == "RUNNING"


def test_fragment_identity_maps_directly_to_pre_registered_native_fragment_unit():
    assert native_fragment_unit_id_for_fragment("query-7", "query-7:node:12") == native_fragment_unit_id_for_node(
        "query-7", "12"
    )
    with pytest.raises(ValueError, match="does not belong to query"):
        native_fragment_unit_id_for_fragment("query-7", "other:node:12")
    with pytest.raises(ValueError, match="invalid native fragment_id"):
        native_fragment_unit_id_for_fragment("query-7", "query-7:task:12")
