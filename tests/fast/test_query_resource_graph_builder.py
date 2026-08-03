# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

from duckdb.runners.ray.cluster_resource_coordinator import ActorResourceBundle
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
                "num_partitions": 36,
                "udf_payload": None,
            },
            {
                "node_id": "2",
                "node_name": "StreamingUDF",
                "input_node_ids": ["1"],
                "is_sink": False,
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


def test_builder_sizes_upstream_retention_window_for_downstream_compute_batch():
    metadata = _metadata()
    producer = metadata["nodes"][1]["udf_payload"]
    consumer = metadata["nodes"][2]["udf_payload"]
    producer["udf_output_target_max_bytes"] = 1024
    consumer["udf_task_input_max_bytes"] = 64 * 1024

    graph = build_query_resource_graph(metadata, env={})
    cpu_udf = graph.unit_by_id(udf_unit_id_for_node("query-7", "2"))

    assert cpu_udf.target_output_block_bytes == 1024
    assert cpu_udf.generator_buffer_blocks == 64
    assert cpu_udf.output_window_bytes == 64 * 1024


def test_builder_leaves_udf_heap_unreserved_when_memory_is_not_declared():
    metadata = _metadata()
    del metadata["nodes"][1]["udf_payload"]["memory_bytes"]
    del metadata["nodes"][2]["udf_payload"]["memory_bytes"]

    graph = build_query_resource_graph(metadata, env={})

    assert graph.unit_by_id(udf_unit_id_for_node("query-7", "2")).per_task.heap_bytes == 0
    assert graph.unit_by_id(udf_unit_id_for_node("query-7", "3")).resident_per_actor.heap_bytes == 0

    cluster = ResourceVector(cpu=64, gpu=4, heap_bytes=64 * GIB, object_store_bytes=64 * GIB)
    demand = build_query_demand(graph, cluster)
    assert demand.minimum.heap_bytes == 0
    assert demand.desired.heap_bytes == 0


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


def test_query_demand_reserves_only_remote_process_bundles():
    graph = build_query_resource_graph(_metadata(), env={})
    cluster = ResourceVector(cpu=64, gpu=4, heap_bytes=64 * GIB, object_store_bytes=64 * GIB)

    demand = build_query_demand(graph, cluster)

    assert demand.query_id == graph.query_id
    assert demand.desired == ResourceVector(
        cpu=64,
        gpu=1,
        heap_bytes=64 * GIB,
        object_store_bytes=64 * GIB,
    )
    assert demand.actor_bundles == (
        ActorResourceBundle(
            resource_unit_id="resource:query-7:udf:node:3",
            actor_index=0,
            resources=ResourceVector(
                cpu=1,
                gpu=1,
                heap_bytes=3 * GIB,
            ),
        ),
    )
    assert demand.task_bundles == (ResourceVector(cpu=1, heap_bytes=1536 * MIB),)
    assert demand.minimum == ResourceVector(
        cpu=2,
        gpu=1,
        heap_bytes=3 * GIB + 1536 * MIB,
    )


def test_pure_native_query_demands_only_a_soft_object_store_budget():
    metadata = _metadata()
    for node in metadata["nodes"]:
        node["udf_payload"] = None
    graph = build_query_resource_graph(metadata, env={})
    cluster = ResourceVector(cpu=64, gpu=4, heap_bytes=64 * GIB, object_store_bytes=8 * GIB)

    demand = build_query_demand(graph, cluster)

    assert all(unit.backend == "ray_worker" for unit in graph.units)
    assert all(unit.per_task == ResourceVector() for unit in graph.units)
    assert demand.minimum == ResourceVector()
    assert demand.task_bundles == ()
    assert demand.actor_bundles == ()
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

    demand = build_query_demand(graph, cluster)

    assert demand.minimum.object_store_bytes == 0
    assert demand.desired.object_store_bytes == 512 * MIB
    assert all(bundle.resources.object_store_bytes == 0 for bundle in demand.actor_bundles)
    assert all(bundle.object_store_bytes == 0 for bundle in demand.task_bundles)


def test_query_demand_reserves_gpu_ray_task_as_an_indivisible_task_bundle():
    metadata = _metadata()
    metadata["nodes"][1]["udf_payload"]["gpus"] = 1.0
    graph = build_query_resource_graph(metadata, env={})
    cluster = ResourceVector(cpu=64, gpu=4, heap_bytes=64 * GIB, object_store_bytes=64 * GIB)

    demand = build_query_demand(graph, cluster)

    assert demand.minimum.gpu == 2
    assert demand.desired.gpu == 2
    assert demand.actor_bundles[0].resources.gpu == 1
    assert demand.task_bundles[0].gpu == 1


def test_query_demand_reserves_every_fractional_gpu_actor_in_fixed_pool():
    metadata = _metadata()
    actor_payload = metadata["nodes"][2]["udf_payload"]
    actor_payload["actor_pool_size"] = 3
    actor_payload["gpus"] = 0.25
    graph = build_query_resource_graph(metadata, env={})
    cluster = ResourceVector(cpu=64, gpu=4, heap_bytes=64 * GIB, object_store_bytes=64 * GIB)

    demand = build_query_demand(graph, cluster)
    actor_unit = graph.unit_by_id(udf_unit_id_for_node("query-7", "3"))

    assert actor_unit.actor_pool_size == 3
    assert [bundle.actor_index for bundle in demand.actor_bundles] == [0, 1, 2]
    assert all(bundle.resources.gpu == 0.25 for bundle in demand.actor_bundles)
    assert demand.minimum.gpu == 0.75
    assert demand.desired.gpu == demand.minimum.gpu


def test_fragment_identity_maps_directly_to_pre_registered_native_fragment_unit():
    assert native_fragment_unit_id_for_fragment("query-7", "query-7:node:12") == native_fragment_unit_id_for_node(
        "query-7", "12"
    )
    with pytest.raises(ValueError, match="does not belong to query"):
        native_fragment_unit_id_for_fragment("query-7", "other:node:12")
    with pytest.raises(ValueError, match="invalid native fragment_id"):
        native_fragment_unit_id_for_fragment("query-7", "query-7:task:12")
