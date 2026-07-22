# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

from vane.runners.ray.cluster_resource_coordinator import ActorResourceBundle
from vane.runners.ray.query_execution_graph import ResourceVector
from vane.runners.ray.query_graph_builder import (
    build_query_demand,
    build_query_execution_graph,
    fte_stage_id_for_fragment,
    fte_stage_id_for_node,
    udf_stage_id_for_node,
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
                    "stage_id": udf_stage_id_for_node(query_id, "2"),
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
                    "stage_id": udf_stage_id_for_node(query_id, "3"),
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


def test_builder_registers_complete_pipeline_and_nested_udf_stages_before_execution():
    graph = build_query_execution_graph(_metadata(), env={})

    assert graph.query_id == "query-7"
    assert graph.plan_digest.startswith("sha256:")
    assert len(graph.stages) == 6
    assert graph.topological_stage_ids() == (
        fte_stage_id_for_node("query-7", "1"),
        fte_stage_id_for_node("query-7", "2"),
        udf_stage_id_for_node("query-7", "2"),
        fte_stage_id_for_node("query-7", "3"),
        udf_stage_id_for_node("query-7", "3"),
        fte_stage_id_for_node("query-7", "4"),
    )
    assert graph.terminal_stage_ids == (fte_stage_id_for_node("query-7", "4"),)


def test_builder_counts_each_nested_ray_process_and_never_uses_zero_heap():
    graph = build_query_execution_graph(_metadata(), env={})
    cpu_udf = graph.stage_by_id(udf_stage_id_for_node("query-7", "2"))
    gpu_udf = graph.stage_by_id(udf_stage_id_for_node("query-7", "3"))
    scan = graph.stage_by_id(fte_stage_id_for_node("query-7", "1"))
    cpu_udf_parent = graph.stage_by_id(fte_stage_id_for_node("query-7", "2"))
    gpu_udf_parent = graph.stage_by_id(fte_stage_id_for_node("query-7", "3"))
    native_sink = graph.stage_by_id(fte_stage_id_for_node("query-7", "4"))

    assert scan.backend == "ray_worker"
    # Node 1 is the task-producing feeder immediately before the remote UDF.
    assert scan.per_task == ResourceVector(cpu=1, heap_bytes=512 * MIB)
    assert cpu_udf_parent.per_task == ResourceVector(cpu=1, heap_bytes=512 * MIB)
    assert gpu_udf_parent.per_task == ResourceVector(cpu=1, heap_bytes=512 * MIB)
    assert native_sink.per_task == ResourceVector(cpu=1, heap_bytes=2 * GIB)
    assert cpu_udf.backend == "ray_task"
    assert cpu_udf.per_task == ResourceVector(cpu=1, heap_bytes=1536 * MIB, object_store_bytes=128 * MIB)
    assert cpu_udf.max_concurrency is None
    assert gpu_udf.backend == "ray_actor"
    assert gpu_udf.resident_per_actor == ResourceVector(cpu=1, gpu=1, heap_bytes=3 * GIB)
    assert gpu_udf.per_task == ResourceVector(object_store_bytes=128 * MIB)
    assert gpu_udf.max_concurrency is None
    assert gpu_udf.actor_min_size == 1
    assert gpu_udf.actor_max_size == 1
    assert gpu_udf.actor_prefetch_depth == 2


def test_builder_configures_stateless_actor_prefetch_and_disables_it_for_stateful_udfs():
    configured = build_query_execution_graph(
        _metadata(),
        env={"VANE_RAY_ACTOR_PREFETCH_DEPTH": "3"},
    )
    assert configured.stage_by_id(udf_stage_id_for_node("query-7", "3")).actor_prefetch_depth == 3

    stateful_metadata = _metadata()
    stateful_metadata["nodes"][2]["udf_payload"]["stateful"] = True
    stateful = build_query_execution_graph(
        stateful_metadata,
        env={"VANE_RAY_ACTOR_PREFETCH_DEPTH": "3"},
    )
    assert stateful.stage_by_id(udf_stage_id_for_node("query-7", "3")).actor_prefetch_depth == 1

    with pytest.raises(ValueError, match="VANE_RAY_ACTOR_PREFETCH_DEPTH"):
        build_query_execution_graph(
            _metadata(),
            env={"VANE_RAY_ACTOR_PREFETCH_DEPTH": "0"},
        )


def test_builder_uses_two_logical_output_blocks_for_all_streaming_stages():
    graph = build_query_execution_graph(_metadata(), env={})
    cpu_udf = graph.stage_by_id(udf_stage_id_for_node("query-7", "2"))
    gpu_udf = graph.stage_by_id(udf_stage_id_for_node("query-7", "3"))

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

    graph = build_query_execution_graph(metadata, env={})
    cpu_udf = graph.stage_by_id(udf_stage_id_for_node("query-7", "2"))

    assert cpu_udf.target_output_block_bytes == 1024
    assert cpu_udf.generator_buffer_blocks == 64
    assert cpu_udf.output_window_bytes == 64 * 1024


def test_builder_defaults_are_new_positive_production_limits_not_host_memory():
    metadata = _metadata()
    del metadata["nodes"][1]["udf_payload"]["memory_bytes"]
    del metadata["nodes"][2]["udf_payload"]["memory_bytes"]

    graph = build_query_execution_graph(metadata, env={})

    assert graph.stage_by_id(udf_stage_id_for_node("query-7", "2")).per_task.heap_bytes == 2 * GIB
    assert graph.stage_by_id(udf_stage_id_for_node("query-7", "3")).resident_per_actor.heap_bytes == 4 * GIB


def test_builder_accepts_only_positive_new_resource_configuration():
    with pytest.raises(ValueError, match="VANE_FTE_TASK_HEAP_BYTES"):
        build_query_execution_graph(_metadata(), env={"VANE_FTE_TASK_HEAP_BYTES": "0"})
    with pytest.raises(ValueError, match="VANE_FTE_UDF_DRIVER_HEAP_BYTES"):
        build_query_execution_graph(
            _metadata(),
            env={"VANE_FTE_UDF_DRIVER_HEAP_BYTES": "0"},
        )
    with pytest.raises(ValueError, match="memory_bytes"):
        metadata = _metadata()
        metadata["nodes"][1]["udf_payload"]["memory_bytes"] = 0
        build_query_execution_graph(metadata, env={})


def test_builder_rejects_missing_or_mismatched_preannotated_udf_stage_identity():
    missing = _metadata()
    missing["nodes"][1]["udf_payload"].pop("stage_id")
    with pytest.raises(ValueError, match="missing pre-registered stage_id"):
        build_query_execution_graph(missing, env={})

    mismatch = _metadata()
    mismatch["nodes"][1]["udf_payload"]["stage_id"] = "stage:legacy:operator"
    with pytest.raises(ValueError, match="stage_id mismatch"):
        build_query_execution_graph(mismatch, env={})


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
        build_query_execution_graph(metadata, env={})


def test_plan_digest_is_stable_for_node_order_but_changes_with_resources():
    first = _metadata()
    reordered = _metadata()
    reordered["nodes"] = list(reversed(reordered["nodes"]))
    changed = _metadata()
    changed["nodes"][1]["udf_payload"]["memory_bytes"] += 1

    graph_a = build_query_execution_graph(first, env={})
    graph_b = build_query_execution_graph(reordered, env={})
    graph_c = build_query_execution_graph(changed, env={})

    assert graph_a.plan_digest == graph_b.plan_digest
    assert graph_a.plan_digest != graph_c.plan_digest


def test_query_demand_reserves_actor_minima_and_downstream_fte_progress_slot():
    graph = build_query_execution_graph(_metadata(), env={})
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
            stage_id="stage:query-7:node:3:udf",
            actor_index=0,
            resources=ResourceVector(
                cpu=1,
                gpu=1,
                heap_bytes=3 * GIB,
                object_store_bytes=192 * MIB,
            ),
        ),
    )
    assert demand.task_bundles == (
        ResourceVector(cpu=1, heap_bytes=1536 * MIB, object_store_bytes=256 * MIB),
        ResourceVector(cpu=1, heap_bytes=2 * GIB, object_store_bytes=256 * MIB),
        ResourceVector(cpu=1, heap_bytes=2 * GIB, object_store_bytes=256 * MIB),
    )
    assert demand.minimum == ResourceVector(
        cpu=4,
        gpu=1,
        heap_bytes=3 * GIB + 4 * GIB + 1536 * MIB,
        object_store_bytes=(192 + 256 + 256 + 256) * MIB,
    )


def test_query_demand_reserves_gpu_ray_task_as_an_indivisible_task_bundle():
    metadata = _metadata()
    metadata["nodes"][1]["udf_payload"]["gpus"] = 1.0
    graph = build_query_execution_graph(metadata, env={})
    cluster = ResourceVector(cpu=64, gpu=4, heap_bytes=64 * GIB, object_store_bytes=64 * GIB)

    demand = build_query_demand(graph, cluster)

    assert demand.minimum.gpu == 2
    assert demand.desired.gpu == 2
    assert demand.actor_bundles[0].resources.gpu == 1
    assert demand.task_bundles[0].gpu == 1


def test_fragment_identity_maps_directly_to_pre_registered_fte_stage():
    assert fte_stage_id_for_fragment("query-7", "query-7:node:12") == fte_stage_id_for_node("query-7", "12")
    with pytest.raises(ValueError, match="does not belong to query"):
        fte_stage_id_for_fragment("query-7", "other:node:12")
    with pytest.raises(ValueError, match="invalid FTE fragment_id"):
        fte_stage_id_for_fragment("query-7", "query-7:task:12")
