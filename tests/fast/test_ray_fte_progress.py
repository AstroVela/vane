# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io

import pytest

import vane
import vane.runners.progress as shared_progress_mod
import vane.runners.ray.progress as progress_mod
from vane.runners.progress import (
    LocalProgressSnapshotStore,
    _format_bytes,
    _format_count,
    build_local_progress_snapshot,
    progress_enabled,
)
from vane.runners.ray.fte import FteTaskExecution
from vane.runners.ray.progress import ProgressRenderer, build_progress_snapshot, format_progress_snapshot
from vane.runners.ray.query_execution_graph import (
    NodeResourceAllocation,
    QueryAllocation,
    QueryExecutionGraph,
    ResourceVector,
    StageResourceSpec,
)
from vane.runners.ray.query_resource_manager import TaskRequest
from vane.runners.ray.query_resource_runtime import (
    clear_query_resource_managers,
    register_query_graph,
)


def _topology(
    *pipelines: tuple[int, list[str], list[dict[str, str]] | None],
) -> dict[str, object]:
    return {
        "schema": "pipeline_topology",
        "pipelines": [
            {
                "pipeline_id": pipeline_id,
                "operators": operators,
                "operator_details": (operator_details if operator_details is not None else [{} for _ in operators]),
                "stage_ids": [],
            }
            for pipeline_id, operators, operator_details in pipelines
        ],
    }


def test_progress_snapshot_includes_query_resource_manager():
    clear_query_resource_managers()
    try:
        stage = StageResourceSpec(
            query_id="q-progress",
            stage_id="stage:q-progress:node:1:fte",
            physical_node_id="node:1:fte",
            stage_kind="fte",
            backend="ray_worker",
            input_stage_ids=(),
            per_task=ResourceVector(cpu=1, heap_bytes=1),
            target_output_block_bytes=1,
            generator_buffer_blocks=1,
            max_concurrency=1,
        )
        manager = register_query_graph(
            QueryExecutionGraph(
                query_id="q-progress",
                plan_digest="sha256:q-progress",
                stages=(stage,),
                terminal_stage_ids=(stage.stage_id,),
            ),
            QueryAllocation(
                resources=ResourceVector(cpu=1, heap_bytes=2, object_store_bytes=1),
                node_allocations=(
                    NodeResourceAllocation(
                        node_id="node-a",
                        resources=ResourceVector(cpu=1, heap_bytes=2, object_store_bytes=1),
                    ),
                ),
                actor_placements=(),
                generation=1,
            ),
        )
        manager.update_stage_state(stage.stage_id, runnable=True)
        grant = manager.try_acquire_task(TaskRequest("q-progress", stage.stage_id, "task-1", "attempt-1", "node-a"))
        assert grant.granted

        snapshot = build_progress_snapshot(
            {"queries": {"q-progress": {"query_id": "q-progress", "fragment_executions": {}}}},
            "q-progress",
        )

        assert snapshot["query"]["query_resource_manager"]["stages"][stage.stage_id]["active_task_count"] == 1
    finally:
        clear_query_resource_managers()


def test_progress_enabled_auto_supports_local_runner(monkeypatch):
    monkeypatch.delenv("VANE_PROGRESS", raising=False)

    assert progress_enabled("local")


def test_local_progress_snapshot_uses_common_progress_shape():
    snapshot = build_local_progress_snapshot(
        {
            "processed_input_rows": 50,
            "processed_input_bytes": 1024,
            "total_pipeline_tasks": 4,
            "queued_pipeline_tasks": 0,
            "running_pipeline_tasks": 2,
            "completed_pipeline_tasks": 2,
            "pipelines": [
                {
                    "pipeline_id": 2,
                    "operators": ["TABLE_SCAN", "INOUT_FUNCTION"],
                    "operator_details": [{}, {"pipeline_role": "sink", "udf_name": "udf"}],
                    "input_rows": 50,
                    "input_bytes": 1024,
                    "output_rows": 50,
                    "output_bytes": 1024,
                    "total_pipeline_tasks": 2,
                    "queued_pipeline_tasks": 0,
                    "running_pipeline_tasks": 0,
                    "completed_pipeline_tasks": 2,
                },
                {
                    "pipeline_id": 1,
                    "operators": ["INOUT_FUNCTION", "COPY_TO_FILE"],
                    "operator_details": [{"pipeline_role": "source", "udf_name": "udf"}, {}],
                    "input_rows": 50,
                    "input_bytes": 1024,
                    "output_rows": 50,
                    "output_bytes": 1024,
                    "total_pipeline_tasks": 2,
                    "queued_pipeline_tasks": 0,
                    "running_pipeline_tasks": 2,
                    "completed_pipeline_tasks": 0,
                },
            ],
        },
        "local-query",
        started_at=0,
    )

    assert snapshot["schema"] == "progress"
    assert snapshot["query_id"] == "local-query"
    assert "scheduled" not in snapshot
    assert "progress_percentage" not in snapshot
    assert "running_percentage" not in snapshot
    assert snapshot["total_pipeline_tasks"] == 4
    assert snapshot["running_pipeline_tasks"] == 2
    assert snapshot["processed_rows"] == 50
    assert set(snapshot["fragments"][0]) == {
        "id",
        "display_id",
        "name",
        "pending_partitions",
        "pipelines",
    }
    assert snapshot["fragments"][0]["pipelines"][0]["name"] == "ScanSource->StreamingUDF(sink)"


def test_progress_formatters_preserve_integer_trailing_zeroes():
    assert _format_count(720_000) == "720K"
    assert _format_count(72_000) == "72K"
    assert _format_count(7_200_000) == "7.2M"
    assert _format_bytes(720 * 1024) == "720KB"
    assert _format_bytes(72 * 1024) == "72KB"
    assert _format_bytes(7.2 * 1024 * 1024) == "7.2MB"


def test_local_progress_snapshot_store_finishes_with_final_stats():
    store = LocalProgressSnapshotStore("local-query", started_at=0)
    store.record(
        {
            "processed_input_rows": 1,
            "total_pipeline_tasks": 4,
            "queued_pipeline_tasks": 0,
            "running_pipeline_tasks": 3,
            "completed_pipeline_tasks": 1,
        }
    )
    store.finish(
        {
            "processed_input_rows": 2,
            "total_pipeline_tasks": 4,
            "queued_pipeline_tasks": 0,
            "running_pipeline_tasks": 0,
            "completed_pipeline_tasks": 4,
        }
    )

    snapshot = store.snapshot()

    assert snapshot["state"] == "FINISHED"
    assert snapshot["completed_pipeline_tasks"] == 4
    assert snapshot["processed_rows"] == 2


def test_progress_snapshot_uses_logical_partition_counts_and_selected_stats():
    registry = {
        "queries": {
            "query-a": {
                "query_id": "query-a",
                "fragment_executions": {
                    "fragment-b": {
                        "fragment_id": "fragment-b",
                        "fragment_execution_id": 2,
                        "pending_submission_count": 1,
                        "progress_topology": _topology(),
                        "partition_count": 4,
                        "running_count": 1,
                        "finished_count": 2,
                        "waiting_for_node_count": 1,
                        "waiting_for_execution_count": 0,
                        "execution_deferred_count": 0,
                        "no_more_partitions": True,
                        "source_node_ids": ["source"],
                        "dynamic_exchange_source_node_ids": ["source"],
                        "partitions": {
                            "0": {
                                "state": "FINISHED",
                                "selected_output_stats": {
                                    "processed_input_rows": 3,
                                    "physical_input_bytes": 11,
                                },
                            },
                            "1": {
                                "state": "FINISHED",
                                "selected_output_stats": {
                                    "processed_input_rows": 5,
                                    "internal_network_input_bytes": 13,
                                },
                            },
                            "2": {
                                "state": "RUNNING",
                                "running_count": 1,
                                "running_attempts": [{"task_stats": {}}],
                            },
                            "3": {"state": "SEALED", "waiting_for_execution": True},
                        },
                    }
                },
            }
        }
    }

    snapshot = build_progress_snapshot(registry, "query-a", started_at=0)

    assert snapshot["total_pipeline_tasks"] == 0
    assert snapshot["queued_pipeline_tasks"] == 0
    assert snapshot["running_pipeline_tasks"] == 0
    assert snapshot["completed_pipeline_tasks"] == 0
    assert snapshot["total_partitions"] == 4
    assert snapshot["queued_partitions"] == 1
    assert snapshot["running_partitions"] == 1
    assert snapshot["completed_partitions"] == 2
    assert snapshot["processed_rows"] == 8
    assert snapshot["processed_bytes"] == 24
    assert snapshot["fragments"][0]["display_id"] == "1"
    assert snapshot["fragments"][0]["pipelines"] == []


def test_progress_snapshot_uses_native_pipeline_driver_lifecycle():
    registry = {
        "queries": {
            "query-a": {
                "query_id": "query-a",
                "fragment_executions": {
                    "fragment-a": {
                        "fragment_id": "fragment-a",
                        "fragment_execution_id": 1,
                        "pending_submission_count": 0,
                        "progress_topology": _topology(
                            (
                                2,
                                ["TABLE_SCAN", "INOUT_FUNCTION"],
                                [{}, {"pipeline_role": "sink", "udf_name": "udf"}],
                            ),
                            (
                                1,
                                ["INOUT_FUNCTION", "COPY_TO_FILE"],
                                [{"pipeline_role": "source", "udf_name": "udf"}, {}],
                            ),
                        ),
                        "partitions": {
                            "0": {
                                "state": "RUNNING",
                                "running_attempts": [
                                    {
                                        "task_stats": {
                                            "total_pipeline_tasks": 5,
                                            "queued_pipeline_tasks": 1,
                                            "running_pipeline_tasks": 2,
                                            "completed_pipeline_tasks": 2,
                                            "pipelines": [
                                                {
                                                    "pipeline_id": 2,
                                                    "operators": ["TABLE_SCAN", "INOUT_FUNCTION"],
                                                    "operator_details": [
                                                        {},
                                                        {"pipeline_role": "sink", "udf_name": "udf"},
                                                    ],
                                                    "input_rows": 8,
                                                    "input_bytes": 80,
                                                    "output_rows": 8,
                                                    "output_bytes": 80,
                                                    "total_pipeline_tasks": 3,
                                                    "queued_pipeline_tasks": 0,
                                                    "running_pipeline_tasks": 1,
                                                    "completed_pipeline_tasks": 2,
                                                },
                                                {
                                                    "pipeline_id": 1,
                                                    "operators": ["INOUT_FUNCTION", "COPY_TO_FILE"],
                                                    "operator_details": [
                                                        {"pipeline_role": "source", "udf_name": "udf"},
                                                        {},
                                                    ],
                                                    "input_rows": 3,
                                                    "input_bytes": 30,
                                                    "output_rows": 3,
                                                    "output_bytes": 30,
                                                    "total_pipeline_tasks": 2,
                                                    "queued_pipeline_tasks": 1,
                                                    "running_pipeline_tasks": 1,
                                                    "completed_pipeline_tasks": 0,
                                                },
                                            ],
                                        }
                                    }
                                ],
                            }
                        },
                    }
                },
            }
        }
    }

    snapshot = build_progress_snapshot(registry, "query-a", started_at=0)

    assert snapshot["total_pipeline_tasks"] == 5
    assert snapshot["queued_pipeline_tasks"] == 1
    assert snapshot["running_pipeline_tasks"] == 2
    assert snapshot["completed_pipeline_tasks"] == 2
    pipelines = snapshot["fragments"][0]["pipelines"]
    assert [pipeline["total_pipeline_tasks"] for pipeline in pipelines] == [3, 2]
    assert [pipeline["completed_pipeline_tasks"] for pipeline in pipelines] == [2, 0]
    assert [pipeline["running_pipeline_tasks"] for pipeline in pipelines] == [1, 1]


def test_progress_snapshot_publishes_native_topology_before_first_partition_runs():
    topology = {
        "schema": "pipeline_topology",
        "pipelines": [
            {
                "pipeline_id": 1,
                "operators": ["RESULT_COLLECTOR"],
                "operator_details": [{}],
                "stage_ids": [],
            },
            {
                "pipeline_id": 2,
                "operators": ["TABLE_SCAN", "PROJECTION", "RESULT_COLLECTOR"],
                "operator_details": [{}, {}, {}],
                "stage_ids": [],
            },
        ],
    }
    registry = {
        "queries": {
            "query-a": {
                "query_id": "query-a",
                "fragment_executions": {
                    "fragment-a": {
                        "fragment_id": "fragment-a",
                        "fragment_execution_id": 0,
                        "pending_submission_count": 0,
                        "no_more_partitions": False,
                        "partitions": {},
                        "progress_topology": topology,
                    }
                },
            }
        }
    }

    snapshot = build_progress_snapshot(registry, "query-a", started_at=0)
    pipelines = snapshot["fragments"][0]["pipelines"]
    lines = format_progress_snapshot(snapshot, width=120)

    assert [pipeline["name"] for pipeline in pipelines] == [
        "ScanSource->Projection->RESULT_COLLECTOR",
        "RESULT_COLLECTOR",
    ]
    assert [pipeline["state"] for pipeline in pipelines] == ["P", "P"]
    assert snapshot["total_pipeline_tasks"] == 0
    assert lines[2] == "Fragment 1 [PENDING 0]"
    assert len(lines) == 5


def test_progress_snapshot_overlays_only_live_counters_matching_planned_topology():
    topology = {
        "schema": "pipeline_topology",
        "pipelines": [
            {
                "pipeline_id": 1,
                "operators": ["RESULT_COLLECTOR"],
                "operator_details": [{}],
                "stage_ids": [],
            },
            {
                "pipeline_id": 2,
                "operators": ["TABLE_SCAN", "PROJECTION", "RESULT_COLLECTOR"],
                "operator_details": [{}, {}, {}],
                "stage_ids": [],
            },
        ],
    }
    live_stats = {
        "processed_input_rows": 7,
        "total_pipeline_tasks": 1,
        "queued_pipeline_tasks": 0,
        "running_pipeline_tasks": 1,
        "completed_pipeline_tasks": 0,
        "pipelines": [
            {
                "pipeline_id": 2,
                "operators": ["TABLE_SCAN", "PROJECTION", "RESULT_COLLECTOR"],
                "operator_details": [{}, {}, {}],
                "input_rows": 7,
                "input_bytes": 70,
                "total_pipeline_tasks": 1,
                "queued_pipeline_tasks": 0,
                "running_pipeline_tasks": 1,
                "completed_pipeline_tasks": 0,
            },
            {
                "pipeline_id": 99,
                "operators": ["TABLE_SCAN"],
                "operator_details": [{}],
                "input_rows": 999,
                "total_pipeline_tasks": 1,
                "running_pipeline_tasks": 1,
            },
        ],
    }
    registry = {
        "queries": {
            "query-a": {
                "query_id": "query-a",
                "fragment_executions": {
                    "fragment-a": {
                        "fragment_id": "fragment-a",
                        "fragment_execution_id": 1,
                        "pending_submission_count": 0,
                        "no_more_partitions": True,
                        "partitions": {
                            "0": {
                                "state": "RUNNING",
                                "running_attempts": [{"task_stats": live_stats}],
                            }
                        },
                        "progress_topology": topology,
                    }
                },
            }
        }
    }

    with pytest.raises(RuntimeError, match="unknown pipeline_id 99"):
        build_progress_snapshot(registry, "query-a", started_at=0)


def test_fragment_does_not_mix_deferred_partitions_with_native_pipeline_tasks():
    task_stats = {
        "total_pipeline_tasks": 40,
        "queued_pipeline_tasks": 2,
        "running_pipeline_tasks": 2,
        "completed_pipeline_tasks": 36,
        "pipelines": [
            {
                "pipeline_id": 5,
                "name": "ExchangeSource->Transform",
                "operators": ["EXCHANGE_SOURCE", "PROJECTION"],
                "operator_details": [{}, {}],
                "stage_ids": [],
                "total_pipeline_tasks": 36,
                "queued_pipeline_tasks": 0,
                "running_pipeline_tasks": 0,
                "completed_pipeline_tasks": 36,
            },
            *[
                {
                    "pipeline_id": pipeline_id,
                    "name": f"Pipeline{pipeline_id}",
                    "operators": ["PROJECTION"],
                    "operator_details": [{}],
                    "stage_ids": [],
                    "total_pipeline_tasks": 1,
                    "queued_pipeline_tasks": int(pipeline_id <= 2),
                    "running_pipeline_tasks": int(pipeline_id in {3, 4}),
                    "completed_pipeline_tasks": 0,
                }
                for pipeline_id in range(1, 5)
            ],
        ],
    }
    partitions = {
        **{
            str(partition_id): {
                "state": "RUNNING",
                "running_count": 1,
                "running_attempts": [{"task_stats": task_stats}],
            }
            for partition_id in range(7)
        },
        **{
            str(partition_id): {
                "state": "SEALED",
                "execution_ready_deferred": True,
            }
            for partition_id in range(7, 36)
        },
    }
    registry = {
        "queries": {
            "query-a": {
                "fragment_executions": {
                    "fragment-a": {
                        "fragment_id": "fragment-a",
                        "fragment_execution_id": 1,
                        "pending_submission_count": 29,
                        "progress_topology": _topology(
                            (5, ["EXCHANGE_SOURCE", "PROJECTION"], None),
                            *((pipeline_id, ["PROJECTION"], None) for pipeline_id in range(1, 5)),
                        ),
                        "partition_count": 36,
                        "running_count": 7,
                        "finished_count": 0,
                        "failed_count": 0,
                        "no_more_partitions": True,
                        "partitions": partitions,
                    }
                }
            }
        }
    }

    snapshot = build_progress_snapshot(registry, "query-a")
    fragment = snapshot["fragments"][0]
    lines = format_progress_snapshot(snapshot, width=160)

    assert snapshot["total_pipeline_tasks"] == 280
    assert snapshot["queued_pipeline_tasks"] == 14
    assert snapshot["running_pipeline_tasks"] == 14
    assert snapshot["completed_pipeline_tasks"] == 252
    assert snapshot["total_partitions"] == 36
    assert snapshot["queued_partitions"] == 29
    assert snapshot["running_partitions"] == 7
    assert snapshot["completed_partitions"] == 0
    assert snapshot["pending_partitions"] == 29
    assert set(fragment) == {"id", "display_id", "name", "pending_partitions", "pipelines"}
    assert fragment["pending_partitions"] == 29
    assert sum(pipeline["queued_pipeline_tasks"] for pipeline in fragment["pipelines"]) == 14
    assert sum(pipeline["running_pipeline_tasks"] for pipeline in fragment["pipelines"]) == 14
    assert sum(pipeline["completed_pipeline_tasks"] for pipeline in fragment["pipelines"]) == 252
    assert lines[2] == "Fragment 1 [PENDING 29]"


def test_speculative_attempts_count_as_one_running_logical_partition():
    running_task_stats = {
        "total_pipeline_tasks": 1,
        "queued_pipeline_tasks": 0,
        "running_pipeline_tasks": 1,
        "completed_pipeline_tasks": 0,
    }
    registry = {
        "queries": {
            "query-a": {
                "fragment_executions": {
                    "fragment-a": {
                        "fragment_id": "fragment-a",
                        "fragment_execution_id": 1,
                        "pending_submission_count": 1,
                        "progress_topology": _topology(),
                        "partition_count": 2,
                        "running_count": 2,
                        "finished_count": 0,
                        "no_more_partitions": True,
                        "partitions": {
                            "0": {
                                "state": "RUNNING",
                                "running_count": 2,
                                "running_attempts": [
                                    {"task_stats": running_task_stats},
                                    {"task_stats": running_task_stats},
                                ],
                            },
                            "1": {"state": "SEALED"},
                        },
                    }
                }
            }
        }
    }

    snapshot = build_progress_snapshot(registry, "query-a")

    assert snapshot["total_partitions"] == 2
    assert snapshot["running_partitions"] == 1
    assert snapshot["queued_partitions"] == 1
    assert snapshot["total_pipeline_tasks"] == 2
    assert snapshot["running_pipeline_tasks"] == 2


def test_progress_snapshot_stays_running_until_query_finishes():
    registry = {
        "queries": {
            "query-a": {
                "finished": False,
                "fragment_executions": {
                    "fragment-scan": {
                        "fragment_id": "fragment-scan",
                        "fragment_execution_id": 0,
                        "pending_submission_count": 0,
                        "progress_topology": _topology(),
                        "partition_count": 1,
                        "running_count": 0,
                        "finished_count": 1,
                        "waiting_for_node_count": 0,
                        "waiting_for_execution_count": 0,
                        "finished": True,
                        "partitions": {
                            "0": {
                                "state": "FINISHED",
                                "selected_output_stats": {"processed_input_rows": 10},
                            }
                        },
                    }
                },
            }
        }
    }

    snapshot = build_progress_snapshot(registry, "query-a")

    assert snapshot["state"] == "RUNNING"
    assert snapshot["completed_pipeline_tasks"] == 0
    assert snapshot["total_pipeline_tasks"] == 0
    assert snapshot["completed_partitions"] == snapshot["total_partitions"] == 1


def test_progress_snapshot_ignores_legacy_row_byte_fields():
    registry = {
        "queries": {
            "query-a": {
                "fragment_executions": {
                    "fragment-a": {
                        "fragment_id": "fragment-a",
                        "fragment_execution_id": 1,
                        "pending_submission_count": 0,
                        "progress_topology": _topology(),
                        "partition_count": 1,
                        "running_count": 0,
                        "finished_count": 1,
                        "waiting_for_node_count": 0,
                        "waiting_for_execution_count": 0,
                        "partitions": {
                            "0": {
                                "selected_output_stats": {
                                    "rows": 3,
                                    "total_rows": 4,
                                    "output_rows": 5,
                                    "num_rows": 6,
                                    "positions": 7,
                                    "total_bytes": 11,
                                    "output_bytes": 13,
                                    "size_bytes": 17,
                                    "bytes": 19,
                                }
                            }
                        },
                    }
                },
            }
        }
    }

    snapshot = build_progress_snapshot(registry, "query-a")

    assert snapshot["processed_rows"] == 0
    assert snapshot["processed_bytes"] == 0


def test_finished_query_without_fragments_is_terminal():
    snapshot = build_progress_snapshot(
        {"queries": {"query-a": {"finished": True, "fragment_executions": {}}}},
        "query-a",
    )

    assert snapshot["state"] == "FINISHED"


def test_progress_renderer_success_finish_forces_logical_partition_counters():
    snapshot = {
        "schema": "progress",
        "query_id": "query-a",
        "state": "RUNNING",
        "processed_rows": 10,
        "processed_bytes": 100,
        "total_pipeline_tasks": 1,
        "queued_pipeline_tasks": 0,
        "running_pipeline_tasks": 1,
        "completed_pipeline_tasks": 0,
        "total_partitions": 2,
        "queued_partitions": 1,
        "running_partitions": 1,
        "completed_partitions": 0,
        "failed_partitions": 0,
        "pending_partitions": 2,
        "fragments": [
            {
                "id": "fragment-a",
                "display_id": "1",
                "name": "Fragment 1",
                "pending_partitions": 2,
                "pipelines": [],
            }
        ],
    }
    renderer = ProgressRenderer(lambda: snapshot, stream=io.StringIO(), interval_s=0.1)

    renderer.update(force=True)
    renderer.finish(final_state="FINISHED")

    final = renderer._last_snapshot
    assert final is not None
    assert final["completed_partitions"] == final["total_partitions"] == 2
    assert final["queued_partitions"] == 0
    assert final["running_partitions"] == 0
    assert final["pending_partitions"] == 0
    assert final["fragments"][0]["pending_partitions"] == 0
    assert set(final["fragments"][0]) == {
        "id",
        "display_id",
        "name",
        "pending_partitions",
        "pipelines",
    }


def test_progress_renderer_finishes_from_preteardown_snapshot_without_remote_lookup():
    stream = io.StringIO()
    final_snapshot = {
        "schema": "progress",
        "query_id": "query-a",
        "state": "FINISHED",
        "processed_rows": 10,
        "processed_bytes": 100,
        "total_pipeline_tasks": 1,
        "queued_pipeline_tasks": 0,
        "running_pipeline_tasks": 0,
        "completed_pipeline_tasks": 1,
        "fragments": [],
    }
    remote_lookups = []
    renderer = ProgressRenderer(
        lambda: remote_lookups.append("lookup"),
        stream=stream,
        interval_s=0.1,
    )

    renderer.finish(
        final_state="FINISHED",
        final_snapshot=final_snapshot,
    )

    assert remote_lookups == []
    assert "10 rows" in stream.getvalue()


def test_progress_snapshot_query_rows_sum_all_fragments():
    registry = {
        "queries": {
            "query-a": {
                "fragment_executions": {
                    "fragment-source": {
                        "fragment_id": "fragment-source",
                        "fragment_execution_id": 1,
                        "pending_submission_count": 0,
                        "progress_topology": _topology(),
                        "partition_count": 1,
                        "running_count": 1,
                        "finished_count": 0,
                        "waiting_for_node_count": 0,
                        "waiting_for_execution_count": 0,
                        "partitions": {
                            "0": {"selected_output_stats": {"processed_input_rows": 0}},
                        },
                    },
                    "fragment-copy": {
                        "fragment_id": "fragment-copy",
                        "fragment_execution_id": 2,
                        "pending_submission_count": 0,
                        "progress_topology": _topology(),
                        "partition_count": 1,
                        "running_count": 1,
                        "finished_count": 0,
                        "waiting_for_node_count": 0,
                        "waiting_for_execution_count": 0,
                        "partitions": {
                            "0": {"selected_output_stats": {"processed_input_rows": 1000}},
                        },
                    },
                },
            }
        }
    }

    snapshot = build_progress_snapshot(registry, "query-a", started_at=0)

    assert snapshot["processed_rows"] == 1000


def test_progress_snapshot_sums_pending_fte_descriptors_across_fragments():
    registry = {
        "queries": {
            "query-a": {
                "fragment_executions": {
                    "fragment-source": {
                        "fragment_id": "fragment-source",
                        "fragment_execution_id": 1,
                        "pending_submission_count": 0,
                        "progress_topology": _topology(),
                        "partition_count": 1,
                        "running_count": 0,
                        "finished_count": 1,
                        "waiting_for_node_count": 0,
                        "waiting_for_execution_count": 0,
                        "no_more_partitions": True,
                        "partitions": {"0": {"state": "FINISHED"}},
                    },
                    "fragment-udf": {
                        "fragment_id": "fragment-udf",
                        "fragment_execution_id": 2,
                        "pending_submission_count": 12,
                        "progress_topology": _topology(),
                        "partition_count": 16,
                        "running_count": 4,
                        "finished_count": 0,
                        "waiting_for_node_count": 12,
                        "waiting_for_execution_count": 0,
                        "no_more_partitions": True,
                        "partitions": {
                            **{
                                str(index): {
                                    "state": "RUNNING",
                                    "running_count": 1,
                                    "running_attempts": [{"task_stats": {}}],
                                }
                                for index in range(4)
                            },
                            **{
                                str(index): {"state": "SEALED", "waiting_for_execution": True} for index in range(4, 16)
                            },
                        },
                    },
                },
            }
        }
    }

    snapshot = build_progress_snapshot(registry, "query-a", started_at=0)

    assert snapshot["total_pipeline_tasks"] == 0
    assert snapshot["completed_pipeline_tasks"] == 0
    assert snapshot["running_pipeline_tasks"] == 0
    assert snapshot["total_partitions"] == 17
    assert snapshot["queued_partitions"] == 12
    assert snapshot["running_partitions"] == 4
    assert snapshot["completed_partitions"] == 1
    assert snapshot["pending_partitions"] == 12
    assert [fragment["pending_partitions"] for fragment in snapshot["fragments"]] == [0, 12]


def test_progress_snapshot_uses_native_driver_lifecycle_not_split_queue_counts():
    registry = {
        "queries": {
            "query-a": {
                "fragment_executions": {
                    "fragment-source": {
                        "fragment_id": "fragment-source",
                        "fragment_execution_id": 1,
                        "pending_submission_count": 0,
                        "progress_topology": _topology(),
                        "partition_count": 1,
                        "running_count": 1,
                        "finished_count": 0,
                        "waiting_for_node_count": 0,
                        "waiting_for_execution_count": 0,
                        "no_more_partitions": True,
                        "partitions": {
                            "0": {
                                "state": "RUNNING",
                                "running_count": 1,
                                "initial_split_count_by_source": {"scan": 100},
                                "running_attempts": [
                                    {
                                        "task_stats": {
                                            "submitted_split_count": 100,
                                            "queued_split_count": 90,
                                            "consumed_split_count": 10,
                                            "completed_split_count": 3,
                                            "total_pipeline_tasks": 4,
                                            "queued_pipeline_tasks": 0,
                                            "running_pipeline_tasks": 2,
                                            "completed_pipeline_tasks": 2,
                                            "processed_input_rows": 600,
                                            "physical_input_bytes": 1024,
                                        }
                                    }
                                ],
                            }
                        },
                    }
                },
            }
        }
    }

    snapshot = build_progress_snapshot(registry, "query-a", started_at=0)

    assert snapshot["total_pipeline_tasks"] == 4
    assert snapshot["queued_pipeline_tasks"] == 0
    assert snapshot["running_pipeline_tasks"] == 2
    assert snapshot["completed_pipeline_tasks"] == 2
    assert "total_pipeline_tasks" not in snapshot["fragments"][0]
    assert snapshot["running_partitions"] == 1


def test_progress_snapshot_finished_merged_task_ignores_abandoned_splits():
    registry = {
        "queries": {
            "query-a": {
                "finished": True,
                "fragment_executions": {
                    "fragment-source": {
                        "fragment_id": "fragment-source",
                        "fragment_execution_id": 1,
                        "pending_submission_count": 0,
                        "progress_topology": _topology(),
                        "partition_count": 1,
                        "running_count": 0,
                        "finished_count": 1,
                        "waiting_for_node_count": 0,
                        "waiting_for_execution_count": 0,
                        "no_more_partitions": True,
                        "partitions": {
                            "0": {
                                "state": "FINISHED",
                                "running_count": 0,
                                "initial_split_count_by_source": {"scan": 36},
                                "selected_output_stats": {
                                    "submitted_split_count": 36,
                                    "queued_split_count": 24,
                                    "consumed_split_count": 12,
                                    "completed_split_count": 12,
                                    "processed_input_rows": 6000,
                                    "physical_input_bytes": 327680,
                                },
                            }
                        },
                    }
                },
            }
        }
    }

    snapshot = build_progress_snapshot(registry, "query-a", started_at=0)

    assert snapshot["total_pipeline_tasks"] == 0
    assert snapshot["queued_pipeline_tasks"] == 0
    assert snapshot["running_pipeline_tasks"] == 0
    assert snapshot["completed_pipeline_tasks"] == 0
    assert snapshot["completed_partitions"] == snapshot["total_partitions"] == 1


def test_progress_snapshot_does_not_invent_native_tasks_before_stats_arrive():
    registry = {
        "queries": {
            "query-a": {
                "fragment_executions": {
                    "fragment-source": {
                        "fragment_id": "fragment-source",
                        "fragment_execution_id": 1,
                        "pending_submission_count": 0,
                        "progress_topology": _topology(),
                        "partitions": {
                            "0": {
                                "state": "RUNNING",
                                "running_count": 1,
                                "running_attempts": [
                                    {
                                        "task_stats": {
                                            "submitted_split_count": 36,
                                            "consumed_split_count": 36,
                                            "completed_split_count": 36,
                                            "processed_input_rows": 100,
                                        }
                                    }
                                ],
                            }
                        },
                    }
                }
            }
        }
    }

    snapshot = build_progress_snapshot(registry, "query-a")
    lines = format_progress_snapshot(snapshot, width=100)

    assert snapshot["total_pipeline_tasks"] == 0
    assert snapshot["running_pipeline_tasks"] == 0
    assert snapshot["total_partitions"] == 1
    assert snapshot["running_partitions"] == 1
    assert "%" not in lines[0]
    assert ">" not in lines[0]


def test_progress_consumes_structured_native_pipeline_metrics():
    registry = {
        "queries": {
            "query-a": {
                "fragment_executions": {
                    "fragment-udf-copy": {
                        "fragment_id": "fragment-udf-copy",
                        "fragment_execution_id": 2,
                        "pending_submission_count": 0,
                        "progress_topology": _topology(
                            (
                                3,
                                ["EXCHANGE_SOURCE", "STREAMING_UDF"],
                                [
                                    {},
                                    {
                                        "udf_name": "_decode_and_transform",
                                        "pipeline_role": "sink",
                                    },
                                ],
                            ),
                            (
                                2,
                                ["STREAMING_UDF", "PROJECTION", "STREAMING_UDF"],
                                [
                                    {
                                        "udf_name": "_decode_and_transform",
                                        "pipeline_role": "source",
                                    },
                                    {},
                                    {"udf_name": "ResNetModel", "pipeline_role": "sink"},
                                ],
                            ),
                            (
                                1,
                                ["STREAMING_UDF", "PROJECTION", "BATCH_COPY_TO_FILE"],
                                [
                                    {"udf_name": "ResNetModel", "pipeline_role": "source"},
                                    {},
                                    {},
                                ],
                            ),
                        ),
                        "partition_count": 1,
                        "running_count": 0,
                        "finished_count": 1,
                        "waiting_for_node_count": 0,
                        "waiting_for_execution_count": 0,
                        "partitions": {
                            "0": {
                                "selected_output_stats": {
                                    "processed_input_rows": 10,
                                    "processed_input_bytes": 100,
                                    "pipelines": [
                                        {
                                            "pipeline_id": 3,
                                            "operators": ["EXCHANGE_SOURCE", "STREAMING_UDF"],
                                            "operator_details": [
                                                {},
                                                {
                                                    "udf_name": "_decode_and_transform",
                                                    "pipeline_role": "sink",
                                                },
                                            ],
                                            "input_rows": 8,
                                            "input_bytes": 80,
                                            "output_rows": 8,
                                            "output_bytes": 80,
                                        },
                                        {
                                            "pipeline_id": 2,
                                            "operators": ["STREAMING_UDF", "PROJECTION", "STREAMING_UDF"],
                                            "operator_details": [
                                                {
                                                    "udf_name": "_decode_and_transform",
                                                    "pipeline_role": "source",
                                                },
                                                {},
                                                {
                                                    "udf_name": "ResNetModel",
                                                    "pipeline_role": "sink",
                                                },
                                            ],
                                            "input_rows": 7,
                                            "input_bytes": 700,
                                            "output_rows": 6,
                                            "output_bytes": 600,
                                        },
                                        {
                                            "pipeline_id": 1,
                                            "operators": ["STREAMING_UDF", "PROJECTION", "BATCH_COPY_TO_FILE"],
                                            "operator_details": [
                                                {
                                                    "udf_name": "ResNetModel",
                                                    "pipeline_role": "source",
                                                },
                                                {},
                                                {},
                                            ],
                                            "input_rows": 5,
                                            "input_bytes": 50,
                                            "output_rows": 5,
                                            "output_bytes": 50,
                                        },
                                    ],
                                }
                            }
                        },
                    }
                },
            }
        }
    }

    snapshot = build_progress_snapshot(registry, "query-a")

    pipelines = snapshot["fragments"][0]["pipelines"]
    assert [pipeline["display_id"] for pipeline in pipelines] == ["1.1", "1.2", "1.3"]
    assert [pipeline["name"] for pipeline in pipelines] == [
        "ExchangeSource->_decode_and_transform(sink)",
        "_decode_and_transform(source)->Projection->ResNetModel(sink)",
        "ResNetModel(source)->Projection->CopySink",
    ]
    assert [pipeline["processed_rows"] for pipeline in pipelines] == [8, 7, 5]
    assert [pipeline["processed_bytes"] for pipeline in pipelines] == [80, 700, 50]


def test_progress_sink_counts_accepted_input_before_streaming_submit_completes():
    registry = {
        "queries": {
            "query-a": {
                "fragment_executions": {
                    "fragment-udf": {
                        "fragment_id": "fragment-udf",
                        "fragment_execution_id": 1,
                        "pending_submission_count": 0,
                        "progress_topology": _topology(
                            (
                                1,
                                ["EXCHANGE_SOURCE", "STREAMING_UDF"],
                                [{}, {"udf_name": "decode", "pipeline_role": "sink"}],
                            ),
                        ),
                        "partitions": {
                            "0": {
                                "state": "RUNNING",
                                "running_attempts": [
                                    {
                                        "task_stats": {
                                            "total_pipeline_tasks": 1,
                                            "queued_pipeline_tasks": 0,
                                            "running_pipeline_tasks": 1,
                                            "completed_pipeline_tasks": 0,
                                            "pipelines": [
                                                {
                                                    "pipeline_id": 1,
                                                    "operators": ["EXCHANGE_SOURCE", "STREAMING_UDF"],
                                                    "operator_details": [
                                                        {},
                                                        {
                                                            "udf_name": "decode",
                                                            "pipeline_role": "sink",
                                                            "udf_accepted_input_rows": "8",
                                                            "udf_accepted_input_bytes": "80",
                                                            "udf_completed_input_rows": "0",
                                                            "udf_completed_input_bytes": "0",
                                                            "udf_emitted_output_rows": "3",
                                                            "udf_emitted_output_bytes": "30",
                                                        },
                                                    ],
                                                    "input_rows": 8,
                                                    "input_bytes": 80,
                                                    "output_rows": 8,
                                                    "output_bytes": 80,
                                                }
                                            ],
                                        }
                                    }
                                ],
                            }
                        },
                    }
                }
            }
        }
    }

    snapshot = build_progress_snapshot(registry, "query-a")
    pipeline = snapshot["fragments"][0]["pipelines"][0]

    assert pipeline["name"] == "ExchangeSource->decode(sink)"
    assert pipeline["processed_rows"] == 8
    assert pipeline["processed_bytes"] == 80


def test_progress_displays_native_pipelines_in_source_to_sink_order():
    registry = {
        "queries": {
            "query-a": {
                "fragment_executions": {
                    "fragment-copy": {
                        "fragment_id": "fragment-copy",
                        "fragment_execution_id": 1,
                        "pending_submission_count": 0,
                        "progress_topology": _topology(
                            (1, ["RESULT_COLLECTOR"], None),
                            (2, ["COPY_TO_FILE", "RESULT_COLLECTOR"], None),
                            (
                                3,
                                ["STREAMING_UDF", "PROJECTION", "COPY_TO_FILE"],
                                [{"udf_name": "Embedder", "pipeline_role": "source"}, {}, {}],
                            ),
                            (
                                4,
                                ["STREAMING_UDF", "PROJECTION", "STREAMING_UDF"],
                                [
                                    {"udf_name": "chunker", "pipeline_role": "source"},
                                    {},
                                    {"udf_name": "Embedder", "pipeline_role": "sink"},
                                ],
                            ),
                            (
                                5,
                                ["STREAMING_UDF", "PROJECTION", "STREAMING_UDF"],
                                [
                                    {
                                        "udf_name": "extract_text_from_pdf",
                                        "pipeline_role": "source",
                                    },
                                    {},
                                    {"udf_name": "chunker", "pipeline_role": "sink"},
                                ],
                            ),
                            (
                                6,
                                ["TABLE_SCAN", "PROJECTION", "STREAMING_UDF"],
                                [
                                    {},
                                    {},
                                    {
                                        "udf_name": "extract_text_from_pdf",
                                        "pipeline_role": "sink",
                                    },
                                ],
                            ),
                        ),
                        "partitions": {
                            "0": {
                                "state": "FINISHED",
                                "selected_output_stats": {
                                    "pipelines": [
                                        {"pipeline_id": 1, "operators": ["RESULT_COLLECTOR"]},
                                        {
                                            "pipeline_id": 2,
                                            "operators": ["COPY_TO_FILE", "RESULT_COLLECTOR"],
                                        },
                                        {
                                            "pipeline_id": 3,
                                            "operators": ["STREAMING_UDF", "PROJECTION", "COPY_TO_FILE"],
                                            "operator_details": [
                                                {"udf_name": "Embedder", "pipeline_role": "source"},
                                                {},
                                                {},
                                            ],
                                        },
                                        {
                                            "pipeline_id": 4,
                                            "operators": ["STREAMING_UDF", "PROJECTION", "STREAMING_UDF"],
                                            "operator_details": [
                                                {"udf_name": "chunker", "pipeline_role": "source"},
                                                {},
                                                {"udf_name": "Embedder", "pipeline_role": "sink"},
                                            ],
                                        },
                                        {
                                            "pipeline_id": 5,
                                            "operators": ["STREAMING_UDF", "PROJECTION", "STREAMING_UDF"],
                                            "operator_details": [
                                                {
                                                    "udf_name": "extract_text_from_pdf",
                                                    "pipeline_role": "source",
                                                },
                                                {},
                                                {"udf_name": "chunker", "pipeline_role": "sink"},
                                            ],
                                        },
                                        {
                                            "pipeline_id": 6,
                                            "operators": ["TABLE_SCAN", "PROJECTION", "STREAMING_UDF"],
                                            "operator_details": [
                                                {},
                                                {},
                                                {
                                                    "udf_name": "extract_text_from_pdf",
                                                    "pipeline_role": "sink",
                                                },
                                            ],
                                        },
                                    ]
                                },
                            }
                        },
                    }
                }
            }
        }
    }

    snapshot = build_progress_snapshot(registry, "query-a")

    assert [pipeline["name"] for pipeline in snapshot["fragments"][0]["pipelines"]] == [
        "ScanSource->Projection->extract_text_from_pdf(sink)",
        "extract_text_from_pdf(source)->Projection->chunker(sink)",
        "chunker(source)->Projection->Embedder(sink)",
        "Embedder(source)->Projection->CopySink",
        "CopySink->RESULT_COLLECTOR",
        "RESULT_COLLECTOR",
    ]


def test_progress_consumes_structured_native_repartition_pipelines():
    registry = {
        "queries": {
            "query-a": {
                "fragment_executions": {
                    "fragment-native-copy": {
                        "fragment_id": "fragment-native-copy",
                        "fragment_execution_id": 1,
                        "pending_submission_count": 0,
                        "progress_topology": _topology(
                            (
                                3,
                                ["TABLE_SCAN", "PROJECTION", "REPARTITION"],
                                [{}, {}, {"pipeline_role": "sink"}],
                            ),
                            (
                                2,
                                ["REPARTITION", "INOUT_FUNCTION"],
                                [
                                    {"pipeline_role": "source"},
                                    {
                                        "udf_name": "_decode_and_transform",
                                        "pipeline_role": "sink",
                                    },
                                ],
                            ),
                            (
                                1,
                                ["INOUT_FUNCTION", "COPY_TO_FILE"],
                                [
                                    {
                                        "udf_name": "_decode_and_transform",
                                        "pipeline_role": "source",
                                    },
                                    {},
                                ],
                            ),
                        ),
                        "partition_count": 1,
                        "running_count": 0,
                        "finished_count": 1,
                        "waiting_for_node_count": 0,
                        "waiting_for_execution_count": 0,
                        "partitions": {
                            "0": {
                                "selected_output_stats": {
                                    "processed_input_rows": 10,
                                    "processed_input_bytes": 100,
                                    "pipelines": [
                                        {
                                            "pipeline_id": 3,
                                            "operators": ["TABLE_SCAN", "PROJECTION", "REPARTITION"],
                                            "operator_details": [
                                                {},
                                                {},
                                                {"pipeline_role": "sink"},
                                            ],
                                            "input_rows": 10,
                                            "input_bytes": 100,
                                            "output_rows": 10,
                                            "output_bytes": 100,
                                        },
                                        {
                                            "pipeline_id": 2,
                                            "operators": ["REPARTITION", "INOUT_FUNCTION"],
                                            "operator_details": [
                                                {"pipeline_role": "source"},
                                                {
                                                    "udf_name": "_decode_and_transform",
                                                    "pipeline_role": "sink",
                                                },
                                            ],
                                            "input_rows": 10,
                                            "input_bytes": 100,
                                            "output_rows": 8,
                                            "output_bytes": 80,
                                        },
                                        {
                                            "pipeline_id": 1,
                                            "operators": ["INOUT_FUNCTION", "COPY_TO_FILE"],
                                            "operator_details": [
                                                {
                                                    "udf_name": "_decode_and_transform",
                                                    "pipeline_role": "source",
                                                },
                                                {},
                                            ],
                                            "input_rows": 7,
                                            "input_bytes": 700,
                                            "output_rows": 7,
                                            "output_bytes": 700,
                                        },
                                    ],
                                }
                            }
                        },
                    }
                },
            }
        }
    }

    snapshot = build_progress_snapshot(registry, "query-a")

    pipelines = snapshot["fragments"][0]["pipelines"]
    assert [pipeline["display_id"] for pipeline in pipelines] == ["1.1", "1.2", "1.3"]
    assert [pipeline["name"] for pipeline in pipelines] == [
        "ScanSource->Projection->Repartition(sink)",
        "Repartition(source)->_decode_and_transform(sink)",
        "_decode_and_transform(source)->CopySink",
    ]
    assert [pipeline["processed_rows"] for pipeline in pipelines] == [10, 10, 7]
    assert [pipeline["processed_bytes"] for pipeline in pipelines] == [100, 100, 700]


def test_progress_distinguishes_remote_exchange_sink_from_local_repartition():
    assert (
        shared_progress_mod._display_pipeline_name(
            {
                "operators": ["TABLE_SCAN", "EXCHANGE_SINK"],
                "operator_details": [{}, {"pipeline_role": "sink"}],
            }
        )
        == "ScanSource->ExchangeSink(sink)"
    )
    assert (
        shared_progress_mod._display_pipeline_name(
            {
                "operators": ["TABLE_SCAN", "REPARTITION"],
                "operator_details": [{}, {"pipeline_role": "sink"}],
            }
        )
        == "ScanSource->Repartition(sink)"
    )


def test_progress_displays_native_pipeline_starting_at_streaming_udf():
    registry = {
        "queries": {
            "query-a": {
                "fragment_executions": {
                    "fragment-copy": {
                        "fragment_id": "fragment-copy",
                        "fragment_execution_id": 2,
                        "pending_submission_count": 0,
                        "progress_topology": _topology(
                            (
                                1,
                                ["STREAMING_UDF", "PROJECTION", "BATCH_COPY_TO_FILE"],
                                [
                                    {"udf_name": "ResNetModel", "pipeline_role": "source"},
                                    {},
                                    {},
                                ],
                            ),
                        ),
                        "partition_count": 1,
                        "running_count": 0,
                        "finished_count": 1,
                        "waiting_for_node_count": 0,
                        "waiting_for_execution_count": 0,
                        "partitions": {
                            "0": {
                                "selected_output_stats": {
                                    "processed_input_rows": 10,
                                    "pipelines": [
                                        {
                                            "pipeline_id": 1,
                                            "operators": [
                                                "STREAMING_UDF",
                                                "PROJECTION",
                                                "BATCH_COPY_TO_FILE",
                                            ],
                                            "operator_details": [
                                                {"udf_name": "ResNetModel", "pipeline_role": "source"},
                                                {},
                                                {},
                                            ],
                                            "input_rows": 10,
                                            "input_bytes": 100,
                                            "output_rows": 10,
                                            "output_bytes": 100,
                                        }
                                    ],
                                }
                            }
                        },
                    }
                },
            }
        }
    }

    snapshot = build_progress_snapshot(registry, "query-a")

    pipelines = snapshot["fragments"][0]["pipelines"]
    assert len(pipelines) == 1
    assert pipelines[0]["name"] == "ResNetModel(source)->Projection->CopySink"


def test_progress_snapshot_requires_exact_query_id():
    registry = {
        "queries": {
            "actual-query": {
                "fragment_executions": {
                    "fragment-a": {
                        "fragment_id": "fragment-a",
                        "fragment_execution_id": 1,
                        "pending_submission_count": 0,
                        "progress_topology": _topology(),
                        "partition_count": 1,
                        "running_count": 1,
                        "finished_count": 0,
                        "waiting_for_node_count": 0,
                        "waiting_for_execution_count": 0,
                        "partitions": {},
                    }
                }
            }
        }
    }

    snapshot = build_progress_snapshot(registry, "requested-query")

    assert snapshot["query_id"] == "requested-query"
    assert snapshot["requested_query_id"] == "requested-query"
    assert snapshot["fragments"] == []


def test_progress_task_status_poll_uses_timeout(monkeypatch):
    import vane.runners.ray.worker_handle as worker_handle_mod

    class _PendingFuture:
        def result(self, timeout):
            assert timeout == pytest.approx(0.001)
            raise TimeoutError

    class _PendingObjectRef:
        def future(self):
            return _PendingFuture()

    class _RemoteMethod:
        def remote(self, *args):
            assert args
            return _PendingObjectRef()

    class _Actor:
        fte_get_task_status = _RemoteMethod()

    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_MAX_ATTEMPTS", "1")

    handle = worker_handle_mod.RayWorkerActorHandle(
        _Actor(),
        memory_capacity_bytes=1 << 60,
        worker_id="worker-a",
        node_id="node-a",
    )

    with pytest.raises(TimeoutError):
        handle.fte_get_task_status(
            {"query_id": "q", "fragment_execution_id": 1, "partition_id": 0, "attempt_id": 0},
            timeout_s=0.001,
        )


def test_fte_progress_snapshot_renders_before_no_more_partitions():
    registry = {
        "queries": {
            "query-a": {
                "fragment_executions": {
                    "fragment-a": {
                        "fragment_id": "fragment-a",
                        "fragment_execution_id": 1,
                        "pending_submission_count": 2,
                        "progress_topology": _topology(),
                        "partition_count": 4,
                        "running_count": 1,
                        "finished_count": 1,
                        "waiting_for_node_count": 0,
                        "waiting_for_execution_count": 0,
                        "no_more_partitions": False,
                        "partitions": {
                            "0": {"state": "FINISHED"},
                            "1": {
                                "state": "RUNNING",
                                "running_count": 1,
                                "running_attempts": [{"task_stats": {}}],
                            },
                            "2": {"state": "SEALED", "waiting_for_execution": True},
                            "3": {"state": "SEALED", "waiting_for_execution": True},
                        },
                    }
                }
            }
        }
    }

    snapshot = build_progress_snapshot(registry, "query-a")
    lines = format_progress_snapshot(snapshot, width=100)

    assert "%" not in lines[0]
    assert ">" not in lines[0]
    assert lines[1].startswith("FRAGMENTS")


def test_unsealed_fragment_cannot_finish_when_all_known_partitions_are_done():
    registry = {
        "queries": {
            "query-a": {
                "finished": True,
                "fragment_executions": {
                    "fragment-a": {
                        "fragment_id": "fragment-a",
                        "fragment_execution_id": 1,
                        "pending_submission_count": 0,
                        "progress_topology": _topology(),
                        "partition_count": 1,
                        "running_count": 0,
                        "finished_count": 1,
                        "finished": True,
                        "no_more_partitions": False,
                        "partitions": {"0": {"state": "FINISHED"}},
                    }
                },
            }
        }
    }

    snapshot = build_progress_snapshot(registry, "query-a")

    assert snapshot["state"] == "RUNNING"


def test_progress_snapshot_renders_without_pipeline_tasks():
    registry = {
        "queries": {
            "query-a": {
                "fragment_executions": {
                    "fragment-a": {
                        "fragment_id": "fragment-a",
                        "fragment_execution_id": 1,
                        "pending_submission_count": 0,
                        "progress_topology": _topology(),
                        "partition_count": 0,
                        "running_count": 0,
                        "finished_count": 0,
                        "waiting_for_node_count": 0,
                        "waiting_for_execution_count": 0,
                        "no_more_partitions": False,
                        "partitions": {},
                    }
                }
            }
        }
    }

    snapshot = build_progress_snapshot(registry, "query-a")
    lines = format_progress_snapshot(snapshot, width=100)

    assert "%" not in lines[0]
    assert ">" not in lines[0]


def test_progress_format_omits_fragment_header_without_fragments():
    snapshot = {
        "elapsed_s": 1,
        "processed_rows": 1000,
        "processed_bytes": 1024,
        "fragments": [],
    }

    lines = format_progress_snapshot(snapshot, width=120)

    assert len(lines) == 1
    assert "FRAGMENTS" not in lines[0]


def test_progress_format_omits_percentage_and_bar():
    snapshot = {
        "elapsed_s": 1,
        "processed_rows": 1,
        "processed_bytes": 1,
        "fragments": [],
    }

    lines = format_progress_snapshot(snapshot, width=100)

    assert lines == ["0:01 [    1 rows,     1B] [    1 rows/s,     1B/s]"]


def test_progress_stage_table_rate_cells_show_per_second_units():
    snapshot = {
        "elapsed_s": 1,
        "processed_rows": 1000,
        "processed_bytes": 1024,
        "fragments": [
            {
                "display_id": "1",
                "pending_partitions": 0,
                "state": "R",
                "processed_rows": 1000,
                "processed_bytes": 1024,
                "pipelines": [
                    {
                        "display_id": "1.1",
                        "name": "ScanSource",
                        "state": "R",
                        "processed_rows": 1000,
                        "processed_bytes": 1024,
                        "queued_pipeline_tasks": 0,
                        "running_pipeline_tasks": 1,
                        "completed_pipeline_tasks": 0,
                    }
                ],
            }
        ],
    }

    lines = format_progress_snapshot(snapshot, width=120)

    assert "rows/s" in lines[0]
    assert "KB/s" in lines[0]
    assert "ROWS/s" in lines[1]
    assert "BYTES/s" in lines[1]
    assert lines[2] == "Fragment 1 [PENDING 0]"
    assert "1K/s" in lines[3]
    assert "1KB/s" in lines[3]


def test_progress_tree_aligns_fragment_and_pipeline_columns():
    snapshot = {
        "elapsed_s": 10,
        "processed_rows": 1000,
        "processed_bytes": 1024,
        "fragments": [
            {
                "display_id": "1",
                "pending_partitions": 0,
                "state": "R",
                "processed_rows": 1000,
                "processed_bytes": 1024,
                "pipelines": [
                    {
                        "display_id": "1.1",
                        "name": "ScanSource->Projection->_stream_chunk_rows(sink)",
                        "state": "R",
                        "processed_rows": 1000,
                        "processed_bytes": 1024,
                        "queued_pipeline_tasks": 0,
                        "running_pipeline_tasks": 1,
                        "completed_pipeline_tasks": 0,
                    },
                    {
                        "display_id": "1.2",
                        "name": "_stream_chunk_rows(source)->Projection->Embedder(sink)",
                        "state": "R",
                        "processed_rows": 1000,
                        "processed_bytes": 1024,
                        "queued_pipeline_tasks": 0,
                        "running_pipeline_tasks": 1,
                        "completed_pipeline_tasks": 0,
                    },
                ],
            }
        ],
    }

    lines = format_progress_snapshot(snapshot, width=160)
    state_columns = [line.rfind(" R ") for line in lines[3:5]]

    assert lines[2] == "Fragment 1 [PENDING 0]"
    assert state_columns[0] > 0
    assert state_columns == [state_columns[0]] * len(state_columns)


def test_progress_tree_rates_use_item_rate_elapsed():
    snapshot = {
        "elapsed_s": 10,
        "processed_rows": 1000,
        "processed_bytes": 1024,
        "fragments": [
            {
                "display_id": "1",
                "pending_partitions": 0,
                "state": "D",
                "processed_rows": 1000,
                "processed_bytes": 1024,
                "rate_elapsed_s": 2,
                "pipelines": [
                    {
                        "display_id": "1.1",
                        "name": "ScanSource",
                        "state": "D",
                        "processed_rows": 1000,
                        "processed_bytes": 1024,
                        "rate_elapsed_s": 2,
                        "queued_pipeline_tasks": 0,
                        "running_pipeline_tasks": 0,
                        "completed_pipeline_tasks": 1,
                    }
                ],
            }
        ],
    }

    lines = format_progress_snapshot(snapshot, width=120)

    assert lines[2] == "Fragment 1 [PENDING 0]"
    assert "500/s" in lines[3]
    assert "512B/s" in lines[3]


def test_progress_renderer_freezes_completed_tree_item_rates(monkeypatch):
    monkeypatch.setenv("VANE_PROGRESS", "raylog")
    times = iter([100.0, 110.0, 120.0])
    monkeypatch.setattr(progress_mod.time, "time", lambda: next(times))

    def snapshot_getter():
        return {
            "processed_rows": 1000,
            "processed_bytes": 1024,
            "fragments": [
                {
                    "id": "fragment-a",
                    "display_id": "1",
                    "pending_partitions": 0,
                    "state": "D",
                    "processed_rows": 1000,
                    "processed_bytes": 1024,
                    "pipelines": [
                        {
                            "id": "1.1",
                            "display_id": "1.1",
                            "name": "ScanSource",
                            "state": "D",
                            "processed_rows": 1000,
                            "processed_bytes": 1024,
                            "queued_pipeline_tasks": 0,
                            "running_pipeline_tasks": 0,
                            "completed_pipeline_tasks": 1,
                        }
                    ],
                }
            ],
        }

    stream = io.StringIO()
    renderer = ProgressRenderer(snapshot_getter, stream=stream)
    renderer.update(force=True)
    renderer.update(force=True)

    lines = stream.getvalue().splitlines()
    assert lines[2] == "Fragment 1 [PENDING 0]"
    assert lines[6] == "Fragment 1 [PENDING 0]"
    assert "100/s" in lines[3]
    assert "100/s" in lines[7]


def test_progress_renderer_reuses_last_fragment_tree_on_transient_empty_snapshot(monkeypatch):
    monkeypatch.setenv("VANE_PROGRESS", "raylog")
    times = iter([100.0, 101.0, 102.0])
    monkeypatch.setattr(progress_mod.time, "time", lambda: next(times))

    snapshots = iter(
        [
            {
                "processed_rows": 1000,
                "processed_bytes": 1024,
                "state": "RUNNING",
                "fragments": [
                    {
                        "id": "fragment-a",
                        "display_id": "1",
                        "pending_partitions": 0,
                        "state": "R",
                        "processed_rows": 1000,
                        "processed_bytes": 1024,
                        "pipelines": [],
                    }
                ],
            },
            {
                "processed_rows": 1000,
                "processed_bytes": 1024,
                "state": "RUNNING",
                "fragments": [],
            },
        ]
    )

    stream = io.StringIO()
    renderer = ProgressRenderer(lambda: next(snapshots), stream=stream)
    renderer.update(force=True)
    renderer.update(force=True)

    output = stream.getvalue()
    assert output.count("FRAGMENTS") == 2
    assert output.count("Fragment 1") == 2


def test_progress_renderer_reuses_last_complete_snapshot_on_final_empty_snapshot(monkeypatch):
    monkeypatch.setenv("VANE_PROGRESS", "raylog")
    times = iter([100.0, 101.0, 102.0])
    monkeypatch.setattr(progress_mod.time, "time", lambda: next(times))

    snapshots = iter(
        [
            {
                "processed_rows": 1000,
                "processed_bytes": 1024,
                "state": "FINISHED",
                "fragments": [
                    {
                        "id": "fragment-a",
                        "display_id": "1",
                        "pending_partitions": 0,
                        "state": "D",
                        "processed_rows": 1000,
                        "processed_bytes": 1024,
                        "pipelines": [],
                    }
                ],
            },
            {
                "processed_rows": 0,
                "processed_bytes": 0,
                "state": "FINISHED",
                "fragments": [],
            },
        ]
    )

    stream = io.StringIO()
    renderer = ProgressRenderer(lambda: next(snapshots), stream=stream)
    renderer.update(force=True)
    renderer.finish()

    output = stream.getvalue()
    assert output.count("Fragment 1") == 2
    assert "0 rows,     0B" not in output


def test_progress_dynamic_renderer_clears_previous_extra_lines(monkeypatch):
    monkeypatch.delenv("VANE_PROGRESS", raising=False)

    class TTYStream(io.StringIO):
        def isatty(self) -> bool:
            return True

    times = iter([100.0, 101.0, 102.0])
    monkeypatch.setattr(progress_mod.time, "time", lambda: next(times))

    snapshots = iter(
        [
            {
                "processed_rows": 1000,
                "processed_bytes": 1024,
                "state": "RUNNING",
                "fragments": [
                    {
                        "display_id": "1",
                        "pending_partitions": 0,
                        "state": "R",
                        "processed_rows": 1000,
                        "processed_bytes": 1024,
                        "pipelines": [
                            {
                                "display_id": "1.1",
                                "name": "ScanSource",
                                "state": "R",
                                "processed_rows": 1000,
                                "processed_bytes": 1024,
                                "queued_pipeline_tasks": 0,
                                "running_pipeline_tasks": 1,
                                "completed_pipeline_tasks": 0,
                            }
                        ],
                    }
                ],
            },
            {
                "processed_rows": 1000,
                "processed_bytes": 1024,
                "state": "FINISHED",
                "fragments": [],
            },
        ]
    )

    stream = TTYStream()
    renderer = ProgressRenderer(lambda: next(snapshots), stream=stream)
    renderer.update(force=True)
    renderer.update(force=True)

    output = stream.getvalue()
    assert "\x1b[3A" in output
    assert "\x1b[J" in output
    assert "\n\x1b[2K" in output
    assert not output.endswith("\n")


def test_progress_dynamic_renderer_adds_newline_only_on_finish(monkeypatch):
    monkeypatch.delenv("VANE_PROGRESS", raising=False)

    class TTYStream(io.StringIO):
        def isatty(self) -> bool:
            return True

    times = iter([100.0, 101.0, 102.0])
    monkeypatch.setattr(progress_mod.time, "time", lambda: next(times))

    snapshot = {
        "processed_rows": 1000,
        "processed_bytes": 1024,
        "state": "FINISHED",
        "fragments": [],
    }

    stream = TTYStream()
    renderer = ProgressRenderer(lambda: snapshot, stream=stream)
    renderer.update(force=True)
    assert not stream.getvalue().endswith("\n")

    renderer.finish()
    assert stream.getvalue().endswith("\n")


def test_progress_auto_uses_log_mode_for_non_tty(monkeypatch):
    monkeypatch.delenv("VANE_PROGRESS", raising=False)

    snapshot = {
        "processed_rows": 1000,
        "processed_bytes": 1024,
        "state": "FINISHED",
        "fragments": [],
    }

    stream = io.StringIO()
    renderer = ProgressRenderer(lambda: snapshot, stream=stream)
    renderer.update(force=True)

    output = stream.getvalue()
    assert "\x1b[" not in output
    assert output.endswith("\n")


def test_fte_task_execution_extracts_terminal_task_stats_from_native_result_tuple():
    execution = FteTaskExecution(
        {
            "task_id": {
                "query_id": "query-a",
                "fragment_execution_id": 1,
                "partition_id": 0,
                "attempt_id": 0,
            },
            "fragment_id": "fragment-a",
        },
        execute_fn=None,  # type: ignore[arg-type]
        default_task_memory_bytes=1,
    )

    stats = execution._extract_task_stats(
        (
            ["payload"],
            [{"num_rows": 3, "size_bytes": 11}],
            None,
            [],
            0,
            None,
            {"processed_input_rows": 3, "processed_input_bytes": 11},
        )
    )

    assert stats == {"processed_input_rows": 3, "processed_input_bytes": 11}


def test_exchange_source_task_descriptor_preserves_file_rows_for_progress():
    handles = [
        {
            "partition_id": 0,
            "attempt_id": 1,
            "node_id": "node-a",
            "flight_port": 5010,
            "files": [{"path": "shuffle__sink_0__attempt_1", "rows": 37, "file_size": 4096}],
        }
    ]

    raw = vane.ray_cxx.make_exchange_source_task_descriptor_for_test(handles, [0], 1, 1)

    assert vane.ray_cxx.exchange_source_task_source_handles_for_test(raw) == handles
