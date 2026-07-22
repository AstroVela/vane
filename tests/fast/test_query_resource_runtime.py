# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

from vane.runners.ray.query_execution_graph import (
    NodeResourceAllocation,
    QueryAllocation,
    QueryExecutionGraph,
    ResourceVector,
    StageResourceSpec,
)
from vane.runners.ray.query_resource_runtime import (
    clear_query_resource_managers,
    get_query_resource_manager,
    query_resource_manager_snapshot,
    register_query_graph,
    release_query_resource_manager,
)


@pytest.fixture(autouse=True)
def _clear_registry():
    clear_query_resource_managers()
    yield
    clear_query_resource_managers()


def _graph(digest="sha256:a"):
    stage = StageResourceSpec(
        query_id="q",
        stage_id="stage:f:scan",
        physical_node_id="scan",
        stage_kind="fte",
        backend="ray_worker",
        input_stage_ids=(),
        per_task=ResourceVector(cpu=1, heap_bytes=100),
        target_output_block_bytes=10,
        generator_buffer_blocks=2,
        max_concurrency=4,
    )
    return QueryExecutionGraph("q", digest, (stage,), (stage.stage_id,))


def _allocation(generation=1):
    resources = ResourceVector(cpu=4, heap_bytes=1_000, object_store_bytes=1_000)
    return QueryAllocation(
        resources=resources,
        node_allocations=(NodeResourceAllocation(node_id="node-a", resources=resources),),
        actor_placements=(),
        generation=generation,
    )


def test_runtime_never_lazily_creates_manager_before_graph_registration():
    with pytest.raises(KeyError, match="query graph is not registered"):
        get_query_resource_manager("q")
    assert query_resource_manager_snapshot("q") == {}


def test_runtime_registers_graph_atomically_and_rejects_every_duplicate():
    manager = register_query_graph(_graph(), _allocation())

    assert get_query_resource_manager("q") is manager
    assert query_resource_manager_snapshot("q")["graph"]["plan_digest"] == "sha256:a"
    with pytest.raises(ValueError, match="already registered"):
        register_query_graph(_graph(), _allocation())
    with pytest.raises(ValueError, match="already registered"):
        register_query_graph(_graph("sha256:different"), _allocation())


def test_runtime_graph_validation_finishes_before_registry_visibility():
    graph = _graph()
    too_small_resources = ResourceVector(cpu=1, heap_bytes=99, object_store_bytes=1_000)
    too_small = QueryAllocation(
        resources=too_small_resources,
        node_allocations=(NodeResourceAllocation(node_id="node-a", resources=too_small_resources),),
        actor_placements=(),
        generation=1,
    )

    with pytest.raises(ValueError, match="heap_bytes"):
        register_query_graph(graph, too_small)
    assert query_resource_manager_snapshot("q") == {}


def test_runtime_release_cancels_and_removes_manager_idempotently():
    register_query_graph(_graph(), _allocation())

    first = release_query_resource_manager("q", reason="completed")
    second = release_query_resource_manager("q", reason="completed")

    assert first["released"] is True
    assert first["task_lease_count"] == 0
    assert first["output_lease_count"] == 0
    assert second == {"released": False, "task_lease_count": 0, "output_lease_count": 0}
    assert query_resource_manager_snapshot("q") == {}
