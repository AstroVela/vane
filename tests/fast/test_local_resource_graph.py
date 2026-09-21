# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import itertools
import json
import subprocess
import sys
import uuid

import pytest

import vane
from vane.execution.local_resource_graph import LocalResourceGraphAdapter, build_local_resource_graph
from vane.execution.request_admission import RequestAdmissionLimits
from vane.execution.resource_graph_metadata import ResourceGraphMetadataProvider
from vane.execution.udf_local_model import LocalModelRuntime
from vane.execution.udf_runtime_admission import TaskAdmissionLimits
from vane.runners.ray.query_resource_graph_builder import build_query_resource_graph
from vane.runners.ray.resource_graph_adapter import RayResourceGraphAdapter


def _plan(relation, conn):
    return vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(conn)


def _identity(table):
    return table


def _collect(provider: ResourceGraphMetadataProvider, conn):
    return provider.collect_resource_graph_metadata(conn=conn)


@pytest.mark.parametrize(
    "backend,adapter",
    [
        ("subprocess_task", LocalResourceGraphAdapter),
        ("subprocess_actor", LocalResourceGraphAdapter),
        ("ray_task", RayResourceGraphAdapter),
        ("ray_actor", RayResourceGraphAdapter),
    ],
)
def test_backend_adapters_share_native_metadata_contract(backend, adapter):
    class Identity:
        def __call__(self, table):
            return table

    with vane.connect() as conn:
        plan = _plan(
            conn.sql("SELECT 3::INTEGER AS x").map_batches(
                Identity if backend.endswith("actor") else _identity,
                schema={"x": vane.sqltypes.INTEGER},
                execution_backend=backend,
                **({"actor_number": 1} if backend.endswith("actor") else {}),
            ),
            conn,
        )
        before = plan.collect_udf_nodes(conn=conn)
        metadata = _collect(adapter(plan), conn)
        assert metadata == _collect(adapter(plan), conn)
        assert set(metadata) == {"query_id", "nodes", "terminal_node_ids", "udf_node_ids"}
        physical = {str(node["node_id"]): node["payload"] for node in plan.collect_udf_nodes(conn=conn)}
        for node in metadata["nodes"]:
            if node["udf_payload"] is not None:
                assert node["udf_payload"] == physical[metadata["udf_node_ids"][node["node_id"]]]
        if backend.startswith("subprocess"):
            assert plan.collect_udf_nodes(conn=conn) == before
            assert all("query_id" not in node["payload"] for node in before)
            build_local_resource_graph(metadata, query_id="execution")
        else:
            graph = build_query_resource_graph(metadata, env={})
            for payload in physical.values():
                assert payload["query_id"] == plan.idx()
                assert graph.unit_by_id(payload["resource_unit_id"]).backend == backend
            legacy = plan.collect_query_resource_graph_metadata(conn=conn)
            assert legacy == {key: value for key, value in metadata.items() if key != "udf_node_ids"}


def test_readonly_collection_restores_payloads_after_conversion_failure():
    with vane.connect() as conn:
        plan = _plan(
            conn.sql("SELECT 3::INTEGER AS x").map_batches(
                _identity,
                schema={"x": vane.sqltypes.INTEGER},
                execution_backend="subprocess_task",
            ),
            conn,
        )
        before = plan.collect_udf_nodes(conn=conn)
        with pytest.raises(vane.InternalException, match="Connection object"):
            _collect(LocalResourceGraphAdapter(plan), object())
        assert plan.collect_udf_nodes(conn=conn) == before
        metadata = _collect(LocalResourceGraphAdapter(plan), conn)
        assert metadata["udf_node_ids"]
        assert plan.collect_udf_nodes(conn=conn) == before


@pytest.mark.parametrize(
    ("transform", "has_barrier"),
    [
        (lambda relation: relation.order("x"), True),
        (lambda relation: relation.aggregate("sum(x) AS x"), False),
        (lambda relation: relation.order("x").limit(3), True),
    ],
)
def test_local_and_ray_share_structural_barriers_and_phase_calculation(tmp_path, transform, has_barrier):
    with vane.connect() as conn:
        path = tmp_path / "input.parquet"
        conn.execute(f"COPY (SELECT i AS x FROM range(8) tbl(i)) TO '{path}' (FORMAT PARQUET)")
        plan = _plan(transform(conn.read_parquet(str(path)).repartition(2)), conn)
        local_metadata = _collect(LocalResourceGraphAdapter(plan), conn)
        ray_metadata = _collect(RayResourceGraphAdapter(plan), conn)
        assert local_metadata == ray_metadata
        local = build_local_resource_graph(local_metadata, query_id=plan.idx())
        ray = build_query_resource_graph(ray_metadata, env={})
        assert bool(local.materialization_barriers) == has_barrier
        assert local.materialization_barriers == ray.materialization_barriers
        assert local.topological_unit_ids() == ray.topological_unit_ids()
        ids = [barrier.barrier_id for barrier in local.materialization_barriers]
        for count in range(len(ids) + 1):
            for completed in itertools.combinations(ids, count):
                assert local.eligible_resource_unit_ids(set(completed)) == ray.eligible_resource_unit_ids(
                    set(completed)
                )
                assert local.frontier_materialization_barriers(set(completed)) == ray.frontier_materialization_barriers(
                    set(completed)
                )


def test_reordered_branch_bindings_reach_the_matching_native_executor(monkeypatch):
    from vane.execution import udf_subprocess

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    monkeypatch.setenv("VANE_DISTRIBUTED_JOIN_STRATEGY", "broadcast_right")
    seen = []
    initialize = udf_subprocess.UDFExecutor.__init__

    def observe(executor, payload, options=None):
        initialize(executor, payload, options)
        seen.append((payload["cpus"], executor.resource_identity()))

    monkeypatch.setattr(udf_subprocess.UDFExecutor, "__init__", observe)
    with vane.connect() as conn:
        left = (
            conn.sql("SELECT * FROM (VALUES (0), (1), (2), (3)) t(x)")
            .map_batches(
                _identity,
                schema={"x": vane.sqltypes.INTEGER},
                execution_backend="subprocess_task",
                cpus=1,
            )
            .set_alias("l")
        )
        right = (
            conn.sql("SELECT * FROM (VALUES (0), (1), (2), (3)) t(x)")
            .map_batches(
                _identity,
                schema={"x": vane.sqltypes.INTEGER},
                execution_backend="subprocess_task",
                cpus=2,
            )
            .set_alias("r")
        )
        plan = _plan(left.join(right, "l.x = r.x").project("l.x"), conn)
        metadata = _collect(LocalResourceGraphAdapter(plan), conn)
        before = plan.collect_udf_nodes(conn=conn)
        expected = {
            node["udf_payload"]["cpus"]: f"node:{node['node_id']}:udf"
            for node in metadata["nodes"]
            if node["udf_payload"] is not None
        }
        physical = {str(node["node_id"]): node["payload"] for node in before}
        for node in metadata["nodes"]:
            if node["udf_payload"] is not None:
                assert node["udf_payload"] == physical[metadata["udf_node_ids"][node["node_id"]]]
        with LocalModelRuntime(
            session_id=plan.session_id(),
            session_config=plan.session_config(),
            track_graph=True,
            request_limit=RequestAdmissionLimits(1, 1),
        ) as runtime:
            request = runtime.request()
            result = request.execute(plan, {}, conn=conn)
            assert sorted(
                value for table in result.partition_payloads for value in table.column(0).to_pylist()
            ) == list(range(4))
            assert {cpu: identity["physical_node_id"] for cpu, identity in seen} == expected
            snapshot = request.resource_graph_snapshot()
            assert {identity["query_id"] for _, identity in seen} == {snapshot["graph"]["query_id"]}
            assert snapshot["phase_tracking"] == "structural_only"
            assert snapshot["graph"]["materialization_barriers"]
            assert "function_pickle" not in json.dumps(snapshot)
            assert not runtime.resource_snapshot()["prepared_query_graphs"]
        assert plan.collect_udf_nodes(conn=conn) == before


@pytest.mark.parametrize("task_limit", [None, TaskAdmissionLimits(2, 2)], ids=["graph_only", "task_limited"])
def test_chained_udfs_have_independent_units_and_each_execution_has_a_new_identity(monkeypatch, task_limit):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as conn:
        relation = conn.sql("SELECT 7::INTEGER AS x")
        for _ in range(2):
            relation = relation.map_batches(
                _identity, schema={"x": vane.sqltypes.INTEGER}, execution_backend="subprocess_task"
            )
        plan = _plan(relation, conn)
        with LocalModelRuntime(
            session_id=plan.session_id(),
            session_config=plan.session_config(),
            track_graph=True,
            task_limit=task_limit,
        ) as runtime:
            query_ids = set()
            for _ in range(2):
                resources = runtime.prepare(plan, {}, conn=conn)
                try:
                    (snapshot,) = runtime.resource_snapshot()["prepared_query_graphs"]
                    graph = snapshot["graph"]
                    query_ids.add(graph["query_id"])
                    udf_units = [unit for unit in graph["units"] if unit["backend"] == "subprocess_task"]
                    assert len({unit["resource_unit_id"] for unit in udf_units}) == 2
                    result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(conn, plan)
                    assert [value for table in result.partition_payloads for value in table.column(0).to_pylist()] == [
                        7
                    ]
                finally:
                    for resource in resources:
                        resource.shutdown()
                assert not runtime.resource_snapshot()["prepared_query_graphs"]
            assert len(query_ids) == 2


@pytest.mark.parametrize("task_limit", [None, TaskAdmissionLimits(2, 2)], ids=["graph_only", "task_limited"])
def test_graph_publication_failure_retires_diagnostics(task_limit):
    with vane.connect() as conn:
        native = _plan(
            conn.sql("SELECT 7::INTEGER AS x").map_batches(
                _identity,
                schema={"x": vane.sqltypes.INTEGER},
                execution_backend="subprocess_task",
            ),
            conn,
        )

        class Plan:
            session_id = native.session_id
            session_config = native.session_config
            collect_udf_nodes = native.collect_udf_nodes
            collect_resource_graph_metadata = native.collect_resource_graph_metadata

            def set_udf_actor_handles(self, *args, **kwargs):
                raise RuntimeError("publication failed")

        with LocalModelRuntime(
            session_id=native.session_id(),
            session_config=native.session_config(),
            track_graph=True,
            task_limit=task_limit,
        ) as runtime:
            with pytest.raises(RuntimeError, match="publication failed"):
                runtime.prepare(Plan(), {}, conn=conn)
            assert not runtime.resource_snapshot()["prepared_query_graphs"]
            runtime.drain()
            with pytest.raises(RuntimeError, match="draining"):
                runtime.prepare(native, {}, conn=conn)
            assert not runtime.resource_snapshot()["prepared_query_graphs"]


def test_binding_map_rejects_missing_or_duplicate_physical_udf_ids():
    with vane.connect() as conn:
        relation = conn.sql("SELECT 7::INTEGER AS x")
        for _ in range(2):
            relation = relation.map_batches(
                _identity, schema={"x": vane.sqltypes.INTEGER}, execution_backend="subprocess_task"
            )
        metadata = _collect(LocalResourceGraphAdapter(_plan(relation, conn)), conn)
        for mapping in ({}, dict.fromkeys(metadata["udf_node_ids"], "0")):
            invalid = copy.deepcopy(metadata)
            invalid["udf_node_ids"] = mapping
            with pytest.raises(ValueError, match="udf_node_ids"):
                build_local_resource_graph(invalid, query_id="execution")


def test_local_graph_import_does_not_import_ray_policy():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import sys
import vane
before = set(sys.modules)
from vane.execution.local_resource_graph import LocalResourceGraphAdapter
assert 'vane.runners.ray.query_resource_graph' not in sys.modules
assert 'vane.runners.ray.cluster_resource_coordinator' not in sys.modules
assert not any(name == 'ray' or name.startswith('ray.') for name in set(sys.modules) - before)
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("backend", ["ray_task", "ray_actor"])
def test_local_resource_context_is_rejected_by_ray_routing(backend):
    from vane.execution.udf import build_executor

    with pytest.raises(ValueError, match="local resource graphs require local subprocess UDFs"):
        build_executor({"execution_backend": backend, "call_mode": "map_batches"}, {"local_resource_unit": object()})
