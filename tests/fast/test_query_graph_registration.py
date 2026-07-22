# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import uuid

import pytest

import vane
from vane.runners.ray.query_graph_builder import build_query_execution_graph


def _physical_plan(relation, con, prefix):
    return vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        f"{prefix}-{uuid.uuid4().hex[:8]}",
    ).to_physical_plan(con)


def _parquet_relation(con, tmp_path):
    path = tmp_path / "input.parquet"
    con.execute(f"COPY (SELECT i::BIGINT AS x FROM range(8) tbl(i)) TO '{path}' (FORMAT PARQUET)")
    return con.read_parquet(str(path))


def test_physical_plan_exports_complete_deterministic_execution_stage_metadata(tmp_path):
    con = vane.connect()
    try:
        plan = _physical_plan(_parquet_relation(con, tmp_path), con, "graph-plain")

        first = plan.collect_execution_stages(conn=con)
        second = plan.collect_execution_stages(conn=con)
        graph = build_query_execution_graph(first, env={})

        assert first == second
        assert first["query_id"] == plan.idx()
        assert first["nodes"]
        assert first["terminal_node_ids"]
        assert graph.query_id == plan.idx()
        assert all(node["node_id"] for node in first["nodes"])
        assert all(node["num_partitions"] >= 1 for node in first["nodes"])
    finally:
        con.close()


def test_stage_collection_does_not_treat_generic_inout_as_python_udf(tmp_path):
    con = vane.connect()
    try:
        path = tmp_path / "generic_inout.parquet"
        con.execute(f"COPY (SELECT i::BIGINT AS x FROM range(2) tbl(i)) TO '{path}' (FORMAT PARQUET)")
        con.execute("SET scalar_subquery_error_on_multiple_rows=false")
        relation = con.sql(f"SELECT * FROM unnest((SELECT [x, x + 1] FROM read_parquet('{path}')))")
        plan = _physical_plan(relation, con, "graph-generic-inout")

        metadata = plan.collect_execution_stages(conn=con)
        graph = build_query_execution_graph(metadata, env={})

        assert graph.query_id == plan.idx()
        assert all(node["udf_payload"] is None for node in metadata["nodes"])
    finally:
        con.close()


def test_stage_collection_preannotates_ray_udf_payload_on_original_plan(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    class Identity:
        def __call__(self, table):
            return pa.table({"y": table.column(0)})

    con = vane.connect()
    try:
        relation = _parquet_relation(con, tmp_path).map_batches(
            Identity,
            schema={"y": vane.sqltypes.BIGINT},
            execution_backend="ray_actor",
            actor_number=1,
            gpus=0.0,
            streaming_breaker=False,
        )
        plan = _physical_plan(relation, con, "graph-udf")

        metadata = plan.collect_execution_stages(conn=con)
        udf_nodes = [node for node in metadata["nodes"] if node["udf_payload"] is not None]
        replay_nodes = plan.collect_udf_nodes(conn=con)

        assert len(udf_nodes) == 1
        assert len(replay_nodes) == 1
        payload = udf_nodes[0]["udf_payload"]
        replay_payload = replay_nodes[0]["payload"]
        assert payload["query_id"] == plan.idx()
        assert payload["stage_id"].endswith(f":node:{udf_nodes[0]['node_id']}:udf")
        assert replay_payload["query_id"] == payload["query_id"]
        assert replay_payload["stage_id"] == payload["stage_id"]
        graph = build_query_execution_graph(metadata, env={})
        assert graph.stage_by_id(payload["stage_id"]).backend == "ray_actor"
    finally:
        con.close()


def test_stage_collection_preserves_distinct_stage_identity_for_nested_udfs(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    def first(table):
        return pa.table({"first": table.column(0)})

    class Second:
        def __call__(self, table):
            return pa.table({"second": table.column(0)})

    con = vane.connect()
    try:
        relation = (
            _parquet_relation(con, tmp_path)
            .map_batches(
                first,
                schema={"first": vane.sqltypes.BIGINT},
                execution_backend="ray_task",
            )
            .map_batches(
                Second,
                schema={"second": vane.sqltypes.BIGINT},
                execution_backend="ray_actor",
                actor_number=1,
                gpus=0.0,
                streaming_breaker=False,
            )
        )
        plan = _physical_plan(relation, con, "graph-nested-udf")

        metadata = plan.collect_execution_stages(conn=con)
        udf_nodes = [node for node in metadata["nodes"] if node["udf_payload"] is not None]
        replay_nodes = plan.collect_udf_nodes(conn=con)

        assert len(udf_nodes) == 2
        assert len(replay_nodes) == 2
        metadata_by_stage = {node["udf_payload"]["stage_id"]: node for node in udf_nodes}
        replay_by_stage = {node["payload"]["stage_id"]: node for node in replay_nodes}
        assert metadata_by_stage.keys() == replay_by_stage.keys()
        assert {node["udf_payload"]["execution_backend"] for node in udf_nodes} == {
            "ray_task",
            "ray_actor",
        }
        for stage_id, node in metadata_by_stage.items():
            assert stage_id.endswith(f":node:{node['node_id']}:udf")
            assert replay_by_stage[stage_id]["payload"]["execution_backend"] == node["udf_payload"]["execution_backend"]
    finally:
        con.close()
