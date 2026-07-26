# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pickle
import uuid

import pytest

import duckdb
from duckdb.runners.ray.query_graph_builder import build_query_execution_graph


def _physical_plan(relation, con, prefix):
    return duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        f"{prefix}-{uuid.uuid4().hex[:8]}",
    ).to_physical_plan(con)


def _parquet_relation(con, tmp_path):
    path = tmp_path / "input.parquet"
    con.execute(f"COPY (SELECT i::BIGINT AS x FROM range(8) tbl(i)) TO '{path}' (FORMAT PARQUET)")
    return con.read_parquet(str(path))


def test_physical_plan_exports_complete_deterministic_execution_stage_metadata(tmp_path):
    con = duckdb.connect()
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
    con = duckdb.connect()
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

    con = duckdb.connect()
    try:
        relation = _parquet_relation(con, tmp_path).map_batches(
            Identity,
            schema={"y": duckdb.sqltypes.BIGINT},
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

    con = duckdb.connect()
    try:
        relation = (
            _parquet_relation(con, tmp_path)
            .map_batches(
                first,
                schema={"first": duckdb.sqltypes.BIGINT},
                execution_backend="ray_task",
            )
            .map_batches(
                Second,
                schema={"second": duckdb.sqltypes.BIGINT},
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


def test_stage_collection_pairs_reordered_branch_udfs_by_stable_identity(monkeypatch, tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    def left_udf(table):
        return pa.table({"id": table.column("id"), "left_value": table.column("left_value")})

    def right_udf(table):
        return pa.table({"id": table.column("id"), "right_value": table.column("right_value")})

    monkeypatch.setenv("VANE_DISTRIBUTED_JOIN_STRATEGY", "broadcast_right")
    left_path = tmp_path / "left.parquet"
    right_path = tmp_path / "right.parquet"
    con = duckdb.connect()
    try:
        con.execute(
            f"COPY (SELECT i::BIGINT AS id, (i + 10)::BIGINT AS left_value FROM range(8) tbl(i)) "
            f"TO '{left_path}' (FORMAT PARQUET)"
        )
        con.execute(
            f"COPY (SELECT i::BIGINT AS id, (i + 20)::BIGINT AS right_value FROM range(8) tbl(i)) "
            f"TO '{right_path}' (FORMAT PARQUET)"
        )
        left = (
            con.read_parquet(str(left_path))
            .map_batches(
                left_udf,
                schema={"id": duckdb.sqltypes.BIGINT, "left_value": duckdb.sqltypes.BIGINT},
                execution_backend="ray_task",
                cpus=1.0,
                memory_bytes=1 << 30,
                streaming_breaker=False,
            )
            .set_alias("l")
        )
        right = (
            con.read_parquet(str(right_path))
            .map_batches(
                right_udf,
                schema={"id": duckdb.sqltypes.BIGINT, "right_value": duckdb.sqltypes.BIGINT},
                execution_backend="ray_task",
                cpus=3.0,
                memory_bytes=3 << 30,
                streaming_breaker=False,
            )
            .set_alias("r")
        )
        relation = left.join(right, "l.id = r.id").project("l.id, l.left_value, r.right_value")
        plan = _physical_plan(relation, con, "graph-branch-udfs")

        metadata = plan.collect_execution_stages(conn=con)
        assert metadata == plan.collect_execution_stages(conn=con)
        assert "Swapped: true" in plan.repr_ascii(False)

        udf_nodes = [node for node in metadata["nodes"] if node["udf_payload"] is not None]
        by_name = {node["udf_payload"]["udf_name"].rsplit(".", 1)[-1]: node for node in udf_nodes}
        assert by_name.keys() == {"left_udf", "right_udf"}
        left_node = by_name["left_udf"]
        right_node = by_name["right_udf"]

        # Physical translation assigns the left branch first even though the
        # broadcast pipeline exposes the right branch first.
        assert int(left_node["node_id"]) < int(right_node["node_id"])
        assert left_node["udf_payload"]["cpus"] == 1.0
        assert right_node["udf_payload"]["cpus"] == 3.0
        assert left_node["udf_payload"]["memory_bytes"] == 1 << 30
        assert right_node["udf_payload"]["memory_bytes"] == 3 << 30
        assert left_node["udf_payload"]["stage_id"].endswith(f":node:{left_node['node_id']}:udf")
        assert right_node["udf_payload"]["stage_id"].endswith(f":node:{right_node['node_id']}:udf")
        assert left_node["udf_payload"]["_vane_udf_operator_id"] != right_node["udf_payload"]["_vane_udf_operator_id"]

        replay_by_name = {
            node["payload"]["udf_name"].rsplit(".", 1)[-1]: node for node in plan.collect_udf_nodes(conn=con)
        }
        assert replay_by_name["left_udf"]["payload"]["stage_id"] == left_node["udf_payload"]["stage_id"]
        assert replay_by_name["right_udf"]["payload"]["stage_id"] == right_node["udf_payload"]["stage_id"]

        graph = build_query_execution_graph(metadata, env={})
        left_stage = graph.stage_by_id(left_node["udf_payload"]["stage_id"])
        right_stage = graph.stage_by_id(right_node["udf_payload"]["stage_id"])
        assert left_stage.per_task.cpu == 1.0
        assert right_stage.per_task.cpu == 3.0
        assert left_stage.per_task.heap_bytes == 1 << 30
        assert right_stage.per_task.heap_bytes == 3 << 30

        operator_ids = [
            left_node["udf_payload"]["_vane_udf_operator_id"],
            right_node["udf_payload"]["_vane_udf_operator_id"],
        ]
        serialized_plan = pickle.dumps(plan)
        assert all(serialized_plan.count(operator_id.encode()) == 1 for operator_id in operator_ids)
        corrupted_plan = pickle.loads(
            serialized_plan.replace(operator_ids[1].encode(), operator_ids[0].encode())
        ).clone(con)
        with pytest.raises(duckdb.InvalidInputException, match="duplicate physical UDF operator identity"):
            corrupted_plan.collect_execution_stages(conn=con)
    finally:
        con.close()
