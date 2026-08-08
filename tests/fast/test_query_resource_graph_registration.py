# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pickle
import uuid

import pytest

import vane
from vane.runners.ray.query_resource_graph_builder import (
    build_query_resource_graph,
    native_fragment_unit_id_for_node,
    udf_unit_id_for_node,
)


def _physical_plan(relation, con, prefix):
    return vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        f"{prefix}-{uuid.uuid4().hex[:8]}",
    ).to_physical_plan(con)


def _parquet_relation(con, tmp_path):
    path = tmp_path / "input.parquet"
    con.execute(f"COPY (SELECT i::BIGINT AS x FROM range(8) tbl(i)) TO '{path}' (FORMAT PARQUET)")
    return con.read_parquet(str(path))


def test_physical_plan_exports_complete_deterministic_resource_unit_metadata(tmp_path):
    con = vane.connect()
    try:
        plan = _physical_plan(_parquet_relation(con, tmp_path), con, "graph-plain")

        first = plan.collect_query_resource_graph_metadata(conn=con)
        second = plan.collect_query_resource_graph_metadata(conn=con)
        graph = build_query_resource_graph(first, env={})

        assert first == second
        assert first["query_id"] == plan.idx()
        assert first["nodes"]
        assert first["terminal_node_ids"]
        assert graph.query_id == plan.idx()
        assert all(node["node_id"] for node in first["nodes"])
        assert all(node["num_partitions"] >= 1 for node in first["nodes"])
        assert all(
            bool(node["materialized_input_node_ids"]) == node["is_materialization_barrier"] for node in first["nodes"]
        )
    finally:
        con.close()


@pytest.mark.parametrize(
    ("transform", "expected_nodes"),
    [
        (lambda relation: relation.repartition(4), {"Repartition": False}),
        (
            lambda relation: relation.repartition(4).order("x"),
            {"Repartition": False, "OrderBy": True},
        ),
        (
            lambda relation: relation.repartition(4).limit(3),
            {"Repartition": False, "StreamingLimit": True},
        ),
    ],
)
def test_physical_plan_marks_only_true_materialization_barriers(tmp_path, transform, expected_nodes):
    con = vane.connect()
    try:
        plan = _physical_plan(transform(_parquet_relation(con, tmp_path)), con, "graph-barrier")
        metadata = plan.collect_query_resource_graph_metadata(conn=con)
        barrier_by_name = {
            node["node_name"]: node["is_materialization_barrier"]
            for node in metadata["nodes"]
            if node["node_name"] in expected_nodes
        }

        assert barrier_by_name == expected_nodes
        for node in metadata["nodes"]:
            if node["node_name"] not in expected_nodes:
                continue
            assert bool(node["materialized_input_node_ids"]) == expected_nodes[node["node_name"]]
            assert set(node["materialized_input_node_ids"]).issubset(node["input_node_ids"])
        graph = build_query_resource_graph(metadata, env={})
        assert {barrier.physical_node_id for barrier in graph.materialization_barriers} == {
            node["node_id"] for node in metadata["nodes"] if node["is_materialization_barrier"]
        }
    finally:
        con.close()


def test_broadcast_join_barrier_materializes_only_broadcaster_input(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("VANE_DISTRIBUTED_JOIN_STRATEGY", "broadcast")
    monkeypatch.setenv("VANE_DISTRIBUTED_BROADCAST_JOIN_RECEIVER_REPARTITION", "0")
    con = vane.connect()
    try:
        path = tmp_path / "broadcast_input.parquet"
        con.execute(f"COPY (SELECT i::BIGINT AS x FROM range(8) tbl(i)) TO '{path}' (FORMAT PARQUET)")
        relation = con.sql(f"SELECT l.x FROM read_parquet('{path}') l JOIN read_parquet('{path}') r ON l.x = r.x")
        plan = _physical_plan(relation, con, "graph-broadcast")
        metadata = plan.collect_query_resource_graph_metadata(conn=con)
        broadcast = next(node for node in metadata["nodes"] if node["node_name"] == "BroadcastJoin")

        assert len(broadcast["input_node_ids"]) == 2
        assert len(broadcast["materialized_input_node_ids"]) == 1
        assert set(broadcast["materialized_input_node_ids"]).issubset(broadcast["input_node_ids"])

        graph = build_query_resource_graph(metadata, env={})
        barrier = graph.barrier_for_physical_node(broadcast["node_id"])
        assert barrier.materialized_input_unit_ids == (
            native_fragment_unit_id_for_node(
                graph.query_id,
                broadcast["materialized_input_node_ids"][0],
            ),
        )
    finally:
        con.close()


def test_resource_unit_collection_does_not_treat_generic_inout_as_python_udf(tmp_path):
    con = vane.connect()
    try:
        path = tmp_path / "generic_inout.parquet"
        con.execute(f"COPY (SELECT i::BIGINT AS x FROM range(2) tbl(i)) TO '{path}' (FORMAT PARQUET)")
        con.execute("SET scalar_subquery_error_on_multiple_rows=false")
        relation = con.sql(f"SELECT * FROM unnest((SELECT [x, x + 1] FROM read_parquet('{path}')))")
        plan = _physical_plan(relation, con, "graph-generic-inout")

        metadata = plan.collect_query_resource_graph_metadata(conn=con)
        graph = build_query_resource_graph(metadata, env={})

        assert graph.query_id == plan.idx()
        assert all(node["udf_payload"] is None for node in metadata["nodes"])
    finally:
        con.close()


def test_resource_unit_collection_preannotates_ray_udf_payload_on_original_plan(tmp_path):
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
        )
        plan = _physical_plan(relation, con, "graph-udf")

        metadata = plan.collect_query_resource_graph_metadata(conn=con)
        udf_nodes = [node for node in metadata["nodes"] if node["udf_payload"] is not None]
        replay_nodes = plan.collect_udf_nodes(conn=con)

        assert len(udf_nodes) == 1
        assert len(replay_nodes) == 1
        payload = udf_nodes[0]["udf_payload"]
        replay_payload = replay_nodes[0]["payload"]
        assert payload["query_id"] == plan.idx()
        assert payload["resource_unit_id"] == udf_unit_id_for_node(plan.idx(), udf_nodes[0]["node_id"])
        assert replay_payload["query_id"] == payload["query_id"]
        assert replay_payload["resource_unit_id"] == payload["resource_unit_id"]
        graph = build_query_resource_graph(metadata, env={})
        assert graph.unit_by_id(payload["resource_unit_id"]).backend == "ray_actor"
    finally:
        con.close()


def test_resource_unit_collection_preserves_distinct_identity_for_nested_udfs(tmp_path):
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
            )
        )
        plan = _physical_plan(relation, con, "graph-nested-udf")

        metadata = plan.collect_query_resource_graph_metadata(conn=con)
        udf_nodes = [node for node in metadata["nodes"] if node["udf_payload"] is not None]
        replay_nodes = plan.collect_udf_nodes(conn=con)

        assert len(udf_nodes) == 2
        assert len(replay_nodes) == 2
        metadata_by_unit = {node["udf_payload"]["resource_unit_id"]: node for node in udf_nodes}
        replay_by_unit = {node["payload"]["resource_unit_id"]: node for node in replay_nodes}
        assert metadata_by_unit.keys() == replay_by_unit.keys()
        assert {node["udf_payload"]["execution_backend"] for node in udf_nodes} == {
            "ray_task",
            "ray_actor",
        }
        for resource_unit_id, node in metadata_by_unit.items():
            assert resource_unit_id == udf_unit_id_for_node(plan.idx(), node["node_id"])
            assert (
                replay_by_unit[resource_unit_id]["payload"]["execution_backend"]
                == node["udf_payload"]["execution_backend"]
            )
    finally:
        con.close()


def test_resource_unit_collection_pairs_reordered_branch_udfs_by_stable_identity(monkeypatch, tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    def left_udf(table):
        return pa.table({"id": table.column("id"), "left_value": table.column("left_value")})

    def right_udf(table):
        return pa.table({"id": table.column("id"), "right_value": table.column("right_value")})

    monkeypatch.setenv("VANE_DISTRIBUTED_JOIN_STRATEGY", "broadcast_right")
    left_path = tmp_path / "left.parquet"
    right_path = tmp_path / "right.parquet"
    con = vane.connect()
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
                schema={"id": vane.sqltypes.BIGINT, "left_value": vane.sqltypes.BIGINT},
                execution_backend="ray_task",
                cpus=1.0,
                memory_bytes=1 << 30,
            )
            .set_alias("l")
        )
        right = (
            con.read_parquet(str(right_path))
            .map_batches(
                right_udf,
                schema={"id": vane.sqltypes.BIGINT, "right_value": vane.sqltypes.BIGINT},
                execution_backend="ray_task",
                cpus=3.0,
                memory_bytes=3 << 30,
            )
            .set_alias("r")
        )
        relation = left.join(right, "l.id = r.id").project("l.id, l.left_value, r.right_value")
        plan = _physical_plan(relation, con, "graph-branch-udfs")

        metadata = plan.collect_query_resource_graph_metadata(conn=con)
        assert metadata == plan.collect_query_resource_graph_metadata(conn=con)
        assert "Broadcast side: right" in plan.repr_ascii(False)

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
        assert left_node["udf_payload"]["resource_unit_id"] == udf_unit_id_for_node(plan.idx(), left_node["node_id"])
        assert right_node["udf_payload"]["resource_unit_id"] == udf_unit_id_for_node(plan.idx(), right_node["node_id"])
        assert left_node["udf_payload"]["_vane_udf_operator_id"] != right_node["udf_payload"]["_vane_udf_operator_id"]

        replay_by_name = {
            node["payload"]["udf_name"].rsplit(".", 1)[-1]: node for node in plan.collect_udf_nodes(conn=con)
        }
        assert replay_by_name["left_udf"]["payload"]["resource_unit_id"] == left_node["udf_payload"]["resource_unit_id"]
        assert (
            replay_by_name["right_udf"]["payload"]["resource_unit_id"] == right_node["udf_payload"]["resource_unit_id"]
        )

        graph = build_query_resource_graph(metadata, env={})
        left_unit = graph.unit_by_id(left_node["udf_payload"]["resource_unit_id"])
        right_unit = graph.unit_by_id(right_node["udf_payload"]["resource_unit_id"])
        assert left_unit.per_task.cpu == 1.0
        assert right_unit.per_task.cpu == 3.0
        assert left_unit.per_task.heap_bytes == 1 << 30
        assert right_unit.per_task.heap_bytes == 3 << 30

        operator_ids = [
            left_node["udf_payload"]["_vane_udf_operator_id"],
            right_node["udf_payload"]["_vane_udf_operator_id"],
        ]
        serialized_plan = pickle.dumps(plan)
        assert all(serialized_plan.count(operator_id.encode()) == 1 for operator_id in operator_ids)
        corrupted_plan = pickle.loads(
            serialized_plan.replace(operator_ids[1].encode(), operator_ids[0].encode())
        ).clone(con)
        with pytest.raises(vane.InvalidInputException, match="duplicate physical UDF operator identity"):
            corrupted_plan.collect_query_resource_graph_metadata(conn=con)
    finally:
        con.close()
