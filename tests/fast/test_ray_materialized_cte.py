# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from uuid import uuid4

import pytest

import vane


def _physical_plan(con, sql):
    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(con.sql(sql), str(uuid4()))
    return logical.to_physical_plan(con)


@pytest.fixture
def ray_con(ray_local):
    vane.configure(runner="ray")
    con = vane.connect(config={"threads": 2})
    try:
        yield con
    finally:
        con.close()


def test_ray_cte_reads_one_materialization_from_two_references(ray_con, tmp_path):
    # Each invocation creates its own marker. A second evaluation of the same input fails,
    # even when a duplicated producer would otherwise return exactly the same join rows.
    def add_one(i):
        (Path(tmp_path) / f"row-{i}").touch(exist_ok=False)
        return i + 1

    definition = "SELECT cte_probe_udf(i) AS x FROM range(3) t(i)"
    sql = f"WITH c AS MATERIALIZED ({definition}) SELECT a.x, b.x FROM c a JOIN c b ON a.x=b.x ORDER BY a.x"
    vane.attach_function(
        add_one, alias="cte_probe_udf", parameters=["BIGINT"], return_dtype="BIGINT", connection=ray_con
    )
    try:
        plan = _physical_plan(ray_con, sql)
        graph = plan.collect_query_resource_graph_metadata(conn=ray_con)
        assert graph == plan.collect_query_resource_graph_metadata(conn=ray_con)
        nodes = graph["nodes"]
        (producer,) = [node for node in nodes if node["node_name"] == "CTEInput"]
        readers = [node for node in nodes if node["node_name"] == "CTEScan"]
        assert len(readers) == 2
        assert all(reader["input_node_ids"] == [producer["node_id"]] for reader in readers)
        assert len([node for node in nodes if node["udf_payload"] is not None]) == 1
        assert producer["is_materialization_barrier"]
        assert ray_con.sql(sql).fetchall() == [(1, 1), (2, 2), (3, 3)]
        assert sorted(p.name for p in tmp_path.glob("row-*")) == ["row-0", "row-1", "row-2"]
    finally:
        vane.detach_function("cte_probe_udf", connection=ray_con)


def test_ray_cte_types_empty_aggregate_and_binding_scopes(ray_con):
    cases = [
        ("WITH c AS MATERIALIZED (SELECT i AS x FROM range(3) t(i)) SELECT x FROM c ORDER BY x", [(0,), (1,), (2,)]),
        ("WITH c AS MATERIALIZED (SELECT i AS id, 'x' AS tag FROM range(3) t(i)) SELECT count(*) FROM c", [(3,)]),
        ("WITH c AS MATERIALIZED (SELECT i FROM range(0) t(i)) SELECT count(*) FROM c", [(0,)]),
        ("WITH c AS MATERIALIZED (SELECT count(*) AS n FROM range(0)) SELECT n FROM c", [(0,)]),
        (
            """WITH c AS MATERIALIZED (SELECT 1 AS x),
                 d AS MATERIALIZED (WITH c AS MATERIALIZED (SELECT 2 AS x) SELECT x FROM c)
            SELECT c.x, d.x FROM c CROSS JOIN d""",
            [(1, 2)],
        ),
        (
            """WITH c AS MATERIALIZED (SELECT i AS x FROM range(3) t(i))
            SELECT c.x FROM c JOIN range(2) t(i) ON x=i ORDER BY x""",
            [(0,), (1,)],
        ),
        (
            """SELECT * FROM
                 ((WITH c(x) AS MATERIALIZED (VALUES (2),(3)) SELECT x FROM c) UNION ALL SELECT 4) a
                 POSITIONAL JOIN
                 ((WITH c(y) AS MATERIALIZED (VALUES (5),(6)) SELECT y FROM c) UNION ALL SELECT 7) b""",
            [(2, 5), (3, 6), (4, 7)],
        ),
        (
            """WITH c(x) AS MATERIALIZED (VALUES (NULL::BIGINT), (7), (7))
            SELECT x FROM c ORDER BY x NULLS FIRST""",
            [(None,), (7,), (7,)],
        ),
        (
            """WITH c AS MATERIALIZED (SELECT 'b'::ENUM('z','b') AS x)
            SELECT x::VARCHAR, enum_range(x)::VARCHAR[] FROM c""",
            [("b", ["z", "b"])],
        ),
        (
            """WITH c AS MATERIALIZED (SELECT {'v': ['z'::ENUM('z','b'), NULL]} AS x)
            SELECT x.v::VARCHAR[] FROM c""",
            [(["z", None],)],
        ),
        (
            """WITH c AS MATERIALIZED (SELECT {'j':'{}'::JSON,'e':'z'::ENUM('z','b')} AS x)
            SELECT x.j::VARCHAR, x.e::VARCHAR FROM c""",
            [("{}", "z")],
        ),
    ]
    for sql, expected in cases:
        assert ray_con.sql(sql).fetchall() == expected, sql


def test_ray_cte_preserves_file_splits_and_optimizer_generated_cte(ray_con, tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import ray

    from vane.runners.ray.driver import RayQueryDriverClient

    # Inspect the live scheduler after the first result, while the query still owns its fragments.
    snapshots = []
    stream_plan = RayQueryDriverClient.stream_plan

    def observe_scan(client, plan):
        for index, partition in enumerate(stream_plan(client, plan)):
            if index == 0:
                snapshot = ray.get(
                    client.runner.progress_snapshot.remote(client._owner_id, plan.session_id(), plan.idx())
                )
                snapshots.append((str(plan.idx()), snapshot))
            yield partition

    monkeypatch.setattr(RayQueryDriverClient, "stream_plan", observe_scan)

    monkeypatch.setenv("VANE_RAY_SCAN_TASK_SIZE_GROUPING", "0")
    monkeypatch.setenv("VANE_RAY_SCAN_TASK_MIN_PARTITION_NUM", "4")
    monkeypatch.setenv("VANE_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION", "1")
    monkeypatch.setenv("VANE_RAY_MAX_TASK_BACKLOG", "1")
    for file_id in range(4):
        pq.write_table(pa.table({"x": list(range(file_id * 3, file_id * 3 + 3))}), tmp_path / f"{file_id}.parquet")
    source = "read_parquet('" + (tmp_path / "*.parquet").as_posix().replace("'", "''") + "')"
    direct = f"SELECT count(*) FROM {source}"
    explicit = f"WITH c AS MATERIALIZED (SELECT x FROM {source}) SELECT count(*) FROM c"
    repeated = f"SELECT DISTINCT x FROM {source}"
    automatic = f"SELECT a.x FROM ({repeated}) a JOIN ({repeated}) b USING (x) ORDER BY a.x"
    for sql, cte_count, expected in [
        (direct, 0, [(12,)]),
        (explicit, 1, [(12,)]),
        (automatic, 1, [(x,) for x in range(12)]),
    ]:
        plan = _physical_plan(ray_con, sql)
        graph = plan.collect_query_resource_graph_metadata(conn=ray_con)
        nodes = graph["nodes"]
        assert sum(n["node_name"] == "CTEInput" for n in nodes) == cte_count
        (scan,) = [n for n in nodes if n["node_name"] == "ScanSource"]
        assert scan["num_partitions"] == 4
        splits = plan.scan_split_batch_map()
        assert [len(batch) for batch in splits.values()] == [4]
        assert ray_con.sql(sql).fetchall() == expected
        ((query_id, snapshot),) = snapshots
        snapshots.clear()
        (scan_fragment,) = [f for f in snapshot["fragments"] if f["id"] == f"{query_id}:node:{scan['node_id']}"]
        assert scan_fragment["total_partitions"] == scan_fragment["completed_partitions"] == 4
    # The UNION continues pulling the external scan after its enclosing CTE returns the stream.
    assert ray_con.sql(
        f"WITH c(x) AS MATERIALIZED (VALUES (-1)) SELECT x FROM (SELECT x FROM c UNION ALL SELECT x FROM {source}) ORDER BY x"
    ).fetchall() == [(x,) for x in range(-1, 12)]
    for index in range(4, 70):
        (tmp_path / f"{index}.parquet").write_bytes((tmp_path / "0.parquet").read_bytes())
    wide = _physical_plan(ray_con, explicit).collect_query_resource_graph_metadata(conn=ray_con)["nodes"]
    assert [node["num_partitions"] for node in wide if node["node_name"] == "ScanSource"] == [70]
    assert [node["num_partitions"] for node in wide if node["node_name"] == "CTEInput"] == [64]


def test_ray_cte_planning_rejects_hidden_reader_and_prints_shared_nodes_once(ray_con, monkeypatch):
    for sql in [
        """WITH c(x) AS MATERIALIZED (VALUES (1),(2),(3))
           SELECT i,(SELECT count(*) FROM c WHERE x<i) FROM range(3) t(i)""",
        """SELECT i,(WITH c AS MATERIALIZED (SELECT x FROM range(3)t(x) WHERE x<i)
           SELECT count(*) FROM c) FROM range(3)t(i)""",
    ]:
        with pytest.raises(vane.InvalidInputException, match="inside an unsupported native subtree"):
            _physical_plan(ray_con, sql).collect_query_resource_graph_metadata(conn=ray_con)
    definitions = ["c0 AS MATERIALIZED (SELECT i AS x FROM range(3) t(i))"]
    for index in range(1, 9):
        name = f"c{index - 1}"
        definitions.append(f"c{index} AS MATERIALIZED (SELECT a.x FROM {name} a JOIN {name} b USING (x))")
    plan = _physical_plan(ray_con, "WITH " + ", ".join(definitions) + " SELECT * FROM c8")
    graph = plan.collect_query_resource_graph_metadata(conn=ray_con)
    tree = plan.repr_ascii(True)
    assert "reference to node" in tree
    assert len(tree.splitlines()) < 4 * len(graph["nodes"])
    monkeypatch.setenv("VANE_RAY_MATERIALIZED_CTE", "0")
    with pytest.raises(vane.InvalidInputException, match="disabled by VANE_RAY_MATERIALIZED_CTE"):
        _physical_plan(ray_con, "WITH c AS MATERIALIZED (SELECT 1) SELECT * FROM c").repr_ascii(True)


def test_ray_cte_producer_failure_releases_query(ray_con):
    def fail(i):
        raise ValueError("cte-producer-failed")

    vane.attach_function(fail, alias="cte_failure", parameters=["BIGINT"], return_dtype="BIGINT", connection=ray_con)
    try:
        with pytest.raises(Exception, match="cte-producer-failed"):
            ray_con.sql(
                "WITH c AS MATERIALIZED (SELECT cte_failure(i) AS x FROM range(3) t(i)) SELECT x FROM c"
            ).fetchall()
        assert ray_con.sql("SELECT 42").fetchall() == [(42,)]
    finally:
        vane.detach_function("cte_failure", connection=ray_con)
