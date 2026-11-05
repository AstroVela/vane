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
    ]
    for sql, expected in cases:
        assert ray_con.sql(sql).fetchall() == expected, sql


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
