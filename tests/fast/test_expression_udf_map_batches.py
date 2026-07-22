# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Expression-level batch UDF tests for ``vane.func.batch``."""

from __future__ import annotations

import os

import pytest


def test_vane_function_batch_expression_local_single_output():
    import pyarrow as pa

    import vane

    def add_one_batch(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(5) t(i)")

    expr = vane.func.batch(
        add_one_batch,
        inputs={"x": vane.col("x")},
        schema={"y": "INTEGER"},
        batch_size=2,
    )
    out = rel.select(expr.alias("y"))

    assert sorted(out.fetchall()) == [(1,), (2,), (3,), (4,), (5,)]


def test_vane_function_batch_immediate_call_without_expression_inputs():
    import pyarrow as pa

    import vane

    def add_one_batch(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    result = vane.func.batch(
        add_one_batch,
        inputs={"x": [1, 2, 3]},
        batch_size=2,
    )

    assert result.to_pydict() == {"y": [2, 3, 4]}


def test_vane_function_batch_requires_mapping_inputs():
    import vane

    def identity(table):
        return table

    with pytest.raises(vane.InvalidInputException, match="inputs must be a non-empty mapping"):
        vane.func.batch(identity, inputs=[], schema={"x": "INTEGER"})


def test_vane_function_batch_v1_requires_single_output_schema():
    import vane

    def identity(table):
        return table

    with pytest.raises(vane.InvalidInputException, match="exactly one output column"):
        vane.func.batch(identity, inputs={"x": vane.col("x")}, schema={"x": "INTEGER", "y": "INTEGER"})


def test_vane_function_batch_expression_requires_schema():
    import vane

    def identity(table):
        return table

    with pytest.raises(vane.InvalidInputException, match="schema must be a non-empty mapping"):
        vane.func.batch(identity, inputs={"x": vane.col("x")})


def test_vane_function_batch_v1_rejects_passthrough_projection_without_row_preserving():
    import pyarrow as pa

    import vane

    def add_one_batch(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    con = vane.connect()
    rel = con.sql("select 1::INTEGER as x")
    expr = vane.func.batch(add_one_batch, inputs={"x": vane.col("x")}, schema={"y": "INTEGER"})

    with pytest.raises(Exception, match=r"row_preserving=True|only output|unique output"):
        rel.select(vane.col("x"), expr.alias("y")).fetchall()


def test_vane_function_batch_v1_rejects_multiple_result_only_batch_udfs():
    import pyarrow as pa

    import vane

    def add_one_batch(table):
        values = table.column("x").to_pylist()
        return pa.table({"a": [value + 1 for value in values]})

    def times_two_batch(table):
        values = table.column("x").to_pylist()
        return pa.table({"b": [value * 2 for value in values]})

    con = vane.connect()
    rel = con.sql("select 1::INTEGER as x")
    a_expr = vane.func.batch(add_one_batch, inputs={"x": vane.col("x")}, schema={"a": "INTEGER"})
    b_expr = vane.func.batch(times_two_batch, inputs={"x": vane.col("x")}, schema={"b": "INTEGER"})

    with pytest.raises(Exception, match=r"row_preserving=True|one batch UDF|only output"):
        rel.select(a_expr.alias("a"), b_expr.alias("b")).fetchall()


def test_vane_function_batch_v1_rejects_nested_result_only_batch_udfs():
    import pyarrow as pa

    import vane

    def add_one_batch(table):
        values = table.column("x").to_pylist()
        return pa.table({"a": [value + 1 for value in values]})

    def times_two_batch(table):
        values = table.column("a").to_pylist()
        return pa.table({"b": [value * 2 for value in values]})

    con = vane.connect()
    rel = con.sql("select 1::INTEGER as x")
    a_expr = vane.func.batch(add_one_batch, inputs={"x": vane.col("x")}, schema={"a": "INTEGER"})
    b_expr = vane.func.batch(times_two_batch, inputs={"a": a_expr}, schema={"b": "INTEGER"})

    with pytest.raises(Exception, match=r"row_preserving=True|nested batch UDF|top-level"):
        rel.select(b_expr.alias("b")).fetchall()


def test_vane_function_batch_expression_ray_backend_explain():
    import pyarrow as pa

    import vane

    old_runner = os.environ.get("VANE_RUNNER")
    try:
        vane.configure(runner="ray")

        def add_one_batch(table):
            values = table.column("x").to_pylist()
            return pa.table({"y": [value + 1 for value in values]})

        con = vane.connect()
        rel = con.sql("select i::INTEGER as x from range(3) t(i)")
        expr = vane.func.batch(add_one_batch, inputs={"x": vane.col("x")}, schema={"y": "INTEGER"})
        plan = rel.select(expr.alias("y")).explain()

        assert "execution_backend:" in plan
        assert "ray_task" in plan
        assert "ray_block_stream_output:" in plan
        assert "direct_block_metadata_pair" in plan
    finally:
        if old_runner is None:
            os.environ.pop("VANE_RUNNER", None)
        else:
            os.environ["VANE_RUNNER"] = old_runner


def test_vane_function_batch_uses_local_fast_runner_from_env(monkeypatch):
    import pyarrow as pa

    import vane

    monkeypatch.setenv("VANE_RUNNER", "local-fast")

    def add_one_batch(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    expr = vane.func.batch(add_one_batch, inputs={"x": vane.col("x")}, schema={"y": "INTEGER"})
    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(3) t(i)")
    plan = rel.select(expr.alias("y")).explain()

    assert "execution_backend:" in plan
    assert "subprocess_task" in plan
    assert "local_shm_ref_bundle" in plan
    assert "ray_task" not in plan
    assert "direct_block_metadata_pair" not in plan


@pytest.mark.parametrize("row_preserving", [False, True])
def test_vane_function_batch_local_fast_runner_executes(monkeypatch, row_preserving):
    import pyarrow as pa

    import vane

    monkeypatch.setenv("VANE_RUNNER", "local-fast")

    def add_one_batch(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    expr = vane.func.batch(
        add_one_batch,
        inputs={"x": vane.col("x")},
        schema={"y": "INTEGER"},
        row_preserving=row_preserving,
    )
    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(3) t(i)")

    if row_preserving:
        result = rel.select(vane.col("x"), expr.alias("y")).fetchall()
        assert result == [(0, 1), (1, 2), (2, 3)]
    else:
        result = rel.select(expr.alias("y")).fetchall()
        assert result == [(1,), (2,), (3,)]


def test_vane_function_batch_local_fast_runner_rewrites_streaming_contract(monkeypatch):
    import uuid

    import pyarrow as pa

    import vane

    monkeypatch.setenv("VANE_RUNNER", "local-fast")

    def add_one_batch(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    expr = vane.func.batch(
        add_one_batch,
        inputs={"x": vane.col("x")},
        schema={"y": "INTEGER"},
        row_preserving=True,
    )
    con = vane.connect()
    try:
        relation = con.sql("select i::INTEGER as x from range(3) t(i)").select(expr.alias("y"))
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
            relation,
            str(uuid.uuid4()),
        ).to_physical_plan(con)
        nodes = plan.collect_udf_nodes(conn=con)
    finally:
        con.close()

    assert len(nodes) == 1
    payload = nodes[0]["payload"]
    assert payload["execution_backend"] == "subprocess_task"
    assert payload["produce_ray_block_stream"] is False
    assert payload["produce_ref_bundle_output"] is True
    assert payload["streaming_output_mode"] == "local_shm_ref_bundle"


def test_vane_internal_batch_actor_expression_uses_actor_backend(monkeypatch):
    import pyarrow as pa

    import vane
    from vane._expression_udf import _build_actor_map_batches_expression

    monkeypatch.setenv("VANE_RUNNER", "local-fast")

    class AddOneActor:
        def __call__(self, table) -> pa.Table:
            values = table.column("x").to_pylist()
            return pa.table({"y": [value + 1 for value in values]})

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(3) t(i)")
    expr = _build_actor_map_batches_expression(
        AddOneActor,
        name="add_one_actor",
        inputs={"x": vane.col("x")},
        schema={"y": "INTEGER"},
        batch_size=2,
        row_preserving=True,
        actor_number=2,
        gpus=0,
    )

    plan = rel.select(vane.col("x"), expr.alias("y")).explain()

    assert "execution_backend:" in plan
    assert "subprocess_actor" in plan
    assert "actor_number:" in plan
    assert "2" in plan


def test_vane_function_batch_row_preserving_keeps_passthrough_columns():
    import pyarrow as pa

    import vane

    def add_one_batch(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(4) t(i)")
    expr = vane.func.batch(
        add_one_batch,
        inputs={"x": vane.col("x")},
        schema={"y": "INTEGER"},
        row_preserving=True,
    )

    assert rel.select(vane.col("x"), expr.alias("y")).fetchall() == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_vane_function_batch_row_preserving_allows_multiple_udfs():
    import pyarrow as pa

    import vane

    def add_one_batch(table):
        values = table.column("x").to_pylist()
        return pa.table({"a": [value + 1 for value in values]})

    def times_two_batch(table):
        values = table.column("x").to_pylist()
        return pa.table({"b": [value * 2 for value in values]})

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(4) t(i)")
    a_expr = vane.func.batch(
        add_one_batch,
        inputs={"x": vane.col("x")},
        schema={"a": "INTEGER"},
        row_preserving=True,
    )
    b_expr = vane.func.batch(
        times_two_batch,
        inputs={"x": vane.col("x")},
        schema={"b": "INTEGER"},
        row_preserving=True,
    )

    assert rel.select(vane.col("x"), a_expr.alias("a"), b_expr.alias("b")).fetchall() == [
        (0, 1, 0),
        (1, 2, 2),
        (2, 3, 4),
        (3, 4, 6),
    ]


def test_vane_function_batch_row_preserving_allows_nested_udfs():
    import pyarrow as pa

    import vane

    def add_one_batch(table):
        values = table.column("x").to_pylist()
        return pa.table({"a": [value + 1 for value in values]})

    def times_two_batch(table):
        values = table.column("a").to_pylist()
        return pa.table({"b": [value * 2 for value in values]})

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(4) t(i)")
    a_expr = vane.func.batch(
        add_one_batch,
        inputs={"x": vane.col("x")},
        schema={"a": "INTEGER"},
        row_preserving=True,
    )
    b_expr = vane.func.batch(
        times_two_batch,
        inputs={"a": a_expr},
        schema={"b": "INTEGER"},
        row_preserving=True,
    )

    assert rel.select(vane.col("x"), b_expr.alias("b")).fetchall() == [(0, 2), (1, 4), (2, 6), (3, 8)]


def test_vane_function_batch_row_preserving_rejects_row_count_mismatch():
    import pyarrow as pa

    import vane

    def too_short_batch(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values[:-1]]})

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(4) t(i)")
    expr = vane.func.batch(
        too_short_batch,
        inputs={"x": vane.col("x")},
        schema={"y": "INTEGER"},
        row_preserving=True,
    )

    with pytest.raises(Exception, match=r"row count|input rows|does not match"):
        rel.select(vane.col("x"), expr.alias("y")).fetchall()


def test_vane_function_batch_row_preserving_explain_uses_streaming_layout():
    import pyarrow as pa

    import vane

    def add_one_batch(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(2) t(i)")
    expr = vane.func.batch(
        add_one_batch,
        inputs={"x": vane.col("x")},
        schema={"y": "INTEGER"},
        row_preserving=True,
    )
    plan = rel.select(vane.col("x"), expr.alias("y")).explain()

    assert "call_mode:" in plan
    assert "map_batches_rows" in plan
    assert "row_preserving:" in plan
    assert "true" in plan
    assert "STREAMING_UDF" in plan
    assert "local_shm_ref_bundle" in plan


def test_vane_function_batch_row_preserving_ray_block_stream_explain_when_gpu_requested():
    import pyarrow as pa

    import vane

    old_runner = os.environ.get("VANE_RUNNER")
    try:
        vane.configure(runner="ray")

        def add_one_batch(table):
            values = table.column("x").to_pylist()
            return pa.table({"y": [value + 1 for value in values]})

        con = vane.connect()
        rel = con.sql("select i::INTEGER as x, ('row-' || i::VARCHAR) as label from range(2) t(i)")
        expr = vane.func.batch(
            add_one_batch,
            inputs={"x": vane.col("x")},
            schema={"y": "INTEGER"},
            row_preserving=True,
            gpus=1.0,
        )
        plan = rel.select(vane.col("label"), expr.alias("y")).explain()

        assert "STREAMING_UDF" in plan
        assert "execution_backend:" in plan
        assert "ray_task" in plan
        assert "call_mode:" in plan
        assert "map_batches_rows" in plan
        assert "row_preserving:" in plan
        assert "true" in plan
        assert "ray_block_stream_output:" in plan
        assert "direct_block_metadata_pair" in plan
    finally:
        if old_runner is None:
            os.environ.pop("VANE_RUNNER", None)
        else:
            os.environ["VANE_RUNNER"] = old_runner


def test_vane_function_batch_row_preserving_gpu_zero_stays_streaming():
    import pyarrow as pa

    import vane

    old_runner = os.environ.get("VANE_RUNNER")
    try:
        vane.configure(runner="ray")

        def add_one_batch(table):
            values = table.column("x").to_pylist()
            return pa.table({"y": [value + 1 for value in values]})

        con = vane.connect()
        rel = con.sql("select i::INTEGER as x from range(2) t(i)")
        expr = vane.func.batch(
            add_one_batch,
            inputs={"x": vane.col("x")},
            schema={"y": "INTEGER"},
            row_preserving=True,
            gpus=0.0,
        )
        plan = rel.select(vane.col("x"), expr.alias("y")).explain()

        assert "STREAMING_UDF" in plan
        assert "ray_block_stream_output:" in plan
    finally:
        if old_runner is None:
            os.environ.pop("VANE_RUNNER", None)
        else:
            os.environ["VANE_RUNNER"] = old_runner


def test_vane_function_batch_streaming_output_is_sliced_before_grouped_aggregate():
    """Keep an oversized streaming batch from reaching fixed-size DuckDB vectors."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import pyarrow as pa
        import vane

        def record_batch_size(table):
            return pa.table({"seen": [table.num_rows] * table.num_rows})

        vane.configure(runner="local-fast")
        con = vane.connect()
        relation = con.sql("SELECT i::INTEGER AS x FROM range(5000) t(i)")
        expression = vane.func.batch(
            record_batch_size,
            inputs={"x": vane.col("x")},
            schema={"seen": "BIGINT"},
            batch_size=4096,
        )
        result = (
            relation.select(expression.alias("seen"))
            .aggregate("seen, count(*) AS n")
            .order("seen")
            .fetchall()
        )
        assert result == [(904, 904), (4096, 4096)], result
        con.close()
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_vane_function_batch_row_preserving_batch_size_is_backend_independent():
    import pyarrow as pa

    import vane

    def record_batch_size(table):
        return pa.table({"seen": [table.num_rows] * table.num_rows})

    vane.configure(runner="local-fast")
    con = vane.connect()
    relation = con.sql("SELECT i::INTEGER AS x FROM range(5000) t(i)")
    expression = vane.func.batch(
        record_batch_size,
        inputs={"x": vane.col("x")},
        schema={"seen": "BIGINT"},
        batch_size=4096,
        row_preserving=True,
    )

    result = relation.select(expression.alias("seen")).aggregate("seen, count(*) AS n").order("seen").fetchall()
    assert result == [(904, 904), (4096, 4096)]


def test_vane_function_batch_row_preserving_chain_promotes_ref_bundle_without_gpu():
    import pyarrow as pa

    import vane

    old_runner = os.environ.get("VANE_RUNNER")
    try:
        vane.configure(runner="ray")

        def add_one_batch(table):
            values = table.column("x").to_pylist()
            return pa.table({"a": [value + 1 for value in values]})

        def times_two_batch(table):
            values = table.column("a").to_pylist()
            return pa.table({"b": [value * 2 for value in values]})

        con = vane.connect()
        rel = con.sql("select i::INTEGER as x from range(2) t(i)")
        a_expr = vane.func.batch(
            add_one_batch,
            inputs={"x": vane.col("x")},
            schema={"a": "INTEGER"},
            row_preserving=True,
        )
        b_expr = vane.func.batch(
            times_two_batch,
            inputs={"a": a_expr},
            schema={"b": "INTEGER"},
            row_preserving=True,
        )
        plan = rel.select(vane.col("x"), b_expr.alias("b")).explain()

        assert plan.count("STREAMING_UDF") == 2
        assert plan.count("ray_block_stream_output:") == 2
        assert "direct_block_metadata_pair" in plan
        assert plan.count("map_batches_rows") == 2
        assert plan.count("row_preserving: true") == 2
    finally:
        if old_runner is None:
            os.environ.pop("VANE_RUNNER", None)
        else:
            os.environ["VANE_RUNNER"] = old_runner


def test_vane_function_batch_row_preserving_chain_promotion_uses_ray_runner(monkeypatch):
    import pyarrow as pa

    import vane

    monkeypatch.setenv("VANE_RUNNER", "ray")

    def add_one_batch(table):
        values = table.column("x").to_pylist()
        return pa.table({"a": [value + 1 for value in values]})

    def times_two_batch(table):
        values = table.column("a").to_pylist()
        return pa.table({"b": [value * 2 for value in values]})

    a_expr = vane.func.batch(
        add_one_batch,
        inputs={"x": vane.col("x")},
        schema={"a": "INTEGER"},
        row_preserving=True,
    )
    b_expr = vane.func.batch(
        times_two_batch,
        inputs={"a": a_expr},
        schema={"b": "INTEGER"},
        row_preserving=True,
    )
    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(2) t(i)")
    plan = rel.select(vane.col("x"), b_expr.alias("b")).explain()

    assert plan.count("STREAMING_UDF") == 2
    assert plan.count("ray_block_stream_output:") == 2
    assert "ray_task" in plan
    assert "direct_block_metadata_pair" in plan
