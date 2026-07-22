# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Expression-level actor UDF tests for ``vane.cls.batch``."""

from __future__ import annotations

import pytest


def test_vane_cls_batch_immediate_call_with_constructor_args():
    import pyarrow as pa

    import vane

    @vane.cls.batch(actor_number=1, schema={"score": "INTEGER"})
    class Scorer:
        def __init__(self, offset):
            self.offset = offset

        def __call__(self, table: pa.Table) -> pa.Table:
            values = table.column("value").to_pylist()
            return pa.table({"score": [value + self.offset for value in values]})

    result = Scorer(10)(value=[1, 2, 3])

    assert result.to_pydict() == {"score": [11, 12, 13]}


def test_vane_cls_batch_expression_local():
    import pyarrow as pa

    import vane

    @vane.cls.batch(actor_number=1, schema={"score": "INTEGER"}, row_preserving=True)
    class Scorer:
        def __init__(self, offset):
            self.offset = offset

        def __call__(self, table: pa.Table) -> pa.Table:
            values = table.column("value").to_pylist()
            return pa.table({"score": [value + self.offset for value in values]})

    con = vane.connect()
    rel = con.sql("select i::INTEGER as value from range(3) t(i)")
    expr = Scorer(10)(value=vane.col("value"))

    assert rel.select(vane.col("value"), expr.alias("score")).fetchall() == [(0, 10), (1, 11), (2, 12)]


def test_vane_cls_batch_expression_can_change_cardinality_when_not_row_preserving():
    import pyarrow as pa

    import vane

    @vane.cls.batch(actor_number=1, schema={"even_value": "INTEGER"}, row_preserving=False)
    class KeepEvenValues:
        def __call__(self, table: pa.Table) -> pa.Table:
            values = table.column("value").to_pylist()
            return pa.table({"even_value": [value for value in values if value % 2 == 0]})

    con = vane.connect()
    try:
        relation = con.sql("SELECT i::INTEGER AS value FROM range(6) t(i)")
        expression = KeepEvenValues()(value=vane.col("value"))
        rows = relation.select(expression.alias("even_value")).fetchall()
    finally:
        con.close()

    assert rows == [(0,), (2,), (4,)]
    assert len(rows) != 6


def test_vane_cls_batch_expression_local_reuses_state_across_batches():
    from collections import Counter

    import pyarrow as pa

    import vane

    @vane.cls.batch(
        actor_number=1,
        schema={"batch_call": "INTEGER"},
        row_preserving=True,
    )
    class BatchCounter:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, table: pa.Table) -> pa.Table:
            self.calls += 1
            return pa.table({"batch_call": [self.calls] * table.num_rows})

    con = vane.connect()
    # DuckDB's 2048-row standard vector size makes this three actor submits.
    rel = con.sql("select i::INTEGER as value from range(4097) t(i)")
    expr = BatchCounter()(value=vane.col("value"))
    out = rel.select(vane.col("value"), expr.alias("batch_call")).order("value")

    rows = out.fetchall()
    assert len(rows) == 4097
    calls = Counter(row[1] for row in rows)
    assert sorted(calls) == [1, 2, 3]
    assert sorted(calls.values()) == [1, 2048, 2048]


def test_vane_cls_batch_instances_do_not_share_state():
    import pyarrow as pa

    import vane

    @vane.cls.batch(actor_number=1, schema={"batch_call": "INTEGER"})
    class BatchCounter:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, table: pa.Table) -> pa.Table:
            self.calls += 1
            return pa.table({"batch_call": [self.calls] * table.num_rows})

    first = BatchCounter()
    second = BatchCounter()

    assert first(value=[10]).to_pydict() == {"batch_call": [1]}
    assert first(value=[20]).to_pydict() == {"batch_call": [2]}
    assert second(value=[30]).to_pydict() == {"batch_call": [1]}


@pytest.mark.parametrize("actor_number", [None, False, True, 0, 1.0, 2])
def test_vane_cls_batch_requires_exactly_one_strict_integer_actor(actor_number):
    import vane

    class Identity:
        def __call__(self, table):
            return table

    with pytest.raises(
        vane.InvalidInputException,
        match=r"actor_number must be exactly 1.*multi-actor state",
    ):
        vane.cls.batch(actor_number=actor_number, schema={"x": "INTEGER"})(Identity)


def test_vane_cls_batch_requires_single_output_schema():
    import vane

    class Identity:
        def __call__(self, table):
            return table

    with pytest.raises(vane.InvalidInputException, match="exactly one output column"):
        vane.cls.batch(actor_number=1, schema={"x": "INTEGER", "y": "INTEGER"})(Identity)


def test_vane_cls_batch_physical_payload_marks_stateful_side_effects(monkeypatch):
    import uuid

    import pyarrow as pa

    import vane

    monkeypatch.setenv("VANE_RUNNER", "ray")

    @vane.cls.batch(
        actor_number=1,
        schema={"score": "INTEGER"},
        name="stateful_batch_counter",
        row_preserving=True,
    )
    class BatchCounter:
        def __call__(self, table: pa.Table) -> pa.Table:
            return pa.table({"score": table.column("value")})

    con = vane.connect()
    try:
        relation = con.sql("select 1::INTEGER as value").select(BatchCounter()(value=vane.col("value")).alias("score"))
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
            relation,
            str(uuid.uuid4()),
        ).to_physical_plan(con)
        nodes = plan.collect_udf_nodes(conn=con)
    finally:
        con.close()

    assert len(nodes) == 1
    payload = nodes[0]["payload"]
    assert payload["udf_name"] == "stateful_batch_counter"
    assert payload.get("stateful") is True
    assert payload.get("side_effects") is True
    assert payload["actor_number"] == 1
    assert "state_scope" not in payload


def test_vane_cls_batch_lazy_payload_arguments_are_forwarded(monkeypatch):
    import pyarrow as pa

    import vane
    import vane._expression_udf as expression_udf

    monkeypatch.setenv("VANE_RUNNER", "ray")
    captured = {}
    sentinel = object()

    def fake_build_actor_map_batches_expression(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(expression_udf, "_build_actor_map_batches_expression", fake_build_actor_map_batches_expression)

    @vane.cls.batch(
        actor_number=1,
        schema={"score": "INTEGER"},
        name="score_batch",
        batch_size=128,
        row_preserving=True,
        gpus=0.25,
    )
    class Scorer:
        def __call__(self, table: pa.Table) -> pa.Table:
            return pa.table({"score": table.column("value")})

    assert Scorer()(value=vane.col("value")) is sentinel
    assert captured["kwargs"]["name"] == "score_batch"
    assert captured["kwargs"]["schema"] == {"score": vane.sqltype("INTEGER")}
    assert captured["kwargs"]["batch_size"] == 128
    assert captured["kwargs"]["row_preserving"] is True
    assert captured["kwargs"]["actor_number"] == 1
    assert captured["kwargs"]["gpus"] == 0.25
    assert list(captured["kwargs"]["inputs"]) == ["value"]


@pytest.mark.parametrize("actor_number", [False, True])
def test_vane_cls_batch_rejects_actor_number_bool(actor_number):
    import vane

    class Identity:
        def __call__(self, table):
            return table

    with pytest.raises(
        vane.InvalidInputException,
        match=r"actor_number.*bool|actor_number.*exactly 1|actor_number.*integer",
    ):
        vane.cls.batch(actor_number=actor_number, schema={"result": "INTEGER"})(Identity)


def test_vane_cls_batch_schema_pyarrow_int64_expression_round_trip():
    import pyarrow as pa

    import vane

    @vane.cls.batch(actor_number=1, schema={"result": pa.int64()}, row_preserving=True)
    class WidenBatch:
        def __call__(self, table):
            values = table.column("value").to_pylist()
            return pa.table({"result": [value + 2**40 for value in values]})

    conn = vane.connect()
    result = conn.sql("SELECT 1::INTEGER AS value").select(WidenBatch()(value=vane.col("value")).alias("result"))

    assert [str(dtype) for dtype in result.types] == ["BIGINT"]
    assert result.fetchall() == [(2**40 + 1,)]
