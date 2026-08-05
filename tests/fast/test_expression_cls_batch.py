# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest


def test_vane_cls_batch_immediate_call_with_constructor_args():
    import pyarrow as pa
    import pyarrow.compute as pc

    import vane

    @vane.cls.batch(actor_number=1, return_dtype=pa.int64())
    class AddOffset:
        def __init__(self, offset):
            self.offset = offset

        def __call__(self, values):
            assert isinstance(values, (pa.Array, pa.ChunkedArray))
            return pc.add(values, self.offset)

    result = AddOffset(3)(pa.chunked_array([[1, 2]]))

    assert isinstance(result, pa.Array)
    assert result.to_pylist() == [4, 5]


def test_vane_cls_batch_expression_local():
    import pyarrow as pa
    import pyarrow.compute as pc

    import vane

    @vane.cls.batch(actor_number=1, return_dtype=pa.int32(), batch_size=2)
    class AddOffset:
        def __init__(self, offset):
            self.offset = offset

        def __call__(self, values):
            return pc.add(values, self.offset)

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(4) t(i)")
    expression = AddOffset(10)(vane.col("x"))

    assert rel.select(vane.col("x"), expression.alias("score")).fetchall() == [
        (0, 10),
        (1, 11),
        (2, 12),
        (3, 13),
    ]


def test_vane_cls_batch_expression_supports_keyword_columns():
    import pyarrow as pa
    import pyarrow.compute as pc

    import vane

    @vane.cls.batch(actor_number=1, return_dtype=pa.int32())
    class Subtract:
        def __call__(self, left, *, right):
            return pc.subtract(left, right)

    con = vane.connect()
    rel = con.sql("select 7::INTEGER as x, 2::INTEGER as y")

    assert rel.select(Subtract()(vane.col("x"), right=vane.col("y")).alias("result")).fetchall() == [(5,)]


def test_vane_cls_batch_expression_local_reuses_state_across_batches():
    import pyarrow as pa

    import vane

    @vane.cls.batch(actor_number=1, return_dtype=pa.int32(), batch_size=2)
    class BatchCounter:
        def __init__(self):
            self.calls = 0

        def __call__(self, values):
            self.calls += 1
            return pa.array([self.calls] * len(values), type=pa.int32())

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(5) t(i)")
    expression = BatchCounter()(vane.col("x"))

    assert rel.select(expression.alias("batch_call")).fetchall() == [(1,), (1,), (2,), (2,), (3,)]


def test_vane_cls_batch_instances_do_not_share_state():
    import pyarrow as pa

    import vane

    @vane.cls.batch(actor_number=1, return_dtype=pa.int64())
    class BatchCounter:
        def __init__(self):
            self.calls = 0

        def __call__(self, values):
            self.calls += 1
            return pa.array([self.calls] * len(values), type=pa.int64())

    first = BatchCounter()
    second = BatchCounter()

    assert first(pa.array([1])).to_pylist() == [1]
    assert first(pa.array([1])).to_pylist() == [2]
    assert second(pa.array([1])).to_pylist() == [1]


@pytest.mark.parametrize("actor_number", [None, 0, 2, True, 1.0, "1"])
def test_vane_cls_batch_requires_exactly_one_strict_integer_actor(actor_number):
    import pyarrow as pa

    import vane

    with pytest.raises(vane.InvalidInputException, match="actor_number must be exactly 1"):

        @vane.cls.batch(actor_number=actor_number, return_dtype=pa.int64())
        class Identity:
            def __call__(self, values):
                return values


def test_vane_cls_batch_requires_return_dtype_and_rejects_old_options():
    import vane

    with pytest.raises(TypeError, match="return_dtype"):
        vane.cls.batch(actor_number=1)

    with pytest.raises(TypeError, match="unexpected keyword argument 'schema'"):
        vane.cls.batch(actor_number=1, schema={"x": "INTEGER"})

    with pytest.raises(TypeError, match="unexpected keyword argument 'row_preserving'"):
        vane.cls.batch(actor_number=1, return_dtype="INTEGER", row_preserving=True)


def test_vane_cls_batch_rejects_async_call():
    import pyarrow as pa

    import vane

    with pytest.raises(TypeError, match="requires a synchronous Python __call__"):

        @vane.cls.batch(actor_number=1, return_dtype=pa.int32())
        class AsyncBatch:
            async def __call__(self, values):
                return values


def test_vane_cls_batch_rejects_variadic_call_protocol():
    import vane

    with pytest.raises(vane.InvalidInputException, match=r"cannot use \*args or \*\*kwargs"):

        @vane.cls.batch(actor_number=1, return_dtype="INTEGER")
        class Variadic:
            def __call__(self, *values):
                return values[0]


def test_vane_cls_batch_rejects_table_output():
    import pyarrow as pa

    import vane

    @vane.cls.batch(actor_number=1, return_dtype=pa.int64())
    class Identity:
        def __call__(self, values):
            return pa.table({"value": values})

    with pytest.raises(vane.InvalidInputException, match="must return pyarrow.Array or pyarrow.ChunkedArray"):
        Identity()(pa.array([1]))


def test_vane_cls_batch_struct_unnest_executes_one_actor_udf():
    import pyarrow as pa

    import vane

    result_type = pa.struct([pa.field("score", pa.int32()), pa.field("reason", pa.string())])

    @vane.cls.batch(actor_number=1, return_dtype=result_type, unnest=True)
    class Analyze:
        def __call__(self, values):
            python_values = values.to_pylist()
            return pa.StructArray.from_arrays(
                [
                    pa.array(python_values, type=pa.int32()),
                    pa.array([f"value={value}" for value in python_values]),
                ],
                fields=list(result_type),
            )

    con = vane.connect()
    rel = con.sql("select i::INTEGER as id, i::INTEGER as value from range(2) t(i)")
    selected = rel.select(vane.col("id"), Analyze()(vane.col("value")))

    assert selected.explain().count("STREAMING_UDF") == 1
    assert selected.fetchall() == [(0, 0, "value=0"), (1, 1, "value=1")]


def test_vane_cls_batch_physical_payload_marks_stateful_side_effects(monkeypatch):
    import uuid

    import pyarrow as pa

    import duckdb
    import vane

    monkeypatch.setenv("VANE_RUNNER", "local-fast")

    @vane.cls.batch(actor_number=1, return_dtype=pa.int32(), batch_size=2, gpus=0)
    class Identity:
        def __call__(self, values):
            return values

    con = vane.connect()
    try:
        relation = con.sql("select i::INTEGER as x from range(3) t(i)").select(Identity()(vane.col("x")))
        plan = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, str(uuid.uuid4())).to_physical_plan(con)
        nodes = plan.collect_udf_nodes(conn=con)
    finally:
        con.close()

    assert len(nodes) == 1
    payload = nodes[0]["payload"]
    assert payload["execution_backend"] == "subprocess_actor"
    assert payload["actor_number"] == 1
    assert payload["stateful"] is True
    assert payload["side_effects"] is True
    assert payload["row_preserving"] is True
    assert payload["call_mode"] == "map_batches_rows"
    assert payload["expression_id"]


def test_vane_cls_batch_return_dtype_pyarrow_int64_expression_round_trip():
    import pyarrow as pa

    import vane

    @vane.cls.batch(actor_number=1, return_dtype=pa.int64())
    class Identity:
        def __call__(self, values):
            return values.cast(pa.int64())

    con = vane.connect()
    rel = con.sql("select 42::BIGINT as x")

    assert rel.select(Identity()(vane.col("x")).alias("result")).fetchall() == [(42,)]


def test_vane_cls_batch_explicit_none_gpus_means_no_gpu():
    import pyarrow as pa

    import vane

    @vane.cls.batch(actor_number=1, return_dtype=pa.int32(), gpus=None)
    class Identity:
        def __call__(self, values):
            return values

    con = vane.connect()
    rel = con.sql("select 42::INTEGER as x")

    assert rel.select(Identity()(vane.col("x"))).fetchall() == [(42,)]
