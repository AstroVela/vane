# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Expression-level Arrow column batch UDF tests for ``vane.func.batch``."""

from __future__ import annotations

import os

import pytest


def test_vane_function_batch_is_decorator_only():
    import pyarrow as pa

    import vane

    def identity(values):
        return values

    with pytest.raises(TypeError, match="positional argument"):
        vane.func.batch(identity, return_dtype=pa.int64())

    with pytest.raises(TypeError, match="unexpected keyword argument 'inputs'"):
        vane.func.batch(return_dtype=pa.int64(), inputs={"x": vane.col("x")})

    with pytest.raises(TypeError, match="unexpected keyword argument 'schema'"):
        vane.func.batch(schema={"x": "BIGINT"})

    with pytest.raises(TypeError, match="unexpected keyword argument 'row_preserving'"):
        vane.func.batch(return_dtype=pa.int64(), row_preserving=False)


def test_vane_function_batch_requires_return_dtype():
    import vane

    with pytest.raises(TypeError, match="return_dtype"):
        vane.func.batch()


def test_vane_function_batch_rejects_async_function():
    import pyarrow as pa

    import vane

    with pytest.raises(TypeError, match="generic UDF callables must be synchronous"):

        @vane.func.batch(return_dtype=pa.int32())
        async def identity(values):
            return values


def test_vane_function_batch_eager_array_and_chunked_array_inputs():
    import pyarrow as pa
    import pyarrow.compute as pc

    import vane

    seen_types = []

    @vane.func.batch(return_dtype=pa.int64())
    def add(left, right):
        seen_types.append((type(left), type(right)))
        return pc.add(left, right)

    single_chunk = add(pa.array([1, 2]), pa.chunked_array([[10, 20]]))
    multiple_chunks = add(pa.chunked_array([[1], [2]]), pa.chunked_array([[10], [20]]))

    assert single_chunk.to_pylist() == [11, 22]
    assert multiple_chunks.to_pylist() == [11, 22]
    assert seen_types[0] == (pa.Int64Array, pa.Int64Array)
    assert seen_types[1] == (pa.ChunkedArray, pa.ChunkedArray)


def test_vane_function_batch_eager_call_requires_an_input_column():
    import pyarrow as pa

    import vane

    @vane.func.batch(return_dtype=pa.int32())
    def no_inputs():
        return pa.array([], type=pa.int32())

    with pytest.raises(vane.InvalidInputException, match="batch UDFs require at least one input column"):
        no_inputs()


def test_vane_function_batch_rejects_non_arrow_column_inputs():
    import pyarrow as pa

    import vane

    @vane.func.batch(return_dtype=pa.int64())
    def identity(values):
        return values

    for value in ([1, 2], pa.table({"value": [1, 2]})):
        with pytest.raises(vane.InvalidInputException, match="must be pyarrow.Array or pyarrow.ChunkedArray"):
            identity(value)


def test_vane_function_batch_expression_receives_arrow_columns():
    import pyarrow as pa
    import pyarrow.compute as pc

    import vane

    @vane.func.batch(return_dtype=pa.int32(), batch_size=2)
    def add_one(values):
        assert isinstance(values, (pa.Array, pa.ChunkedArray))
        return pc.add(values, 1)

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(5) t(i)")

    assert rel.select(vane.col("x"), add_one(vane.col("x")).alias("y")).fetchall() == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    ]


def test_vane_function_batch_expression_supports_keyword_columns():
    import pyarrow as pa
    import pyarrow.compute as pc

    import vane

    @vane.func.batch(return_dtype=pa.int32())
    def subtract(left, *, right):
        return pc.subtract(left, right)

    con = vane.connect()
    rel = con.sql("select 7::INTEGER as x, 2::INTEGER as y")

    assert rel.select(subtract(vane.col("x"), right=vane.col("y")).alias("result")).fetchall() == [(5,)]


def test_vane_function_batch_rejects_table_output():
    import pyarrow as pa

    import vane

    @vane.func.batch(return_dtype=pa.int64())
    def identity(values):
        return pa.table({"value": values})

    with pytest.raises(vane.InvalidInputException, match="must return pyarrow.Array or pyarrow.ChunkedArray"):
        identity(pa.array([1, 2]))


def test_vane_function_batch_rejects_row_count_mismatch():
    import pyarrow as pa

    import vane

    @vane.func.batch(return_dtype=pa.int32())
    def too_short(values):
        return values.slice(0, max(0, len(values) - 1))

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(4) t(i)")

    with pytest.raises(Exception, match=r"returned 3 rows for 4 input rows|row count"):
        rel.select(too_short(vane.col("x")).alias("y")).fetchall()


def test_vane_function_batch_casts_output_to_declared_arrow_type():
    import pyarrow as pa

    import vane

    @vane.func.batch(return_dtype=pa.int32())
    def as_int32(values):
        return pa.array(values.to_pylist(), type=pa.int64())

    result = as_int32(pa.array([1, 2], type=pa.int64()))

    assert result.type == pa.int32()
    assert result.to_pylist() == [1, 2]


def test_vane_function_batch_struct_is_one_logical_output_column():
    import pyarrow as pa

    import vane

    result_type = pa.struct(
        [
            pa.field("label", pa.string()),
            pa.field("score", pa.float64()),
            pa.field("reason", pa.string()),
        ]
    )

    @vane.func.batch(return_dtype=result_type)
    def analyze(text):
        labels = pa.array(["positive" if value > 0 else "negative" for value in text.to_pylist()])
        scores = pa.array([abs(value) / 10 for value in text.to_pylist()], type=pa.float64())
        reasons = pa.array([f"value={value}" for value in text.to_pylist()])
        return pa.StructArray.from_arrays([labels, scores, reasons], fields=list(result_type))

    con = vane.connect()
    rel = con.sql("select i::INTEGER as id, (i * 2 - 1)::INTEGER as value from range(2) t(i)")
    rows = rel.select(vane.col("id"), analyze(vane.col("value")).alias("analysis")).fetchall()

    assert rows == [
        (0, {"label": "negative", "score": 0.1, "reason": "value=-1"}),
        (1, {"label": "positive", "score": 0.1, "reason": "value=1"}),
    ]


def test_vane_function_batch_unnest_expands_struct_once(tmp_path):
    import pyarrow as pa

    import vane

    calls_path = tmp_path / "calls"
    result_type = pa.struct(
        [
            pa.field("label", pa.string()),
            pa.field("score", pa.float64()),
            pa.field("reason", pa.string()),
        ]
    )

    @vane.func.batch(return_dtype=result_type, unnest=True)
    def analyze(text):
        with calls_path.open("a", encoding="utf-8") as calls:
            calls.write("batch\n")
        values = text.to_pylist()
        return pa.StructArray.from_arrays(
            [
                pa.array(["ok"] * len(values)),
                pa.array([float(value) for value in values]),
                pa.array([f"value={value}" for value in values]),
            ],
            fields=list(result_type),
        )

    con = vane.connect()
    rel = con.sql("select i::INTEGER as id, i::INTEGER as value from range(3) t(i)")
    selected = rel.select(vane.col("id"), analyze(vane.col("value")))

    assert selected.explain().count("STREAMING_UDF") == 1
    assert selected.fetchall() == [
        (0, "ok", 0.0, "value=0"),
        (1, "ok", 1.0, "value=1"),
        (2, "ok", 2.0, "value=2"),
    ]
    assert calls_path.read_text(encoding="utf-8").splitlines() == ["batch"]


def test_vane_function_batch_unnest_requires_struct_return_dtype():
    import pyarrow as pa

    import vane

    with pytest.raises(vane.InvalidInputException, match="unnest=True requires a Struct"):

        @vane.func.batch(return_dtype=pa.int64(), unnest=True)
        def identity(values):
            return values


def test_vane_function_batch_separate_calls_have_separate_expression_ids():
    import pyarrow as pa

    import vane

    result_type = pa.struct([pa.field("value", pa.int32())])

    @vane.func.batch(return_dtype=result_type, unnest=True)
    def wrap(values):
        return pa.StructArray.from_arrays([values], fields=list(result_type))

    con = vane.connect()
    rel = con.sql("select 1::INTEGER as left_value, 10::INTEGER as right_value")
    selected = rel.select(wrap(vane.col("left_value")), wrap(vane.col("right_value")))

    assert selected.explain().count("STREAMING_UDF") == 2
    assert selected.fetchall() == [(1, 10)]


def test_vane_function_batch_allows_multiple_and_nested_udfs():
    import pyarrow as pa
    import pyarrow.compute as pc

    import vane

    @vane.func.batch(return_dtype=pa.int32())
    def add_one(values):
        return pc.add(values, 1)

    @vane.func.batch(return_dtype=pa.int32())
    def times_two(values):
        return pc.multiply(values, 2)

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(3) t(i)")
    a = add_one(vane.col("x"))
    b = times_two(vane.col("x"))
    nested = times_two(a)

    assert rel.select(a.alias("a"), b.alias("b"), nested.alias("nested")).fetchall() == [
        (1, 0, 2),
        (2, 2, 4),
        (3, 4, 6),
    ]


def test_vane_function_batch_local_fast_runner_rewrites_streaming_contract(monkeypatch):
    import uuid

    import pyarrow as pa
    import pyarrow.compute as pc

    import duckdb
    import vane

    monkeypatch.setenv("VANE_RUNNER", "local-fast")

    @vane.func.batch(return_dtype=pa.int32())
    def add_one(values):
        return pc.add(values, 1)

    con = vane.connect()
    try:
        relation = con.sql("select i::INTEGER as x from range(3) t(i)").select(add_one(vane.col("x")))
        plan = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, str(uuid.uuid4())).to_physical_plan(con)
        nodes = plan.collect_udf_nodes(conn=con)
    finally:
        con.close()

    assert len(nodes) == 1
    payload = nodes[0]["payload"]
    assert payload["execution_backend"] == "subprocess_task"
    assert payload["produce_ray_block_stream"] is False
    assert payload["produce_ref_bundle_output"] is True
    assert payload["streaming_output_mode"] == "local_shm_ref_bundle"
    assert payload["call_mode"] == "map_batches_rows"
    assert payload["row_preserving"] is True
    assert payload["expression_id"]


def test_vane_function_batch_ray_backend_explain():
    import pyarrow as pa
    import pyarrow.compute as pc

    import vane

    old_runner = os.environ.get("VANE_RUNNER")
    try:
        vane.configure(runner="ray")

        @vane.func.batch(return_dtype=pa.int32())
        def add_one(values):
            return pc.add(values, 1)

        con = vane.connect()
        rel = con.sql("select i::INTEGER as x from range(3) t(i)")
        plan = rel.select(add_one(vane.col("x"))).explain()

        assert "execution_backend:" in plan
        assert "ray_task" in plan
        assert "ray_block_stream_output:" in plan
        assert "direct_block_metadata_pair" in plan
    finally:
        if old_runner is None:
            os.environ.pop("VANE_RUNNER", None)
        else:
            os.environ["VANE_RUNNER"] = old_runner


def test_vane_function_batch_batch_size_is_backend_independent():
    import pyarrow as pa

    import vane

    old_runner = os.environ.get("VANE_RUNNER")
    con = None
    try:
        vane.configure(runner="local")

        @vane.func.batch(return_dtype=pa.int64(), batch_size=4096)
        def record_batch_size(values):
            return pa.array([len(values)] * len(values), type=pa.int64())

        con = vane.connect()
        relation = con.sql("SELECT i::INTEGER AS x FROM range(5000) t(i)")
        result = (
            relation.select(record_batch_size(vane.col("x")).alias("seen"))
            .aggregate("seen, count(*) AS n")
            .order("seen")
            .fetchall()
        )
        assert result == [(904, 904), (4096, 4096)]
    finally:
        try:
            if con is not None:
                con.close()
        finally:
            if old_runner is None:
                os.environ.pop("VANE_RUNNER", None)
            else:
                os.environ["VANE_RUNNER"] = old_runner


def test_vane_function_batch_gpu_zero_stays_streaming():
    import pyarrow as pa

    import vane

    old_runner = os.environ.get("VANE_RUNNER")
    try:
        vane.configure(runner="ray")

        @vane.func.batch(return_dtype=pa.int32(), gpus=0)
        def identity(values):
            return values

        con = vane.connect()
        rel = con.sql("select i::INTEGER as x from range(2) t(i)")
        plan = rel.select(vane.col("x"), identity(vane.col("x"))).explain()

        assert "STREAMING_UDF" in plan
        assert "ray_block_stream_output:" in plan
    finally:
        if old_runner is None:
            os.environ.pop("VANE_RUNNER", None)
        else:
            os.environ["VANE_RUNNER"] = old_runner


def test_vane_function_batch_descriptor_binds_instance():
    import pyarrow as pa
    import pyarrow.compute as pc

    import vane

    class Scorer:
        def __init__(self, offset):
            self.offset = offset

        @vane.func.batch(return_dtype=pa.int64())
        def score(self, values):
            return pc.add(values, self.offset)

    result = Scorer(3).score(pa.array([1, 2], type=pa.int64()))

    assert result.to_pylist() == [4, 5]
