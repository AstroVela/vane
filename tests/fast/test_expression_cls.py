# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Expression-level actor UDF tests for ``vane.cls``."""

from __future__ import annotations

import pytest


def test_vane_cls_public_api():
    import vane

    assert "cls" in vane.__all__
    assert callable(vane.cls)
    assert callable(vane.cls.batch)
    assert isinstance(vane.col("x"), vane.Expression)


@pytest.mark.parametrize("actor_number", [None, False, True, 0, 1.0, 2])
def test_vane_cls_requires_exactly_one_strict_integer_actor(actor_number):
    import vane

    class Prefixer:
        def __call__(self, text):
            return text

    with pytest.raises(
        vane.InvalidInputException,
        match=r"actor_number must be exactly 1.*multi-actor state",
    ):
        vane.cls(actor_number=actor_number, return_dtype="VARCHAR")(Prefixer)


def test_vane_cls_requires_return_dtype():
    import vane

    class Prefixer:
        def __call__(self, text):
            return text

    with pytest.raises(vane.InvalidInputException, match="return_dtype"):
        vane.cls(actor_number=1)(Prefixer)


def test_vane_cls_rejects_empty_explicit_name_without_defaulting():
    import vane

    with pytest.raises(vane.InvalidInputException, match="name must be a non-empty string"):

        @vane.cls(actor_number=1, return_dtype="VARCHAR", name="")
        class Prefixer:
            def __call__(self, text):
                return text


def test_vane_cls_physical_payload_marks_stateful_side_effects(monkeypatch):
    import uuid

    import vane

    monkeypatch.setenv("VANE_RUNNER", "ray")

    @vane.cls(actor_number=1, return_dtype="INTEGER", name="stateful_counter")
    class Counter:
        def __call__(self, value):
            return value

    con = vane.connect()
    try:
        relation = con.sql("select 1::INTEGER as value").select(Counter()(vane.col("value")).alias("out"))
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
            relation,
            str(uuid.uuid4()),
        ).to_physical_plan(con)
        nodes = plan.collect_udf_nodes(conn=con)
    finally:
        con.close()

    assert len(nodes) == 1
    payload = nodes[0]["payload"]
    assert payload["udf_name"] == "stateful_counter"
    assert payload.get("stateful") is True
    assert payload.get("side_effects") is True
    assert payload["actor_number"] == 1
    assert "state_scope" not in payload


def test_vane_cls_immediate_call_reuses_eager_instance():
    import vane

    created: list[object] = []

    @vane.cls(actor_number=1, return_dtype="VARCHAR")
    class Prefixer:
        def __init__(self, prefix):
            created.append(self)
            self.prefix = prefix

        def __call__(self, text):
            return f"{self.prefix}{text}"

    prefixer = Prefixer("p:")

    assert prefixer("x") == "p:x"
    assert prefixer("y") == "p:y"
    assert len(created) == 1


def test_vane_cls_expression_local_single_column():
    import vane

    @vane.cls(actor_number=1, return_dtype="VARCHAR")
    class Prefixer:
        def __init__(self, prefix):
            self.prefix = prefix

        def __call__(self, text):
            return f"{self.prefix}{text}"

    con = vane.connect()
    rel = con.sql("select 'a'::VARCHAR as text union all select 'b'::VARCHAR as text")
    expr = Prefixer("p:")(vane.col("text"))

    assert sorted(rel.select(expr.alias("out")).fetchall()) == [("p:a",), ("p:b",)]


def test_vane_cls_expression_pickles_duckdb_type_annotations():
    import vane

    class AddOne:
        def __call__(self, value):
            return value + 1

    AddOne.__call__.__annotations__ = {
        "value": vane.sqltypes.INTEGER,
        "return": vane.sqltypes.INTEGER,
    }
    decorated = vane.cls(actor_number=1, return_dtype="INTEGER")(AddOne)

    con = vane.connect()
    rel = con.sql("select i::INTEGER as x from range(3) t(i)")

    assert rel.select(decorated()(vane.col("x")).alias("y")).fetchall() == [(1,), (2,), (3,)]


def test_vane_cls_expression_local_multi_column_and_literal_kwargs():
    import vane

    @vane.cls(actor_number=1, return_dtype="VARCHAR")
    class Joiner:
        def __call__(self, left, right, *, sep):
            return f"{left}{sep}{right}"

    con = vane.connect()
    rel = con.sql("select 'a'::VARCHAR as left_value, 'b'::VARCHAR as right_value")
    expr = Joiner()(vane.col("left_value"), vane.col("right_value"), sep=":")

    assert rel.select(expr.alias("out")).fetchall() == [("a:b",)]


def test_vane_cls_rejects_expression_keyword_arguments():
    import vane

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class AddOne:
        def __call__(self, value):
            return value + 1

    with pytest.raises(vane.InvalidInputException, match="keyword arguments"):
        AddOne()(value=vane.col("x"))


def test_vane_cls_row_actor_adapter_propagates_nulls_without_calling_user_code():
    import pyarrow as pa

    from vane._expression_udf import _build_row_actor_class

    calls: list[str] = []

    class Upper:
        def __call__(self, value):
            calls.append(value)
            return value.upper()

    Actor = _build_row_actor_class(Upper, (), {}, ["text"], "out", pa.string())
    result = Actor()(pa.table({"text": ["a", None, "b"]}))

    assert result.to_pydict() == {"out": ["A", None, "B"]}
    assert calls == ["a", "b"]


def test_vane_cls_row_actor_adapter_reuses_and_isolates_instance_state():
    import pyarrow as pa

    from vane._expression_udf import _build_row_actor_class

    class Counter:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, _value: object) -> int:
            self.calls += 1
            return self.calls

    Actor = _build_row_actor_class(Counter, (), {}, ["value"], "calls", pa.int64())
    first_actor = Actor()
    second_actor = Actor()

    assert first_actor(pa.table({"value": [10, 20]})).to_pydict() == {"calls": [1, 2]}
    assert first_actor(pa.table({"value": [30]})).to_pydict() == {"calls": [3]}
    assert second_actor(pa.table({"value": [40]})).to_pydict() == {"calls": [1]}


def test_vane_cls_row_actor_adapter_uses_fixed_size_list_arrow_type():
    import pyarrow as pa

    from vane._expression_udf import _build_row_actor_class

    class Embed:
        def __call__(self, value):
            return [float(value), float(value + 1), float(value + 2)]

    arrow_type = pa.list_(pa.float32(), 3)
    Actor = _build_row_actor_class(Embed, (), {}, ["x"], "embedding", arrow_type)
    result = Actor()(pa.table({"x": [1, 2]}))

    assert result.schema.field("embedding").type == arrow_type
    assert result.to_pydict() == {"embedding": [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]}


def test_vane_cls_lazy_payload_arguments_are_forwarded(monkeypatch):
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

    @vane.cls(actor_number=1, return_dtype="INTEGER", name="plus", gpus=0.5)
    class Plus:
        def __call__(self, value):
            return value + 1

    assert Plus()(vane.col("x")) is sentinel
    assert captured["kwargs"]["name"] == "plus"
    assert captured["kwargs"]["schema"] == {"plus": "INTEGER"}
    assert captured["kwargs"]["row_preserving"] is True
    assert captured["kwargs"]["actor_number"] == 1
    assert captured["kwargs"]["gpus"] == 0.5
    assert list(captured["kwargs"]["inputs"]) == ["value"]


def test_vane_cls_receiver_name_is_not_required_to_be_self(monkeypatch):
    import vane
    import vane._expression_udf as expression_udf

    captured = {}
    sentinel = object()

    def fake_build_actor_map_batches_expression(*args, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(expression_udf, "_build_actor_map_batches_expression", fake_build_actor_map_batches_expression)

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class AddOne:
        def __call__(this, value):
            return value + 1

    assert AddOne()(vane.col("value")) is sentinel
    assert list(captured["inputs"]) == ["value"]


def test_vane_cls_expression_supports_literal_keyword_only_call_config():
    import vane

    @vane.cls(actor_number=1, return_dtype="VARCHAR")
    class Joiner:
        def __call__(self, left, right, *, separator):
            return f"{left}{separator}{right}"

    conn = vane.connect()
    expression = Joiner()(vane.col("left_value"), vane.col("right_value"), separator="::")
    rows = conn.sql("SELECT 'a' AS left_value, 'b' AS right_value").select(expression.alias("result")).fetchall()

    assert rows == [("a::b",)]


def test_vane_cls_expression_supports_staticmethod_call():
    import vane

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class AddOne:
        @staticmethod
        def __call__(value):
            return value + 1

    conn = vane.connect()
    result = conn.sql("SELECT 1::INTEGER AS value").select(AddOne()(vane.col("value")).alias("result"))

    assert result.fetchall() == [(2,)]


def test_vane_cls_expression_supports_classmethod_call():
    import vane

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class AddOne:
        @classmethod
        def __call__(cls, value):
            return value + 1

    conn = vane.connect()
    result = conn.sql("SELECT 1::INTEGER AS value").select(AddOne()(vane.col("value")).alias("result"))

    assert result.fetchall() == [(2,)]


@pytest.mark.parametrize("actor_number", [False, True])
def test_vane_cls_rejects_actor_number_bool(actor_number):
    import vane

    class Identity:
        def __call__(self, value):
            return value

    with pytest.raises(
        vane.InvalidInputException,
        match=r"actor_number.*bool|actor_number.*exactly 1|actor_number.*integer",
    ):
        vane.cls(actor_number=actor_number, return_dtype="INTEGER")(Identity)


def test_vane_cls_return_dtype_pyarrow_int64_expression_round_trip():
    import pyarrow as pa

    import vane

    @vane.cls(actor_number=1, return_dtype=pa.int64())
    class Widen:
        def __call__(self, value):
            return value + 2**40

    conn = vane.connect()
    result = conn.sql("SELECT 1::INTEGER AS value").select(Widen()(vane.col("value")).alias("result"))

    assert [str(dtype) for dtype in result.types] == ["BIGINT"]
    assert result.fetchall() == [(2**40 + 1,)]


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("struct", id="struct"),
        pytest.param("map", id="map"),
    ],
)
def test_unsupported_pyarrow_datatype_is_rejected_during_canonicalization(dtype):
    import pyarrow as pa

    import vane

    unsupported = {
        "struct": pa.struct([("value", pa.int64())]),
        "map": pa.map_(pa.string(), pa.int64()),
    }[dtype]

    class Identity:
        def __call__(self, value):
            return value

    with pytest.raises(vane.InvalidInputException) as exc_info:
        vane.cls(actor_number=1, return_dtype=unsupported)(Identity)

    assert str(unsupported) in str(exc_info.value)
    assert "not supported" in str(exc_info.value)


def test_vane_cls_timestamp_naive_datetime_preserves_wall_clock_and_microseconds():
    from datetime import datetime

    import vane

    expected = datetime(2024, 2, 3, 4, 5, 6, 789123)

    @vane.cls(actor_number=1, return_dtype="TIMESTAMP")
    class NaiveTimestamp:
        def __call__(self, _value):
            return expected

    conn = vane.connect()
    result = conn.sql("SELECT 1::INTEGER AS value").select(NaiveTimestamp()(vane.col("value")).alias("result"))

    assert [str(dtype) for dtype in result.types] == ["TIMESTAMP"]
    assert result.fetchall() == [(expected,)]


def test_vane_cls_eager_call_preserves_aware_datetime_object_semantics():
    from datetime import datetime, timedelta, timezone

    import vane

    aware = datetime(
        2024,
        2,
        3,
        4,
        5,
        6,
        789123,
        tzinfo=timezone(timedelta(hours=5, minutes=30)),
    )

    @vane.cls(actor_number=1, return_dtype="TIMESTAMP")
    class EchoTimestamp:
        def __call__(self, value):
            return value

    result = EchoTimestamp()(aware)

    assert result is aware
    assert result.utcoffset() == timedelta(hours=5, minutes=30)


def _assert_expression_rejects_aware_timestamp(offset_hours):
    from datetime import datetime, timedelta, timezone

    import vane

    aware = datetime(2024, 2, 3, 4, 5, 6, 789123, tzinfo=timezone(timedelta(hours=offset_hours)))

    @vane.cls(actor_number=1, return_dtype="TIMESTAMP")
    class AwareTimestamp:
        def __call__(self, _value):
            return aware

    conn = vane.connect()
    result = conn.sql("SELECT 1::INTEGER AS value").select(AwareTimestamp()(vane.col("value")).alias("result"))
    with pytest.raises(
        Exception,
        match=r"TIMESTAMP is timezone-naive; use a naive datetime or a supported TIMESTAMPTZ contract",
    ):
        result.fetchall()


def test_vane_cls_timestamp_rejects_positive_offset_aware_datetime():
    _assert_expression_rejects_aware_timestamp(5.5)


def test_vane_cls_timestamp_rejects_negative_offset_aware_datetime():
    _assert_expression_rejects_aware_timestamp(-7)


def test_timezone_aware_pyarrow_timestamp_is_rejected_if_timestamptz_is_unsupported():
    import pyarrow as pa

    import vane

    class Identity:
        def __call__(self, value):
            return value

    with pytest.raises(
        vane.InvalidInputException,
        match=r"timestamp\[us, tz=UTC\].*not supported|timezone-aware.*TIMESTAMPTZ",
    ):
        vane.cls(actor_number=1, return_dtype=pa.timestamp("us", tz="UTC"))(Identity)


def test_vane_cls_eager_zero_argument_call_remains_supported():
    import vane

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class Constant:
        def __call__(self):
            return 42

    assert Constant()() == 42


def test_vane_cls_null_contract_skips_user_code_only_for_expression_path():
    import vane

    @vane.cls(actor_number=1, return_dtype="VARCHAR")
    class NullAware:
        def __call__(self, value):
            return "called-with-null" if value is None else value.upper()

    instance = NullAware()
    assert instance(None) == "called-with-null"

    conn = vane.connect()
    result = conn.sql("SELECT value FROM (VALUES ('x'::VARCHAR), (NULL::VARCHAR)) AS t(value)").select(
        instance(vane.col("value")).alias("result")
    )
    assert sorted(result.fetchall(), key=lambda row: row[0] is None) == [("X",), (None,)]
