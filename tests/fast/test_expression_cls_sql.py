"""SQL registration tests for ``vane.cls`` and ``vane.cls.batch``."""

from __future__ import annotations

import pytest


def _assert_sql_alias_absent(conn, alias):
    assert conn.execute(
        "SELECT count(*) FROM duckdb_functions() WHERE function_name = ?",
        [alias],
    ).fetchone() == (0,)


def _sql_function_contract(conn, alias):
    return conn.execute(
        """
        SELECT return_type, parameters, parameter_types, has_side_effects, function_oid
        FROM duckdb_functions()
        WHERE function_name = ?
        ORDER BY parameter_types
        """,
        [alias],
    ).fetchall()


def test_vane_cls_registered_for_sql_projection_and_named_arguments():
    import vane

    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="VARCHAR", name="prefixer")
    class Prefixer:
        def __init__(self, prefix):
            self.prefix = prefix

        def __call__(self, text):
            return f"{self.prefix}{text}"

    vane.attach_function(Prefixer("p:"), connection=conn, alias="prefixer_sql", parameters=["VARCHAR"])

    rows = conn.sql("""
        SELECT prefixer_sql(text := value) AS y
        FROM (VALUES ('a'), ('b')) AS t(value)
        ORDER BY value
    """).fetchall()

    assert rows == [("p:a",), ("p:b",)]


def test_vane_cls_sql_reuses_state_across_batches():
    import vane

    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class Counter:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, _value: object) -> int:
            self.calls += 1
            return self.calls

    vane.attach_function(
        Counter(),
        connection=conn,
        alias="stateful_counter_sql",
        parameters=["INTEGER"],
    )

    # DuckDB's 2048-row standard vector size makes this three actor submits.
    rows = conn.sql("""
        SELECT stateful_counter_sql(i::INTEGER) AS call
        FROM range(4097) t(i)
        ORDER BY i
    """).fetchall()

    assert len(rows) == 4097
    assert sorted(row[0] for row in rows) == list(range(1, 4098))


def test_vane_cls_batch_registered_for_sql_projection():
    import pyarrow as pa

    import vane

    conn = vane.connect()

    @vane.cls.batch(actor_number=1, schema={"score": "INTEGER"}, row_preserving=True)
    class Scorer:
        def __init__(self, offset):
            self.offset = offset

        def __call__(self, table: pa.Table) -> pa.Table:
            values = table.column("value").to_pylist()
            return pa.table({"score": [value + self.offset for value in values]})

    vane.attach_function(
        Scorer(10),
        connection=conn,
        alias="score_sql",
        input_names=["value"],
        parameters=["INTEGER"],
    )

    rows = conn.sql("""
        SELECT score_sql(i::INTEGER) AS y
        FROM range(3) t(i)
        ORDER BY i
    """).fetchall()

    assert rows == [(10,), (11,), (12,)]


def test_vane_cls_batch_sql_rejects_non_row_preserving_before_registration():
    import pyarrow as pa

    import vane

    conn = vane.connect()

    @vane.cls.batch(actor_number=1, schema={"value": "INTEGER"}, row_preserving=False)
    class ExpandingBatch:
        def __call__(self, table: pa.Table) -> pa.Table:
            values = table.column("value").to_pylist()
            return pa.table({"value": values + values})

    with pytest.raises(
        vane.InvalidInputException,
        match=r"row_preserving=False.*expression API|SQL attach.*row-preserving",
    ):
        vane.attach_function(
            ExpandingBatch(),
            connection=conn,
            alias="expanding_batch_sql",
            input_names=["value"],
            parameters=["INTEGER"],
        )

    with pytest.raises(Exception, match="expanding_batch_sql"):
        conn.sql("SELECT expanding_batch_sql(1::INTEGER)").fetchall()


def test_failed_non_row_preserving_attach_does_not_replace_existing_alias():
    import pyarrow as pa

    import vane

    conn = vane.connect()

    @vane.cls.batch(actor_number=1, schema={"value": "INTEGER"}, row_preserving=True)
    class StableBatch:
        def __call__(self, table: pa.Table) -> pa.Table:
            values = table.column("value").to_pylist()
            return pa.table({"value": [value + 1 for value in values]})

    @vane.cls.batch(actor_number=1, schema={"value": "INTEGER"}, row_preserving=False)
    class UnsafeBatch:
        def __call__(self, table: pa.Table) -> pa.Table:
            values = table.column("value").to_pylist()
            return pa.table({"value": values + values})

    vane.attach_function(
        StableBatch(),
        connection=conn,
        alias="preserved_batch_sql",
        input_names=["value"],
        parameters=["INTEGER"],
    )

    with pytest.raises(vane.InvalidInputException, match="row_preserving=False"):
        vane.attach_function(
            UnsafeBatch(),
            connection=conn,
            alias="preserved_batch_sql",
            input_names=["value"],
            parameters=["INTEGER"],
            replace=True,
        )

    assert conn.sql("SELECT preserved_batch_sql(1::INTEGER)").fetchall() == [(2,)]


def test_vane_cls_replace_validation_failure_preserves_old_alias():
    import vane

    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class AddOne:
        def __call__(self, value):
            return value + 1

    vane.attach_function(AddOne(), connection=conn, alias="class_rollback_sql", parameters=["INTEGER"])

    with pytest.raises(vane.InvalidInputException, match="input_names count"):
        vane.attach_function(
            AddOne(),
            connection=conn,
            alias="class_rollback_sql",
            parameters=["INTEGER"],
            input_names=["value", "extra"],
            replace=True,
        )

    assert conn.sql("SELECT class_rollback_sql(1::INTEGER)").fetchall() == [(2,)]


def test_vane_cls_batch_sql_reuses_state_across_batches():
    from collections import Counter

    import pyarrow as pa

    import vane

    conn = vane.connect()

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

    vane.attach_function(
        BatchCounter(),
        connection=conn,
        alias="stateful_batch_counter_sql",
        input_names=["value"],
        parameters=["INTEGER"],
    )

    # DuckDB's 2048-row standard vector size makes this three actor submits.
    rows = conn.sql("""
        SELECT stateful_batch_counter_sql(i::INTEGER) AS batch_call
        FROM range(4097) t(i)
        ORDER BY i
    """).fetchall()

    assert len(rows) == 4097
    calls = Counter(row[0] for row in rows)
    assert sorted(calls) == [1, 2, 3]
    assert sorted(calls.values()) == [1, 2048, 2048]


def test_vane_cls_python_expression_and_sql_results_match():
    import vane

    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="VARCHAR", name="prefixer")
    class Prefixer:
        def __init__(self, prefix):
            self.prefix = prefix

        def __call__(self, text):
            return f"{self.prefix}{text}"

    prefixer = Prefixer("p:")
    expr_rows = (
        conn.sql("select 'a'::VARCHAR as text union all select 'b'::VARCHAR as text")
        .select(prefixer(vane.col("text")).alias("y"))
        .order("y")
        .fetchall()
    )

    vane.attach_function(prefixer, connection=conn, alias="prefixer_sql", parameters=["VARCHAR"])
    sql_rows = conn.sql("""
        SELECT prefixer_sql(text) AS y
        FROM (VALUES ('a'), ('b')) AS t(text)
        ORDER BY y
    """).fetchall()

    assert expr_rows == sql_rows


def test_vane_cls_detach_removes_sql_function():
    import vane

    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class AddOne:
        def __call__(self, value):
            return value + 1

    vane.attach_function(AddOne(), connection=conn, alias="actor_add_one", parameters=["INTEGER"])
    assert conn.sql("SELECT actor_add_one(1::INTEGER)").fetchall() == [(2,)]

    vane.detach_function("actor_add_one", connection=conn)
    with pytest.raises(Exception, match="actor_add_one"):
        conn.sql("SELECT actor_add_one(1::INTEGER)").fetchall()


def test_vane_cls_batch_sql_requires_input_names():
    import vane

    @vane.cls.batch(actor_number=1, schema={"x": "INTEGER"}, row_preserving=True)
    class Identity:
        def __call__(self, table):
            return table

    with pytest.raises(vane.InvalidInputException, match="input_names is required"):
        vane.attach_function(Identity(), alias="bad_identity", parameters=["INTEGER"])


def test_raw_actor_class_with_required_constructor_args_fails_at_attach():
    import pyarrow as pa

    import vane

    class RawScorer:
        def __init__(self, offset):
            self.offset = offset

        def __call__(self, table: pa.Table) -> pa.Table:
            values = table.column("value").to_pylist()
            return pa.table({"score": [value + self.offset for value in values]})

    with pytest.raises(vane.InvalidInputException, match="zero-argument"):
        vane.attach_function(
            RawScorer,
            alias="raw_scorer",
            input_names=["value"],
            schema={"score": "INTEGER"},
            parameters=["INTEGER"],
            actor_number=1,
            gpus=0,
        )


def test_vane_cls_sql_explain_resolves_actor_backend_from_current_runner(monkeypatch):
    import vane

    monkeypatch.setenv("VANE_RUNNER", "ray")
    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="VARCHAR", name="prefixer")
    class Prefixer:
        def __init__(self, prefix):
            self.prefix = prefix

        def __call__(self, text):
            return f"{self.prefix}{text}"

    vane.attach_function(Prefixer("p:"), connection=conn, alias="prefixer_sql", parameters=["VARCHAR"])
    plan = conn.sql("EXPLAIN SELECT prefixer_sql('x'::VARCHAR)").fetchall()
    text = "\n".join(str(row) for row in plan)

    assert "ray_actor" in text
    assert "actor_pool_size" in text
    assert "subprocess_actor" not in text


def test_vane_cls_batch_sql_explain_resolves_actor_backend_from_current_runner(monkeypatch):
    import pyarrow as pa

    import vane

    monkeypatch.setenv("VANE_RUNNER", "ray")
    conn = vane.connect()

    @vane.cls.batch(actor_number=1, schema={"score": "INTEGER"}, row_preserving=True)
    class Scorer:
        def __call__(self, table: pa.Table) -> pa.Table:
            return pa.table({"score": table.column("value")})

    vane.attach_function(
        Scorer(),
        connection=conn,
        alias="score_sql",
        input_names=["value"],
        parameters=["INTEGER"],
    )
    plan = conn.sql("EXPLAIN SELECT score_sql(1::INTEGER)").fetchall()
    text = "\n".join(str(row) for row in plan)

    assert "ray_actor" in text
    assert "actor_pool_size" in text
    assert "subprocess_actor" not in text


def test_vane_cls_attach_rejects_uninstantiated_decorator():
    import vane

    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class AddOne:
        def __call__(self, value):
            return value + 1

    alias = "uninstantiated_cls_sql"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"SQL registration for vane\.cls requires an instantiated class; use AddOne\(\)",
    ):
        vane.attach_function(AddOne, connection=conn, alias=alias, parameters=["INTEGER"])

    _assert_sql_alias_absent(conn, alias)


def test_vane_cls_attach_rejects_uninstantiated_decorator_even_with_return_dtype():
    import vane

    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class AddOne:
        def __call__(self, value):
            return value + 1

    alias = "uninstantiated_cls_with_dtype_sql"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"SQL registration for vane\.cls requires an instantiated class; use AddOne\(\)",
    ):
        vane.attach_function(
            AddOne,
            connection=conn,
            alias=alias,
            parameters=["INTEGER"],
            return_dtype="INTEGER",
        )

    _assert_sql_alias_absent(conn, alias)


def test_vane_cls_batch_attach_rejects_uninstantiated_decorator():
    import pyarrow as pa

    import vane

    conn = vane.connect()

    @vane.cls.batch(actor_number=1, schema={"result": "INTEGER"}, row_preserving=True)
    class IdentityBatch:
        def __call__(self, table):
            return pa.table({"result": table.column("value")})

    alias = "uninstantiated_cls_batch_sql"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"SQL registration for vane\.cls\.batch requires an instantiated class; use IdentityBatch\(\)",
    ):
        vane.attach_function(
            IdentityBatch,
            connection=conn,
            alias=alias,
            input_names=["value"],
            parameters=["INTEGER"],
        )

    _assert_sql_alias_absent(conn, alias)


def test_vane_cls_attach_requires_parameters_with_explicit_input_names():
    import vane

    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class AddOne:
        def __call__(self, value):
            return value + 1

    alias = "cls_requires_parameters_sql"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"parameters is required for SQL vane\.cls registration",
    ):
        vane.attach_function(AddOne(), connection=conn, alias=alias, input_names=["value"])

    _assert_sql_alias_absent(conn, alias)


def test_vane_cls_batch_attach_requires_parameters():
    import pyarrow as pa

    import vane

    conn = vane.connect()

    @vane.cls.batch(actor_number=1, schema={"result": "INTEGER"}, row_preserving=True)
    class IdentityBatch:
        def __call__(self, table):
            return pa.table({"result": table.column("value")})

    alias = "cls_batch_requires_parameters_sql"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"parameters is required for SQL vane\.cls\.batch registration",
    ):
        vane.attach_function(IdentityBatch(), connection=conn, alias=alias, input_names=["value"])

    _assert_sql_alias_absent(conn, alias)


def test_vane_cls_attach_rejects_return_dtype_override():
    import vane

    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class Identity:
        def __call__(self, value):
            return value

    alias = "cls_return_dtype_override_sql"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"return_dtype cannot override vane\.cls instance configuration.*@vane\.cls",
    ):
        vane.attach_function(
            Identity(),
            connection=conn,
            alias=alias,
            parameters=["INTEGER"],
            return_dtype="BIGINT",
        )

    _assert_sql_alias_absent(conn, alias)


def test_vane_cls_attach_rejects_schema_argument():
    import vane

    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class Identity:
        def __call__(self, value):
            return value

    alias = "cls_schema_override_sql"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"schema is not valid for SQL vane\.cls registration.*@vane\.cls",
    ):
        vane.attach_function(
            Identity(),
            connection=conn,
            alias=alias,
            parameters=["INTEGER"],
            schema={"result": "INTEGER"},
        )

    _assert_sql_alias_absent(conn, alias)


def test_vane_cls_attach_rejects_gpus_override():
    import vane

    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class Identity:
        def __call__(self, value):
            return value

    alias = "cls_gpus_override_sql"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"gpus cannot override vane\.cls instance configuration.*@vane\.cls",
    ):
        vane.attach_function(Identity(), connection=conn, alias=alias, parameters=["INTEGER"], gpus=0)

    _assert_sql_alias_absent(conn, alias)


def test_vane_cls_attach_rejects_actor_number_override():
    import vane

    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class Identity:
        def __call__(self, value):
            return value

    alias = "cls_actor_number_override_sql"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"actor_number cannot override vane\.cls instance configuration.*@vane\.cls",
    ):
        vane.attach_function(Identity(), connection=conn, alias=alias, parameters=["INTEGER"], actor_number=1)

    _assert_sql_alias_absent(conn, alias)


def test_vane_cls_batch_attach_rejects_return_dtype_argument():
    import vane

    conn = vane.connect()

    @vane.cls.batch(actor_number=1, schema={"result": "INTEGER"}, row_preserving=True)
    class IdentityBatch:
        def __call__(self, table):
            return table

    alias = "cls_batch_return_dtype_override_sql"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"return_dtype.*vane\.cls\.batch.*@vane\.cls\.batch",
    ):
        vane.attach_function(
            IdentityBatch(),
            connection=conn,
            alias=alias,
            input_names=["value"],
            parameters=["INTEGER"],
            return_dtype="INTEGER",
        )

    _assert_sql_alias_absent(conn, alias)


def test_vane_cls_batch_attach_rejects_gpus_override():
    import vane

    conn = vane.connect()

    @vane.cls.batch(actor_number=1, schema={"result": "INTEGER"}, row_preserving=True)
    class IdentityBatch:
        def __call__(self, table):
            return table

    alias = "cls_batch_gpus_override_sql"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"gpus cannot override vane\.cls\.batch instance configuration.*@vane\.cls\.batch",
    ):
        vane.attach_function(
            IdentityBatch(),
            connection=conn,
            alias=alias,
            input_names=["value"],
            parameters=["INTEGER"],
            gpus=0,
        )

    _assert_sql_alias_absent(conn, alias)


def test_vane_cls_batch_attach_rejects_actor_number_override():
    import vane

    conn = vane.connect()

    @vane.cls.batch(actor_number=1, schema={"result": "INTEGER"}, row_preserving=True)
    class IdentityBatch:
        def __call__(self, table):
            return table

    alias = "cls_batch_actor_number_override_sql"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"actor_number cannot override vane\.cls\.batch instance configuration.*@vane\.cls\.batch",
    ):
        vane.attach_function(
            IdentityBatch(),
            connection=conn,
            alias=alias,
            input_names=["value"],
            parameters=["INTEGER"],
            actor_number=1,
        )

    _assert_sql_alias_absent(conn, alias)


def test_vane_cls_attach_infers_receiver_named_this():
    import vane

    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class AddOne:
        def __call__(this, value):
            return value + 1

    vane.attach_function(AddOne(), connection=conn, alias="receiver_this_sql", parameters=["INTEGER"])

    assert conn.sql("SELECT receiver_this_sql(value := 1::INTEGER)").fetchall() == [(2,)]


def test_vane_cls_attach_rejects_too_few_required_arguments():
    import vane

    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class Add:
        def __call__(self, left, right):
            return left + right

    alias = "cls_too_few_required_sql"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"vane\.cls.*requires at least 2.*received 1|1.*fewer than.*2",
    ):
        vane.attach_function(Add(), connection=conn, alias=alias, parameters=["INTEGER"])

    _assert_sql_alias_absent(conn, alias)


def test_vane_cls_attach_allows_omitted_defaulted_argument():
    import vane

    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class Add:
        def __call__(self, left, right=10):
            return left + right

    vane.attach_function(Add(), connection=conn, alias="cls_defaulted_sql", parameters=["INTEGER"])

    assert conn.sql("SELECT cls_defaulted_sql(left := 2::INTEGER)").fetchall() == [(12,)]


def test_vane_cls_attach_rejects_required_keyword_only_argument():
    import vane

    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class Scale:
        def __call__(self, value, *, factor):
            return value * factor

    alias = "cls_required_keyword_only_sql"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"required keyword-only.*factor|factor.*cannot be supplied by SQL",
    ):
        vane.attach_function(Scale(), connection=conn, alias=alias, parameters=["INTEGER"])

    _assert_sql_alias_absent(conn, alias)


def test_vane_cls_attach_supports_staticmethod_call():
    import vane

    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class AddOne:
        @staticmethod
        def __call__(value):
            return value + 1

    vane.attach_function(AddOne(), connection=conn, alias="cls_staticmethod_sql", parameters=["INTEGER"])

    assert conn.sql("SELECT cls_staticmethod_sql(value := 1::INTEGER)").fetchall() == [(2,)]


def test_vane_cls_attach_supports_classmethod_call():
    import vane

    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class AddOne:
        @classmethod
        def __call__(cls, value):
            return value + 1

    vane.attach_function(AddOne(), connection=conn, alias="cls_classmethod_sql", parameters=["INTEGER"])

    assert conn.sql("SELECT cls_classmethod_sql(value := 1::INTEGER)").fetchall() == [(2,)]


def test_vane_cls_attach_allows_varargs_with_explicit_input_names():
    import vane

    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class SumValues:
        def __call__(self, first, *rest):
            return first + sum(rest)

    vane.attach_function(
        SumValues(),
        connection=conn,
        alias="cls_varargs_sql",
        input_names=["first", "second", "third"],
        parameters=["INTEGER", "INTEGER", "INTEGER"],
    )

    assert conn.sql("SELECT cls_varargs_sql(third := 3, first := 1, second := 2)").fetchall() == [(6,)]


def test_vane_cls_attach_validates_explicit_input_names_arity():
    import vane

    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class Add:
        def __call__(self, left, right):
            return left + right

    alias = "cls_explicit_arity_sql"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"vane\.cls.*requires at least 2.*received 1|input_names.*arity",
    ):
        vane.attach_function(
            Add(),
            connection=conn,
            alias=alias,
            input_names=["left"],
            parameters=["INTEGER"],
        )

    _assert_sql_alias_absent(conn, alias)


def test_vane_cls_attach_rejects_zero_input_sql_udf():
    import vane

    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class Constant:
        def __call__(self):
            return 42

    alias = "cls_zero_input_sql"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"zero-input.*vane\.cls.*SQL.*not supported|SQL.*zero-input.*not supported",
    ):
        vane.attach_function(Constant(), connection=conn, alias=alias, parameters=[])

    _assert_sql_alias_absent(conn, alias)


def test_failed_class_preflight_with_replace_preserves_old_alias():
    import vane

    conn = vane.connect()

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class AddOne:
        def __call__(self, value):
            return value + 1

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class AddHundred:
        def __call__(self, value):
            return value + 100

    alias = "class_preflight_rollback_sql"
    vane.attach_function(AddOne(), connection=conn, alias=alias, parameters=["INTEGER"])
    before = _sql_function_contract(conn, alias)

    with pytest.raises(vane.InvalidInputException, match=r"return_dtype cannot override vane\.cls"):
        vane.attach_function(
            AddHundred(),
            connection=conn,
            alias=alias,
            parameters=["INTEGER"],
            return_dtype="BIGINT",
            replace=True,
        )

    assert _sql_function_contract(conn, alias) == before
    assert conn.sql(f"SELECT {alias}(1::INTEGER), typeof({alias}(1::INTEGER))").fetchall() == [(2, "INTEGER")]


def test_vane_cls_replace_apply_phase_serialization_failure_preserves_old_alias():
    import gc
    import pickle
    import weakref

    import vane

    class Offset:
        def __init__(self, value):
            self.value = value

    class UnpicklableConstructorArgument:
        def __reduce__(self):
            raise pickle.PicklingError("intentional vane.cls replacement serialization failure")

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class AddOffset:
        def __init__(self, offset):
            self.offset = offset

        def __call__(self, value):
            return value + self.offset.value

    conn = vane.connect()
    alias = "class_apply_rollback_sql"
    old_offset = Offset(1)
    old_offset_ref = weakref.ref(old_offset)
    old_instance = AddOffset(old_offset)
    vane.attach_function(old_instance, connection=conn, alias=alias, parameters=["INTEGER"])
    before = _sql_function_contract(conn, alias)

    with pytest.raises(pickle.PicklingError, match=r"intentional vane.cls replacement serialization failure"):
        vane.attach_function(
            AddOffset(UnpicklableConstructorArgument()),
            connection=conn,
            alias=alias,
            parameters=["INTEGER"],
            replace=True,
        )

    del old_instance, old_offset
    gc.collect()
    assert old_offset_ref() is not None
    assert _sql_function_contract(conn, alias) == before
    assert conn.sql(f"SELECT {alias}(1::INTEGER), typeof({alias}(1::INTEGER))").fetchall() == [(2, "INTEGER")]


def test_vane_cls_batch_replace_apply_phase_serialization_failure_preserves_old_alias():
    import gc
    import pickle
    import weakref

    import pyarrow as pa

    import vane

    class Offset:
        def __init__(self, value):
            self.value = value

    class UnpicklableConstructorArgument:
        def __reduce__(self):
            raise pickle.PicklingError("intentional vane.cls.batch replacement serialization failure")

    @vane.cls.batch(actor_number=1, schema={"result": "INTEGER"}, row_preserving=True)
    class AddOffsetBatch:
        def __init__(self, offset):
            self.offset = offset

        def __call__(self, table):
            values = table.column("value").to_pylist()
            return pa.table({"result": [value + self.offset.value for value in values]})

    conn = vane.connect()
    alias = "class_batch_apply_rollback_sql"
    old_offset = Offset(1)
    old_offset_ref = weakref.ref(old_offset)
    old_instance = AddOffsetBatch(old_offset)
    vane.attach_function(
        old_instance,
        connection=conn,
        alias=alias,
        input_names=["value"],
        parameters=["INTEGER"],
    )
    before = _sql_function_contract(conn, alias)

    with pytest.raises(pickle.PicklingError, match=r"intentional vane\.cls\.batch replacement serialization failure"):
        vane.attach_function(
            AddOffsetBatch(UnpicklableConstructorArgument()),
            connection=conn,
            alias=alias,
            input_names=["value"],
            parameters=["INTEGER"],
            replace=True,
        )

    del old_instance, old_offset
    gc.collect()
    assert old_offset_ref() is not None
    assert _sql_function_contract(conn, alias) == before
    assert conn.sql(f"SELECT {alias}(1::INTEGER), typeof({alias}(1::INTEGER))").fetchall() == [(2, "INTEGER")]


def test_vane_cls_sql_timestamp_naive_preserves_wall_clock_and_microseconds():
    from datetime import datetime

    import vane

    expected = datetime(2024, 2, 3, 4, 5, 6, 789123)

    @vane.cls(actor_number=1, return_dtype="TIMESTAMP")
    class NaiveTimestamp:
        def __call__(self, _value):
            return expected

    conn = vane.connect()
    vane.attach_function(NaiveTimestamp(), connection=conn, alias="cls_naive_timestamp_sql", parameters=["INTEGER"])

    assert conn.sql(
        "SELECT cls_naive_timestamp_sql(1::INTEGER), typeof(cls_naive_timestamp_sql(1::INTEGER))"
    ).fetchall() == [(expected, "TIMESTAMP")]


def _assert_sql_rejects_aware_timestamp(offset_hours, alias):
    from datetime import datetime, timedelta, timezone

    import vane

    aware = datetime(2024, 2, 3, 4, 5, 6, 789123, tzinfo=timezone(timedelta(hours=offset_hours)))

    @vane.cls(actor_number=1, return_dtype="TIMESTAMP")
    class AwareTimestamp:
        def __call__(self, _value):
            return aware

    conn = vane.connect()
    vane.attach_function(AwareTimestamp(), connection=conn, alias=alias, parameters=["INTEGER"])
    with pytest.raises(
        Exception,
        match=r"TIMESTAMP is timezone-naive; use a naive datetime or a supported TIMESTAMPTZ contract",
    ):
        conn.sql(f"SELECT {alias}(1::INTEGER)").fetchall()


def test_vane_cls_sql_timestamp_rejects_positive_offset_aware_output():
    _assert_sql_rejects_aware_timestamp(5.5, "cls_positive_aware_timestamp_sql")


def test_vane_cls_sql_timestamp_rejects_negative_offset_aware_output():
    _assert_sql_rejects_aware_timestamp(-7, "cls_negative_aware_timestamp_sql")


def test_vane_cls_return_dtype_pyarrow_int64_sql_round_trip():
    import pyarrow as pa

    import vane

    @vane.cls(actor_number=1, return_dtype=pa.int64())
    class Widen:
        def __call__(self, value):
            return value + 2**40

    conn = vane.connect()
    vane.attach_function(Widen(), connection=conn, alias="cls_pyarrow_int64_sql", parameters=["INTEGER"])

    assert conn.sql(
        "SELECT cls_pyarrow_int64_sql(1::INTEGER), typeof(cls_pyarrow_int64_sql(1::INTEGER))"
    ).fetchall() == [(2**40 + 1, "BIGINT")]


def test_vane_cls_batch_schema_pyarrow_int64_sql_round_trip():
    import pyarrow as pa

    import vane

    @vane.cls.batch(actor_number=1, schema={"result": pa.int64()}, row_preserving=True)
    class WidenBatch:
        def __call__(self, table):
            values = table.column("value").to_pylist()
            return pa.table({"result": [value + 2**40 for value in values]})

    conn = vane.connect()
    vane.attach_function(
        WidenBatch(),
        connection=conn,
        alias="cls_batch_pyarrow_int64_sql",
        input_names=["value"],
        parameters=["INTEGER"],
    )

    assert conn.sql(
        "SELECT cls_batch_pyarrow_int64_sql(1::INTEGER), typeof(cls_batch_pyarrow_int64_sql(1::INTEGER))"
    ).fetchall() == [(2**40 + 1, "BIGINT")]


def test_vane_cls_eager_expression_sql_null_contract():
    import vane

    @vane.cls(actor_number=1, return_dtype="VARCHAR")
    class NullAware:
        def __call__(self, value):
            return "called-with-null" if value is None else value.upper()

    instance = NullAware()
    assert instance(None) == "called-with-null"

    conn = vane.connect()
    source = conn.sql("SELECT value FROM (VALUES ('x'::VARCHAR), (NULL::VARCHAR)) AS t(value)")
    expression_rows = source.select(instance(vane.col("value")).alias("result")).fetchall()

    vane.attach_function(instance, connection=conn, alias="cls_null_contract_sql", parameters=["VARCHAR"])
    sql_rows = conn.sql(
        "SELECT cls_null_contract_sql(value) FROM (VALUES ('x'::VARCHAR), (NULL::VARCHAR)) AS t(value)"
    ).fetchall()

    assert sorted(expression_rows, key=lambda row: row[0] is None) == [("X",), (None,)]
    assert sorted(sql_rows, key=lambda row: row[0] is None) == [("X",), (None,)]
