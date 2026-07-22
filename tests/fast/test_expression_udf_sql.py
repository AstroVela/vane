# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
import uuid

import pytest

import vane


def _assert_sql_alias_absent(conn, alias):
    assert conn.execute(
        "SELECT count(*) FROM duckdb_functions() WHERE function_name = ?",
        [alias],
    ).fetchone() == (0,)


def _fresh_physical_udf_payload(relation):
    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    )
    restored = pickle.loads(pickle.dumps(logical))
    target = vane.connect()
    try:
        physical = restored.to_physical_plan(target)
        nodes = physical.collect_udf_nodes()
    finally:
        target.close()

    assert len(nodes) == 1
    return nodes[0]["payload"]


def _attach_binary_batch(conn, alias, fn):
    import pyarrow as pa

    def batch(table):
        left = table.column("left").to_pylist()
        right = table.column("right").to_pylist()
        return pa.table({"result": [fn(a, b) for a, b in zip(left, right, strict=True)]})

    vane.attach_function(
        batch,
        connection=conn,
        alias=alias,
        input_names=["left", "right"],
        schema={"result": "INTEGER"},
        parameters=["INTEGER", "INTEGER"],
    )


def test_vane_function_registered_for_sql_projection():
    conn = vane.connect()

    @vane.func(return_dtype="INTEGER")
    def add_one(value):
        return value + 1

    vane.attach_function(
        add_one,
        connection=conn,
        alias="add_one_sql",
        parameters=["INTEGER"],
    )

    rows = conn.sql("""
        SELECT add_one_sql(i::INTEGER) AS y
        FROM range(3) t(i)
        ORDER BY i
    """).fetchall()

    assert rows == [(1,), (2,), (3,)]


def test_vane_function_sql_can_mix_with_normal_projection_expressions():
    conn = vane.connect()

    @vane.func(return_dtype="INTEGER")
    def add_one(value):
        return value + 1

    vane.attach_function(
        add_one,
        connection=conn,
        alias="add_one_sql",
        parameters=["INTEGER"],
    )

    rows = conn.sql("""
        SELECT
            i::INTEGER AS x,
            add_one_sql(i::INTEGER) AS y,
            add_one_sql(i::INTEGER) + 10 AS z
        FROM range(2) t(i)
        ORDER BY i
    """).fetchall()

    assert rows == [(0, 1, 11), (1, 2, 12)]


def test_vane_function_sql_rejects_missing_return_dtype():
    conn = vane.connect()

    @vane.func
    def add_one(value):
        return value + 1

    with pytest.raises(vane.InvalidInputException, match="return_dtype is required"):
        vane.attach_function(
            add_one,
            connection=conn,
            alias="bad_add_one_sql",
            parameters=["INTEGER"],
        )


def test_vane_function_sql_removal_uses_existing_remove_function():
    conn = vane.connect()

    @vane.func(return_dtype="INTEGER")
    def add_one(value):
        return value + 1

    vane.attach_function(
        add_one,
        connection=conn,
        alias="add_one_sql",
        parameters=["INTEGER"],
    )
    conn.remove_function("add_one_sql")

    with pytest.raises(Exception, match="add_one_sql"):
        conn.sql("SELECT add_one_sql(1)").fetchall()


def test_attach_function_replace_swaps_implementation():
    conn = vane.connect()

    @vane.func(return_dtype="INTEGER")
    def add_one(value):
        return value + 1

    @vane.func(return_dtype="INTEGER")
    def add_two(value):
        return value + 2

    vane.attach_function(add_one, connection=conn, alias="replaceable_sql", parameters=["INTEGER"])
    vane.attach_function(add_two, connection=conn, alias="replaceable_sql", parameters=["INTEGER"], replace=True)

    assert conn.sql("SELECT replaceable_sql(1)").fetchall() == [(3,)]
    assert conn.sql("SELECT count(*) FROM duckdb_functions() WHERE function_name = 'replaceable_sql'").fetchone() == (
        1,
    )


def test_attach_function_replace_validation_failure_preserves_old_alias():
    conn = vane.connect()

    @vane.func(return_dtype="INTEGER")
    def add_one(value):
        return value + 1

    @vane.func(return_dtype="NOT_A_DUCKDB_TYPE")
    def invalid_replacement(value):
        return value + 100

    vane.attach_function(add_one, connection=conn, alias="rollback_sql", parameters=["INTEGER"])

    with pytest.raises(vane.CatalogException):
        vane.attach_function(
            invalid_replacement,
            connection=conn,
            alias="rollback_sql",
            parameters=["INTEGER"],
            replace=True,
        )

    assert conn.sql("SELECT rollback_sql(1)").fetchall() == [(2,)]


def test_attach_function_replace_pickle_failure_preserves_old_alias():
    conn = vane.connect()

    @vane.func(return_dtype="INTEGER")
    def add_one(value):
        return value + 1

    class UnpicklableCapture:
        def __reduce__(self):
            raise pickle.PicklingError("intentional replacement pickle failure")

    captured = UnpicklableCapture()

    @vane.func(return_dtype="INTEGER")
    def invalid_replacement(value):
        return value + (100 if captured else 0)

    vane.attach_function(add_one, connection=conn, alias="pickle_rollback_sql", parameters=["INTEGER"])

    with pytest.raises(pickle.PicklingError, match=r"pickle|Pickling|serialize"):
        vane.attach_function(
            invalid_replacement,
            connection=conn,
            alias="pickle_rollback_sql",
            parameters=["INTEGER"],
            replace=True,
        )

    assert conn.sql("SELECT pickle_rollback_sql(1)").fetchall() == [(2,)]


def test_batch_replace_parameter_name_mismatch_preserves_old_alias():
    import pyarrow as pa

    conn = vane.connect()

    def old_batch(table):
        values = table.column("value").to_pylist()
        return pa.table({"result": [value + 1 for value in values]})

    vane.attach_function(
        old_batch,
        connection=conn,
        alias="batch_parameter_rollback_sql",
        input_names=["value"],
        schema={"result": "INTEGER"},
        parameters=["INTEGER"],
    )

    with pytest.raises(vane.InvalidInputException, match="input_names count"):
        vane.attach_function(
            old_batch,
            connection=conn,
            alias="batch_parameter_rollback_sql",
            input_names=["value", "extra"],
            schema={"result": "INTEGER"},
            parameters=["INTEGER"],
            replace=True,
        )

    assert conn.sql("SELECT batch_parameter_rollback_sql(1::INTEGER)").fetchall() == [(2,)]


def test_batch_replace_multi_output_schema_preserves_old_alias():
    import pyarrow as pa

    conn = vane.connect()

    def old_batch(table):
        values = table.column("value").to_pylist()
        return pa.table({"result": [value + 1 for value in values]})

    vane.attach_function(
        old_batch,
        connection=conn,
        alias="batch_schema_rollback_sql",
        input_names=["value"],
        schema={"result": "INTEGER"},
        parameters=["INTEGER"],
    )

    with pytest.raises(vane.InvalidInputException, match="exactly one output column"):
        vane.attach_function(
            old_batch,
            connection=conn,
            alias="batch_schema_rollback_sql",
            input_names=["value"],
            schema={"result": "INTEGER", "extra": "INTEGER"},
            parameters=["INTEGER"],
            replace=True,
        )

    assert conn.sql("SELECT batch_schema_rollback_sql(1::INTEGER)").fetchall() == [(2,)]


def test_replace_different_catalog_signature_preserves_old_alias():
    conn = vane.connect()

    @vane.func(return_dtype="INTEGER")
    def add_one(value):
        return value + 1

    @vane.func(return_dtype="INTEGER")
    def add_ten(value):
        return value + 10

    vane.attach_function(
        add_one,
        connection=conn,
        alias="signature_rollback_sql",
        parameters=["INTEGER"],
    )

    with pytest.raises(vane.InvalidInputException, match="different SQL signature"):
        vane.attach_function(
            add_ten,
            connection=conn,
            alias="signature_rollback_sql",
            parameters=["BIGINT"],
            replace=True,
        )

    assert conn.sql("SELECT signature_rollback_sql(1::INTEGER)").fetchall() == [(2,)]


def test_attach_function_replace_rejects_builtin_and_preserves_it():
    conn = vane.connect()

    @vane.func(return_dtype="DOUBLE")
    def fake_sqrt(value):
        return -1.0

    with pytest.raises(vane.InvalidInputException, match=r"owned|builtin|registered"):
        vane.attach_function(
            fake_sqrt,
            connection=conn,
            alias="sqrt",
            parameters=["DOUBLE"],
            replace=True,
        )

    assert conn.sql("SELECT sqrt(9::DOUBLE)").fetchall() == [(3.0,)]


def test_detach_scalar_alias_preserves_same_name_builtin_table_function():
    conn = vane.connect()
    table_function_count = conn.execute(
        """
        SELECT count(*)
        FROM duckdb_functions()
        WHERE function_name = 'read_csv' AND function_type = 'table'
        """
    ).fetchone()[0]
    assert table_function_count > 0

    @vane.func(return_dtype="INTEGER")
    def add_one(value):
        return value + 1

    vane.attach_function(add_one, connection=conn, alias="read_csv", parameters=["INTEGER"])
    assert conn.sql("SELECT read_csv(1)").fetchall() == [(2,)]

    vane.detach_function("read_csv", connection=conn)

    assert conn.execute(
        """
        SELECT count(*)
        FROM duckdb_functions()
        WHERE function_name = 'read_csv' AND function_type = 'table'
        """
    ).fetchone() == (table_function_count,)
    assert conn.execute(
        """
        SELECT count(*)
        FROM duckdb_functions()
        WHERE function_name = 'read_csv' AND function_type = 'scalar'
        """
    ).fetchone() == (0,)


def test_attached_aliases_with_same_name_are_connection_scoped():
    left_conn = vane.connect()
    right_conn = vane.connect()

    @vane.func(return_dtype="INTEGER")
    def add_one(value):
        return value + 1

    @vane.func(return_dtype="INTEGER")
    def add_ten(value):
        return value + 10

    vane.attach_function(add_one, connection=left_conn, alias="connection_local_sql", parameters=["INTEGER"])
    vane.attach_function(add_ten, connection=right_conn, alias="connection_local_sql", parameters=["INTEGER"])

    assert left_conn.sql("SELECT connection_local_sql(1)").fetchall() == [(2,)]
    assert right_conn.sql("SELECT connection_local_sql(1)").fetchall() == [(11,)]


def test_replace_rejects_vane_alias_owned_by_another_cursor():
    owner = vane.connect()
    other = owner.cursor()

    @vane.func(return_dtype="INTEGER")
    def add_one(value):
        return value + 1

    @vane.func(return_dtype="INTEGER")
    def add_ten(value):
        return value + 10

    vane.attach_function(add_one, connection=owner, alias="cursor_owned_sql", parameters=["INTEGER"])

    with pytest.raises(vane.InvalidInputException, match="not a Vane alias owned by this connection"):
        vane.attach_function(
            add_ten,
            connection=other,
            alias="cursor_owned_sql",
            parameters=["INTEGER"],
            replace=True,
        )

    assert owner.sql("SELECT cursor_owned_sql(1)").fetchall() == [(2,)]


def test_replace_rejects_non_vane_python_udf_owned_by_same_connection():
    conn = vane.connect()
    conn.create_function("ordinary_python_sql", lambda value: value + 1, ["INTEGER"], "INTEGER")

    @vane.func(return_dtype="INTEGER")
    def add_ten(value):
        return value + 10

    with pytest.raises(vane.InvalidInputException, match="not a registered Vane SQL alias"):
        vane.attach_function(
            add_ten,
            connection=conn,
            alias="ordinary_python_sql",
            parameters=["INTEGER"],
            replace=True,
        )

    assert conn.sql("SELECT ordinary_python_sql(1)").fetchall() == [(2,)]


def test_attach_function_replace_tolerates_missing_alias():
    conn = vane.connect()

    @vane.func(return_dtype="INTEGER")
    def add_one(value):
        return value + 1

    vane.attach_function(add_one, connection=conn, alias="never_attached_sql", parameters=["INTEGER"], replace=True)

    assert conn.sql("SELECT never_attached_sql(1)").fetchall() == [(2,)]


def test_attach_function_replace_delegates_atomic_registration_errors():
    class FailingAtomicConnection:
        def remove_function(self, alias):
            raise AssertionError(f"replace must not remove {alias} before registration")

        def _create_vane_function(self, *args, **kwargs):
            assert kwargs["replace"] is True
            raise RuntimeError("catalog locked during atomic registration")

    @vane.func(return_dtype="INTEGER")
    def add_one(value):
        return value + 1

    with pytest.raises(RuntimeError, match="catalog locked during atomic registration"):
        vane.attach_function(
            add_one,
            connection=FailingAtomicConnection(),
            alias="bad_replace_sql",
            parameters=["INTEGER"],
            replace=True,
        )


def test_vane_batch_function_registered_for_sql_projection():
    import pyarrow as pa

    conn = vane.connect()

    def add_one_batch(table: pa.Table) -> pa.Table:
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    vane.attach_function(
        add_one_batch,
        connection=conn,
        alias="batch_add_one_sql",
        input_names=["x"],
        schema={"y": "INTEGER"},
        parameters=["INTEGER"],
        batch_size=2,
    )

    rows = conn.sql("""
        SELECT batch_add_one_sql(i::INTEGER) AS y
        FROM range(5) t(i)
        ORDER BY i
    """).fetchall()

    assert rows == [(1,), (2,), (3,), (4,), (5,)]


def test_vane_batch_function_sql_without_actor_number_keeps_task_backend():
    import pyarrow as pa

    conn = vane.connect()

    def add_one_batch(table: pa.Table) -> pa.Table:
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    vane.attach_function(
        add_one_batch,
        connection=conn,
        alias="batch_add_one_task_sql",
        input_names=["x"],
        schema={"y": "INTEGER"},
        parameters=["INTEGER"],
    )

    rows = conn.sql("SELECT batch_add_one_task_sql(1::INTEGER)").fetchall()
    plan = conn.sql("EXPLAIN SELECT batch_add_one_task_sql(1::INTEGER)").fetchall()
    text = "\n".join(str(row) for row in plan)

    assert rows == [(2,)]
    assert "subprocess_task" in text
    assert "subprocess_actor" not in text
    assert "ray_actor" not in text


def test_vane_batch_function_sql_uses_declared_input_names():
    import pyarrow as pa

    conn = vane.connect()

    def combine_batch(table: pa.Table) -> pa.Table:
        left = table.column("left_value").to_pylist()
        right = table.column("right_value").to_pylist()
        return pa.table({"sum_value": [a + b for a, b in zip(left, right, strict=True)]})

    vane.attach_function(
        combine_batch,
        connection=conn,
        alias="batch_sum_sql",
        input_names=["left_value", "right_value"],
        schema={"sum_value": "INTEGER"},
        parameters=["INTEGER", "INTEGER"],
        batch_size=2,
    )

    rows = conn.sql("""
        SELECT batch_sum_sql(i::INTEGER, (i * 10)::INTEGER) AS y
        FROM range(3) t(i)
        ORDER BY i
    """).fetchall()

    assert rows == [(0,), (11,), (22,)]


def test_vane_batch_sql_reorders_reversed_named_arguments():
    conn = vane.connect()
    _attach_binary_batch(conn, "subtract_named_sql", lambda left, right: left - right)

    assert conn.sql("SELECT subtract_named_sql(right := 2, left := 10)").fetchall() == [(8,)]


def test_vane_batch_sql_supports_positional_prefix_and_named_suffix():
    conn = vane.connect()
    _attach_binary_batch(conn, "subtract_mixed_sql", lambda left, right: left - right)

    assert conn.sql("SELECT subtract_mixed_sql(10, right := 2)").fetchall() == [(8,)]


def test_vane_batch_sql_reorders_named_arguments_before_type_binding():
    import pyarrow as pa

    conn = vane.connect()

    def repeat_batch(table):
        text = table.column("text").to_pylist()
        repeat = table.column("repeat").to_pylist()
        return pa.table({"result": [value * count for value, count in zip(text, repeat, strict=True)]})

    vane.attach_function(
        repeat_batch,
        connection=conn,
        alias="repeat_named_sql",
        input_names=["text", "repeat"],
        schema={"result": "VARCHAR"},
        parameters=["VARCHAR", "INTEGER"],
    )

    assert conn.sql("SELECT repeat_named_sql(repeat := 2, text := 'x')").fetchall() == [("xx",)]


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ("unknown := 2, left := 10", "unknown"),
        ("left := 10, left := 2", "duplicate|left"),
        ("10, left := 2", "duplicate|left|positional"),
        ("left := 10", "missing|right"),
    ],
)
def test_vane_batch_sql_rejects_invalid_named_arguments(arguments, error):
    conn = vane.connect()
    _attach_binary_batch(conn, "invalid_named_sql", lambda left, right: left - right)

    with pytest.raises(Exception, match=error):
        conn.sql(f"SELECT invalid_named_sql({arguments})").fetchall()


def test_vane_scalar_sql_rejects_named_arguments_without_declared_names():
    conn = vane.connect()

    @vane.func(return_dtype="INTEGER")
    def subtract(left, right):
        return left - right

    vane.attach_function(
        subtract,
        connection=conn,
        alias="scalar_named_sql",
        parameters=["INTEGER", "INTEGER"],
    )

    with pytest.raises(vane.BinderException, match=r"named arguments.*not supported|does not accept named"):
        conn.sql("SELECT scalar_named_sql(right := 2, left := 10)").fetchall()


def test_vane_batch_function_sql_rejects_input_name_count_mismatch():
    import pyarrow as pa

    conn = vane.connect()

    def identity_batch(table: pa.Table) -> pa.Table:
        return pa.table({"y": table.column("x")})

    with pytest.raises(vane.InvalidInputException, match="input_names count must match"):
        vane.attach_function(
            identity_batch,
            connection=conn,
            alias="bad_batch_sql",
            input_names=["x", "extra"],
            schema={"y": "INTEGER"},
            parameters=["INTEGER"],
        )


def test_vane_batch_function_sql_rejects_multi_output_schema_in_v1():
    import pyarrow as pa

    conn = vane.connect()

    def two_outputs(table: pa.Table) -> pa.Table:
        values = table.column("x").to_pylist()
        return pa.table({"a": values, "b": values})

    with pytest.raises(vane.InvalidInputException, match="exactly one output column"):
        vane.attach_function(
            two_outputs,
            connection=conn,
            alias="bad_multi_output_sql",
            input_names=["x"],
            schema={"a": "INTEGER", "b": "INTEGER"},
            parameters=["INTEGER"],
        )


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("batch_size", 32),
        ("gpus", 0.5),
        ("actor_number", 1),
    ],
)
def test_vane_function_attach_rejects_batch_only_options(option, value):
    conn = vane.connect()

    @vane.func(return_dtype="INTEGER")
    def add_one(value):
        return value + 1

    alias = f"decorated_scalar_rejects_{option}"
    with pytest.raises(
        vane.InvalidInputException,
        match=rf"{option}.*SQL vane\.func|SQL vane\.func.*{option}",
    ):
        vane.attach_function(
            add_one,
            connection=conn,
            alias=alias,
            parameters=["INTEGER"],
            **{option: value},
        )

    _assert_sql_alias_absent(conn, alias)


def test_vane_function_cannot_be_reclassified_as_batch():
    conn = vane.connect()

    @vane.func(return_dtype="INTEGER")
    def add_one(value):
        return value + 1

    alias = "decorated_scalar_is_not_batch"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"input_names|schema.*SQL vane\.func|SQL vane\.func.*batch",
    ):
        vane.attach_function(
            add_one,
            connection=conn,
            alias=alias,
            parameters=["INTEGER"],
            input_names=["value"],
            schema={"result": "INTEGER"},
        )

    _assert_sql_alias_absent(conn, alias)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("batch_size", 32),
        ("gpus", 0.5),
        ("actor_number", 1),
    ],
)
def test_raw_scalar_attach_rejects_batch_size_gpus_and_actor_number(option, value):
    conn = vane.connect()

    def add_one(value):
        return value + 1

    alias = f"raw_scalar_rejects_{option}"
    with pytest.raises(
        vane.InvalidInputException,
        match=rf"{option}.*raw scalar|raw scalar.*{option}",
    ):
        vane.attach_function(
            add_one,
            connection=conn,
            alias=alias,
            parameters=["INTEGER"],
            return_dtype="INTEGER",
            **{option: value},
        )

    _assert_sql_alias_absent(conn, alias)


@pytest.mark.parametrize(
    ("metadata", "missing"),
    [
        ({"input_names": ["value"]}, "schema"),
        ({"schema": {"result": "INTEGER"}}, "input_names"),
    ],
)
def test_raw_batch_attach_requires_input_names_and_schema_together(metadata, missing):
    conn = vane.connect()

    def identity_batch(table):
        return table

    alias = f"incomplete_raw_batch_{missing}"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"input_names.*schema.*together|schema.*input_names.*together",
    ):
        vane.attach_function(
            identity_batch,
            connection=conn,
            alias=alias,
            parameters=["INTEGER"],
            **metadata,
        )

    _assert_sql_alias_absent(conn, alias)


def test_raw_batch_attach_requires_parameters():
    conn = vane.connect()

    def identity_batch(table):
        return table

    alias = "raw_batch_requires_parameters"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"parameters is required.*raw batch|raw batch.*parameters is required",
    ):
        vane.attach_function(
            identity_batch,
            connection=conn,
            alias=alias,
            input_names=["value"],
            schema={"result": "INTEGER"},
        )

    _assert_sql_alias_absent(conn, alias)


def test_raw_batch_attach_rejects_return_dtype():
    conn = vane.connect()

    def identity_batch(table):
        return table

    alias = "raw_batch_rejects_return_dtype"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"return_dtype.*raw batch|raw batch.*return_dtype",
    ):
        vane.attach_function(
            identity_batch,
            connection=conn,
            alias=alias,
            parameters=["INTEGER"],
            input_names=["value"],
            schema={"result": "INTEGER"},
            return_dtype="INTEGER",
        )

    _assert_sql_alias_absent(conn, alias)


def test_raw_batch_actor_number_requires_zero_argument_class():
    conn = vane.connect()

    def identity_batch(table):
        return table

    alias = "raw_actor_requires_class"
    with pytest.raises(vane.InvalidInputException, match=r"actor.*callable class"):
        vane.attach_function(
            identity_batch,
            connection=conn,
            alias=alias,
            parameters=["INTEGER"],
            input_names=["value"],
            schema={"result": "INTEGER"},
            actor_number=1,
        )

    _assert_sql_alias_absent(conn, alias)


def test_vane_cls_sql_batch_size_override_reaches_fresh_physical_payload_and_executes():
    conn = vane.connect()

    @vane.cls(
        actor_number=1,
        return_dtype="INTEGER",
        name="row_override_result",
    )
    class AddOne:
        def __call__(self, value):
            return value + 1

    vane.attach_function(
        AddOne(),
        connection=conn,
        alias="row_class_batch_size_override_sql",
        parameters=["INTEGER"],
        batch_size=2,
    )
    relation = conn.sql("SELECT row_class_batch_size_override_sql(i::INTEGER) AS result FROM range(5) t(i) ORDER BY i")

    payload = _fresh_physical_udf_payload(relation)
    assert payload["batch_size"] == 2
    assert payload["output_schema"] == [
        {
            "name": "row_override_result",
            "kind": "duckdb_type",
            "type": "INTEGER",
            "dtype": None,
            "shape": None,
        }
    ]
    assert relation.fetchall() == [(1,), (2,), (3,), (4,), (5,)]


def test_vane_cls_batch_sql_schema_and_batch_size_overrides_reach_payload_and_execute():
    import pyarrow as pa

    conn = vane.connect()

    @vane.cls.batch(
        actor_number=1,
        batch_size=17,
        schema={"decorated_result": "INTEGER"},
        name="decorated_batch_result",
        row_preserving=True,
    )
    class AddTwoBatch:
        def __call__(self, table):
            values = table.column("value").to_pylist()
            return pa.table(
                {
                    "override_result": pa.array(
                        [value + 2 for value in values],
                        type=pa.int64(),
                    )
                }
            )

    vane.attach_function(
        AddTwoBatch(),
        connection=conn,
        alias="batch_class_metadata_override_sql",
        input_names=["value"],
        parameters=["INTEGER"],
        schema={"override_result": "BIGINT"},
        batch_size=3,
    )
    relation = conn.sql("SELECT batch_class_metadata_override_sql(i::INTEGER) AS result FROM range(5) t(i) ORDER BY i")

    payload = _fresh_physical_udf_payload(relation)
    assert payload["batch_size"] == 3
    assert payload["output_schema"] == [
        {
            "name": "override_result",
            "kind": "duckdb_type",
            "type": "BIGINT",
            "dtype": None,
            "shape": None,
        }
    ]
    assert relation.fetchall() == [(2,), (3,), (4,), (5,), (6,)]


def test_raw_batch_actor_requires_callable_instances():
    conn = vane.connect()

    class NotCallableBatch:
        pass

    alias = "raw_actor_requires_callable_instances"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"callable class.*instances implement __call__",
    ):
        vane.attach_function(
            NotCallableBatch,
            connection=conn,
            alias=alias,
            parameters=["INTEGER"],
            input_names=["value"],
            schema={"result": "INTEGER"},
            actor_number=1,
        )

    _assert_sql_alias_absent(conn, alias)


def test_raw_batch_actor_requires_a_concrete_callable_class():
    from abc import ABC, abstractmethod

    conn = vane.connect()

    class AbstractBatch(ABC):
        @abstractmethod
        def __call__(self, table):
            raise NotImplementedError

    alias = "raw_actor_requires_concrete_class"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"concrete callable class",
    ):
        vane.attach_function(
            AbstractBatch,
            connection=conn,
            alias=alias,
            parameters=["INTEGER"],
            input_names=["value"],
            schema={"result": "INTEGER"},
            actor_number=1,
        )

    _assert_sql_alias_absent(conn, alias)


def test_raw_batch_actor_rejects_uninspectable_constructor_before_registration():
    conn = vane.connect()

    class UninspectableBatch:
        __signature__ = object()

        def __call__(self, table):
            return table

    alias = "raw_actor_uninspectable_constructor"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"constructor signature cannot be inspected.*vane\.cls",
    ):
        vane.attach_function(
            UninspectableBatch,
            connection=conn,
            alias=alias,
            parameters=["INTEGER"],
            input_names=["value"],
            schema={"result": "INTEGER"},
            actor_number=1,
        )

    _assert_sql_alias_absent(conn, alias)


@pytest.mark.parametrize("actor_number", [False, True])
def test_raw_batch_actor_number_rejects_bool(actor_number):
    conn = vane.connect()

    class IdentityBatch:
        def __call__(self, table):
            return table

    alias = f"raw_actor_rejects_bool_{actor_number}"
    with pytest.raises(
        vane.InvalidInputException,
        match=r"actor_number.*bool|actor_number.*positive integer",
    ):
        vane.attach_function(
            IdentityBatch,
            connection=conn,
            alias=alias,
            parameters=["INTEGER"],
            input_names=["value"],
            schema={"result": "INTEGER"},
            actor_number=actor_number,
        )

    _assert_sql_alias_absent(conn, alias)


def test_generic_attach_preflight_failure_with_replace_preserves_old_alias():
    conn = vane.connect()

    @vane.func(return_dtype="INTEGER")
    def add_one(value):
        return value + 1

    @vane.func(return_dtype="INTEGER")
    def add_hundred(value):
        return value + 100

    alias = "generic_preflight_rollback_sql"
    vane.attach_function(add_one, connection=conn, alias=alias, parameters=["INTEGER"])
    before = conn.execute(
        "SELECT return_type, parameter_types, function_oid FROM duckdb_functions() WHERE function_name = ?",
        [alias],
    ).fetchall()

    with pytest.raises(vane.InvalidInputException, match=r"batch_size.*SQL vane\.func"):
        vane.attach_function(
            add_hundred,
            connection=conn,
            alias=alias,
            parameters=["INTEGER"],
            batch_size=8,
            replace=True,
        )

    after = conn.execute(
        "SELECT return_type, parameter_types, function_oid FROM duckdb_functions() WHERE function_name = ?",
        [alias],
    ).fetchall()
    assert after == before
    assert conn.sql(f"SELECT {alias}(1::INTEGER), typeof({alias}(1::INTEGER))").fetchall() == [(2, "INTEGER")]


def test_vane_function_return_dtype_pyarrow_int64_expression_and_sql():
    import pyarrow as pa

    conn = vane.connect()

    @vane.func(return_dtype=pa.int64())
    def widen(value):
        return value + 2**40

    expression_result = conn.sql("SELECT 1::INTEGER AS value").select(widen(vane.col("value")).alias("result"))
    assert [str(dtype) for dtype in expression_result.types] == ["BIGINT"]
    assert expression_result.fetchall() == [(2**40 + 1,)]

    vane.attach_function(widen, connection=conn, alias="widen_pyarrow_sql", parameters=["INTEGER"])
    assert conn.sql("SELECT widen_pyarrow_sql(1::INTEGER), typeof(widen_pyarrow_sql(1::INTEGER))").fetchall() == [
        (2**40 + 1, "BIGINT")
    ]
