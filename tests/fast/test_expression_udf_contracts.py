# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Cross-cutting expression UDF capability-boundary tests."""

from __future__ import annotations

import re

import pytest


@pytest.fixture
def sql_udf_contract_connection():
    import pyarrow as pa

    import vane

    con = vane.connect()

    @vane.func(return_dtype="INTEGER")
    def scalar_contract(value):
        return value + 1

    def batch_contract(table):
        values = table.column("value").to_pylist()
        return pa.table({"result": [value + 1 for value in values]})

    @vane.cls(actor_number=2, return_dtype="INTEGER", name="actor_contract")
    class ActorContract:
        def __call__(self, value):
            return value

    vane.attach_function(
        scalar_contract,
        alias="scalar_contract",
        connection=con,
        parameters=["INTEGER"],
    )
    vane.attach_function(
        batch_contract,
        alias="batch_contract",
        connection=con,
        parameters=["INTEGER"],
        input_names=["value"],
        schema={"result": "INTEGER"},
    )
    vane.attach_function(
        ActorContract(),
        alias="actor_contract",
        connection=con,
        parameters=["INTEGER"],
    )
    try:
        yield con
    finally:
        con.close()


@pytest.fixture
def recorded_local_actor_pools(monkeypatch):
    from vane.execution import udf_subprocess

    original = udf_subprocess.LocalSubprocessActorPool
    pools = []

    def create_pool(*args, **kwargs):
        pool = original(*args, **kwargs)
        pools.append(pool)
        return pool

    monkeypatch.setattr(udf_subprocess, "LocalSubprocessActorPool", create_pool)
    try:
        yield pools
    finally:
        for pool in pools:
            pool.shutdown(kill=True)


def _assert_local_actor_pools_released(pools):
    for pool in pools:
        assert pool.worker_pids() == []
        assert not pool.cleanup_pending()


@pytest.mark.parametrize("parameterized", [False, True])
def test_prepared_actor_udf_recreates_query_scoped_resources(
    sql_udf_contract_connection, recorded_local_actor_pools, parameterized
):
    con = sql_udf_contract_connection
    pools = recorded_local_actor_pools
    argument = "$1::INTEGER" if parameterized else "42"
    con.execute(f"PREPARE actor_query AS SELECT actor_contract({argument})")
    assert pools == []
    for execution_count, value in enumerate((42, 73), start=1):
        execute = f"EXECUTE actor_query({value})" if parameterized else "EXECUTE actor_query"
        expected = value if parameterized else 42
        assert con.execute(execute).fetchall() == [(expected,)]
        assert len(pools) == execution_count
        _assert_local_actor_pools_released(pools)


def test_executemany_actor_udf_recreates_query_scoped_resources(
    sql_udf_contract_connection, recorded_local_actor_pools
):
    con = sql_udf_contract_connection
    con.executemany("SELECT actor_contract(?::INTEGER)", [[42], [73]])
    assert con.fetchall() == [(73,)]
    assert len(recorded_local_actor_pools) == 2
    _assert_local_actor_pools_released(recorded_local_actor_pools)


def test_prepared_actor_udf_releases_failed_query_and_can_execute_again(
    sql_udf_contract_connection, recorded_local_actor_pools
):
    import vane

    @vane.cls(actor_number=1, return_dtype="INTEGER", name="recoverable_actor_contract")
    class RecoverableActorContract:
        def __call__(self, value):
            if value < 0:
                raise ValueError("actor contract rejects negative values")
            return value

    con = sql_udf_contract_connection
    vane.attach_function(
        RecoverableActorContract(),
        alias="recoverable_actor_contract",
        connection=con,
        parameters=["INTEGER"],
    )
    con.execute("PREPARE actor_query AS SELECT recoverable_actor_contract($1::INTEGER)")
    assert recorded_local_actor_pools == []
    for execution_count, value in enumerate((42, -1, 73), start=1):
        if value < 0:
            with pytest.raises(Exception, match="actor contract rejects negative values"):
                con.execute(f"EXECUTE actor_query({value})").fetchall()
        else:
            assert con.execute(f"EXECUTE actor_query({value})").fetchall() == [(value,)]
        assert len(recorded_local_actor_pools) == execution_count
        _assert_local_actor_pools_released(recorded_local_actor_pools)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT i FROM range(3) t(i) WHERE scalar_contract(i::INTEGER) > 0",
        "SELECT sum(batch_contract(i::INTEGER)) FROM range(3) t(i)",
        "SELECT * FROM range(3) a(i) JOIN range(3) b(j) ON actor_contract(a.i::INTEGER) = b.j",
        "SELECT ai_prompt(text), count(*) FROM (VALUES ('x')) t(text) GROUP BY ai_prompt(text)",
        "SELECT count(*) FROM (VALUES (1), (2)) t(i) HAVING scalar_contract(count(*)::INTEGER) > 0",
    ],
    ids=["where-scalar", "aggregate-batch", "join-actor-class", "group-by-ai", "having-scalar"],
)
def test_expression_udfs_reject_non_projection_positions(sql_udf_contract_connection, sql):
    message = "udf can only be used in a projection and must be planned as a UDF operator"
    with pytest.raises(Exception, match=re.escape(message)):
        sql_udf_contract_connection.execute(sql)


@pytest.mark.parametrize(
    ("dtype", "arrow_type"),
    [
        ("BOOLEAN", "bool"),
        ("TINYINT", "int8"),
        ("UTINYINT", "uint8"),
        ("SMALLINT", "int16"),
        ("USMALLINT", "uint16"),
        ("INTEGER", "int32"),
        ("UINTEGER", "uint32"),
        ("BIGINT", "int64"),
        ("UBIGINT", "uint64"),
        ("FLOAT", "float"),
        ("DOUBLE", "double"),
        ("VARCHAR", "string"),
        ("BLOB", "binary"),
        ("DATE", "date32[day]"),
        ("TIMESTAMP", "timestamp[us]"),
        ("TIMESTAMP_NS", "timestamp[ns]"),
        ("TIMESTAMP_MS", "timestamp[ms]"),
        ("TIMESTAMP_S", "timestamp[s]"),
        ("DECIMAL(10,2)", "decimal128(10, 2)"),
        ("FLOAT[]", "list<item: float>"),
        ("FLOAT[4]", "fixed_size_list<item: float>[4]"),
        ("STRUCT(value INTEGER)", "struct<value: int32>"),
    ],
)
def test_vane_cls_arrow_return_type_support_matrix(dtype, arrow_type):
    from vane._expression_udf import _dtype_to_arrow

    assert str(_dtype_to_arrow(dtype)) == arrow_type


@pytest.mark.parametrize(
    "dtype",
    [
        "TIME",
        "INTERVAL",
        "UUID",
        "TIMESTAMPTZ",
        "ENUM('red', 'green')",
        "MAP(VARCHAR, INTEGER)",
    ],
)
def test_vane_cls_rejects_unsupported_arrow_return_types_with_original_type(dtype):
    import vane
    from vane._expression_udf import _dtype_to_arrow

    with pytest.raises(vane.InvalidInputException, match=re.escape(dtype)):
        _dtype_to_arrow(dtype)


@pytest.mark.parametrize(
    ("arrow_type", "duckdb_type"),
    [
        pytest.param("bool", "BOOLEAN", id="bool"),
        pytest.param("int8", "TINYINT", id="int8"),
        pytest.param("uint8", "UTINYINT", id="uint8"),
        pytest.param("int16", "SMALLINT", id="int16"),
        pytest.param("uint16", "USMALLINT", id="uint16"),
        pytest.param("int32", "INTEGER", id="int32"),
        pytest.param("uint32", "UINTEGER", id="uint32"),
        pytest.param("int64", "BIGINT", id="int64"),
        pytest.param("uint64", "UBIGINT", id="uint64"),
        pytest.param("float32", "FLOAT", id="float32"),
        pytest.param("float64", "DOUBLE", id="float64"),
        pytest.param("string", "VARCHAR", id="string"),
        pytest.param("binary", "BLOB", id="binary"),
        pytest.param("date32", "DATE", id="date32"),
        pytest.param("decimal", "DECIMAL(18,4)", id="decimal"),
        pytest.param("list", "BIGINT[]", id="list"),
        pytest.param("fixed_list", "BIGINT[3]", id="fixed-list"),
        pytest.param("nested_list", "INTEGER[][]", id="nested-list"),
        pytest.param("nested_fixed_list", "INTEGER[][2]", id="nested-fixed-list"),
        pytest.param("timestamp_us", "TIMESTAMP", id="timestamp-us"),
        pytest.param("timestamp_ns", "TIMESTAMP_NS", id="timestamp-ns"),
        pytest.param("timestamp_ms", "TIMESTAMP_MS", id="timestamp-ms"),
        pytest.param("timestamp_s", "TIMESTAMP_S", id="timestamp-s"),
        pytest.param(
            "struct",
            'STRUCT("value" BIGINT, nested STRUCT(score DOUBLE))',
            id="nested-struct",
        ),
    ],
)
def test_pyarrow_datatype_canonicalization_matrix(arrow_type, duckdb_type):
    import pyarrow as pa

    from vane._expression_udf import _canonicalize_dtype

    types = {
        "bool": pa.bool_(),
        "int8": pa.int8(),
        "uint8": pa.uint8(),
        "int16": pa.int16(),
        "uint16": pa.uint16(),
        "int32": pa.int32(),
        "uint32": pa.uint32(),
        "int64": pa.int64(),
        "uint64": pa.uint64(),
        "float32": pa.float32(),
        "float64": pa.float64(),
        "string": pa.string(),
        "binary": pa.binary(),
        "date32": pa.date32(),
        "decimal": pa.decimal128(18, 4),
        "list": pa.list_(pa.int64()),
        "fixed_list": pa.list_(pa.int64(), 3),
        "nested_list": pa.list_(pa.list_(pa.int32())),
        "nested_fixed_list": pa.list_(pa.list_(pa.int32()), 2),
        "timestamp_us": pa.timestamp("us"),
        "timestamp_ns": pa.timestamp("ns"),
        "timestamp_ms": pa.timestamp("ms"),
        "timestamp_s": pa.timestamp("s"),
        "struct": pa.struct(
            [
                pa.field("value", pa.int64()),
                pa.field("nested", pa.struct([pa.field("score", pa.float64())])),
            ]
        ),
    }
    expected_arrow = types[arrow_type]

    normalized_duckdb, normalized_arrow = _canonicalize_dtype(expected_arrow)

    assert str(normalized_duckdb) == duckdb_type
    assert normalized_arrow == expected_arrow


@pytest.mark.parametrize(
    "unsupported",
    [
        pytest.param("map", id="map"),
        pytest.param("duration", id="duration"),
        pytest.param("dictionary", id="dictionary"),
        pytest.param("timezone", id="timezone-aware-timestamp"),
    ],
)
def test_unsupported_pyarrow_datatype_matrix_is_rejected_during_canonicalization(unsupported):
    import pyarrow as pa

    import vane
    from vane._expression_udf import _canonicalize_dtype

    dtype = {
        "map": pa.map_(pa.string(), pa.int64()),
        "duration": pa.duration("us"),
        "dictionary": pa.dictionary(pa.int8(), pa.string()),
        "timezone": pa.timestamp("us", tz="UTC"),
    }[unsupported]

    with pytest.raises(vane.InvalidInputException) as exc_info:
        _canonicalize_dtype(dtype)

    assert str(dtype) in str(exc_info.value)
    assert "not supported" in str(exc_info.value) or "TIMESTAMPTZ" in str(exc_info.value)


@pytest.mark.parametrize("duplicate_name", ["value", "VALUE"])
def test_pyarrow_struct_duplicate_field_names_are_rejected_without_collapsing(duplicate_name):
    import pyarrow as pa

    import vane
    from vane._expression_udf import _canonicalize_dtype

    dtype = pa.struct([pa.field("value", pa.int32()), pa.field(duplicate_name, pa.string())])

    with pytest.raises(vane.InvalidInputException, match="Struct field names must be unique case-insensitively"):
        _canonicalize_dtype(dtype)


def test_actor_gpu_is_rejected_when_resolved_backend_is_local(monkeypatch):
    import pyarrow as pa

    import vane

    monkeypatch.setenv("VANE_RUNNER", "ray")

    @vane.cls.batch(
        actor_number=1,
        return_dtype=pa.int32(),
        gpus=0.75,
    )
    class IdentityBatch:
        def __call__(self, value):
            return value

    monkeypatch.setenv("VANE_RUNNER", "local")
    expression = IdentityBatch()(vane.col("value"))
    with vane.connect() as connection:
        relation = connection.sql("SELECT 1 AS value").select(expression)
        with pytest.raises(vane.InvalidInputException, match="GPU UDF execution requires a registered local GPU model"):
            relation.fetchall()


def test_direct_udf_operator_and_attached_expression_aliases_are_volatile():
    import vane

    @vane.func(return_dtype="INTEGER")
    def identity(value):
        return value

    @vane.cls(actor_number=2, return_dtype="INTEGER")
    class ActorIdentity:
        def __call__(self, value):
            return value

    con = vane.connect()
    try:
        vane.attach_function(
            identity,
            alias="volatile_scalar_alias",
            connection=con,
            parameters=["INTEGER"],
        )
        vane.attach_function(
            ActorIdentity(),
            alias="volatile_actor_alias",
            connection=con,
            parameters=["INTEGER"],
        )
        stability = dict(
            con.execute(
                """
                SELECT function_name, has_side_effects
                FROM duckdb_functions()
                WHERE function_type = 'scalar'
                  AND function_name IN ('udf', 'volatile_scalar_alias', 'volatile_actor_alias')
                """
            ).fetchall()
        )
    finally:
        con.close()

    assert stability == {
        "udf": True,
        "volatile_scalar_alias": True,
        "volatile_actor_alias": True,
    }
