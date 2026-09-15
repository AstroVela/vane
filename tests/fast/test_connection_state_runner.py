# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Client metadata is routed by bound source identity, including composed queries."""

from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import vane
from tests.fast.test_bound_plan_runner import install_runner
from tests.fast.test_distributed_result_consumers import _TransportedPlanRunner


def query(connection, entry, sql, params=None):
    if entry == "execute":
        return connection.execute(sql, params).fetchall()
    return connection.sql(sql, params=params).fetchall()


@pytest.fixture
def forbid_ray(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")

    def initialize(*_args, **_kwargs):
        raise AssertionError("client reads and admission errors must not initialize Ray")

    monkeypatch.setattr(vane._native, "set_runner_ray", initialize)


@pytest.mark.parametrize("entry", ["execute", "sql"])
@pytest.mark.parametrize("parameterized", [False, True])
def test_direct_state_reads_use_the_owning_connection(forbid_ray, entry, parameterized):
    with vane.connect() as connection:
        connection.execute("SET threads=3")
        connection.execute("SET VARIABLE state_value=7")
        key = "$key" if parameterized else "'threads'"
        params = {"key": "threads"} if parameterized else None
        sql = f"SELECT current_setting({key}), getvariable('state_value'), current_schema(), current_database(), 42"
        assert query(connection, entry, sql, params) == [(3, 7, "main", "memory", 42)]
        connection.execute("SET threads=5")
        assert query(connection, entry, sql, params)[0][0] == 5
        ids = query(connection, entry, "SELECT current_connection_id(), current_query_id(), current_transaction_id()")
        assert all(isinstance(value, int) for value in ids[0])
        sql = "SELECT current_query()"
        assert query(connection, entry, sql) == [(sql,)]
        connection.begin()
        timestamp = query(connection, entry, "SELECT now()")[0][0]
        assert query(connection, entry, "SELECT transaction_timestamp()")[0][0] == timestamp
        assert isinstance(query(connection, entry, "SELECT txid_current()")[0][0], int)
        connection.commit()


@pytest.mark.parametrize("value", [None, 8, "text", [1, 2], {"key": 3}])
def test_native_variable_binding_and_lazy_rebinding(forbid_ray, value):
    with vane.connect() as connection:
        connection.execute("SET VARIABLE state_value=$value", {"value": value})
        assert connection.sql("SELECT getvariable($key)", params={"key": "state_value"}).fetchall() == [(value,)]
        connection.execute("SET threads=2")
        relation = connection.sql("SELECT current_setting('threads')")
        connection.execute("SET threads=3")
        assert relation.fetchall() == [(3,)]
        connection.execute("SET threads=4")
        assert relation.fetchall() == [(4,)]


@pytest.mark.parametrize("entry", ["execute", "sql"])
@pytest.mark.parametrize("expression, value", [("-1", -1), ("+1", 1), ("-(1 + 1)", -2)])
def test_state_computations_use_native_binding(forbid_ray, entry, expression, value):
    with vane.connect() as connection:
        assert query(connection, entry, f"SELECT current_schema(), {expression}") == [("main", value)]


@pytest.mark.parametrize("entry", ["execute", "sql", "table_function"])
@pytest.mark.parametrize(
    "function, column, expected",
    [
        ("duckdb_tables", "table_name", "marker"),
        ("duckdb_views", "view_name", "marker_view"),
        ("duckdb_schemas", "schema_name", "main"),
        ("duckdb_databases", "database_name", "attached"),
        ("duckdb_settings", "name", "threads"),
        ("duckdb_variables", "name", "state_value"),
        ("duckdb_extensions", "extension_name", "parquet"),
        ("duckdb_sequences", "sequence_name", "marker_sequence"),
    ],
)
def test_direct_metadata_reads_use_client_catalog(forbid_ray, entry, function, column, expected):
    with vane.connect() as connection:
        connection.execute("CREATE TABLE marker(value INTEGER)")
        connection.execute("CREATE VIEW marker_view AS SELECT 1 AS value")
        connection.execute("CREATE SEQUENCE marker_sequence")
        connection.execute("SET VARIABLE state_value=42")
        connection.execute("ATTACH ':memory:' AS attached")
        connection.begin()
        if entry == "table_function":
            relation = connection.table_function(function)
            index = relation.columns.index(column)
            values = [row[index] for row in relation.fetchall()]
        else:
            values = [row[0] for row in query(connection, entry, f"SELECT m.{column} AS value FROM {function}() m")]
        assert expected in values
        # Bare star is also supported; expression-based COLUMNS/REPLACE is not.
        assert query(connection, "sql", f"SELECT m.* FROM {function}() m")
        connection.commit()


@pytest.mark.parametrize("entry", ["execute", "sql"])
@pytest.mark.parametrize("qualifier", ["system", "system.main"])
def test_catalog_qualified_reads_and_transaction_visibility(forbid_ray, entry, qualifier):
    with vane.connect() as connection:
        connection.execute("SET threads=3")
        connection.begin()
        connection.execute("CREATE TABLE transaction_marker(value INTEGER)")
        assert query(connection, entry, f"SELECT {qualifier}.current_setting($key)", {"key": "threads"}) == [(3,)]
        assert ("transaction_marker",) in query(
            connection, entry, f"SELECT table_name FROM {qualifier}.duckdb_tables()"
        )
        connection.rollback()
        assert ("transaction_marker",) not in query(connection, entry, "SELECT table_name FROM duckdb_tables()")
        connection.execute("CREATE SCHEMA system")
        with pytest.raises(vane.BinderException, match="Ambiguous reference to catalog or schema"):
            query(connection, entry, "SELECT system.current_setting('threads')")
        assert query(connection, entry, "SELECT system.main.current_setting('threads')") == [(3,)]


@pytest.mark.parametrize("entry", ["execute", "sql"])
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT current_setting(concat('th', 'reads'))",
        "SELECT current_setting('threads') FROM range(1)",
        "SELECT * FROM duckdb_tables(), range(1)",
        "SELECT * FROM query('SHOW TABLES')",
    ],
)
def test_mixed_or_unapproved_dependencies_are_not_native_fallbacks(forbid_ray, entry, sql):
    with vane.connect() as connection:
        with pytest.raises(vane.NotImplementedException, match="client-context|client connection|client metadata"):
            query(connection, entry, sql)


@pytest.mark.parametrize("entry", ["execute", "sql"])
@pytest.mark.parametrize(
    "source",
    [
        "duckdb_columns()",
        "pragma_table_info('marker_view')",
        "pragma_show('marker_view')",
        "duckdb_functions()",
        "duckdb_types()",
        "duckdb_memory()",
        "which_secret('https://example.invalid', 'http')",
    ],
)
def test_unlisted_metadata_is_rejected_before_view_rebinding(forbid_ray, entry, source):
    with vane.connect() as connection:
        connection.execute("CREATE SEQUENCE metadata_sequence")
        connection.execute("CREATE VIEW marker_view AS SELECT * FROM range(nextval('metadata_sequence'))")
        before = connection.execute("SELECT last_value FROM duckdb_sequences()").fetchall()
        with pytest.raises(vane.NotImplementedException, match="client-context table function"):
            query(connection, entry, f"SELECT * FROM {source}")
        assert connection.execute("SELECT last_value FROM duckdb_sequences()").fetchall() == before
        # Native DDL keeps DuckDB's view binding behavior without a metadata guard.
        connection.execute("COMMENT ON COLUMN marker_view.range IS 'native DDL'")


@pytest.mark.parametrize("entry", ["execute", "sql"])
def test_macro_and_view_references_preserve_source_identity(forbid_ray, entry):
    with vane.connect() as connection:
        connection.execute("CREATE SCHEMA custom")
        connection.execute("CREATE MACRO custom.current_setting(key) AS system.main.current_setting(key)")
        connection.execute("CREATE VIEW state_view AS SELECT current_schema() AS schema_name")
        connection.execute("SET threads=3")
        assert query(connection, entry, "SELECT custom.current_setting('threads')") == [(3,)]
        assert query(connection, entry, "SELECT * FROM state_view") == [("main",)]


@pytest.mark.parametrize("reader", ["current_setting", "getvariable"])
@pytest.mark.parametrize("entry", ["execute", "sql"])
def test_state_table_arguments_fail_before_opening_files(forbid_ray, tmp_path, entry, reader):
    with vane.connect() as connection:
        connection.execute("SET VARIABLE threads=3")
        sql = f"SELECT * FROM read_parquet(concat('{tmp_path}/', {reader}($key), '.parquet'))"
        with pytest.raises(vane.NotImplementedException, match="client-context|client metadata"):
            query(connection, entry, sql, {"key": "threads"})


@pytest.mark.parametrize("entry", ["execute", "sql"])
def test_unsupported_transaction_queries_reject_before_binding(forbid_ray, tmp_path, entry):
    with vane.connect() as connection:
        connection.begin()
        connection.execute("CREATE TABLE transaction_marker(value INTEGER)")
        sql = f"SELECT current_schema() FROM (VALUES ((SELECT count(*) FROM read_parquet('{tmp_path}/missing')))) t(v)"
        with pytest.raises(vane.BinderException, match="auto-commit"):
            query(connection, entry, sql)
        assert ("transaction_marker",) in connection.execute("SELECT table_name FROM duckdb_tables()").fetchall()
        connection.commit()


@pytest.mark.parametrize(
    "sql",
    [
        "SHOW TABLES",
        "DESCRIBE SELECT 1 AS x",
        "PRAGMA disable_profiling",
    ],
)
@pytest.mark.parametrize("derive", ["project", "filter", "order", "limit"])
def test_relations_do_not_inherit_a_client_read_exemption(forbid_ray, sql, derive):
    with vane.connect() as connection:
        relation = connection.sql(sql)
        with pytest.raises(vane.NotImplementedException, match="client-context|client connection|command results"):
            if derive == "project":
                relation.project("*").fetchall()
            elif derive == "filter":
                relation.filter("TRUE").fetchall()
            elif derive == "order":
                relation.order("1").fetchall()
            else:
                relation.limit(1).fetchall()


@pytest.mark.parametrize("reader", ["current_setting('threads')", "getvariable('state_value')", "current_query()"])
@pytest.mark.parametrize("entry", ["execute", "sql", "relation", "transport"])
def test_client_reads_cannot_enter_writes_or_explicit_transports(forbid_ray, tmp_path, entry, reader):
    destination = tmp_path / "rejected.parquet"
    with vane.connect() as connection:
        connection.execute("SET VARIABLE state_value=3")
        with pytest.raises((ValueError, vane.NotImplementedException), match="client-context function"):
            if entry == "relation":
                connection.sql(f"SELECT {reader} AS value").write_parquet(str(destination))
            elif entry == "transport":
                vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(connection.sql(f"SELECT {reader} AS value"), None)
            else:
                getattr(connection, entry)(f"COPY (SELECT {reader}) TO '{destination}' (FORMAT PARQUET)")
    assert not destination.exists()


@pytest.mark.parametrize("entry", ["execute", "sql"])
def test_local_fast_keeps_native_query_composition(monkeypatch, tmp_path, entry):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    destination = tmp_path / "native.parquet"
    with vane.connect() as connection:
        connection.execute("SET threads=3")
        sql = "SELECT range AS value, current_setting('threads') AS threads FROM range(2)"
        assert query(connection, entry, sql) == [(0, 3), (1, 3)]
        assert connection.sql("SELECT current_schema()").project("*").fetchall() == [("main",)]
        assert query(connection, entry, "SELECT count(*) FROM duckdb_columns()")
        getattr(connection, entry)(f"COPY ({sql}) TO '{destination}' (FORMAT PARQUET)")
    assert pq.read_table(destination).to_pydict() == {"value": [0, 1], "threads": [3, 3]}


def test_connection_state_isolated_between_concurrent_connections(forbid_ray):
    with vane.connect() as first, vane.connect() as second:
        first.execute("SET threads=2")
        second.execute("SET threads=5")
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(lambda con: query(con, "sql", "SELECT current_setting('threads')"), [first, second])
            )
        assert results == [[(2,)], [(5,)]]


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
def test_client_reads_do_not_change_data_routing(monkeypatch, tmp_path, entry):
    source = tmp_path / "data.parquet"
    pq.write_table(pa.table({"value": [1, 2]}), source)
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    runner = _TransportedPlanRunner()
    install_runner(monkeypatch, runner)
    try:
        with vane.connect() as connection:
            assert connection.sql("SELECT current_schema()").fetchall() == [("main",)]
            assert not runner.plans
            sql = f"SELECT value + $offset AS value FROM read_parquet('{source}') ORDER BY value"
            if entry == "relation":
                rows = connection.sql(sql, params={"offset": 2}).project("value").fetchall()
            else:
                rows = query(connection, entry, sql, {"offset": 2})
            assert rows == [(3,), (4,)]
            assert len(runner.plans) == 1
            assert connection.sql("SELECT current_schema()").fetchall() == [("main",)]
            assert len(runner.plans) == 1
    finally:
        runner.worker.close()


@pytest.mark.parametrize("entry", ["execute", "sql"])
@pytest.mark.parametrize("sql", ["SELECT current_schema()", "SELECT * FROM duckdb_tables()"])
def test_client_reads_do_not_silently_skip_native_verification(forbid_ray, entry, sql):
    with vane.connect() as connection:
        connection.execute("PRAGMA enable_verification")
        with pytest.raises(vane.NotImplementedException, match="query verification requires a local-fast"):
            query(connection, entry, sql)
        connection.execute("PRAGMA disable_verification")
        expected = [("main",)] if "current_schema" in sql else []
        assert query(connection, entry, sql) == expected


@pytest.mark.parametrize("entry", ["execute", "sql"])
def test_metadata_operators_stay_on_owning_connection(forbid_ray, entry):
    with vane.connect() as connection, vane.connect() as other:
        connection.begin()
        connection.execute("CREATE TABLE alpha(value INTEGER)")
        connection.execute("CREATE TABLE beta(value INTEGER)")
        assert query(
            connection,
            entry,
            "SELECT table_name FROM duckdb_tables() WHERE table_name IN ($a, $b) "
            "ORDER BY table_name DESC LIMIT $limit OFFSET $offset",
            {"a": "alpha", "b": "beta", "limit": 1, "offset": 0},
        ) == [("beta",)]
        assert query(other, entry, "SELECT count(*) FROM duckdb_tables() WHERE table_name='alpha'") == [(0,)]
        assert query(
            connection,
            entry,
            "SELECT schema_name, count(*), min(table_name), max(table_name), sum(column_count) "
            "FROM duckdb_tables() WHERE table_name IN ('alpha', 'beta') "
            "GROUP BY schema_name HAVING count(*) > 1 ORDER BY schema_name",
        ) == [("main", 2, "alpha", "beta", 2)]
        assert query(
            connection,
            entry,
            "SELECT upper(table_name), length(table_name) + 1, "
            "CASE WHEN table_name = 'alpha' THEN 1 ELSE 2 END "
            "FROM duckdb_tables() WHERE NOT (table_name != 'alpha') AND table_name IS NOT NULL",
        ) == [("ALPHA", 6, 1)]
        assert query(
            connection,
            entry,
            "SELECT loaded, install_mode FROM duckdb_extensions() WHERE extension_name = ?",
            ["parquet"],
        ) == [(True, "STATICALLY_LINKED")]
        connection.rollback()
        assert query(connection, entry, "SELECT count(*) FROM duckdb_tables() WHERE table_name='alpha'") == [(0,)]


@pytest.mark.parametrize("entry", ["execute", "sql"])
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT random() FROM duckdb_tables()",
        "SELECT nextval('metadata_probe') FROM duckdb_tables()",
        "SELECT count(*) FROM duckdb_tables(), range(1)",
    ],
)
def test_metadata_operators_reject_other_dependencies(forbid_ray, entry, sql):
    with vane.connect() as connection:
        connection.execute("CREATE TABLE alpha(value INTEGER)")
        connection.execute("CREATE SEQUENCE metadata_probe")
        before = query(connection, entry, "SELECT last_value FROM duckdb_sequences()")
        with pytest.raises(
            vane.NotImplementedException, match="client-context|client connection|client metadata|database-modifying"
        ):
            query(connection, entry, sql)
        assert query(connection, entry, "SELECT last_value FROM duckdb_sequences()") == before


@pytest.mark.parametrize("entry", ["execute", "sql", "relation", "transport"])
def test_metadata_computations_cannot_enter_writes_or_transports(forbid_ray, tmp_path, entry):
    destination = tmp_path / "metadata.parquet"
    sql = "SELECT count(*) AS n FROM duckdb_tables() WHERE table_name = 'alpha'"
    with vane.connect() as connection:
        with pytest.raises(
            (ValueError, vane.NotImplementedException), match="client-context|client connection|client metadata"
        ):
            if entry == "relation":
                connection.sql(sql).write_parquet(str(destination))
            elif entry == "transport":
                vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(connection.sql(sql), None)
            else:
                getattr(connection, entry)(f"COPY ({sql}) TO '{destination}' (FORMAT PARQUET)")
    assert not destination.exists()


def test_metadata_query_rebinds_after_catalog_change(forbid_ray):
    with vane.connect() as connection:
        relation = connection.sql(
            "SELECT count(*) FROM duckdb_tables() WHERE table_name = $name", params={"name": "alpha"}
        )
        assert relation.fetchall() == [(0,)]
        connection.execute("CREATE TABLE alpha(value INTEGER)")
        assert relation.fetchall() == [(1,)]
        connection.execute("DROP TABLE alpha")
        assert relation.fetchall() == [(0,)]


@pytest.mark.parametrize("entry", ["execute", "sql"])
def test_metadata_computations_do_not_invoke_python_udfs(forbid_ray, entry):
    calls = []

    def metadata_probe(value: str) -> str:
        calls.append(value)
        return value

    with vane.connect() as connection:
        vane.attach_function(
            metadata_probe,
            connection=connection,
            alias="metadata_probe",
            parameters=["VARCHAR"],
            return_dtype="VARCHAR",
        )
        with pytest.raises(vane.NotImplementedException, match="client-context|client connection|client metadata"):
            query(connection, entry, "SELECT metadata_probe(schema_name) FROM duckdb_schemas()")
        assert calls == []


@pytest.mark.parametrize("entry", ["execute", "sql"])
def test_metadata_boolean_literals_and_empty_aggregates(forbid_ray, entry):
    with vane.connect() as connection:
        connection.begin()
        assert query(connection, entry, "SELECT count(*) FROM duckdb_schemas() WHERE false") == [(0,)]
        assert query(connection, entry, "SELECT count(*) FROM duckdb_schemas() WHERE true")[0][0] > 0
        assert (
            query(connection, entry, "SELECT count(*) FROM duckdb_schemas() WHERE schema_name = 'main' AND true")[0][0]
            > 0
        )
        connection.rollback()


@pytest.mark.parametrize("entry", ["execute", "sql"])
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT t.table_name FROM duckdb_tables() t JOIN duckdb_schemas() s ON t.schema_oid=s.oid WHERE t.table_name='alpha'",
        "WITH tables AS (SELECT table_name FROM duckdb_tables()) SELECT table_name FROM tables WHERE table_name='alpha'",
        "WITH tables AS MATERIALIZED (SELECT table_name FROM duckdb_tables()) SELECT table_name FROM tables WHERE table_name='alpha'",
        "SELECT table_name FROM (SELECT table_name FROM duckdb_tables()) WHERE table_name=(SELECT 'alpha')",
        "SELECT table_name FROM duckdb_tables() WHERE table_name='alpha' AND EXISTS (SELECT 1 FROM duckdb_schemas())",
        "SELECT table_name FROM duckdb_tables() WHERE table_name IN (SELECT table_name FROM duckdb_tables() WHERE table_name='alpha')",
        "SELECT table_name FROM duckdb_tables() WHERE table_name='alpha' UNION SELECT table_name FROM duckdb_tables() WHERE false",
        "SELECT DISTINCT table_name FROM duckdb_tables() WHERE table_name='alpha'",
        "SELECT table_name FROM metadata_view WHERE table_name='alpha'",
    ],
)
def test_metadata_source_identity_survives_query_composition(forbid_ray, entry, sql):
    with vane.connect() as connection:
        connection.execute("CREATE TABLE alpha(value INTEGER)")
        connection.execute("CREATE VIEW metadata_view AS SELECT table_name FROM duckdb_tables()")
        connection.begin()
        assert query(connection, entry, sql) == [("alpha",)]
        connection.rollback()


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM duckdb_tables() JOIN ordinary ON true",
        "SELECT * FROM ordinary JOIN duckdb_tables() ON true",
        "WITH data AS (SELECT * FROM ordinary) SELECT * FROM duckdb_tables(), data",
        "SELECT * FROM duckdb_tables() WHERE EXISTS (SELECT * FROM ordinary)",
        "SELECT * FROM mixed_view",
    ],
)
def test_metadata_queries_reject_ordinary_sources_in_any_position(forbid_ray, sql):
    with vane.connect() as connection:
        connection.execute("CREATE TABLE ordinary(value INTEGER)")
        connection.execute("CREATE VIEW mixed_view AS SELECT * FROM ordinary, duckdb_tables()")
        with pytest.raises(vane.NotImplementedException, match="Client metadata queries cannot mix"):
            connection.sql(sql).fetchall()


@pytest.mark.parametrize("entry", ["sql", "table_function"])
def test_metadata_relations_keep_source_identity(forbid_ray, entry):
    with vane.connect() as connection:
        connection.execute("CREATE TABLE alpha(value INTEGER)")
        relation = (
            connection.sql("SELECT * FROM duckdb_tables()")
            if entry == "sql"
            else connection.table_function("duckdb_tables")
        )
        connection.begin()
        assert relation.filter("table_name='alpha'").project("table_name").order("table_name").limit(1).fetchall() == [
            ("alpha",)
        ]
        assert relation.filter("table_name='alpha'").aggregate("count(*)").fetchall() == [(1,)]
        connection.rollback()


@pytest.mark.parametrize("entry", ["execute", "sql"])
def test_metadata_windows_check_computation_eligibility(forbid_ray, entry):
    with vane.connect() as connection:
        connection.begin()
        connection.execute("CREATE TABLE alpha(value INTEGER)")
        assert query(
            connection,
            entry,
            "SELECT table_name, row_number() OVER (ORDER BY table_name), count(*) OVER () "
            "FROM duckdb_tables() WHERE table_name='alpha'",
        ) == [("alpha", 1, 1)]
        with pytest.raises(vane.BinderException, match="client metadata does not support function string_agg"):
            query(connection, entry, "SELECT string_agg(table_name) OVER () FROM duckdb_tables()")
        connection.rollback()


@pytest.mark.parametrize("entry", ["execute", "sql"])
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM duckdb_tables(), range(1)",
        "SELECT * FROM range(1), duckdb_tables()",
        "SELECT random() FROM duckdb_tables()",
    ],
)
def test_metadata_admission_errors_preserve_explicit_transaction(forbid_ray, entry, sql):
    with vane.connect() as connection:
        connection.begin()
        connection.execute("CREATE TABLE transaction_marker(value INTEGER)")
        with pytest.raises(vane.BinderException):
            query(connection, entry, sql)
        assert query(
            connection, entry, "SELECT table_name FROM duckdb_tables() WHERE table_name='transaction_marker'"
        ) == [("transaction_marker",)]
        connection.commit()
