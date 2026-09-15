# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Client metadata routing follows the successful native bound plan."""

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
        raise AssertionError("metadata and rejected plans must not initialize Ray")

    monkeypatch.setattr(vane._native, "set_runner_ray", initialize)


@pytest.fixture
def transported_runner(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    runner = _TransportedPlanRunner()
    install_runner(monkeypatch, runner)
    try:
        yield runner
    finally:
        runner.worker.close()


@pytest.mark.parametrize("entry", ["execute", "sql", "table_function"])
@pytest.mark.parametrize(
    "function,column,expected",
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
def test_metadata_reads_use_the_owning_client(forbid_ray, entry, function, column, expected):
    with vane.connect() as connection:
        connection.execute("CREATE TABLE marker(value INTEGER)")
        connection.execute("CREATE VIEW marker_view AS SELECT 1 AS value")
        connection.execute("CREATE SEQUENCE marker_sequence")
        connection.execute("SET VARIABLE state_value=42")
        connection.execute("ATTACH ':memory:' AS attached")
        connection.begin()
        if entry == "table_function":
            rows = connection.table_function(function).filter(f"{column} = '{expected}'").project(column).fetchall()
        else:
            rows = query(
                connection,
                entry,
                f"SELECT m.{column} FROM {function}() m WHERE m.{column} = $value",
                {"value": expected},
            )
        assert (expected,) in rows
        connection.commit()


@pytest.mark.parametrize("entry", ["execute", "sql"])
@pytest.mark.parametrize(
    "sql,expected",
    [
        ("SELECT count(*) FROM duckdb_tables() WHERE table_name='marker'", [(1,)]),
        ("SELECT upper(table_name) FROM duckdb_tables() WHERE table_name='marker'", [("MARKER",)]),
        ("WITH m AS (SELECT table_name FROM duckdb_tables()) SELECT * FROM m WHERE table_name='marker'", [("marker",)]),
        ("SELECT table_name FROM (SELECT * FROM duckdb_tables()) WHERE table_name='marker'", [("marker",)]),
        (
            "SELECT t.table_name FROM duckdb_tables() t JOIN duckdb_schemas() s USING (schema_name) "
            "WHERE t.table_name='marker' AND s.schema_name='main' AND s.database_name=t.database_name",
            [("marker",)],
        ),
        (
            "SELECT table_name FROM duckdb_tables() WHERE table_name='marker' UNION ALL SELECT 'constant'",
            [("marker",), ("constant",)],
        ),
        ("SELECT table_name FROM duckdb_tables() WHERE table_name='marker' EXCEPT SELECT 'other'", [("marker",)]),
        (
            "SELECT table_name FROM duckdb_tables() WHERE table_name IN ('marker','other') ORDER BY 1 LIMIT 1",
            [("marker",)],
        ),
        (
            "SELECT table_name, row_number() OVER (ORDER BY table_name) FROM duckdb_tables() WHERE table_name='marker'",
            [("marker", 1)],
        ),
        ("SELECT COLUMNS('table_name') FROM duckdb_tables() WHERE table_name='marker'", [("marker",)]),
    ],
)
def test_metadata_sql_composition_uses_native_operators(forbid_ray, entry, sql, expected):
    with vane.connect() as connection:
        connection.execute("CREATE TABLE marker(value INTEGER)")
        assert query(connection, entry, sql) == expected


@pytest.mark.parametrize("entry", ["execute", "sql"])
@pytest.mark.parametrize(
    "source", ["metadata_view", "query_table('metadata_view')", "query('SELECT * FROM metadata_view')"]
)
def test_native_expansion_preserves_actual_metadata_sources(forbid_ray, entry, source):
    with vane.connect() as connection:
        connection.execute("CREATE TABLE marker(value INTEGER)")
        connection.execute("CREATE VIEW metadata_view AS SELECT table_name FROM duckdb_tables()")
        assert query(connection, entry, f"SELECT * FROM {source} WHERE table_name='marker'") == [("marker",)]


@pytest.mark.parametrize("entry", ["execute", "sql"])
@pytest.mark.parametrize("qualifier", ["system", "system.main"])
def test_catalog_qualification_and_transaction_visibility(forbid_ray, entry, qualifier):
    with vane.connect() as connection:
        connection.begin()
        connection.execute("CREATE TABLE transaction_marker(value INTEGER)")
        assert ("transaction_marker",) in query(
            connection, entry, f"SELECT table_name FROM {qualifier}.duckdb_tables()"
        )
        connection.rollback()
        assert ("transaction_marker",) not in query(connection, entry, "SELECT table_name FROM duckdb_tables()")


@pytest.mark.parametrize("entry", ["execute", "sql"])
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM duckdb_tables(), range(1)",
        "SELECT * FROM range(1), duckdb_tables()",
        "SELECT table_name FROM duckdb_tables() UNION ALL SELECT CAST(range AS VARCHAR) FROM range(1)",
        "SELECT current_schema(), range FROM range(1)",
        "SELECT * FROM duckdb_tables() WHERE EXISTS (SELECT * FROM range(1))",
    ],
)
def test_mixed_sources_are_rejected_before_runner_execution(forbid_ray, entry, sql):
    with vane.connect() as connection:
        with pytest.raises(vane.NotImplementedException, match="Client metadata queries cannot mix"):
            query(connection, entry, sql)


@pytest.mark.parametrize("entry", ["execute", "sql", "relation", "transport"])
def test_metadata_cannot_enter_writes_or_explicit_transports(forbid_ray, tmp_path, entry):
    destination = tmp_path / "metadata.parquet"
    with vane.connect() as connection:
        with pytest.raises((ValueError, vane.NotImplementedException), match="client metadata"):
            if entry == "relation":
                connection.sql("SELECT * FROM duckdb_tables()").write_parquet(str(destination))
            elif entry == "transport":
                vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(connection.sql("SELECT * FROM duckdb_tables()"), None)
            else:
                getattr(connection, entry)(f"COPY (SELECT * FROM duckdb_tables()) TO '{destination}' (FORMAT PARQUET)")
    assert not destination.exists()


@pytest.mark.parametrize("entry", ["execute", "sql"])
def test_data_transaction_rejection_preserves_client_work(forbid_ray, entry):
    with vane.connect() as connection:
        connection.begin()
        connection.execute("CREATE TABLE transaction_marker(value INTEGER)")
        with pytest.raises(vane.BinderException, match="auto-commit"):
            query(connection, entry, "SELECT * FROM range(1)")
        assert ("transaction_marker",) in query(connection, entry, "SELECT table_name FROM duckdb_tables()")
        connection.commit()


@pytest.mark.parametrize("entry", ["execute", "sql"])
def test_binding_retains_native_schema_io(forbid_ray, tmp_path, entry):
    with vane.connect() as connection:
        # Native binding finishes before routing. A missing source can fail
        # before Vane has a complete plan to classify, regardless of FROM order.
        for sources in [
            f"duckdb_tables(), read_parquet('{tmp_path}/missing')",
            f"read_parquet('{tmp_path}/missing'), duckdb_tables()",
        ]:
            with pytest.raises(vane.IOException, match="No files found"):
                query(connection, entry, f"SELECT * FROM {sources}")


@pytest.mark.parametrize("entry", ["execute", "sql"])
def test_live_client_read_expressions_compose_natively(forbid_ray, entry):
    with vane.connect() as connection:
        connection.execute("SET threads=3")
        assert query(connection, entry, "SELECT upper(current_schema()), current_setting('threads') + 1") == [
            ("MAIN", 4)
        ]
        assert query(connection, entry, "SELECT list_transform(['x'], lambda x: current_schema() || x)") == [
            (["mainx"],)
        ]
        assert query(connection, entry, "SELECT current_schema(), +1, -(1+1)") == [("main", 1, -2)]


def test_folded_client_values_are_transported_as_constants(transported_runner):
    with vane.connect() as connection:
        connection.execute("SET VARIABLE snapshot_value=7")
        relation = connection.sql("SELECT getvariable($name)::BIGINT AS value", params={"name": "snapshot_value"})
        assert relation.fetchall() == [(7,)]
        connection.execute("SET VARIABLE snapshot_value=9")
        assert relation.fetchall() == [(9,)]
        assert len(transported_runner.plans) == 2


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
def test_metadata_does_not_change_data_routing(transported_runner, tmp_path, entry):
    source = tmp_path / "data.parquet"
    pq.write_table(pa.table({"value": [1, 2]}), source)
    with vane.connect() as connection:
        assert connection.sql("SELECT count(*) FROM duckdb_extensions()").fetchall()[0][0] > 0
        assert not transported_runner.plans
        sql = f"SELECT value + $offset AS value FROM read_parquet('{source}') ORDER BY value"
        rows = (
            connection.sql(sql, params={"offset": 2}).project("value").fetchall()
            if entry == "relation"
            else query(connection, entry, sql, {"offset": 2})
        )
        assert rows == [(3,), (4,)]
        assert len(transported_runner.plans) == 1


def test_builtin_spelling_does_not_grant_metadata_routing(transported_runner):
    with vane.connect() as connection:
        connection.execute("CREATE MACRO duckdb_tables() AS TABLE SELECT 7::BIGINT AS value")
        assert connection.sql("SELECT * FROM duckdb_tables()").fetchall() == [(7,)]
        assert len(transported_runner.plans) == 1


def test_connection_state_isolated_between_concurrent_connections(forbid_ray):
    with vane.connect() as first, vane.connect() as second:
        first.execute("SET threads=2")
        second.execute("SET threads=5")
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(
                pool.map(lambda con: con.sql("SELECT current_setting('threads')").fetchall(), [first, second])
            ) == [[(2,)], [(5,)]]


@pytest.mark.parametrize("entry", ["execute", "sql"])
def test_metadata_queries_reject_native_verification(forbid_ray, entry):
    with vane.connect() as connection:
        connection.execute("PRAGMA enable_verification")
        with pytest.raises(vane.NotImplementedException, match="query verification requires a local-fast"):
            query(connection, entry, "SELECT * FROM duckdb_tables()")
        connection.execute("PRAGMA disable_verification")
        assert query(connection, entry, "SELECT * FROM duckdb_tables()") == []


@pytest.mark.parametrize("entry", ["execute", "sql"])
def test_table_arguments_capture_native_binding_values(transported_runner, entry):
    with vane.connect() as connection:
        connection.execute("SET threads=3")
        assert query(connection, entry, "SELECT * FROM range(current_setting('threads'))") == [(0,), (1,), (2,)]
        assert len(transported_runner.plans) == 1


def test_native_binding_can_apply_effects_before_admission(forbid_ray, monkeypatch, tmp_path):
    database = str(tmp_path / "binding.duckdb")
    with vane.connect(database) as connection:
        connection.execute("CREATE SEQUENCE seq")
        connection.begin()
        connection.execute("CREATE TABLE transaction_marker(value INTEGER)")
        with pytest.raises(vane.BinderException, match="database-modifying expressions"):
            connection.execute("SELECT * FROM range(nextval('seq'))")
        connection.commit()
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(database) as inspector:
        assert inspector.execute("SELECT nextval('seq')").fetchone()[0] > 1


def test_native_setting_binding_reports_its_own_error(forbid_ray):
    with vane.connect(
        config={"autoload_known_extensions": "false", "autoinstall_known_extensions": "false"}
    ) as connection:
        with pytest.raises(vane.CatalogException, match="exists in the azure extension"):
            connection.sql("SELECT current_setting('azure_storage_connection_string')")


@pytest.mark.real_ray
def test_metadata_then_parameterized_data_on_real_ray(ray_local, monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    source = tmp_path / "ray-data.parquet"
    pq.write_table(pa.table({"value": [1, 2, 3]}), source)
    try:
        with vane.connect() as connection:
            connection.execute("CREATE TABLE client_only_marker(value INTEGER)")
            assert connection.sql(
                "SELECT count(*) FROM duckdb_tables() WHERE table_name=$name", params={"name": "client_only_marker"}
            ).fetchall() == [(1,)]
            relation = connection.sql(
                "SELECT value + $offset AS value FROM read_parquet($source)",
                params={"offset": 10, "source": str(source)},
            )
            assert relation.filter("value > 11").order("value").fetchall() == [(12,), (13,)]
    finally:
        vane.teardown_runner()


@pytest.mark.parametrize("entry", ["execute", "sql", "statement"])
def test_parameter_capture_does_not_interpolate_query_text(forbid_ray, entry):
    with vane.connect() as connection:
        sql = "SELECT current_query() AS query_text, $value AS value"
        params = {"value": "bound-secret-value"}
        result = (
            connection.execute(connection.extract_statements(sql)[0], params).fetchall()
            if entry == "statement"
            else query(connection, entry, sql, params)
        )
        assert result[0][1] == params["value"]
        assert params["value"] not in result[0][0]
        assert "$value" in result[0][0]


@pytest.mark.parametrize("entry", ["execute", "sql", "executemany", "statement"])
def test_direct_pragma_keeps_native_preprocessing(forbid_ray, entry):
    with vane.connect() as connection:
        connection.execute("CREATE TABLE marker(value INTEGER)")
        sql = "PRAGMA table_info('marker')"
        if entry == "statement":
            result = connection.execute(connection.extract_statements(sql)[0])
        elif entry == "executemany":
            result = connection.executemany(sql, [[]])
        else:
            result = getattr(connection, entry)(sql)
        assert result.fetchall() == [(0, "value", "INTEGER", False, None, False)]


@pytest.mark.parametrize("runner", ["ray", "local-fast"])
@pytest.mark.parametrize("entry", ["execute", "sql", "executemany", "statement"])
def test_pragma_named_arguments_reach_native_binding(forbid_ray, monkeypatch, runner, entry):
    monkeypatch.setenv("VANE_RUNNER", runner)
    with vane.connect() as connection:
        connection.execute("CREATE TABLE marker(value INTEGER)")
        sql = "PRAGMA table_info('marker', unexpected=10)"
        with pytest.raises(vane.BinderException, match='Invalid named parameter "unexpected"'):
            if entry == "statement":
                connection.execute(connection.extract_statements(sql)[0])
            elif entry == "executemany":
                connection.executemany(sql, [[]])
            else:
                getattr(connection, entry)(sql)
        assert connection.execute("PRAGMA table_info('marker')").fetchall() == [
            (0, "value", "INTEGER", False, None, False)
        ]


def test_completed_pragma_rows_can_be_composed_as_data(transported_runner):
    with vane.connect() as connection:
        connection.execute("CREATE TABLE marker(value INTEGER)")
        relation = connection.sql("PRAGMA show_tables")
        assert relation.fetchall() == [("marker",)]
        connection.execute("CREATE TABLE later(value INTEGER)")
        assert relation.project("name").fetchall() == [("marker",)]
        assert len(transported_runner.plans) == 1
