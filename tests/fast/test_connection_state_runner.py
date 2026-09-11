# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Connection state is read from its owner, including in transported queries."""

import pickle
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
    relation = connection.sql(sql, params=params)
    if entry == "relation":
        relation = relation.project("*")
    return relation.fetchall()


@pytest.fixture
def forbid_ray(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")

    def initialize(*_args, **_kwargs):
        raise AssertionError("a connection-only query must not initialize Ray")

    monkeypatch.setattr(vane._native, "set_runner_ray", initialize)


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
@pytest.mark.parametrize("parameterized", [False, True])
def test_state_scalars_use_the_owning_connection(forbid_ray, entry, parameterized):
    with vane.connect() as connection:
        connection.execute("SET threads=3")
        key = "$key" if parameterized else "'threads'"
        params = {"key": "threads"} if parameterized else None
        sql = f"SELECT current_setting({key}), current_schema(), current_database(), list_contains(current_schemas(true), 'main')"
        assert query(connection, entry, sql, params) == [(3, "main", "memory", True)]
        connection.execute("SET threads=5")
        assert query(connection, entry, sql, params)[0][0] == 5
        ids = query(connection, entry, "SELECT current_connection_id(), current_query_id(), current_transaction_id()")
        assert all(isinstance(value, int) for value in ids[0])
        assert query(connection, entry, "SELECT current_connection_id()")[0][0] == ids[0][0]
        connection.execute("SET VARIABLE state_marker=7")
        assert query(connection, entry, "SELECT getvariable('state_marker')") == [(7,)]
        connection.execute("SET VARIABLE state_marker=9")
        assert query(connection, entry, "SELECT getvariable('state_marker')") == [(9,)]


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
@pytest.mark.parametrize(
    "expression",
    [
        "current_query()",
        "txid_current()",
        "current_catalog()",
        "in_search_path('memory', 'main')",
        "now()",
        "CURRENT_TIMESTAMP",
        "transaction_timestamp()",
        "CURRENT_DATE",
        "today()",
        "CURRENT_TIME",
        "LOCALTIME",
        "LOCALTIMESTAMP",
        "current_localtime()",
        "current_localtimestamp()",
        "list_transform([1], lambda x: current_query())",
    ],
)
def test_declared_state_readers_remain_client_local(forbid_ray, entry, expression):
    with vane.connect() as connection:
        assert query(connection, entry, f"SELECT ({expression}) IS NOT NULL") == [(True,)]


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
@pytest.mark.parametrize(
    "source, predicate",
    [
        ("duckdb_tables()", "table_name = 'marker'"),
        ("duckdb_columns()", "table_name = 'marker' AND column_name = 'value'"),
        ("duckdb_views()", "view_name = 'marker_view'"),
        ("duckdb_schemas()", "schema_name = 'main'"),
        ("duckdb_databases()", "database_name = 'memory'"),
        ("duckdb_settings()", "name = 'threads' AND value = '3'"),
        ("duckdb_variables()", "name = 'marker_variable'"),
        ("duckdb_extensions()", "extension_name = 'parquet' AND loaded"),
        ("duckdb_functions()", "function_name = 'current_setting'"),
        ("duckdb_types()", "type_name = 'varchar'"),
        ("pragma_table_info('marker')", "name = 'value'"),
        ("pragma_database_size()", "database_name = 'memory'"),
        ("duckdb_memory()", "TRUE"),
        ("pragma_version()", "library_version IS NOT NULL"),
        ("pragma_platform()", "platform IS NOT NULL"),
    ],
)
def test_metadata_reads_use_client_catalog(forbid_ray, entry, source, predicate):
    with vane.connect() as connection:
        connection.execute("SET threads=3")
        connection.execute("CREATE TABLE marker(value INTEGER)")
        connection.execute("CREATE VIEW marker_view AS SELECT 1 AS value")
        connection.execute("SET VARIABLE marker_variable=42")
        assert query(connection, entry, f"SELECT count(*) > 0 FROM {source} WHERE {predicate}") == [(True,)]


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
def test_connection_reads_observe_explicit_transaction(forbid_ray, entry):
    with vane.connect() as connection:
        connection.begin()
        connection.execute("CREATE TABLE transaction_marker(value INTEGER)")
        sql = "SELECT table_name FROM duckdb_tables() WHERE table_name = $name"
        assert query(connection, entry, sql, {"name": "transaction_marker"}) == [("transaction_marker",)]
        timestamp = query(connection, entry, "SELECT CURRENT_TIMESTAMP")[0][0]
        assert query(connection, entry, "SELECT now()")[0][0] == timestamp
        with pytest.raises(vane.BinderException, match="explicit transaction"):
            query(connection, entry, "SELECT current_setting('threads') FROM range(1)")
        connection.rollback()
        assert query(connection, entry, sql, {"name": "transaction_marker"}) == []


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
def test_metadata_parameters_and_nested_composition(forbid_ray, entry):
    with vane.connect() as connection:
        connection.execute("CREATE TABLE marker(value INTEGER)")
        sql = """
            WITH metadata AS (SELECT table_name FROM duckdb_tables())
            SELECT table_name FROM metadata
            WHERE table_name = $name
              AND EXISTS (SELECT 1 FROM duckdb_columns() WHERE table_name = $name)
            ORDER BY table_name
        """
        assert query(connection, entry, sql, {"name": "marker"}) == [("marker",)]
        connection.execute("DROP TABLE marker")
        assert query(connection, entry, sql, {"name": "marker"}) == []


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
def test_mixed_metadata_queries_do_not_run_business_scans_locally(forbid_ray, entry):
    with vane.connect() as connection:
        for sql in [
            "SELECT value FROM duckdb_settings(), range(2) WHERE name='threads'",
            "SELECT range FROM range(2) WHERE EXISTS (SELECT 1 FROM duckdb_tables())",
            "WITH metadata AS (SELECT * FROM duckdb_settings()) SELECT value FROM metadata, range(2)",
        ]:
            with pytest.raises(vane.NotImplementedException, match="client-context table function"):
                query(connection, entry, sql)


def test_composed_show_query_cannot_move_business_scan_to_client(forbid_ray):
    with vane.connect() as connection:
        metadata = connection.sql("SHOW TABLES").set_alias("metadata")
        data = connection.sql("SELECT range FROM range(2)").set_alias("data")
        with pytest.raises(vane.NotImplementedException, match="client connection queries"):
            metadata.join(data, "TRUE").fetchall()


def test_row_dependent_state_function_is_rejected_in_data_query(forbid_ray):
    with vane.connect() as connection:
        with pytest.raises(vane.NotImplementedException, match="client-context function in_search_path"):
            connection.execute("SELECT in_search_path(range::VARCHAR, 'main') FROM range(2)")


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
def test_mixed_state_scalars_are_captured_before_transport(monkeypatch, entry):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    runner = _TransportedPlanRunner()
    runner.worker.execute("SET threads=1")
    install_runner(monkeypatch, runner)
    try:
        with vane.connect() as connection:
            connection.execute("SET threads=3")
            identity = connection.execute("SELECT current_connection_id()").fetchone()[0]
            sql = """
                SELECT range, current_setting($key), current_connection_id(),
                       current_schema(), now(), CURRENT_TIMESTAMP,
                       list_transform([range], lambda x: x + current_setting($key)), current_query()
                FROM range(3) ORDER BY range
            """
            rows = query(connection, entry, sql, {"key": "threads"})
            assert len(runner.plans) == 1
            assert [row[:4] for row in rows] == [(i, 3, identity, "main") for i in range(3)]
            assert len({row[4] for row in rows}) == 1
            assert all(row[4] == row[5] and row[6] == [row[0] + 3] for row in rows)
            assert all("FROM range(3)" in row[7] for row in rows)
            # Replaying the serialized plan must keep the original time and IDs.
            original = pickle.loads(pickle.dumps(runner.plans[0]))
            replay = pa.concat_tables(list(runner.run_iter_tables(original))).to_pylist()
            assert list(replay[0].values()) == list(rows[0])
    finally:
        runner.worker.close()


def test_lazy_relation_captures_state_on_each_execution(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    runner = _TransportedPlanRunner()
    install_runner(monkeypatch, runner)
    try:
        with vane.connect() as connection:
            connection.execute("SET threads=2")
            relation = connection.sql("SELECT current_setting('threads') AS value FROM range(1)").project("value")
            connection.execute("SET threads=3")
            assert relation.fetchall() == [(3,)]
            connection.execute("SET threads=4")
            relation.execute()
            assert relation.fetchall() == [(4,)]
            assert len(runner.plans) == 2
    finally:
        runner.worker.close()


def test_connection_state_isolated_between_concurrent_connections(forbid_ray):
    with vane.connect() as first, vane.connect() as second:
        first.execute("SET threads=2")
        second.execute("SET threads=5")
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(lambda con: query(con, "sql", "SELECT current_setting('threads')"), [first, second])
            )
        assert results == [[(2,)], [(5,)]]


@pytest.mark.real_ray
@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
def test_ray_state_capture_agrees_with_worker_time_zone(ray_local, monkeypatch, tmp_path, entry):
    path = tmp_path / "events.parquet"
    pq.write_table(pa.table({"value": list(range(10000))}), path, row_group_size=1000)
    monkeypatch.setenv("VANE_RUNNER", "ray")
    with vane.connect() as connection:
        connection.execute("SET TimeZone='America/Los_Angeles'")
        sql = """
            SELECT count(*)::BIGINT, min(current_setting('TimeZone')),
                   min(strftime(TIMESTAMPTZ '2000-01-01 00:00:00+00' + value * INTERVAL '1 minute', '%Y-%m-%d %H:%M'))
            FROM read_parquet($path)
        """
        assert query(connection, entry, sql, {"path": str(path)}) == [
            (10000, "America/Los_Angeles", "1999-12-31 16:00")
        ]
