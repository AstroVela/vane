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
@pytest.mark.parametrize(
    "extension, sql",
    [("inet", "SELECT html_escape('value')"), ("excel", "SELECT count(*) FROM read_xlsx('missing.xlsx')")],
)
def test_transaction_classification_does_not_autoload_extensions(forbid_ray, tmp_path, entry, extension, sql):
    extension_directory = tmp_path / "extensions"
    config = {
        "autoload_known_extensions": "true",
        "autoinstall_known_extensions": "true",
        "custom_extension_repository": "http://127.0.0.1:9",
        "extension_directory": str(extension_directory),
    }
    with vane.connect(config=config) as connection:
        source = connection.sql("SELECT 1 AS value")
        extensions = "SELECT extension_name, loaded FROM duckdb_extensions() ORDER BY extension_name"
        before = connection.execute(extensions).fetchall()
        assert not dict(before)[extension]
        connection.begin()
        connection.execute("CREATE TABLE transaction_marker(value INTEGER)")
        with pytest.raises(vane.BinderException, match="explicit transaction"):
            if entry == "relation":
                source.project(f"({sql}) AS extension_result").fetchall()
            else:
                getattr(connection, entry)(sql)
        assert connection.execute(extensions).fetchall() == before
        assert connection.execute(
            "SELECT table_name FROM duckdb_tables() WHERE table_name = 'transaction_marker'"
        ).fetchall() == [("transaction_marker",)]
        connection.rollback()
    assert not extension_directory.exists()


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


@pytest.mark.parametrize("sql", ["SHOW TABLES", "PRAGMA disable_profiling"])
def test_composed_client_query_cannot_move_business_scan_to_client(forbid_ray, sql):
    with vane.connect() as connection:
        metadata = connection.sql(sql).set_alias("metadata")
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
                       current_schema(), now() AS clock_a, CURRENT_TIMESTAMP AS clock_b,
                       list_transform([range], lambda x: x + current_setting($key)), current_query()
                FROM range(3) ORDER BY range
            """
            rows = query(connection, entry, sql, {"key": "threads"})
            assert len(runner.plans) == 1
            assert [row[:4] for row in rows] == [(i, 3, identity, "main") for i in range(3)]
            assert len({row[4] for row in rows}) == 1
            assert all(row[4] == row[5] and row[6] == [row[0] + 3] for row in rows)
            assert all("current_query()" in row[7] and "range" in row[7] for row in rows)
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


@pytest.mark.parametrize("runner_type", ["local-fast", "ray"])
def test_explicit_plan_export_captures_its_own_query_state(monkeypatch, runner_type):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    runner = _TransportedPlanRunner()
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    try:
        with vane.connect() as connection:
            connection.execute("SET threads=3")
            previous_id = connection.execute("SELECT current_query_id() FROM duckdb_settings() LIMIT 2").fetchone()[0]
            relation = connection.sql(
                "SELECT range, current_query_id() AS query_id, current_query() AS query_text, "
                "current_setting('threads') AS threads, now() AS clock_a, CURRENT_TIMESTAMP AS clock_b "
                "FROM range(2)"
            ).project("*")
            plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, None)
            # The low-level transport fixture exposes physical column names;
            # the public result adapter restores the Relation's logical names.
            table = pa.concat_tables(list(runner.run_iter_tables(plan)))
            rows = table.rename_columns(relation.columns).to_pylist()
            assert len(rows) == 2
            assert all(row["query_id"] != previous_id for row in rows)
            assert len({row["query_id"] for row in rows}) == 1
            assert all("range" in row["query_text"] and "duckdb_settings" not in row["query_text"] for row in rows)
            assert all(row["threads"] == 3 and row["clock_a"] == row["clock_b"] for row in rows)
            # Export completes its binding transaction and leaves the connection usable.
            connection.begin()
            connection.execute("CREATE TABLE after_export(value INTEGER)")
            connection.rollback()
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


@pytest.mark.real_ray
@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
def test_ray_writes_preserve_captured_client_state(ray_local, monkeypatch, tmp_path, entry):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    destination = tmp_path / "captured.parquet"
    with vane.connect() as connection:
        connection.execute("SET threads=3")
        identity = connection.execute("SELECT current_connection_id()").fetchone()[0]
        source = """
            SELECT range AS value, current_setting('threads') AS threads,
                   current_connection_id() AS connection_id, current_query() AS query_text,
                   now() AS clock_a, CURRENT_TIMESTAMP AS clock_b
            FROM range(6)
        """
        if entry == "relation":
            connection.sql(source).write_parquet(str(destination))
        else:
            result = getattr(connection, entry)(f"COPY ({source}) TO '{destination}' (FORMAT PARQUET)")
            if entry == "execute":
                assert result.fetchall() == [(6,)]
            else:
                assert result is None
        rows = pq.read_table(destination).to_pylist()
        assert sorted(row["value"] for row in rows) == list(range(6))
        assert all(row["threads"] == 3 and row["connection_id"] == identity for row in rows)
        if entry == "relation":
            # Native Relation write terminals have no SQL query text.
            assert all(row["query_text"] == "" for row in rows)
        else:
            assert all("current_query()" in row["query_text"] for row in rows)
        assert len({row["clock_a"] for row in rows}) == 1
        assert all(row["clock_a"] == row["clock_b"] for row in rows)
