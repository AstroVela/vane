# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Pure state queries use native client execution; mixed queries are rejected."""

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
        variable_key = "$key" if parameterized else "'state_marker'"
        variable_params = {"key": "state_marker"} if parameterized else None
        assert query(connection, entry, f"SELECT getvariable({variable_key})", variable_params) == [(7,)]
        connection.execute("SET VARIABLE state_marker=9")
        assert query(connection, entry, "SELECT getvariable('state_marker')") == [(9,)]


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
@pytest.mark.parametrize("parameterized", [False, True])
@pytest.mark.parametrize(
    "setting, expected", [("disabled_filesystems", ""), ("lock_configuration", False), ("threads", 3)]
)
def test_current_setting_survives_native_query_verification(monkeypatch, entry, parameterized, setting, expected):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as connection:
        connection.execute("SET threads=3")
        connection.execute("PRAGMA enable_verification")
        key = "$key" if parameterized else f"'{setting}'"
        params = {"key": setting} if parameterized else None
        assert query(connection, entry, f"SELECT current_setting({key})", params) == [(expected,)]
        connection.execute("SET threads=5")
        assert query(connection, entry, "SELECT current_setting('threads')") == [(5,)]


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
@pytest.mark.parametrize(
    "expression",
    [
        "current_query()",
        "txid_current()",
        "current_catalog()",
        "CURRENT_CATALOG",
        "pg_catalog.current_database()",
        "pg_catalog.current_schema()",
        "pg_catalog.current_query()",
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
        connection.execute("SET threads=3")
        connection.execute("SET VARIABLE transaction_value=7")
        connection.begin()
        connection.execute("CREATE TABLE transaction_marker(value INTEGER)")
        sql = "SELECT table_name FROM duckdb_tables() WHERE table_name = $name"
        assert query(connection, entry, sql, {"name": "transaction_marker"}) == [("transaction_marker",)]
        timestamp = query(connection, entry, "SELECT CURRENT_TIMESTAMP")[0][0]
        assert query(connection, entry, "SELECT now()")[0][0] == timestamp
        assert query(
            connection,
            entry,
            "SELECT current_setting($key), getvariable('transaction_value')",
            {"key": "threads"},
        ) == [(3, 7)]
        with pytest.raises(vane.BinderException, match="explicit transaction"):
            query(connection, entry, "SELECT current_setting('threads') FROM range(1)")
        connection.rollback()
        assert query(connection, entry, sql, {"name": "transaction_marker"}) == []


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
@pytest.mark.parametrize(
    "extension, sql",
    [
        ("inet", "SELECT html_escape('value')"),
        ("excel", "SELECT count(*) FROM read_xlsx('missing.xlsx')"),
        ("inet", "SELECT html_escape(current_schema())"),
        ("inet", "SELECT current_schema() IS NOT NULL AND CAST('127.0.0.1' AS INET) IS NOT NULL"),
        ("inet", "SELECT CAST('127.0.0.1' AS INET) IS NOT NULL FROM duckdb_tables()"),
    ],
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
            SELECT table_name FROM (SELECT table_name FROM duckdb_tables()) AS metadata
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
def test_lazy_state_relations_rebind_natively(forbid_ray, entry):
    with vane.connect() as connection:
        connection.execute("SET threads=2")
        connection.execute("SET VARIABLE state_value=7")
        relation = connection.sql("SELECT current_setting('threads'), getvariable('state_value')").project("*")
        connection.execute("SET threads=3")
        connection.execute("SET VARIABLE state_value=9")
        assert relation.fetchall() == [(3, 9)]
        connection.execute("SET threads=4")
        assert query(connection, entry, "SELECT current_setting('threads')") == [(4,)]


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
@pytest.mark.parametrize("parameterized", [False, True])
@pytest.mark.parametrize("reader", ["current_setting", "getvariable"])
def test_state_table_arguments_are_rejected_before_opening_files(forbid_ray, tmp_path, entry, parameterized, reader):
    # The missing file proves rejection happens before the data source is opened.
    with vane.connect() as connection:
        connection.execute("SET VARIABLE threads=3")
        key = "$key" if parameterized else "'threads'"
        params = {"key": "threads"} if parameterized else None
        # Bind the state expression as a table argument so the native binder's
        # early effect check runs before any file-system work.
        sql = f"SELECT * FROM read_parquet(concat('{tmp_path}/', {reader}({key}), '.parquet'))"
        with pytest.raises(vane.NotImplementedException, match=f"client-context function {reader}"):
            query(connection, entry, sql, params)


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
@pytest.mark.parametrize("reader", ["current_setting('threads')", "getvariable('threads')", "current_query()"])
def test_state_scalar_cannot_be_written_by_runner(forbid_ray, tmp_path, entry, reader):
    destination = tmp_path / "rejected.parquet"
    with vane.connect() as connection:
        connection.execute("SET VARIABLE threads=3")
        with pytest.raises(vane.NotImplementedException, match="client-context function"):
            if entry == "relation":
                connection.sql(f"SELECT {reader} AS value").write_parquet(str(destination))
            else:
                getattr(connection, entry)(f"COPY (SELECT {reader}) TO '{destination}' (FORMAT PARQUET)")
    assert not destination.exists()


@pytest.mark.parametrize("reader", ["current_setting('threads')", "getvariable('threads')", "current_query()"])
def test_explicit_transport_rejects_native_state_reads(monkeypatch, reader):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as connection:
        connection.execute("SET VARIABLE threads=3")
        relation = connection.sql(f"SELECT {reader} AS value")
        with pytest.raises(ValueError, match="client-context function"):
            vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, None)


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
@pytest.mark.parametrize("reader", ["current_setting('threads')", "getvariable('threads')", "current_query()"])
def test_state_projection_over_parquet_is_rejected(forbid_ray, tmp_path, entry, reader):
    source = tmp_path / "values.parquet"
    pq.write_table(pa.table({"value": [1, 2]}), source)
    with vane.connect() as connection:
        connection.execute("SET VARIABLE threads=3")
        with pytest.raises(vane.NotImplementedException, match="client-context function"):
            query(connection, entry, f"SELECT {reader}, value FROM read_parquet('{source}')")


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
def test_local_fast_keeps_native_mixed_reads_and_writes(monkeypatch, tmp_path, entry):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    destination = tmp_path / "native.parquet"
    with vane.connect() as connection:
        connection.execute("SET threads=3")
        connection.execute("SET VARIABLE state_value=7")
        sql = "SELECT range AS value, current_setting('threads') AS threads, getvariable('state_value') AS marker FROM range(2)"
        assert query(connection, entry, sql) == [(0, 3, 7), (1, 3, 7)]
        if entry == "relation":
            connection.sql(sql).write_parquet(str(destination))
        else:
            getattr(connection, entry)(f"COPY ({sql}) TO '{destination}' (FORMAT PARQUET)")
    assert pq.read_table(destination).to_pydict() == {"value": [0, 1], "threads": [3, 3], "marker": [7, 7]}


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
def test_client_reads_do_not_change_subsequent_data_routing(monkeypatch, entry):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    runner = _TransportedPlanRunner()
    install_runner(monkeypatch, runner)
    try:
        with vane.connect() as connection:
            connection.execute("SET threads=3")
            assert query(connection, entry, "SELECT current_setting($key)", {"key": "threads"}) == [(3,)]
            assert not runner.plans
            assert query(connection, entry, "SELECT range FROM range($rows)", {"rows": 2}) == [(0,), (1,)]
            assert len(runner.plans) == 1
            assert query(connection, entry, "SELECT current_schema()") == [("main",)]
            assert len(runner.plans) == 1
    finally:
        runner.worker.close()


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
@pytest.mark.parametrize("reader", ["current_schema()", "current_query()", "current_schemas(true)"])
def test_parent_binders_cannot_erase_mixed_state_reads(forbid_ray, entry, reader):
    with vane.connect() as connection:
        with pytest.raises(vane.NotImplementedException, match="client-context function"):
            query(connection, entry, f"SELECT typeof({reader}) FROM range(1)")


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
def test_user_macro_cannot_claim_a_builtin_alias_route(forbid_ray, entry):
    with vane.connect() as connection:
        connection.execute("CREATE SCHEMA custom")
        connection.execute("CREATE MACRO custom.current_catalog() AS current_setting('threads')")
        with pytest.raises(vane.NotImplementedException, match="client-context function current_setting"):
            query(connection, entry, "SELECT custom.current_catalog()")


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
@pytest.mark.parametrize("expression, expected", [("TRUE", True), ("FALSE", False), ("CAST('t' AS BOOLEAN)", True)])
def test_native_boolean_literals_in_client_queries(forbid_ray, entry, expression, expected):
    with vane.connect() as connection:
        connection.execute("SET threads=3")
        assert query(connection, entry, f"SELECT current_setting('threads'), {expression}") == [(3, expected)]


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
def test_user_type_cannot_claim_a_builtin_boolean_route(forbid_ray, entry):
    with vane.connect() as connection:
        connection.execute("SET threads=3")
        connection.execute("CREATE SCHEMA custom")
        connection.execute("CREATE TYPE custom.\"BOOLEAN\" AS ENUM ('t', 'f')")
        connection.execute("SET search_path=custom")
        assert query(connection, entry, "SELECT current_setting('threads'), TRUE") == [(3, True)]
        with pytest.raises(vane.NotImplementedException, match="client-context function"):
            query(connection, entry, "SELECT current_setting('threads'), CAST('t' AS custom.\"BOOLEAN\")")


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
@pytest.mark.parametrize("parameterized", [False, True])
def test_which_secret_reads_the_owning_connection(forbid_ray, entry, parameterized):
    with vane.connect() as connection:
        connection.execute(
            "CREATE SECRET state_test_secret "
            "(TYPE HTTP, BEARER_TOKEN 'test-token', SCOPE 'https://runner-state.invalid')"
        )
        path = "$path" if parameterized else "'https://runner-state.invalid/data'"
        params = {"path": "https://runner-state.invalid/data"} if parameterized else None
        assert query(connection, entry, f"SELECT name FROM which_secret({path}, 'http')", params) == [
            ("state_test_secret",)
        ]


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
def test_which_secret_cannot_mix_with_data_scans(forbid_ray, entry):
    with vane.connect() as connection:
        with pytest.raises(vane.NotImplementedException, match="client-context table function which_secret"):
            query(
                connection,
                entry,
                "SELECT name FROM which_secret('https://runner-state.invalid/data', 'http'), range(1)",
            )


@pytest.mark.parametrize("runner", ["local-fast", "ray"])
def test_which_secret_cannot_be_exported(monkeypatch, runner):
    monkeypatch.setenv("VANE_RUNNER", runner)
    with vane.connect() as connection:
        relation = connection.sql("SELECT name FROM which_secret('https://runner-state.invalid/data', 'http')")
        with pytest.raises(
            (ValueError, vane.NotImplementedException), match="client-context table function which_secret"
        ):
            vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, None)


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
@pytest.mark.parametrize("transaction", [False, True])
@pytest.mark.parametrize(
    "sql, expected",
    [
        ("SELECT current_setting('threads'), v FROM (VALUES (1), (2)) t(v) ORDER BY v", [(3, 1), (3, 2)]),
        ("SELECT * FROM (VALUES (current_setting('threads'))) t(v)", [(3,)]),
    ],
)
def test_constant_values_sources_keep_state_queries_native(forbid_ray, entry, transaction, sql, expected):
    with vane.connect() as connection:
        connection.execute("SET threads=3")
        if transaction:
            connection.begin()
        assert query(connection, entry, sql) == expected
        if transaction:
            connection.rollback()


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
def test_values_cannot_hide_a_data_scan_from_transaction_admission(forbid_ray, tmp_path, entry):
    missing = tmp_path / "missing.parquet"
    sql = f"SELECT current_setting('threads') FROM (VALUES ((SELECT count(*) FROM read_parquet('{missing}')))) t(v)"
    with vane.connect() as connection:
        connection.begin()
        with pytest.raises(vane.BinderException, match="auto-commit"):
            query(connection, entry, sql)
        connection.commit()


@pytest.mark.parametrize("command", ["SHOW TABLES", "PRAGMA show_tables", "PRAGMA disable_profiling"])
@pytest.mark.parametrize("composition", ["join_left", "join_right", "subquery", "projection"])
def test_command_composition_rejects_data_before_binding(forbid_ray, tmp_path, command, composition):
    source = tmp_path / "data.parquet"
    pq.write_table(pa.table({"value": [1]}), source)
    with vane.connect() as connection:
        metadata = connection.sql(command).set_alias("metadata")
        data = connection.sql(f"SELECT * FROM read_parquet('{source}')").set_alias("data")
        source.unlink()
        connection.begin()
        connection.execute("CREATE TABLE transaction_marker(value INTEGER)")
        with pytest.raises(vane.BinderException, match="auto-commit"):
            if composition == "join_left":
                metadata.join(data, "TRUE").fetchall()
            elif composition == "join_right":
                data.join(metadata, "TRUE").fetchall()
            elif composition == "projection":
                metadata.project(f"*, (SELECT count(*) FROM read_parquet('{source}')) AS rows").fetchall()
            else:
                metadata.query(
                    "metadata", f"SELECT * FROM metadata WHERE EXISTS (SELECT 1 FROM read_parquet('{source}'))"
                ).fetchall()
        assert connection.sql(
            "SELECT count(*) FROM duckdb_tables() WHERE table_name='transaction_marker'"
        ).fetchall() == [(1,)]
        connection.rollback()


@pytest.mark.parametrize("command", ["SHOW TABLES", "PRAGMA disable_profiling"])
def test_pure_command_results_remain_native_in_transactions(forbid_ray, command):
    with vane.connect() as connection:
        connection.begin()
        connection.execute("CREATE TABLE transaction_marker(value INTEGER)")
        metadata = connection.sql(command)
        expected = metadata.fetchall()
        assert metadata.project("*").fetchall() == expected
        connection.rollback()


@pytest.mark.parametrize("entry", ["execute", "sql"])
def test_direct_native_pragma_expansion_in_transactions(forbid_ray, entry):
    with vane.connect() as connection:
        connection.begin()
        connection.execute("CREATE TABLE transaction_marker(value INTEGER)")
        assert query(connection, entry, "PRAGMA show_tables") == [("transaction_marker",)]
        connection.rollback()


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
@pytest.mark.parametrize("transaction", [False, True])
@pytest.mark.parametrize("qualifier", ["system", "system.main"])
def test_catalog_qualified_state_reads_use_native_resolution(forbid_ray, entry, transaction, qualifier):
    with vane.connect() as connection:
        connection.execute("SET threads=3")
        if transaction:
            connection.begin()
        assert query(connection, entry, f"SELECT {qualifier}.current_setting($key)", {"key": "threads"}) == [(3,)]
        assert query(connection, entry, f"SELECT count(*) FROM {qualifier}.duckdb_settings() WHERE name='threads'") == [
            (1,)
        ]
        assert query(connection, entry, "SELECT current_setting('threads'), CAST('t' AS memory.\"BOOLEAN\")") == [
            (3, True)
        ]
        if transaction:
            connection.commit()


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
def test_catalog_schema_ambiguity_keeps_native_errors(forbid_ray, entry):
    with vane.connect() as connection:
        connection.execute("CREATE SCHEMA system")
        connection.begin()
        for sql in [
            "SELECT system.current_setting('threads')",
            "SELECT count(*) FROM system.duckdb_settings()",
        ]:
            with pytest.raises(vane.BinderException, match="Ambiguous reference to catalog or schema"):
                query(connection, entry, sql)
        assert query(connection, entry, "SELECT count(*) FROM system.main.duckdb_settings() WHERE name='threads'") == [
            (1,)
        ]
        connection.commit()


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
@pytest.mark.parametrize("function", ["pragma_table_info", "pragma_show"])
@pytest.mark.parametrize("parameterized", [False, True])
@pytest.mark.parametrize("transaction", [False, True])
def test_view_metadata_rejects_hidden_sequence_effects(forbid_ray, entry, function, parameterized, transaction):
    with vane.connect() as connection:
        connection.execute("CREATE SEQUENCE metadata_sequence")
        connection.execute("CREATE VIEW metadata_view AS SELECT * FROM range(nextval('metadata_sequence'))")
        sequence = "SELECT last_value FROM duckdb_sequences() WHERE sequence_name='metadata_sequence'"
        before = connection.execute(sequence).fetchall()
        assert before == [(1,)]
        if transaction:
            connection.begin()
            connection.execute("CREATE TABLE transaction_marker(value INTEGER)")
        target = "$target" if parameterized else "'metadata_view'"
        params = {"target": "metadata_view"} if parameterized else None
        with pytest.raises(vane.BinderException, match="metadata cannot rebind a view"):
            query(connection, entry, f"SELECT * FROM {function}({target})", params)
        assert connection.execute(sequence).fetchall() == before
        if transaction:
            assert connection.execute(
                "SELECT count(*) FROM duckdb_tables() WHERE table_name='transaction_marker'"
            ).fetchall() == [(1,)]
            connection.commit()
        assert connection.execute(sequence).fetchall() == before


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
@pytest.mark.parametrize("function", ["pragma_table_info", "pragma_show"])
@pytest.mark.parametrize("parameterized", [False, True])
@pytest.mark.parametrize("transaction", [False, True])
def test_view_metadata_rejects_file_binding(forbid_ray, tmp_path, entry, function, parameterized, transaction):
    source = tmp_path / "view.parquet"
    pq.write_table(pa.table({"value": [1]}), source)
    with vane.connect() as connection:
        connection.execute(f"CREATE VIEW metadata_view AS SELECT * FROM read_parquet('{source}')")
        source.unlink()
        if transaction:
            connection.begin()
        target = "$target" if parameterized else "'metadata_view'"
        params = {"target": "metadata_view"} if parameterized else None
        with pytest.raises(vane.BinderException, match="metadata cannot rebind a view"):
            query(connection, entry, f"SELECT * FROM {function}({target})", params)
        assert connection.execute("SELECT current_schema()").fetchall() == [("main",)]
        if transaction:
            connection.commit()


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
@pytest.mark.parametrize("function, column", [("pragma_table_info", "name"), ("pragma_show", "column_name")])
def test_view_metadata_keeps_proven_native_bindings(forbid_ray, entry, function, column):
    with vane.connect() as connection:
        connection.execute("CREATE TABLE metadata_table(value INTEGER)")
        connection.execute("CREATE VIEW constant_view AS SELECT 1 AS value")
        connection.execute("CREATE VIEW state_view AS SELECT system.current_setting('threads') AS value")
        connection.begin()
        for target in ["metadata_table", "constant_view", "state_view"]:
            assert query(connection, entry, f"SELECT {column} FROM {function}($target)", {"target": target}) == [
                ("value",)
            ]
        connection.commit()


@pytest.mark.parametrize("function", ["pragma_table_info", "pragma_show"])
def test_view_metadata_checks_the_view_search_path(forbid_ray, function):
    with vane.connect() as connection:
        connection.execute("CREATE SEQUENCE metadata_sequence")
        connection.execute("CREATE SCHEMA custom")
        connection.execute("CREATE MACRO custom.pi() AS nextval('metadata_sequence')")
        connection.execute("CREATE VIEW custom.metadata_view AS SELECT pi() AS value")
        with pytest.raises(vane.BinderException, match="metadata cannot rebind a view"):
            connection.execute(f"SELECT * FROM {function}('custom.metadata_view')")
        assert connection.execute(
            "SELECT last_value FROM duckdb_sequences() WHERE sequence_name='metadata_sequence'"
        ).fetchall() == [(None,)]


@pytest.mark.parametrize("function", ["pragma_table_info", "pragma_show"])
def test_view_metadata_relation_rechecks_replaced_views(forbid_ray, function):
    with vane.connect() as connection:
        connection.execute("CREATE VIEW metadata_view AS SELECT 1 AS value")
        relation = connection.sql(f"SELECT * FROM {function}('metadata_view')").project("*")
        connection.execute("CREATE OR REPLACE VIEW metadata_view AS SELECT * FROM range(1)")
        with pytest.raises(vane.BinderException, match="metadata cannot rebind a view"):
            relation.fetchall()


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
def test_columns_metadata_does_not_bind_effectful_unbound_views(forbid_ray, entry):
    with vane.connect() as connection:
        connection.execute("CREATE SCHEMA custom")
        connection.execute("SET search_path=custom")
        connection.execute("CREATE SEQUENCE metadata_sequence")
        # The built-in duckdb_constraints view is initially unbound. Native
        # view binding resolves this table macro through the client search path.
        connection.execute(
            "CREATE MACRO duckdb_constraints() AS TABLE SELECT * FROM range(nextval('metadata_sequence'))"
        )
        sequence = "SELECT last_value FROM duckdb_sequences() WHERE sequence_name='metadata_sequence'"
        # Native CREATE MACRO validates the body once before this metadata read.
        before = connection.execute(sequence).fetchall()
        assert before == [(1,)]
        connection.begin()
        assert query(
            connection,
            entry,
            "SELECT column_name FROM duckdb_columns() "
            "WHERE database_name='system' AND schema_name='main' AND table_name='duckdb_constraints'",
        ) == [(None,)]
        assert connection.execute(sequence).fetchall() == before
        connection.commit()


@pytest.mark.parametrize("function", ["pragma_table_info", "pragma_show"])
def test_local_fast_keeps_native_data_view_metadata(monkeypatch, function):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as connection:
        connection.execute("CREATE VIEW metadata_view AS SELECT * FROM range(1)")
        connection.begin()
        assert len(connection.execute(f"SELECT * FROM {function}('metadata_view')").fetchall()) == 1
        connection.commit()
