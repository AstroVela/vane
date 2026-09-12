# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Client catalog operations and temporary tables must not become remote state."""

import pyarrow as pa
import pytest

import vane

pytestmark = pytest.mark.local_fast(reason="Native execution and runner contract")


@pytest.fixture
def no_runner(monkeypatch):
    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("client catalog handling must precede runner initialization")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    monkeypatch.setattr(vane._native, "set_runner_local", forbid_initialization)


@pytest.mark.parametrize("runner_type", ["local", "ray"])
@pytest.mark.parametrize("entry", ["execute", "sql", "executemany"])
@pytest.mark.parametrize(
    "query",
    ["SHOW TABLES", "SHOW DATABASES", "SHOW VARIABLES", "SHOW SCHEMAS", "SHOW ALL TABLES", "SHOW TABLES FROM attached"],
)
def test_show_catalog_commands_observe_client_state(monkeypatch, no_runner, runner_type, entry, query):
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    with vane.connect() as connection:
        connection.execute("ATTACH ':memory:' AS attached")
        connection.execute("CREATE TABLE attached.attached_table(value INTEGER)")
        connection.execute("SET VARIABLE client_value=7")
        connection.begin()
        connection.execute("CREATE TABLE client_table(value INTEGER)")
        result = connection.executemany(query, [[]]) if entry == "executemany" else getattr(connection, entry)(query)
        rows = result.fetchall()
        if query == "SHOW TABLES":
            assert ("client_table",) in rows
        elif query == "SHOW DATABASES":
            assert ("attached",) in rows
        elif query == "SHOW VARIABLES":
            assert ("client_value", "7", "INTEGER") in rows
        elif query == "SHOW SCHEMAS":
            assert ("attached", "main") in [row[:2] for row in rows]
        elif query == "SHOW ALL TABLES":
            assert ("attached", "main", "attached_table") in [row[:3] for row in rows]
        else:
            assert rows == [("attached_table",)]
        connection.rollback()
        with pytest.raises(vane.CatalogException, match="client_table"):
            connection.table("client_table")


@pytest.mark.parametrize("runner_type", ["local-fast", "local", "ray"])
@pytest.mark.parametrize("entry", ["execute", "sql", "executemany"])
@pytest.mark.parametrize("transaction", [False, True])
@pytest.mark.parametrize(
    "query, params, column",
    [
        ("DESCRIBE SELECT ? AS x", [7], "x"),
        ("DESCRIBE SELECT $value + 1 AS x", {"value": 7}, "x"),
        ("DESCRIBE SELECT value FROM marker", {}, "value"),
        ("DESCRIBE marker", {}, "value"),
    ],
)
def test_direct_describe_returns_client_schema(
    monkeypatch, no_runner, runner_type, entry, transaction, query, params, column
):
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    with vane.connect() as connection:
        if transaction:
            connection.begin()
        connection.execute("CREATE TEMP TABLE marker(value INTEGER)")
        if entry == "executemany":
            result = connection.executemany(query, [params])
        elif entry == "sql":
            result = connection.sql(query, params=params)
        else:
            result = connection.execute(query, params)
        assert result.fetchall() == [(column, "INTEGER", "YES", None, None, None)]
        if transaction:
            connection.rollback()


@pytest.mark.parametrize("entry", ["execute", "sql", "executemany"])
@pytest.mark.parametrize("parameterized", [False, True])
def test_describe_python_source_keeps_client_routing_after_capture(monkeypatch, no_runner, entry, parameterized):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    description_input = pa.table({"value": [1, 2]})
    expected = [(description_input.column_names[0], "BIGINT", "YES", None, None, None)]
    query = "DESCRIBE SELECT value FROM description_input"
    params = {}
    if parameterized:
        query = "DESCRIBE SELECT value + $offset AS value FROM description_input"
        params = {"offset": 1}
    with vane.connect() as connection:
        if entry == "executemany":
            result = connection.executemany(query, [params])
        elif entry == "sql":
            result = connection.sql(query, params=params)
        else:
            result = connection.execute(query, params)
        del description_input
        assert result.fetchall() == expected
        if entry == "sql":
            # The captured source and command origin survive lazy rebinding.
            assert result.fetchall() == expected


@pytest.mark.parametrize("entry", ["execute", "sql", "executemany"])
def test_describe_does_not_execute_the_inner_select(monkeypatch, no_runner, entry):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    with vane.connect() as connection:
        query = "DESCRIBE SELECT CAST(error('DESCRIBE executed its input') AS INTEGER) AS value FROM range(3)"
        result = connection.executemany(query, [[]]) if entry == "executemany" else getattr(connection, entry)(query)
        assert result.fetchall() == [("value", "INTEGER", "YES", None, None, None)]


@pytest.mark.parametrize(
    "operation", ["subquery", "cte", "mixed", "macro", "copy", "ctas", "relation_copy", "transport", "datasink"]
)
def test_describe_cannot_be_embedded_in_runner_queries(monkeypatch, no_runner, tmp_path, operation):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    destination = tmp_path / "description.parquet"
    describe = "DESCRIBE SELECT 1 AS value"
    with vane.connect() as connection:
        with pytest.raises((vane.NotImplementedException, ValueError), match="client connection quer"):
            if operation == "subquery":
                connection.execute(f"SELECT * FROM ({describe})")
            elif operation == "cte":
                connection.execute(f"WITH description AS (SELECT * FROM ({describe})) SELECT * FROM description")
            elif operation == "mixed":
                connection.execute(f"SELECT column_name FROM ({describe}), range(2)")
            elif operation == "macro":
                connection.execute(f"CREATE MACRO description() AS TABLE SELECT * FROM ({describe})")
                connection.execute("SELECT * FROM description()")
            elif operation == "copy":
                connection.execute(f"COPY (SELECT * FROM ({describe})) TO '{destination}' (FORMAT PARQUET)")
            elif operation == "ctas":
                connection.execute(f"CREATE TABLE created AS SELECT * FROM ({describe})")
            else:
                relation = connection.sql(describe)
                if operation == "relation_copy":
                    relation.write_parquet(str(destination))
                elif operation == "transport":
                    vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, None)
                else:
                    vane.ray_cxx.PyLogicalPlan.from_duckdb_datasink_relation(
                        relation._mark_datasink("description"), None
                    )
        with pytest.raises(vane.CatalogException, match="created"):
            connection.table("created")
    assert not destination.exists()


@pytest.mark.parametrize("entry", ["execute", "sql", "executemany"])
@pytest.mark.parametrize("argument", ["literal", "parameter", "nested", "macro"])
@pytest.mark.parametrize("catalog_query", ["SHOW TABLES", "SHOW DATABASES", "SHOW VARIABLES", "DESCRIBE SELECT 1 AS x"])
@pytest.mark.parametrize("runner_type", ["local-fast", "local", "ray"])
def test_query_expansion_preserves_client_catalog_origin(
    monkeypatch, no_runner, entry, argument, catalog_query, runner_type
):
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    with vane.connect() as connection:
        connection.execute("ATTACH ':memory:' AS attached")
        connection.execute("CREATE TABLE client_table(value INTEGER)")
        connection.execute("SET VARIABLE client_value=7")
        expected = connection.execute(catalog_query).fetchall()
        query = f"SELECT * FROM query('{catalog_query}')"
        params = {}
        if argument == "parameter":
            query = "SELECT * FROM query($catalog_query)"
            params = {"catalog_query": catalog_query}
        elif argument == "nested":
            query = "SELECT * FROM query('" + query.replace("'", "''") + "')"
        elif argument == "macro":
            connection.execute(f"CREATE MACRO client_catalog() AS TABLE {query}")
            query = "SELECT * FROM client_catalog()"

        def execute_query():
            if entry == "executemany":
                return connection.executemany(query, [params])
            if entry == "sql":
                return connection.sql(query, params=params)
            return connection.execute(query, params)

        if runner_type == "ray":
            with pytest.raises(vane.NotImplementedException, match="client connection queries"):
                execute_query().fetchall()
        else:
            assert execute_query().fetchall() == expected


@pytest.mark.parametrize(
    "operation",
    ["sql_copy", "sql_insert", "sql_ctas", "relation_copy", "relation_insert", "relation_create", "read", "datasink"],
)
def test_query_catalog_origin_does_not_relax_write_or_transport_guards(monkeypatch, no_runner, tmp_path, operation):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    database = str(tmp_path / "query_catalog.duckdb")
    destination = tmp_path / "catalog.parquet"
    with vane.connect(database) as connection:
        connection.execute("CREATE TABLE target(name VARCHAR)")
        connection.execute("CREATE SEQUENCE seq")
        connection.execute("CREATE MACRO row_count() AS 1")
        connection.execute("CREATE MACRO client_catalog() AS TABLE SELECT 'target' AS name")
        query = "SELECT name FROM client_catalog(), range(row_count())"
        relation = connection.sql(query)
        # A lazy relation must reject newly introduced client state before the
        # second source can evaluate its sequence argument during rebinding.
        connection.execute("CREATE OR REPLACE MACRO client_catalog() AS TABLE SELECT * FROM query('SHOW TABLES')")
        connection.execute("CREATE OR REPLACE MACRO row_count() AS nextval('seq')")
        with pytest.raises((vane.NotImplementedException, ValueError), match="client connection queries"):
            if operation == "sql_copy":
                connection.execute(f"COPY ({query}) TO '{destination}' (FORMAT PARQUET)")
            elif operation == "sql_insert":
                connection.execute(f"INSERT INTO target {query}")
            elif operation == "sql_ctas":
                connection.execute(f"CREATE TABLE created AS {query}")
            elif operation == "relation_copy":
                relation.write_parquet(str(destination))
            elif operation == "relation_insert":
                relation.insert_into("target")
            elif operation == "relation_create":
                relation.create("created")
            elif operation == "datasink":
                vane.ray_cxx.PyLogicalPlan.from_duckdb_datasink_relation(relation._mark_datasink("catalog"), None)
            else:
                vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, None)
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(database) as inspector:
        assert inspector.execute("SELECT nextval('seq')").fetchone() == (1,)
        assert inspector.table("target").fetchall() == []
        with pytest.raises(vane.CatalogException, match="created"):
            inspector.table("created")
    assert not destination.exists()


@pytest.mark.parametrize("runner_type", ["local-fast", "local"])
@pytest.mark.parametrize("derive", ["project", "filter", "order", "limit"])
def test_native_materialized_command_results_remain_composable(monkeypatch, no_runner, runner_type, derive):
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    with vane.connect() as connection:
        relation = connection.sql("PRAGMA disable_profiling").set_alias("command")
        expected = relation.fetchall()
        columns = relation.columns
        if derive == "project":
            relation = relation.project("command.*")
        elif derive == "filter":
            relation = relation.filter("TRUE")
        elif derive == "order":
            relation = relation.order("1")
        else:
            relation = relation.limit(100)
        assert relation.fetchall() == expected
        assert relation.columns == columns


_TEMPORARY_QUERIES = [
    "SELECT * FROM temporary_source",
    "SELECT * FROM source_view",
    "INSERT INTO temporary_target VALUES (2)",
    "INSERT INTO temporary_target SELECT 2",
    "UPDATE temporary_target SET value=2",
    "DELETE FROM temporary_target",
    "MERGE INTO temporary_target t USING (SELECT 2 AS value) s ON t.value=s.value "
    "WHEN NOT MATCHED THEN INSERT VALUES (s.value)",
    "INSERT INTO target SELECT * FROM temporary_source",
    "UPDATE target SET value=2 FROM temporary_source s WHERE target.value=s.value",
    "DELETE FROM target USING temporary_source s WHERE target.value=s.value",
    "MERGE INTO target t USING temporary_source s ON t.value=s.value WHEN NOT MATCHED THEN INSERT VALUES (s.value)",
    "CREATE TABLE created AS SELECT * FROM temporary_source",
    "COPY temporary_source TO $path (FORMAT PARQUET)",
]


@pytest.mark.parametrize("entry", ["execute", "sql", "executemany"])
@pytest.mark.parametrize("query", _TEMPORARY_QUERIES)
def test_ray_sql_rejects_temporary_sources_and_targets_before_dispatch(monkeypatch, no_runner, tmp_path, entry, query):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    path = tmp_path / "temporary.parquet"
    with vane.connect() as connection:
        connection.execute("CREATE TABLE target(value INTEGER)")
        connection.execute("CREATE TEMP TABLE temporary_target(value INTEGER)")
        connection.execute("CREATE TEMP TABLE temporary_source(value INTEGER)")
        connection.execute("CREATE TEMP VIEW source_view AS SELECT * FROM temporary_source")
        params = {"path": str(path)} if "$path" in query else {}
        with pytest.raises(vane.NotImplementedException, match="temporary table"):
            if entry == "execute":
                result = connection.execute(query, params)
            elif entry == "executemany":
                result = connection.executemany(query, [params])
            else:
                result = connection.sql(query, params=params)
            if result is not None:
                result.fetchall()
        with pytest.raises(vane.CatalogException, match="created"):
            connection.table("created")
    assert not path.exists()


@pytest.mark.parametrize("operation", ["read", "insert", "update", "delete", "merge", "copy", "create"])
def test_ray_relations_reject_temporary_tables_before_dispatch(monkeypatch, no_runner, tmp_path, operation):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    path = tmp_path / "temporary.parquet"
    with vane.connect() as connection:
        connection.execute("CREATE TEMP TABLE temporary_target(value INTEGER)")
        connection.execute("CREATE TEMP TABLE temporary_source(value INTEGER)")
        source = connection.table("temporary_source")
        target = connection.table("temporary_target")
        with pytest.raises(vane.NotImplementedException, match="temporary table"):
            if operation == "read":
                source.fetchall()
            elif operation == "insert":
                connection.sql("SELECT 2 AS value").insert_into("temporary_target")
            elif operation == "update":
                target.update({"value": vane.ConstantExpression(2)})
            elif operation == "delete":
                target.delete()
            elif operation == "merge":
                source.merge_into(
                    "temporary_target",
                    "target.value=source.value",
                    ["WHEN NOT MATCHED THEN INSERT VALUES (source.value)"],
                )
            elif operation == "copy":
                source.write_parquet(str(path))
            else:
                source.create("created")
    assert not path.exists()


@pytest.mark.parametrize("runner_type", ["local-fast", "local", "ray"])
@pytest.mark.parametrize("factory", ["read", "datasink"])
def test_explicit_plan_factories_reject_temporary_tables(monkeypatch, no_runner, runner_type, factory):
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    with vane.connect() as connection:
        connection.execute("CREATE TEMP TABLE source(value INTEGER)")
        relation = connection.table("source")
        if factory == "datasink":
            relation = relation._mark_datasink("temporary-source")
        make_plan = getattr(
            vane.ray_cxx.PyLogicalPlan, f"from_duckdb_{'datasink_' if factory == 'datasink' else ''}relation"
        )
        with pytest.raises(ValueError, match="temporary table"):
            make_plan(relation, None)


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
def test_local_fte_copy_rejects_temporary_sources(monkeypatch, no_runner, tmp_path, entry):
    monkeypatch.setenv("VANE_RUNNER", "local")
    path = tmp_path / "temporary.parquet"
    with vane.connect() as connection:
        connection.execute("CREATE TEMP TABLE source(value INTEGER)")
        with pytest.raises(vane.NotImplementedException, match="temporary table"):
            if entry == "relation":
                connection.table("source").write_parquet(str(path))
            else:
                getattr(connection, entry)(f"COPY source TO '{path}' (FORMAT PARQUET)")
    assert not path.exists()


@pytest.mark.parametrize("runner_type", ["local-fast", "local"])
def test_native_temporary_table_reads_remain_available(monkeypatch, no_runner, runner_type):
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    with vane.connect() as connection:
        connection.execute("CREATE TEMP TABLE source(value INTEGER)")
        if runner_type == "local-fast":
            connection.execute("INSERT INTO source VALUES (7)")
        assert connection.table("source").fetchall() == ([(7,)] if runner_type == "local-fast" else [])


def test_temporary_arrow_views_remain_transportable(ray_local, monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    try:
        with vane.connect() as connection:
            connection.register("input", pa.table({"value": [1, 2]}))
            assert connection.sql("SELECT sum(value)::BIGINT AS value FROM input").fetchall() == [(3,)]
    finally:
        vane.teardown_runner()
