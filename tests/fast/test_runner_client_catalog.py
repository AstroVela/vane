# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Client catalog operations and temporary tables must not become remote state."""

import pyarrow as pa
import pytest

import vane
from tests.fast.test_bound_plan_runner import install_runner
from tests.fast.test_distributed_result_consumers import _TransportedPlanRunner


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
    runner = _TransportedPlanRunner()
    install_runner(monkeypatch, runner)
    with vane.connect() as connection:
        connection.register("input", pa.table({"value": [1, 2]}))
        assert connection.sql("SELECT sum(value)::BIGINT AS value FROM input").fetchall() == [(3,)]
        assert len(runner.plans) == 1
