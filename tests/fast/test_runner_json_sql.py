# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""JSON helpers that bind or execute SQL must retain their client context."""

import json

import pytest

import vane
from tests.fast.test_bound_plan_runner import _run_sql_entry, install_runner
from tests.fast.test_distributed_result_consumers import _TransportedPlanRunner

_PLAN_OPTIONS = [
    "",
    ", optimize := true",
    ", optimize := true, skip_null := true",
    ", optimize := true, skip_null := true, skip_empty := true",
    ", skip_null := true, skip_empty := true, skip_default := true, format := true",
]
_INNER_QUERY = "SELECT getvariable('answer') AS value"
_INNER_QUERY_LITERAL = "'SELECT getvariable(''answer'') AS value'"


@pytest.mark.parametrize("options", _PLAN_OPTIONS)
@pytest.mark.parametrize("entry", ["execute", "sql", "parameterized_sql", "relation_query", "relation"])
@pytest.mark.parametrize(
    "runner_type, operation",
    [("ray", "select"), ("ray", "copy"), ("local", "copy"), ("ray", "insert"), ("ray", "ctas")],
)
def test_runner_rejects_json_plan_serialization(monkeypatch, tmp_path, options, entry, runner_type, operation):
    monkeypatch.setenv("VANE_RUNNER", runner_type)

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("JSON plan serialization must fail before initializing a runner")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    monkeypatch.setattr(vane._native, "set_runner_local", forbid_initialization)
    destination = tmp_path / "rejected.parquet"
    with vane.connect() as connection:
        connection.execute("SET VARIABLE answer=41")
        connection.execute("CREATE TABLE target(value JSON)")
        argument = "$query" if entry == "parameterized_sql" else _INNER_QUERY_LITERAL
        source = f"SELECT json_serialize_plan({argument}{options}) AS value"
        query = {
            "select": source,
            "copy": f"COPY ({source}) TO '{destination}' (FORMAT PARQUET)",
            "insert": f"INSERT INTO target {source}",
            "ctas": f"CREATE TABLE created AS {source}",
        }[operation]
        with pytest.raises(vane.NotImplementedException, match="client-context function json_serialize_plan"):
            if entry == "parameterized_sql":
                result = connection.sql(query, params={"query": _INNER_QUERY})
            elif entry == "relation":
                relation = connection.sql(source)
                if operation == "copy":
                    result = relation.write_parquet(str(destination))
                elif operation == "insert":
                    result = relation.insert_into("target")
                elif operation == "ctas":
                    result = relation.create("created")
                else:
                    result = relation
            else:
                result = _run_sql_entry(connection, entry, query)
            if result is not None:
                result.fetchall()
    assert not destination.exists()


@pytest.mark.parametrize("runner_type", ["local-fast", "local", "ray"])
@pytest.mark.parametrize("factory", ["read", "datasink"])
def test_plan_factory_rechecks_json_plan_serialization(monkeypatch, runner_type, factory):
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    with vane.connect() as connection:
        connection.execute("CREATE MACRO describe_query(query_text VARCHAR) AS '{}'::JSON")
        relation = connection.sql(f"SELECT describe_query({_INNER_QUERY_LITERAL}) AS value")
        if factory == "datasink":
            relation = relation._mark_datasink("json-plan-admission")
        connection.execute(
            "CREATE OR REPLACE MACRO describe_query(query_text VARCHAR) AS json_serialize_plan(query_text)"
        )
        make_plan = getattr(
            vane.ray_cxx.PyLogicalPlan, f"from_duckdb_{'datasink_' if factory == 'datasink' else ''}relation"
        )
        with pytest.raises(ValueError, match="client-context function json_serialize_plan"):
            make_plan(relation, None)


@pytest.mark.parametrize("runner_type", ["local-fast", "local"])
@pytest.mark.parametrize("options", _PLAN_OPTIONS)
def test_native_json_plan_serialization_observes_client_variables(monkeypatch, runner_type, options):
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    with vane.connect() as connection:
        connection.execute("SET VARIABLE answer=41")
        actual = connection.execute(f"SELECT json_serialize_plan({_INNER_QUERY_LITERAL}{options})").fetchone()[0]
        expected = connection.execute(f"SELECT json_serialize_plan('SELECT 41 AS value'{options})").fetchone()[0]
        assert json.loads(actual)["error"] is False
        assert json.loads(actual) == json.loads(expected)


@pytest.mark.parametrize("runner_type, operation", [("ray", "select"), ("ray", "copy"), ("local", "copy")])
@pytest.mark.parametrize("entry", ["execute", "sql", "executemany", "relation_query"])
def test_runner_rejects_json_execution_before_nested_binding(monkeypatch, tmp_path, runner_type, operation, entry):
    monkeypatch.setenv("VANE_RUNNER", runner_type)

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("nested JSON SQL must fail before initializing a runner")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    monkeypatch.setattr(vane._native, "set_runner_local", forbid_initialization)
    database = str(tmp_path / "json_nested_binding.duckdb")
    destination = tmp_path / "rejected.parquet"
    with vane.connect(database) as connection:
        connection.execute("CREATE SEQUENCE seq")
        source = (
            "SELECT * FROM json_execute_serialized_sql(json_serialize_sql('SELECT * FROM range(nextval(''seq''))'))"
        )
        query = source if operation == "select" else f"COPY ({source}) TO '{destination}' (FORMAT PARQUET)"
        with pytest.raises(
            vane.NotImplementedException, match="client-context table function json_execute_serialized_sql"
        ):
            result = _run_sql_entry(connection, entry, query)
            if result is not None:
                result.fetchall()
    assert not destination.exists()
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(database) as inspector:
        assert inspector.execute("SELECT nextval('seq')").fetchone() == (1,)


@pytest.mark.parametrize(
    "runner_type, factory", [("local-fast", "read"), ("local", "read"), ("local-fast", "datasink")]
)
def test_plan_factory_rejects_json_execution(monkeypatch, runner_type, factory):
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    with vane.connect() as connection:
        relation = connection.sql("SELECT * FROM json_execute_serialized_sql(json_serialize_sql('SELECT 42 AS value'))")
        if factory == "datasink":
            relation = relation._mark_datasink("json-execution-admission")
        make_plan = getattr(
            vane.ray_cxx.PyLogicalPlan, f"from_duckdb_{'datasink_' if factory == 'datasink' else ''}relation"
        )
        with pytest.raises(ValueError, match="client-context table function json_execute_serialized_sql"):
            make_plan(relation, None)


@pytest.mark.parametrize("runner_type", ["local-fast", "local"])
def test_native_json_execution_still_runs(monkeypatch, runner_type):
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    with vane.connect() as connection:
        assert connection.execute(
            "SELECT * FROM json_execute_serialized_sql(json_serialize_sql('SELECT 42 AS value'))"
        ).fetchall() == [(42,)]


def test_runner_keeps_pure_json_sql_codecs(monkeypatch):
    runner = _TransportedPlanRunner()
    install_runner(monkeypatch, runner)
    try:
        with vane.connect() as connection:
            serialized = connection.execute("SELECT json_serialize_sql('SELECT 42 AS value')").fetchone()[0]
            assert json.loads(serialized)["error"] is False
            query = connection.execute("SELECT json_deserialize_sql(?)", [serialized]).fetchone()[0]
            assert connection.execute(query).fetchall() == [(42,)]
    finally:
        runner.worker.close()


def test_ray_json_execution_pragma_uses_client_variables(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as serializer:
        serialized = serializer.execute("SELECT json_serialize_sql(?)", [_INNER_QUERY]).fetchone()[0]

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("query PRAGMAs must keep the client connection")

    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    with vane.connect() as connection:
        connection.execute("SET VARIABLE answer=41")
        argument = serialized.replace("'", "''")
        assert connection.execute(f"PRAGMA json_execute_serialized_sql('{argument}')").fetchone() == (41,)
