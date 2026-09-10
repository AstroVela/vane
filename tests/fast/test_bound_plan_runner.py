# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""SQL and Relation execution share admission of an already-bound plan."""

import pickle
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pytest

import vane
from tests.fast.test_distributed_result_consumers import _TransportedPlanRunner


class RecordingRunner:
    def __init__(self):
        self.reads = []
        self.writes = []

    def run_iter_tables(self, plan):
        assert isinstance(plan, vane.ray_cxx.PyLogicalPlan)
        self.reads.append(plan)
        yield pa.table({"value": pa.array([42], pa.int64())})

    def run_write(self, plan):
        assert isinstance(plan, vane.ray_cxx.PyLogicalPlan)
        self.writes.append(pickle.loads(pickle.dumps(plan)))
        return {"copy_operation_id": plan.idx(), "rows_copied": 3}


def install_runner(monkeypatch, runner):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setattr(vane._native, "set_runner_ray", lambda *_args, **_kwargs: runner)


def test_explicit_plan_factory_allocates_stable_unique_query_ids():
    with vane.connect() as connection:
        relation = connection.sql("SELECT 7::BIGINT AS value")
        first = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, None)
        second = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, None)
        assert uuid.UUID(first.idx()) != uuid.UUID(second.idx())
        assert pickle.loads(pickle.dumps(first)).idx() == first.idx()
        assert (
            vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "explicit-query-id").idx() == "explicit-query-id"
        )


@pytest.mark.parametrize("runner_type", ["local-fast", "local", "ray"])
@pytest.mark.parametrize("factory", ["read", "datasink"])
@pytest.mark.parametrize(
    "query, message",
    [
        ("SELECT current_query()", "client-context function"),
        ("SELECT concat('query: ', current_query())", "client-context function"),
        ("SELECT * FROM duckdb_settings()", "client-context table function"),
        ("SELECT nextval('seq')", "database-modifying expressions"),
        ("SELECT list_transform([1], lambda x: current_query())", "client-context function"),
        ("SELECT list_transform([1], lambda x: nextval('seq'))", "database-modifying expressions"),
    ],
)
def test_explicit_plan_factories_apply_runner_admission(monkeypatch, runner_type, factory, query, message):
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    with vane.connect() as connection:
        connection.execute("CREATE SEQUENCE seq")
        relation = connection.sql(query)
        if factory == "datasink":
            relation = relation._mark_datasink("factory-admission")
        make_plan = getattr(
            vane.ray_cxx.PyLogicalPlan, f"from_duckdb_{'datasink_' if factory == 'datasink' else ''}relation"
        )
        with pytest.raises(ValueError, match=message):
            make_plan(relation, None)


@pytest.mark.parametrize("runner_type", ["local-fast", "local", "ray"])
def test_explicit_plan_factory_rejects_client_query_origin(monkeypatch, runner_type):
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    with vane.connect() as connection:
        relation = connection.sql("PRAGMA show_tables").project("name")
        with pytest.raises(ValueError, match="client connection queries|client-context table function"):
            vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, None)


@pytest.mark.parametrize("runner_type", ["local-fast", "local", "ray"])
def test_explicit_read_factory_rechecks_transaction_before_binding(monkeypatch, runner_type):
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    with vane.connect() as connection:
        connection.execute("CREATE SEQUENCE seq")
        connection.execute("CREATE MACRO row_count() AS 2")
        relation = connection.sql("SELECT * FROM range(row_count())")
        connection.begin()
        connection.execute("CREATE OR REPLACE MACRO row_count() AS nextval('seq')")
        with pytest.raises(ValueError, match="cannot participate.*explicit transaction"):
            vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, None)
        connection.execute("CREATE TABLE still_active(value INTEGER)")
        connection.rollback()
        with pytest.raises(vane.CatalogException, match="still_active"):
            connection.table("still_active")


@pytest.mark.parametrize("runner_type", ["local-fast", "local", "ray"])
def test_explicit_plan_factory_checks_bind_time_effects_after_macro_replacement(monkeypatch, tmp_path, runner_type):
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    database = str(tmp_path / "factory_effects.duckdb")
    with vane.connect(database) as connection:
        connection.execute("CREATE SEQUENCE seq")
        connection.execute("CREATE MACRO row_count() AS 2")
        relation = connection.sql("SELECT * FROM range(row_count())")
        connection.execute("CREATE OR REPLACE MACRO row_count() AS nextval('seq')")
        with pytest.raises(ValueError, match="database-modifying expressions"):
            vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, None)
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(database) as inspector:
        assert inspector.execute("SELECT nextval('seq')").fetchone() == (1,)


@pytest.mark.parametrize("derive", ["project", "filter", "order"])
def test_runner_relation_rebinding_checks_bind_time_effects(monkeypatch, tmp_path, derive):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    database = str(tmp_path / "relation_effects.duckdb")
    with vane.connect(database) as connection:
        connection.execute("CREATE SEQUENCE seq")
        connection.execute("CREATE MACRO row_count() AS 2")
        relation = connection.sql("SELECT range AS value FROM range(row_count())")
        argument = "value > 0" if derive == "filter" else "value"
        relation = getattr(relation, derive)(argument)
        connection.execute("CREATE OR REPLACE MACRO row_count() AS nextval('seq')")
        with pytest.raises(vane.NotImplementedException, match="database-modifying expressions"):
            relation.fetchall()
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(database) as inspector:
        assert inspector.execute("SELECT nextval('seq')").fetchone() == (1,)


@pytest.mark.parametrize("runner_type", ["local-fast", "local", "ray"])
@pytest.mark.parametrize(
    "expression, expected",
    [("range + $offset", [5, 6, 7]), ("list_transform([range], lambda x: x + $offset)", [[5], [6], [7]])],
)
def test_explicit_plan_factory_preserves_portable_binding(monkeypatch, runner_type, expression, expected):
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    with vane.connect() as connection:
        connection.execute("SET VARIABLE row_count=3")
        relation = connection.sql(
            f"SELECT {expression} AS value FROM range(getvariable('row_count'))", params={"offset": 5}
        )
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, None)
        runner = _TransportedPlanRunner()
        try:
            result = pa.concat_tables(list(runner.run_iter_tables(plan)))
            assert result.num_columns == 1
            assert result.column(0).to_pylist() == expected
        finally:
            runner.worker.close()


@pytest.mark.parametrize("entry", ["execute", "sql", "executemany", "relation_query"])
@pytest.mark.parametrize(
    "runner_type, operation",
    [("ray", "select"), *[(runner, op) for runner in ["local", "ray"] for op in ["copy", "insert", "ctas"]]],
)
@pytest.mark.parametrize(
    "expression",
    [
        "nextval('seq')",
        "nextval('seq') + 1",
        "hidden_sequence()",
        "getvariable(CAST(nextval('seq') AS VARCHAR))",
        "CASE WHEN FALSE THEN nextval('seq') ELSE 2 END",
        "array_length(list_transform([1], x -> nextval('seq')))",
    ],
)
def test_runner_rejects_bind_time_effects_in_autocommit(
    monkeypatch, tmp_path, entry, runner_type, operation, expression
):
    monkeypatch.setenv("VANE_RUNNER", runner_type)

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("bind-time effects must be rejected before initializing a runner")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    monkeypatch.setattr(vane._native, "set_runner_local", forbid_initialization)
    database = str(tmp_path / "bind_effects.duckdb")
    destination = tmp_path / "rejected.parquet"
    with vane.connect(database) as connection:
        connection.execute("CREATE SEQUENCE seq")
        connection.execute("CREATE TABLE target(value BIGINT)")
        connection.execute("CREATE MACRO hidden_sequence() AS nextval('seq')")
        source = f"SELECT range AS value FROM range({expression})"
        query = {
            "select": source,
            "copy": f"COPY ({source}) TO '{destination}' (FORMAT PARQUET)",
            "insert": f"INSERT INTO target {source}",
            "ctas": f"CREATE TABLE created AS {source}",
        }[operation]
        message = "database-modifying expressions"
        errors = (vane.NotImplementedException,)
        if runner_type == "local" and operation in {"insert", "ctas"}:
            # Local FTE can reject the terminal itself before inspecting a
            # runtime lambda body; it does not support these table writes.
            message += "|requires a ray or local-fast connection"
            errors += (vane.InvalidInputException,)
        with pytest.raises(errors, match=message):
            result = _run_sql_entry(connection, entry, query)
            # Lambda arguments can select range's table-in/table-out overload,
            # whose argument is evaluated at execution rather than binding.
            if result is not None:
                result.fetchall()
        connection.execute("CREATE TABLE followup(value INTEGER)")
    assert not destination.exists()
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(database) as inspector:
        assert inspector.execute("SELECT nextval('seq')").fetchone() == (1,)
        assert inspector.table("target").fetchall() == []
        assert inspector.table("followup").fetchall() == []
        with pytest.raises(vane.CatalogException, match="created"):
            inspector.table("created")


@pytest.mark.parametrize("entry", ["execute", "sql", "executemany"])
def test_runner_rejects_parameterized_bind_time_effects(monkeypatch, tmp_path, entry):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    database = str(tmp_path / "parameter_effects.duckdb")
    with vane.connect(database) as connection:
        connection.execute("CREATE SEQUENCE seq")
        query = "SELECT * FROM range(nextval($sequence))"
        params = {"sequence": "seq"}
        with pytest.raises(vane.NotImplementedException, match="database-modifying expressions"):
            if entry == "sql":
                connection.sql(query, params=params)
            elif entry == "executemany":
                connection.executemany(query, [params])
            else:
                connection.execute(query, params)
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(database) as inspector:
        assert inspector.execute("SELECT nextval('seq')").fetchone() == (1,)


@pytest.mark.parametrize("expression", ["length(current_query())", "length(getvariable(current_query()))"])
def test_runner_checks_client_context_before_table_argument_folding(monkeypatch, expression):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    with vane.connect() as connection:
        with pytest.raises(vane.NotImplementedException, match="client-context function"):
            connection.sql(f"SELECT * FROM range({expression})")


@pytest.mark.parametrize(
    "expression, message",
    [
        ("CAST(nextval('seq') AS VARCHAR)", "database-modifying expressions"),
        ("current_query()", "client-context function"),
        ("getvariable(current_query())", "client-context function"),
    ],
)
def test_runner_checks_lambda_effects_in_bind_time_table_arguments(monkeypatch, tmp_path, expression, message):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    database = str(tmp_path / "lambda_effects.duckdb")
    with vane.connect(database) as connection:
        connection.execute("CREATE SEQUENCE seq")
        # Unlike range, read_csv always evaluates these arguments during binding.
        query = f"SELECT * FROM read_csv(list_transform(['unused'], lambda path: {expression}))"
        with pytest.raises(vane.NotImplementedException, match=message):
            connection.sql(query)
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(database) as inspector:
        assert inspector.execute("SELECT nextval('seq')").fetchone() == (1,)


@pytest.mark.parametrize("runner_type", ["local-fast", "local"])
def test_native_read_binding_keeps_table_argument_effects(monkeypatch, runner_type):
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    with vane.connect() as connection:
        connection.begin()
        connection.execute("CREATE SEQUENCE seq")
        # Native preparation/rebinding evaluates this argument twice, as it did
        # before runner bind-time admission was introduced.
        assert connection.execute("SELECT * FROM range(nextval('seq'))").fetchall() == [(0,), (1,)]
        assert connection.execute("SELECT nextval('seq')").fetchone() == (3,)
        connection.commit()


@pytest.mark.parametrize("runner_type", ["local", "ray"])
@pytest.mark.parametrize("entry", ["execute", "sql", "executemany", "relation_query"])
@pytest.mark.parametrize("query", ["CALL checkpoint()", "CALL range(3)"])
def test_sql_call_is_not_dispatched_as_a_distributed_read(monkeypatch, runner_type, entry, query):
    monkeypatch.setenv("VANE_RUNNER", runner_type)

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("SQL CALL must fail before runner initialization")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    monkeypatch.setattr(vane._native, "set_runner_local", forbid_initialization)
    with vane.connect() as connection:
        with pytest.raises(vane.NotImplementedException, match="SQL CALL"):
            if entry == "executemany":
                connection.executemany(query, [[]])
            elif entry == "relation_query":
                connection.sql("SELECT 1").query("input", query)
            else:
                getattr(connection, entry)(query)


def _run_sql_entry(connection, entry, query, query_relation=None):
    if entry == "executemany":
        return connection.executemany(query, [[]])
    if entry == "relation_query":
        if query_relation is None:
            query_relation = connection.sql("SELECT 1")
        return query_relation.query("input", query)
    return getattr(connection, entry)(query)


@pytest.mark.parametrize("runner_type", ["local", "ray"])
@pytest.mark.parametrize("entry", ["execute", "sql", "executemany", "relation_query"])
@pytest.mark.parametrize("transactional", [False, True])
@pytest.mark.parametrize(
    "query",
    [
        "PREPARE p AS SELECT * FROM range(nextval('seq'))",
        "EXPLAIN ANALYZE SELECT * FROM range(nextval('seq'))",
        "CALL range(nextval('seq'))",
        "EXPLAIN PREPARE p AS SELECT * FROM range(nextval('seq'))",
        "EXPLAIN CALL range(nextval('seq'))",
    ],
)
def test_rejected_wrappers_do_not_bind_effectful_arguments(
    monkeypatch, tmp_path, runner_type, entry, transactional, query
):
    monkeypatch.setenv("VANE_RUNNER", runner_type)

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("unsupported wrappers must fail before runner initialization")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    monkeypatch.setattr(vane._native, "set_runner_local", forbid_initialization)
    database = str(tmp_path / "wrappers.duckdb")
    with vane.connect(database) as connection:
        connection.execute("CREATE SEQUENCE seq")
        query_relation = connection.sql("SELECT 1") if entry == "relation_query" else None
        if transactional:
            connection.begin()
            connection.execute("CREATE TABLE marker(value INTEGER)")
        with pytest.raises(vane.NotImplementedException, match="SQL CALL|SQL PREPARE, EXECUTE, or EXPLAIN ANALYZE"):
            _run_sql_entry(connection, entry, query, query_relation)
        if transactional:
            connection.execute("CREATE TABLE followup(value INTEGER)")
            connection.commit()
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(database) as inspector:
        assert inspector.execute("SELECT nextval('seq')").fetchone() == (1,)
        if transactional:
            assert inspector.table("marker").fetchall() == []
            assert inspector.table("followup").fetchall() == []


@pytest.mark.parametrize("runner_type", ["local", "ray"])
@pytest.mark.parametrize("entry", ["execute", "sql", "executemany", "relation_query"])
@pytest.mark.parametrize("operation", ["copy_from", "client_context", "insert"])
def test_bound_admission_errors_preserve_transactional_work(monkeypatch, tmp_path, runner_type, entry, operation):
    monkeypatch.setenv("VANE_RUNNER", runner_type)

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("admission rejection must precede runner initialization")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    monkeypatch.setattr(vane._native, "set_runner_local", forbid_initialization)
    source = tmp_path / "input.csv"
    source.write_text("value\n1\n")
    destination = tmp_path / "rejected.parquet"
    query = {
        "copy_from": f"COPY marker FROM '{source}' (FORMAT CSV, HEADER)",
        "client_context": f"COPY (SELECT current_query()) TO '{destination}' (FORMAT PARQUET)",
        "insert": "INSERT INTO marker VALUES (1)",
    }[operation]
    database = str(tmp_path / "admission.duckdb")
    with vane.connect(database) as connection:
        query_relation = connection.sql("SELECT 1") if entry == "relation_query" else None
        connection.begin()
        connection.execute("CREATE TABLE marker(value INTEGER)")
        with pytest.raises(vane.BinderException, match="cannot participate.*explicit transaction"):
            _run_sql_entry(connection, entry, query, query_relation)
        connection.execute("CREATE TABLE followup(value INTEGER)")
        connection.commit()
    assert not destination.exists()
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(database) as inspector:
        assert inspector.table("marker").fetchall() == []
        assert inspector.table("followup").fetchall() == []


@pytest.mark.parametrize("entry", ["execute", "sql", "executemany", "relation_query"])
@pytest.mark.parametrize(
    ("runner_type", "operation"),
    [
        ("ray", "select"),
        *[(runner, operation) for runner in ["local", "ray"] for operation in ["copy", "insert", "ctas"]],
    ],
)
def test_transaction_rejection_precedes_table_function_argument_evaluation(
    monkeypatch, tmp_path, entry, runner_type, operation
):
    monkeypatch.setenv("VANE_RUNNER", runner_type)

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("transaction admission must precede runner initialization")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    monkeypatch.setattr(vane._native, "set_runner_local", forbid_initialization)
    database = str(tmp_path / "transaction-effects.duckdb")
    destination = tmp_path / "rejected.parquet"
    source = "SELECT * FROM range(nextval('seq'))"
    query = {
        "select": source,
        "copy": f"COPY ({source}) TO '{destination}' (FORMAT PARQUET)",
        "insert": f"INSERT INTO marker {source}",
        "ctas": f"CREATE TABLE created AS {source}",
    }[operation]
    with vane.connect(database) as connection:
        connection.execute("CREATE SEQUENCE seq")
        query_relation = connection.sql("SELECT 1") if entry == "relation_query" else None
        connection.begin()
        connection.execute("CREATE TABLE marker(value INTEGER)")
        with pytest.raises(vane.BinderException, match="cannot participate.*explicit transaction"):
            _run_sql_entry(connection, entry, query, query_relation)
        connection.execute("CREATE TABLE followup(value INTEGER)")
        connection.commit()
    assert not destination.exists()
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(database) as inspector:
        assert inspector.execute("SELECT nextval('seq')").fetchone() == (1,)
        assert inspector.table("marker").fetchall() == []
        assert inspector.table("followup").fetchall() == []
        with pytest.raises(vane.CatalogException, match="created"):
            inspector.table("created")


@pytest.mark.parametrize("entry", ["sql", "relation_query", "project"])
def test_ray_relation_schema_binding_checks_transactions_before_effects(monkeypatch, tmp_path, entry):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    database = str(tmp_path / "relation-effects.duckdb")
    with vane.connect(database) as connection:
        connection.execute("CREATE SEQUENCE seq")
        source = connection.sql("SELECT 1 AS value")
        connection.begin()
        connection.execute("CREATE TABLE marker(value INTEGER)")
        with pytest.raises(vane.BinderException, match="cannot participate.*explicit transaction"):
            if entry == "sql":
                connection.sql("SELECT * FROM range(nextval($sequence))", params={"sequence": "seq"})
            elif entry == "relation_query":
                source.query("input", "SELECT * FROM range(nextval('seq'))")
            else:
                source.project("(SELECT count(*) FROM range(nextval('seq'))) AS value")
        connection.commit()
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(database) as inspector:
        assert inspector.execute("SELECT nextval('seq')").fetchone() == (1,)
        assert inspector.table("marker").fetchall() == []


@pytest.mark.parametrize("runner_type", ["local", "ray"])
def test_transaction_precheck_keeps_client_pragma_composition_native(monkeypatch, runner_type):
    monkeypatch.setenv("VANE_RUNNER", runner_type)

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("client PRAGMA queries must stay native")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    monkeypatch.setattr(vane._native, "set_runner_local", forbid_initialization)
    with vane.connect() as connection:
        connection.begin()
        connection.execute("CREATE TABLE marker(value INTEGER)")
        assert connection.sql("PRAGMA table_info('marker')").project("name").fetchall() == [("value",)]
        assert connection.execute("PRAGMA show_tables").fetchall() == [("marker",)]
        connection.commit()


@pytest.mark.parametrize("runner_type", ["local-fast", "local"])
def test_transaction_precheck_keeps_native_read_effects(monkeypatch, runner_type):
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    with vane.connect() as connection:
        connection.begin()
        connection.execute("CREATE SEQUENCE seq")
        assert connection.execute("SELECT nextval('seq')").fetchone() == (1,)
        connection.commit()


@pytest.mark.parametrize("runner_type", ["local", "ray"])
@pytest.mark.parametrize("entry", ["execute", "sql"])
def test_pragma_control_commands_stay_on_the_client(monkeypatch, runner_type, entry):
    monkeypatch.setenv("VANE_RUNNER", runner_type)

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("PRAGMA control commands must not initialize a runner")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    monkeypatch.setattr(vane._native, "set_runner_local", forbid_initialization)
    with vane.connect() as connection:
        for query in ["PRAGMA threads=1", "PRAGMA enable_profiling", "PRAGMA disable_profiling"]:
            result = getattr(connection, entry)(query)
            if result is not None:
                assert result.fetchall() == []


@pytest.mark.parametrize("runner_type", ["local", "ray"])
@pytest.mark.parametrize("entry", ["execute", "sql", "executemany"])
@pytest.mark.parametrize("query", ["PRAGMA show_tables", "PRAGMA database_size", "PRAGMA table_info('client_table')"])
def test_query_pragmas_observe_the_client_catalog(monkeypatch, runner_type, entry, query):
    monkeypatch.setenv("VANE_RUNNER", runner_type)

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("client catalog inspection must not initialize a runner")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    monkeypatch.setattr(vane._native, "set_runner_local", forbid_initialization)
    with vane.connect() as connection:
        connection.execute("CREATE TABLE client_table(value INTEGER)")
        connection.execute("ATTACH ':memory:' AS client_catalog")
        result = connection.executemany(query, [[]]) if entry == "executemany" else getattr(connection, entry)(query)
        rows = result.fetchall()
        if "show_tables" in query:
            assert rows == [("client_table",)]
        elif "database_size" in query:
            assert "client_catalog" in {row[0] for row in rows}
        else:
            assert rows == [(0, "value", "INTEGER", False, None, False)]


@pytest.mark.parametrize("derive", ["filter", "project", "order", "persisted_view"])
def test_pragma_query_origin_survives_relation_composition(monkeypatch, tmp_path, derive):
    monkeypatch.setenv("VANE_RUNNER", "ray")

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("a derived PRAGMA query must still inspect its client catalog")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    database = str(tmp_path / "catalog.duckdb")
    with vane.connect(database) as connection:
        connection.execute("CREATE TABLE client_table(value INTEGER)")
        relation = connection.sql("PRAGMA show_tables")
        if derive == "filter":
            result = relation.filter("name = 'client_table'")
        elif derive == "project":
            result = relation.project("name")
        elif derive == "order":
            result = relation.order("name")
        else:
            relation.filter("name = 'client_table'").create_view("catalog_view")
            result = connection.sql("SELECT name FROM catalog_view")
        assert result.fetchall() == [("client_table",)]
    if derive == "persisted_view":
        # Reopening exercises the stored query node's serialization, not only Copy().
        with vane.connect(database) as connection:
            assert connection.sql("SELECT name FROM catalog_view").fetchall() == [("client_table",)]


@pytest.mark.parametrize("entry", ["sql", "relation"])
def test_runner_write_cannot_make_client_pragma_queries_run_remotely(monkeypatch, tmp_path, entry):
    monkeypatch.setenv("VANE_RUNNER", "ray")

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("a client catalog query cannot be embedded in a runner write")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    destination = tmp_path / "catalog.parquet"
    with vane.connect() as connection:
        connection.execute("CREATE TABLE client_table(value INTEGER)")
        relation = connection.sql("PRAGMA show_tables")
        with pytest.raises(vane.NotImplementedException, match="client connection quer"):
            if entry == "sql":
                relation.create_view("catalog_view")
                connection.sql(f"COPY catalog_view TO '{destination}' (FORMAT PARQUET)")
            else:
                relation.write_parquet(str(destination))
    assert not destination.exists()


@pytest.mark.parametrize("runner_type", ["local", "ray"])
@pytest.mark.parametrize("entry", ["execute", "sql"])
def test_native_controls_can_disable_query_verification(monkeypatch, tmp_path, runner_type, entry):
    monkeypatch.setenv("VANE_RUNNER", runner_type)

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("verification rejection and native controls must precede runner initialization")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    monkeypatch.setattr(vane._native, "set_runner_local", forbid_initialization)
    database = str(tmp_path / "verification.duckdb")
    with vane.connect(database) as connection:
        connection.execute("CREATE SEQUENCE seq")
        result = getattr(connection, entry)("PRAGMA enable_verification")
        if result is not None:
            assert result.fetchall() == []
        connection.execute("SET threads=1")
        connection.execute("CREATE TABLE client_table(value INTEGER)")
        assert connection.sql("PRAGMA show_tables").fetchall() == [("client_table",)]
        # Native verification must not execute this SELECT before admission rejects it.
        with pytest.raises(vane.NotImplementedException, match="query verification"):
            getattr(connection, entry)("SELECT nextval('seq')").fetchall()
        destination = tmp_path / "verified.parquet"
        with pytest.raises(vane.NotImplementedException, match="query verification"):
            getattr(connection, entry)(f"COPY (SELECT 1) TO '{destination}' (FORMAT PARQUET)")
        assert not destination.exists()
        result = getattr(connection, entry)("PRAGMA disable_verification")
        if result is not None:
            assert result.fetchall() == []
        runner = RecordingRunner()
        monkeypatch.setattr(vane._native, "set_runner_ray", lambda *_args, **_kwargs: runner)
        assert connection.execute("SELECT 42::BIGINT AS value").fetchall() == [(42,)]
        assert len(runner.reads) == (1 if runner_type == "ray" else 0)
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(database) as inspector:
        assert inspector.execute("SELECT nextval('seq')").fetchone() == (1,)


@pytest.mark.parametrize("entry", ["execute", "sql", "executemany", "relation"])
@pytest.mark.parametrize("sequence", ["'seq'", "seq_name", "$sequence"])
def test_ray_rejects_database_modifying_reads_before_execution(monkeypatch, tmp_path, entry, sequence):
    monkeypatch.setenv("VANE_RUNNER", "ray")

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("a database-modifying read must fail before runner initialization")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    database = str(tmp_path / "sequence.duckdb")
    with vane.connect(database) as connection:
        connection.execute("CREATE SEQUENCE seq")
        query = f"SELECT nextval({sequence}) AS value FROM (VALUES ('seq')) t(seq_name)"
        params = {"sequence": "seq"} if sequence == "$sequence" else {}
        # Nonconstant sequence names are rejected by the binder itself.
        error = "requires a constant sequence" if sequence == "seq_name" else "database-modifying expressions"
        with pytest.raises(vane.NotImplementedException, match=error):
            if entry == "execute":
                connection.execute(query, params).fetchall()
            elif entry == "executemany":
                connection.executemany(query, [params]).fetchall()
            else:
                result = connection.sql(query, params=params)
                if entry == "relation":
                    result = result.filter("value > 0")
                result.fetchall()
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(database) as inspector:
        assert inspector.execute("SELECT nextval('seq')").fetchone() == (1,)


@pytest.mark.parametrize("operation", ["copy", "insert_default", "ctas"])
def test_ray_writes_reject_untransportable_expression_effects(monkeypatch, tmp_path, operation):
    monkeypatch.setenv("VANE_RUNNER", "ray")

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("a write must not transport database-modifying expressions")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    with vane.connect() as connection:
        connection.execute("CREATE SEQUENCE seq")
        connection.execute("CREATE TABLE target(value BIGINT DEFAULT nextval('seq'))")
        query = {
            "copy": f"COPY (SELECT nextval('seq')) TO '{tmp_path / 'sequence.parquet'}' (FORMAT PARQUET)",
            "insert_default": "INSERT INTO target DEFAULT VALUES",
            "ctas": "CREATE TABLE created AS SELECT nextval('seq') AS value",
        }[operation]
        with pytest.raises(vane.NotImplementedException, match="database-modifying expressions"):
            connection.execute(query)


@pytest.mark.parametrize("entry", ["execute", "sql", "executemany", "relation", "relation_query"])
@pytest.mark.parametrize("operation", ["insert", "update", "merge"])
@pytest.mark.parametrize("modifies_database", [False, True])
def test_write_constraint_effects_are_checked_before_runner_initialization(
    monkeypatch, tmp_path, entry, operation, modifies_database
):
    runner = RecordingRunner()
    install_runner(monkeypatch, runner)
    if modifies_database:

        def forbid_initialization(*_args, **_kwargs):
            raise AssertionError("a database-modifying CHECK must fail before runner initialization")

        monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    database = str(tmp_path / "constraints.duckdb")
    with vane.connect(database) as connection:
        connection.execute("CREATE SEQUENCE seq")
        expression = "value > 0 AND nextval('seq') > 0" if modifies_database else "value > 0"
        connection.execute(f"CREATE TABLE target(value BIGINT NOT NULL CHECK({expression}))")
        query = {
            "insert": "INSERT INTO target VALUES (1)",
            "update": "UPDATE target SET value=1",
            "merge": "MERGE INTO target t USING (SELECT 1::BIGINT AS value) s ON t.value=s.value "
            "WHEN MATCHED THEN UPDATE SET value=s.value WHEN NOT MATCHED THEN INSERT VALUES (s.value)",
        }[operation]

        def write():
            if entry == "execute":
                connection.execute(query)
            elif entry == "sql":
                connection.sql(query)
            elif entry == "executemany":
                connection.executemany(query, [[]])
            elif entry == "relation_query":
                connection.sql("SELECT 1 AS unused").query("source_view", query)
            elif operation == "insert":
                connection.sql("SELECT 1::BIGINT AS value").insert_into("target")
            elif operation == "update":
                connection.table("target").update({"value": vane.ConstantExpression(1)})
            else:
                connection.sql("SELECT 1::BIGINT AS value").merge_into(
                    "target",
                    "target.value = source.value",
                    [
                        "WHEN MATCHED THEN UPDATE SET value=source.value",
                        "WHEN NOT MATCHED THEN INSERT VALUES (source.value)",
                    ],
                )

        if modifies_database:
            with pytest.raises(vane.NotImplementedException, match="database-modifying expressions"):
                write()
            assert runner.writes == []
        else:
            write()
            assert len(runner.writes) == 1
        assert runner.reads == []
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(database) as inspector:
        assert inspector.execute("SELECT nextval('seq')").fetchone() == (1,)
        assert inspector.execute("SELECT * FROM target").fetchall() == []


@pytest.mark.parametrize("entry", ["execute", "relation"])
@pytest.mark.parametrize("operation", ["insert", "update", "merge"])
@pytest.mark.parametrize("expression", ["value + 1", "nextval('seq')"])
def test_runner_writes_reject_generated_target_columns(monkeypatch, tmp_path, entry, operation, expression):
    monkeypatch.setenv("VANE_RUNNER", "ray")

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("generated target columns must fail before runner initialization")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    database = str(tmp_path / "generated.duckdb")
    with vane.connect(database) as connection:
        connection.execute("CREATE SEQUENCE seq")
        connection.execute(f"CREATE TABLE target(value BIGINT, derived BIGINT GENERATED ALWAYS AS ({expression}))")
        with pytest.raises(vane.NotImplementedException, match="generated target columns"):
            if entry == "execute":
                connection.execute(
                    {
                        "insert": "INSERT INTO target VALUES (1)",
                        "update": "UPDATE target SET value=1",
                        "merge": "MERGE INTO target t USING (SELECT 1::BIGINT AS value) s ON t.value=s.value "
                        "WHEN MATCHED THEN UPDATE SET value=s.value WHEN NOT MATCHED THEN INSERT VALUES (s.value)",
                    }[operation]
                )
            elif operation == "insert":
                connection.sql("SELECT 1::BIGINT AS value").insert_into("target")
            elif operation == "update":
                connection.table("target").update({"value": vane.ConstantExpression(1)})
            else:
                connection.sql("SELECT 1::BIGINT AS value").merge_into(
                    "target",
                    "target.value = source.value",
                    [
                        "WHEN MATCHED THEN UPDATE SET value=source.value",
                        "WHEN NOT MATCHED THEN INSERT VALUES (source.value)",
                    ],
                )
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(database) as inspector:
        assert inspector.execute("SELECT nextval('seq')").fetchone() == (1,)
        assert inspector.execute("SELECT value FROM target").fetchall() == []


def test_local_fast_keeps_native_generated_target_writes(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as connection:
        connection.execute("CREATE TABLE target(value BIGINT, derived BIGINT GENERATED ALWAYS AS (value + 1))")
        connection.sql("SELECT 1::BIGINT AS value").insert_into("target")
        connection.execute("UPDATE target SET value=2")
        assert connection.execute("SELECT * FROM target").fetchall() == [(2, 3)]


_CLIENT_CONTEXT_EXPRESSIONS = [
    "current_query()",
    "list_transform([1], lambda x: current_query())",
    "list_transform([1], lambda x: list_transform([2], lambda y: current_query()))",
    "txid_current()",
    "current_query_id()",
    "current_transaction_id()",
    "current_connection_id()",
    "currval('seq')",
    "setseed(0.25)",
    "write_log('runner guard')",
    "parse_duckdb_log_message('QueryLog', 'runner guard')",
    "current_schema()",
    "current_database()",
    "current_catalog()",
    "current_schemas(true)",
    "in_search_path('memory', 'main')",
    "current_setting('threads')",
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
    "age(TIMESTAMP '2026-09-10 11:00:00')",
    "age(TIMESTAMPTZ '2026-09-10 11:00:00+00')",
]


@pytest.mark.parametrize("entry", ["execute", "sql", "executemany", "relation"])
@pytest.mark.parametrize("expression", _CLIENT_CONTEXT_EXPRESSIONS)
def test_runner_reads_reject_client_context_functions_before_initialization(monkeypatch, entry, expression):
    monkeypatch.setenv("VANE_RUNNER", "ray")

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("client-context functions must fail before runner initialization")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    with vane.connect() as connection:
        connection.execute("CREATE SEQUENCE seq")
        query = f"SELECT {expression} AS value"
        with pytest.raises(vane.NotImplementedException, match="client-context function"):
            if entry == "executemany":
                connection.executemany(query, [[]])
            elif entry == "relation":
                connection.sql(query).project("CAST(value AS VARCHAR) AS value").fetchall()
            else:
                getattr(connection, entry)(query).fetchall()


@pytest.mark.parametrize(
    "expression",
    [
        "current_query()",
        "txid_current()",
        "write_log('runner guard')",
        "parse_duckdb_log_message('QueryLog', 'runner guard')",
    ],
)
@pytest.mark.parametrize("operation", ["copy", "insert", "update", "merge", "ctas", "default", "check"])
def test_runner_writes_reject_client_context_functions(monkeypatch, tmp_path, expression, operation):
    monkeypatch.setenv("VANE_RUNNER", "ray")

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("client-context functions must not reach the write runner")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    with vane.connect() as connection:
        value = f"CAST({expression} AS VARCHAR)"
        definition = "value VARCHAR"
        if operation == "default":
            definition += f" DEFAULT {value}"
        elif operation == "check":
            definition += f" CHECK(length({value}) > 0)"
        connection.execute(f"CREATE TABLE target({definition})")
        query = {
            "copy": f"COPY (SELECT {value}) TO '{tmp_path / 'context.parquet'}' (FORMAT PARQUET)",
            "insert": f"INSERT INTO target SELECT {value}",
            "update": f"UPDATE target SET value={value}",
            "merge": "MERGE INTO target t USING (SELECT 'value' AS value) s ON t.value=s.value "
            f"WHEN NOT MATCHED THEN INSERT VALUES ({value})",
            "ctas": f"CREATE TABLE created AS SELECT {value} AS value",
            "default": "INSERT INTO target DEFAULT VALUES",
            "check": "INSERT INTO target VALUES ('value')",
        }[operation]
        with pytest.raises(vane.NotImplementedException, match="client-context function"):
            connection.execute(query)
    assert not (tmp_path / "context.parquet").exists()


@pytest.fixture
def client_extension_state(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    database = str(tmp_path / "extension_state.duckdb")
    extension_directory = tmp_path / "extensions"
    config = {
        "autoload_known_extensions": "true",
        "autoinstall_known_extensions": "true",
        "custom_extension_repository": "http://127.0.0.1:9",
        "extension_directory": str(extension_directory),
    }
    with vane.connect(database, config=config) as inspector:
        query = "SELECT extension_name, loaded FROM duckdb_extensions() ORDER BY extension_name"
        before = inspector.execute(query).fetchall()
        # httpfs is statically loaded in the default build; Azure exercises
        # a setting whose bind callback would actually enter the autoloader.
        assert not dict(before).get("azure", False)
        yield database, config
        assert inspector.execute(query).fetchall() == before
        assert not extension_directory.exists()


@pytest.mark.parametrize("entry", ["execute", "sql", "parameterized_sql", "executemany", "relation_query"])
@pytest.mark.parametrize(
    "runner_type, operation",
    [("ray", "select"), *[(runner, op) for runner in ["local", "ray"] for op in ["copy", "insert", "ctas"]]],
)
@pytest.mark.parametrize(
    "expression, setting",
    [
        ("current_setting({key})", "s3_region"),
        ("current_setting({key})", "azure_storage_connection_string"),
        ("upper(current_setting({key}))", "azure_storage_connection_string"),
        ("list_transform([1], lambda x: current_setting({key}))", "azure_storage_connection_string"),
    ],
)
def test_runner_rejects_scalar_bind_callbacks_before_extension_autoload(
    monkeypatch, tmp_path, client_extension_state, entry, runner_type, operation, expression, setting
):
    database, config = client_extension_state
    monkeypatch.setenv("VANE_RUNNER", runner_type)

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("scalar bind callbacks must be rejected before initializing a runner")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    monkeypatch.setattr(vane._native, "set_runner_local", forbid_initialization)
    destination = tmp_path / "rejected.parquet"
    with vane.connect(database, config=config) as connection:
        connection.execute("CREATE TABLE target(value VARCHAR)")
        expression = expression.format(key="$setting" if entry == "parameterized_sql" else f"'{setting}'")
        source = f"SELECT {expression} AS value"
        query = {
            "select": source,
            "copy": f"COPY ({source}) TO '{destination}' (FORMAT PARQUET)",
            "insert": f"INSERT INTO target {source}",
            "ctas": f"CREATE TABLE created AS {source}",
        }[operation]
        with pytest.raises(vane.NotImplementedException, match="client-context function current_setting"):
            if entry == "parameterized_sql":
                result = connection.sql(query, params={"setting": setting})
            else:
                result = _run_sql_entry(connection, entry, query)
            if result is not None:
                result.fetchall()
    assert not destination.exists()


@pytest.mark.parametrize("runner_type", ["local-fast", "local", "ray"])
@pytest.mark.parametrize("factory", ["read", "datasink"])
def test_explicit_plan_factory_rejects_scalar_bind_callbacks_after_macro_replacement(
    monkeypatch, client_extension_state, runner_type, factory
):
    database, config = client_extension_state
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    with vane.connect(database, config=config) as connection:
        connection.execute("CREATE MACRO source_setting(key) AS 'initial'")
        relation = connection.sql("SELECT source_setting('azure_storage_connection_string') AS value")
        if factory == "datasink":
            relation = relation._mark_datasink("scalar-bind-admission")
        connection.execute("CREATE OR REPLACE MACRO source_setting(key) AS current_setting(key)")
        make_plan = getattr(
            vane.ray_cxx.PyLogicalPlan, f"from_duckdb_{'datasink_' if factory == 'datasink' else ''}relation"
        )
        with pytest.raises(ValueError, match="client-context function current_setting"):
            make_plan(relation, None)


@pytest.mark.parametrize("runner_type", ["local-fast", "local"])
def test_native_reads_keep_client_context_functions(monkeypatch, runner_type):
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    with vane.connect() as connection:
        query = "SELECT current_query()"
        assert connection.execute(query).fetchone() == (query,)
        for function in ["txid_current", "current_query_id", "current_transaction_id", "current_connection_id"]:
            assert isinstance(connection.execute(f"SELECT {function}()").fetchone()[0], int)
        connection.execute("CREATE SEQUENCE seq")
        assert connection.execute("SELECT nextval('seq')").fetchone() == (1,)
        assert connection.execute("SELECT currval('seq')").fetchone() == (1,)
        assert connection.execute("SELECT setseed(0.25)").fetchone() == (None,)
        assert connection.execute(
            "SELECT write_log('native guard', return_value := 42, disable_logging := true)"
        ).fetchone() == (42,)
        assert connection.execute("SELECT parse_duckdb_log_message('QueryLog', 'native guard')").fetchone() == (
            {"message": "native guard"},
        )
        assert connection.execute(
            "SELECT CURRENT_TIMESTAMP IS NOT NULL, current_setting('threads') > 0"
        ).fetchone() == (
            True,
            True,
        )


@pytest.mark.parametrize("entry", ["execute", "sql", "executemany", "relation"])
@pytest.mark.parametrize(
    "function",
    [
        "duckdb_settings()",
        "duckdb_variables()",
        "duckdb_prepared_statements()",
        "duckdb_tables()",
        "duckdb_views()",
        "duckdb_columns()",
        "duckdb_schemas()",
        "duckdb_databases()",
        "duckdb_memory()",
        "duckdb_logs()",
        "enable_logging()",
        "checkpoint()",
    ],
)
def test_runner_reads_reject_client_context_table_functions(monkeypatch, entry, function):
    monkeypatch.setenv("VANE_RUNNER", "ray")

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("client-context table functions must fail before runner initialization")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    with vane.connect() as connection:
        query = f"SELECT count(*) AS value FROM {function}"
        with pytest.raises(vane.NotImplementedException, match="client-context table function"):
            if entry == "relation":
                connection.sql(query).project("value").fetchall()
            else:
                _run_sql_entry(connection, entry, query).fetchall()


@pytest.fixture
def client_file_logs(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    database = str(tmp_path / "logs.duckdb")
    log_directory = tmp_path / "logs"
    with vane.connect(database) as connection:
        connection.execute(f"CALL enable_logging(['QueryLog'], storage_path='{log_directory}', storage_normalize=true)")
        connection.execute("SELECT 'client log marker'").fetchall()
        connection.execute("CALL disable_logging()")
        yield database, connection, log_directory


_FILE_LOG_FUNCTIONS = ["duckdb_logs()", "duckdb_logs(denormalized_table=true)", "duckdb_log_contexts()"]


@pytest.mark.parametrize("function", _FILE_LOG_FUNCTIONS)
@pytest.mark.parametrize("runner_type, operation", [("ray", "select"), ("ray", "copy"), ("local", "copy")])
@pytest.mark.parametrize("entry", ["execute", "sql", "parameterized_sql", "relation_query", "relation"])
def test_runner_rejects_file_log_bind_replacement(
    monkeypatch, tmp_path, client_file_logs, function, runner_type, operation, entry
):
    database, _, log_directory = client_file_logs
    monkeypatch.setenv("VANE_RUNNER", runner_type)

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("file log functions must fail before initializing a runner")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    monkeypatch.setattr(vane._native, "set_runner_local", forbid_initialization)
    destination = tmp_path / "rejected.parquet"
    with vane.connect(database) as connection:
        source = f"SELECT count(*) AS value FROM {function}"
        # A local read may bind and flush natively before becoming a write.
        relation = connection.sql(source).project("value") if entry == "relation" and runner_type == "local" else None
        # Disabling logging leaves the buffered storage available for scans.
        # Admission must reject before bind replacement flushes those buffers.
        before = {path.name: path.read_bytes() for path in log_directory.glob("*")}
        query = source if operation == "select" else f"COPY ({source}) TO '{destination}' (FORMAT PARQUET)"
        with pytest.raises(vane.NotImplementedException, match="client-context table function duckdb_log"):
            if entry == "parameterized_sql":
                parameterized = source + " WHERE context_id >= $minimum"
                if operation == "copy":
                    parameterized = f"COPY ({parameterized}) TO '{destination}' (FORMAT PARQUET)"
                result = connection.sql(parameterized, params={"minimum": 0})
            elif entry == "relation":
                if relation is None:
                    relation = connection.sql(source).project("value")
                result = relation if operation == "select" else relation.write_parquet(str(destination))
            else:
                result = _run_sql_entry(connection, entry, query)
            if result is not None:
                result.fetchall()
        assert {path.name: path.read_bytes() for path in log_directory.glob("*")} == before
    assert not destination.exists()


@pytest.mark.parametrize("function", _FILE_LOG_FUNCTIONS)
@pytest.mark.parametrize(
    "runner_type, factory", [("local-fast", "read"), ("local", "read"), ("local-fast", "datasink")]
)
def test_explicit_plan_factory_rejects_file_log_bind_replacement(
    monkeypatch, client_file_logs, function, runner_type, factory
):
    database, _, log_directory = client_file_logs
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    with vane.connect(database) as connection:
        relation = connection.sql(f"SELECT * FROM {function}")
        if factory == "datasink":
            relation = relation._mark_datasink("file-log-admission")
        make_plan = getattr(
            vane.ray_cxx.PyLogicalPlan, f"from_duckdb_{'datasink_' if factory == 'datasink' else ''}relation"
        )
        before = {path.name: path.read_bytes() for path in log_directory.glob("*")}
        with pytest.raises(ValueError, match="client-context table function duckdb_log"):
            make_plan(relation, None)
        assert {path.name: path.read_bytes() for path in log_directory.glob("*")} == before


@pytest.mark.parametrize("runner_type", ["local-fast", "local"])
@pytest.mark.parametrize("function", _FILE_LOG_FUNCTIONS)
def test_native_reads_keep_file_log_bind_replacement(monkeypatch, client_file_logs, runner_type, function):
    database, _, _ = client_file_logs
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    with vane.connect(database) as connection:
        assert connection.execute(f"SELECT count(*) > 0 FROM {function}").fetchone() == (True,)
        assert connection.sql(
            "SELECT message FROM duckdb_logs() WHERE message = $message",
            params={"message": "SELECT 'client log marker'"},
        ).fetchall() == [("SELECT 'client log marker'",)]


@pytest.mark.parametrize("source", ["query('SELECT 42 AS value')", "query_table('portable')"])
def test_runner_keeps_portable_bind_replacements(monkeypatch, source):
    runner = _TransportedPlanRunner()
    install_runner(monkeypatch, runner)
    try:
        with vane.connect() as connection:
            connection.execute("CREATE VIEW portable AS SELECT 42 AS value")
            assert connection.sql(f"SELECT * FROM {source}").fetchall() == [(42,)]
    finally:
        runner.worker.close()


@pytest.mark.parametrize("entry", ["execute", "sql", "executemany", "relation"])
@pytest.mark.parametrize("operation", ["copy", "insert", "ctas"])
@pytest.mark.parametrize(
    "source",
    [
        "SELECT value FROM duckdb_settings() WHERE name = 'threads'",
        "SELECT table_name AS value FROM duckdb_tables()",
    ],
)
def test_runner_writes_reject_client_context_table_functions(monkeypatch, tmp_path, entry, operation, source):
    monkeypatch.setenv("VANE_RUNNER", "ray")

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("writes must not substitute worker settings or catalog state")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    destination = tmp_path / "context.parquet"
    with vane.connect() as connection:
        connection.execute("SET threads=1")
        connection.execute("CREATE TABLE target(value VARCHAR)")
        with pytest.raises(vane.NotImplementedException, match="client-context table function"):
            if entry == "relation":
                relation = connection.sql(source)
                if operation == "copy":
                    relation.write_parquet(str(destination))
                elif operation == "insert":
                    relation.insert_into("target")
                else:
                    relation.create("created")
            else:
                query = {
                    "copy": f"COPY ({source}) TO '{destination}' (FORMAT PARQUET)",
                    "insert": f"INSERT INTO target {source}",
                    "ctas": f"CREATE TABLE created AS {source}",
                }[operation]
                _run_sql_entry(connection, entry, query)
    assert not destination.exists()


@pytest.mark.parametrize("runner_type", ["local-fast", "local"])
def test_native_reads_keep_client_context_table_functions(monkeypatch, runner_type):
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    with vane.connect() as connection:
        connection.execute("SET threads=1")
        connection.execute("CREATE TABLE client_marker(value INTEGER)")
        assert connection.sql("SELECT value FROM duckdb_settings() WHERE name='threads'").fetchone() == ("1",)
        assert connection.execute(
            "SELECT table_name FROM duckdb_tables() WHERE table_name='client_marker'"
        ).fetchall() == [("client_marker",)]


@pytest.mark.parametrize("function", ["duckdb_keywords()", "duckdb_optimizers()"])
def test_runner_reads_keep_static_table_functions(monkeypatch, function):
    install_runner(monkeypatch, _TransportedPlanRunner())
    with vane.connect() as connection:
        assert connection.execute(f"SELECT count(*) > 0 AS value FROM {function}").fetchone() == (True,)


def test_local_fast_executemany_reuses_the_bound_native_plan(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as connection:
        # Native Prepare uses a read-only transaction in auto-commit mode.
        # Bind-time sequence changes require an existing write transaction.
        connection.begin()
        connection.execute("CREATE SEQUENCE seq")
        connection.execute("CREATE TABLE target(value BIGINT)")
        # Native table-function arguments are evaluated at bind time. Rebinding
        # each parameter set would advance the sequence and insert 1+2+3 rows.
        connection.executemany("INSERT INTO target SELECT range FROM range(nextval('seq'))", [[], [], []])
        assert connection.execute("SELECT value FROM target").fetchall() == [(0,), (0,), (0,)]
        assert connection.execute("SELECT nextval('seq')").fetchone() == (2,)
        connection.commit()


@pytest.mark.parametrize("operation", ["insert", "ctas"])
def test_local_fast_writes_keep_client_query_text(monkeypatch, operation):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as connection:
        if operation == "insert":
            connection.execute("CREATE TABLE target(value VARCHAR)")
            query = "INSERT INTO target SELECT current_query()"
        else:
            query = "CREATE TABLE target AS SELECT current_query() AS value"
        assert connection.execute(query).fetchone() == (1,)
        assert connection.execute("SELECT value FROM target").fetchone() == (query,)


@pytest.mark.parametrize("entry", ["execute", "relation"])
@pytest.mark.parametrize("timestamp_type", ["TIMESTAMP", "TIMESTAMPTZ"])
def test_runner_reads_keep_binary_age_with_explicit_operands(monkeypatch, entry, timestamp_type):
    runner = _TransportedPlanRunner()
    install_runner(monkeypatch, runner)
    with vane.connect() as connection:
        query = f"SELECT epoch(age({timestamp_type} '2026-09-10 12:00:00+00', "
        query += f"{timestamp_type} '2026-09-08 12:00:00+00')) AS value"
        if entry == "execute":
            result = connection.execute(query)
        else:
            result = connection.sql(query).project("value")
        assert result.fetchall() == [(172800.0,)]


def test_runner_reads_keep_bound_variable_values_and_pure_functions(monkeypatch):
    runner = _TransportedPlanRunner()
    install_runner(monkeypatch, runner)
    with vane.connect() as connection:
        connection.execute("SET VARIABLE answer=41")
        assert connection.execute("SELECT getvariable('answer') + 1, upper('portable')").fetchall() == [
            (42, "PORTABLE")
        ]


@pytest.mark.parametrize("runner_type", ["local-fast", "local"])
def test_native_read_policy_keeps_sequence_effects(monkeypatch, runner_type):
    monkeypatch.setenv("VANE_RUNNER", runner_type)
    with vane.connect() as connection:
        connection.execute("CREATE SEQUENCE seq")
        assert connection.execute("SELECT nextval('seq')").fetchone() == (1,)
        assert connection.sql("SELECT nextval($sequence)", params={"sequence": "seq"}).fetchone() == (2,)


def test_local_fast_native_verification_still_executes(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as connection:
        connection.execute("PRAGMA enable_verification")
        try:
            assert connection.execute("SELECT sum(i) FROM range(5) t(i)").fetchall() == [(10,)]
            assert connection.sql("PRAGMA show_tables").fetchall() == []
        finally:
            connection.execute("PRAGMA disable_verification")


def test_local_fast_call_keeps_native_results(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as connection:
        assert connection.execute("CALL range(3)").fetchall() == [(0,), (1,), (2,)]


@pytest.mark.parametrize("entry", ["execute", "sql", "query", "from_query", "relation"])
def test_read_runner_receives_bound_parameters_without_rebinding(monkeypatch, entry):
    runner = _TransportedPlanRunner()
    install_runner(monkeypatch, runner)
    with vane.connect() as connection:

        def forbid_rebinding(*_args, **_kwargs):
            raise AssertionError("a runner must receive a plan that is already bound")

        monkeypatch.setattr(vane.ray_cxx.PyLogicalPlan, "from_duckdb_relation", forbid_rebinding)
        if entry == "relation":
            result = connection.sql("SELECT $value::BIGINT AS value", params={"value": 7}).filter("value > 1")
        elif entry == "execute":
            result = connection.execute("SELECT $value::BIGINT AS value", {"value": 7})
        else:
            result = getattr(connection, entry)("SELECT $value::BIGINT AS value", params={"value": 7})
        assert result.fetchall() == [(7,)]
        assert len(runner.plans) == 1


_WRITES = [
    ("insert_values", "INSERT INTO target VALUES ($value)", {"value": 7}),
    ("insert_select", "INSERT INTO target SELECT $value::INTEGER", {"value": 8}),
    ("update", "UPDATE target SET value=$value", {"value": 9}),
    ("delete", "DELETE FROM target WHERE value=$value", {"value": 10}),
    (
        "merge",
        "MERGE INTO target t USING (SELECT $value::INTEGER AS value) s ON t.value=s.value "
        "WHEN NOT MATCHED THEN INSERT VALUES (s.value)",
        {"value": 11},
    ),
    ("ctas", "CREATE TABLE created AS SELECT $value::INTEGER AS value", {"value": 12}),
    ("copy", "COPY (SELECT $value::INTEGER AS value) TO $path (FORMAT PARQUET)", {"value": 13}),
]


@pytest.mark.parametrize("entry", ["execute", "sql", "executemany", "relation", "relation_query"])
@pytest.mark.parametrize("operation, query, values", _WRITES, ids=[write[0] for write in _WRITES])
def test_write_entrypoints_dispatch_bound_plans_without_local_mutation(
    monkeypatch, tmp_path, entry, operation, query, values
):
    database = str(tmp_path / "source.duckdb")
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(database) as native:
        native.execute("CREATE TABLE target(value INTEGER); INSERT INTO target VALUES (1), (2)")
    runner = RecordingRunner()
    install_runner(monkeypatch, runner)
    values = dict(values)
    path = tmp_path / "result.parquet"
    if operation == "copy":
        values["path"] = str(path)

    def forbid_rebinding(*_args, **_kwargs):
        raise AssertionError("write runner must not bind a Relation again")

    monkeypatch.setattr(vane.ray_cxx.PyLogicalPlan, "from_duckdb_write_relation", forbid_rebinding)
    with vane.connect(database) as connection:
        if entry == "execute":
            assert connection.execute(query, values).fetchall() == [(3,)]
            assert connection.description[0][0] == "Count"
        elif entry == "sql":
            assert connection.sql(query, params=values) is None
        elif entry == "executemany":
            assert connection.executemany(query, [values, values]).fetchall() == [(3,)]
        elif entry == "relation_query":
            for key, value in values.items():
                query = query.replace(f"${key}", str(vane.ConstantExpression(value)))
            assert connection.sql("SELECT 1 AS unused").query("source_view", query) is None
        else:
            source = connection.sql("SELECT $value::INTEGER AS value", params={"value": values["value"]})
            if operation == "insert_values":
                connection.table("target").insert([values["value"]])
            elif operation == "insert_select":
                source.insert_into("target")
            elif operation == "update":
                connection.table("target").update({"value": vane.ConstantExpression(values["value"])})
            elif operation == "delete":
                connection.table("target").delete(condition=vane.ColumnExpression("value") == values["value"])
            elif operation == "merge":
                source.merge_into(
                    "target", "target.value = source.value", ["WHEN NOT MATCHED THEN INSERT VALUES (source.value)"]
                )
            elif operation == "ctas":
                source.create("created")
            else:
                source.write_parquet(str(path))
        assert len(runner.writes) == (2 if entry == "executemany" else 1)
        assert runner.reads == []
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(database) as native:
        assert native.execute("SELECT * FROM target ORDER BY value").fetchall() == [(1,), (2,)]
        assert native.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name='created'").fetchone() == (0,)
    assert not path.exists()


@pytest.mark.parametrize("entry", ["execute", "sql", "relation"])
@pytest.mark.parametrize("hook", ["failure", "close", "replace", "begin", "interrupt"])
def test_runner_initialization_cannot_execute_an_abandoned_bound_query(monkeypatch, tmp_path, entry, hook):
    runner = RecordingRunner()
    install_runner(monkeypatch, runner)
    connection = vane.connect()
    path = tmp_path / "abandoned.parquet"

    def initialize(*_args, **_kwargs):
        if hook == "failure":
            raise RuntimeError("runner initialization failed")
        if hook == "replace":
            connection.execute("SET threads=2")
        else:
            getattr(connection, hook)()
        return runner

    monkeypatch.setattr(vane._native, "set_runner_ray", initialize)
    try:
        with pytest.raises(
            (RuntimeError, vane.InvalidInputException, vane.ConnectionException, vane.InterruptException)
        ):
            if entry == "execute":
                connection.execute("SELECT 7::BIGINT AS value")
            elif entry == "sql":
                connection.sql(f"COPY (SELECT 7 AS value) TO '{path}' (FORMAT PARQUET)")
            else:
                connection.sql("SELECT 7 AS value").write_parquet(str(path))
        assert runner.reads == runner.writes == []
        assert not path.exists()
        if hook != "close":
            if hook == "begin":
                connection.rollback()
            install_runner(monkeypatch, runner)
            # Recovery must start a new query; the abandoned binding transaction
            # must not retain catalog locks or erase this result during cleanup.
            connection.execute("CREATE TABLE recovered(value INTEGER)")
            assert connection.execute("SELECT 1::BIGINT AS value").fetchall() == [(42,)]
    finally:
        connection.close()


@pytest.mark.parametrize(
    "query, message",
    [
        ("INSERT INTO target VALUES (3) RETURNING value", "default Count"),
        ("UPDATE target SET value=3 RETURNING value", "default Count"),
        ("DELETE FROM target RETURNING value", "default Count"),
        ("INSERT INTO target VALUES (3) ON CONFLICT DO NOTHING", "ON CONFLICT"),
        ("INSERT INTO target VALUES (3) ON CONFLICT DO UPDATE SET value=excluded.value", "ON CONFLICT"),
        ("INSERT OR IGNORE INTO target VALUES (3)", "ON CONFLICT"),
        ("INSERT OR REPLACE INTO target VALUES (3)", "ON CONFLICT"),
        ("CREATE TEMP TABLE created AS SELECT 3 AS value", "TEMPORARY"),
        ("CREATE OR REPLACE TABLE created AS SELECT 3 AS value", "OR REPLACE"),
        ("CREATE TABLE IF NOT EXISTS created AS SELECT 3 AS value", "IF NOT EXISTS"),
    ],
)
def test_unsupported_write_semantics_fail_before_runner_initialization(monkeypatch, query, message):
    monkeypatch.setenv("VANE_RUNNER", "ray")

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("unsupported plans must fail admission before initializing a runner")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    with vane.connect() as connection:
        connection.execute("CREATE TABLE target(value INTEGER PRIMARY KEY)")
        with pytest.raises(vane.NotImplementedException, match=message):
            connection.execute(query)
        with pytest.raises(vane.CatalogException, match="created"):
            connection.table("created")


@pytest.mark.parametrize("entry", ["execute", "sql"])
@pytest.mark.parametrize(
    "operation, query, values", [("select", "SELECT $value::INTEGER AS value", {"value": 7}), *_WRITES]
)
def test_runner_transaction_rejection_preserves_the_client_transaction(
    monkeypatch, tmp_path, entry, operation, query, values
):
    runner = RecordingRunner()
    install_runner(monkeypatch, runner)
    values = dict(values)
    if operation == "copy":
        values["path"] = str(tmp_path / "rejected.parquet")
    with vane.connect() as connection:
        connection.begin()
        connection.execute("CREATE TABLE target(value INTEGER)")
        with pytest.raises(vane.BinderException, match="cannot participate.*explicit transaction"):
            if entry == "execute":
                connection.execute(query, values)
            else:
                result = connection.sql(query, params=values)
                if result is not None:
                    result.fetchall()
        # Binding rejection must neither abort nor commit the transaction.
        connection.execute("CREATE TABLE still_active(value INTEGER)")
        connection.rollback()
        for table in ["target", "still_active"]:
            with pytest.raises(vane.CatalogException, match=table):
                connection.table(table)
        assert runner.reads == runner.writes == []


@pytest.mark.parametrize("entry", ["execute", "relation"])
def test_read_only_write_target_fails_before_runner_initialization(monkeypatch, tmp_path, entry):
    database = str(tmp_path / "readonly.duckdb")
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(database) as connection:
        connection.execute("CREATE TABLE target(value INTEGER)")
    monkeypatch.setenv("VANE_RUNNER", "ray")

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("read-only writes must fail before initializing Ray")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    with vane.connect(database, read_only=True) as connection:
        with pytest.raises(vane.InvalidInputException, match="read-only"):
            if entry == "execute":
                connection.execute("INSERT INTO target VALUES (3)")
            else:
                connection.sql("SELECT 3 AS value").insert_into("target")


@pytest.mark.parametrize("entry", ["execute", "sql", "executemany", "relation_query"])
@pytest.mark.parametrize("clause", ["WITH (location={})", "PARTITIONED BY ({})", "SORTED BY ({})"])
@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("current_query()", "client-context function"),
        ("current_date", "client-context function"),
        ("concat('prefix:', current_query())", "client-context function"),
        ("getvariable(current_query())", "client-context function"),
        ("CASE WHEN TRUE THEN 'static' ELSE current_query() END", "client-context function"),
        ("hidden_context('prefix:')", "client-context function"),
        ("nextval('seq')", "database-modifying expressions"),
        ("hidden_sequence()", "database-modifying expressions"),
        ("(SELECT nextval('seq'))", "metadata does not support subqueries"),
        ("hidden_subquery()", "metadata does not support subqueries"),
    ],
)
def test_ctas_metadata_rejects_effects_before_runner_initialization(
    monkeypatch, tmp_path, entry, clause, expression, message
):
    monkeypatch.setenv("VANE_RUNNER", "ray")

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("CTAS metadata effects must fail before runner initialization")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    database = str(tmp_path / "metadata.duckdb")
    with vane.connect(database) as connection:
        connection.execute("CREATE SEQUENCE seq")
        connection.execute("CREATE MACRO hidden_context(x) AS concat(x, current_query())")
        connection.execute("CREATE MACRO hidden_sequence() AS nextval('seq')")
        connection.execute("CREATE MACRO hidden_subquery() AS (SELECT nextval('seq'))")
        query = f"CREATE TABLE created {clause.format(expression)} AS SELECT 7 AS value"
        with pytest.raises(vane.NotImplementedException, match=message):
            _run_sql_entry(connection, entry, query)
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect(database) as inspector:
        assert inspector.execute("SELECT nextval('seq')").fetchone() == (1,)
        assert inspector.execute("SELECT table_name FROM duckdb_tables()").fetchall() == []


@pytest.mark.parametrize("clause", ["PARTITIONED BY ({})", "SORTED BY ({})"])
@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("bucket(nextval('seq'), value)", "database-modifying expressions"),
        ("bucket(current_query(), value)", "client-context function"),
        ("bucket(hidden_context(value), value)", "client-context function"),
        ("bucket((SELECT nextval('seq')), value)", "metadata does not support subqueries"),
        ("bucket(unregistered(value), value)", "unregistered.*does not exist"),
        ("list_transform([value], lambda x: x + 1)", "metadata does not support lambda expressions"),
    ],
)
def test_ctas_transform_arguments_require_validated_sql_expressions(monkeypatch, clause, expression, message):
    monkeypatch.setenv("VANE_RUNNER", "ray")

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("invalid CTAS transform arguments must not reach the runner")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    with vane.connect() as connection:
        connection.execute("CREATE SEQUENCE seq")
        connection.execute("CREATE MACRO hidden_context(x) AS concat(x, current_query())")
        query = f"CREATE TABLE created {clause.format(expression)} AS SELECT 7 AS value"
        with pytest.raises((vane.NotImplementedException, vane.CatalogException), match=message):
            connection.execute(query)


@pytest.mark.parametrize("metadata", ["properties", "partition_by"])
@pytest.mark.parametrize("expression", ["current_query()", "nextval('seq')", "hidden_context()"])
def test_relation_create_metadata_uses_the_same_effect_checks(monkeypatch, metadata, expression):
    monkeypatch.setenv("VANE_RUNNER", "ray")

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("Relation CTAS metadata must fail before runner initialization")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    with vane.connect() as connection:
        connection.execute("CREATE SEQUENCE seq")
        connection.execute("CREATE MACRO hidden_context() AS current_query()")
        expression = vane.SQLExpression(expression)
        arguments = {metadata: {"location": expression} if metadata == "properties" else [expression]}
        with pytest.raises(
            vane.NotImplementedException, match="client-context function|database-modifying expressions"
        ):
            connection.sql("SELECT 7 AS value").create("created", **arguments)


@pytest.mark.parametrize(
    "metadata",
    ["WITH (location=$setting)", "PARTITIONED BY (bucket($setting, value))", "SORTED BY (value + $setting)"],
)
def test_ctas_metadata_captures_parameters_in_transported_plan(monkeypatch, metadata):
    runner = RecordingRunner()
    install_runner(monkeypatch, runner)
    value = "s3://warehouse/captured" if metadata.startswith("WITH") else 29
    query = f"CREATE TABLE created {metadata} AS SELECT $value::INTEGER AS value"
    with vane.connect() as connection:
        connection.execute(query, {"setting": value, "value": 7})
        payload = runner.writes[0].__getstate__()[1]
        # CreateInfo retains the original SQL for diagnostics; executable metadata
        # must contain typed constants instead of unbound parameter identifiers.
        assert b"setting" not in payload.replace(query.encode(), b"")
        assert runner.writes[0].to_physical_plan(connection) is not None


@pytest.mark.parametrize(
    "metadata",
    [
        "WITH (location=concat('s3://warehouse/', getvariable('setting')))",
        "WITH (location=metadata_value('table'))",
        "WITH (location=coalesce(NULL, 's3://warehouse/table'))",
        "PARTITIONED BY (bucket(getvariable('setting'), value))",
        "PARTITIONED BY (metadata_value(value))",
        "SORTED BY (value + getvariable('setting'))",
        "SORTED BY (metadata_value(value))",
        "SORTED BY (age(TIMESTAMP '2025-01-01', TIMESTAMP '2020-01-01'))",
    ],
)
def test_ctas_metadata_captures_client_bindings_and_keeps_pure_expressions(monkeypatch, metadata):
    runner = RecordingRunner()
    install_runner(monkeypatch, runner)
    query = f"CREATE TABLE created(value) {metadata} AS SELECT 7 AS source_name"
    with vane.connect() as connection:
        connection.execute("SET VARIABLE setting=29")
        connection.execute("CREATE MACRO metadata_value(x) AS x || getvariable('setting')")
        connection.execute(query)
        payload = runner.writes[0].__getstate__()[1].replace(query.encode(), b"")
        assert b"getvariable" not in payload
        assert b"metadata_value" not in payload
        # The planning connection has neither the client's macro nor variable.
        with vane.connect() as driver:
            assert runner.writes[0].to_physical_plan(driver) is not None


def test_local_fast_sql_writes_keep_native_transactions(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")

    def forbid_initialization(*_args, **_kwargs):
        raise AssertionError("native execution must not initialize Ray")

    monkeypatch.setattr(vane._native, "set_runner_ray", forbid_initialization)
    with vane.connect() as connection:
        connection.execute("CREATE TABLE target(value INTEGER DEFAULT 7)")
        connection.begin()
        assert connection.execute("INSERT INTO target DEFAULT VALUES").fetchall() == [(1,)]
        assert connection.execute("UPDATE target SET value=? RETURNING value", [8]).fetchall() == [(8,)]
        connection.rollback()
        assert connection.execute("SELECT count(*) FROM target").fetchone() == (0,)
        connection.sql("INSERT INTO target VALUES (?)", params=[9])
        assert connection.execute("CREATE TABLE created AS SELECT * FROM target").fetchall() == [(1,)]
        path = tmp_path / "native.parquet"
        assert connection.execute("COPY created TO ? (FORMAT PARQUET)", [str(path)]).fetchall() == [(1,)]
        assert connection.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchall() == [(9,)]


@pytest.mark.parametrize("entry", ["execute", "sql", "executemany"])
def test_local_fast_concurrent_writes_on_one_connection_are_serialized(monkeypatch, entry):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    workers, rounds, rows_per_statement = 4, 8, 32768
    start = threading.Barrier(workers)
    query = "INSERT INTO target SELECT ?::INTEGER, i FROM range(?) t(i)"
    with vane.connect() as connection:
        connection.execute("CREATE TABLE target(worker INTEGER, value BIGINT)")

        def insert(worker):
            start.wait(timeout=10)
            for _ in range(rounds):
                params = [worker, rows_per_statement]
                if entry == "sql":
                    connection.sql(query, params=params)
                elif entry == "executemany":
                    connection.executemany(query, [params])
                else:
                    connection.execute(query, params)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(insert, worker) for worker in range(workers)]
            for future in futures:
                future.result(timeout=30)
        assert connection.execute("SELECT worker, count(*) FROM target GROUP BY worker ORDER BY worker").fetchall() == [
            (worker, rounds * rows_per_statement) for worker in range(workers)
        ]
