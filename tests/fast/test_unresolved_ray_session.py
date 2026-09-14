# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Client requests cross a serialization boundary before driver binding."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import cloudpickle
import pyarrow as pa
import pytest

import vane
from vane._ray_connection import RayConnection, RayRelation
from vane._unresolved import DriverPlanReference, UnresolvedExpression
from vane.experimental import connect_driver_session as ray_connect
from vane.runners.ray.unresolved import prepare_request, resolve_plan_reference


class _InProcessDriver:
    def __init__(self):
        self.sessions = {}
        self.requests = []
        self.reads = []
        self.writes = []
        self.kinds = []
        self.closed_streams = 0

    def _client_for_session(self, session_id):
        return self

    def _ensure_session(self, request):
        if request.session not in self.sessions:
            database, read_only, options = request.bootstrap
            connection = (
                self.sessions[request.parent_session].connection.cursor()
                if request.parent_session
                else vane._native._connect_with_runner(
                    "ray", database=database, read_only=read_only, config=dict(options), driver_owned=True
                )
            )
            self.sessions[request.session] = SimpleNamespace(
                connection=connection,
                bootstrap=request.bootstrap,
                operation_lock=threading.Lock(),
                condition=threading.Condition(),
                unresolved_plans={},
            )
        return request.session, request.session_config(), self

    def unresolved_request(self, request, mode):
        request = cloudpickle.loads(cloudpickle.dumps(request))
        self._ensure_session(request)
        self.requests.append((request, mode))
        response = prepare_request(self.sessions[request.session], request, mode)
        if "kind" in response:
            self.kinds.append(response["kind"])
        return cloudpickle.loads(cloudpickle.dumps(response))

    def stream_plan(self, reference):
        assert isinstance(reference, DriverPlanReference)
        session = self.sessions[reference.session]
        logical = resolve_plan_reference(session, reference)
        self.reads.append(logical)
        with vane._native._connect_with_runner("local-fast") as worker:
            physical = logical.to_physical_plan(worker)
            result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(worker, physical)
            try:
                for table in result.partition_payloads:
                    yield SimpleNamespace(partition=lambda value=table: value)
            finally:
                self.closed_streams += 1

    def run_copy_plan(self, reference):
        session = self.sessions[reference.session]
        self.writes.append(resolve_plan_reference(session, reference))
        self._close_plan_after_stream(self, session_id=reference.session, plan_id=reference.query)
        return {"copy_operation_id": reference.query, "rows_copied": 3}

    def _close_plan_after_stream(self, runner, *, session_id, plan_id):
        self.sessions[session_id].unresolved_plans.pop(plan_id, None)

    def close_session(self, session_id):
        session = self.sessions.pop(session_id, None)
        if session is not None:
            session.unresolved_plans.clear()
            session.connection.close()

    def close(self):
        for session in tuple(self.sessions):
            self.close_session(session)


@pytest.fixture
def driver(monkeypatch):
    driver = _InProcessDriver()
    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setattr(vane._native, "set_runner_ray", lambda *_args, **_kwargs: driver)
    try:
        yield driver
    finally:
        driver.close()


def test_local_fast_keeps_native_binding_and_execution(driver, monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as connection:
        assert type(connection) is vane.DuckDBPyConnection
        relation = connection.sql("SELECT ?::BIGINT AS value", params=[2])
        assert type(relation) is vane.DuckDBPyRelation
        with pytest.raises(vane.BinderException):
            relation.project("missing")
        monkeypatch.setenv("VANE_RUNNER", "ray")
        connection.execute("CREATE TABLE t(value BIGINT)")
        relation.insert_into("t")
        connection.begin()
        connection.execute("UPDATE t SET value = value + 1")
        connection.rollback()
        assert connection.table("t").fetchall() == [(2,)]
        path = str(tmp_path / "local.parquet")
        connection.table("t").write_parquet(path)
        assert connection.read_parquet(path).fetchall() == [(2,)]
    assert driver.requests == []


def test_ray_connection_and_relation_construction_do_not_start_runtime(driver):
    with ray_connect() as connection:
        assert type(connection) is RayConnection
        relation = (
            connection.table("created_later")
            .filter(vane.col("value") > 1)
            .select((vane.col("value") + 2).alias("value"))
        )
        assert type(relation) is RayRelation
        assert driver.requests == []
        connection.execute("CREATE VIEW created_later AS SELECT i AS value FROM range(4) t(i)")
        assert relation.order("value").fetchall() == [(4,), (5,)]
        request = next(
            request for request, mode in driver.requests if mode == "execute" and request.plan.kind == "relation"
        )
        assert isinstance(request.plan.inputs[0].arguments[0], UnresolvedExpression)


@pytest.mark.parametrize("method", ["execute", "sql"])
@pytest.mark.parametrize("parameters", [[7], {"value": 7}])
def test_sql_and_relations_use_driver_bindings(driver, method, parameters):
    marker = "?" if isinstance(parameters, list) else "$value"
    with ray_connect() as connection:
        connection.execute("CREATE MACRO increment(x) AS x + 1")
        query = f"SELECT increment(i + {marker}) AS value FROM range(2) t(i) ORDER BY value"
        result = (
            connection.execute(query, parameters) if method == "execute" else connection.sql(query, params=parameters)
        )
        assert result.fetchall() == [(8,), (9,)]
        assert driver.reads
        with pytest.raises(vane.CatalogException):
            connection._result_context.sql("SELECT increment(1)")


def test_driver_session_settings_and_schema_analysis(driver):
    with ray_connect() as connection:
        connection.execute("SET VARIABLE amount = 4")
        connection.execute("SET TimeZone = 'Asia/Shanghai'")
        relation = connection.sql("SELECT i + getvariable('amount') AS value FROM range(2) t(i)")
        assert relation.columns == ["value"]
        assert driver.reads == []
        assert relation.order("value").fetchall() == [(4,), (5,)]
        query = "SELECT current_query() AS query, current_schema() AS schema FROM range(2)"
        assert connection.execute(query).fetchall() == [(query, "main"), (query, "main")]
        result = connection.execute("SELECT TIMESTAMPTZ '2026-01-01 00:00:00+00' FROM range(1)").fetchone()[0]
        assert result.hour == 8
        assert str(result.tzinfo) == "Asia/Shanghai"


def test_driver_catalog_metadata_and_native_client_state_are_separate(driver):
    with ray_connect() as connection:
        connection.execute("CREATE SCHEMA driver_catalog")
        connection.execute("CREATE TABLE driver_catalog.only_here(value INTEGER)")
        assert connection.execute(
            "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'driver_catalog'"
        ).fetchall() == [("only_here",)]
        assert driver.kinds[-1] == "native"
        assert (
            connection._result_context.execute(
                "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'driver_catalog'"
            ).fetchall()
            == []
        )
        connection.begin()
        assert connection.execute("SELECT count(*) FROM duckdb_tables()").fetchone() == (1,)
        connection.rollback()


def test_unresolved_relations_rebind_after_driver_ddl(driver):
    with ray_connect() as connection:
        connection.execute("CREATE VIEW source AS SELECT i AS value FROM range(1) t(i)")
        relation = connection.table("source")
        assert relation.columns == ["value"]
        connection.execute("CREATE OR REPLACE VIEW source AS SELECT i + 9 AS value FROM range(2) t(i)")
        assert relation.order("value").fetchall() == [(9,), (10,)]


def test_parameters_are_captured_without_binding_client_catalog(driver):
    with ray_connect() as connection:
        parameters = {"items": [1, 2]}
        relation = connection.sql("SELECT $items::BIGINT[] AS value FROM range(1)", params=parameters)
        parameters["items"].append(3)
        assert relation.fetchall() == [([1, 2],)]
        assert relation.set_alias("items").alias == "items"


def test_cursor_inherits_runner_and_database_after_environment_change(driver, monkeypatch):
    with ray_connect() as connection:
        connection.execute("CREATE VIEW source AS SELECT i FROM range(2) t(i)")
        monkeypatch.setenv("VANE_RUNNER", "invalid")
        with connection.cursor() as cursor:
            assert isinstance(cursor, RayConnection)
            assert cursor.table("source").order("i").fetchall() == [(0,), (1,)]
        assert connection.table("source").aggregate("count(*)").fetchone() == (2,)
        with pytest.raises(vane.InvalidInputException):
            vane.connect()


def test_standard_ray_connection_keeps_the_client_binding_path(driver):
    with vane.connect() as connection:
        assert type(connection) is vane.DuckDBPyConnection
        assert type(connection.sql("SELECT 1")) is vane.DuckDBPyRelation
    assert driver.requests == []


def test_driver_write_admission_is_shared_by_sql_and_relation(driver, tmp_path):
    with ray_connect() as connection:
        target = str(tmp_path / "dataset.parquet")
        assert connection.execute(f"COPY (SELECT * FROM range(3)) TO '{target}' (FORMAT PARQUET)").fetchall() == [(3,)]
        connection.table_function("range", [3]).write_parquet(target)
        assert len(driver.writes) == 2
        assert connection.sql(f"COPY (SELECT * FROM range(3)) TO '{target}' (FORMAT PARQUET)") is None
        assert len(driver.writes) == 3
        with pytest.raises(vane.NotImplementedException, match="COPY FROM"):
            connection.execute("CREATE TABLE target(i BIGINT)")
            connection.execute(f"COPY target FROM '{target}' (FORMAT PARQUET)")


def test_driver_rejects_unsupported_distribution_without_running_locally(driver):
    with ray_connect() as connection:
        connection.execute("CREATE SEQUENCE s")
        with pytest.raises(vane.NotImplementedException, match="database-modifying"):
            connection.execute("SELECT nextval('s') FROM range(2)")
        assert driver.reads == []
        with pytest.raises(vane.NotImplementedException, match="PREPARE"):
            connection.execute("PREPARE p AS SELECT * FROM range(1)")
        assert connection.execute("SELECT i FROM range(1) t(i)").fetchall() == [(0,)]


def test_partial_results_release_driver_plan_dependencies(driver):
    with ray_connect() as connection:
        relation = connection.table_function("range", [10000])
        assert relation.fetchone() == (0,)
        session = driver.sessions[connection._session]
        assert len(session.unresolved_plans) == 1
        relation.close()
        assert session.unresolved_plans == {}
        assert driver.closed_streams == 1


def test_native_arguments_nested_in_source_objects_can_cross_transport(driver):
    with ray_connect() as connection:
        expression = vane.col("value").cast(vane.sqltypes.BIGINT)
        restored = cloudpickle.loads(cloudpickle.dumps({"expression": expression, "type": vane.sqltypes.BIGINT}))
        relation = connection.from_arrow(pa.table({"value": [1, 2]})).select(restored["expression"])
        assert relation.columns == ["value"]
        assert relation.types == [restored["type"]]


@pytest.mark.real_ray
def test_unresolved_sql_relation_and_copy_on_real_ray(ray_local, monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    try:
        with ray_connect() as connection:
            connection.execute("SET VARIABLE amount = 6")
            connection.execute("CREATE VIEW numbers AS SELECT i FROM range(7) t(i)")
            assert connection.table("numbers").filter(vane.col("i") < 2).order("i").fetchall() == [(0,), (1,)]
            assert connection.sql(
                "SELECT i + getvariable('amount') + ? AS value FROM numbers WHERE i < 2 ORDER BY i", params=[1]
            ).fetchall() == [(7,), (8,)]
            target = str(tmp_path / "ray-dataset.parquet")
            assert connection.execute(f"COPY numbers TO '{target}' (FORMAT PARQUET)").fetchone() == (7,)
            assert connection.read_parquet(target + "/*.parquet").aggregate("count(*)").fetchone() == (7,)
    finally:
        vane.teardown_runner()
