# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle

import pytest

import vane
from tests.fast.test_distributed_result_consumers import _install_fake_ray_runner
from tests.fast.test_sql_insert_runner import _invoke
from vane.runners.copy_outcome import CopyOutcomeUnknownError, CopyResultUnavailableError

_TARGET = '"target schema"."created table"'


class _CreateRunner:
    def __init__(self, database=None):
        self.database = database
        self.plans = []
        self.outcome = {"copy_operation_id": "ctas-test", "rows_copied": 3}
        self.error = None

    def run_write(self, relation):
        assert relation.type == "CREATE_TABLE_RELATION"
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_write_relation(relation, f"ctas-{len(self.plans)}")
        self.plans.append(pickle.loads(pickle.dumps(plan)))
        if self.error is not None:
            raise self.error
        if self.database is not None:
            # Check the serialized SQL plan's native semantics in a separate
            # write transaction. Distributed target support is tested separately.
            with vane.connect(self.database) as worker:
                physical = plan.to_physical_plan(worker)
                worker.begin()
                worker.execute("UPDATE write_guard SET id = id WHERE false")
                result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(worker, physical)
                worker.commit()
                rows = [row for table in result.partition_payloads for row in table.to_pylist()]
                return {"copy_operation_id": "native-ctas-test", "rows_copied": next(iter(rows[0].values()))}
        return self.outcome


def _database(monkeypatch, tmp_path):
    path = str(tmp_path / "ctas.db")
    with monkeypatch.context() as setup:
        setup.setenv("VANE_RUNNER", "local-fast")
        inspector = vane.connect(path)
    inspector.execute('CREATE SCHEMA "target schema"')
    inspector.execute("CREATE TABLE write_guard(id INTEGER)")
    return path, inspector


def _assert_absent(connection, target=_TARGET):
    with pytest.raises(vane.CatalogException, match="does not exist"):
        connection.table(target)


@pytest.mark.parametrize("method", ["execute", "sql", "query", "from_query", "executemany"])
@pytest.mark.parametrize("selected", ["ray", "", None])
def test_ctas_uses_fixed_runner_without_native_creation(monkeypatch, tmp_path, method, selected):
    path, inspector = _database(monkeypatch, tmp_path)
    runner = _CreateRunner()
    calls = _install_fake_ray_runner(monkeypatch, runner)
    if selected is None:
        monkeypatch.delenv("VANE_RUNNER")
    else:
        monkeypatch.setenv("VANE_RUNNER", selected)
    with inspector, vane.connect(path) as connection:
        assert calls == []
        monkeypatch.setenv("VANE_RUNNER", "local-fast")
        result = _invoke(connection, method, f"CREATE TABLE {_TARGET} AS SELECT ?::BIGINT AS id", [8])
        if method in {"execute", "executemany"}:
            assert result is connection
            assert [column[0] for column in connection.description] == ["Count"]
            assert connection.fetchall() == [(3,)]
            assert connection.fetchall() == []
        else:
            assert result is None
        assert len(runner.plans) == len(calls) == 1
        _assert_absent(inspector)


@pytest.mark.parametrize("method", ["execute", "sql"])
@pytest.mark.parametrize(
    ("definition", "parameters", "columns", "expected"),
    [
        ("AS SELECT ?::BIGINT AS id, ?::VARCHAR AS value", [4, "q'; data"], ["id", "value"], [(4, "q'; data")]),
        ("(renamed) AS SELECT $v::INTEGER AS original", {"v": 7}, ["renamed"], [(7,)]),
        ("AS WITH source AS (SELECT $v::BIGINT AS id) SELECT * FROM source", {"v": 9}, ["id"], [(9,)]),
        ("AS SELECT i AS id FROM range(?) t(i)", [3], ["id"], [(0,), (1,), (2,)]),
        ("AS SELECT ?::VARCHAR AS value", [None], ["value"], [(None,)]),
        ("AS SELECT 1 AS id WITH NO DATA", None, ["id"], []),
    ],
)
def test_ctas_preserves_bound_parameters_columns_and_rows(
    monkeypatch, tmp_path, method, definition, parameters, columns, expected
):
    path, inspector = _database(monkeypatch, tmp_path)
    runner = _CreateRunner(path)
    _install_fake_ray_runner(monkeypatch, runner)
    with inspector, vane.connect(path) as connection:
        result = _invoke(connection, method, f'CREATE TABLE "ctas".{_TARGET} {definition}', parameters)
        if method == "execute":
            assert connection.fetchall() == [(len(expected),)]
        else:
            assert result is None
        table = inspector.table(_TARGET)
        assert table.columns == columns
        assert table.order(columns[0]).fetchall() == expected
        assert len(runner.plans) == 1


def test_ctas_statement_object_and_preceding_statement(monkeypatch, tmp_path):
    path, inspector = _database(monkeypatch, tmp_path)
    runner = _CreateRunner(path)
    _install_fake_ray_runner(monkeypatch, runner)
    with inspector, vane.connect(path) as connection:
        statement = connection.extract_statements(f"CREATE TABLE {_TARGET} AS SELECT ?::BIGINT AS id")[0]
        connection.execute(statement, [13])
        assert connection.fetchall() == [(1,)]
        connection.execute("CREATE TABLE preceding AS SELECT 2 AS id; SET threads=2")
        assert inspector.table(_TARGET).fetchall() == [(13,)]
        assert inspector.table("preceding").fetchall() == [(2,)]
        assert len(runner.plans) == 2


@pytest.mark.parametrize("option", ["PARTITIONED BY (id)", "WITH (location = 'test-location')"])
def test_ctas_keeps_creation_options_in_serialized_plan(monkeypatch, tmp_path, option):
    path, inspector = _database(monkeypatch, tmp_path)
    runner = _CreateRunner(path)
    _install_fake_ray_runner(monkeypatch, runner)
    with inspector, vane.connect(path) as connection:
        with pytest.raises(ValueError, match="(PARTITIONED BY|WITH clause) is not supported"):
            connection.execute(f"CREATE TABLE {_TARGET} {option} AS SELECT 1 AS id")
        assert len(runner.plans) == 1
        _assert_absent(inspector)


@pytest.mark.parametrize("method", ["execute", "sql", "executemany"])
@pytest.mark.parametrize("mode", ["TEMPORARY", "OR REPLACE", "IF NOT EXISTS"])
def test_ctas_rejects_unsupported_creation_modes_before_runner(monkeypatch, tmp_path, method, mode):
    path, inspector = _database(monkeypatch, tmp_path)
    if mode != "TEMPORARY":
        inspector.execute("CREATE TABLE unsupported AS SELECT 9 AS id")
    runner = _CreateRunner()
    calls = _install_fake_ray_runner(monkeypatch, runner)
    query = {
        "TEMPORARY": "CREATE TEMPORARY TABLE unsupported AS SELECT 1 AS id",
        "OR REPLACE": "CREATE OR REPLACE TABLE unsupported AS SELECT 1 AS id",
        "IF NOT EXISTS": "CREATE TABLE IF NOT EXISTS unsupported AS SELECT 1 AS id",
    }[mode]
    with inspector, vane.connect(path) as connection:
        with pytest.raises(vane.NotImplementedException, match="Runner SQL CTAS does not support"):
            _invoke(connection, method, query)
        assert calls == runner.plans == []
        if mode == "TEMPORARY":
            _assert_absent(connection, "unsupported")
        else:
            assert inspector.table("unsupported").fetchall() == [(9,)]


@pytest.mark.parametrize("restriction", ["local", "transaction", "parameter-begin", "parameter-close", "parameters"])
def test_ctas_revalidates_before_runner(monkeypatch, tmp_path, restriction):
    path, inspector = _database(monkeypatch, tmp_path)
    runner = _CreateRunner()
    calls = _install_fake_ray_runner(monkeypatch, runner)
    if restriction == "local":
        monkeypatch.setenv("VANE_RUNNER", "local")
    with inspector, vane.connect(path) as connection:
        parameters = [1]
        if restriction == "transaction":
            connection.begin()
        elif restriction == "parameters":
            parameters = []
        elif restriction.startswith("parameter-"):

            class Parameters(list):
                triggered = False

                def __len__(self):
                    if not self.triggered:
                        self.triggered = True
                        getattr(connection, restriction.removeprefix("parameter-"))()
                    return super().__len__()

            parameters = Parameters(parameters)
        with pytest.raises((vane.InvalidInputException, vane.ConnectionException)):
            connection.execute(f"CREATE TABLE {_TARGET} AS SELECT ?::INTEGER AS id", parameters)
        assert calls == runner.plans == []
        if restriction in {"transaction", "parameter-begin"}:
            connection.rollback()
        _assert_absent(inspector)


@pytest.mark.parametrize("error_kind", ["execution", "unknown", "committed-result"])
def test_ctas_failure_never_creates_locally(monkeypatch, tmp_path, error_kind):
    path, inspector = _database(monkeypatch, tmp_path)
    runner = _CreateRunner()
    _install_fake_ray_runner(monkeypatch, runner)
    if error_kind == "execution":
        runner.error = RuntimeError("injected CTAS failure")
    elif error_kind == "unknown":
        runner.error = CopyOutcomeUnknownError("ctas-test")
    else:
        runner.outcome["rows_copied"] = -1
    with inspector, vane.connect(path) as connection:
        with pytest.raises((RuntimeError, CopyResultUnavailableError)) as raised:
            connection.execute(f"CREATE TABLE {_TARGET} AS SELECT 1 AS id")
        if error_kind != "execution":
            assert raised.value.safe_to_retry is False
        if error_kind == "committed-result":
            assert raised.value.write_state == "committed"
        assert len(runner.plans) == 1
        _assert_absent(inspector)


def test_local_fast_ctas_keeps_native_modes_and_transactions(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as connection:
        monkeypatch.setenv("VANE_RUNNER", "ray")
        connection.begin()
        connection.execute("CREATE TEMPORARY TABLE native_target AS SELECT ? AS value", [1])
        connection.execute("CREATE OR REPLACE TEMPORARY TABLE native_target AS SELECT 2 AS value")
        connection.execute("CREATE TEMPORARY TABLE IF NOT EXISTS native_target AS SELECT 3 AS value")
        assert connection.table("native_target").fetchall() == [(2,)]
        connection.rollback()
        _assert_absent(connection, "native_target")


def test_plain_create_table_keeps_client_catalog_path(monkeypatch):
    runner = _CreateRunner()
    calls = _install_fake_ray_runner(monkeypatch, runner)
    with vane.connect() as connection:
        connection.execute("CREATE TABLE plain_target(id INTEGER)")
        assert connection.table("plain_target").columns == ["id"]
        assert calls == runner.plans == []


@pytest.mark.real_ray
def test_ctas_captures_python_source_before_entering_runner(monkeypatch, ray_local):
    import pandas as pd

    from vane.datasource import _memory

    runner = _CreateRunner()
    _install_fake_ray_runner(monkeypatch, runner)
    source_frame = pd.DataFrame({"id": [3, 4]})
    snapshots = []
    put_partition = _memory._put_memory_partition

    def capture_partition(table):
        snapshots.append(table.to_pylist())
        return put_partition(table)

    monkeypatch.setattr(_memory, "_put_memory_partition", capture_partition)
    with vane.connect() as connection:
        connection.execute("CREATE TABLE source_target AS SELECT id + ? AS value FROM source_frame", [10])
        assert snapshots == [source_frame.to_dict("records")]
        assert runner.plans[0]._memory_source_ref_count_for_test() == 1
        _assert_absent(connection, "source_target")


@pytest.mark.real_ray
def test_real_ray_ctas_rejects_ordinary_catalog_before_creation(monkeypatch, ray_local):
    from vane import runners

    monkeypatch.setenv("VANE_RUNNER", "ray")
    runners.set_runner_ray(noop_if_initialized=True)
    try:
        with vane.connect() as connection:
            with pytest.raises(Exception, match="Distributed pipeline does not support operator type: CREATE_TABLE_AS"):
                connection.execute("CREATE TABLE unsupported_target AS SELECT 1 AS id")
            _assert_absent(connection, "unsupported_target")
    finally:
        vane.teardown_runner()
