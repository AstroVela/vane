# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle

import pytest

import vane
from tests.fast.test_distributed_result_consumers import _install_fake_ray_runner
from vane.runners.copy_outcome import CopyOutcomeUnknownError, CopyResultUnavailableError


class _InsertRunner:
    def __init__(self, database=None):
        self.database = database
        self.plans = []
        self.outcome = {"copy_operation_id": "insert-test", "rows_copied": 3}
        self.error = None

    def run_write(self, relation):
        assert relation.type == "INSERT_RELATION"
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_write_relation(relation, f"insert-{len(self.plans)}")
        restored = pickle.loads(pickle.dumps(plan))
        self.plans.append(restored)
        if self.error is not None:
            raise self.error
        if self.database is not None:
            # This test runner checks bound SQL semantics in the shared native
            # database. execute_native assumes read-only statement properties,
            # so explicitly establish its write transaction on a separate
            # connection. Distributed target rejection is tested separately.
            with vane.connect(self.database) as worker:
                physical = plan.to_physical_plan(worker)
                worker.begin()
                worker.execute(f"UPDATE {_TARGET} SET id = id WHERE false")
                result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(worker, physical)
                worker.commit()
                rows = [row for table in result.partition_payloads for row in table.to_pylist()]
                return {"copy_operation_id": "transported-insert", "rows_copied": next(iter(rows[0].values()))}
        return self.outcome


def _database(monkeypatch, tmp_path):
    path = str(tmp_path / "insert.db")
    with monkeypatch.context() as setup:
        setup.setenv("VANE_RUNNER", "local-fast")
        inspector = vane.connect(path)
    inspector.execute('CREATE SCHEMA "target schema"')
    inspector.execute('CREATE TABLE "target schema"."target table" (id BIGINT DEFAULT 7, value VARCHAR DEFAULT \'d\')')
    return path, inspector


def _invoke(connection, method, query, parameters=None):
    if method in {"execute", "executemany"}:
        return getattr(connection, method)(query, [parameters or []] if method == "executemany" else parameters)
    return getattr(connection, method)(query, params=parameters)


_TARGET = '"target schema"."target table"'


@pytest.mark.parametrize("method", ["execute", "sql", "query", "from_query", "executemany"])
@pytest.mark.parametrize("selected", ["ray", "", None])
def test_sql_insert_uses_fixed_runner_without_local_mutation(monkeypatch, tmp_path, method, selected):
    path, inspector = _database(monkeypatch, tmp_path)
    runner = _InsertRunner()
    calls = _install_fake_ray_runner(monkeypatch, runner)
    if selected is None:
        monkeypatch.delenv("VANE_RUNNER")
    else:
        monkeypatch.setenv("VANE_RUNNER", selected)
    with inspector, vane.connect(path) as connection:
        assert calls == []
        monkeypatch.setenv("VANE_RUNNER", "local-fast")
        result = _invoke(connection, method, f"INSERT INTO {_TARGET} VALUES ($id, $value)", {"id": 4, "value": "v"})
        if method in {"execute", "executemany"}:
            assert result is connection
            assert [column[0] for column in connection.description] == ["Count"]
            assert connection.fetchall() == [(3,)]
            assert connection.fetchall() == []
        else:
            assert result is None
        assert len(runner.plans) == len(calls) == 1
        assert inspector.execute(f"SELECT count(*) FROM {_TARGET}").fetchone() == (0,)


@pytest.mark.parametrize(
    ("suffix", "parameters", "expected"),
    [
        ("VALUES (?, ?)", [2, "q'; INSERT is data"], [(2, "q'; INSERT is data")]),
        ("(value, id) VALUES ($v, $i)", {"v": "v", "i": 4}, [(4, "v")]),
        ("BY NAME SELECT $v AS value, $i AS id", {"v": "n", "i": 5}, [(5, "n")]),
        ("(id) VALUES (?), (DEFAULT)", [9], [(7, "d"), (9, "d")]),
        ("DEFAULT VALUES", None, [(7, "d")]),
        ("SELECT i, 'r' FROM range(?) t(i) WHERE i > 0", [3], [(1, "r"), (2, "r")]),
        ("VALUES (?, ?)", [None, None], [(None, None)]),
        ("SELECT 1, 'empty' WHERE false", None, []),
    ],
)
def test_serialized_insert_preserves_sql_binding(monkeypatch, tmp_path, suffix, parameters, expected):
    path, inspector = _database(monkeypatch, tmp_path)
    runner = _InsertRunner(path)
    _install_fake_ray_runner(monkeypatch, runner)
    with inspector, vane.connect(path) as connection:
        connection.execute(f"INSERT INTO {_TARGET} {suffix}", parameters)
        assert connection.fetchall() == [(len(expected),)]
        assert inspector.execute(f"SELECT * FROM {_TARGET} ORDER BY id").fetchall() == expected
        assert len(runner.plans) == 1


def test_insert_cte_and_executemany_share_transport(monkeypatch, tmp_path):
    path, inspector = _database(monkeypatch, tmp_path)
    runner = _InsertRunner(path)
    _install_fake_ray_runner(monkeypatch, runner)
    with inspector, vane.connect(path) as connection:
        connection.executemany(
            f"WITH source AS (SELECT $id::BIGINT AS id) INSERT INTO {_TARGET} SELECT id, $v FROM source",
            [{"id": 1, "v": "a"}, {"id": 2, "v": "b"}],
        )
        assert connection.fetchall() == [(1,)]
        assert inspector.execute(f"SELECT * FROM {_TARGET} ORDER BY id").fetchall() == [(1, "a"), (2, "b")]
        assert len(runner.plans) == 2


def test_insert_statement_object_keeps_catalog_qualification(monkeypatch, tmp_path):
    path, inspector = _database(monkeypatch, tmp_path)
    runner = _InsertRunner(path)
    _install_fake_ray_runner(monkeypatch, runner)
    with inspector, vane.connect(path) as connection:
        statement = connection.extract_statements(f'INSERT INTO "insert".{_TARGET} VALUES (?, ?)')[0]
        connection.execute(statement, [11, "qualified"])
        assert connection.fetchall() == [(1,)]
        assert inspector.execute(f"SELECT * FROM {_TARGET}").fetchall() == [(11, "qualified")]


@pytest.mark.parametrize("parameters", [[], [1], [1, "v", 3], {"missing": 1}])
def test_insert_rejects_invalid_parameters_before_runner(monkeypatch, tmp_path, parameters):
    path, inspector = _database(monkeypatch, tmp_path)
    runner = _InsertRunner()
    calls = _install_fake_ray_runner(monkeypatch, runner)
    with inspector, vane.connect(path) as connection:
        with pytest.raises(vane.InvalidInputException):
            connection.execute(f"INSERT INTO {_TARGET} VALUES (?, ?)", parameters)
        assert calls == runner.plans == []
        assert inspector.execute(f"SELECT count(*) FROM {_TARGET}").fetchone() == (0,)


@pytest.mark.real_ray
@pytest.mark.parametrize("source_kind", ["select", "cte", "values-default", "aliases", "shadow"])
def test_insert_replacement_scan_survives_logical_serialization(monkeypatch, tmp_path, ray_local, source_kind):
    import pandas as pd

    from vane.datasource import _memory

    path, inspector = _database(monkeypatch, tmp_path)
    runner = _InsertRunner()
    _install_fake_ray_runner(monkeypatch, runner)
    source_frame = pd.DataFrame({"id": [3, 4], "value": ["a", "b"]})
    queries = {
        "select": f"INSERT INTO {_TARGET} SELECT * FROM source_frame",
        "cte": f"WITH source AS (SELECT * FROM source_frame) INSERT INTO {_TARGET} SELECT * FROM source",
        "values-default": f"INSERT INTO {_TARGET} VALUES ((SELECT max(id) FROM source_frame), DEFAULT)",
        "aliases": f"INSERT INTO {_TARGET} SELECT d.x, d.y FROM source_frame AS d(x, y)",
        "shadow": f"INSERT INTO {_TARGET} SELECT * FROM source_frame UNION ALL "
        "SELECT * FROM (WITH source_frame AS (SELECT 9 AS id, 'cte' AS value) SELECT * FROM source_frame)",
    }
    snapshots = []
    put_partition = _memory._put_memory_partition

    def capture_partition(table):
        snapshots.append(table.to_pylist())
        return put_partition(table)

    monkeypatch.setattr(_memory, "_put_memory_partition", capture_partition)
    with inspector, vane.connect(path) as connection:
        connection.execute(queries[source_kind])
        assert connection.fetchall() == [(3,)]
        expected = source_frame[["id"]] if source_kind == "values-default" else source_frame
        if source_kind == "aliases":
            expected = expected.rename(columns={"id": "x", "value": "y"})
        assert snapshots == [expected.to_dict("records")]
        assert runner.plans[0]._memory_source_ref_count_for_test() == 1
        assert inspector.execute(f"SELECT * FROM {_TARGET}").fetchall() == []


@pytest.mark.parametrize("method", ["execute", "sql"])
def test_preceding_insert_uses_write_runner(monkeypatch, tmp_path, method):
    path, inspector = _database(monkeypatch, tmp_path)
    runner = _InsertRunner(path)
    _install_fake_ray_runner(monkeypatch, runner)
    with inspector, vane.connect(path) as connection:
        _invoke(
            connection,
            method,
            f"INSERT INTO {_TARGET} VALUES (1, 'first'); INSERT INTO {_TARGET} VALUES (?, ?)",
            [2, "last"],
        )
        assert len(runner.plans) == 2
        assert inspector.execute(f"SELECT * FROM {_TARGET} ORDER BY id").fetchall() == [(1, "first"), (2, "last")]


@pytest.mark.parametrize("method", ["execute", "sql", "executemany"])
@pytest.mark.parametrize("form", ["returning", "ignore", "replace", "conflict"])
def test_unsupported_insert_forms_fail_before_runner(monkeypatch, tmp_path, method, form):
    path, inspector = _database(monkeypatch, tmp_path)
    runner = _InsertRunner()
    calls = _install_fake_ray_runner(monkeypatch, runner)
    queries = {
        "returning": f"INSERT INTO {_TARGET} VALUES (1, 'v') RETURNING id",
        "ignore": f"INSERT OR IGNORE INTO {_TARGET} VALUES (1, 'v')",
        "replace": f"INSERT OR REPLACE INTO {_TARGET} VALUES (1, 'v')",
        "conflict": f"INSERT INTO {_TARGET} VALUES (1, 'v') ON CONFLICT DO NOTHING",
    }
    with inspector, vane.connect(path) as connection:
        with pytest.raises(vane.NotImplementedException, match="RETURNING or ON CONFLICT"):
            _invoke(connection, method, queries[form])
        assert calls == runner.plans == []
        assert inspector.execute(f"SELECT count(*) FROM {_TARGET}").fetchone() == (0,)


@pytest.mark.parametrize("restriction", ["local", "transaction", "parameter-begin", "parameter-close"])
def test_insert_revalidates_policy_and_transaction_before_runner(monkeypatch, tmp_path, restriction):
    path, inspector = _database(monkeypatch, tmp_path)
    runner = _InsertRunner()
    calls = _install_fake_ray_runner(monkeypatch, runner)
    if restriction == "local":
        monkeypatch.setenv("VANE_RUNNER", "local")
    with inspector, vane.connect(path) as connection:
        parameters = [1, "v"]
        if restriction == "transaction":
            connection.begin()
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
            connection.execute(f"INSERT INTO {_TARGET} VALUES (?, ?)", parameters)
        assert calls == runner.plans == []
        if restriction in {"transaction", "parameter-begin"}:
            connection.rollback()
        assert inspector.execute(f"SELECT count(*) FROM {_TARGET}").fetchone() == (0,)


@pytest.mark.parametrize("error_kind", ["execution", "unknown", "committed-result"])
def test_insert_failure_never_replays_locally(monkeypatch, tmp_path, error_kind):
    path, inspector = _database(monkeypatch, tmp_path)
    runner = _InsertRunner()
    _install_fake_ray_runner(monkeypatch, runner)
    if error_kind == "execution":
        runner.error = RuntimeError("injected INSERT execution failure")
    elif error_kind == "unknown":
        runner.error = CopyOutcomeUnknownError("insert-test")
    else:
        runner.outcome["rows_copied"] = "invalid count"
    with inspector, vane.connect(path) as connection:
        with pytest.raises((RuntimeError, CopyResultUnavailableError)) as raised:
            connection.execute(f"INSERT INTO {_TARGET} VALUES (1, 'v')")
        if error_kind != "execution":
            assert raised.value.safe_to_retry is False
        if error_kind == "committed-result":
            assert raised.value.write_state == "committed"
        assert len(runner.plans) == 1
        assert inspector.execute(f"SELECT count(*) FROM {_TARGET}").fetchone() == (0,)


def test_local_fast_insert_preserves_native_returning_and_transactions(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as connection:
        connection.execute("CREATE TABLE target(id INTEGER PRIMARY KEY, value VARCHAR)")
        monkeypatch.setenv("VANE_RUNNER", "ray")
        connection.begin()
        assert connection.sql("INSERT INTO target VALUES (?, ?) RETURNING *", params=[1, "a"]).fetchall() == [(1, "a")]
        connection.execute("INSERT OR REPLACE INTO target VALUES (1, 'b')")
        connection.rollback()
        assert connection.execute("SELECT * FROM target").fetchall() == []


def test_insert_rejects_ordinary_duckdb_target_before_worker_execution(monkeypatch, tmp_path):
    from vane.runners.fte.backends.native import NativeFteWorkerManagerBackend

    path, inspector = _database(monkeypatch, tmp_path)
    runner = _InsertRunner()
    _install_fake_ray_runner(monkeypatch, runner)
    with inspector, vane.connect(path) as connection:
        plans = []

        def run_write(relation):
            plans.append(vane.ray_cxx.PyLogicalPlan.from_duckdb_write_relation(relation, "unsupported-insert"))
            return runner.outcome

        monkeypatch.setattr(runner, "run_write", run_write)
        connection.execute(f"INSERT INTO {_TARGET} VALUES (1, 'v')")
        physical = plans[0].to_physical_plan(connection)
        backend_calls = []

        def execute_backend(request):
            backend_calls.append(request)
            raise AssertionError("unsupported INSERT must fail before worker execution")

        backend = NativeFteWorkerManagerBackend(execute_fn=execute_backend)
        try:
            with pytest.raises(ValueError, match="Distributed pipeline does not support operator type: INSERT"):
                vane.ray_cxx.DistributedPhysicalPlanRunner(backend).run_copy_plan(physical)
            assert backend_calls == []
            assert inspector.execute(f"SELECT count(*) FROM {_TARGET}").fetchone() == (0,)
        finally:
            backend.shutdown()


@pytest.mark.real_ray
def test_real_ray_insert_rejects_client_table_without_mutation(monkeypatch, ray_local):
    from vane import runners

    monkeypatch.setenv("VANE_RUNNER", "ray")
    runners.set_runner_ray(noop_if_initialized=True)
    try:
        with vane.connect() as connection:
            connection.execute("CREATE TABLE client_target(id BIGINT)")
            with pytest.raises(Exception, match="Table with name client_target does not exist"):
                connection.execute("INSERT INTO client_target VALUES (1)")
            assert connection.execute("UPDATE client_target SET id = id RETURNING id").fetchall() == []
    finally:
        vane.teardown_runner()
