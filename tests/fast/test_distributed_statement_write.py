# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import gc
import pickle

import pytest

import vane

_MERGE_SQL = """
MERGE INTO merge_target AS target
USING merge_source AS source
ON target.id = source.id
WHEN MATCHED THEN UPDATE SET value = source.value
WHEN NOT MATCHED THEN INSERT (id, value) VALUES (source.id, source.value)
"""


def _require_ray_cxx():
    ray_cxx = getattr(vane, "ray_cxx", None)
    if (
        ray_cxx is None
        or not hasattr(ray_cxx, "PyLogicalPlan")
        or not hasattr(ray_cxx.PyLogicalPlan, "from_duckdb_write_statement")
    ):
        pytest.skip("Vane statement-write bindings are not available in this environment")
    return ray_cxx


def _merge_connection(database=None):
    connection = vane.connect() if database is None else vane.connect(str(database))
    connection.execute("CREATE TABLE merge_target (id INTEGER PRIMARY KEY, value VARCHAR)")
    connection.execute("INSERT INTO merge_target VALUES (1, 'old'), (3, 'keep')")
    connection.execute("CREATE TABLE merge_source (id INTEGER, value VARCHAR)")
    connection.execute("INSERT INTO merge_source VALUES (1, 'new'), (2, 'inserted')")
    return connection


def test_execute_distributed_write_rejects_non_ray_before_runner_creation(monkeypatch):
    import vane.runners as runners

    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "local")
    monkeypatch.setattr(
        runners,
        "get_or_create_runner",
        lambda: pytest.fail("unsupported runners must be rejected before runner creation"),
    )

    with pytest.raises(ValueError, match="require the Ray runner"):
        vane.execute_distributed_write(_MERGE_SQL, connection=object())


def test_execute_distributed_write_dispatches_to_ray_runner(monkeypatch):
    import vane.runners as runners

    connection = object()
    expected = {"rows_copied": 2, "extension_write": True}
    calls = []

    class _Runner:
        name = "ray"

        def run_statement_write(self, actual_connection, statement):
            calls.append((actual_connection, statement))
            return expected

    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "ray")
    monkeypatch.setattr(runners, "get_or_create_runner", _Runner)

    assert vane.execute_distributed_write(_MERGE_SQL, connection=connection) == expected
    assert calls == [(connection, _MERGE_SQL)]


def test_execute_distributed_write_propagates_runner_failure_without_local_execution(monkeypatch):
    import vane.runners as runners

    class _Connection:
        execute_calls = 0

        def execute(self, _statement):
            self.execute_calls += 1

    class _Runner:
        name = "ray"

        def run_statement_write(self, _connection, _statement):
            raise RuntimeError("distributed write failed")

    connection = _Connection()
    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "ray")
    monkeypatch.setattr(runners, "get_or_create_runner", _Runner)

    with pytest.raises(RuntimeError, match="distributed write failed"):
        vane.execute_distributed_write(_MERGE_SQL, connection=connection)
    assert connection.execute_calls == 0


def test_execute_distributed_write_rejects_invalid_runner_result(monkeypatch):
    import vane.runners as runners

    class _Runner:
        name = "ray"

        def run_statement_write(self, _connection, _statement):
            return None

    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "ray")
    monkeypatch.setattr(runners, "get_or_create_runner", _Runner)

    with pytest.raises(TypeError, match=r"Runner\.run_statement_write\(\) returned NoneType"):
        vane.execute_distributed_write(_MERGE_SQL, connection=object())


def test_ray_runner_uses_statement_write_factory_and_copy_lifecycle(monkeypatch):
    from vane.runners.ray import runner as runner_module

    connection = object()
    statement = object()
    expected = {"rows_copied": 4}
    calls = []

    class _Plan:
        def session_id(self):
            return "statement-write-session"

    class _LogicalPlan:
        @staticmethod
        def from_duckdb_write_statement(actual_connection, actual_statement, query_id):
            calls.append((actual_connection, actual_statement, query_id))
            return _Plan()

    class _Client:
        def run_copy_plan(self, plan):
            assert isinstance(plan, _Plan)
            return expected

    ray_runner = object.__new__(runner_module.RayRunner)
    monkeypatch.setattr(runner_module, "require_ray_cxx_attr", lambda _name, *, hint: _LogicalPlan)
    monkeypatch.setattr(ray_runner, "_client_for_session", lambda session_id: _Client())

    assert ray_runner.run_statement_write(connection, statement) is expected
    assert len(calls) == 1
    assert calls[0][0:2] == (connection, statement)
    assert calls[0][2]


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        ("", "exactly one statement"),
        ("SELECT 1", "only MERGE INTO"),
        (_MERGE_SQL + "; SELECT 1", "exactly one statement"),
        (_MERGE_SQL.replace("target.id = source.id", "target.id = $merge_id"), "prepared parameters"),
    ],
)
def test_statement_write_factory_rejects_invalid_statements(statement, message):
    ray_cxx = _require_ray_cxx()
    connection = _merge_connection()
    try:
        with pytest.raises(ValueError, match=message):
            ray_cxx.PyLogicalPlan.from_duckdb_write_statement(connection, statement, "invalid-merge")
    finally:
        connection.close()


def test_statement_write_factory_rejects_invalid_input_type():
    ray_cxx = _require_ray_cxx()
    connection = _merge_connection()
    try:
        with pytest.raises(TypeError, match="SQL text or a Vane Statement"):
            ray_cxx.PyLogicalPlan.from_duckdb_write_statement(connection, object(), "invalid-type")
    finally:
        connection.close()


def test_statement_write_factory_propagates_parser_error_without_mutation():
    ray_cxx = _require_ray_cxx()
    connection = _merge_connection()
    try:
        with pytest.raises(ValueError, match='"exception_type":"Parser"'):
            ray_cxx.PyLogicalPlan.from_duckdb_write_statement(connection, "MERGE INTO", "parser-error")
        assert connection.execute("SELECT * FROM merge_target ORDER BY id").fetchall() == [(1, "old"), (3, "keep")]
    finally:
        connection.close()


def test_statement_write_factory_propagates_binding_error_without_mutation():
    ray_cxx = _require_ray_cxx()
    connection = _merge_connection()
    try:
        missing_target_merge = _MERGE_SQL.replace("merge_target", "missing_target")
        with pytest.raises(ValueError, match='"exception_type":"Catalog"'):
            ray_cxx.PyLogicalPlan.from_duckdb_write_statement(connection, missing_target_merge, "binding-error")
        assert connection.execute("SELECT * FROM merge_target ORDER BY id").fetchall() == [(1, "old"), (3, "keep")]
    finally:
        connection.close()


def test_statement_write_factory_rejects_explicit_transaction():
    ray_cxx = _require_ray_cxx()
    connection = _merge_connection()
    connection.execute("BEGIN")
    try:
        with pytest.raises(ValueError, match="cannot participate in an explicit transaction"):
            ray_cxx.PyLogicalPlan.from_duckdb_write_statement(connection, _MERGE_SQL, "transaction-merge")
    finally:
        connection.execute("ROLLBACK")
        connection.close()


def test_merge_statement_logical_plan_round_trip_does_not_execute_locally(tmp_path):
    ray_cxx = _require_ray_cxx()
    database_path = tmp_path / "merge-round-trip.duckdb"
    connection = _merge_connection(database_path)
    try:
        statement = connection.extract_statements(_MERGE_SQL)[0]
        logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_write_statement(
            connection,
            statement,
            "merge-round-trip",
        )
        snapshot = logical_plan.__getstate__()[3]
        assert snapshot["vane_session"]["id"] == logical_plan.session_id()
        assert snapshot["bootstrap"]["database"] == str(database_path)
        assert connection.execute("SELECT * FROM merge_target ORDER BY id").fetchall() == [(1, "old"), (3, "keep")]
    finally:
        connection.close()

    transported_plan = pickle.loads(pickle.dumps(logical_plan))
    planning_connection = vane.connect()
    try:
        physical_plan = transported_plan.to_physical_plan(planning_connection)
    finally:
        planning_connection.close()

    assert physical_plan is not None
    assert logical_plan.session_id()
    del physical_plan
    del transported_plan
    del logical_plan
    gc.collect()

    verification_connection = vane.connect(str(database_path))
    try:
        assert verification_connection.execute("SELECT * FROM merge_target ORDER BY id").fetchall() == [
            (1, "old"),
            (3, "keep"),
        ]
    finally:
        verification_connection.close()


def test_statement_write_rejects_native_merge_before_backend_mutation(tmp_path):
    from vane.runners.fte.backends.native import NativeFteWorkerManagerBackend

    ray_cxx = _require_ray_cxx()
    database_path = tmp_path / "native-merge-rejection.duckdb"
    connection = vane.connect(str(database_path))
    try:
        connection.execute("CREATE TABLE merge_target (id INTEGER PRIMARY KEY, value VARCHAR)")
        connection.execute("INSERT INTO merge_target VALUES (1, 'old')")
        logical_plan = ray_cxx.PyLogicalPlan.from_duckdb_write_statement(
            connection,
            """
            MERGE INTO merge_target AS target
            USING (SELECT i::INTEGER AS id, 'new' AS value FROM range(2, 3) rows(i)) AS source
            ON false
            WHEN NOT MATCHED THEN INSERT (id, value) VALUES (source.id, source.value)
            """,
            "native-merge-rejection",
        )
    finally:
        connection.close()

    transported_plan = pickle.loads(pickle.dumps(logical_plan))
    planning_connection = vane.connect()
    try:
        physical_plan = transported_plan.to_physical_plan(planning_connection)
    finally:
        planning_connection.close()

    backend_calls = []

    def execute_backend(request):
        backend_calls.append(request)
        raise AssertionError("native MERGE must be rejected before worker execution")

    backend = NativeFteWorkerManagerBackend(execute_fn=execute_backend)
    runner = ray_cxx.DistributedPhysicalPlanRunner(backend)
    try:
        with pytest.raises(ValueError, match="does not support operator type: MERGE_INTO"):
            runner.run_copy_plan(physical_plan)
        assert backend_calls == []
    finally:
        backend.shutdown()

    del runner
    del backend
    del physical_plan
    del transported_plan
    del logical_plan
    gc.collect()

    verification_connection = vane.connect(str(database_path))
    try:
        assert verification_connection.execute("SELECT * FROM merge_target").fetchall() == [(1, "old")]
    finally:
        verification_connection.close()
