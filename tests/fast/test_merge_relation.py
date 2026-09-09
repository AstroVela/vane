# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import gc
import pickle

import pytest

import vane

_WHEN_CLAUSES = [
    "WHEN MATCHED THEN UPDATE SET value = source.value",
    "WHEN NOT MATCHED THEN INSERT (id, value) VALUES (source.id, source.value)",
]


def _require_ray_cxx():
    ray_cxx = getattr(vane, "ray_cxx", None)
    if (
        ray_cxx is None
        or not hasattr(ray_cxx, "PyLogicalPlan")
        or not hasattr(ray_cxx.PyLogicalPlan, "from_duckdb_write_relation")
    ):
        pytest.skip("Vane write-relation bindings are not available in this environment")
    return ray_cxx


def _merge_connection(monkeypatch, database):
    with monkeypatch.context() as setup:
        setup.setenv("VANE_RUNNER", "local-fast")
        with vane.connect(str(database)) as connection:
            connection.execute("CREATE TABLE merge_target (id INTEGER PRIMARY KEY, value VARCHAR)")
            connection.execute("INSERT INTO merge_target VALUES (1, 'old'), (3, 'keep')")
            connection.execute("CREATE TABLE merge_source (id INTEGER, value VARCHAR)")
            connection.execute("INSERT INTO merge_source VALUES (1, 'new'), (2, 'inserted')")
    return vane.connect(str(database))


def _merge(source, **kwargs):
    return source.merge_into(
        "merge_target",
        "target.id = source.id",
        _WHEN_CLAUSES,
        **kwargs,
    )


def _install_fake_ray_runner(monkeypatch, run_write):
    class FakeRayRunner:
        def run_write(self, relation, **_kwargs):
            return run_write(relation)

    monkeypatch.setattr(vane._native, "set_runner_ray", lambda *_args, **_kwargs: FakeRayRunner())


def _local_target_rows(monkeypatch, connection):
    # Inspect the coordinator table through the native executor explicitly;
    # changing the environment cannot change this connection's policy.
    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        connection.sql("SELECT * FROM merge_target ORDER BY id"), "inspect-merge-target"
    )
    physical = logical.to_physical_plan(connection)
    result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(connection, physical)
    return [tuple(row.values()) for table in result.partition_payloads for row in table.to_pylist()]


def test_merge_relation_runs_with_explicit_local_fast(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    connection = _merge_connection(monkeypatch, tmp_path / "merge.duckdb")
    try:
        assert _merge(connection.table("merge_source")) is None
        assert connection.execute("SELECT * FROM merge_target ORDER BY id").fetchall() == [
            (1, "new"),
            (2, "inserted"),
            (3, "keep"),
        ]
    finally:
        connection.close()


def test_merge_relation_supports_using_columns(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    connection = _merge_connection(monkeypatch, tmp_path / "merge.duckdb")
    try:
        connection.table("merge_source").merge_into(
            "merge_target",
            ["id"],
            _WHEN_CLAUSES,
        )
        assert connection.execute("SELECT * FROM merge_target ORDER BY id").fetchall() == [
            (1, "new"),
            (2, "inserted"),
            (3, "keep"),
        ]
    finally:
        connection.close()


def test_merge_relation_supports_expression_condition_and_custom_aliases(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    connection = _merge_connection(monkeypatch, tmp_path / "merge.duckdb")
    try:
        condition = vane.ColumnExpression("destination.id") == vane.ColumnExpression("changes.id")
        connection.table("merge_source").merge_into(
            "merge_target",
            condition,
            [
                "WHEN MATCHED THEN UPDATE SET value = changes.value",
                "WHEN NOT MATCHED THEN INSERT (id, value) VALUES (changes.id, changes.value)",
            ],
            target_alias="destination",
            source_alias="changes",
        )
        assert connection.execute("SELECT * FROM merge_target ORDER BY id").fetchall() == [
            (1, "new"),
            (2, "inserted"),
            (3, "keep"),
        ]
    finally:
        connection.close()


def test_merge_relation_separates_line_comment_clauses(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    connection = _merge_connection(monkeypatch, tmp_path / "merge.duckdb")
    try:
        connection.table("merge_source").merge_into(
            "merge_target",
            "target.id = source.id",
            [
                "WHEN MATCHED THEN UPDATE SET value = source.value -- update existing rows",
                "WHEN NOT MATCHED THEN INSERT (id, value) VALUES (source.id, source.value)",
            ],
        )
        assert connection.execute("SELECT * FROM merge_target ORDER BY id").fetchall() == [
            (1, "new"),
            (2, "inserted"),
            (3, "keep"),
        ]
    finally:
        connection.close()


def test_merge_relation_accepts_sql_whitespace_after_when(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    connection = _merge_connection(monkeypatch, tmp_path / "merge.duckdb")
    try:
        connection.table("merge_source").merge_into(
            "merge_target",
            "target.id = source.id",
            [
                "WHEN\tMATCHED THEN UPDATE SET value = source.value",
                "WHEN\nNOT MATCHED THEN INSERT (id, value) VALUES (source.id, source.value)",
            ],
        )
        assert connection.execute("SELECT * FROM merge_target ORDER BY id").fetchall() == [
            (1, "new"),
            (2, "inserted"),
            (3, "keep"),
        ]
    finally:
        connection.close()


def test_merge_relation_accepts_sql_comments_after_when(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    connection = _merge_connection(monkeypatch, tmp_path / "merge.duckdb")
    try:
        connection.table("merge_source").merge_into(
            "merge_target",
            "target.id = source.id",
            [
                "WHEN/* update existing rows */MATCHED THEN UPDATE SET value = source.value",
                "WHEN-- insert missing rows\nNOT MATCHED THEN INSERT (id, value) VALUES (source.id, source.value)",
            ],
        )
        assert connection.execute("SELECT * FROM merge_target ORDER BY id").fetchall() == [
            (1, "new"),
            (2, "inserted"),
            (3, "keep"),
        ]
    finally:
        connection.close()


def test_merge_relation_dispatches_as_write_without_local_execution(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    ray_cxx = _require_ray_cxx()
    logical_plans = []

    def run_write(relation):
        logical_plans.append(relation)
        return {"copy_operation_id": relation.idx(), "rows_copied": 1}

    _install_fake_ray_runner(monkeypatch, run_write)
    connection = _merge_connection(monkeypatch, tmp_path / "merge.duckdb")
    try:
        assert _merge(connection.table("merge_source")) is None
        assert isinstance(logical_plans[0], ray_cxx.PyLogicalPlan)
        assert len(logical_plans) == 1
        assert _local_target_rows(monkeypatch, connection) == [
            (1, "old"),
            (3, "keep"),
        ]
    finally:
        connection.close()


def test_merge_relation_preserves_non_sql_source_operators(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    _require_ray_cxx()
    logical_plans = []

    def run_write(relation):
        logical_plans.append(relation)
        return {"copy_operation_id": relation.idx(), "rows_copied": 1}

    _install_fake_ray_runner(monkeypatch, run_write)
    connection = _merge_connection(monkeypatch, tmp_path / "merge.duckdb")
    try:
        source = connection.table("merge_source").repartition(2, "id")
        _merge(source)
        assert len(logical_plans) == 1
        assert _local_target_rows(monkeypatch, connection) == [
            (1, "old"),
            (3, "keep"),
        ]
    finally:
        connection.close()


def test_merge_relation_failure_never_executes_locally(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "ray")

    def run_write(_relation):
        raise RuntimeError("injected distributed merge failure")

    _install_fake_ray_runner(monkeypatch, run_write)
    connection = _merge_connection(monkeypatch, tmp_path / "merge.duckdb")
    try:
        with pytest.raises(RuntimeError, match="injected distributed merge failure"):
            _merge(connection.table("merge_source"))
        assert _local_target_rows(monkeypatch, connection) == [
            (1, "old"),
            (3, "keep"),
        ]
    finally:
        connection.close()


def test_merge_relation_rejects_explicit_transaction_before_dispatch(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    calls = []
    _install_fake_ray_runner(monkeypatch, calls.append)
    connection = _merge_connection(monkeypatch, tmp_path / "merge.duckdb")
    connection.execute("BEGIN")
    try:
        with pytest.raises(vane.BinderException, match="Runner MERGE_INTO requires DuckDB auto-commit mode"):
            _merge(connection.table("merge_source"))
        assert calls == []
        assert _local_target_rows(monkeypatch, connection) == [
            (1, "old"),
            (3, "keep"),
        ]
    finally:
        connection.execute("ROLLBACK")
        connection.close()


def test_merge_relation_rejects_local_fte_runner(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "local")
    connection = _merge_connection(monkeypatch, tmp_path / "merge.duckdb")
    try:
        with pytest.raises(
            vane.InvalidInputException,
            match="MERGE_INTO requires a ray or local-fast connection",
        ):
            _merge(connection.table("merge_source"))
        assert connection.execute("SELECT * FROM merge_target ORDER BY id").fetchall() == [
            (1, "old"),
            (3, "keep"),
        ]
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("condition", "when_clauses", "kwargs", "message"),
    [
        ("", _WHEN_CLAUSES, {}, "non-empty MERGE condition"),
        ([], _WHEN_CLAUSES, {}, "at least one MERGE USING column"),
        (["id", 2], _WHEN_CLAUSES, {}, "MERGE USING columns as strings"),
        ("target.id = source.id", [], {}, "at least one MERGE WHEN clause"),
        ("target.id = source.id", "WHEN MATCHED THEN DELETE", {}, "sequence of SQL strings"),
        ("target.id = source.id", ["UPDATE SET value = source.value"], {}, "must start with WHEN"),
        ("target.id = source.id", ["WHENEVER MATCHED THEN DELETE"], {}, "must start with WHEN"),
        ("target.id = source.id", _WHEN_CLAUSES, {"source_alias": "target"}, "must be different"),
    ],
)
def test_merge_relation_validates_api_inputs(monkeypatch, condition, when_clauses, kwargs, message, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    connection = _merge_connection(monkeypatch, tmp_path / "merge.duckdb")
    try:
        with pytest.raises(vane.InvalidInputException, match=message):
            connection.table("merge_source").merge_into(
                "merge_target",
                condition,
                when_clauses,
                **kwargs,
            )
    finally:
        connection.close()


def test_merge_relation_rejects_parameters_before_dispatch(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    calls = []
    _install_fake_ray_runner(monkeypatch, calls.append)
    connection = _merge_connection(monkeypatch, tmp_path / "merge.duckdb")
    try:
        with pytest.raises(vane.InvalidInputException, match="does not accept prepared parameters"):
            connection.table("merge_source").merge_into(
                "merge_target",
                "target.id = $merge_id",
                _WHEN_CLAUSES,
            )
        assert calls == []
    finally:
        connection.close()


def test_merge_relation_propagates_parser_and_binding_errors_before_dispatch(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    calls = []
    _install_fake_ray_runner(monkeypatch, calls.append)
    connection = _merge_connection(monkeypatch, tmp_path / "merge.duckdb")
    try:
        with pytest.raises(vane.ParserException):
            connection.table("merge_source").merge_into(
                "merge_target",
                "target.id = source.id",
                ["WHEN MATCHED THEN UPDATE SET"],
            )
        with pytest.raises(vane.CatalogException, match="missing_target"):
            connection.table("merge_source").merge_into(
                "missing_target",
                "target.id = source.id",
                _WHEN_CLAUSES,
            )
        assert calls == []
        assert _local_target_rows(monkeypatch, connection) == [
            (1, "old"),
            (3, "keep"),
        ]
    finally:
        connection.close()


def test_merge_relation_reaches_runner_as_bound_plan(monkeypatch, tmp_path):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    ray_cxx = _require_ray_cxx()
    checks = []

    def run_write(relation):
        assert isinstance(relation, ray_cxx.PyLogicalPlan)
        checks.append(relation)
        return {"copy_operation_id": relation.idx(), "rows_copied": 1}

    _install_fake_ray_runner(monkeypatch, run_write)
    connection = _merge_connection(monkeypatch, tmp_path / "merge.duckdb")
    try:
        _merge(connection.table("merge_source"))
        assert len(checks) == 1
    finally:
        connection.close()


def test_merge_relation_logical_plan_round_trip_does_not_execute_locally(tmp_path, monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    _require_ray_cxx()
    logical_plans = []

    def run_write(relation):
        logical_plans.append(relation)
        return {"copy_operation_id": relation.idx(), "rows_copied": 1}

    _install_fake_ray_runner(monkeypatch, run_write)
    database_path = tmp_path / "merge-round-trip.duckdb"
    connection = _merge_connection(monkeypatch, database_path)
    try:
        _merge(connection.table("merge_source"))
        logical_plan = logical_plans[0]
        snapshot = logical_plan.__getstate__()[3]
        assert snapshot["vane_session"]["id"] == logical_plan.session_id()
        assert snapshot["bootstrap"]["database"] == str(database_path)
        assert _local_target_rows(monkeypatch, connection) == [
            (1, "old"),
            (3, "keep"),
        ]
    finally:
        connection.close()

    transported_plan = pickle.loads(pickle.dumps(logical_plan))
    planning_connection = vane.connect()
    try:
        physical_plan = transported_plan.to_physical_plan(planning_connection)
    finally:
        planning_connection.close()

    assert physical_plan is not None
    del physical_plan
    del transported_plan
    del logical_plan
    gc.collect()

    verification_connection = vane.connect(str(database_path))
    try:
        assert _local_target_rows(monkeypatch, verification_connection) == [
            (1, "old"),
            (3, "keep"),
        ]
    finally:
        verification_connection.close()


def test_merge_relation_rejects_ordinary_duckdb_target_before_backend_mutation(tmp_path, monkeypatch):
    from vane.runners.fte.backends.native import NativeFteWorkerManagerBackend

    monkeypatch.setenv("VANE_RUNNER", "ray")
    ray_cxx = _require_ray_cxx()
    logical_plans = []

    def run_write(relation):
        logical_plans.append(relation)
        return {"copy_operation_id": relation.idx(), "rows_copied": 1}

    _install_fake_ray_runner(monkeypatch, run_write)
    database_path = tmp_path / "native-merge-rejection.duckdb"
    connection = _merge_connection(monkeypatch, database_path)
    try:
        source = connection.sql("SELECT i::INTEGER AS id, 'new' AS value FROM range(2, 3) rows(i)")
        _merge(source)
    finally:
        connection.close()

    transported_plan = pickle.loads(pickle.dumps(logical_plans[0]))
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
        with pytest.raises(ValueError, match="Distributed execution does not support"):
            runner.run_copy_plan(physical_plan)
        assert backend_calls == []
    finally:
        backend.shutdown()

    del runner
    del backend
    del physical_plan
    del transported_plan
    gc.collect()

    verification_connection = vane.connect(str(database_path))
    try:
        assert _local_target_rows(monkeypatch, verification_connection) == [
            (1, "old"),
            (3, "keep"),
        ]
    finally:
        verification_connection.close()


def test_statement_write_api_is_not_public():
    ray_cxx = _require_ray_cxx()
    assert not hasattr(vane, "execute_distributed_write")
    assert not hasattr(ray_cxx.PyLogicalPlan, "from_duckdb_write_statement")
