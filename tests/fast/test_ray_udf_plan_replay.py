# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
import uuid

import pytest

import vane
from vane import runners as _runners


def _table_from_native_result(result):
    pa = pytest.importorskip("pyarrow")

    assert isinstance(result, vane.ray_cxx.NativeDistributedTaskResult)
    payloads = list(result.partition_payloads)
    assert payloads
    if len(payloads) == 1:
        return payloads[0]
    return pa.concat_tables(payloads)


def _round_trip_to_fresh_physical_plan(relation):
    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, str(uuid.uuid4()))
    serialized = pickle.dumps(logical)
    restored = pickle.loads(serialized)
    target = vane.connect()
    return target, restored.to_physical_plan(target), serialized


def _execute_fresh_physical_plan(target, physical):
    from vane.execution.udf_subprocess import ensure_local_subprocess_actor_pools_for_plan

    pools, _ = ensure_local_subprocess_actor_pools_for_plan(physical, conn=target)
    try:
        runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
        result = runner.execute_native(target.cursor(), physical, None, None)
        return _table_from_native_result(result)
    finally:
        for pool in pools:
            pool.shutdown(kill=True)


def test_attached_scalar_alias_survives_logical_plan_pickle_to_fresh_connection():
    source = vane.connect()

    def add_one(value):
        return value + 1

    vane.attach_function(
        add_one,
        connection=source,
        alias="fresh_scalar_sql",
        parameters=["INTEGER"],
        return_dtype="INTEGER",
    )
    relation = source.sql("SELECT fresh_scalar_sql(i::INTEGER) AS value FROM range(3) t(i) ORDER BY i")

    target, physical, serialized = _round_trip_to_fresh_physical_plan(relation)
    udf_node = physical.collect_udf_nodes()[0]
    table = _execute_fresh_physical_plan(target, physical)

    assert table.column(0).to_pylist() == [1, 2, 3]
    assert udf_node["payload"]["function_pickle_size_bytes"] > 0
    assert 0 < len(serialized) < 1_000_000


@pytest.mark.parametrize(
    ("corruption", "error"),
    [
        ("payload_version", "unsupported payload_version 999"),
        ("logical_return_type", "payload missing method_return_type or output_schema"),
        ("return_type", "serialized return type 'VARCHAR' does not match payload return type 'INTEGER'"),
    ],
)
def test_physical_udf_deserialize_rejects_corrupt_metadata(monkeypatch, corruption, error):
    source = vane.connect()

    def add_one(value):
        return value + 1

    vane.attach_function(
        add_one,
        connection=source,
        alias="physical_payload_version_sql",
        parameters=["INTEGER"],
        return_dtype="INTEGER",
    )
    relation = source.sql("SELECT physical_payload_version_sql(1::INTEGER) AS value")
    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, str(uuid.uuid4()))
    physical = logical.to_physical_plan(source)

    monkeypatch.setenv("VANE_ENABLE_UDF_TEST_HOOKS", "1")
    monkeypatch.setenv("VANE_TEST_CORRUPT_UDF_PHYSICAL_PAYLOAD", corruption)
    serialized = pickle.dumps(physical)
    monkeypatch.delenv("VANE_TEST_CORRUPT_UDF_PHYSICAL_PAYLOAD")

    restored = pickle.loads(serialized)
    target = vane.connect()
    with pytest.raises(Exception, match=error):
        restored.clone(target)


def test_attached_batch_alias_survives_logical_plan_pickle_to_fresh_connection():
    pa = pytest.importorskip("pyarrow")
    source = vane.connect()

    def add_one_batch(table):
        values = table.column("value").to_pylist()
        return pa.table({"result": [value + 1 for value in values]})

    vane.attach_function(
        add_one_batch,
        connection=source,
        alias="fresh_batch_sql",
        input_names=["value"],
        schema={"result": "INTEGER"},
        parameters=["INTEGER"],
    )
    relation = source.sql("SELECT fresh_batch_sql(i::INTEGER) AS value FROM range(3) t(i) ORDER BY i")

    target, physical, _ = _round_trip_to_fresh_physical_plan(relation)
    table = _execute_fresh_physical_plan(target, physical)

    assert table.column(0).to_pylist() == [1, 2, 3]


def test_replaced_batch_named_arguments_survive_pickle_with_new_dependency():
    pa = pytest.importorskip("pyarrow")
    source = vane.connect()

    def old_batch(table):
        left = table.column("left").to_pylist()
        right = table.column("right").to_pylist()
        return pa.table({"result": [a - b for a, b in zip(left, right, strict=True)]})

    def new_batch(table):
        left = table.column("left").to_pylist()
        right = table.column("right").to_pylist()
        return pa.table({"result": [(a - b) * 10 for a, b in zip(left, right, strict=True)]})

    registration = {
        "connection": source,
        "alias": "fresh_replaced_named_sql",
        "input_names": ["left", "right"],
        "schema": {"result": "INTEGER"},
        "parameters": ["INTEGER", "INTEGER"],
    }
    vane.attach_function(old_batch, **registration)
    vane.attach_function(new_batch, **registration, replace=True)
    relation = source.sql(
        "SELECT fresh_replaced_named_sql(right := 2, left := i::INTEGER) AS value FROM range(3, 6) t(i) ORDER BY i"
    )

    target, physical, _ = _round_trip_to_fresh_physical_plan(relation)
    table = _execute_fresh_physical_plan(target, physical)

    assert table.column(0).to_pylist() == [10, 20, 30]


def test_failed_cpp_catalog_replace_rolls_back_dependency_and_fresh_plan(monkeypatch):
    source = vane.connect()

    @vane.func(return_dtype="INTEGER")
    def old_add(value):
        return value + 1

    @vane.func(return_dtype="INTEGER")
    def new_add(value):
        return value + 100

    alias = "fault_injected_replace_sql"
    registration = {
        "connection": source,
        "alias": alias,
        "parameters": ["INTEGER"],
    }
    vane.attach_function(old_add, **registration)
    monkeypatch.setenv("VANE_ENABLE_UDF_TEST_HOOKS", "1")
    monkeypatch.setenv("VANE_TEST_FAIL_UDF_CATALOG_REGISTRATION", alias)

    with pytest.raises(Exception, match="injected Vane catalog registration failure after dependency swap"):
        vane.attach_function(new_add, **registration, replace=True)

    monkeypatch.delenv("VANE_TEST_FAIL_UDF_CATALOG_REGISTRATION")
    assert source.sql(f"SELECT {alias}(1::INTEGER)").fetchall() == [(2,)]

    relation = source.sql(f"SELECT {alias}(i::INTEGER) AS value FROM range(3) t(i) ORDER BY i")
    target, physical, _ = _round_trip_to_fresh_physical_plan(relation)
    table = _execute_fresh_physical_plan(target, physical)

    assert table.column(0).to_pylist() == [1, 2, 3]


def test_attached_vane_cls_alias_survives_logical_plan_pickle_to_fresh_connection():
    source = vane.connect()

    @vane.cls(actor_number=1, return_dtype="INTEGER")
    class Offset:
        def __init__(self, offset):
            self.offset = offset

        def __call__(self, value):
            return value + self.offset

    vane.attach_function(
        Offset(10),
        connection=source,
        alias="fresh_class_sql",
        parameters=["INTEGER"],
    )
    relation = source.sql("SELECT fresh_class_sql(i::INTEGER) AS value FROM range(3) t(i) ORDER BY i")

    target, physical, _ = _round_trip_to_fresh_physical_plan(relation)
    table = _execute_fresh_physical_plan(target, physical)

    assert table.column(0).to_pylist() == [10, 11, 12]


def test_attached_vane_cls_batch_alias_survives_logical_plan_pickle_to_fresh_connection():
    pa = pytest.importorskip("pyarrow")
    source = vane.connect()

    @vane.cls.batch(actor_number=1, schema={"result": "INTEGER"}, row_preserving=True)
    class BatchOffset:
        def __init__(self, offset):
            self.offset = offset

        def __call__(self, table):
            values = table.column("value").to_pylist()
            return pa.table({"result": [value + self.offset for value in values]})

    vane.attach_function(
        BatchOffset(20),
        connection=source,
        alias="fresh_class_batch_sql",
        input_names=["value"],
        parameters=["INTEGER"],
    )
    relation = source.sql("SELECT fresh_class_batch_sql(i::INTEGER) AS value FROM range(3) t(i) ORDER BY i")

    target, physical, _ = _round_trip_to_fresh_physical_plan(relation)
    table = _execute_fresh_physical_plan(target, physical)

    assert table.column(0).to_pylist() == [20, 21, 22]


def _reset_udf_executor_counters():
    import _vane_duckdb

    _vane_duckdb._reset_udf_executor_debug_counters()


def _udf_executor_counters():
    import _vane_duckdb

    return dict(_vane_duckdb._udf_executor_debug_counters())


def _assert_no_udf_direct_output_conversion():
    counters = _udf_executor_counters()
    assert counters["udf_direct_output_arrow_table_conversion_count"] == 0
    assert counters["udf_distributed_direct_table_rejected_events"] == 0
    return counters


@pytest.fixture(autouse=True)
def _stop_native_udf_dispatcher_after_test():
    yield
    import _vane_duckdb

    _vane_duckdb._shutdown_udf_executor_dispatcher()


def test_execute_native_rejects_ray_scalar_without_registered_query_graph(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    pytest.importorskip("ray")

    monkeypatch.setenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

    con = vane.connect()
    parquet_path = tmp_path / "udf_input.parquet"
    con.execute(
        f"""
        COPY (
            SELECT i::INTEGER AS a
            FROM range(6) tbl(i)
        ) TO '{parquet_path}' (FORMAT PARQUET)
        """
    )

    def plus_one(value):
        return value + 1

    relation = (
        con.sql(
            f"""
            SELECT a
            FROM read_parquet('{parquet_path}')
            WHERE a < 4
            ORDER BY a
            """
        )
        .map(plus_one, return_type=vane.sqltype("INTEGER"), execution_backend="ray_task")
        .project("a, value AS out")
    )
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)

    runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
    with pytest.raises(ValueError, match="requires query_id"):
        runner.execute_native(con.cursor(), plan, None, None)
    del runner, plan, relation
    con.close()


def test_execute_native_subprocess_udf_reports_admission_task_stats(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    import vane.execution.udf_subprocess as subprocess_exec

    subprocess_exec._shutdown_global_task_runtime()

    con = vane.connect()
    con.execute("SET threads=2")
    parquet_path = tmp_path / "subprocess_udf_admission_input.parquet"
    con.execute(
        f"""
        COPY (
            SELECT i::INTEGER AS a
            FROM range(4) tbl(i)
        ) TO '{parquet_path}' (FORMAT PARQUET)
        """
    )

    def double_values(table):
        values = table.column(0).to_pylist()
        return pa.table({"x": [value * 2 for value in values]})

    try:
        relation = con.sql(
            f"""
            SELECT a
            FROM read_parquet('{parquet_path}')
            ORDER BY a
            """
        ).map_batches(
            double_values,
            schema={"x": vane.sqltype("INTEGER")},
            execution_backend="subprocess_task",
            batch_size=1,
            streaming_breaker=True,
        )
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
            relation,
            str(uuid.uuid4()),
        ).to_physical_plan(con)

        _reset_udf_executor_counters()
        runner = vane.ray_cxx.DistributedPhysicalPlanRunner()
        result = runner.execute_native(con.cursor(), plan, None, None)
        table = _table_from_native_result(result)
        stats = result.task_stats

        assert sorted(table.column(0).to_pylist()) == [0, 2, 4, 6]
        counters = _assert_no_udf_direct_output_conversion()
        assert counters["udf_distributed_ref_bundle_data_events"] >= 1
        assert stats["udf_max_running_tasks"] >= 1
        assert stats["udf_running_task_count"] >= 0
        assert stats["udf_queued_task_count"] >= 0
    finally:
        subprocess_exec._shutdown_global_task_runtime()
        if "relation" in locals():
            del relation
        con.close()


@pytest.mark.usefixtures("ray_local")
def test_ray_runner_replays_map_batches_udf_via_task_plan_pickle(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    pytest.importorskip("ray")
    import pyarrow as pa

    monkeypatch.setenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

    con = vane.connect()
    parquet_path = tmp_path / "table_udf_pickled_input.parquet"
    con.execute(
        f"""
        COPY (
            SELECT i::INTEGER AS a
            FROM range(6) tbl(i)
        ) TO '{parquet_path}' (FORMAT PARQUET)
        """
    )

    def double_values(table):
        values = table.column(0).to_pylist()
        return pa.table({"x": [value * 2 for value in values]})

    relation = con.sql(
        f"""
        SELECT a
        FROM read_parquet('{parquet_path}')
        WHERE a < 4
        ORDER BY a
        """
    ).map_batches(
        double_values,
        schema={"x": vane.sqltype("INTEGER")},
        execution_backend="ray_task",
        batch_size=64,
    )
    _runners.set_runner_ray(noop_if_initialized=True)
    runner = _runners.get_or_create_runner()
    parts = list(runner.run_iter_tables(relation, results_buffer_size=1))

    result = pa.concat_tables([part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts])
    assert result.column(0).to_pylist() == [0, 2, 4, 6]
    runner.close()
    del result, parts, relation
    con.close()


def test_map_batches_ray_task_rejects_direct_execution_without_query_graph(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    pytest.importorskip("ray")
    import pyarrow as pa

    monkeypatch.setenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

    con = vane.connect()
    parquet_path = tmp_path / "table_udf_direct_input.parquet"
    con.execute(
        f"""
        COPY (
            SELECT i::INTEGER AS a
            FROM range(6) tbl(i)
        ) TO '{parquet_path}' (FORMAT PARQUET)
        """
    )

    def double_values(table):
        values = table.column(0).to_pylist()
        return pa.table({"x": [value * 2 for value in values]})

    relation = con.sql(
        f"""
        SELECT a
        FROM read_parquet('{parquet_path}')
        WHERE a < 4
        ORDER BY a
        """
    ).map_batches(
        double_values,
        schema={"x": vane.sqltype("INTEGER")},
        execution_backend="ray_task",
        batch_size=64,
    )

    with pytest.raises(Exception, match="requires query_id"):
        relation.arrow().read_all()
    del relation
    con.close()
