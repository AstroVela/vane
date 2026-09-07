# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import os
import pickle
import sys
import types
import weakref
from collections.abc import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import vane
from tests.image_helpers import assert_image_equal, make_image
from vane._image import image_arrow_type


class _FakeRayRunner:
    def __init__(self, tables: list[pa.Table]) -> None:
        self.tables = tables
        self.calls: list[vane.DuckDBPyRelation] = []
        self.closed_iterators = 0

    def run_iter_tables(self, relation: vane.DuckDBPyRelation) -> Iterator[pa.Table]:
        self.calls.append(relation)
        try:
            yield from self.tables
        finally:
            self.closed_iterators += 1


def _install_fake_ray_runner(monkeypatch: pytest.MonkeyPatch, runner: object) -> list[tuple[object, bool]]:
    monkeypatch.setenv("VANE_RUNNER", "ray")
    runners = types.ModuleType("vane.runners")
    factory_calls: list[tuple[object, bool]] = []

    def set_runner_ray(address=None, noop_if_initialized=False):
        factory_calls.append((address, noop_if_initialized))
        return runner

    runners.set_runner_ray = set_runner_ray
    monkeypatch.setitem(sys.modules, "vane.runners", runners)
    return factory_calls


def _two_column_relation() -> vane.DuckDBPyRelation:
    return vane.connect().sql("SELECT 999::BIGINT AS value, 'local'::VARCHAR AS label")


def _two_column_tables() -> list[pa.Table]:
    return [
        pa.table({"c0": pa.array([1, 2], pa.int64()), "c1": ["one", "two"]}),
        pa.table({"c0": pa.array([3], pa.int64()), "c1": ["three"]}),
    ]


@pytest.mark.parametrize("configured", ["local-fast", "local", "ray", "  RAY  ", "", None])
def test_connection_execute_select_uses_configured_runner(monkeypatch, configured):
    runner = _FakeRayRunner(_two_column_tables())
    factory_calls = _install_fake_ray_runner(monkeypatch, runner)
    if configured is None:
        monkeypatch.delenv("VANE_RUNNER")
    else:
        monkeypatch.setenv("VANE_RUNNER", configured)

    with vane.connect() as connection:
        assert connection.execute("SELECT 999::BIGINT AS value, 'local' AS label") is connection
        assert [column[0] for column in connection.description] == ["value", "label"]
        if configured in {"local-fast", "local"}:
            assert connection.fetchall() == [(999, "local")]
            assert factory_calls == []
        else:
            assert len(runner.calls) == 1
            assert connection.fetchone() == (1, "one")
            assert connection.fetchmany(1) == [(2, "two")]
            assert connection.fetchall() == [(3, "three")]
            assert runner.closed_iterators == 1
        # Connection cursors are exhausted permanently until another execute().
        call_count = len(runner.calls)
        assert connection.fetchall() == []
        assert connection.fetchone() is None
        assert connection.fetchmany(2) == []
        assert len(runner.calls) == call_count


@pytest.mark.parametrize("consumer", ["df", "fetchnumpy", "to_arrow_table", "to_arrow_reader"])
def test_connection_execute_ray_bulk_consumers_do_not_reexecute(monkeypatch, consumer):
    runner = _FakeRayRunner(_two_column_tables())
    _install_fake_ray_runner(monkeypatch, runner)
    with vane.connect() as connection:
        connection.execute("SELECT 999::BIGINT AS value, 'local' AS label")
        result = getattr(connection, consumer)()
        if consumer == "df":
            result = result.to_dict(orient="list")
        elif consumer == "fetchnumpy":
            result = {name: values.tolist() for name, values in result.items()}
        else:
            if consumer == "to_arrow_reader":
                result = result.read_all()
            result = result.to_pydict()
        assert result == {"value": [1, 2, 3], "label": ["one", "two", "three"]}
        if consumer == "to_arrow_reader":
            with pytest.raises(vane.InvalidInputException, match="result closed"):
                connection.fetchall()
        else:
            assert connection.fetchall() == []
        assert len(runner.calls) == 1


class _TransportedPlanRunner:
    """Exercise real binding and plan transport without starting a Ray cluster."""

    def __init__(self):
        self.plans = []
        self.closed_iterators = 0

    def run_iter_tables(self, relation):
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, f"execute-{len(self.plans)}")
        self.plans.append(plan)
        restored = pickle.loads(pickle.dumps(plan))
        with vane.connect() as worker:
            physical_plan = restored.to_physical_plan(worker)
            native_result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(worker, physical_plan)
            try:
                yield from native_result.partition_payloads
            finally:
                self.closed_iterators += 1


@pytest.mark.parametrize(
    ("query", "parameters"),
    [
        ("SELECT ?::BIGINT AS value", [11]),
        ("SELECT $VALUE::BIGINT + $value::BIGINT AS value", {"value": 6}),
        ("SELECT ? + 1, typeof(?)", [7, 7]),
        ("SELECT ?::DATE AS value", ["2026-08-01"]),
        ("SELECT ?::VARCHAR AS value", ["quotes ' ; SELECT ' stay data"]),
        ("SELECT ?::INTEGER[] AS value", [[1, None, 3]]),
        ("SELECT ?::BIGINT AS value", [None]),
        ("SELECT $2::BIGINT AS value", {"2": 17}),
        ("SELECT i % ? AS value, count(*) AS n FROM range(6) t(i) GROUP BY value ORDER BY value", [2]),
        (
            "WITH data AS (SELECT i FROM range($rows) t(i)) "
            "SELECT i + (SELECT $offset::BIGINT) AS value FROM data WHERE i >= $start ORDER BY i LIMIT $limit",
            {"rows": 5, "offset": 10, "start": 2, "limit": 2},
        ),
    ],
)
def test_connection_execute_ray_binds_parameters_before_plan_transport(monkeypatch, query, parameters):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as reference:
        expected = reference.execute(query, parameters).fetchall()
        expected_description = reference.description

    runner = _TransportedPlanRunner()
    _install_fake_ray_runner(monkeypatch, runner)
    with vane.connect() as connection:
        assert connection.execute(query, parameters).fetchall() == expected
        assert connection.description == expected_description
        assert len(runner.plans) == 1
        assert runner.closed_iterators == 1


@pytest.mark.parametrize(
    ("query", "parameters", "message"),
    [
        ("SELECT ?", None, "Values were not provided"),
        ("SELECT 1", [1], "excess parameters"),
        ("SELECT $expected", {"unexpected": 1}, "Values were not provided"),
        ("SELECT ?", object(), "list or a dictionary"),
    ],
)
def test_connection_execute_ray_rejects_invalid_parameters_before_runner(monkeypatch, query, parameters, message):
    runner = _FakeRayRunner([])
    factory_calls = _install_fake_ray_runner(monkeypatch, runner)
    with vane.connect() as connection:
        with pytest.raises(vane.InvalidInputException, match=message):
            connection.execute(query, parameters)
        assert connection.description is None
    assert factory_calls == []


def test_connection_execute_ray_keeps_control_and_write_statements_on_connection(monkeypatch):
    runner = _FakeRayRunner([])
    factory_calls = _install_fake_ray_runner(monkeypatch, runner)
    with vane.connect() as connection:
        connection.execute("SET threads=2; CREATE TABLE items(value BIGINT)")
        assert connection.execute("INSERT INTO items VALUES (?)", [11]).fetchall() == [(1,)]
        connection.begin()
        connection.execute("INSERT INTO items VALUES (12)")
        connection.rollback()
        assert factory_calls == []
        monkeypatch.setenv("VANE_RUNNER", "local-fast")
        assert connection.execute("SELECT * FROM items").fetchall() == [(11,)]


def test_connection_execute_ray_rejects_select_in_explicit_transaction(monkeypatch):
    runner = _FakeRayRunner([])
    factory_calls = _install_fake_ray_runner(monkeypatch, runner)
    with vane.connect() as connection:
        connection.begin()
        with pytest.raises(vane.InvalidInputException, match="cannot participate.*explicit transaction"):
            connection.execute("SELECT 1")
        connection.rollback()
        assert factory_calls == []
        monkeypatch.setenv("VANE_RUNNER", "local-fast")
        connection.begin()
        assert connection.execute("SELECT 1").fetchone() == (1,)
        connection.rollback()


@pytest.mark.parametrize("table_kind", ["TABLE", "TEMP TABLE"])
@pytest.mark.parametrize("combined_statements", [False, True])
def test_connection_execute_ray_rejects_coordinator_table_without_fallback(
    monkeypatch, table_kind, combined_statements
):
    runner = _TransportedPlanRunner()
    _install_fake_ray_runner(monkeypatch, runner)
    with vane.connect() as connection:
        setup = f"CREATE {table_kind} items AS SELECT 11::BIGINT AS value"
        query = "SELECT value FROM items"
        if combined_statements:
            query = f"{setup}; {query}"
        else:
            connection.execute(setup)
        with pytest.raises((vane.CatalogException, ValueError), match="Table with name items does not exist"):
            connection.execute(query)
        assert len(runner.plans) == 1
        assert connection.description is None
        monkeypatch.setenv("VANE_RUNNER", "local-fast")
        assert connection.execute("SELECT value FROM items").fetchall() == [(11,)]


def test_connection_execute_ray_drains_preceding_queries_and_retains_last_result(monkeypatch):
    runner = _FakeRayRunner([pa.table({"value": pa.array([41, 42], pa.int64())})])
    _install_fake_ray_runner(monkeypatch, runner)
    with vane.connect() as connection:
        connection.execute("SELECT 1::BIGINT AS value; SET threads=2; SELECT ?::BIGINT AS value", [2])
        assert len(runner.calls) == 2
        assert runner.closed_iterators == 1
        assert connection.fetchall() == [(41,), (42,)]
        assert runner.closed_iterators == 2
        with pytest.raises(vane.NotImplementedException, match="only supported for the last statement"):
            connection.execute("SELECT ?; SELECT 1", [1])
        assert len(runner.calls) == 2


def test_connection_execute_ray_accepts_extracted_statements_and_module_entrypoint(monkeypatch):
    runner = _FakeRayRunner([pa.table({"value": pa.array([42], pa.int64())})])
    _install_fake_ray_runner(monkeypatch, runner)
    with vane.connect() as connection:
        statement = connection.extract_statements("SELECT ?::BIGINT AS value")[0]
        assert connection.execute(statement, [1]).fetchall() == [(42,)]
        assert vane.execute(statement, [2], connection=connection).fetchall() == [(42,)]
        assert len(runner.calls) == 2


def test_connection_execute_ray_failure_closes_old_cursor_without_local_fallback(monkeypatch):
    runner = _FakeRayRunner([pa.table({"value": pa.array([42], pa.int64())})])
    _install_fake_ray_runner(monkeypatch, runner)
    with vane.connect() as connection:
        connection.execute("SELECT 1::BIGINT AS value")
        assert runner.closed_iterators == 0

        def fail(_relation):
            raise RuntimeError("Ray submission failed")

        monkeypatch.setattr(runner, "run_iter_tables", fail)
        with pytest.raises(RuntimeError, match="Ray submission failed"):
            connection.execute("SELECT 2::BIGINT AS value")
        assert runner.closed_iterators == 1
        assert connection.description is None
        with pytest.raises(vane.InvalidInputException, match="No open result set"):
            connection.fetchall()


@pytest.mark.parametrize("close_explicitly", [False, True])
def test_connection_execute_ray_releases_abandoned_cursor_and_plan(monkeypatch, close_explicitly):
    runner = _TransportedPlanRunner()
    _install_fake_ray_runner(monkeypatch, runner)
    connection = vane.connect()
    reference = weakref.ref(connection)
    connection.execute("SELECT i FROM range(10) t(i)")
    assert runner.closed_iterators == 0
    if close_explicitly:
        connection.close()
    del connection
    gc.collect()
    assert reference() is None
    assert runner.closed_iterators == 1


def test_connection_execute_arrow_reader_keeps_connection_alive(monkeypatch):
    runner = _TransportedPlanRunner()
    _install_fake_ray_runner(monkeypatch, runner)
    connection = vane.connect()
    reference = weakref.ref(connection)
    reader = connection.execute("SELECT i FROM range(10) t(i)").to_arrow_reader(batch_size=2)
    del connection
    gc.collect()
    assert reference() is not None
    assert reader.read_all().to_pydict() == {"i": list(range(10))}
    reader.close()
    del reader
    gc.collect()
    assert reference() is None
    assert runner.closed_iterators == 1


def test_connection_execute_ray_does_not_evaluate_parameterized_udf_locally(monkeypatch):
    runner = _FakeRayRunner([pa.table({"value": pa.array([42], pa.int64())})])
    _install_fake_ray_runner(monkeypatch, runner)

    def local_execution_is_an_error(value):
        raise AssertionError(f"query executed locally with {value}")

    with vane.connect() as connection:
        vane.attach_function(
            local_execution_is_an_error,
            connection=connection,
            alias="must_run_on_ray",
            parameters=["BIGINT"],
            return_dtype="BIGINT",
        )
        assert connection.execute("SELECT must_run_on_ray(?) AS value", [7]).fetchone() == (42,)
        assert len(runner.calls) == 1


def test_connection_execute_keeps_runner_selected_before_parameter_binding(monkeypatch):
    runner = _FakeRayRunner([pa.table({"value": pa.array([42], pa.int64())})])
    _install_fake_ray_runner(monkeypatch, runner)

    class Parameters(list):
        def __len__(self):
            monkeypatch.setenv("VANE_RUNNER", "local-fast")
            return super().__len__()

    with vane.connect() as connection:
        assert connection.execute("SELECT ?::BIGINT AS value", Parameters([7])).fetchall() == [(42,)]
        assert len(runner.calls) == 1


def test_connection_execute_ray_empty_results_preserve_description(monkeypatch):
    runner = _FakeRayRunner([])
    _install_fake_ray_runner(monkeypatch, runner)
    with vane.connect() as connection:
        connection.execute("SELECT 1::BIGINT AS value WHERE false")
        assert connection.fetchone() is None
        assert connection.fetchall() == []
        assert connection.description[0][0] == "value"
        assert len(runner.calls) == 1


def test_connection_execute_uses_real_ray_runner(ray_local, monkeypatch, tmp_path):
    import ray

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    data = tmp_path / "execute.parquet"
    pq.write_table(pa.table({"value": list(range(100))}), data)
    with vane.connect() as connection:
        vane.attach_function(
            lambda value: os.getpid(),
            connection=connection,
            alias="worker_pid",
            parameters=["BIGINT"],
            return_dtype="BIGINT",
        )
        try:
            runner = vane.set_runner_ray(noop_if_initialized=True)
            assert ray.is_initialized()
            assert runner.name == "ray"
            rows = connection.execute(
                "SELECT value, worker_pid(value) AS pid FROM read_parquet(?) WHERE value >= ? ORDER BY value",
                [str(data), 97],
            ).fetchall()
            assert [row[0] for row in rows] == [97, 98, 99]
            assert all(row[1] != os.getpid() for row in rows)
            assert connection.fetchall() == []
        finally:
            vane.teardown_runner()


def _assert_typed_empty_bulk_result(result, consumer: str) -> None:
    if consumer == "fetchdf":
        assert result.empty
        assert list(result.columns) == ["value"]
        assert str(result.dtypes["value"]) == "int64"
    else:
        assert list(result) == ["value"]
        assert result["value"].tolist() == []
        assert str(result["value"].dtype) == "int64"


@pytest.mark.parametrize("consumer", ["fetchdf", "fetchnumpy"])
def test_local_bulk_result_preserves_schema_after_row_cursor_exhaustion(monkeypatch, consumer):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    relation = vane.connect().sql("SELECT 1::BIGINT AS value")

    assert relation.fetchmany(2) == [(1,)]

    _assert_typed_empty_bulk_result(getattr(relation, consumer)(), consumer)


@pytest.mark.parametrize("consumer", ["fetchdf", "fetchnumpy"])
def test_distributed_bulk_result_preserves_schema_after_row_cursor_exhaustion(monkeypatch, consumer):
    runner = _FakeRayRunner([pa.table({"c0": pa.array([1], pa.int64())})])
    _install_fake_ray_runner(monkeypatch, runner)
    relation = vane.connect().sql("SELECT 999::BIGINT AS value")

    assert relation.fetchmany(2) == [(1,)]

    _assert_typed_empty_bulk_result(getattr(relation, consumer)(), consumer)


def test_distributed_row_cursor_is_shared_across_fetch_methods(monkeypatch):
    runner = _FakeRayRunner(_two_column_tables())
    _install_fake_ray_runner(monkeypatch, runner)
    relation = _two_column_relation()

    assert relation.fetchone() == (1, "one")
    assert relation.fetchmany(1) == [(2, "two")]
    assert relation.fetchall() == [(3, "three")]

    assert len(runner.calls) == 1
    assert runner.closed_iterators == 1


def test_distributed_execute_preserves_result_for_later_consumption(monkeypatch):
    runner = _FakeRayRunner(_two_column_tables())
    _install_fake_ray_runner(monkeypatch, runner)
    relation = _two_column_relation()

    relation.execute()

    assert len(runner.calls) == 1
    assert relation.fetchall() == [
        (1, "one"),
        (2, "two"),
        (3, "three"),
    ]
    assert len(runner.calls) == 1


def test_distributed_execute_starts_runner_and_close_releases_iterator(monkeypatch):
    runner = _FakeRayRunner(_two_column_tables())
    _install_fake_ray_runner(monkeypatch, runner)
    relation = _two_column_relation()

    relation.execute()

    assert len(runner.calls) == 1
    assert runner.closed_iterators == 0

    relation.close()

    assert runner.closed_iterators == 1


def test_distributed_execute_reports_runner_start_error(monkeypatch):
    class _StartFailRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run_iter_tables(self, _relation):
            self.calls += 1
            raise RuntimeError("runner failed before first partition")
            yield  # pragma: no cover

    runner = _StartFailRunner()
    _install_fake_ray_runner(monkeypatch, runner)
    relation = _two_column_relation()

    with pytest.raises(RuntimeError, match="runner failed before first partition"):
        relation.execute()

    assert runner.calls == 1


def test_distributed_result_reports_midstream_runner_error(monkeypatch):
    class _MidstreamFailRunner:
        def __init__(self) -> None:
            self.closed_iterators = 0

        def run_iter_tables(self, _relation):
            try:
                yield pa.table({"c0": pa.array([1], pa.int64())})
                raise RuntimeError("runner failed after first partition")
            finally:
                self.closed_iterators += 1

    runner = _MidstreamFailRunner()
    _install_fake_ray_runner(monkeypatch, runner)
    relation = vane.connect().sql("SELECT 1::BIGINT AS value")

    with pytest.raises(vane.InvalidInputException, match="runner failed after first partition"):
        relation.fetchall()

    assert runner.closed_iterators == 1

    with pytest.raises(vane.InvalidInputException, match="runner failed after first partition"):
        relation.fetchall()

    assert runner.closed_iterators == 1


def test_distributed_numpy_and_pandas_use_relation_names(monkeypatch):
    runner = _FakeRayRunner(_two_column_tables())
    _install_fake_ray_runner(monkeypatch, runner)

    numpy_result = _two_column_relation().fetchnumpy()
    assert list(numpy_result) == ["value", "label"]
    assert numpy_result["value"].tolist() == [1, 2, 3]
    assert numpy_result["label"].tolist() == ["one", "two", "three"]

    frame = _two_column_relation().df()
    assert frame.to_dict(orient="list") == {
        "value": [1, 2, 3],
        "label": ["one", "two", "three"],
    }


@pytest.mark.parametrize(
    ("consumer", "result_kind", "deprecated"),
    [
        ("fetchdf", "frame", False),
        ("to_df", "frame", False),
        ("arrow", "reader", False),
        ("fetch_arrow_table", "table", True),
        ("fetch_record_batch", "reader", True),
        ("fetch_arrow_reader", "reader", True),
    ],
)
def test_distributed_consumer_aliases(monkeypatch, consumer, result_kind, deprecated):
    runner = _FakeRayRunner(_two_column_tables())
    _install_fake_ray_runner(monkeypatch, runner)

    if deprecated:
        with pytest.warns(DeprecationWarning):
            result = getattr(_two_column_relation(), consumer)()
    else:
        result = getattr(_two_column_relation(), consumer)()

    if result_kind == "frame":
        assert result.to_dict(orient="list") == {
            "value": [1, 2, 3],
            "label": ["one", "two", "three"],
        }
    else:
        if result_kind == "reader":
            result = result.read_all()
        assert result.to_pydict() == {
            "value": [1, 2, 3],
            "label": ["one", "two", "three"],
        }
    assert len(runner.calls) == 1


@pytest.mark.parametrize(
    ("consumer", "module_name", "converter_name"),
    [
        ("torch", "torch", "from_numpy"),
        ("tf", "tensorflow", "convert_to_tensor"),
    ],
)
def test_distributed_tensor_consumers_receive_numpy_results(monkeypatch, consumer, module_name, converter_name):
    runner = _FakeRayRunner([pa.table({"c0": pa.array([1, 2, 3], pa.int64())})])
    _install_fake_ray_runner(monkeypatch, runner)
    framework = types.ModuleType(module_name)
    setattr(framework, converter_name, lambda array: array.tolist())
    monkeypatch.setitem(sys.modules, module_name, framework)
    relation = vane.connect().sql("SELECT 999::BIGINT AS value")

    assert getattr(relation, consumer)() == {"value": [1, 2, 3]}
    assert len(runner.calls) == 1


def test_distributed_polars_eager_and_lazy(monkeypatch):
    pytest.importorskip("polars")
    runner = _FakeRayRunner(_two_column_tables())
    _install_fake_ray_runner(monkeypatch, runner)

    eager = _two_column_relation().pl()
    lazy_relation = _two_column_relation()
    lazy = lazy_relation.pl(lazy=True).collect()

    expected = [
        {"value": 1, "label": "one"},
        {"value": 2, "label": "two"},
        {"value": 3, "label": "three"},
    ]
    assert eager.to_dicts() == expected
    assert lazy.to_dicts() == expected
    assert len(runner.calls) == 2


def test_distributed_df_chunks_preserve_cursor_state(monkeypatch):
    first = pa.table(
        {
            "c0": pa.array(range(3000), pa.int64()),
            "c1": [f"row-{index}" for index in range(3000)],
        }
    )
    runner = _FakeRayRunner([first])
    _install_fake_ray_runner(monkeypatch, runner)
    relation = _two_column_relation()

    first_chunk = relation.fetch_df_chunk(vectors_per_chunk=1)
    second_chunk = relation.fetch_df_chunk(vectors_per_chunk=1)
    third_chunk = relation.fetch_df_chunk(vectors_per_chunk=1)
    fourth_chunk = relation.fetch_df_chunk(vectors_per_chunk=100)
    fifth_chunk = relation.fetch_df_chunk(vectors_per_chunk=0)

    assert first_chunk["value"].tolist() == list(range(2048))
    assert second_chunk["value"].tolist() == list(range(2048, 3000))
    assert third_chunk.empty
    assert fourth_chunk.empty
    assert fifth_chunk.empty
    assert list(fourth_chunk.columns) == ["value", "label"]
    assert len(runner.calls) == 1

    relation.close()
    with pytest.raises(vane.InvalidInputException, match="result closed"):
        relation.fetch_df_chunk()


def test_distributed_arrow_table_and_reader_stream_partitions(monkeypatch):
    runner = _FakeRayRunner(_two_column_tables())
    _install_fake_ray_runner(monkeypatch, runner)

    table = _two_column_relation().to_arrow_table(batch_size=1)
    assert table.schema.names == ["value", "label"]
    assert table.to_pydict() == {
        "value": [1, 2, 3],
        "label": ["one", "two", "three"],
    }

    reader = _two_column_relation().to_arrow_reader(batch_size=2)
    assert [batch.num_rows for batch in reader] == [2, 1]


@pytest.mark.parametrize("consumer", ["to_arrow_reader", "to_arrow_table"])
def test_local_arrow_consumers_reject_zero_batch_size_without_consuming_result(monkeypatch, consumer):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    connection = vane.connect()
    connection.execute("SELECT 1::BIGINT AS value")

    with pytest.raises(RuntimeError, match="Approximate Batch Size of Record Batch MUST be higher than 0"):
        getattr(connection, consumer)(batch_size=0)

    assert connection.fetchall() == [(1,)]


@pytest.mark.parametrize("consumer", ["to_arrow_reader", "to_arrow_table"])
def test_fresh_local_relation_arrow_consumers_reject_zero_batch_size_without_consuming_result(monkeypatch, consumer):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    relation = vane.connect().sql("SELECT 1::BIGINT AS value")

    with pytest.raises(RuntimeError, match="Approximate Batch Size of Record Batch MUST be higher than 0"):
        getattr(relation, consumer)(batch_size=0)

    assert relation.fetchall() == [(1,)]


@pytest.mark.parametrize("consumer", ["to_arrow_reader", "to_arrow_table"])
def test_distributed_arrow_consumers_reject_zero_batch_size_without_consuming_result(monkeypatch, consumer):
    runner = _FakeRayRunner(_two_column_tables())
    _install_fake_ray_runner(monkeypatch, runner)
    relation = _two_column_relation()

    with pytest.raises(RuntimeError, match="Approximate Batch Size of Record Batch MUST be higher than 0"):
        getattr(relation, consumer)(batch_size=0)

    assert relation.fetchall() == [
        (1, "one"),
        (2, "two"),
        (3, "three"),
    ]
    assert len(runner.calls) == 1


def test_distributed_arrow_capsule_protocol(monkeypatch):
    runner = _FakeRayRunner(_two_column_tables())
    _install_fake_ray_runner(monkeypatch, runner)

    reader = pa.RecordBatchReader.from_stream(_two_column_relation())
    assert reader.read_all().to_pydict() == {
        "value": [1, 2, 3],
        "label": ["one", "two", "three"],
    }


def test_distributed_result_rejects_switching_cursor_modes(monkeypatch):
    runner = _FakeRayRunner(_two_column_tables())
    _install_fake_ray_runner(monkeypatch, runner)
    relation = _two_column_relation()

    assert relation.fetchone() == (1, "one")
    with pytest.raises(vane.InvalidInputException, match="partially consumed row result"):
        relation.to_arrow_table()


def test_distributed_result_preserves_duplicate_names(monkeypatch):
    table = pa.Table.from_arrays([pa.array([10]), pa.array([20])], names=["c0", "c1"])
    runner = _FakeRayRunner([table])
    _install_fake_ray_runner(monkeypatch, runner)

    relation = vane.connect().sql("SELECT 1::BIGINT AS a, 2::BIGINT AS a")
    result = relation.to_arrow_table()

    assert result.schema.names == ["a", "a"]
    assert result.column(0).to_pylist() == [10]
    assert result.column(1).to_pylist() == [20]


def test_distributed_empty_result_keeps_schema(monkeypatch):
    runner = _FakeRayRunner([])
    _install_fake_ray_runner(monkeypatch, runner)

    row_relation = vane.connect().sql("SELECT NULL::VARCHAR AS name WHERE FALSE")
    assert row_relation.fetchall() == []

    arrow_relation = vane.connect().sql("SELECT NULL::VARCHAR AS name WHERE FALSE")
    result = arrow_relation.to_arrow_table()
    assert result.schema.names == ["name"]
    assert result.schema.types == [pa.string()]
    assert result.num_rows == 0


def test_distributed_result_rejects_partition_schema_mismatch(monkeypatch):
    runner = _FakeRayRunner([pa.table({"c0": ["wrong type"]})])
    _install_fake_ray_runner(monkeypatch, runner)
    relation = vane.connect().sql("SELECT 1::BIGINT AS value")

    with pytest.raises(vane.InvalidInputException, match="has Arrow type string, expected int64"):
        relation.fetchall()


def test_distributed_result_rejects_safe_but_noncanonical_partition_type(monkeypatch):
    runner = _FakeRayRunner([pa.table({"c0": pa.array([1], pa.int32())})])
    _install_fake_ray_runner(monkeypatch, runner)
    relation = vane.connect().sql("SELECT 1::BIGINT AS value")

    with pytest.raises(vane.InvalidInputException, match="has Arrow type int32, expected int64"):
        relation.fetchall()


@pytest.mark.parametrize(
    "query",
    [
        "SELECT 1::HUGEINT AS value",
        "SELECT 1::UHUGEINT AS value",
        "SELECT '00112233-4455-6677-8899-aabbccddeeff'::UUID AS value",
        "SELECT '10101'::BIT AS value",
        "SELECT '12:34:56+02:00'::TIMETZ AS value",
        "SELECT '{\"key\": 1}'::JSON AS value",
        "SELECT [1::HUGEINT] AS value",
    ],
)
def test_distributed_result_rejects_lossy_types_before_starting_runner(monkeypatch, query):
    runner = _FakeRayRunner([])
    factory_calls = _install_fake_ray_runner(monkeypatch, runner)

    with pytest.raises(
        vane.NotImplementedException,
        match="cannot preserve result type.*arrow_lossless_conversion",
    ):
        vane.connect().sql(query).fetchall()

    assert factory_calls == []
    assert runner.calls == []


@pytest.mark.parametrize(
    ("partition_query", "relation_query", "expected"),
    [
        ("SELECT 1::HUGEINT AS c0", "SELECT 999::HUGEINT AS value", "1"),
        ("SELECT 1::UHUGEINT AS c0", "SELECT 999::UHUGEINT AS value", "1"),
        (
            "SELECT '00112233-4455-6677-8899-aabbccddeeff'::UUID AS c0",
            "SELECT 'ffffffff-ffff-ffff-ffff-ffffffffffff'::UUID AS value",
            "00112233-4455-6677-8899-aabbccddeeff",
        ),
        ("SELECT '10101'::BIT AS c0", "SELECT '111'::BIT AS value", "10101"),
        (
            "SELECT '{\"key\": 1}'::JSON AS c0",
            "SELECT '{\"local\": true}'::JSON AS value",
            '{"key": 1}',
        ),
        (
            "SELECT '12:34:56+02:00'::TIMETZ AS c0",
            "SELECT '01:02:03+01:00'::TIMETZ AS value",
            "12:34:56+02:00",
        ),
    ],
)
def test_distributed_result_accepts_lossless_arrow_extension_types(
    monkeypatch, partition_query, relation_query, expected
):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    connection = vane.connect()
    connection.execute("SET arrow_lossless_conversion = true")
    table = connection.sql(partition_query).to_arrow_table()

    runner = _FakeRayRunner([table])
    _install_fake_ray_runner(monkeypatch, runner)

    assert str(connection.sql(relation_query).fetchone()[0]) == expected
    assert len(runner.calls) == 1


@pytest.mark.parametrize(
    ("partition_query", "relation_query", "expected"),
    [
        (
            "SELECT ['{\"ray\": true}'::JSON] AS c0",
            "SELECT ['{\"local\": true}'::JSON] AS value",
            ['{"ray": true}'],
        ),
        ("SELECT ['10101'::BIT] AS c0", "SELECT ['111'::BIT] AS value", ["10101"]),
        (
            "SELECT [123456789012345678901234567890::BIGNUM] AS c0",
            "SELECT [1::BIGNUM] AS value",
            ["123456789012345678901234567890"],
        ),
        (
            "SELECT {'json_value': '{\"ray\": true}'::JSON} AS c0",
            "SELECT {'json_value': '{\"local\": true}'::JSON} AS value",
            {"json_value": '{"ray": true}'},
        ),
        (
            "SELECT ['{\"ray\": true}'::JSON]::JSON[1] AS c0",
            "SELECT ['{\"local\": true}'::JSON]::JSON[1] AS value",
            ('{"ray": true}',),
        ),
        (
            "SELECT map(['key'], ['{\"ray\": true}'::JSON]) AS c0",
            "SELECT map(['key'], ['{\"local\": true}'::JSON]) AS value",
            {"key": '{"ray": true}'},
        ),
    ],
)
def test_distributed_result_normalizes_nested_lossless_arrow_extension_storage(
    monkeypatch, partition_query, relation_query, expected
):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    producer = vane.connect()
    producer.execute("SET arrow_lossless_conversion = true")
    producer.execute("SET arrow_large_buffer_size = true")
    table = producer.sql(partition_query).to_arrow_table()

    runner = _FakeRayRunner([table])
    _install_fake_ray_runner(monkeypatch, runner)
    consumer = vane.connect()
    consumer.execute("SET arrow_lossless_conversion = true")

    assert consumer.sql(relation_query).fetchone() == (expected,)
    assert len(runner.calls) == 1


def test_distributed_result_normalizes_sliced_sparse_union_children(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    producer = vane.connect()
    producer.execute("SET arrow_large_buffer_size = true")
    table = producer.sql("""
        SELECT
            CASE
                WHEN i % 2 = 0
                    THEN ('ray-' || i::VARCHAR)::UNION(s VARCHAR, i BIGINT)
                ELSE i::BIGINT::UNION(s VARCHAR, i BIGINT)
            END AS c0
        FROM range(6) AS t(i)
    """).to_arrow_table()
    table = table.slice(1, 4)

    runner = _FakeRayRunner([table])
    _install_fake_ray_runner(monkeypatch, runner)

    relation = vane.connect().sql("SELECT NULL::UNION(s VARCHAR, i BIGINT) AS value")
    assert relation.fetchall() == [(1,), ("ray-2",), (3,), ("ray-4",)]
    assert len(runner.calls) == 1


def test_distributed_result_normalizes_timestamp_timezone_metadata(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    producer = vane.connect()
    producer.execute("SET TimeZone = 'UTC'")
    table = producer.sql("SELECT TIMESTAMPTZ '2024-01-01 12:00:00+00' AS c0").to_arrow_table()
    assert table.schema.field(0).type.tz == "UTC"

    runner = _FakeRayRunner([table])
    _install_fake_ray_runner(monkeypatch, runner)
    consumer = vane.connect()
    consumer.execute("SET TimeZone = 'America/New_York'")

    value = consumer.sql("SELECT TIMESTAMPTZ '2024-01-01 12:00:00+00' AS value").fetchone()[0]
    assert value.isoformat() == "2024-01-01T07:00:00-05:00"
    assert len(runner.calls) == 1


def test_distributed_result_does_not_reinterpret_naive_timestamp_as_timestamp_timezone(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    table = vane.connect().sql("SELECT TIMESTAMP '2024-01-01 12:00:00' AS c0").to_arrow_table()

    runner = _FakeRayRunner([table])
    _install_fake_ray_runner(monkeypatch, runner)
    consumer = vane.connect()
    consumer.execute("SET TimeZone = 'America/New_York'")

    relation = consumer.sql("SELECT TIMESTAMPTZ '2024-01-01 12:00:00+00' AS value")
    with pytest.raises(
        vane.InvalidInputException,
        match=r"has Arrow type timestamp\[us\], expected timestamp\[us, tz=America/New_York\]",
    ):
        relation.fetchall()


@pytest.mark.parametrize(
    "query",
    [
        "SELECT 'red'::ENUM('red', 'blue') AS value",
        "SELECT 42::VARIANT AS value",
        "SELECT sum(i) EXPORT_STATE AS value FROM range(3) t(i)",
    ],
)
def test_distributed_result_rejects_untransportable_types_before_starting_runner_even_when_lossless(monkeypatch, query):
    connection = vane.connect()
    connection.execute("SET arrow_lossless_conversion = true")
    runner = _FakeRayRunner([])
    factory_calls = _install_fake_ray_runner(monkeypatch, runner)

    with pytest.raises(
        vane.NotImplementedException,
        match="cannot preserve result type.*Arrow transport",
    ):
        connection.sql(query).fetchall()

    assert factory_calls == []
    assert runner.calls == []


@pytest.mark.parametrize(
    ("query", "table", "expected"),
    [
        (
            "SELECT 'local'::VARCHAR AS value",
            pa.table({"c0": pa.array(["distributed"], pa.large_string())}),
            [("distributed",)],
        ),
        (
            "SELECT 'local'::BLOB AS value",
            pa.table({"c0": pa.array([b"distributed"], pa.large_binary())}),
            [(b"distributed",)],
        ),
        (
            "SELECT ['local'::VARCHAR] AS value",
            pa.table({"c0": pa.array([["distributed"]], pa.large_list(pa.large_string()))}),
            [(["distributed"],)],
        ),
        (
            "SELECT ['local'::VARCHAR] AS value",
            pa.table(
                {
                    "c0": pa.ListViewArray.from_arrays(
                        pa.array([0, 1], pa.int32()),
                        pa.array([2, 3], pa.int32()),
                        pa.array(["first", "second", "third", "fourth"], pa.large_string()),
                    )
                }
            ),
            [(["first", "second"],), (["second", "third", "fourth"],)],
        ),
    ],
)
def test_distributed_result_normalizes_arrow_offset_widths(monkeypatch, query, table, expected):
    runner = _FakeRayRunner([table])
    _install_fake_ray_runner(monkeypatch, runner)

    assert vane.connect().sql(query).fetchall() == expected


def test_distributed_result_normalizes_file_tensor_storage(monkeypatch):
    file_type = pa.struct(
        [
            pa.field("url", pa.large_string()),
            pa.field("content_type", pa.large_string()),
            pa.field("position", pa.int64()),
            pa.field("size", pa.int64()),
            pa.field("checksum", pa.large_string()),
        ]
    )
    records = [
        {
            "url": "memory://first",
            "content_type": None,
            "position": None,
            "size": None,
            "checksum": None,
        },
        {
            "url": "memory://second",
            "content_type": None,
            "position": None,
            "size": None,
            "checksum": None,
        },
    ]
    files = pa.array(records, type=file_type)
    storage = pa.FixedSizeListArray.from_arrays(files, 2)
    tensor = pa.ExtensionArray.from_storage(pa.fixed_shape_tensor(file_type, (2,)), storage)
    runner = _FakeRayRunner([pa.table({"c0": tensor})])
    _install_fake_ray_runner(monkeypatch, runner)

    def unused(table):
        return table

    relation = (
        vane.connect()
        .sql("SELECT 1 AS value")
        .map_batches(
            unused,
            schema={"documents": vane.tensor_type(vane.file_type(), (2,))},
            execution_backend="subprocess_task",
        )
    )

    assert relation.fetchone() == (
        (
            vane.File("memory://first"),
            vane.File("memory://second"),
        ),
    )


def test_distributed_result_restores_decoded_image_type(monkeypatch):
    image_type = image_arrow_type(vane.image_type()).storage_type
    pixels = bytes(range(6))
    images = pa.array(
        [{"data": list(pixels), "channel": 3, "height": 1, "width": 2, "mode": 3}],
        type=image_type,
    )
    images = pa.ExtensionArray.from_storage(image_arrow_type(vane.image_type()), images)
    runner = _FakeRayRunner([pa.table({"c0": images})])
    _install_fake_ray_runner(monkeypatch, runner)

    def unused(table):
        return table

    relation = (
        vane.connect()
        .sql("SELECT 1 AS value")
        .map_batches(
            unused,
            schema={"image": vane.image_type()},
            execution_backend="subprocess_task",
        )
    )

    assert relation.types[0].is_image()
    assert_image_equal(relation.fetchone(), (make_image(pixels, 2, 1, "RGB"),))


@pytest.mark.parametrize(
    ("consumer", "error_type"),
    [
        pytest.param("fetchone", vane.InvalidInputException, id="row"),
        pytest.param("to_arrow_table", OSError, id="arrow"),
    ],
)
@pytest.mark.parametrize("nested", [False, True], ids=["top-level", "nested"])
def test_distributed_result_rejects_malformed_decoded_image_before_consumption(
    monkeypatch, consumer, error_type, nested
):
    image_type = image_arrow_type(vane.image_type()).storage_type
    malformed = {"data": list(b"\x00"), "channel": 3, "height": 1, "width": 1, "mode": 3}
    images = pa.ExtensionArray.from_storage(image_arrow_type(vane.image_type()), pa.array([malformed], type=image_type))
    if nested:
        values = pa.StructArray.from_arrays([images], names=["image"])
        result_type = vane.struct_type({"image": vane.image_type()})
    else:
        values = images
        result_type = vane.image_type()

    runner = _FakeRayRunner([pa.table({"c0": values})])
    _install_fake_ray_runner(monkeypatch, runner)

    def unused(table):
        return table

    relation = (
        vane.connect()
        .sql("SELECT 1 AS value")
        .map_batches(
            unused,
            schema={"value": result_type},
            execution_backend="subprocess_task",
        )
    )

    with pytest.raises(
        error_type,
        match=r"Distributed result partition 0 column 0 failed IMAGE validation",
    ):
        getattr(relation, consumer)()

    assert runner.closed_iterators == 1


def test_distributed_partition_error_is_terminal_and_closes_iterator(monkeypatch):
    runner = _FakeRayRunner(
        [
            pa.table({"c0": ["bad"]}),
            pa.table({"c0": pa.array([22], pa.int64())}),
        ]
    )
    _install_fake_ray_runner(monkeypatch, runner)
    relation = vane.connect().sql("SELECT 1::BIGINT AS value")

    with pytest.raises(vane.InvalidInputException, match="has Arrow type string, expected int64"):
        relation.fetchall()

    assert runner.closed_iterators == 1

    with pytest.raises(vane.InvalidInputException, match="has Arrow type string, expected int64"):
        relation.fetchall()

    assert runner.closed_iterators == 1


def test_distributed_runner_error_does_not_fall_back_to_local(monkeypatch):
    class _UnsupportedPlanRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run_iter_tables(self, _relation):
            self.calls += 1
            raise NotImplementedError("unsupported distributed plan")
            yield  # pragma: no cover

    runner = _UnsupportedPlanRunner()
    _install_fake_ray_runner(monkeypatch, runner)
    relation = vane.connect().sql("SELECT source_id FROM pragma_version()")

    with pytest.raises(NotImplementedError, match="unsupported distributed plan"):
        relation.fetchone()

    assert runner.calls == 1


def test_distributed_result_close_closes_runner_iterator(monkeypatch):
    runner = _FakeRayRunner(_two_column_tables())
    _install_fake_ray_runner(monkeypatch, runner)
    relation = _two_column_relation()

    assert relation.fetchone() == (1, "one")
    relation.close()

    assert runner.closed_iterators == 1
    with pytest.raises(vane.InvalidInputException, match="result closed"):
        relation.fetchall()


def test_distributed_partial_result_lifecycle_stress(monkeypatch):
    runner = _FakeRayRunner(_two_column_tables())
    _install_fake_ray_runner(monkeypatch, runner)

    row_iterations = 32
    for index in range(row_iterations):
        relation = _two_column_relation()
        assert relation.fetchone() == (1, "one")
        if index % 2 == 0:
            relation.close()
        del relation

    arrow_iterations = 32
    for index in range(arrow_iterations):
        relation = _two_column_relation()
        reader = relation.to_arrow_reader(batch_size=1)
        assert reader.read_next_batch().to_pydict() == {
            "value": [1],
            "label": ["one"],
        }
        if index % 2 == 0:
            reader.close()
        del reader
        del relation

    gc.collect()

    assert len(runner.calls) == row_iterations + arrow_iterations
    assert runner.closed_iterators == row_iterations + arrow_iterations


def test_distributed_len_and_shape_use_runner(monkeypatch):
    runner = _FakeRayRunner([pa.table({"c0": pa.array([3], pa.int64())})])
    _install_fake_ray_runner(monkeypatch, runner)

    relation = _two_column_relation()
    assert len(relation) == 3
    assert relation.shape == (3, 2)
    assert len(runner.calls) == 2


def test_distributed_repr_uses_common_result_source(monkeypatch):
    runner = _FakeRayRunner([pa.table({"c0": pa.array([41, 42], pa.int64())})])
    _install_fake_ray_runner(monkeypatch, runner)

    output = repr(vane.connect().sql("SELECT 999::BIGINT AS value"))

    assert "41" in output
    assert "42" in output
    assert "999" not in output
    assert len(runner.calls) == 1
    assert "LIMIT 10000" in runner.calls[0].sql_query().upper()
