# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import sys
import types
from collections.abc import Iterator

import pyarrow as pa
import pytest

import duckdb


class _FakeRayRunner:
    def __init__(self, tables: list[pa.Table]) -> None:
        self.tables = tables
        self.calls: list[tuple[duckdb.DuckDBPyRelation, int | None]] = []
        self.closed_iterators = 0

    def run_iter_tables(
        self, relation: duckdb.DuckDBPyRelation, results_buffer_size: int | None = None
    ) -> Iterator[pa.Table]:
        self.calls.append((relation, results_buffer_size))
        try:
            yield from self.tables
        finally:
            self.closed_iterators += 1


def _install_fake_ray_runner(monkeypatch: pytest.MonkeyPatch, runner: object) -> None:
    monkeypatch.setenv("VANE_RUNNER", "ray")
    runners = types.ModuleType("duckdb.runners")
    runners.set_runner_ray = lambda *_args, **_kwargs: runner
    monkeypatch.setitem(sys.modules, "duckdb.runners", runners)


def _two_column_relation() -> duckdb.DuckDBPyRelation:
    return duckdb.connect().sql("SELECT 999::BIGINT AS value, 'local'::VARCHAR AS label")


def _two_column_tables() -> list[pa.Table]:
    return [
        pa.table({"c0": pa.array([1, 2], pa.int64()), "c1": ["one", "two"]}),
        pa.table({"c0": pa.array([3], pa.int64()), "c1": ["three"]}),
    ]


def test_distributed_row_cursor_is_shared_across_fetch_methods(monkeypatch):
    runner = _FakeRayRunner(_two_column_tables())
    _install_fake_ray_runner(monkeypatch, runner)
    relation = _two_column_relation()

    assert relation.fetchone() == (1, "one")
    assert relation.fetchmany(1) == [(2, "two")]
    assert relation.fetchall() == [(3, "three")]

    assert len(runner.calls) == 1
    assert runner.calls[0][1] == 1
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

        def run_iter_tables(self, _relation, results_buffer_size=None):
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

        def run_iter_tables(self, _relation, results_buffer_size=None):
            del results_buffer_size
            try:
                yield pa.table({"c0": pa.array([1], pa.int64())})
                raise RuntimeError("runner failed after first partition")
            finally:
                self.closed_iterators += 1

    runner = _MidstreamFailRunner()
    _install_fake_ray_runner(monkeypatch, runner)
    relation = duckdb.connect().sql("SELECT 1::BIGINT AS value")

    with pytest.raises(duckdb.InvalidInputException, match="runner failed after first partition"):
        relation.fetchall()

    assert runner.closed_iterators == 1

    with pytest.raises(duckdb.InvalidInputException, match="runner failed after first partition"):
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
    relation = duckdb.connect().sql("SELECT 999::BIGINT AS value")

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
    with pytest.raises(duckdb.InvalidInputException, match="result closed"):
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
    connection = duckdb.connect()
    connection.execute("SELECT 1::BIGINT AS value")

    with pytest.raises(RuntimeError, match="Approximate Batch Size of Record Batch MUST be higher than 0"):
        getattr(connection, consumer)(batch_size=0)

    assert connection.fetchall() == [(1,)]


@pytest.mark.parametrize("consumer", ["to_arrow_reader", "to_arrow_table"])
def test_fresh_local_relation_arrow_consumers_reject_zero_batch_size_without_consuming_result(monkeypatch, consumer):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    relation = duckdb.connect().sql("SELECT 1::BIGINT AS value")

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
    with pytest.raises(duckdb.InvalidInputException, match="partially consumed row result"):
        relation.to_arrow_table()


def test_distributed_result_preserves_duplicate_names(monkeypatch):
    table = pa.Table.from_arrays([pa.array([10]), pa.array([20])], names=["c0", "c1"])
    runner = _FakeRayRunner([table])
    _install_fake_ray_runner(monkeypatch, runner)

    relation = duckdb.connect().sql("SELECT 1::BIGINT AS a, 2::BIGINT AS a")
    result = relation.to_arrow_table()

    assert result.schema.names == ["a", "a"]
    assert result.column(0).to_pylist() == [10]
    assert result.column(1).to_pylist() == [20]


def test_distributed_empty_result_keeps_schema(monkeypatch):
    runner = _FakeRayRunner([])
    _install_fake_ray_runner(monkeypatch, runner)

    row_relation = duckdb.connect().sql("SELECT NULL::VARCHAR AS name")
    assert row_relation.fetchall() == []

    arrow_relation = duckdb.connect().sql("SELECT NULL::VARCHAR AS name")
    result = arrow_relation.to_arrow_table()
    assert result.schema.names == ["name"]
    assert result.schema.types == [pa.string()]
    assert result.num_rows == 0


def test_distributed_result_rejects_partition_schema_mismatch(monkeypatch):
    runner = _FakeRayRunner([pa.table({"c0": ["wrong type"]})])
    _install_fake_ray_runner(monkeypatch, runner)
    relation = duckdb.connect().sql("SELECT 1::BIGINT AS value")

    with pytest.raises(duckdb.InvalidInputException, match="has Arrow type string, expected int64"):
        relation.fetchall()


def test_distributed_result_rejects_safe_but_noncanonical_partition_type(monkeypatch):
    runner = _FakeRayRunner([pa.table({"c0": pa.array([1], pa.int32())})])
    _install_fake_ray_runner(monkeypatch, runner)
    relation = duckdb.connect().sql("SELECT 1::BIGINT AS value")

    with pytest.raises(duckdb.InvalidInputException, match="has Arrow type int32, expected int64"):
        relation.fetchall()


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
    ],
)
def test_distributed_result_normalizes_only_arrow_offset_widths(monkeypatch, query, table, expected):
    runner = _FakeRayRunner([table])
    _install_fake_ray_runner(monkeypatch, runner)

    assert duckdb.connect().sql(query).fetchall() == expected


def test_distributed_partition_error_is_terminal_and_closes_iterator(monkeypatch):
    runner = _FakeRayRunner(
        [
            pa.table({"c0": ["bad"]}),
            pa.table({"c0": pa.array([22], pa.int64())}),
        ]
    )
    _install_fake_ray_runner(monkeypatch, runner)
    relation = duckdb.connect().sql("SELECT 1::BIGINT AS value")

    with pytest.raises(duckdb.InvalidInputException, match="has Arrow type string, expected int64"):
        relation.fetchall()

    assert runner.closed_iterators == 1

    with pytest.raises(duckdb.InvalidInputException, match="has Arrow type string, expected int64"):
        relation.fetchall()

    assert runner.closed_iterators == 1


def test_distributed_runner_error_does_not_fall_back_to_local(monkeypatch):
    class _UnsupportedPlanRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run_iter_tables(self, _relation, results_buffer_size=None):
            self.calls += 1
            raise NotImplementedError("unsupported distributed plan")
            yield  # pragma: no cover

    runner = _UnsupportedPlanRunner()
    _install_fake_ray_runner(monkeypatch, runner)
    relation = duckdb.connect().sql("SELECT source_id FROM pragma_version()")

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
    with pytest.raises(duckdb.InvalidInputException, match="result closed"):
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

    output = repr(duckdb.connect().sql("SELECT 999::BIGINT AS value"))

    assert "41" in output
    assert "42" in output
    assert "999" not in output
    assert len(runner.calls) == 1
    assert "LIMIT 10000" in runner.calls[0][0].sql_query().upper()
