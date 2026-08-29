# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pytest

try:
    import ray
except Exception:
    ray = None

import vane
from vane import runners

pytestmark = [
    pytest.mark.skipif(ray is None, reason="ray not installed"),
    pytest.mark.usefixtures("ray_local"),
]


@pytest.fixture(autouse=True)
def _ray_execution_env(monkeypatch):
    monkeypatch.setenv("VANE_SHUFFLE_ALGORITHM", "flight_shuffle")
    monkeypatch.setenv("VANE_SHUFFLE_LOCAL_DIRS", "/tmp/duckdb_shuffle")
    monkeypatch.setenv("RAY_DEDUP_LOGS", "0")


@pytest.fixture
def connection():
    con = vane.connect()
    try:
        yield con
    finally:
        con.close()


def _memory_relation(con, source_kind: str, row_count: int):
    values = range(row_count)
    if source_kind == "pandas":
        return con.from_df(pd.DataFrame({"id": values, "value": [value * 10 for value in values]}))
    if source_kind == "arrow":
        return con.from_arrow(
            pa.table(
                {
                    "id": pa.array(values, type=pa.int64()),
                    "value": pa.array((value * 10 for value in values), type=pa.int64()),
                }
            )
        )
    raise AssertionError(f"unknown source kind: {source_kind}")


@pytest.mark.parametrize("source_kind", ["pandas", "arrow"])
def test_python_memory_plan_uses_compact_query_owned_object_ref(connection, source_kind):
    relation = _memory_relation(connection, source_kind, 100_000)

    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, f"compact-{source_kind}-memory-source")
    serialized = pickle.dumps(logical)

    assert logical._memory_source_ref_count_for_test() == 1
    assert len(serialized) < 64 * 1024

    restored = pickle.loads(serialized)
    assert restored._memory_source_ref_count_for_test() == 1
    physical = restored.to_physical_plan(connection)
    assert physical._memory_source_ref_count_for_test() == 1
    split_batches = physical.scan_split_batch_map()
    assert [len(batches) for batches in split_batches.values()] == [1]
    assert sum(len(batch) for batches in split_batches.values() for batch in batches) < 4096


def test_repeated_arrow_source_is_snapshotted_once(connection, monkeypatch):
    from vane.datasource import _memory

    source = pa.table({"id": [1, 2, 3], "value": [10, 20, 30]})
    snapshot_calls = 0
    original = _memory._as_arrow_table

    def count_snapshot(value, source_kind):
        nonlocal snapshot_calls
        snapshot_calls += 1
        return original(value, source_kind)

    monkeypatch.setattr(_memory, "_as_arrow_table", count_snapshot)
    left = connection.from_arrow(source).set_alias("left_source")
    right = connection.from_arrow(source).set_alias("right_source")
    relation = left.join(right, "left_source.id = right_source.id")

    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "deduplicated-arrow-memory-source")

    assert snapshot_calls == 1
    assert logical._memory_source_ref_count_for_test() == 1
    physical = logical.to_physical_plan(connection)
    assert sorted(len(batches) for batches in physical.scan_split_batch_map().values()) == [1, 1]


def test_repeated_pandas_source_is_snapshotted_once(connection, monkeypatch):
    from vane.datasource import _memory

    source = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
    snapshot_calls = 0
    original = _memory._as_arrow_table

    def count_snapshot(value, source_kind):
        nonlocal snapshot_calls
        snapshot_calls += 1
        return original(value, source_kind)

    monkeypatch.setattr(_memory, "_as_arrow_table", count_snapshot)
    left = connection.from_df(source).set_alias("left_source")
    right = connection.from_df(source).set_alias("right_source")
    relation = left.join(right, "left_source.id = right_source.id")

    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "deduplicated-pandas-memory-source")

    assert snapshot_calls == 1
    assert logical._memory_source_ref_count_for_test() == 1
    physical = logical.to_physical_plan(connection)
    assert sorted(len(batches) for batches in physical.scan_split_batch_map().values()) == [1, 1]


def test_repeated_arrow_backed_pandas_source_is_snapshotted_once(connection, monkeypatch):
    from vane.datasource import _memory

    arrow_int = pd.ArrowDtype(pa.int64())
    source = pd.DataFrame(
        {
            "id": pd.Series([1, 2, 3], dtype=arrow_int),
            "value": pd.Series([10, 20, 30], dtype=arrow_int),
        }
    )
    snapshot_calls = 0
    original = _memory._as_arrow_table

    def count_snapshot(value, source_kind):
        nonlocal snapshot_calls
        snapshot_calls += 1
        assert source_kind == "arrow"
        return original(value, source_kind)

    monkeypatch.setattr(_memory, "_as_arrow_table", count_snapshot)
    left = connection.from_df(source).set_alias("left_source")
    right = connection.from_df(source).set_alias("right_source")
    relation = left.join(right, "left_source.id = right_source.id")

    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation, "deduplicated-arrow-backed-pandas-memory-source"
    )

    assert snapshot_calls == 1
    assert logical._memory_source_ref_count_for_test() == 1


def test_rebound_mutated_pandas_source_preserves_each_snapshot(connection, monkeypatch):
    from vane.datasource import _memory

    source = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
    left = connection.from_df(source).set_alias("left_source")
    source["value"] = [100, 200, 300]
    right = connection.from_df(source).set_alias("right_source")
    snapshots = []
    original = _memory._as_arrow_table

    def capture_snapshot(value, source_kind):
        if source_kind == "pandas":
            snapshots.append(tuple(value["value"]))
        return original(value, source_kind)

    monkeypatch.setattr(_memory, "_as_arrow_table", capture_snapshot)
    relation = left.union(right)

    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "versioned-pandas-memory-source")

    assert snapshots == [(10, 20, 30), (100, 200, 300)]
    assert logical._memory_source_ref_count_for_test() == 2


def test_rebound_mutated_arrow_backed_pandas_source_preserves_each_snapshot(connection, monkeypatch):
    from vane.datasource import _memory

    arrow_int = pd.ArrowDtype(pa.int64())
    source = pd.DataFrame(
        {
            "id": pd.Series([1, 2, 3], dtype=arrow_int),
            "value": pd.Series([10, 20, 30], dtype=arrow_int),
        }
    )
    left = connection.from_df(source).set_alias("left_source")
    source["value"] = pd.Series([100, 200, 300], dtype=arrow_int)
    right = connection.from_df(source).set_alias("right_source")
    snapshots = []
    original = _memory._as_arrow_table

    def capture_snapshot(value, source_kind):
        if source_kind == "arrow":
            snapshots.append(tuple(value["value"].to_pylist()))
        return original(value, source_kind)

    monkeypatch.setattr(_memory, "_as_arrow_table", capture_snapshot)
    relation = left.union(right)

    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "versioned-arrow-backed-pandas-memory-source")

    assert snapshots == [(10, 20, 30), (100, 200, 300)]
    assert logical._memory_source_ref_count_for_test() == 2


def test_pandas_snapshot_prunes_unreferenced_unconvertible_columns(connection, monkeypatch):
    from vane.datasource import _memory

    class Unconvertible:
        pass

    source = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "value": [10, 20, 30],
            "ignored": [Unconvertible(), Unconvertible(), Unconvertible()],
        }
    )
    snapshot_columns = []
    original = _memory._as_arrow_table

    def capture_snapshot(value, source_kind):
        if source_kind == "pandas":
            snapshot_columns.append(tuple(value.columns))
        return original(value, source_kind)

    monkeypatch.setattr(_memory, "_as_arrow_table", capture_snapshot)
    left = connection.from_df(source).project("id")
    right = connection.from_df(source).project("value")

    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(left.union(right), "pruned-pandas-memory-source")

    assert snapshot_columns == [("id", "value")]
    assert logical._memory_source_ref_count_for_test() == 1

    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    result = pa.concat_tables(list(runner.run_iter_tables(left.union(right))))
    assert sorted(result.column(0).to_pylist()) == [1, 2, 3, 10, 20, 30]


def test_large_arrow_memory_source_is_partitioned_without_growing_plan(connection, monkeypatch):
    row_count = 1_200_000
    source = pa.table(
        {
            "id": pa.array(range(row_count), type=pa.int64()),
            "value": pa.array(range(row_count), type=pa.int64()),
        }
    )
    relation = connection.from_arrow(source)
    put_partitions = []
    original_put = ray.put

    def capture_partition(value):
        if isinstance(value, pa.Table):
            put_partitions.append(value)
        return original_put(value)

    monkeypatch.setattr(ray, "put", capture_partition)

    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "partitioned-arrow-memory-source")
    physical = logical.to_physical_plan(connection)

    assert len(pickle.dumps(logical)) < 64 * 1024
    assert logical._memory_source_ref_count_for_test() == 2
    assert physical._memory_source_ref_count_for_test() == 2
    assert [len(batches) for batches in physical.scan_split_batch_map().values()] == [2]
    assert len(put_partitions) == 2
    assert sum(partition.get_total_buffer_size() for partition in put_partitions) == source.get_total_buffer_size()


def test_dictionary_arrow_partitions_trim_unused_dictionary_values(connection, monkeypatch):
    from vane.datasource import _memory

    dictionary = pa.array([f"category-{index}-" + "x" * 64 for index in range(8)])
    values = pa.DictionaryArray.from_arrays(pa.array(range(8), type=pa.int8()), dictionary)
    source = pa.Table.from_arrays([values], names=["category"])
    put_partitions = []
    original_put = ray.put

    def capture_partition(value):
        if isinstance(value, pa.Table):
            put_partitions.append(value)
        return original_put(value)

    monkeypatch.setattr(_memory, "_TARGET_PARTITION_BYTES", 1)
    monkeypatch.setattr(ray, "put", capture_partition)

    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        connection.from_arrow(source), "trimmed-dictionary-arrow-memory-source"
    )

    assert logical._memory_source_ref_count_for_test() == 8
    assert len(put_partitions) == 8
    partition_dictionaries = [partition.column(0).chunk(0).dictionary for partition in put_partitions]
    assert [len(partition_dictionary) for partition_dictionary in partition_dictionaries] == [1] * 8
    assert [
        partition_dictionary[0].as_py() for partition_dictionary in partition_dictionaries
    ] == dictionary.to_pylist()


def test_pandas_memory_scan_rewrite_preserves_join_cardinality(connection):
    small = connection.from_arrow(pa.table({"small_id": [1, 2, 3]})).set_alias("small_source")
    large = connection.from_df(pd.DataFrame({"large_id": range(10_000), "payload": range(10_000)})).set_alias(
        "large_source"
    )
    relation = small.join(large, "small_source.small_id = large_source.large_id")

    physical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation, "cardinality-preserving-memory-source"
    ).to_physical_plan(connection)

    cardinalities = physical._datasource_scan_cardinalities_for_test()
    assert cardinalities[("large_id", "payload")] == 10_000


def test_pandas_and_arrow_memory_relations_execute_through_ray(connection):
    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()

    for source_kind in ("pandas", "arrow"):
        relation = _memory_relation(connection, source_kind, 4).filter("id >= 2").project("id, value * 2")
        result = pa.concat_tables(list(runner.run_iter_tables(relation)))
        rows = sorted(zip(result.column(0).to_pylist(), result.column(1).to_pylist(), strict=True))
        assert rows == [(2, 40), (3, 60)]


def test_pandas_snapshot_preserves_bound_object_integer_type(connection):
    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    source = pd.DataFrame({"small_integer": pd.Series([1, 2, None], dtype=object)})
    relation = connection.from_df(source).project("small_integer + 1")

    result = pa.concat_tables(list(runner.run_iter_tables(relation)))

    assert result.column(0).to_pylist() == [2, 3, None]


@pytest.mark.parametrize("source_kind", ["dataset", "scanner"])
def test_lazy_arrow_source_requires_explicit_materialization(connection, source_kind):
    dataset = ds.dataset(pa.table({"id": [1, 2, 3]}))
    source = dataset if source_kind == "dataset" else dataset.scanner()
    relation = connection.from_arrow(source)

    with pytest.raises(TypeError, match="Materialize lazy or streaming Arrow sources explicitly"):
        vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, f"lazy-arrow-{source_kind}")
