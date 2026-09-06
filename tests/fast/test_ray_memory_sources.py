# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import pickle
import uuid
from datetime import time, timedelta, timezone

import numpy as np
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
from vane.datasource import _memory

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
    source = pa.table({"id": [1, 2, 3], "value": [10, 20, 30]})
    snapshot_calls = 0
    original = _memory._put_memory_partition

    def count_snapshot(value):
        nonlocal snapshot_calls
        if isinstance(value, pa.Table):
            snapshot_calls += 1
        return original(value)

    monkeypatch.setattr(_memory, "_put_memory_partition", count_snapshot)
    left = connection.from_arrow(source).set_alias("left_source")
    right = connection.from_arrow(source).set_alias("right_source")
    relation = left.join(right, "left_source.id = right_source.id")

    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "deduplicated-arrow-memory-source")

    assert snapshot_calls == 1
    assert logical._memory_source_ref_count_for_test() == 1
    physical = logical.to_physical_plan(connection)
    assert sorted(len(batches) for batches in physical.scan_split_batch_map().values()) == [1, 1]


def test_repeated_pandas_source_is_snapshotted_once(connection, monkeypatch):
    source = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
    snapshot_calls = 0
    original = _memory._put_memory_partition

    def count_snapshot(value):
        nonlocal snapshot_calls
        if isinstance(value, pa.Table):
            snapshot_calls += 1
        return original(value)

    monkeypatch.setattr(_memory, "_put_memory_partition", count_snapshot)
    left = connection.from_df(source).set_alias("left_source")
    right = connection.from_df(source).set_alias("right_source")
    relation = left.join(right, "left_source.id = right_source.id")

    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "deduplicated-pandas-memory-source")

    assert snapshot_calls == 1
    assert logical._memory_source_ref_count_for_test() == 1
    physical = logical.to_physical_plan(connection)
    assert sorted(len(batches) for batches in physical.scan_split_batch_map().values()) == [1, 1]


def test_repeated_arrow_backed_pandas_source_is_snapshotted_once(connection, monkeypatch):
    arrow_int = pd.ArrowDtype(pa.int64())
    source = pd.DataFrame(
        {
            "id": pd.Series([1, 2, 3], dtype=arrow_int),
            "value": pd.Series([10, 20, 30], dtype=arrow_int),
        }
    )
    snapshot_calls = 0
    original = _memory._put_memory_partition

    def count_snapshot(value):
        nonlocal snapshot_calls
        if isinstance(value, pa.Table):
            snapshot_calls += 1
        return original(value)

    monkeypatch.setattr(_memory, "_put_memory_partition", count_snapshot)
    left = connection.from_df(source).set_alias("left_source")
    right = connection.from_df(source).set_alias("right_source")
    relation = left.join(right, "left_source.id = right_source.id")

    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation, "deduplicated-arrow-backed-pandas-memory-source"
    )

    assert snapshot_calls == 1
    assert logical._memory_source_ref_count_for_test() == 1


def test_rebound_mutated_pandas_source_preserves_each_snapshot(connection, monkeypatch):
    source = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
    left = connection.from_df(source).set_alias("left_source")
    source["value"] = [100, 200, 300]
    right = connection.from_df(source).set_alias("right_source")
    snapshots = []
    original = _memory._put_memory_partition

    def capture_snapshot(value):
        if isinstance(value, pa.Table):
            snapshots.append(tuple(value["value"].to_pylist()))
        return original(value)

    monkeypatch.setattr(_memory, "_put_memory_partition", capture_snapshot)
    relation = left.union(right)

    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "versioned-pandas-memory-source")

    assert snapshots == [(10, 20, 30), (100, 200, 300)]
    assert logical._memory_source_ref_count_for_test() == 2


def test_rebound_mutated_arrow_backed_pandas_source_preserves_each_snapshot(connection, monkeypatch):
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
    original = _memory._put_memory_partition

    def capture_snapshot(value):
        if isinstance(value, pa.Table):
            snapshots.append(tuple(value["value"].to_pylist()))
        return original(value)

    monkeypatch.setattr(_memory, "_put_memory_partition", capture_snapshot)
    relation = left.union(right)

    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "versioned-arrow-backed-pandas-memory-source")

    assert snapshots == [(10, 20, 30), (100, 200, 300)]
    assert logical._memory_source_ref_count_for_test() == 2


def test_rebound_mutated_nested_dictionary_arrow_backed_pandas_preserves_each_snapshot(connection, monkeypatch):
    indices = pa.array([0, 1], type=pa.int8())
    offsets = pa.array([0, 1, 2], type=pa.int32())

    def nested_dictionary(values):
        dictionary = pa.DictionaryArray.from_arrays(indices, pa.array(values))
        return pa.ListArray.from_arrays(offsets, dictionary)

    original_array = nested_dictionary(["old-a", "old-b"])
    updated_array = nested_dictionary(["new-a", "new-b"])
    assert original_array.buffers() == updated_array.buffers()

    arrow_type = pd.ArrowDtype(original_array.type)
    source = pd.DataFrame({"value": pd.Series(original_array, dtype=arrow_type)})
    left = connection.from_df(source).set_alias("left_source")
    source["value"] = pd.Series(updated_array, dtype=arrow_type)
    right = connection.from_df(source).set_alias("right_source")
    snapshots = []
    original = _memory._put_memory_partition

    def capture_snapshot(value):
        if isinstance(value, pa.Table):
            snapshots.append(value["value"].to_pylist())
        return original(value)

    monkeypatch.setattr(_memory, "_put_memory_partition", capture_snapshot)
    relation = left.union(right)

    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation, "versioned-nested-dictionary-arrow-backed-pandas-memory-source"
    )

    assert snapshots == [[["old-a"], ["old-b"]], [["new-a"], ["new-b"]]]
    assert logical._memory_source_ref_count_for_test() == 2


@pytest.mark.parametrize("view_type", [pa.string_view(), pa.binary_view()])
@pytest.mark.parametrize("nested", [False, True])
def test_rebound_arrow_views_preserve_distinct_variadic_buffers(connection, monkeypatch, view_type, nested):
    original_array = pa.array(["sharedprefix-first"], type=view_type)
    buffers = original_array.buffers()
    updated_array = pa.Array.from_buffers(
        view_type, len(original_array), [buffers[0], buffers[1], pa.py_buffer(b"sharedprefix-other")]
    )
    # Both arrays share the validity and view descriptors, including the inline
    # string prefix. Only the out-of-line buffer changes between the two binds.
    if nested:
        offsets = pa.array([0, 1], type=pa.int32())
        original_array = pa.ListArray.from_arrays(offsets, original_array)
        updated_array = pa.ListArray.from_arrays(offsets, updated_array)
    arrow_type = pd.ArrowDtype(original_array.type)
    source = pd.DataFrame({"value": pd.Series(original_array, dtype=arrow_type)})
    left = connection.from_df(source)
    source["value"] = pd.Series(updated_array, dtype=arrow_type)
    right = connection.from_df(source)
    snapshots = []
    original_put = _memory._put_memory_partition

    def capture_snapshot(value):
        if isinstance(value, pa.Table):
            snapshots.append(value.column(0).to_pylist())
        return original_put(value)

    monkeypatch.setattr(_memory, "_put_memory_partition", capture_snapshot)
    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()

    result = pa.concat_tables(list(runner.run_iter_tables(left.union(right))))

    expected = original_array.to_pylist() + updated_array.to_pylist()
    assert sorted(result.column(0).to_pylist()) == sorted(expected)
    assert snapshots == [original_array.to_pylist(), updated_array.to_pylist()]


def test_pandas_snapshot_prunes_unreferenced_unconvertible_columns(connection, monkeypatch):
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
    original = _memory._put_memory_partition

    def capture_snapshot(value):
        if isinstance(value, pa.Table):
            snapshot_columns.append(tuple(value.column_names))
        return original(value)

    monkeypatch.setattr(_memory, "_put_memory_partition", capture_snapshot)
    left = connection.from_df(source).project("id")
    right = connection.from_df(source).project("value")

    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(left.union(right), "pruned-pandas-memory-source")

    assert snapshot_columns == [("id", "value")]
    assert logical._memory_source_ref_count_for_test() == 1

    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    result = pa.concat_tables(list(runner.run_iter_tables(left.union(right))))
    assert sorted(result.column(0).to_pylist()) == [1, 2, 3, 10, 20, 30]


def test_pandas_row_count_scan_does_not_snapshot_uuid_object_column(connection, monkeypatch):
    source = pd.DataFrame({"u": pd.Series([uuid.uuid4(), uuid.uuid4(), uuid.uuid4()], dtype=object)})
    snapshot_columns = []
    original = _memory._put_memory_partition

    def capture_snapshot(value):
        if isinstance(value, pa.Table):
            snapshot_columns.append(tuple(value.column_names))
        return original(value)

    monkeypatch.setattr(_memory, "_put_memory_partition", capture_snapshot)
    relation = connection.from_df(source).aggregate("count(*)")

    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    result = pa.concat_tables(list(runner.run_iter_tables(relation)))

    assert result.column(0).to_pylist() == [3]
    assert len(snapshot_columns) == 1
    assert len(snapshot_columns[0]) == 1
    assert snapshot_columns[0][0].startswith("__vane_row_count_")


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
    original_put = _memory._put_memory_partition

    def capture_partition(value):
        if isinstance(value, pa.Table):
            put_partitions.append(value)
        return original_put(value)

    monkeypatch.setattr(_memory, "_put_memory_partition", capture_partition)

    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "partitioned-arrow-memory-source")
    physical = logical.to_physical_plan(connection)

    assert len(pickle.dumps(logical)) < 64 * 1024
    assert logical._memory_source_ref_count_for_test() == 2
    assert physical._memory_source_ref_count_for_test() == 2
    assert [len(batches) for batches in physical.scan_split_batch_map().values()] == [2]
    assert len(put_partitions) == 2
    assert sum(partition.get_total_buffer_size() for partition in put_partitions) == source.get_total_buffer_size()


def test_single_partition_arrow_slice_drops_unreferenced_backing_buffers(connection, monkeypatch):
    source = pa.table({"value": pa.array(range(1_000_000), type=pa.int64())}).slice(500_000, 3)
    put_partitions = []
    original_put = _memory._put_memory_partition

    def capture_partition(value):
        if isinstance(value, pa.Table):
            put_partitions.append(value)
        return original_put(value)

    monkeypatch.setattr(_memory, "_put_memory_partition", capture_partition)

    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        connection.from_arrow(source), "materialized-single-arrow-partition"
    )

    assert logical._memory_source_ref_count_for_test() == 1
    assert len(put_partitions) == 1
    partition = put_partitions[0]
    assert partition.column(0).chunk(0).offset == 0
    assert partition.get_total_buffer_size() == 3 * pa.int64().bit_width // 8
    assert partition.column(0).chunk(0).buffers()[1].address != source.column(0).chunk(0).buffers()[1].address


@pytest.mark.parametrize("view_type", [pa.string_view(), pa.binary_view()])
def test_arrow_view_partitions_drop_unreferenced_variadic_buffers(connection, view_type):
    values = pa.array([f"{index:08d}" + "x" * 992 for index in range(100)], type=view_type)
    source = pa.table({"value": values}).slice(50, 1)

    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(connection.from_arrow(source), str(uuid.uuid4()))
    refs = next(iter(logical.__getstate__()[-1].values()))
    assert len(refs) == 1
    partition = ray.get(refs[0])

    assert isinstance(partition, pa.Table)
    assert partition.schema == source.schema
    assert partition.to_pydict() == source.to_pydict()
    assert partition.get_total_buffer_size() < 2048


def test_memory_partition_transport_preserves_unaligned_column_chunks():
    row_count = 1000
    table = pa.table(
        {
            "one_chunk": pa.array(range(row_count), type=pa.int64()),
            "many_chunks": pa.chunked_array([pa.array([i], type=pa.int64()) for i in range(row_count)]),
        }
    )
    buffers = []

    serialized = pickle.dumps(_memory._MemoryPartitionTransport(table), protocol=5, buffer_callback=buffers.append)
    restored = pickle.loads(serialized, buffers=buffers)

    assert restored.equals(table)
    assert [column.num_chunks for column in restored.columns] == [1, row_count]
    assert sum(buffer.raw().nbytes for buffer in buffers) <= table.nbytes * 2


def test_dictionary_arrow_partitions_trim_unused_dictionary_values(connection, monkeypatch):
    dictionary = pa.array([f"category-{index}-" + "x" * 64 for index in range(8)])
    values = pa.DictionaryArray.from_arrays(pa.array(range(8), type=pa.int8()), dictionary)
    source = pa.Table.from_arrays([values], names=["category"])
    put_partitions = []
    original_put = _memory._put_memory_partition

    def capture_partition(value):
        if isinstance(value, pa.Table):
            put_partitions.append(value)
        return original_put(value)

    monkeypatch.setattr(_memory, "_TARGET_PARTITION_BYTES", 1)
    monkeypatch.setattr(_memory, "_put_memory_partition", capture_partition)

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


@pytest.mark.parametrize("container_kind", ["list", "struct", "map"])
def test_nested_dictionary_partition_materialization_trims_unused_values(container_kind):
    dictionary = pa.array([f"category-{index}-" + "x" * 64 for index in range(3)])
    values = pa.DictionaryArray.from_arrays(pa.array(range(3), type=pa.int8()), dictionary)
    if container_kind == "list":
        array = pa.ListArray.from_arrays(pa.array(range(4), type=pa.int32()), values)
    elif container_kind == "struct":
        array = pa.StructArray.from_arrays([values], names=["category"])
    else:
        array = pa.MapArray.from_arrays(
            pa.array(range(4), type=pa.int32()),
            pa.array([f"key-{index}" for index in range(3)]),
            values,
        )

    column = pa.chunked_array([array])
    materialized = [_memory._materialize_partition_column(column.slice(index, 1)).chunk(0) for index in range(3)]
    if container_kind == "list":
        dictionaries = [item.values.dictionary for item in materialized]
    elif container_kind == "struct":
        dictionaries = [item.field("category").dictionary for item in materialized]
    else:
        dictionaries = [item.items.dictionary for item in materialized]

    assert [len(item) for item in dictionaries] == [1, 1, 1]
    assert [item[0].as_py() for item in dictionaries] == dictionary.to_pylist()


@pytest.mark.parametrize("value_kind", ["list", "struct"])
@pytest.mark.parametrize("chunked", [False, True])
def test_dictionary_partitions_support_nested_values_and_preserve_dictionary_order(value_kind, chunked):
    values = (
        [["first"], ["unused"], ["last"]]
        if value_kind == "list"
        else [
            {"label": "first"},
            {"label": "unused"},
            {"label": "last"},
        ]
    )
    dictionary = pa.array(values)
    array = pa.DictionaryArray.from_arrays(pa.array([2, 0, None, 2], type=pa.int8()), dictionary, ordered=True)
    chunks = [array]
    if chunked:
        chunks.append(
            pa.DictionaryArray.from_arrays(array.indices, dictionary.slice(0, 2).take([1, 0, 1]), ordered=True)
        )
    column = pa.chunked_array(chunks)
    materialized_column = _memory._materialize_partition_column(column)
    materialized = materialized_column.chunk(0)

    assert materialized.type == array.type
    assert materialized.to_pylist() == array.to_pylist()
    assert materialized.indices.to_pylist() == [1, 0, None, 1]
    assert materialized.dictionary.to_pylist() == [values[0], values[2]]
    assert materialized_column.to_pylist() == column.to_pylist()
    assert materialized_column.num_chunks == len(chunks)


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("value_kind", ["list", "string_view", "binary_view"])
def test_dictionary_memory_scans_preserve_nested_and_view_values(connection, chunked, value_kind):
    if value_kind == "list":
        dictionary = pa.array([["first"], ["unused"], ["last"]])
    else:
        view_type = pa.string_view() if value_kind == "string_view" else pa.binary_view()
        dictionary = pa.array(["long-first-value", "long-unused-value", "long-last-value"], type=view_type)
    array = pa.DictionaryArray.from_arrays(pa.array([2, 0, None, 2], type=pa.int8()), dictionary, ordered=True)
    chunks = [array]
    if chunked:
        reordered = pa.array(
            [dictionary[1].as_py(), dictionary[0].as_py(), dictionary[1].as_py()], type=dictionary.type
        )
        chunks.append(pa.DictionaryArray.from_arrays(array.indices, reordered, ordered=True))
    column = pa.chunked_array(chunks)
    source = pa.Table.from_arrays([column], names=["value"])  # noqa: F841 - replacement scan
    query = "SELECT CAST(value AS VARCHAR) AS value FROM source"
    expected = [row[0] for row in connection.execute(query).fetchall()]
    relation = connection.sql(query)
    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()

    result = pa.concat_tables(list(runner.run_iter_tables(relation)))

    assert result.column(0).to_pylist() == expected


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


def test_pandas_categorical_memory_source_preserves_enum_semantics(connection):
    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    source = pd.DataFrame(
        {"category": pd.Categorical(["red", "blue", None, "red"], categories=["red", "blue", "unused"], ordered=True)}
    )
    relation = connection.from_df(source).filter("category != 'unused'").project("category")

    result = pa.concat_tables(list(runner.run_iter_tables(relation)))

    assert sorted(value for value in result.column(0).to_pylist() if value is not None) == [
        "blue",
        "red",
        "red",
    ]


def test_categorical_snapshot_does_not_copy_all_categories_per_native_chunk(connection, monkeypatch):
    row_count = 16_384
    source = pd.DataFrame({"category": pd.Categorical([f"category-{i:06}" for i in range(row_count)])})
    snapshot_sizes = []
    original = _memory._snapshot_and_put_memory_source

    def capture_snapshot(table):
        snapshot_sizes.append(table.nbytes)
        return original(table)

    monkeypatch.setattr(_memory, "_snapshot_and_put_memory_source", capture_snapshot)
    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(connection.from_df(source), str(uuid.uuid4()))

    assert logical._memory_source_ref_count_for_test() == 1
    assert len(snapshot_sizes) == 1
    assert snapshot_sizes[0] < row_count * 32


def test_numpy_memory_source_uses_normalized_dictionary_columns(connection):
    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    source = {  # noqa: F841 - resolved by DuckDB's replacement scan
        "id": np.array([1, 2, 3]),
        "label": np.array(["b", "a", "b"]),
        "metric": np.array([1.0, np.nan, 3.0]),
    }
    relation = connection.sql("SELECT id, label, metric FROM source")

    result = pa.concat_tables(list(runner.run_iter_tables(relation)))
    rows = list(
        zip(
            result.column(0).to_pylist(),
            result.column(1).to_pylist(),
            result.column(2).to_pylist(),
            strict=True,
        )
    )

    assert rows == [(1, "b", 1.0), (2, "a", None), (3, "b", 3.0)]


def test_pandas_timezone_memory_source_truncates_nanoseconds_like_pandas_scan(connection):
    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    connection.execute("SET TimeZone = 'UTC'")
    base = pd.Timestamp("2024-01-01T00:00:00Z")
    before_epoch = pd.Timestamp("1970-01-01T00:00:00Z") - pd.Timedelta(1_999, unit="ns")
    source = pd.DataFrame(
        {
            "event_time": pd.Series(
                [before_epoch, base + pd.Timedelta(1, unit="ns"), base + pd.Timedelta(1_999, unit="ns"), pd.NaT],
                dtype="datetime64[ns, UTC]",
            )
        }
    )
    relation = connection.from_df(source).project("event_time")

    result = pa.concat_tables(list(runner.run_iter_tables(relation)))
    values = result.column(0).combine_chunks().cast(pa.int64()).to_pylist()

    assert result.column(0).type == pa.timestamp("us", tz="UTC")
    assert values == [-1, base.value // 1_000, (base.value + 1_999) // 1_000, None]


def test_pandas_timedelta_memory_source_matches_duckdb_interval_normalization(connection):
    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    source = pd.DataFrame(
        {
            "duration": pd.Series(
                [
                    pd.Timedelta(1, unit="ns"),
                    pd.Timedelta(days=31, microseconds=5, nanoseconds=999),
                    pd.Timedelta(-1_001, unit="ns"),
                    pd.NaT,
                ],
                dtype="timedelta64[ns]",
            )
        }
    )
    relation = connection.from_df(source).project("duration")

    result = pa.concat_tables(list(runner.run_iter_tables(relation)))

    assert result.column(0).to_pylist() == [
        pa.MonthDayNano((0, 0, 0)),
        pa.MonthDayNano((1, 1, 5_000)),
        pa.MonthDayNano((0, 0, -1_000)),
        None,
    ]


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


def test_pandas_varchar_object_memory_source_matches_pandas_scan(connection):
    class StringifiedValue:
        def __str__(self) -> str:
            return "custom-value"

    source = pd.DataFrame(
        {
            "value": pd.Series(
                [1, "text", StringifiedValue(), None, float("nan"), pd.NA, pd.NaT],
                dtype=object,
            )
        }
    )
    expected = [value for (value,) in connection.execute("SELECT value FROM source").fetchall()]
    relation = connection.from_df(source).project("value")

    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    result = pa.concat_tables(list(runner.run_iter_tables(relation)))

    assert expected == ["1", "text", "custom-value", None, None, None, None]
    assert result.column(0).to_pylist() == expected


@pytest.mark.parametrize("source_kind", ["pandas", "numpy"])
def test_uuid_object_memory_source_preserves_bound_type(connection, source_kind):
    values = [uuid.UUID("00000000-0000-0000-0000-000000000001"), None, uuid.UUID(int=2)]
    if source_kind == "pandas":
        source = pd.DataFrame({"u": pd.Series(values, dtype=object)})
        relation = connection.from_df(source).project("CAST(u AS VARCHAR) AS u")
    else:
        source = {"u": np.array(values, dtype=object)}  # noqa: F841 - resolved by DuckDB's replacement scan
        relation = connection.sql("SELECT CAST(u AS VARCHAR) AS u FROM source")

    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    result = pa.concat_tables(list(runner.run_iter_tables(relation)))

    assert result.column(0).to_pylist() == [str(values[0]), None, str(values[2])]


def test_pandas_map_object_memory_source_preserves_bound_type(connection):
    source = pd.DataFrame(
        {
            "attributes": pd.Series(
                [
                    {"key": ["a", "b"], "value": [1, 2]},
                    None,
                    {"key": [], "value": []},
                ],
                dtype=object,
            )
        }
    )
    relation = connection.from_df(source).project("attributes")

    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    result = pa.concat_tables(list(runner.run_iter_tables(relation)))

    assert result.column(0).to_pylist() == [[("a", 1), ("b", 2)], None, []]


def test_pandas_file_object_memory_source_preserves_bound_type(connection):
    values = [
        vane.File("memory://first", "text/plain", 1, 2, "sha256:first"),
        None,
        vane.File("memory://second"),
    ]
    source = pd.DataFrame({"file": pd.Series(values, dtype=object)})
    relation = connection.from_df(source).project("file_path(file) AS path")

    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    result = pa.concat_tables(list(runner.run_iter_tables(relation)))

    assert result.column(0).to_pylist() == ["memory://first", None, "memory://second"]


@pytest.mark.parametrize("source_kind", ["pandas", "numpy"])
@pytest.mark.parametrize("value_kind", ["list_file", "struct_file", "time_tz", "list_time_tz", "struct_uuid"])
def test_memory_snapshot_matches_native_nested_and_temporal_types(connection, source_kind, value_kind):
    file = vane.File("memory://nested", "text/plain", 1, 2, "sha256:nested")
    zoned_time = time(12, 34, 56, 123456, timezone(timedelta(hours=5, minutes=30)))
    cases = {
        "list_file": ([[file, None], [], None], "file_path(value[1])"),
        "struct_file": ([{"file": file}, {"file": None}, None], "file_path(value.file)"),
        "time_tz": ([zoned_time, time(1, 2, tzinfo=timezone.utc), None], "CAST(value AS VARCHAR)"),
        "list_time_tz": ([[zoned_time, None], [], None], "CAST(value AS VARCHAR)"),
        "struct_uuid": ([{"id": uuid.UUID(int=1)}, {"id": None}, None], "CAST(value AS VARCHAR)"),
    }
    values, expression = cases[value_kind]
    if source_kind == "pandas":
        source = pd.DataFrame({"value": pd.Series(values, dtype=object)})
    else:
        # Assign into a one-dimensional object array: np.array expands equal-length lists.
        column = np.empty(len(values), dtype=object)
        column[:] = values
        source = {"value": column}  # noqa: F841 - resolved by DuckDB's replacement scan
    query = f"SELECT {expression} AS result FROM source"
    expected = [row[0] for row in connection.execute(query).fetchall()]
    relation = connection.sql(query)

    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()
    result = pa.concat_tables(list(runner.run_iter_tables(relation)))

    assert result.column(0).to_pylist() == expected


@pytest.mark.parametrize("source_kind", ["dataset", "scanner", "reader", "capsule"])
def test_lazy_arrow_source_requires_explicit_materialization(connection, source_kind):
    table = pa.table({"id": [1, 2, 3]})
    dataset = ds.dataset(table)
    source = {
        "dataset": lambda: dataset,
        "scanner": dataset.scanner,
        "reader": table.to_reader,
        "capsule": table.__arrow_c_stream__,
    }[source_kind]()
    relation = connection.from_arrow(source)

    with pytest.raises(TypeError, match="Materialize lazy or streaming Arrow sources explicitly"):
        vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, f"lazy-arrow-{source_kind}")


@pytest.mark.parametrize("source_kind", ["pandas", "arrow", "record_batch"])
@pytest.mark.parametrize("row_count", [0, 3])
def test_memory_sources_preserve_empty_inputs_and_count_only_scans(connection, source_kind, row_count):
    table = pa.table({"id": pa.array(range(row_count), type=pa.int64())})
    if source_kind == "pandas":
        relation = connection.from_df(table.to_pandas())
    elif source_kind == "record_batch":
        relation = connection.from_arrow(pa.RecordBatch.from_arrays([table.column(0).combine_chunks()], ["id"]))
    else:
        relation = connection.from_arrow(table)
    runners.set_runner_ray(noop_if_initialized=True)
    runner = runners.get_or_create_runner()

    result = pa.concat_tables(list(runner.run_iter_tables(relation.aggregate("count(*)"))))

    assert result.column(0).to_pylist() == [row_count]


@pytest.mark.parametrize("source_kind", ["pandas", "arrow"])
@pytest.mark.parametrize("use_fte_queue", [False, True])
def test_memory_snapshot_survives_source_mutation_and_worker_plan_replay(connection, source_kind, use_fte_queue):
    values = np.array([10, 20, 30], dtype=np.int64)
    source = pd.DataFrame({"value": values}, copy=False) if source_kind == "pandas" else pa.table({"value": values})
    relation = connection.from_df(source) if source_kind == "pandas" else connection.from_arrow(source)
    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, str(uuid.uuid4()))
    values[:] = -1
    del source, relation
    gc.collect()

    @ray.remote
    def execute_on_worker(plan, use_fte_queue):
        from vane import _native

        resource_id = str(uuid.uuid4())
        execution_id = f"{resource_id}-retry-1"
        with vane.connect() as worker_connection:
            physical = plan.to_physical_plan(worker_connection)
            splits = physical.scan_split_batch_map()
            node_id, batches = next(iter(splits.items()))
            vane.ray_cxx._register_query_python_replay_state(resource_id, physical)
            try:
                clone = physical.clone(worker_connection)
                task = vane.ray_cxx._make_worker_task_from_plan_for_test(clone, execution_id, resource_id)
                worker_plan = task.plan()
                assert worker_plan._memory_source_ref_count_for_test() == 1
                if use_fte_queue:
                    queue = vane.ray_cxx.FteSplitQueue()
                    for batch in batches:
                        queue.add_scan_split(bytes(batch))
                    queue.no_more_splits()
                    options = {"fte_scan_source_queues": {str(node_id): queue}}
                else:
                    assert len(batches) == 1
                    options = {"scan_split_batch": {str(node_id): bytes(batches[0])}}
                result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(
                    worker_connection, worker_plan, **options
                )
                assert result.completion_status == "ok"
                return [value for table in result.partition_payloads for value in table.column(0).to_pylist()]
            finally:
                _native._release_datasource_factories_for_query(resource_id)
                vane.ray_cxx._cleanup_query_python_replay_state(resource_id)

    assert ray.get(execute_on_worker.remote(logical, use_fte_queue), timeout=60) == [10, 20, 30]


def test_memory_snapshot_refs_follow_query_lifetime_without_out_of_band_pickling(connection, monkeypatch):
    from ray._private import serialization

    # Ray rejects accidental cloudpickle(ObjectRef) calls outside normal task
    # serialization. Source splits must contain only IDs and partition indices.
    monkeypatch.setattr(serialization, "ALLOW_OUT_OF_BAND_OBJECT_REF_SERIALIZATION", False)
    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        connection.from_arrow(pa.table({"value": [41]})), str(uuid.uuid4())
    )
    physical = logical.to_physical_plan(connection)
    refs = logical.__getstate__()[-1]
    source_id = next(iter(refs))
    object_id = refs[source_id][0].hex()
    query_id = physical.resource_query_id()
    vane.ray_cxx._register_query_python_replay_state(query_id, physical)
    try:
        del refs, logical, physical
        gc.collect()
        assert ray.get(vane.ray_cxx._lookup_memory_source_ref(source_id, 0)).column(0).to_pylist() == [41]
    finally:
        vane.ray_cxx._cleanup_query_python_replay_state(query_id)
    gc.collect()
    with pytest.raises(vane.InvalidInputException, match="not retained by an active query"):
        vane.ray_cxx._lookup_memory_source_ref(source_id, 0)
    counts = ray._private.worker.global_worker.core_worker.get_all_reference_counts()
    assert counts.get(object_id, {"local": 0})["local"] == 0


def test_native_memory_snapshot_rejects_changed_storage_types(connection, monkeypatch):
    original = _memory._snapshot_and_put_memory_source

    def change_storage_type(table):
        return original(table.cast(pa.schema([("value", pa.float64())])))

    monkeypatch.setattr(_memory, "_snapshot_and_put_memory_source", change_storage_type)
    with pytest.raises(ValueError, match="Native memory snapshot changed column"):
        vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
            connection.from_df(pd.DataFrame({"value": [41]})), str(uuid.uuid4())
        )
