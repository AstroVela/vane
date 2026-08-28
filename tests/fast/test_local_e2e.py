# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import datetime
import sys

import pytest

import vane
from vane import runners as _runners
from vane.runners.local import set_runner_local


def _teardown_runner_if_supported():
    vane_mod = vane
    if vane_mod is not None and hasattr(vane_mod, "teardown_runner"):
        vane_mod.teardown_runner()


@pytest.fixture
def local_runner():
    _teardown_runner_if_supported()
    try:
        set_runner_local(num_workers=1, max_running_tasks=1)
        runner = _runners.get_or_create_runner()
    except Exception:
        pytest.skip("duckdb local FTE runner API not available in this environment")
    if getattr(runner, "name", None) != "local":
        pytest.skip(f"Local runner not active, got runner={getattr(runner, 'name', None)!r}")
    try:
        yield runner
    finally:
        _teardown_runner_if_supported()


def test_local_run_iter_is_not_implemented(local_runner):
    with pytest.raises(NotImplementedError, match="local FTE run_iter"):
        list(local_runner.run_iter(None))


def test_local_runner_write_parquet_e2e(local_runner, tmp_path, monkeypatch):
    src = tmp_path / "local_e2e_input.parquet"
    dst = tmp_path / "local_e2e_output.parquet"
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    setup_conn = vane.connect()
    try:
        setup_conn.sql("select i::integer as x, (i % 5)::integer as k from range(100) tbl(i)").write_parquet(str(src))
    finally:
        setup_conn.close()

    monkeypatch.setenv("VANE_RUNNER", "local")
    con = vane.connect()
    try:
        con.read_parquet(str(src)).filter("x >= 10 and x < 90").repartition(4).write_parquet(str(dst))
        rows = con.sql(f"select count(*), sum(x), min(k), max(k) from read_parquet('{dst}')").fetchone()
    finally:
        con.close()

    assert rows == (80, 3960, 0, 4)


def test_local_runner_writes_lazy_udf_arrow_output_through_native_copy(local_runner, tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from vane import _native

    def transform(table):
        values = table.column("x").to_pylist()
        chunk_size = 7

        def chunked(items, arrow_type):
            return pa.chunked_array(
                [
                    pa.array(items[offset : offset + chunk_size], type=arrow_type)
                    for offset in range(0, len(items), chunk_size)
                ]
            )

        def dictionary_floats(indices, dictionary):
            return pa.chunked_array(
                [
                    pa.DictionaryArray.from_arrays(
                        pa.array(indices[offset : offset + chunk_size], type=pa.int32()),
                        pa.array(dictionary, type=pa.float64()),
                    )
                    for offset in range(0, len(indices), chunk_size)
                ]
            )

        return pa.table(
            {
                "y": chunked([value * 3 for value in values], pa.int64()),
                "label": chunked([f"row-{value % 7}" for value in values], pa.string()),
                "score": chunked([float("nan")] * len(values), pa.float64()),
                "encoded_score": dictionary_floats([0] * len(values), [float("nan")]),
                "encoded_weight": dictionary_floats([value % 2 for value in values], [1.0, 2.0]),
                "weight": chunked([float(value) for value in values], pa.float32()),
                "samples": chunked([[float("nan"), float(value)] for value in values], pa.list_(pa.float64())),
                "attributes": chunked([[("value", float("nan"))] for _ in values], pa.map_(pa.string(), pa.float64())),
                "event_time": chunked([datetime.datetime(1970, 1, 1)] * len(values), pa.timestamp("us")),
                "event_time_tz": chunked(
                    [datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)] * len(values),
                    pa.timestamp("us", tz="UTC"),
                ),
                "clock": chunked([datetime.time()] * len(values), pa.time64("us")),
            }
        )

    monkeypatch.setenv("VANE_RUNNER", "local")
    output = tmp_path / "lazy_udf_arrow_copy.parquet"
    copy_result = {}
    original_run_write = local_runner.run_write

    def capture_run_write(relation):
        result = original_run_write(relation)
        copy_result.update(result)
        return result

    monkeypatch.setattr(local_runner, "run_write", capture_run_write)
    con = vane.connect()
    try:
        _native._reset_udf_executor_debug_counters()
        relation = con.sql("select i::BIGINT as x from range(4097) t(i)").map_batches(
            transform,
            schema={
                "y": vane.sqltypes.BIGINT,
                "label": vane.sqltypes.VARCHAR,
                "score": vane.sqltypes.DOUBLE,
                "encoded_score": vane.sqltypes.DOUBLE,
                "encoded_weight": vane.sqltypes.DOUBLE,
                "weight": vane.sqltypes.FLOAT,
                "samples": vane.list_type(vane.sqltypes.DOUBLE),
                "attributes": vane.map_type(vane.sqltypes.VARCHAR, vane.sqltypes.DOUBLE),
                "event_time": vane.sqltypes.TIMESTAMP,
                "event_time_tz": vane.sqltypes.TIMESTAMP_TZ,
                "clock": vane.sqltypes.TIME,
            },
            execution_backend="subprocess_task",
            batch_size=4097,
            output_batch_size=4097,
        )
        relation.write_parquet(str(output), row_group_size=64)
        rows = con.sql(f"select count(*), sum(y), count(distinct label) from read_parquet('{output}')").fetchone()
        counters = dict(_native._udf_executor_debug_counters())
        output_files = list(output.glob("*.parquet"))
        metadata = [pq.ParquetFile(path).metadata for path in output_files]
        bloom_metadata = [
            row
            for path in output_files
            for row in con.sql(f"select path_in_schema, bloom_filter_offset from parquet_metadata('{path}')").fetchall()
        ]
    finally:
        con.close()

    assert rows == (4097, 25171968, 7)
    assert metadata
    assert sum(file_metadata.num_rows for file_metadata in metadata) == 4097
    assert all(file_metadata.num_row_groups == (file_metadata.num_rows + 63) // 64 for file_metadata in metadata)
    assert any(file_metadata.num_row_groups > 1 for file_metadata in metadata)
    assert all("parquet-cpp-arrow" in file_metadata.created_by.lower() for file_metadata in metadata)
    assert all(file_metadata.format_version == "1.0" for file_metadata in metadata)
    assert bloom_metadata
    y_bloom_offsets = [offset for path, offset in bloom_metadata if path == "y"]
    repeated_bloom_offsets = [
        offset for path, offset in bloom_metadata if path.split(", ", 1)[0] in {"samples", "attributes"}
    ]
    assert y_bloom_offsets and all(offset is not None for offset in y_bloom_offsets)
    assert repeated_bloom_offsets and all(offset is None for offset in repeated_bloom_offsets)

    copy_files = copy_result["files"]
    assert sum(file_info["row_count"] for file_info in copy_files) == 4097
    assert all(file_info["file_size_bytes"] > 0 for file_info in copy_files)
    assert all(int(file_info["footer_size_bytes"]) > 0 for file_info in copy_files)
    assert all(file_info["column_statistics"] is not None for file_info in copy_files)
    for file_info in copy_files:
        statistics = file_info["column_statistics"]
        score_statistics = statistics[statistics.index('"score"') :]
        score_statistics = score_statistics[: score_statistics.index("}")]
        encoded_score_statistics = statistics[statistics.index('"encoded_score"') :]
        encoded_score_statistics = encoded_score_statistics[: encoded_score_statistics.index("}")]
        encoded_weight_statistics = statistics[statistics.index('"encoded_weight"') :]
        encoded_weight_statistics = encoded_weight_statistics[: encoded_weight_statistics.index("}")]
        weight_statistics = statistics[statistics.index('"weight"') :]
        weight_statistics = weight_statistics[: weight_statistics.index("}")]
        assert "has_nan=true" in score_statistics
        assert "has_nan=true" in encoded_score_statistics
        assert "has_nan=false" in encoded_weight_statistics
        assert "has_nan=false" in weight_statistics
        assert '"samples"."element"' in statistics
        assert '"samples"."list"."element"' not in statistics
        assert '"attributes"."key"' in statistics
        assert '"attributes"."value"' in statistics
        assert '"attributes"."key_value"' not in statistics
        samples_statistics = statistics[statistics.index('"samples"."element"') :]
        samples_statistics = samples_statistics[: samples_statistics.index("}")]
        attributes_statistics = statistics[statistics.index('"attributes"."value"') :]
        attributes_statistics = attributes_statistics[: attributes_statistics.index("}")]
        assert "has_nan=true" in samples_statistics
        assert "has_nan=true" in attributes_statistics
        event_time_statistics = statistics[statistics.index('"event_time"') :]
        event_time_statistics = event_time_statistics[: event_time_statistics.index("}")]
        event_time_tz_statistics = statistics[statistics.index('"event_time_tz"') :]
        event_time_tz_statistics = event_time_tz_statistics[: event_time_tz_statistics.index("}")]
        clock_statistics = statistics[statistics.index('"clock"') :]
        clock_statistics = clock_statistics[: clock_statistics.index("}")]
        assert "1970-01-01 00:00:00" in event_time_statistics
        assert ".000000" not in event_time_statistics
        assert "1970-01-01 00:00:00+00" in event_time_tz_statistics
        assert "Z" not in event_time_tz_statistics
        assert "00:00:00" in clock_statistics
        assert ".000000" not in clock_statistics

    assert counters["udf_external_arrow_stream_export_count"] >= 1
    assert counters["udf_direct_arrow_table_conversion_count"] == 0
    assert counters["udf_python_export_under_client_context_lock_count"] == 0


def test_local_runner_arrow_native_parquet_rejects_duckdb_extension_scalars(local_runner, tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")

    def transform(table):
        values = table.column("x").to_pylist()
        return pa.table({"huge": pa.array(values, type=pa.decimal128(38, 0))})

    monkeypatch.setenv("VANE_RUNNER", "local")
    output = tmp_path / "unsupported_arrow_extension.parquet"
    con = vane.connect()
    try:
        relation = con.sql("select i::BIGINT as x from range(4) t(i)").map_batches(
            transform,
            schema={"huge": vane.sqltypes.HUGEINT},
            execution_backend="subprocess_task",
            batch_size=4,
            output_batch_size=4,
        )
        with pytest.raises(ValueError, match="HUGEINT, UHUGEINT, and TIME WITH TIME ZONE"):
            relation.write_parquet(str(output))
    finally:
        con.close()

    assert not list(output.glob("*.parquet"))


def test_local_runner_arrow_native_parquet_rejects_arrow_bool8(local_runner, tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")

    def transform(table):
        values = [value % 2 for value in table.column("x").to_pylist()]
        return pa.table({"flag": pa.array(values, type=pa.bool8())})

    monkeypatch.setenv("VANE_RUNNER", "local")
    output = tmp_path / "unsupported_arrow_bool8.parquet"
    con = vane.connect()
    try:
        relation = con.sql("select i::BIGINT as x from range(4) t(i)").map_batches(
            transform,
            schema={"flag": vane.sqltypes.BOOLEAN},
            execution_backend="subprocess_task",
            batch_size=4,
            output_batch_size=4,
        )
        with pytest.raises(ValueError, match="arrow.bool8 is not supported by Arrow-native Parquet COPY"):
            relation.write_parquet(str(output))
    finally:
        con.close()

    assert not list(output.glob("*.parquet"))


def test_local_runner_arrow_native_parquet_rejects_dictionary_encoded_nested_values(
    local_runner, tmp_path, monkeypatch
):
    pa = pytest.importorskip("pyarrow")

    def transform(table):
        values = table.column("x").to_pylist()
        dictionary = pa.array(
            [{"reading": float("nan")}, {"reading": 1.0}],
            type=pa.struct([pa.field("reading", pa.float64())]),
        )
        indices = pa.array([value % 2 for value in values], type=pa.int32())
        return pa.table({"encoded": pa.DictionaryArray.from_arrays(indices, dictionary)})

    monkeypatch.setenv("VANE_RUNNER", "local")
    output = tmp_path / "unsupported_arrow_nested_dictionary.parquet"
    con = vane.connect()
    try:
        relation = con.sql("select i::BIGINT as x from range(4) t(i)").map_batches(
            transform,
            schema={"encoded": vane.struct_type({"reading": vane.sqltypes.DOUBLE})},
            execution_backend="subprocess_task",
            batch_size=4,
            output_batch_size=4,
        )
        with pytest.raises(
            ValueError, match="Dictionary-encoded nested values are not supported by Arrow-native Parquet COPY"
        ):
            relation.write_parquet(str(output))
    finally:
        con.close()

    assert not list(output.glob("*.parquet"))


def test_local_runner_arrow_native_parquet_v1_rejects_nanosecond_timestamps(local_runner, tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")

    def transform(table):
        values = [datetime.datetime(1970, 1, 1)] * table.num_rows
        return pa.table({"event_time": pa.array(values, type=pa.timestamp("ns"))})

    monkeypatch.setenv("VANE_RUNNER", "local")
    output = tmp_path / "unsupported_arrow_timestamp_ns.parquet"
    con = vane.connect()
    try:
        relation = con.sql("select i::BIGINT as x from range(4) t(i)").map_batches(
            transform,
            schema={"event_time": vane.sqltypes.TIMESTAMP_NS},
            execution_backend="subprocess_task",
            batch_size=4,
            output_batch_size=4,
        )
        with pytest.raises(
            ValueError, match="Nanosecond Arrow timestamps are not supported by Arrow-native Parquet V1 COPY"
        ):
            relation.write_parquet(str(output))
    finally:
        con.close()

    assert not list(output.glob("*.parquet"))


def test_local_runner_arrow_native_parquet_v1_rejects_uinteger(local_runner, tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")

    def transform(table):
        values = table.column("x").to_pylist()
        return pa.table({"value": pa.array(values, type=pa.uint32())})

    monkeypatch.setenv("VANE_RUNNER", "local")
    output = tmp_path / "unsupported_arrow_uinteger.parquet"
    con = vane.connect()
    try:
        relation = con.sql("select i::BIGINT as x from range(4) t(i)").map_batches(
            transform,
            schema={"value": vane.sqltypes.UINTEGER},
            execution_backend="subprocess_task",
            batch_size=4,
            output_batch_size=4,
        )
        with pytest.raises(ValueError, match="UINTEGER is not supported by Arrow-native Parquet V1 COPY"):
            relation.write_parquet(str(output))
    finally:
        con.close()

    assert not list(output.glob("*.parquet"))


def test_local_runner_arrow_native_parquet_writes_lz4_raw(local_runner, tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")

    def transform(table):
        return pa.table({"value": table.column("x")})

    monkeypatch.setenv("VANE_RUNNER", "local")
    output = tmp_path / "arrow_lz4_raw.parquet"
    con = vane.connect()
    try:
        relation = con.sql("select i::BIGINT as x from range(4) t(i)").map_batches(
            transform,
            schema={"value": vane.sqltypes.BIGINT},
            execution_backend="subprocess_task",
            batch_size=4,
            output_batch_size=4,
        )
        relation.write_parquet(str(output), compression="lz4")
        output_files = list(output.glob("*.parquet"))
        compression = {
            row[0]
            for path in output_files
            for row in con.sql(f"select distinct compression from parquet_metadata('{path}')").fetchall()
        }
    finally:
        con.close()

    assert output_files
    assert compression == {"LZ4_RAW"}


def test_local_runner_arrow_native_parquet_sizes_bloom_filter_from_actual_row_group(
    local_runner, tmp_path, monkeypatch
):
    pa = pytest.importorskip("pyarrow")

    def transform(table):
        return pa.table({"value": table.column("x")})

    monkeypatch.setenv("VANE_RUNNER", "local")
    output = tmp_path / "arrow_small_bloom_filter.parquet"
    con = vane.connect()
    try:
        relation = con.sql("select 42::BIGINT as x").map_batches(
            transform,
            schema={"value": vane.sqltypes.BIGINT},
            execution_backend="subprocess_task",
            batch_size=1,
            output_batch_size=1,
        )
        relation.write_parquet(str(output), row_group_size=122_880)
        output_files = list(output.glob("*.parquet"))
        bloom_metadata = [
            row
            for path in output_files
            for row in con.sql(
                f"select row_group_num_rows, bloom_filter_length from parquet_metadata('{path}') "
                "where path_in_schema = 'value'"
            ).fetchall()
        ]
    finally:
        con.close()

    assert output_files
    assert bloom_metadata
    assert all(row_count == 1 and 0 < bloom_length < 1024 for row_count, bloom_length in bloom_metadata)


def test_local_runner_arrow_native_parquet_rejects_schema_changes_across_rotated_files(
    local_runner, tmp_path, monkeypatch
):
    pa = pytest.importorskip("pyarrow")

    def transform(table):
        values = table.column("x").to_pylist()
        arrow_type = pa.time32("ms") if values[0] == 0 else pa.time64("us")
        return pa.table({"clock": pa.array([datetime.time()] * len(values), type=arrow_type)})

    monkeypatch.setenv("VANE_RUNNER", "local")
    output = tmp_path / "arrow_schema_rotation.parquet"
    escaped_output = str(output).replace("'", "''")
    con = vane.connect()
    try:
        relation = con.sql("select i::BIGINT as x from range(2) t(i)").map_batches(
            transform,
            schema={"clock": vane.sqltypes.TIME},
            execution_backend="subprocess_task",
            batch_size=1,
            output_batch_size=1,
            # Keep the incompatible schemas in separate task results and Arrow streams.
            task_input_max_bytes=1,
        )
        with pytest.raises(vane.InvalidInputException, match="Arrow schema changed during Arrow-native Parquet COPY"):
            relation.query(
                "arrow_schema_rotation",
                f"COPY arrow_schema_rotation TO '{escaped_output}' "
                "(FORMAT PARQUET, ROW_GROUP_SIZE 1, ROW_GROUPS_PER_FILE 1)",
            )
    finally:
        con.close()

    assert len(list(output.glob("*.parquet"))) <= 1


def test_local_runner_arrow_native_parquet_rejects_file_size_rotation(local_runner, tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")

    def transform(table):
        return pa.table({"y": table.column("x")})

    monkeypatch.setenv("VANE_RUNNER", "local")
    output = tmp_path / "unsupported_arrow_file_size.parquet"
    con = vane.connect()
    try:
        relation = con.sql("select i::BIGINT as x from range(4) t(i)").map_batches(
            transform,
            schema={"y": vane.sqltypes.BIGINT},
            execution_backend="subprocess_task",
            batch_size=4,
            output_batch_size=4,
        )
        with pytest.raises(ValueError, match="FILE_SIZE_BYTES is not supported by Arrow-native Parquet COPY"):
            relation.write_parquet(str(output), row_group_size=2, file_size_bytes=1024)
    finally:
        con.close()

    assert not list(output.glob("*.parquet"))


def test_local_runner_direct_target_per_thread_output_allows_sequential_partitions(tmp_path, monkeypatch):
    monkeypatch.delenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", raising=False)
    _teardown_runner_if_supported()
    try:
        set_runner_local(num_workers=1, max_running_tasks=1)
        runner = _runners.get_or_create_runner()
        if getattr(runner, "name", None) != "local":
            pytest.skip(f"Local runner not active, got runner={getattr(runner, 'name', None)!r}")

        src = tmp_path / "local_direct_input.parquet"
        dst = tmp_path / "local_direct_output"
        monkeypatch.setenv("VANE_RUNNER", "local-fast")
        setup_conn = vane.connect()
        try:
            setup_conn.sql("select i::integer as x, (i % 7)::integer as k from range(4096) tbl(i)").write_parquet(
                str(src)
            )
        finally:
            setup_conn.close()

        monkeypatch.setenv("VANE_RUNNER", "local")
        con = vane.connect()
        try:
            con.read_parquet(str(src)).repartition(8).write_parquet(str(dst), per_thread_output=True)
            rows = con.sql(f"select count(*), sum(x), min(k), max(k) from read_parquet('{dst}/*.parquet')").fetchone()
            files = list(dst.glob("*.parquet"))
        finally:
            con.close()

        assert rows == (4096, 8386560, 0, 6)
        assert len(files) >= 2
    finally:
        _teardown_runner_if_supported()


def test_local_runner_without_ray_import(local_runner, tmp_path, monkeypatch):
    for module_name in list(sys.modules):
        if module_name == "ray" or module_name.startswith("ray."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    orig_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ray" or name.startswith("ray."):
            raise ImportError("ray import is blocked in local-runner e2e test")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    src = tmp_path / "local_no_ray_input.parquet"
    dst = tmp_path / "local_no_ray_output.parquet"
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    setup_conn = vane.connect()
    try:
        setup_conn.sql("select i::integer as x from range(10) tbl(i)").write_parquet(str(src))
    finally:
        setup_conn.close()

    monkeypatch.setenv("VANE_RUNNER", "local")
    con = vane.connect()
    try:
        con.read_parquet(str(src)).write_parquet(str(dst))
        rows = con.sql(f"select count(*), sum(x) from read_parquet('{dst}')").fetchone()
    finally:
        con.close()

    assert rows == (10, 45)
