# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
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


@pytest.mark.parametrize("execution_backend", ["subprocess_task", "subprocess_actor"])
def test_local_runner_terminal_udf_writes_parquet_with_arrow(local_runner, tmp_path, monkeypatch, execution_backend):
    src = tmp_path / f"terminal_udf_input_{execution_backend}.parquet"
    dst = tmp_path / f"terminal_udf_output_{execution_backend}.parquet"
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    setup_conn = vane.connect()
    try:
        setup_conn.sql("select i::integer as x from range(257) tbl(i)").write_parquet(str(src))
    finally:
        setup_conn.close()

    def transform(table):
        return pa.table(
            {
                "x": table["x"],
                "doubled": pc.multiply(table["x"], pa.scalar(2, type=pa.int32())),
            }
        )

    class Transform:
        def __call__(self, table):
            return transform(table)

    monkeypatch.setenv("VANE_RUNNER", "local")
    con = vane.connect()
    try:
        udf_options = {"actor_number": 1, "gpus": 0.0} if execution_backend == "subprocess_actor" else {}
        udf = Transform if execution_backend == "subprocess_actor" else transform
        relation = (
            con.read_parquet(str(src))
            .repartition(4)
            .map_batches(
                udf,
                schema={"x": vane.sqltypes.INTEGER, "doubled": vane.sqltypes.INTEGER},
                execution_backend=execution_backend,
                batch_size=64,
                **udf_options,
            )
        )
        relation.write_parquet(str(dst), compression="zstd", row_group_size=32)
        rows = con.sql(f"select count(*), sum(x), sum(doubled) from read_parquet('{dst}')").fetchone()
    finally:
        con.close()

    parquet_files = [dst] if dst.is_file() else sorted(dst.rglob("*.parquet"))
    parquet_metadata = [pq.ParquetFile(path).metadata for path in parquet_files]
    assert rows == (257, 32896, 65792)
    assert len(parquet_files) >= 4
    assert all(metadata.num_rows > 0 for metadata in parquet_metadata)
    assert all("parquet-cpp-arrow" in str(metadata.created_by).lower() for metadata in parquet_metadata)
    assert all(metadata.format_version == "1.0" for metadata in parquet_metadata)
    assert all(
        0 < metadata.row_group(index).num_rows <= 32
        for metadata in parquet_metadata
        for index in range(metadata.num_row_groups)
    )
    assert all(
        metadata.row_group(index).column(0).compression == "ZSTD"
        for metadata in parquet_metadata
        for index in range(metadata.num_row_groups)
    )


@pytest.mark.parametrize("execution_backend", ["subprocess_task", "subprocess_actor"])
def test_local_runner_terminal_flat_map_streams_to_parquet(local_runner, tmp_path, monkeypatch, execution_backend):
    src = tmp_path / f"terminal_flat_map_input_{execution_backend}.parquet"
    dst = tmp_path / f"terminal_flat_map_output_{execution_backend}.parquet"
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    setup_conn = vane.connect()
    try:
        setup_conn.sql("select i::integer as x from range(5) tbl(i)").write_parquet(str(src))
    finally:
        setup_conn.close()

    def expand(row):
        for value in range(row["x"] + 1):
            yield {"x": row["x"], "value": value}

    class Expand:
        def __call__(self, row):
            yield from expand(row)

    monkeypatch.setenv("VANE_RUNNER", "local")
    con = vane.connect()
    try:
        udf_options = {"actor_number": 1, "gpus": 0.0} if execution_backend == "subprocess_actor" else {}
        udf = Expand if execution_backend == "subprocess_actor" else expand
        relation = con.read_parquet(str(src)).flat_map(
            udf,
            schema={"x": vane.sqltypes.INTEGER, "value": vane.sqltypes.INTEGER},
            execution_backend=execution_backend,
            **udf_options,
        )
        relation.write_parquet(str(dst), row_group_size=4)
        rows = con.sql(f"select count(*), sum(x), sum(value) from read_parquet('{dst}')").fetchone()
    finally:
        con.close()

    assert rows == (15, 40, 20)
    parquet_files = [dst] if dst.is_file() else sorted(dst.rglob("*.parquet"))
    assert parquet_files
    assert all("parquet-cpp-arrow" in str(pq.ParquetFile(path).metadata.created_by).lower() for path in parquet_files)


@pytest.mark.parametrize("execution_backend", ["subprocess_task", "subprocess_actor"])
def test_local_runner_terminal_flat_map_writes_empty_parquet(local_runner, tmp_path, monkeypatch, execution_backend):
    src = tmp_path / f"terminal_empty_flat_map_input_{execution_backend}.parquet"
    dst = tmp_path / f"terminal_empty_flat_map_output_{execution_backend}.parquet"
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    setup_conn = vane.connect()
    try:
        setup_conn.sql("select i::integer as x from range(5) tbl(i)").write_parquet(str(src))
    finally:
        setup_conn.close()

    def drop_all(row):
        if False:
            yield row

    class DropAll:
        def __call__(self, row):
            if False:
                yield row

    monkeypatch.setenv("VANE_RUNNER", "local")
    con = vane.connect()
    try:
        udf_options = {"actor_number": 1, "gpus": 0.0} if execution_backend == "subprocess_actor" else {}
        udf = DropAll if execution_backend == "subprocess_actor" else drop_all
        relation = con.read_parquet(str(src)).flat_map(
            udf,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend=execution_backend,
            **udf_options,
        )
        relation.write_parquet(str(dst))
        rows = con.sql(f"select count(*) from read_parquet('{dst}')").fetchone()
    finally:
        con.close()

    parquet_files = [dst] if dst.is_file() else sorted(dst.rglob("*.parquet"))
    assert rows == (0,)
    assert parquet_files
    assert all("parquet-cpp-arrow" in str(pq.ParquetFile(path).metadata.created_by).lower() for path in parquet_files)


@pytest.mark.parametrize("execution_backend", ["subprocess_task", "subprocess_actor"])
def test_local_runner_terminal_udf_writes_empty_parquet(local_runner, tmp_path, monkeypatch, execution_backend):
    src = tmp_path / f"terminal_udf_empty_input_{execution_backend}.parquet"
    dst = tmp_path / f"terminal_udf_empty_output_{execution_backend}.parquet"
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    setup_conn = vane.connect()
    try:
        setup_conn.sql("select i::integer as x from range(0) tbl(i)").write_parquet(str(src))
    finally:
        setup_conn.close()

    def transform(table):
        raise AssertionError("the UDF must not be called for empty input")

    class Transform:
        def __call__(self, table):
            raise AssertionError("the UDF must not be called for empty input")

    monkeypatch.setenv("VANE_RUNNER", "local")
    con = vane.connect()
    try:
        udf_options = {"actor_number": 1, "gpus": 0.0} if execution_backend == "subprocess_actor" else {}
        udf = Transform if execution_backend == "subprocess_actor" else transform
        relation = con.read_parquet(str(src)).map_batches(
            udf,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend=execution_backend,
            **udf_options,
        )
        relation.write_parquet(str(dst))
        rows = con.sql(f"select count(*) from read_parquet('{dst}')").fetchone()
    finally:
        con.close()

    parquet_files = [dst] if dst.is_file() else sorted(dst.rglob("*.parquet"))
    assert rows == (0,)
    assert len(parquet_files) == 1
    assert "parquet-cpp-arrow" in str(pq.ParquetFile(parquet_files[0]).metadata.created_by).lower()
    assert pq.read_schema(parquet_files[0]).names == ["x"]


def test_local_runner_terminal_udf_writes_through_local_staging(local_runner, tmp_path, monkeypatch):
    src = tmp_path / "terminal_udf_staging_input.parquet"
    dst = tmp_path / "terminal_udf_staging_output.parquet"
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    setup_conn = vane.connect()
    try:
        setup_conn.sql("select i::integer as x from range(5) tbl(i)").write_parquet(str(src))
    finally:
        setup_conn.close()

    def identity(table):
        return table

    monkeypatch.setenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", "1")
    monkeypatch.setenv("VANE_RUNNER", "local")
    con = vane.connect()
    try:
        relation = con.read_parquet(str(src)).map_batches(
            identity,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="subprocess_task",
            batch_size=2,
        )
        relation.write_parquet(str(dst))
        rows = con.sql(f"select x from read_parquet('{dst}') order by x").fetchall()
    finally:
        con.close()

    parquet_files = [dst] if dst.is_file() else sorted(dst.rglob("*.parquet"))
    assert rows == [(0,), (1,), (2,), (3,), (4,)]
    assert parquet_files
    assert all("parquet-cpp-arrow" in str(pq.ParquetFile(path).metadata.created_by).lower() for path in parquet_files)
    assert not (tmp_path / "terminal_udf_staging_output.parquet.duckdb_staging").exists()


@pytest.mark.parametrize("execution_backend", ["subprocess_task", "subprocess_actor"])
def test_local_runner_terminal_scalar_udf_writes_parquet(local_runner, tmp_path, monkeypatch, execution_backend):
    src = tmp_path / f"terminal_scalar_udf_input_{execution_backend}.parquet"
    dst = tmp_path / f"terminal_scalar_udf_output_{execution_backend}.parquet"
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    setup_conn = vane.connect()
    try:
        setup_conn.sql("select i::integer as x from range(5) tbl(i)").write_parquet(str(src))
    finally:
        setup_conn.close()

    def plus_one(value):
        return value + 1

    class PlusOne:
        def __call__(self, value):
            return value + 1

    monkeypatch.setenv("VANE_RUNNER", "local")
    con = vane.connect()
    try:
        udf_options = {"actor_number": 1, "gpus": 0.0} if execution_backend == "subprocess_actor" else {}
        udf = PlusOne if execution_backend == "subprocess_actor" else plus_one
        relation = con.read_parquet(str(src)).map(
            udf,
            return_type=vane.sqltypes.INTEGER,
            execution_backend=execution_backend,
            **udf_options,
        )
        relation.write_parquet(str(dst))
        rows = con.sql(f"select x, value from read_parquet('{dst}') order by x").fetchall()
    finally:
        con.close()

    assert rows == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]


@pytest.mark.parametrize(
    ("write_options", "message"),
    [
        ({"per_thread_output": True, "filename_pattern": "custom_{uuid}"}, "FILENAME_PATTERN"),
        ({"append": True}, "APPEND"),
        ({"field_ids": "auto"}, "FIELD_IDS"),
        ({"partition_by": ["x"], "write_partition_columns": True}, "PARTITION_BY"),
        ({"file_size_bytes": "1MB"}, "file rotation"),
    ],
)
def test_local_runner_terminal_udf_rejects_unsupported_copy_options(
    local_runner,
    tmp_path,
    monkeypatch,
    write_options,
    message,
):
    src = tmp_path / "terminal_udf_filename_pattern_input.parquet"
    dst = tmp_path / "terminal_udf_filename_pattern_output"
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    setup_conn = vane.connect()
    try:
        setup_conn.sql("select i::integer as x from range(1) tbl(i)").write_parquet(str(src))
    finally:
        setup_conn.close()

    def identity(table):
        return table

    monkeypatch.setenv("VANE_RUNNER", "local")
    con = vane.connect()
    try:
        relation = con.read_parquet(str(src)).map_batches(
            identity,
            schema={"x": vane.sqltypes.INTEGER},
            execution_backend="subprocess_task",
        )
        with pytest.raises(ValueError, match=rf"does not support {message}"):
            relation.write_parquet(str(dst), **write_options)
    finally:
        con.close()


def test_local_runner_rejects_nonterminal_udf_arrow_parquet_output(local_runner, tmp_path, monkeypatch):
    src = tmp_path / "nonterminal_udf_input.parquet"
    dst = tmp_path / "nonterminal_udf_output.parquet"
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    setup_conn = vane.connect()
    try:
        setup_conn.sql("select i::integer as x from range(2) tbl(i)").write_parquet(str(src))
    finally:
        setup_conn.close()

    def identity(table):
        return table

    monkeypatch.setenv("VANE_RUNNER", "local")
    con = vane.connect()
    try:
        relation = (
            con.read_parquet(str(src))
            .map_batches(
                identity,
                schema={"x": vane.sqltypes.INTEGER},
                execution_backend="subprocess_task",
            )
            .filter("x > 0")
        )
        with pytest.raises(ValueError, match="requires the UDF to be the terminal relational operation"):
            relation.write_parquet(str(dst))
    finally:
        con.close()


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
