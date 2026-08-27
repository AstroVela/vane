# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
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
    from vane import _native

    def transform(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value * 3 for value in values], "label": [f"row-{value % 7}" for value in values]})

    monkeypatch.setenv("VANE_RUNNER", "local")
    output = tmp_path / "lazy_udf_arrow_copy.parquet"
    con = vane.connect()
    try:
        _native._reset_udf_executor_debug_counters()
        relation = con.sql("select i::BIGINT as x from range(4097) t(i)").map_batches(
            transform,
            schema={"y": vane.sqltypes.BIGINT, "label": vane.sqltypes.VARCHAR},
            execution_backend="subprocess_task",
            batch_size=4097,
            output_batch_size=4097,
        )
        relation.write_parquet(str(output), row_group_size=2048)
        rows = con.sql(f"select count(*), sum(y), count(distinct label) from read_parquet('{output}')").fetchone()
        counters = dict(_native._udf_executor_debug_counters())
    finally:
        con.close()

    assert rows == (4097, 25171968, 7)
    assert counters["udf_external_arrow_stream_export_count"] >= 1
    assert counters["udf_direct_arrow_table_conversion_count"] == 0
    assert counters["udf_python_export_under_client_context_lock_count"] == 0


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
