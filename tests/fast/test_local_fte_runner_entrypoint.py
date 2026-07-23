# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys


def test_set_runner_local_entrypoint_in_subprocess():
    script = """
import os
import duckdb.runners as runners
from duckdb.runners.local import LocalRunner

os.environ.pop("VANE_RUNNER", None)
runner = runners.set_runner_local(num_workers=1, max_running_tasks=1)
assert isinstance(runner, LocalRunner)
assert runner.name == "local"
assert os.environ["VANE_RUNNER"] == "local"
assert os.environ["VANE_LOCAL_FTE_WORKERS"] == "1"
assert runner.max_running_tasks == 1
assert runners.get_or_infer_runner_type() == "local"
try:
    runner.run_iter(None)
except NotImplementedError as exc:
    assert "local FTE run_iter" in str(exc)
else:
    raise AssertionError("local FTE run_iter should be hidden until streaming is wired")
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_get_or_create_runner_accepts_local_env_in_subprocess():
    script = """
import os
import duckdb.runners as runners

os.environ["VANE_RUNNER"] = "local"
runner = runners.get_or_create_runner()
assert runner.name == "local"
assert runners.get_or_infer_runner_type() == "local"
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_get_or_create_runner_rejects_native_fte_env_in_subprocess():
    script = """
import os
import duckdb
import duckdb.runners as runners

os.environ["VANE_RUNNER"] = "native-fte"
try:
    runners.get_or_create_runner()
except duckdb.InvalidInputException as exc:
    assert "Please use 'local' or 'ray'" in str(exc)
else:
    raise AssertionError("native-fte should no longer be a public runner")
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_local_runner_preloads_arrow_dataset_imports():
    from duckdb.runners.local.runner import _preload_arrow_dataset_imports

    _preload_arrow_dataset_imports()
    _preload_arrow_dataset_imports()


def test_local_runner_rejects_invalid_num_workers():
    from duckdb.runners.local import _normalize_num_workers
    from duckdb.runners.local.runner import _normalize_num_workers as normalize_runner

    for normalize in (_normalize_num_workers, normalize_runner):
        for value in (0, -1, 1.5, True, "2"):
            try:
                normalize(value)
            except ValueError as exc:
                assert "num_workers must be a positive integer" in str(exc)
            else:
                raise AssertionError(f"expected invalid num_workers for {value!r}")


def test_local_runner_smoke_writes_parquet_in_subprocess():
    script = """
import pathlib
import tempfile

import duckdb
from duckdb.runners.local import set_runner_local

tmp = pathlib.Path(tempfile.mkdtemp())
src = tmp / "input.parquet"
dst = tmp / "output.parquet"

setup_conn = duckdb.connect()
setup_conn.execute(f"COPY (SELECT i::integer as x FROM range(3) tbl(i)) TO '{src}' (FORMAT PARQUET)")

set_runner_local(num_workers=1, max_running_tasks=1)
conn = duckdb.connect()
conn.sql(f"select * from read_parquet('{src}')").write_parquet(str(dst))

assert dst.exists()
assert sorted(row[0] for row in conn.read_parquet(str(dst)).fetchall()) == [0, 1, 2]
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_local_runner_repartition_write_uses_local_exchange_node_in_subprocess():
    script = """
import pathlib
import tempfile

import duckdb
from duckdb.runners.local import set_runner_local

tmp = pathlib.Path(tempfile.mkdtemp())
src = tmp / "input.parquet"
dst = tmp / "output.parquet"

setup_conn = duckdb.connect()
setup_conn.execute(
    f"COPY (SELECT i::integer as x, (i % 3)::integer as k FROM range(20) tbl(i)) TO '{src}' (FORMAT PARQUET)"
)

set_runner_local(num_workers=1, max_running_tasks=1)
conn = duckdb.connect()
conn.read_parquet(str(src)).repartition(4).write_parquet(str(dst))

rows = conn.sql(f"select count(*), sum(x) from read_parquet('{dst}')").fetchone()
assert rows == (20, 190)
"""
    subprocess.run([sys.executable, "-c", script], check=True)
