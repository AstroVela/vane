# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from vane.runners.fte.memory_config import (
    apply_duckdb_memory_limit,
    duckdb_memory_limit_sql,
)
from vane.runners.local.runner import _InProcessFragmentExecutor
from vane.runners.ray.worker import _configure_ray_worker_conn


class _FakeConn:
    def __init__(self):
        self.sql = []

    def execute(self, sql):
        self.sql.append(sql)


def test_duckdb_memory_limit_sql_formats_integer_bytes():
    assert duckdb_memory_limit_sql(1234) == "SET memory_limit='1234B'"
    assert duckdb_memory_limit_sql(None) is None


def test_apply_duckdb_memory_limit_executes_sql():
    conn = _FakeConn()
    apply_duckdb_memory_limit(conn, 4096)

    assert conn.sql == ["SET memory_limit='4096B'"]


def test_local_executor_configures_duckdb_memory_limit_from_env(monkeypatch):
    monkeypatch.setenv("VANE_DUCKDB_MEMORY_BUDGET_BYTES", "2048")
    conn = _FakeConn()

    _InProcessFragmentExecutor._configure_conn(conn)

    assert conn.sql[0] == "SET memory_limit='2048B'"


def test_ray_worker_configures_explicit_duckdb_memory_limit():
    conn = _FakeConn()

    _configure_ray_worker_conn(conn, 8192)

    assert conn.sql[0] == "SET memory_limit='8192B'"
