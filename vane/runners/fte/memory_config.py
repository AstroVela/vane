# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any


class DuckDBMemoryLimitError(RuntimeError):
    pass


def duckdb_memory_limit_sql(memory_bytes: int | None) -> str | None:
    if memory_bytes is None:
        return None
    amount = int(memory_bytes)
    if amount <= 0:
        raise DuckDBMemoryLimitError(f"DuckDB memory limit must be > 0, got {amount}")
    return f"SET memory_limit='{amount}B'"


def apply_duckdb_memory_limit(conn: Any, memory_bytes: int | None) -> None:
    sql = duckdb_memory_limit_sql(memory_bytes)
    if sql is None:
        return
    try:
        conn.execute(sql)
    except Exception as exc:
        raise DuckDBMemoryLimitError(f"failed to apply DuckDB memory limit {memory_bytes}") from exc
