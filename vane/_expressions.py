"""Public expression helper aliases for Vane."""

from __future__ import annotations

from typing import Any

import duckdb


def is_expression(value: Any) -> bool:
    return isinstance(value, duckdb.Expression)


def as_expression(value: Any) -> duckdb.Expression:
    if is_expression(value):
        return value
    return duckdb.ConstantExpression(value)


def col(name: str) -> duckdb.Expression:
    return duckdb.ColumnExpression(name)


def lit(value: Any) -> duckdb.Expression:
    return duckdb.ConstantExpression(value)


def sql_expr(sql: str) -> duckdb.Expression:
    return duckdb.SQLExpression(sql)


__all__ = ["as_expression", "col", "is_expression", "lit", "sql_expr"]
