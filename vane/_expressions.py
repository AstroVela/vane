# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Public expression helper aliases for Vane."""

from __future__ import annotations

from typing import Any

import vane


def is_expression(value: Any) -> bool:
    return isinstance(value, vane.Expression)


def as_expression(value: Any) -> vane.Expression:
    if is_expression(value):
        return value
    return vane.ConstantExpression(value)


def col(name: str) -> vane.Expression:
    return vane.ColumnExpression(name)


def lit(value: Any) -> vane.Expression:
    return vane.ConstantExpression(value)


def sql_expr(sql: str) -> vane.Expression:
    return vane.SQLExpression(sql)


__all__ = ["as_expression", "col", "is_expression", "lit", "sql_expr"]
