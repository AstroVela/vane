"""Relational-API expression construction. Informational, out of the binding gate. See benchmarks/README.md.

This is expression *construction* (ColumnExpression / ConstantExpression / operator overloads), not the
binding-pressure surface the rest of the suite targets. Kept because it carries a real signal (a measured ~35%
construction delta at the cutover), but never part of the gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import duckdb

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

pytestmark = pytest.mark.informational


def test_expr_many(benchmark: BenchmarkFixture) -> None:
    def run() -> int:
        out = []
        for i in range(2000):
            col = duckdb.ColumnExpression(f"col_{i}")
            const = duckdb.ConstantExpression(i)
            out.append(((col + const) * duckdb.ConstantExpression(2)).alias(f"a{i}"))
        return len(out)

    benchmark(run)
