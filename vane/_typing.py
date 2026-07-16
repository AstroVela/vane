"""Convenient type aliases for the most commonly used DuckDB types.

These give shorter, more Pythonic names while preserving full compatibility
with the original DuckDB types (they are the same classes, just re-exported).

Usage::

    import vane

    conn: vane.Connection = vane.connect()
    rel: vane.Relation = conn.sql("SELECT 1")
"""

from __future__ import annotations

from duckdb import (
    DuckDBPyConnection as Connection,
)
from duckdb import (
    DuckDBPyRelation as Relation,
)
from duckdb import (
    Expression,
    Statement,
)

__all__ = [
    "Connection",
    "Expression",
    "Relation",
    "Statement",
]
