# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Convenient type aliases for the most commonly used DuckDB types.

These give shorter, more Pythonic names while preserving full compatibility
with the original DuckDB types (they are the same classes, just re-exported).

Usage::

    import vane

    conn: vane.Connection = vane.connect()
    rel: vane.Relation = conn.sql("SELECT 1")
"""

from __future__ import annotations

from vane import (
    DuckDBPyConnection as Connection,
)
from vane import (
    DuckDBPyRelation as Relation,
)
from vane import (
    Expression,
    Statement,
)

__all__ = [
    "Connection",
    "Expression",
    "Relation",
    "Statement",
]
