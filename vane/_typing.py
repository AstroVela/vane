# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Convenient aliases for Vane's native connection and relation types.

These are the same classes as :class:`vane.DuckDBPyConnection` and
:class:`vane.DuckDBPyRelation`. The official ``duckdb`` distribution exposes
distinct native classes.

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
