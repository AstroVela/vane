# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Public decorator API tests."""

from __future__ import annotations


def test_private_duckdb_func_enums_remain_available():
    from vane import _duckdb_func

    for name in ("NATIVE", "ARROW", "DEFAULT", "SPECIAL", "FunctionNullHandling", "PythonUDFType"):
        assert getattr(_duckdb_func, name) is not None


def test_relation_udf_methods_remain_available():
    import vane

    con = vane.connect()
    rel = con.sql("SELECT 1 AS x")

    assert callable(rel.map)
    assert callable(rel.map_batches)
