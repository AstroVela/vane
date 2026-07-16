"""Public decorator API tests."""

from __future__ import annotations

import importlib
import sys

import pytest


def test_decorator_shortcuts_are_not_public():
    import vane

    removed_names = {
        "method",
        "is_vane_cls",
        "is_vane_func",
        "get_cls_options",
        "get_func_options",
    }

    for name in removed_names:
        assert name not in vane.__all__
        assert not hasattr(vane, name)


def test_vane_func_is_public_expression_udf_factory():
    import vane

    assert callable(vane.func)
    assert callable(vane.func.batch)
    assert callable(vane.cls)
    assert callable(vane.cls.batch)
    assert "func" in vane.__all__
    assert "function" not in vane.__all__
    assert not hasattr(vane, "function")
    assert "cls" in vane.__all__

    for removed_name in ("NATIVE", "ARROW", "DEFAULT", "SPECIAL"):
        assert not hasattr(vane.func, removed_name)


def test_import_vane_func_submodule_is_blocked():
    import vane

    sys.modules.pop("vane.func", None)

    with pytest.raises(ModuleNotFoundError, match=r"No module named 'vane\.func'"):
        importlib.import_module("vane.func")

    assert callable(vane.func)


def test_duckdb_func_remains_available():
    import duckdb.func as duckdb_func

    for name in ("NATIVE", "ARROW", "DEFAULT", "SPECIAL", "FunctionNullHandling", "PythonUDFType"):
        assert getattr(duckdb_func, name) is not None


def test_relation_udf_methods_remain_available():
    import duckdb

    con = duckdb.connect()
    rel = con.sql("SELECT 1 AS x")

    assert callable(rel.map)
    assert callable(rel.map_batches)
