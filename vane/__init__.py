# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Vane — distributed DuckDB powered by Ray.

Vane embeds a private fork of DuckDB and adds distributed execution
capabilities via Ray. DuckDB's Python API is re-exported from ``vane`` without
claiming the public ``duckdb`` package name, so Vane and the official DuckDB
distribution can be installed together.

Quick start::

    import vane

    # Programmatic configuration (optional — env vars also work)
    vane.configure(runner="ray", ray_scan_task_size_grouping=False)

    conn: vane.Connection = vane.connect()
    rel: vane.Relation = conn.sql("SELECT 42")
    rel.show()

Submodules such as ``vane.runners``, ``vane.experimental``, and
``vane.sqltypes`` are part of the Vane package.
"""

import importlib as _importlib
import sys as _sys

import _vane_duckdb as _native

from vane import _duckdb_api as _duckdb_api
from vane._duckdb_api import *  # noqa: F403
from vane._ray_progress_env import configure_ray_progress_logging_defaults as _configure_ray_progress_logging_defaults
from vane._vane_version import get_vane_version

_configure_ray_progress_logging_defaults()

# Install native aliases only after `_vane_duckdb` has finished initializing.
# Importing the public package from inside the extension initializer leaves
# these names unset when applications import `_vane_duckdb` before `vane`.
ray_cxx = _native.ray_cxx
vane_runners_cpp = _native
vane_runners = _native
_sys.modules["vane.ray_cxx"] = ray_cxx
_sys.modules["vane.vane_runners_cpp"] = vane_runners_cpp
_sys.modules["vane.vane_runners"] = vane_runners

# ---------------------------------------------------------------------------
# Version info
# ---------------------------------------------------------------------------

__duckdb_version__ = _duckdb_api.__duckdb_version__
__vane_version__ = get_vane_version()
__version__ = __vane_version__
version = _duckdb_api.version

# ---------------------------------------------------------------------------
# Vane-specific public API
# ---------------------------------------------------------------------------

# Patch DuckDBPyRelation with AI convenience methods (.embed_text(), etc.)
import vane.ai._relation_patch  # noqa: E402,F401
from vane._env import EnvRegistry, env  # noqa: E402,F401
from vane._expression_udf import attach_function, cls, detach_function, func  # noqa: E402,F401
from vane._expressions import col, lit, sql_expr  # noqa: E402,F401
from vane._typing import Connection, Expression, Relation, Statement  # noqa: E402,F401
from vane.config import VaneConfig, configure, current_config  # noqa: E402,F401

# ---------------------------------------------------------------------------
# Submodule lazy-loading
# ---------------------------------------------------------------------------

_SUBMODULES = frozenset(
    {
        "datasource",
        "execution",
        "experimental",
        "query_graph",
        "runners",
        "sqltypes",
        "value",
    }
)

_SUBMODULES = _SUBMODULES | {"ai"}


def __getattr__(name: str) -> object:
    if name in _SUBMODULES:
        mod = _importlib.import_module(f"vane.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'vane' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# __all__ — embedded DuckDB API plus Vane-specific additions
# ---------------------------------------------------------------------------

__all__ = list(
    dict.fromkeys(
        [
            *_duckdb_api.__all__,
            "env",
            "EnvRegistry",
            "Connection",
            "Relation",
            "Expression",
            "Statement",
            "VaneConfig",
            "configure",
            "current_config",
            "col",
            "lit",
            "sql_expr",
            "attach_function",
            "detach_function",
            "func",
            "cls",
            "__duckdb_version__",
            "__vane_version__",
            "__version__",
            "version",
        ]
    )
)
