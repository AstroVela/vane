"""Vane — Distributed DuckDB powered by Ray.

Vane is a thin wrapper around DuckDB that adds distributed execution
capabilities via Ray.  All DuckDB symbols are re-exported, so you can use
``import vane`` as a drop-in replacement for ``import duckdb``.

Quick start::

    import vane

    # Programmatic configuration (optional — env vars also work)
    vane.configure(runner="ray", ray_scan_task_size_grouping=False)

    conn: vane.Connection = vane.connect()
    rel: vane.Relation = conn.sql("SELECT 42")
    rel.show()

Submodules like ``vane.runners``, ``vane.experimental``, and ``vane.sqltypes``
are automatically delegated to the underlying ``duckdb`` package.
"""

import importlib as _importlib

import duckdb
from duckdb import *  # noqa: F403
from duckdb._ray_progress_env import configure_ray_progress_logging_defaults as _configure_ray_progress_logging_defaults
from duckdb._vane_version import get_vane_version

_configure_ray_progress_logging_defaults()

# Extend __path__ so that ``import vane.runners`` (and all other submodules)
# transparently resolves to the corresponding ``duckdb.*`` subpackage.
__path__.extend(duckdb.__path__)

# ---------------------------------------------------------------------------
# Version info
# ---------------------------------------------------------------------------

__duckdb_version__ = duckdb.__duckdb_version__
__vane_version__ = get_vane_version()
__version__ = __vane_version__

# ---------------------------------------------------------------------------
# Vane-specific public API
# ---------------------------------------------------------------------------

# Patch DuckDBPyRelation with AI convenience methods (.embed_text(), etc.)
import vane.ai._relation_patch  # noqa: E402,F401
from vane._env import EnvRegistry, env  # noqa: E402
from vane._expression_udf import attach_function, cls, detach_function, func  # noqa: E402
from vane._expressions import col, lit, sql_expr  # noqa: E402
from vane._typing import Connection, Expression, Relation, Statement  # noqa: E402
from vane.config import VaneConfig, configure, current_config  # noqa: E402

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

# Submodules that live under vane/ directly (not delegated to duckdb).
_VANE_SUBMODULES = frozenset(
    {
        "ai",
    }
)


def __getattr__(name: str):
    if name in _VANE_SUBMODULES:
        mod = _importlib.import_module(f"vane.{name}")
        globals()[name] = mod
        return mod
    if name in _SUBMODULES:
        mod = _importlib.import_module(f"duckdb.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'vane' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# __all__ — curated public API
# ---------------------------------------------------------------------------

__all__ = [
    # --- vane-specific additions ---
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
    # --- version ---
    "__duckdb_version__",
    "__vane_version__",
    "__version__",
    # --- everything from duckdb (via star import) ---
    "connect",
    "sql",
    "execute",
    "executemany",
    "default_connection",
    "set_default_connection",
    "DuckDBPyConnection",
    "DuckDBPyRelation",
]
