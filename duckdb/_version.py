# ----------------------------------------------------------------------
# Version API
# ----------------------------------------------------------------------
import _duckdb

from duckdb._vane_version import get_vane_version

__version__: str = get_vane_version()
"""Version of the Vane package."""

__duckdb_version__: str = _duckdb.__version__
"""Version of DuckDB that is bundled."""


def version() -> str:
    """Human-friendly formatted version string."""
    return f"vane {__version__} (duckdb {_duckdb.__version__})"
