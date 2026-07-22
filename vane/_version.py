# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

# ----------------------------------------------------------------------
# Version API
# ----------------------------------------------------------------------
import _vane_duckdb

from vane._vane_version import get_vane_version

__version__: str = get_vane_version()
"""Version of the Vane package."""

__duckdb_version__: str = _vane_duckdb.__version__
"""Version of DuckDB that is bundled."""


def version() -> str:
    """Human-friendly formatted version string."""
    return f"vane {__version__} (duckdb {_vane_duckdb.__version__})"
