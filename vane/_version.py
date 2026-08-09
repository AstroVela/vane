# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

# ----------------------------------------------------------------------
# Version API
# ----------------------------------------------------------------------
from vane import _native
from vane._vane_version import get_vane_version

__version__: str = get_vane_version()
"""Version of the Vane package."""

__engine_version__: str = _native.__version__
"""Version of the native Vane engine."""


def version() -> str:
    """Human-friendly formatted version string."""
    return f"Vane {__version__} (engine {__engine_version__})"
