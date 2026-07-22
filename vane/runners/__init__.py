# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Runner package - configuration and lifecycle for local FTE and Ray execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import vane as _duckdb_pkg

if TYPE_CHECKING:
    from vane.runners.runner import Runner


# Re-export public runner setup functions from sub-packages.
from vane.runners.local import set_runner_local
from vane.runners.ray import set_runner_ray

__all__ = [
    "get_or_create_runner",
    "get_or_infer_runner_type",
    "set_runner_local",
    "set_runner_ray",
]


def _get_compiled_vane() -> Any:
    """Return the compiled vane module."""
    return _duckdb_pkg.vane_runners_cpp


def get_or_create_runner() -> Runner:
    """Get or create the configured global runner."""
    vane_mod = _get_compiled_vane()
    return vane_mod.get_or_create_runner()


def get_or_infer_runner_type() -> str:
    """Get or infer the configured runner type."""
    vane_mod = _get_compiled_vane()
    return vane_mod.get_or_infer_runner_type()
