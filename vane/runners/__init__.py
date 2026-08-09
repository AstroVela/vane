# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Runner package - configuration and lifecycle for local FTE and Ray execution."""

from __future__ import annotations

from vane import _native
from vane.runners.copy_outcome import CopyOutcomeUnknownError
from vane.runners.local import set_runner_local
from vane.runners.ray import set_runner_ray
from vane.runners.runner import Runner

# Re-export public runner setup functions from sub-packages.

__all__ = [
    "CopyOutcomeUnknownError",
    "get_or_create_runner",
    "get_or_infer_runner_type",
    "set_runner_local",
    "set_runner_ray",
]


def get_or_create_runner() -> Runner:
    """Get or create the configured global runner."""
    return _native.get_or_create_runner()


def get_or_infer_runner_type() -> str:
    """Get or infer the configured runner type."""
    return _native.get_or_infer_runner_type()
