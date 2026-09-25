# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Ray (distributed) runner sub-package."""

from __future__ import annotations

import os

from vane.runners.copy_outcome import CopyOutcomeUnknownError, CopyResultUnavailableError
from vane.runners.ray.committed_copy import (
    force_abort_copy_direct_write_run,
    inspect_copy_direct_write_run,
    read_committed_copy_direct_write_parquet,
)
from vane.runners.ray.lifecycle import cleanup_copy_direct_write_lifecycle_once
from vane.runners.ray.runner import RayRunner
from vane.runners.runner import Runner

__all__ = [
    "CopyOutcomeUnknownError",
    "CopyResultUnavailableError",
    "RayRunner",
    "cleanup_copy_direct_write_lifecycle_once",
    "force_abort_copy_direct_write_run",
    "inspect_copy_direct_write_run",
    "read_committed_copy_direct_write_parquet",
    "set_runner_ray",
]


def set_runner_ray(
    address: str | None = None,
    noop_if_initialized: bool = False,
    max_task_backlog: int | None = None,
) -> Runner:
    """Configure Vane to use the Ray distributed computing framework."""
    from vane import _native

    previous_runner = os.environ.get("VANE_RUNNER")
    os.environ["VANE_RUNNER"] = "ray"
    try:
        return _native.set_runner_ray(
            address,
            noop_if_initialized,
            max_task_backlog,
        )
    except BaseException:
        if previous_runner is None:
            os.environ.pop("VANE_RUNNER", None)
        else:
            os.environ["VANE_RUNNER"] = previous_runner
        raise
