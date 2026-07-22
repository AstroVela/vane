# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Ray (distributed) runner sub-package."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from vane.runners.ray.committed_copy import read_committed_copy_direct_write_parquet
from vane.runners.ray.lifecycle import (
    cleanup_copy_direct_write_lifecycle_once,
    run_copy_direct_write_lifecycle_cleanup_loop,
)
from vane.runners.ray.runner import RayRunner

if TYPE_CHECKING:
    from vane.runners.runner import Runner


__all__ = [
    "RayRunner",
    "cleanup_copy_direct_write_lifecycle_once",
    "read_committed_copy_direct_write_parquet",
    "run_copy_direct_write_lifecycle_cleanup_loop",
    "set_runner_ray",
]


def set_runner_ray(
    address: str | None = None,
    noop_if_initialized: bool = False,
    max_task_backlog: int | None = None,
    force_client_mode: bool = False,
) -> Runner:
    """Configure DuckDB to use the Ray distributed computing framework."""
    import vane as _duckdb_pkg

    os.environ["VANE_RUNNER"] = "ray"
    return _duckdb_pkg.vane_runners_cpp.set_runner_ray(
        address,
        noop_if_initialized,
        max_task_backlog,
        force_client_mode,
    )
