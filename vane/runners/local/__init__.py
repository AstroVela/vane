# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Local FTE runner sub-package."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from vane.runners.local.runner import (
    LocalRunner,
    _normalize_execution_mode,
    _normalize_max_running_tasks,
    _normalize_num_workers,
)

if TYPE_CHECKING:
    from vane.runners.runner import Runner


__all__ = ["LocalRunner", "set_runner_local"]


def set_runner_local(
    num_workers: int | None = 1,
    *,
    max_running_tasks: Any = None,
    execution_mode: str | None = "in_process",
) -> Runner:
    """Configure DuckDB to use the local FTE plan runner."""
    import vane as _duckdb_pkg

    normalized_num_workers = _normalize_num_workers(num_workers)
    normalized_max_running_tasks = _normalize_max_running_tasks(max_running_tasks)
    normalized_execution_mode = _normalize_execution_mode(execution_mode)
    os.environ["VANE_RUNNER"] = "local"
    return _duckdb_pkg.vane_runners_cpp.set_runner_local(
        normalized_num_workers,
        normalized_max_running_tasks,
        normalized_execution_mode,
    )
