# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Native local execution configuration."""

from __future__ import annotations

from vane._env import env

__all__ = ["set_runner_local"]


def set_runner_local() -> None:
    """Select native local execution for new connections.

    Existing connections keep their execution mode. Configure connection threads
    through ``vane.connect(config={"threads": ...})`` and shared admission through
    ``connection.configure_local_runtime()``.
    """
    env.runner = "local"
