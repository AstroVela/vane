# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vane import DuckDBPyConnection, Statement


def execute_distributed_write(
    statement: str | Statement,
    *,
    connection: DuckDBPyConnection,
) -> dict[str, Any]:
    """Execute one MERGE INTO statement using the Ray write lifecycle."""
    from vane.runners import get_or_create_runner, get_or_infer_runner_type

    runner_type = get_or_infer_runner_type()
    if runner_type != "ray":
        raise ValueError(f"distributed statement writes require the Ray runner; configured runner is {runner_type!r}")

    runner = get_or_create_runner()
    if runner.name != "ray":
        raise RuntimeError(f"expected the Ray runner, got {runner.name!r}")
    result = runner.run_statement_write(connection, statement)
    if not isinstance(result, Mapping):
        raise TypeError(f"Runner.run_statement_write() returned {type(result).__name__}, expected a mapping")
    return dict(result)
