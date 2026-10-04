# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]

from vane.execution._common import ensure_table
from vane.execution.udf_row_preserving import fuse_row_preserving_outputs


def execute_scalar_map_layout(
    payload: dict[str, Any],
    table: pa.Table,
    executor: Any,
) -> list[pa.Table]:
    """Execute scalar arguments and emit row-preserving output pieces."""

    table = ensure_table(table)
    arg_count = int(payload.get("scalar_arg_count") or 0)
    if arg_count <= 0:
        raise RuntimeError("map task requires scalar_arg_count > 0")
    if arg_count > table.num_columns:
        raise RuntimeError("scalar_arg_count %d exceeds task input column count %d" % (arg_count, table.num_columns))

    args = table.select(list(range(arg_count)))
    passthrough = table.select(list(range(arg_count, table.num_columns))) if arg_count < table.num_columns else None
    executor.submit(args)
    outputs = [ensure_table(output) for output in executor.drain_outputs()]
    return fuse_row_preserving_outputs(
        payload,
        passthrough,
        outputs,
        expected_rows=table.num_rows,
        mode="map task",
    )


__all__ = ["execute_scalar_map_layout"]
