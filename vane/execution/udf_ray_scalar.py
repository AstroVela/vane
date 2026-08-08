# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pyarrow as pa

from vane.execution._common import ensure_table


def execute_scalar_map_layout(
    payload: dict[str, Any],
    table: pa.Table,
    executor: Any,
) -> pa.Table:
    """Execute scalar arguments and emit the complete row-preserving layout."""

    table = ensure_table(table)
    arg_count = int(payload.get("scalar_arg_count") or 0)
    if arg_count <= 0:
        raise RuntimeError("map task requires scalar_arg_count > 0")
    if arg_count > table.num_columns:
        raise RuntimeError("scalar_arg_count %d exceeds task input column count %d" % (arg_count, table.num_columns))

    args = table.select(list(range(arg_count)))
    passthrough = table.select(list(range(arg_count, table.num_columns)))
    executor.submit(args)
    output = executor.take_ready_result()
    if output is None:
        raise RuntimeError("map task produced no output")
    output = ensure_table(output)
    if output.num_columns != 1:
        raise RuntimeError("map task output must have exactly one column")
    if output.num_rows != passthrough.num_rows:
        raise RuntimeError(
            "map task output rows %d do not match input rows %d" % (output.num_rows, passthrough.num_rows)
        )
    return pa.table(
        list(passthrough.columns) + [output.column(0)],
        names=list(passthrough.schema.names) + ["value"],
    )


__all__ = ["execute_scalar_map_layout"]
