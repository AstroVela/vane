# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared split/fuse helpers for row-preserving UDF layouts.

A ``map_batches_rows`` layout table contains ``scalar_arg_count`` UDF argument
columns followed by passthrough columns. Workers feed only the argument columns
to the UDF, then fuse the single output column back onto the passthrough columns.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

_MISSING_ARG_COUNT = "map_batches_rows requires scalar_arg_count > 0"
_BAD_OUTPUT_COLUMN_COUNT = "map_batches_rows output must have exactly one column"


def row_preserving_arg_count(payload: dict[str, Any]) -> int:
    """Return the positive argument-column count carried by a rows payload."""
    try:
        arg_count = int(payload.get("scalar_arg_count"))
    except (TypeError, ValueError):
        raise RuntimeError(_MISSING_ARG_COUNT) from None
    if arg_count <= 0:
        raise RuntimeError(_MISSING_ARG_COUNT)
    return arg_count


def split_row_preserving_input(payload: dict[str, Any], table: pa.Table) -> tuple[pa.Table, pa.Table | None]:
    """Split a row-preserving layout table into UDF args and passthrough data."""
    arg_count = row_preserving_arg_count(payload)
    if arg_count > table.num_columns:
        msg = f"scalar_arg_count {arg_count} exceeds input column count {table.num_columns}"
        raise RuntimeError(msg)
    if table.num_columns == arg_count:
        return table, None
    args = table.select(list(range(arg_count)))
    passthrough = table.select(list(range(arg_count, table.num_columns)))
    return args, passthrough


def row_preserving_output_name(payload: dict[str, Any], output: pa.Table) -> str:
    """Resolve the single output column name from payload schema or Arrow data."""
    output_name = output.column_names[0] if output.column_names else "value"
    output_schema = payload.get("output_schema") or []
    if len(output_schema) == 1 and isinstance(output_schema[0], dict) and output_schema[0].get("name"):
        output_name = str(output_schema[0]["name"])
    return output_name


def fuse_row_preserving_output(
    payload: dict[str, Any],
    passthrough: pa.Table | None,
    output: pa.Table,
) -> pa.Table:
    """Fuse one output column onto passthrough columns for row-preserving UDFs."""
    if output.num_columns != 1:
        raise RuntimeError(_BAD_OUTPUT_COLUMN_COUNT)
    output_name = row_preserving_output_name(payload, output)
    if passthrough is None:
        return pa.table([output.column(0)], names=[output_name])
    if output.num_rows != passthrough.num_rows:
        msg = f"map_batches_rows output rows {output.num_rows} do not match input rows {passthrough.num_rows}"
        raise RuntimeError(msg)
    return pa.table(
        [*list(passthrough.columns), output.column(0)],
        names=[*list(passthrough.schema.names), output_name],
    )
