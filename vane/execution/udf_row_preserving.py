# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared split/fuse helpers for row-preserving UDF layouts.

A row-preserving layout table contains ``scalar_arg_count`` UDF argument columns
followed by passthrough columns. Workers feed only the argument columns to the
UDF, then fuse each consecutive output piece back onto its matching passthrough
slice.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]

_MISSING_ARG_COUNT = "map_batches_rows requires scalar_arg_count > 0"


def row_preserving_arg_count(payload: dict[str, Any]) -> int:
    """Return the positive argument-column count carried by a rows payload."""
    raw_arg_count = payload.get("scalar_arg_count")
    if raw_arg_count is None:
        raise RuntimeError(_MISSING_ARG_COUNT)
    try:
        arg_count = int(raw_arg_count)
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
    *,
    mode: str = "map_batches_rows",
) -> pa.Table:
    """Fuse one output column onto passthrough columns for row-preserving UDFs."""
    if output.num_columns != 1:
        raise RuntimeError(f"{mode} output must have exactly one column")
    output_name = row_preserving_output_name(payload, output)
    if passthrough is None:
        return pa.table([output.column(0)], names=[output_name])
    if output.num_rows != passthrough.num_rows:
        msg = f"{mode} output rows {output.num_rows} do not match input rows {passthrough.num_rows}"
        raise RuntimeError(msg)
    return pa.table(
        [*list(passthrough.columns), output.column(0)],
        names=[*list(passthrough.schema.names), output_name],
    )


def fuse_row_preserving_outputs(
    payload: dict[str, Any],
    passthrough: pa.Table | None,
    outputs: list[pa.Table],
    *,
    expected_rows: int,
    mode: str,
) -> list[pa.Table]:
    """Fuse output pieces onto matching consecutive passthrough slices."""
    if not outputs:
        raise RuntimeError(f"{mode} produced no output")
    output_rows = sum(output.num_rows for output in outputs)
    if output_rows != expected_rows:
        raise RuntimeError(f"{mode} output rows {output_rows} do not match input rows {expected_rows}")
    if passthrough is not None and passthrough.num_rows != expected_rows:
        raise RuntimeError(f"{mode} passthrough rows {passthrough.num_rows} do not match input rows {expected_rows}")

    fused: list[pa.Table] = []
    row_offset = 0
    for output in outputs:
        output_passthrough = None if passthrough is None else passthrough.slice(row_offset, output.num_rows)
        fused.append(
            fuse_row_preserving_output(
                payload,
                output_passthrough,
                output,
                mode=mode,
            )
        )
        row_offset += output.num_rows
    return fused
