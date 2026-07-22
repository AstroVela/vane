# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pyarrow as pa

from vane.execution._common import ensure_table as _ensure_table


def ref_bundle_slice_rows(slice_desc: Any, metadata: dict[str, Any] | None = None) -> int:
    if slice_desc is None:
        return int((metadata or {}).get("num_rows", 0))
    if isinstance(slice_desc, dict):
        return max(0, int(slice_desc["end"]) - int(slice_desc["start"]))
    start, end = slice_desc
    return max(0, int(end) - int(start))


def apply_ref_bundle_slices(
    blocks: tuple[Any, ...] | list[Any],
    slices: list[Any] | tuple[Any, ...] | None,
    metadata: list[Any] | tuple[Any, ...] | None = None,
    names: list[str] | tuple[str, ...] | None = None,
) -> pa.Table:
    if slices is None:
        slices = [None] * len(blocks)
    if metadata is None:
        metadata = [{} for _ in blocks]
    if len(blocks) != len(slices):
        raise ValueError(f"ref bundle block/slice length mismatch: blocks={len(blocks)} slices={len(slices)}")
    if len(blocks) != len(metadata):
        raise ValueError(f"ref bundle block/metadata length mismatch: blocks={len(blocks)} metadata={len(metadata)}")
    output_names = list(names or [])
    tables: list[pa.Table] = []
    for block, slice_desc, meta in zip(blocks, slices, metadata, strict=False):
        table = _ensure_table(block)
        if isinstance(meta, dict) and meta.get("column_ids") is not None:
            column_ids = [int(column_id) for column_id in meta["column_ids"]]
            for column_id in column_ids:
                if column_id < 0 or column_id >= table.num_columns:
                    raise ValueError(f"invalid ref bundle column id {column_id} for table columns={table.num_columns}")
            table = table.select(column_ids)
        if output_names and len(output_names) != table.num_columns:
            raise ValueError(
                f"ref bundle names length {len(output_names)} does not match table columns={table.num_columns}"
            )
        if slice_desc is not None:
            if isinstance(slice_desc, dict):
                start = int(slice_desc["start"])
                end = int(slice_desc["end"])
            else:
                start, end = slice_desc
                start = int(start)
                end = int(end)
            if start < 0 or end < start or end > table.num_rows:
                raise ValueError(f"invalid ref bundle slice [{start}, {end}) for block rows={table.num_rows}")
            table = table.slice(start, end - start)
        tables.append(table)
    if not tables:
        raise ValueError("empty ref bundle input is not supported")
    result = tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options="default")
    if output_names:
        if len(output_names) != result.num_columns:
            raise ValueError(
                f"ref bundle names length {len(output_names)} does not match result columns={result.num_columns}"
            )
        # Rename after concat. Downstream logical names can be duplicated
        # (e.g. multiple inputs derived from the same STRUCT column), and
        # PyArrow cannot unify multiple per-block schemas with duplicate field
        # names during concat.
        result = result.rename_columns(output_names)
    return result
