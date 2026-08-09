# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa


def test_empty_output_table_from_schema_supports_nested_duckdb_types():
    from vane.execution.udf_output_schema import empty_output_table_from_schema

    feature_type = pa.struct(
        [
            ("label", pa.int64()),
            ("confidence", pa.float32()),
            ("bbox", pa.list_(pa.float32())),
        ]
    )

    table = empty_output_table_from_schema(
        [
            {
                "name": "features",
                "kind": "duckdb_type",
                "type": 'STRUCT("label" BIGINT, confidence FLOAT, bbox FLOAT[])',
            },
            {
                "name": "all_features",
                "kind": "duckdb_type",
                "type": 'STRUCT("label" BIGINT, confidence FLOAT, bbox FLOAT[])[]',
            },
        ]
    )

    assert table.num_rows == 0
    assert table.schema.field("features").type == feature_type
    assert table.schema.field("all_features").type == pa.list_(feature_type)
