# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import _duckdb
from typing_extensions import assert_type


def batch_identity(table: object) -> object:
    return table


schema: dict[str, object] = {"result": object()}
value = _duckdb.ColumnExpression("value")

assert_type(
    _duckdb._VaneUDFMapBatchesExpression(
        batch_identity,
        "typed_actor_udf",
        schema,
        "subprocess_actor",
        ["value"],
        actor_number=1,
    ),
    _duckdb.Expression,
)

assert_type(
    _duckdb._VaneUDFMapBatchesExpression(
        batch_identity,
        "typed_actor_udf_with_expression",
        schema,
        "subprocess_actor",
        ["value"],
        None,
        False,
        0.0,
        1,
        False,
        None,
        value,
    ),
    _duckdb.Expression,
)
