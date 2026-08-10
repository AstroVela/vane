# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing_extensions import assert_type

from vane import _native


def batch_identity(table: object) -> object:
    return table


schema: dict[str, object] = {"result": object()}
value = _native.ColumnExpression("value")

assert_type(
    _native._VaneUDFMapBatchesExpression(
        batch_identity,
        "typed_actor_udf",
        schema,
        "subprocess_actor",
        ["value"],
        actor_number=1,
    ),
    _native.Expression,
)

assert_type(
    _native._VaneUDFMapBatchesExpression(
        batch_identity,
        "typed_actor_udf_with_expression",
        schema,
        "subprocess_actor",
        ["value"],
        None,
        False,
        0.0,
        1,
        None,
        value,
    ),
    _native.Expression,
)
