# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Arrow input type checks shared by DataSink adapters."""

from __future__ import annotations

import pyarrow as pa  # type: ignore[import-not-found, import-untyped, unused-ignore]


def same_input_type(actual: pa.DataType, bound: pa.DataType) -> bool:
    """Allow worker offset widths to differ without accepting logical type drift."""

    if actual == bound:
        return True
    if (pa.types.is_string(actual) or pa.types.is_large_string(actual)) and (
        pa.types.is_string(bound) or pa.types.is_large_string(bound)
    ):
        return True
    if (pa.types.is_binary(actual) or pa.types.is_large_binary(actual)) and (
        pa.types.is_binary(bound) or pa.types.is_large_binary(bound)
    ):
        return True
    variable_lists = (pa.types.is_list(actual) or pa.types.is_large_list(actual)) and (
        pa.types.is_list(bound) or pa.types.is_large_list(bound)
    )
    fixed_lists = (
        pa.types.is_fixed_size_list(actual)
        and pa.types.is_fixed_size_list(bound)
        and actual.list_size == bound.list_size
    )
    if variable_lists or fixed_lists:
        if actual.value_field.nullable != bound.value_field.nullable:
            return False
        return same_input_type(actual.value_type, bound.value_type)
    if pa.types.is_struct(actual) and pa.types.is_struct(bound):
        return len(actual) == len(bound) and all(
            actual_field.name == bound_field.name
            and actual_field.nullable == bound_field.nullable
            and same_input_type(actual_field.type, bound_field.type)
            for actual_field, bound_field in zip(actual, bound, strict=True)
        )
    return False
