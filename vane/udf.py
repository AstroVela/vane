# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0

"""Scalar UDF modes and helpers for Vane."""

import typing

from vane._native._func import (
    ARROW,
    DEFAULT,
    NATIVE,
    SPECIAL,
    FunctionNullHandling,
    PythonUDFType,
)

__all__ = [
    "ARROW",
    "DEFAULT",
    "NATIVE",
    "SPECIAL",
    "FunctionNullHandling",
    "PythonUDFType",
    "vectorized",
]


def vectorized(func: typing.Callable[..., typing.Any]) -> typing.Callable[..., typing.Any]:
    """Decorate a function with annotated function parameters.

    This allows Vane to infer that the function should be provided with pyarrow arrays and should expect
    pyarrow array(s) as output.
    """
    import types
    from inspect import signature

    new_func = types.FunctionType(func.__code__, func.__globals__, func.__name__, func.__defaults__, func.__closure__)
    # Construct the annotations:
    import pyarrow as pa

    new_annotations = {}
    sig = signature(func)
    for param in sig.parameters:
        new_annotations[param] = pa.lib.ChunkedArray

    new_func.__annotations__ = new_annotations
    return new_func
