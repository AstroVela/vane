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
    "no_retries",
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


_Function = typing.TypeVar("_Function", bound=typing.Callable[..., typing.Any])


def no_retries(function: _Function) -> _Function:
    """Disable Ray UDF and enclosing native-fragment retries for this callable.

    Apply to a map_batches function or callable class before binding it. Actor
    reuse remains enabled; an execution failure is surfaced instead of replayed.
    """
    if not callable(function):
        raise TypeError("no_retries requires a function or callable class")
    setattr(function, "_vane_datasink_no_task_retries", True)
    return function
