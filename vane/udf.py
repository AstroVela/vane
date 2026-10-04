# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0

"""Scalar UDF helpers and the experimental batch callable contract."""

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
    "BatchUDF",
    "FunctionNullHandling",
    "PythonUDFType",
    "vectorized",
]


class BatchUDF:
    """Experimental materialized batch UDF for ``map_batches`` Ray actors.

    Construction, ``prepare_batch``, warmup and cleanup run on the actor thread.
    ``__call__`` runs serially on one persistent worker thread and must return
    a dict of materialized columns. Vane constructs Arrow output on the actor
    thread using the declared schema. Ordinary callable classes are unaffected.

    The initial contract supports primitive/list/struct columns and contiguous
    NumPy fixed-shape tensors. It excludes async and generator callables, FILE,
    IMAGE and variable-shape tensors. Thread-local contexts are not inherited
    by the worker. Returned arrays must not be overwritten while output can
    still be consumed downstream; returning an array transfers its ownership.
    """

    def prepare_batch(self, table: typing.Any) -> typing.Any:
        """Adapt an Arrow table on the actor thread before computation."""
        return table

    def __call__(self, prepared: typing.Any) -> dict[str, typing.Any]:
        """Compute one batch on the persistent worker; return column values."""
        raise NotImplementedError


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
