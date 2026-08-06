# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared synchronous-callable contract for generic Python UDFs."""

from __future__ import annotations

import inspect
from typing import Any

_ASYNC_CALLABLE_ERROR = (
    "generic UDF callables must be synchronous; async functions and async __call__ methods are not supported"
)
_AWAITABLE_RESULT_ERROR = "generic UDF callables must return values synchronously; received an awaitable result"


def _is_async_function(value: Any) -> bool:
    unwrapped = inspect.unwrap(value)
    return (
        inspect.iscoroutinefunction(value)
        or inspect.isasyncgenfunction(value)
        or inspect.iscoroutinefunction(unwrapped)
        or inspect.isasyncgenfunction(unwrapped)
    )


def is_async_udf_callable(value: Any) -> bool:
    """Return whether calling *value* uses an async function body."""
    if _is_async_function(value):
        return True

    if inspect.isclass(value):
        call = inspect.getattr_static(value, "__call__")
        if isinstance(call, (classmethod, staticmethod)):
            call = call.__func__
        return _is_async_function(call)

    if not (inspect.isfunction(value) or inspect.ismethod(value)) and callable(value):
        call = inspect.getattr_static(type(value), "__call__")
        if isinstance(call, (classmethod, staticmethod)):
            call = call.__func__
        return _is_async_function(call)

    return False


def validate_synchronous_udf_callable(value: Any) -> None:
    """Reject async generic UDF functions and async callable classes."""
    if is_async_udf_callable(value):
        raise TypeError(_ASYNC_CALLABLE_ERROR)


def ensure_synchronous_udf_result(result: Any) -> Any:
    """Reject and close coroutine results returned by nominally sync UDFs."""
    if not inspect.isawaitable(result):
        return result
    if inspect.iscoroutine(result):
        result.close()
    raise TypeError(_AWAITABLE_RESULT_ERROR)


__all__ = [
    "ensure_synchronous_udf_result",
    "is_async_udf_callable",
    "validate_synchronous_udf_callable",
]
