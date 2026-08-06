# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared synchronous-callable contract for generic Python UDFs."""

from __future__ import annotations

import functools
import inspect
from collections.abc import AsyncIterable
from typing import Any

_ASYNC_CALLABLE_ERROR = (
    "generic UDF callables must be synchronous; async functions, constructors, and __call__ methods are not supported"
)
_ASYNC_RESULT_ERROR = (
    "generic UDF callables must return values synchronously; received an awaitable or async iterable result"
)


def _is_async_function(value: Any) -> bool:
    unwrapped = inspect.unwrap(value)
    return (
        inspect.iscoroutinefunction(value)
        or inspect.isasyncgenfunction(value)
        or _is_generator_coroutine_function(value)
        or inspect.iscoroutinefunction(unwrapped)
        or inspect.isasyncgenfunction(unwrapped)
        or _is_generator_coroutine_function(unwrapped)
    )


def _is_generator_coroutine_function(value: Any) -> bool:
    code = getattr(value, "__code__", None)
    flags = getattr(code, "co_flags", 0)
    return isinstance(flags, int) and bool(flags & inspect.CO_ITERABLE_COROUTINE)


def _unwrap_method_descriptor(value: Any) -> Any:
    if isinstance(value, (classmethod, staticmethod)):
        return value.__func__
    return value


def is_async_udf_callable(value: Any) -> bool:
    """Return whether invoking *value* enters a declared async call protocol."""
    seen: set[int] = set()

    def visit(candidate: Any) -> bool:
        identity = id(candidate)
        if identity in seen:
            return False
        seen.add(identity)

        if isinstance(candidate, (functools.partial, functools.partialmethod)):
            return visit(_unwrap_method_descriptor(candidate.func))

        if _is_async_function(candidate):
            return True

        if inspect.isclass(candidate):
            class_callables = (
                inspect.getattr_static(candidate, "__call__"),
                inspect.getattr_static(type(candidate), "__call__"),
                inspect.getattr_static(candidate, "__new__"),
                inspect.getattr_static(candidate, "__init__"),
            )
            return any(visit(_unwrap_method_descriptor(item)) for item in class_callables)

        if not (inspect.isfunction(candidate) or inspect.ismethod(candidate)) and callable(candidate):
            call = inspect.getattr_static(type(candidate), "__call__")
            return visit(_unwrap_method_descriptor(call))

        return False

    return visit(value)


def validate_synchronous_udf_callable(value: Any) -> None:
    """Reject async generic UDF functions and async callable classes."""
    if is_async_udf_callable(value):
        raise TypeError(_ASYNC_CALLABLE_ERROR)


def ensure_synchronous_udf_result(result: Any) -> Any:
    """Reject results that require an asynchronous execution protocol."""
    is_awaitable = inspect.isawaitable(result)
    if not is_awaitable and not isinstance(result, AsyncIterable):
        return result
    if is_awaitable and (inspect.iscoroutine(result) or inspect.isgenerator(result)):
        result.close()
    raise TypeError(_ASYNC_RESULT_ERROR)


__all__ = [
    "ensure_synchronous_udf_result",
    "is_async_udf_callable",
    "validate_synchronous_udf_callable",
]
