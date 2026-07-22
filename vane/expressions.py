# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Small shim for expression helper classes used in runners."""

from __future__ import annotations

from collections.abc import Iterable


class Expression:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f"Expression({self.args!r},{self.kwargs!r})"

    def __add__(self, other):
        return Expression("add", self, other)

    def __radd__(self, other):
        return Expression("add", other, self)

    def __sub__(self, other):
        return Expression("sub", self, other)

    def __mul__(self, other):
        return Expression("mul", self, other)

    def __truediv__(self, other):
        return Expression("div", self, other)

    def __call__(self, *args, **kwargs):
        # allow Expression("a")(...) if needed
        return Expression(self, *args, **kwargs)


class ExpressionsProjection:
    def __init__(self, exprs: Iterable[Expression]):
        self._exprs = list(exprs)

    def __iter__(self):
        return iter(self._exprs)

    def to_column_expressions(self) -> list[Expression]:
        # In the real implementation this converts to column expressions understood by C++.
        # For testing/import purposes we return the underlying Expression objects.
        return self._exprs


__all__ = ["Expression", "ExpressionsProjection"]
