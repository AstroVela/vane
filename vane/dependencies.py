# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Expose commonly-used optional dependencies (pyarrow, pandas) for vane."""

from __future__ import annotations

try:
    import pyarrow as pa  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pa = None


class _OptionalModule:
    def __init__(self, module):
        self._module = module

    def module_available(self) -> bool:
        return self._module is not None

    def __getattr__(self, name):
        if self._module is None:
            raise AttributeError(f"Optional module not available: {name}")
        return getattr(self._module, name)


try:
    import pandas as _pd  # type: ignore

    pd = _OptionalModule(_pd)
except Exception:  # pragma: no cover - optional dependency
    pd = _OptionalModule(None)

try:
    import numpy as _np  # type: ignore

    np = _OptionalModule(_np)
except Exception:  # pragma: no cover - optional dependency
    np = _OptionalModule(None)

__all__ = ["np", "pa", "pd"]
