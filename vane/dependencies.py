# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Expose commonly-used optional dependencies (pyarrow, pandas) for vane."""

from __future__ import annotations

from types import ModuleType
from typing import Any

pa: ModuleType | None
try:
    import pyarrow as _pa

    pa = _pa
except Exception:  # pragma: no cover - optional dependency
    pa = None


class _OptionalModule:
    def __init__(self, module: ModuleType | None) -> None:
        self._module = module

    def module_available(self) -> bool:
        return self._module is not None

    def __getattr__(self, name: str) -> Any:
        if self._module is None:
            raise AttributeError(f"Optional module not available: {name}")
        return getattr(self._module, name)


try:
    import pandas as _pd

    pd = _OptionalModule(_pd)
except Exception:  # pragma: no cover - optional dependency
    pd = _OptionalModule(None)

try:
    import numpy as _np

    np = _OptionalModule(_np)
except Exception:  # pragma: no cover - optional dependency
    np = _OptionalModule(None)

__all__ = ["np", "pa", "pd"]
