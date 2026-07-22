# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

_DEFAULT_HINT = "Ensure the C++ ray extension is built and importable."


def require_ray_cxx_attr(name: str, *, hint: str | None = None) -> Any:
    """Return a lazily resolved vane.ray_cxx binding or raise a clear error."""
    import _vane_duckdb

    try:
        return getattr(getattr(_vane_duckdb, "ray_cxx"), name)
    except AttributeError as ex:
        raise ImportError(
            f"Required C++ binding `vane.ray_cxx.{name}` is not available. {hint or _DEFAULT_HINT}"
        ) from ex
