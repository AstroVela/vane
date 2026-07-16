from __future__ import annotations

from typing import Any

_DEFAULT_HINT = "Ensure the C++ ray extension is built and importable."


def require_ray_cxx_attr(name: str, *, hint: str | None = None) -> Any:
    """Return a lazily resolved duckdb.ray_cxx binding or raise a clear error."""
    import _duckdb

    try:
        return getattr(_duckdb.ray_cxx, name)
    except AttributeError as ex:
        raise ImportError(
            f"Required C++ binding `duckdb.ray_cxx.{name}` is not available. {hint or _DEFAULT_HINT}"
        ) from ex
