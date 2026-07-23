# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

_DEFAULT_HINT = "Ensure the C++ ray extension is built and importable."


def require_ray_cxx_attr(name: str, *, hint: str | None = None) -> Any:
    """Return a lazily resolved duckdb.ray_cxx binding or raise a clear error."""
    import _duckdb  # type: ignore[import-not-found]

    try:
        return getattr(_duckdb.ray_cxx, name)
    except AttributeError as ex:
        raise ImportError(
            f"Required C++ binding `duckdb.ray_cxx.{name}` is not available. {hint or _DEFAULT_HINT}"
        ) from ex


def validate_plan_serialization_for_submission(plan: Any) -> None:
    """Validate a native physical root before Driver resource registration."""
    validator = getattr(plan, "_validate_serializable_for_submission", None)
    if not callable(validator):
        raise TypeError("distributed physical plan serialization validator must be callable")
    try:
        validator()
    except Exception as exc:
        query_id = str(plan.idx())
        raise RuntimeError(f"distributed physical plan serialization preflight failed for query_id={query_id}") from exc
