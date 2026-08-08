# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from vane._ray_errors import remote_ray_exception

_DEFAULT_HINT = "Ensure the C++ ray extension is built and importable."


def require_ray_cxx_attr(name: str, *, hint: str | None = None) -> Any:
    """Return a lazily resolved vane.ray_cxx binding or raise a clear error."""
    from vane import _native

    try:
        return getattr(_native.ray_cxx, name)
    except AttributeError as ex:
        raise ImportError(
            f"Required C++ binding `vane.ray_cxx.{name}` is not available. {hint or _DEFAULT_HINT}"
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
        message = f"distributed physical plan serialization preflight failed for query_id={query_id}"
        raise remote_ray_exception(message, exc) from exc
