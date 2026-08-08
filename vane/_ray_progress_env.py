# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

_FALSE_VALUES = {"", "0", "false", "no", "off", "none"}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_LOG_PROGRESS_VALUES = {"log", "raylog", "text"}


def _env_truthy(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


def dynamic_ray_progress_enabled() -> bool:
    progress_value = os.getenv("VANE_PROGRESS", "auto").strip().lower()
    if progress_value in _FALSE_VALUES:
        return False
    runner = os.getenv("VANE_RUNNER", "").strip().lower() or "ray"
    if runner != "ray" and progress_value in ("", "auto"):
        return False
    return progress_value not in _LOG_PROGRESS_VALUES


def ray_log_to_driver_default() -> bool:
    ray_override = os.getenv("RAY_LOG_TO_DRIVER")
    if ray_override is not None:
        return _env_truthy(ray_override, default=True)

    return not dynamic_ray_progress_enabled()


def configure_ray_progress_logging_defaults() -> None:
    """Compatibility no-op for imports that previously set global Ray defaults.

    ``RayRunner`` passes Vane's preference directly to ``ray.init`` instead.
    """
