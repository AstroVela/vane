# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Programmatic configuration for Vane.

Instead of setting environment variables manually, users can call
:func:`configure` to set multiple options at once with validation::

    import vane

    vane.configure(
        runner="ray",
        ray_scan_task_size_grouping=False,
    )

Under the hood this writes to ``os.environ`` — env vars remain the
single source of truth consumed by C++ and Python code.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from vane._env import _PUBLIC_RUNNER_ERROR, _PUBLIC_RUNNER_VALUES, EnvRegistry, _Var, env

# Dynamically build the dataclass fields from the registry descriptors
# so we don't have to duplicate the list of variables.
_FIELDS: list[tuple[str, type, Any]] = []
for _attr_name in sorted(dir(EnvRegistry)):
    _desc = getattr(EnvRegistry, _attr_name, None)
    if isinstance(_desc, _Var):
        _FIELDS.append((_attr_name, _desc.type_, _desc.default))
_FIELD_NAMES = frozenset(name for name, _, _ in _FIELDS)


def _make_config_class() -> type:
    """Build ``VaneConfig`` dataclass from registry descriptors."""
    fields = [(name, tp, dataclasses.field(default=default)) for name, tp, default in _FIELDS]
    # Create the class with make_dataclass so fields stay in sync with _env.py
    cls = dataclasses.make_dataclass(
        "VaneConfig",
        fields,
        frozen=False,
        repr=True,
        eq=True,
    )
    cls.__module__ = __name__
    cls.__qualname__ = "VaneConfig"
    cls.__doc__ = (
        "Configuration snapshot for Vane.\n\n"
        "All fields correspond 1-to-1 with ``VANE_*`` environment variables.\n"
        "Use :func:`configure` to create, validate, and apply a config."
    )
    return cls


VaneConfig = _make_config_class()


def configure(**kw: Any) -> VaneConfig:
    """Create a config from keyword arguments and apply it to the environment.

    Unknown keys raise ``AttributeError``.  Returns the applied config for
    inspection.

    Example::

        cfg = vane.configure(runner="ray", ray_scan_task_size_grouping=False)
        print(cfg.runner)  # "ray"
    """
    for key in kw:
        if key not in _FIELD_NAMES:
            raise AttributeError(
                f"Unknown config field: {key!r}. Use current_config().__dict__.keys() to see available names."
            )
    if "runner" in kw:
        runner = str(kw["runner"] or "").strip().lower() or "ray"
        if runner not in _PUBLIC_RUNNER_VALUES:
            raise ValueError(_PUBLIC_RUNNER_ERROR)
        kw["runner"] = runner
    cfg = VaneConfig(**kw)  # type: ignore[call-arg]
    # Apply only the explicitly passed keys (not defaults)
    env.set(**kw)
    return cfg


def current_config() -> VaneConfig:
    """Read the current configuration from the environment.

    Returns a :class:`VaneConfig` snapshot of every registered variable.
    """
    return VaneConfig(**env.as_dict())  # type: ignore[call-arg]
