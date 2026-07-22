# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import ModuleType


def reexport(module: ModuleType, target_globals: dict[str, object]) -> None:
    """Re-export a moved FTE module while preserving private compatibility imports."""
    for name, value in vars(module).items():
        if name.startswith("__") and name.endswith("__"):
            continue
        target_globals[name] = value
    target_globals["__all__"] = [name for name in target_globals if not (name.startswith("__") and name.endswith("__"))]
