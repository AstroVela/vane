# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Vane package version helpers."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

VANE_DISTRIBUTION_NAME = "vane-ai"
VANE_VERSION_FALLBACK = "0.1.0a1"


def get_vane_version() -> str:
    """Return the installed Vane distribution version."""
    try:
        return version(VANE_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return VANE_VERSION_FALLBACK
