# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Vane package version helpers."""

from __future__ import annotations

from importlib.metadata import version

VANE_DISTRIBUTION_NAME = "vane-ai"


def get_vane_version() -> str:
    """Return the installed Vane distribution version."""
    return version(VANE_DISTRIBUTION_NAME)
