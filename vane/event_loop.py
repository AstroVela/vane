# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Minimal event loop helper shim used by runners."""

from __future__ import annotations

import asyncio


def set_event_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    """Set the running event loop for code that expects this helper."""
    if loop is None:
        asyncio.set_event_loop(None)
    else:
        asyncio.set_event_loop(loop)
