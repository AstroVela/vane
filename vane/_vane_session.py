# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared Vane runtime session directory helpers."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

_SESSION_LOCK = threading.Lock()


def ensure_vane_session_dir() -> str:
    """Return VANE_SESSION_DIR, defaulting to $PWD/vane/session_<pid>_<time>."""
    existing = os.environ.get("VANE_SESSION_DIR")
    if existing is not None:
        return _normalize_session_dir(existing)

    with _SESSION_LOCK:
        existing = os.environ.get("VANE_SESSION_DIR")
        if existing is not None:
            return _normalize_session_dir(existing)

        session_dir = Path.cwd() / "vane" / f"session_{os.getpid()}_{time.time_ns()}"
        value = str(session_dir.resolve(strict=False))
        os.environ["VANE_SESSION_DIR"] = value
        return value


def _normalize_session_dir(value: str) -> str:
    text = value.strip()
    if not text:
        raise RuntimeError("VANE_SESSION_DIR is set but empty")
    path = Path(os.path.expanduser(text))
    if not path.is_absolute():
        path = Path.cwd() / path
    normalized = str(path.resolve(strict=False))
    os.environ["VANE_SESSION_DIR"] = normalized
    return normalized
