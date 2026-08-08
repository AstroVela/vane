# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


def fte_task_update_max_splits() -> int:
    raw = os.getenv("VANE_FTE_TASK_UPDATE_MAX_SPLITS", "256")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 256
    return max(1, value)


def fte_task_update_max_payload_bytes() -> int:
    raw = os.getenv("VANE_FTE_TASK_UPDATE_MAX_PAYLOAD_BYTES", str(8 * 1024 * 1024))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 8 * 1024 * 1024
    return max(1, value)


def _estimated_payload_bytes(value: Any) -> int:
    if value is None:
        return 1
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, memoryview):
        return len(value.tobytes())
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, bool):
        return 1
    if isinstance(value, (int, float)):
        return 8
    if isinstance(value, Mapping):
        return sum(_estimated_payload_bytes(k) + _estimated_payload_bytes(v) for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return sum(_estimated_payload_bytes(item) for item in value)
    return len(repr(value).encode("utf-8"))
