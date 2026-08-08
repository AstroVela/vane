# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass

_FALSE_VALUES = {"0", "false", "no", "off"}
_TRUE_VALUES = {"1", "true", "yes", "on"}
FTE_WORKER_RUNTIME = "fte"


def fte_status_wait_timeout_s() -> float:
    raw = os.getenv("VANE_FTE_STATUS_WAIT_TIMEOUT_S", "1.0")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, value)


def fte_split_queue_max_buffered_splits() -> int:
    raw = os.getenv("VANE_FTE_SPLIT_QUEUE_MAX_BUFFERED_SPLITS", "1024")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 1024
    return max(1, value)


@dataclass(frozen=True)
class FteWorkerAdmissionConfig:
    max_running_tasks: int
    mode: str
    memory_budget_bytes: int
    task_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        if int(self.max_running_tasks) <= 0:
            raise ValueError("max_running_tasks must be positive")
        if self.mode not in {"lease", "native", "test"}:
            raise ValueError("FTE admission mode must be lease, native, or test")
        if int(self.memory_budget_bytes) <= 0:
            raise ValueError("memory_budget_bytes must be positive")
        if self.task_memory_bytes is not None and int(self.task_memory_bytes) <= 0:
            raise ValueError("task_memory_bytes must be positive when provided")
