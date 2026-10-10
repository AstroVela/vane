# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Observed task-process memory for local admission, independent of SHM credit."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass(eq=False)
class LocalTaskMemory:
    # Protected by LocalExecutionCapacity._lock, including ready grants and
    # transport waiters. Completed buffered results no longer count as tasks.
    peak_bytes: int = 0
    completed: int = 0
    inflight: int = 0

    @property
    def estimate_bytes(self) -> int:
        return max(64 * 1024**2, (self.peak_bytes * 3 + 1) // 2)


def task_process_peak_bytes(pid: int) -> int:
    """Read high-water RSS before returning a live worker to its idle cache."""
    try:
        if sys.platform == "linux":
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
            return 0  # An exited/zombie process has no address space.
        else:
            info = psutil.Process(pid).memory_info()
            return int(getattr(info, "peak_wset", info.rss))
    except (FileNotFoundError, ProcessLookupError, psutil.NoSuchProcess):
        return 0


def available_process_memory() -> int:
    """Host headroom, also bounded by each containing Linux cgroup v2."""
    available = int(psutil.virtual_memory().available)
    if sys.platform == "linux":
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if not line.startswith("0::"):
                continue
            root = Path("/sys/fs/cgroup")
            directory = root / line[3:].lstrip("/")
            while directory.is_relative_to(root):
                limit_file = directory / "memory.max"
                if limit_file.exists():
                    limit = limit_file.read_text().strip()
                    if limit != "max":
                        used = int((directory / "memory.current").read_text())
                        available = min(available, max(0, int(limit) - used))
                directory = directory.parent
    return available
