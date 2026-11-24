# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Node process resources for local task and resident actor admission."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import psutil

from vane.execution.resources import ResourceVector


class LocalProcessCapacityError(ValueError):
    """Node capacity is temporarily owned; no worker construction was attempted."""


def local_process_capacity() -> ResourceVector:
    cpus = max(1, os.cpu_count() or 1)
    if hasattr(os, "sched_getaffinity"):
        cpus = min(cpus, len(os.sched_getaffinity(0)))
    return ResourceVector(cpu=cpus, heap_bytes=available_process_memory())


def local_task_capacity(resources: ResourceVector, limit: ResourceVector) -> int:
    if resources.cpu <= 0 or resources.gpu or resources.object_store_bytes:
        raise ValueError("local subprocess tasks require positive CPU resources and no GPU/object-store reservation")
    if not resources.fits_within(limit):
        raise ValueError("local task exceeds the node CPU/heap resource capacity")
    slots = math.floor(limit.cpu / resources.cpu + 1e-12)
    if resources.heap_bytes:
        slots = min(slots, limit.heap_bytes // resources.heap_bytes)
    return max(1, slots)


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
