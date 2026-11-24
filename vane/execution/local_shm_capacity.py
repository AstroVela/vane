# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Snapshot platform headroom once to size a local shared-memory arena.

The snapshot is a sizing policy, not an OS reservation. Mapping must still
succeed before execution starts; never retry a query after allocation failure.
"""

from __future__ import annotations

import ctypes
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import psutil

MEMORY_LIMIT_ENV = "VANE_LOCAL_MEMORY_LIMIT_BYTES"
MEMORY_RESERVE_ENV = "VANE_LOCAL_MEMORY_RESERVE_BYTES"


@dataclass(frozen=True)
class ShmCapacity:
    capacity: int
    automatic: bool
    headroom: dict[str, int]


def _configured_bytes(name: str, *, default: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"{name} must be a nonnegative integer byte count")
    return int(raw)


def _windows_headroom() -> dict[str, int]:
    # Fixed-width types keep the ABI testable on non-Windows hosts too.
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
            (name, ctypes.c_uint64)
            for name in (
                "physical_total",
                "physical_available",
                "commit_limit",
                "commit_available",
                "virtual_total",
                "virtual_available",
                "extended_virtual",
            )
        ]

    # Resolve Windows-only ctypes exports at call time on every type-check host.
    kernel = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    query = kernel.GlobalMemoryStatusEx
    query.argtypes, query.restype = [ctypes.POINTER(MemoryStatus)], ctypes.c_int
    result = MemoryStatus(length=ctypes.sizeof(MemoryStatus))
    if not query(ctypes.byref(result)):
        raise getattr(ctypes, "WinError")(getattr(ctypes, "get_last_error")())
    # ullAvailPageFile is the amount the CURRENT PROCESS can commit, at most
    # the system's remaining commit. It is not free physical RAM or swap size.
    return {"commit_available": result.commit_available, "virtual_available": result.virtual_available}


def _unescape_mount(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _cgroup_headroom(proc: Path = Path("/proc/self")) -> dict[str, int]:
    """Read visible containing cgroups, including finite ancestor limits."""
    memberships = []
    for line in (proc / "cgroup").read_text().splitlines():
        hierarchy, controllers, group = line.split(":", 2)
        if hierarchy == "0" and not controllers:
            memberships.append(("cgroup2", group))
        elif "memory" in controllers.split(","):
            memberships.append(("cgroup", group))
    available: dict[str, int] = {}
    resolved = set()
    for line in (proc / "mountinfo").read_text().splitlines():
        fields, filesystem = line.split(" - ", 1)
        left, right = fields.split(), filesystem.split()
        kind = right[0]
        if kind not in {"cgroup", "cgroup2"} or (kind == "cgroup" and "memory" not in right[2].split(",")):
            continue
        mount = Path(_unescape_mount(left[4]))
        mount_root = PurePosixPath(_unescape_mount(left[3]))
        for member_kind, raw_group in memberships:
            if member_kind != kind:
                continue
            group_path = PurePosixPath(raw_group)
            if not group_path.is_absolute() or ".." in group_path.parts:
                raise RuntimeError("invalid cgroup membership path")
            if ".." not in mount_root.parts and group_path.is_relative_to(mount_root):
                relative = group_path.relative_to(mount_root)
            else:
                continue
            directory = mount.joinpath(*relative.parts)
            if not directory.is_dir():
                raise RuntimeError("current cgroup directory is unavailable")
            resolved.add((member_kind, raw_group))
            limit_name, used_name = (
                ("memory.max", "memory.current")
                if kind == "cgroup2"
                else ("memory.limit_in_bytes", "memory.usage_in_bytes")
            )
            while directory.is_relative_to(mount):
                limit_file = directory / limit_name
                if limit_file.exists():
                    value = limit_file.read_text().strip()
                    if value != "max":
                        limit = int(value)
                        used = int((directory / used_name).read_text().strip())
                        remaining = max(0, limit - used)
                        available[kind] = min(available.get(kind, remaining), remaining)
                if directory == mount:
                    break
                directory = directory.parent
    if set(memberships) - resolved:
        raise RuntimeError(
            "cannot resolve current cgroup in visible mounts; a namespace-relative cgroup mount is required"
        )
    return available


def memory_headroom() -> dict[str, int]:
    memory = psutil.virtual_memory()
    process = psutil.Process().memory_info()
    reserve = _configured_bytes(MEMORY_RESERVE_ENV, default=0)
    assert reserve is not None
    values = {"physical_available": max(0, int(memory.available) - reserve)}
    if sys.platform == "win32":
        values.update(_windows_headroom())
    else:
        import resource

        limit, _ = resource.getrlimit(resource.RLIMIT_AS)
        if limit != resource.RLIM_INFINITY:
            values["address_space_available"] = max(0, limit - process.vms)
        if sys.platform == "linux":
            values.update(_cgroup_headroom())
    application_limit = _configured_bytes(MEMORY_LIMIT_ENV)
    if application_limit is not None:
        # Same accounting as the command guard; leave the remainder for the
        # arena and apply the 30% fraction below to retain heap/copy headroom.
        used = int(getattr(process, "private")) if sys.platform == "win32" else process.rss
        values["application_available"] = max(0, application_limit - used)
    return values


def resolve_capacity(requested: int | None = None) -> ShmCapacity:
    headroom = memory_headroom()
    if sys.platform == "linux":
        stat = os.statvfs("/dev/shm")
        headroom["shm_available"] = int(stat.f_frsize) * int(stat.f_bavail)
    if any(value <= 0 for value in headroom.values()):
        raise MemoryError(f"no available memory for local shared-memory arena: {headroom}")
    if requested is None:
        memory = min(value for name, value in headroom.items() if name != "shm_available")
        capacity = min(200 * 1024**3, memory * 3 // 10)
        if "shm_available" in headroom:
            capacity = min(capacity, headroom["shm_available"] * 95 // 100)
    else:
        capacity = requested
    if capacity < 64 or capacity > min(headroom.values()):
        raise MemoryError(f"local shared-memory arena cannot fit: requested_bytes={capacity}, headroom={headroom}")
    return ShmCapacity(capacity, requested is None, headroom)
