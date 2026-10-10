# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Driver-process GPU residency shared by query and registered actor pools."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from vane.execution.udf_local_gpu import _device_ids
from vane.execution.udf_local_resources import LocalProcessCapacityError


def discover_gpu_devices() -> tuple[str, ...]:
    helper = Path(__file__).with_name("local_cuda_inventory.py")
    try:
        result = subprocess.run(
            [sys.executable, "-I", str(helper)], capture_output=True, text=True, timeout=30, check=True
        )
    except (OSError, subprocess.SubprocessError) as error:
        detail = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) else str(error)
        raise RuntimeError(f"local CUDA device discovery failed: {detail}") from error
    devices = json.loads(result.stdout)
    return _device_ids(devices) if devices else ()


class GpuDeviceLease:
    def __init__(self, owner: _DeviceOwners, devices: tuple[str, ...]) -> None:
        self.owner = owner
        self.devices = devices
        self.pid = os.getpid()
        self.released = False

    def release(self) -> None:
        # Inherited finalizers must not enter locks held by vanished threads.
        if os.getpid() != self.pid:
            return
        with self.owner.lock:
            if not self.released:
                self.owner.occupied.difference_update(self.devices)
                self.released = True


class _DeviceOwners:
    def __init__(self, occupied: set[str] | None = None) -> None:
        self.pid = os.getpid()
        self.lock = threading.RLock()
        self.occupied = set() if occupied is None else set(occupied)

    def reserve(self, count: int, devices: tuple[str, ...] | None) -> GpuDeviceLease:
        candidates = discover_gpu_devices() if devices is None else _device_ids(devices)
        if len(candidates) < count or (devices is not None and len(candidates) != count):
            raise ValueError(f"local GPU actor pool needs {count} devices; visible inventory has {len(candidates)}")
        with self.lock:
            free = tuple(device for device in candidates if device not in self.occupied)
            if len(free) < count:
                raise LocalProcessCapacityError(
                    "local GPU devices are occupied by resident actor pools; retry after cleanup"
                )
            lease = GpuDeviceLease(self, free[:count])
            self.occupied.update(lease.devices)
            return lease


_owners = _DeviceOwners()


def reserve_gpu_devices(count: int, devices: tuple[str, ...] | None = None) -> GpuDeviceLease:
    if type(count) is not int or count <= 0:
        raise ValueError("local GPU replica count must be a positive integer")
    global _owners
    if os.getpid() != _owners.pid:
        # Preserve inherited exclusions, but never touch an inherited lock.
        # The fork child cannot reclaim devices still owned by parent workers.
        _owners = _DeviceOwners(_owners.occupied)
    return _owners.reserve(count, devices)
