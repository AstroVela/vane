# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""CUDA enumeration helper, executed in a fresh interpreter, never the driver.

The CUDA driver resolves CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER. NVML
verifies that each resulting UUID identifies a whole physical GPU, not a MIG
instance. No contexts, model frameworks, or Ray imports are needed.
"""

from __future__ import annotations

import ctypes
import json
import sys
import uuid
from typing import Any


def visible_devices() -> list[str]:
    loader = getattr(ctypes, "WinDLL") if sys.platform == "win32" else ctypes.CDLL
    cuda = loader("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")

    def call(library: Any, name: str, *args: Any) -> None:
        code = getattr(library, name)(*args)
        if code != 0:
            raise RuntimeError(f"{name} failed with status {code}")

    status = cuda.cuInit(0)
    if status == 100:  # CUDA_ERROR_NO_DEVICE, including an empty visibility mask.
        return []
    if status != 0:
        raise RuntimeError(f"cuInit failed with status {status}")
    count = ctypes.c_int()
    call(cuda, "cuDeviceGetCount", ctypes.byref(count))
    if not count.value:
        return []
    nvml = loader("nvml.dll" if sys.platform == "win32" else "libnvidia-ml.so.1")
    call(nvml, "nvmlInit_v2")
    try:
        devices = []
        for ordinal in range(count.value):
            device = ctypes.c_int()
            call(cuda, "cuDeviceGet", ctypes.byref(device), ordinal)
            identifier = (ctypes.c_ubyte * 16)()
            call(cuda, "cuDeviceGetUuid_v2", ctypes.byref(identifier), device)
            name = "GPU-" + str(uuid.UUID(bytes=bytes(identifier)))
            handle = ctypes.c_void_p()
            if nvml.nvmlDeviceGetHandleByUUID(name.encode("ascii"), ctypes.byref(handle)) != 0:
                raise ValueError("local query GPU actors require whole physical GPUs; MIG is unsupported")
            devices.append(name)
        return devices
    finally:
        call(nvml, "nvmlShutdown")


if __name__ == "__main__":
    print(json.dumps(visible_devices()))
