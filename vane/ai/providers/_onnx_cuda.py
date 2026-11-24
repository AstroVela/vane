# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Size a CUDA arena from visible device 0 without retaining a query context."""

from __future__ import annotations

import ctypes
import sys
from typing import Any


def _load_driver() -> Any:
    loader = getattr(ctypes, "WinDLL") if sys.platform == "win32" else ctypes.CDLL
    return loader("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")


def cuda_arena_limit() -> int:
    """Leave device headroom for the other sessions and non-arena allocations.

    The three sessions execute serially and return unused CUDA arena regions
    after each Run. This is a per-arena ceiling, not a process/device-wide quota;
    other applications can still consume memory after this initialization probe.
    """
    driver = _load_driver()
    signatures = {
        "cuInit": [ctypes.c_uint],
        "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        "cuDevicePrimaryCtxRetain": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
        "cuCtxPushCurrent_v2": [ctypes.c_void_p],
        "cuMemGetInfo_v2": [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)],
        "cuCtxPopCurrent_v2": [ctypes.POINTER(ctypes.c_void_p)],
        "cuDevicePrimaryCtxRelease_v2": [ctypes.c_int],
    }
    for name, arguments in signatures.items():
        function = getattr(driver, name)
        function.argtypes, function.restype = arguments, ctypes.c_int

    def call(name: str, *args: Any) -> None:
        status = getattr(driver, name)(*args)
        if status != 0:
            raise RuntimeError(f"ONNX CUDA memory query: {name} failed with status {status}")

    call("cuInit", 0)
    device = ctypes.c_int()
    call("cuDeviceGet", ctypes.byref(device), 0)
    context = ctypes.c_void_p()
    call("cuDevicePrimaryCtxRetain", ctypes.byref(context), device)
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    try:
        call("cuCtxPushCurrent_v2", context)
        try:
            call("cuMemGetInfo_v2", ctypes.byref(free), ctypes.byref(total))
        finally:
            previous = ctypes.c_void_p()
            call("cuCtxPopCurrent_v2", ctypes.byref(previous))
    finally:
        call("cuDevicePrimaryCtxRelease_v2", device)
    if not 0 <= free.value <= total.value or not total.value:
        raise RuntimeError("ONNX CUDA memory query returned invalid device capacity")
    reserve = max(total.value // 4, 512 * 1024**2)
    limit = free.value - reserve
    if limit <= 0:
        raise MemoryError(
            f"ONNX CUDA arena has no headroom: free_bytes={free.value}, "
            f"total_bytes={total.value}, reserved_bytes={reserve}"
        )
    return limit
