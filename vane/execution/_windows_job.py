# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Windows job ownership for a worker and all of its subprocess descendants."""

from __future__ import annotations

import ctypes
import sys
import time
import weakref
from ctypes import wintypes
from typing import Any


def _kernel32() -> Any:
    if sys.platform == "win32":
        return ctypes.WinDLL("kernel32", use_last_error=True)
    raise RuntimeError("Windows job objects require Windows")


def _last_error() -> OSError:
    if sys.platform == "win32":
        return ctypes.WinError(ctypes.get_last_error())
    raise RuntimeError("Windows job objects require Windows")


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("process_time", ctypes.c_int64),
        ("job_time", ctypes.c_int64),
        ("flags", wintypes.DWORD),
        ("min_working_set", ctypes.c_size_t),
        ("max_working_set", ctypes.c_size_t),
        ("active_process_limit", wintypes.DWORD),
        ("affinity", ctypes.c_size_t),
        ("priority", wintypes.DWORD),
        ("scheduling_class", wintypes.DWORD),
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("basic", _BasicLimits),
        ("io_counters", ctypes.c_uint64 * 6),
        ("process_memory_limit", ctypes.c_size_t),
        ("job_memory_limit", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t),
        ("peak_job_memory", ctypes.c_size_t),
    ]


class _Accounting(ctypes.Structure):
    _fields_ = [
        ("times", ctypes.c_int64 * 4),
        ("page_faults", wintypes.DWORD),
        ("total_processes", wintypes.DWORD),
        ("active_processes", wintypes.DWORD),
        ("terminated_processes", wintypes.DWORD),
    ]


class WindowsWorkerJob:
    def __init__(self, pid: int) -> None:
        self._api = _kernel32()
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            "OpenProcess": ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
            "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            "SetInformationJobObject": (
                [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD],
                wintypes.BOOL,
            ),
            "QueryInformationJobObject": (
                [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p],
                wintypes.BOOL,
            ),
            "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self._api, name)
            function.argtypes, function.restype = args, result
        handle = self._api.CreateJobObjectW(None, None)
        if not handle:
            raise _last_error()
        self._handle = handle
        self._finalizer = weakref.finalize(self, self._api.CloseHandle, handle)
        try:
            limits = _ExtendedLimits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not self._api.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise _last_error()
            # PROCESS_SET_QUOTA | PROCESS_TERMINATE; no inheritable handle.
            process = self._api.OpenProcess(0x0100 | 0x0001, False, pid)
            if not process:
                raise _last_error()
            try:
                if not self._api.AssignProcessToJobObject(handle, process):
                    raise _last_error()
            finally:
                self._api.CloseHandle(process)
        except BaseException:
            self._finalizer()
            raise

    def close(self, timeout: float) -> None:
        if not self._finalizer.alive:
            return
        if not self._api.TerminateJobObject(self._handle, 1):
            raise _last_error()
        deadline = time.monotonic() + timeout
        while True:
            accounting = _Accounting()
            if not self._api.QueryInformationJobObject(
                self._handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None
            ):
                raise _last_error()
            if accounting.active_processes == 0:
                self._finalizer()
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("UDF worker job still has live processes")
            time.sleep(min(0.01, remaining))
