# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Terminate timed-out Windows pytest processes without running DLL cleanup."""

import faulthandler
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def windows_test_timeout(timeout):
    # A separate interpreter can enforce the deadline even when native code
    # holds the test process's GIL. Leave a second for its traceback dump.
    watchdog = subprocess.Popen(
        [sys.executable, "-I", str(Path(__file__).resolve()), str(os.getpid()), str(timeout + 1)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        if watchdog.stdout.readline() != b"ready\n":
            raise RuntimeError("Windows test timeout watchdog failed to start")
        faulthandler.dump_traceback_later(timeout)
        watchdog.stdin.write(b"1")
        watchdog.stdin.flush()
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
        try:
            watchdog.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            watchdog.kill()
            watchdog.communicate()


def _watch_parent(pid, timeout):
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    # Retain a handle to this process, so PID reuse cannot target another one.
    process = kernel32.OpenProcess(0x0001, False, pid)  # PROCESS_TERMINATE
    if not process:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        sys.stdout.buffer.write(b"ready\n")
        sys.stdout.buffer.flush()
        if sys.stdin.buffer.read(1) != b"1":
            return
        cancelled = threading.Event()

        def cancel_on_parent_exit():
            os.read(sys.stdin.fileno(), 1)
            cancelled.set()

        threading.Thread(target=cancel_on_parent_exit, daemon=True).start()
        if not cancelled.wait(timeout):
            if not kernel32.TerminateProcess(process, 1):
                raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(process)


if __name__ == "__main__":
    _watch_parent(int(sys.argv[1]), float(sys.argv[2]))
