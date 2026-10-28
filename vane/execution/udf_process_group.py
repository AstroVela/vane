# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Cleanup ownership for a subprocess worker and its native engine children."""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from vane.execution._windows_job import WindowsWorkerJob


class WorkerProcessGroup:
    """Retain process identities until the worker's process group is gone.

    Windows job objects retain descendants after their worker exits. POSIX
    workers start a new session before importing user code; group discovery
    still works after the worker exits and its children are reparented. Saved
    psutil identities also retain descendants that create their own sessions.
    POSIX signals use psutil's PID-reuse checks, never an unverified numeric PID.
    """

    def __init__(self, pid: int) -> None:
        self.owner_pid = os.getpid()
        self.pid = pid
        self._lock = threading.RLock()
        self._members: set[psutil.Process] = set()
        try:
            self._root: psutil.Process | None = psutil.Process(pid)
            # Resolve identity while Popen still owns the new worker.
            self._root.create_time()
        except psutil.NoSuchProcess:
            self._root = None
        self._windows_job: WindowsWorkerJob | None = None
        if os.name == "nt":
            from vane.execution._windows_job import WindowsWorkerJob

            # The worker is still blocked on its socket-bootstrap pipe.
            self._windows_job = WindowsWorkerJob(pid)

    def capture(self) -> None:
        if self.owner_pid != os.getpid():
            return
        with self._lock:
            self._capture()

    def _capture(self) -> None:
        if self._windows_job is not None:
            # Job membership survives root exit and cannot race PID reuse.
            return
        if self._root is not None and self._root.is_running():
            try:
                self._members.update(self._root.children(recursive=True))
            except psutil.NoSuchProcess:
                pass
        # The group ID remains reserved while its descendants are alive.
        # If the leader PID has been reused, the old group no longer exists.
        try:
            current_root = psutil.Process(self.pid)
            if current_root != self._root:
                return
        except psutil.NoSuchProcess:
            pass
        for process in psutil.process_iter():
            try:
                if os.getpgid(process.pid) == self.pid:
                    process.create_time()
                    self._members.add(process)
            except (ProcessLookupError, psutil.NoSuchProcess):
                pass

    def close(self, timeout: float) -> None:
        if self.owner_pid != os.getpid():
            return
        with self._lock:
            self._close(timeout)

    def _close(self, timeout: float) -> None:
        if self._windows_job is not None:
            self._windows_job.close(timeout)
            return
        deadline = time.monotonic() + timeout
        while True:
            self._capture()
            alive = []
            for process in self._members:
                try:
                    # Zombies have released memory and GPU handles; only their
                    # parent can reap the remaining process-table entry.
                    if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                        alive.append(process)
                except psutil.NoSuchProcess:
                    pass
            if not alive:
                self._members.clear()
                return
            for process in alive:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("UDF worker process group still has live processes")
            time.sleep(min(0.01, remaining))
