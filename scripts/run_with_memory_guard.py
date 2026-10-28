#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bound a command's memory and lifetime without changing its concurrency."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil


def windows_job(pid: int):
    # Load the standalone helper without importing the native Vane package:
    # the guard also wraps its first build in a clean CI environment.
    path = Path(__file__).resolve().parents[1] / "vane/execution/_windows_job.py"
    spec = importlib.util.spec_from_file_location("vane_guard_windows_job", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WindowsWorkerJob(pid)


def job_processes(job) -> list[psutil.Process]:
    processes = []
    for pid in job.process_ids():
        try:
            process = psutil.Process(pid)
            # Validate membership after capturing the creation time, so a PID
            # reused since the query cannot introduce an unrelated process.
            if job.contains(pid) and process.is_running():
                processes.append(process)
        except (psutil.NoSuchProcess, ProcessLookupError):
            continue
    return processes


def windows_command_child() -> int:
    ready = Path(sys.argv[2])
    deadline = time.monotonic() + 30
    while not ready.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("memory guard did not assign the command's Windows job")
        time.sleep(0.01)
    return subprocess.call(sys.argv[3:])


def owned_process_tree(root: psutil.Process) -> list[psutil.Process]:
    """Exclude unrelated processes whose ancestors' PIDs have been reused."""
    try:
        candidates = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []
    records = {}
    for process in candidates:
        try:
            if process.is_running():
                records[process.pid] = (process, process.ppid(), process.create_time())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if root.pid not in records:
        return []
    owned = []
    for process, _, _ in records.values():
        current = process.pid
        seen = set()
        while current != root.pid:
            if current in seen or current not in records:
                break
            seen.add(current)
            _, parent_pid, created = records[current]
            parent = records.get(parent_pid)
            if parent is None or parent[2] > created:
                break
            current = parent_pid
        else:
            owned.append(process)
    return owned


def stop_processes(root: psutil.Process, known: set[psutil.Process]) -> None:
    deadline = time.monotonic() + 5
    while True:
        known.update(owned_processes(root))
        alive = []
        # Kill the launcher before its workers can schedule more work. Keep its
        # zombie unreaped until cleanup finishes, reserving the session ID and
        # leaving Popen.wait() responsible for the command's actual exit code.
        for process in [root, *(known - {root})]:
            try:
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    alive.append(process)
                    process.kill()
            except psutil.NoSuchProcess:
                pass
        if not alive:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("memory guard still has live command processes")
        time.sleep(0.01)


def owned_processes(root: psutil.Process) -> list[psutil.Process]:
    processes = set(owned_process_tree(root))
    if os.name == "nt" or not root.is_running():
        return list(processes)
    # POSIX commands own a new session. Its membership survives reparenting
    # and shell job-control process groups, unlike a sampled parent/child tree.
    # The leader is not reaped until cleanup, so this session cannot be reused.
    created = root.create_time()
    for process in psutil.process_iter():
        try:
            if process.create_time() >= created and os.getsid(process.pid) == root.pid and process.is_running():
                processes.add(process)
        except (ProcessLookupError, psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return list(processes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-process-mib", type=float, default=4096)
    parser.add_argument("--min-available-mib", type=float, default=4096)
    parser.add_argument("--timeout", type=float, default=0, help="Seconds; zero disables the deadline")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.max_process_mib <= 0 or args.min_available_mib < 0 or args.timeout < 0:
        parser.error("provide a command, a positive process limit and nonnegative reserve/deadline")

    interrupted = threading.Event()
    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, lambda *_: interrupted.set())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    limit = args.max_process_mib * 1024**2
    reserve = args.min_available_mib * 1024**2
    started = time.monotonic()
    peaks: dict[str, int] = {}
    minimum_available = psutil.virtual_memory().available
    reason = None
    code = 1
    child = None
    root = None
    job = None
    ready_directory = None
    known: set[psutil.Process] = set()
    try:
        if os.name == "nt":
            ready_directory = tempfile.TemporaryDirectory(prefix="vane-memory-guard-")
            ready = Path(ready_directory.name) / "ready"
            # Start the base executable directly so a venv redirector cannot
            # create an unassigned child before job ownership is established.
            child = subprocess.Popen(
                [
                    sys._base_executable,
                    "-I",
                    str(Path(__file__).resolve()),
                    "--windows-command-child",
                    str(ready),
                    *command,
                ],
                env={**os.environ, "__PYVENV_LAUNCHER__": sys.executable},
            )
            job = windows_job(child.pid)
            ready.touch()
        else:
            child = subprocess.Popen(command, start_new_session=True)
        try:
            root = psutil.Process(child.pid)
        except psutil.NoSuchProcess:
            code = child.wait()
            return code
        while True:
            if os.name == "nt":
                if child.poll() is not None:
                    break
            elif root.status() == psutil.STATUS_ZOMBIE:
                # Popen.poll() would reap the POSIX session leader too early.
                break
            processes = job_processes(job) if job is not None else owned_processes(root)
            known.update(processes)
            known = {process for process in known if process.is_running()}
            for process in known:
                try:
                    info = process.memory_info()
                    # Windows exposes private committed bytes; POSIX RSS also
                    # counts resident shared pages and is a conservative bound.
                    memory = info.private if os.name == "nt" else info.rss
                    name = process.name()
                    peaks[name] = max(peaks.get(name, 0), memory)
                    if memory > limit:
                        reason = f"memory: {name} PID {process.pid} used {memory / 1024**2:.0f} MiB"
                        code = 137
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            available = psutil.virtual_memory().available
            minimum_available = min(minimum_available, available)
            if available < reserve:
                reason = f"memory: only {available / 1024**2:.0f} MiB physical memory available"
                code = 137
            if reason is None and args.timeout and time.monotonic() - started >= args.timeout:
                reason, code = "deadline exceeded", 124
            if interrupted.is_set():
                reason, code = "guard interrupted", 130
            if reason is not None:
                print(f"Memory guard stopped command: {reason}", file=sys.stderr, flush=True)
                break
            interrupted.wait(0.25)
    finally:
        if job is not None:
            job.close(timeout=5)
        elif child is not None:
            if root is None:
                try:
                    root = psutil.Process(child.pid)
                except psutil.NoSuchProcess:
                    pass
            if root is not None:
                stop_processes(root, known)
        if child is not None:
            child_code = child.wait()
            if reason is None:
                code = child_code
        if ready_directory is not None:
            ready_directory.cleanup()
        args.report.write_text(
            json.dumps(
                {
                    "exit_code": code,
                    "stopped_reason": reason,
                    "memory_metric": "private" if os.name == "nt" else "rss",
                    "max_process_mib": args.max_process_mib,
                    "min_available_mib": args.min_available_mib,
                    "peak_bytes_by_process": peaks,
                    "minimum_available_physical_bytes": minimum_available,
                    "elapsed_seconds": time.monotonic() - started,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return code


if __name__ == "__main__":
    raise SystemExit(windows_command_child() if sys.argv[1:2] == ["--windows-command-child"] else main())
