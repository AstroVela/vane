#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bound a command's memory and lifetime without changing its concurrency."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil


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
    processes = known | set(owned_process_tree(root))
    # Stop the launcher before its workers can schedule more work. Process
    # objects verify creation times before kill(), protecting against PID reuse.
    for process in [root, *(processes - {root})]:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(list(processes | {root}), timeout=5)


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
    known: set[psutil.Process] = set()
    try:
        child = subprocess.Popen(command)
        try:
            root = psutil.Process(child.pid)
        except psutil.NoSuchProcess:
            code = child.wait()
            return code
        while child.poll() is None:
            processes = owned_process_tree(root)
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
                stop_processes(root, known)
                break
            interrupted.wait(0.25)
        child_code = child.wait()
        if reason is None:
            code = child_code
    finally:
        if child is not None and child.poll() is None:
            stop_processes(psutil.Process(child.pid), known)
            child.wait()
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
    raise SystemExit(main())
