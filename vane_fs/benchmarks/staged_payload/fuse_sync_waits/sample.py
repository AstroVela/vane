# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Observe only owned daemon threads and read-only whole-device counters."""

import os
import time
from pathlib import Path

INTERVAL = 0.005
DEVICE = Path("/sys/block/sda/stat")
IO_SYSCALLS = {17: "pread", 18: "pwrite", 74: "fsync", 75: "fdatasync", 277: "sync_file_range"}


def read(path):
    try:
        return dict(value=path.read_text().strip())
    except OSError as error:
        return dict(error=dict(errno=error.errno, kind=type(error).__name__))


def syscall(value):
    if "error" in value:
        return value
    fields = value["value"].split()
    if fields == ["running"]:
        return dict(running=True)
    try:
        number = int(fields[0], 0)
        result = dict(number=number)
        if number >= 0:
            result["fd"] = int(fields[1], 0)
        if number in (17, 18):
            result.update(bytes=int(fields[3], 0), offset=int(fields[4], 0))
        elif number == 277:
            result.update(offset=int(fields[2], 0), bytes=int(fields[3], 0), flags=int(fields[4], 0))
        return result
    except (IndexError, ValueError):
        return dict(error=dict(kind="MalformedSyscall", value=value["value"]))


def file_kind(name):
    name = name.removesuffix(" (deleted)")
    for suffix, kind in ((".payload", "payload"), (".stage-fence", "fence"), ("-wal", "wal"), (".sqlite", "database")):
        if name.endswith(suffix):
            return kind
    return "other"


def disk():
    begin = time.monotonic_ns()
    value = read(DEVICE)
    result = dict(begin_ns=begin, end_ns=time.monotonic_ns())
    if "error" in value:
        result.update(value)
    else:
        try:
            values = list(map(int, value["value"].split()))
            if len(values) != 17 or any(v < 0 for v in values):
                raise ValueError
            result["values"] = values
        except ValueError:
            result["error"] = dict(kind="InvalidDeviceCounters", value=value["value"])
    return result


def observe(task, stack_access):
    begin = time.monotonic_ns()
    before = syscall(read(task / "syscall"))
    wait = read(task / "wchan")
    result = dict(begin_ns=begin, syscall_before=before, wchan=wait)
    if before.get("number") in IO_SYSCALLS:
        try:
            name = os.readlink(task / "fd" / str(before["fd"]))
            result["file"] = dict(name=name, kind=file_kind(name))
        except OSError as error:
            result["file"] = dict(error=dict(errno=error.errno, kind=type(error).__name__))
    if wait.get("value") not in (None, "0") and stack_access["readable"] is not False:
        stack = read(task / "stack")
        result["stack"] = stack
        if "error" in stack:
            stack_access.update(readable=False, denial=stack["error"], tid=int(task.name))
        else:
            stack_access["readable"] = True
    after = syscall(read(task / "syscall"))
    result.update(
        syscall_after=after,
        stable=before == after and "error" not in before and not before.get("running", False),
        end_ns=time.monotonic_ns(),
    )
    return result


def sample(pid, stop, rows):
    tasks = Path(f"/proc/{pid}/task")
    stack_access = dict(readable=None)
    fatal = None
    try:
        while not stop.is_set():
            start, cpu_start = time.monotonic_ns(), time.thread_time_ns()
            counters = disk()
            threads, frequencies = [], {}
            for task in tasks.glob("[0-9]*"):
                try:
                    fields = (task / "stat").read_text().rsplit(") ", 1)[1].split()
                    cpu = int(fields[36])
                    runtime, queued, slices = map(int, (task / "schedstat").read_text().split())
                    if cpu not in frequencies:
                        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
                        frequencies[cpu] = int(path.read_text()) if path.exists() else None
                    threads.append(
                        dict(
                            tid=int(task.name),
                            cpu=cpu,
                            state=fields[0],
                            runtime_ns=runtime,
                            runqueue_ns=queued,
                            slices=slices,
                            wait=observe(task, stack_access),
                        )
                    )
                except (FileNotFoundError, ProcessLookupError):
                    continue
            rows.append(
                dict(
                    epoch_ns=time.time_ns(),
                    begin_monotonic_ns=start,
                    end_monotonic_ns=time.monotonic_ns(),
                    threads=threads,
                    frequency_khz=frequencies,
                    device=counters,
                    stack_access=dict(stack_access),
                    sampler_cpu_ns=time.thread_time_ns() - cpu_start,
                )
            )
            stop.wait(INTERVAL)
    except BaseException as error:
        fatal = dict(kind=type(error).__name__, message=str(error))
    finally:
        now = time.monotonic_ns()
        rows.append(
            dict(
                epoch_ns=time.time_ns(),
                begin_monotonic_ns=now,
                end_monotonic_ns=now,
                threads=[],
                frequency_khz={},
                sampler_finished=True,
                stop_requested=stop.is_set(),
                fatal=fatal,
                stack_access=dict(stack_access),
            )
        )
