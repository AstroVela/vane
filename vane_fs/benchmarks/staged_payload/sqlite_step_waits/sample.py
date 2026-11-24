# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Read kernel waits only for an owned native call still active after 20 ms."""

import importlib.util
import mmap
import os
import struct
import time
from pathlib import Path

PARENT = Path(__file__).resolve().parent.parent / "fuse_sync_waits/sample.py"
spec = importlib.util.spec_from_file_location("wait_reader", PARENT)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
MAGIC = b"VNSQLW1\0"
MINIMUM_NS = 20000000
INTERVAL = 0.005
KINDS = {1: "fsync", 2: "fdatasync", 3: "sync_file_range", 4: "nanosleep"}


def header(mapping):
    magic, version, stride, slots = struct.unpack_from("<8sQQQ", mapping)
    assert magic == MAGIC and (version, stride, slots) == (1, 64, 16)
    assert len(mapping) >= 64 + stride * slots
    return dict(version=version, stride=stride, slots=slots)


def marker(mapping, slot):
    position = 64 + slot * 64
    first = struct.unpack_from("<Q", mapping, position)[0]
    if first % 2:
        return None
    sequence, tid, kind, fd, begin, offset, size, mode = struct.unpack_from("<QQQqQqqq", mapping, position)
    last = struct.unpack_from("<Q", mapping, position)[0]
    if first != sequence or first != last or kind == 0:
        return None
    assert tid > 0 and kind in KINDS and begin > 0
    return dict(
        sequence=sequence, tid=tid, kind=KINDS[kind], fd=fd, begin_ns=begin, offset=offset, bytes=size, mode=mode
    )


def open_marker(pid):
    for descriptor in Path(f"/proc/{pid}/fd").glob("[0-9]*"):
        try:
            if os.readlink(descriptor) == "/memfd:vane-fs-sql-waits (deleted)":
                fd = os.open(descriptor, os.O_RDONLY | os.O_CLOEXEC)
                try:
                    if os.fstat(fd).st_size != 4096:
                        continue
                    mapping = mmap.mmap(fd, 4096, access=mmap.ACCESS_READ)
                finally:
                    os.close(fd)
                try:
                    if mapping[:8] == bytes(8):
                        mapping.close()
                        continue
                    return mapping, dict(fd=int(descriptor.name), **header(mapping))
                except BaseException:
                    mapping.close()
                    raise
        except (FileNotFoundError, ProcessLookupError):
            continue
    return None, None


def sample(pid, stop, rows):
    tasks = Path(f"/proc/{pid}/task")
    mapping = metadata = None
    stack_access, last_observation = dict(readable=None), {}
    fatal = None
    try:
        while not stop.is_set():
            start, cpu_start = time.monotonic_ns(), time.thread_time_ns()
            if mapping is None:
                mapping, metadata = open_marker(pid)
            waits = []
            if mapping is not None:
                for slot in range(16):
                    value = marker(mapping, slot)
                    now = time.monotonic_ns()
                    if value is None or now - value["begin_ns"] < MINIMUM_NS:
                        continue
                    if now - last_observation.get(value["tid"], 0) < MINIMUM_NS:
                        continue
                    wait = reader.observe(tasks / str(value["tid"]), stack_access)
                    after = marker(mapping, slot)
                    waits.append(dict(marker_before=value, marker_after=after, wait=wait))
                    last_observation[value["tid"]] = time.monotonic_ns()
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
                        dict(tid=int(task.name), cpu=cpu, runtime_ns=runtime, runqueue_ns=queued, slices=slices)
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
                    targeted_waits=waits,
                    sampler_cpu_ns=time.thread_time_ns() - cpu_start,
                )
            )
            stop.wait(INTERVAL)
    except BaseException as error:
        fatal = dict(kind=type(error).__name__, message=str(error))
    finally:
        if mapping is not None:
            mapping.close()
        now = time.monotonic_ns()
        rows.append(
            dict(
                epoch_ns=time.time_ns(),
                begin_monotonic_ns=now,
                end_monotonic_ns=now,
                threads=[],
                frequency_khz={},
                targeted_waits=[],
                sampler_cpu_ns=0,
                sampler_finished=True,
                stop_requested=stop.is_set(),
                marker=metadata,
                mapping_read_only=True,
                mapping_closed=True,
                stack_access=stack_access,
                fatal=fatal,
            )
        )
