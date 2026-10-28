# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Read only the owned daemon's CPU residency, frequency and scheduler accounting."""

import time
from pathlib import Path


def sample(pid, stop, rows):
    tasks = Path(f"/proc/{pid}/task")
    while not stop.is_set():
        start = time.monotonic_ns()
        threads = []
        frequencies = {}
        for task in tasks.glob("[0-9]*"):
            try:
                fields = (task / "stat").read_text().rsplit(") ", 1)[1].split()
                cpu = int(fields[36])
                runtime, queued, slices = map(int, (task / "schedstat").read_text().split())
                if cpu not in frequencies:
                    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
                    frequencies[cpu] = int(path.read_text()) if path.exists() else None
                threads.append(dict(tid=int(task.name), cpu=cpu, runtime_ns=runtime, runqueue_ns=queued, slices=slices))
            except (FileNotFoundError, ProcessLookupError):
                continue
        rows.append(
            dict(
                epoch_ns=time.time_ns(),
                begin_monotonic_ns=start,
                end_monotonic_ns=time.monotonic_ns(),
                threads=threads,
                frequency_khz=frequencies,
            )
        )
        stop.wait(0.005)


def policy():
    result = {}
    for cpu in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
        directory = cpu / "cpufreq"
        if directory.exists():
            result[cpu.name] = {
                key: (directory / key).read_text().strip()
                for key in ("scaling_driver", "scaling_governor", "scaling_min_freq", "scaling_max_freq")
            }
    return result
