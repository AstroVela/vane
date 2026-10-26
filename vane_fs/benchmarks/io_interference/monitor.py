"""Retain lightweight host counters; do not modify host or cache configuration."""

import json
import os
import signal
import sys
import time
from pathlib import Path

running = True


def stop(signum, frame):
    global running
    running = False


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
with Path(sys.argv[1]).open("x", buffering=1) as output:
    while running:
        stat = Path("/proc/stat").read_text().splitlines()
        memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        sample = {
            "epoch": time.time(),
            "monotonic": time.monotonic(),
            "load": os.getloadavg(),
            "cpu_ticks": [int(value) for value in stat[0].split()[1:]],
            "processes": dict(line.split() for line in stat if line.startswith("procs_")),
            "memory_kib": {
                key: int(memory[key].split()[0]) for key in ["MemAvailable", "SwapFree", "Dirty", "Writeback"]
            },
            "diskstats": {
                fields[2]: [int(value) for value in fields[3:]]
                for fields in (line.split() for line in Path("/proc/diskstats").read_text().splitlines())
                if fields[2] in {"sda", "sda2"}
            },
            "pressure": {kind: Path("/proc/pressure", kind).read_text().strip() for kind in ["cpu", "io", "memory"]},
        }
        output.write(json.dumps(sample) + "\n")
        time.sleep(1)
