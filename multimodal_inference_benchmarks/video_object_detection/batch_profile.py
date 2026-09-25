# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in video UDF timings; importing this module needs only the standard library."""

from __future__ import annotations

import json
import math
import os
import socket
import time
import uuid
from pathlib import Path


class BatchProfile:
    """Write one JSON record per completed batch in a sequential inference actor.

    This observes the existing computation, without moving tensors or adding
    CUDA synchronization. Ultralytics Results.speed supplies nested model times.
    """

    @classmethod
    def from_env(cls, engine: str):
        directory = os.environ.get("VIDEO_PROFILE_DIR", "").strip()
        return cls(directory, engine) if directory else None

    def __init__(self, directory: str, engine: str):
        self.actor_id = uuid.uuid4().hex
        self.engine = engine
        self.pid = os.getpid()
        self.hostname = socket.gethostname()
        self.path = Path(directory).expanduser().resolve() / f"{engine}-{self.actor_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_index = 0
        self.last_end_ns = None
        self.previous_write_ms = None

    def begin(self, rows: int):
        self.start_ns = self.previous_ns = time.perf_counter_ns()
        self.start_cpu_ns = self.previous_cpu_ns = time.thread_time_ns()
        self.batch_index += 1
        self.record = {
            "schema_version": 1,
            "engine": self.engine,
            "actor_id": self.actor_id,
            "hostname": self.hostname,
            "pid": self.pid,
            "batch_index": self.batch_index,
            "rows": rows,
            "wall_start_ns": time.time_ns(),
            "actor_gap_ms": None if self.last_end_ns is None else (self.start_ns - self.last_end_ns) / 1e6,
            "previous_write_ms": self.previous_write_ms,
            "phases": {},
        }

    def mark(self, name: str):
        now, cpu_now = time.perf_counter_ns(), time.thread_time_ns()
        self.record["phases"][name] = {
            "wall_ms": (now - self.previous_ns) / 1e6,
            "thread_cpu_ms": (cpu_now - self.previous_cpu_ns) / 1e6,
        }
        self.previous_ns, self.previous_cpu_ns = now, cpu_now

    def finish(self, results):
        self.record["actor_body_ms"] = (self.previous_ns - self.start_ns) / 1e6
        self.record["actor_thread_cpu_ms"] = (self.previous_cpu_ns - self.start_cpu_ns) / 1e6
        self.record["model_ms"] = model_timings(results, self.record["rows"])
        # Opening/closing each batch makes completed records available even when
        # a bounded diagnostic run terminates the actor without teardown hooks.
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(self.record, allow_nan=False) + "\n")
        self.last_end_ns = time.perf_counter_ns()
        self.previous_write_ms = (self.last_end_ns - self.previous_ns) / 1e6


def model_timings(results, rows: int):
    """Recover batch times from per-image Results.speed, or report unavailable.

    Ultralytics 8.3.200 divides each synchronized batch interval by the number
    of results. Summing recovers that interval; multiplying each by rows would
    overcount. Missing/invalid metrics are never reported as zero.
    """
    if not rows or len(results) != rows:
        return None
    totals = dict.fromkeys(("preprocess", "inference", "postprocess"), 0.0)
    for result in results:
        speed = getattr(result, "speed", None)
        if not isinstance(speed, dict):
            return None
        for name in totals:
            value = speed.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                return None
            totals[name] += value
    return totals
