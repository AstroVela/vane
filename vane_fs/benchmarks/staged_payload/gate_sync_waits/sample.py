# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Observe only slow active native calls; omit the continuous thread/frequency census."""

import importlib.util
import time
from pathlib import Path

PARENT = Path(__file__).resolve().parent.parent / "sqlite_step_waits/sample.py"
spec = importlib.util.spec_from_file_location("active_wait_reader", PARENT)
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
MAGIC, header, marker, open_marker = parent.MAGIC, parent.header, parent.marker, parent.open_marker
INTERVAL = 0.02
MINIMUM_NS = 20000000
OBSERVATION_GAP_NS = 50000000


def sample(pid, stop, rows):
    tasks = Path(f"/proc/{pid}/task")
    mapping = metadata = None
    stack_access, last_observation = dict(readable=None), {}
    fatal = None
    begin, total_cpu = time.monotonic_ns(), time.thread_time_ns()
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
                    if now - last_observation.get(value["tid"], 0) < OBSERVATION_GAP_NS:
                        continue
                    wait = parent.reader.observe(tasks / str(value["tid"]), stack_access)
                    waits.append(dict(marker_before=value, marker_after=marker(mapping, slot), wait=wait))
                    last_observation[value["tid"]] = time.monotonic_ns()
            rows.append(
                dict(
                    epoch_ns=time.time_ns(),
                    begin_monotonic_ns=start,
                    end_monotonic_ns=time.monotonic_ns(),
                    threads=[],
                    frequency_khz={},
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
        end = time.monotonic_ns()
        rows.append(
            dict(
                epoch_ns=time.time_ns(),
                begin_monotonic_ns=end,
                end_monotonic_ns=end,
                threads=[],
                frequency_khz={},
                targeted_waits=[],
                sampler_cpu_ns=0,
                total_sampler_cpu_ns=time.thread_time_ns() - total_cpu,
                observation_wall_ns=end - begin,
                sampler_finished=True,
                stop_requested=stop.is_set(),
                marker=metadata,
                mapping_read_only=True,
                mapping_closed=True,
                stack_access=stack_access,
                fatal=fatal,
            )
        )
