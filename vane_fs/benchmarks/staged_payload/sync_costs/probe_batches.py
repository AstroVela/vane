# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Separate dirty-byte budgeting from file segmentation in a native ext4 probe."""

import argparse
import functools
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MIB = 1024**2
MODES = ("grow128", "grow16", "sealed16")
ORDERS = [
    ["grow128", "grow16", "sealed16"],
    ["sealed16", "grow16", "grow128"],
    ["grow16", "sealed16", "grow128"],
    ["grow128", "sealed16", "grow16"],
    ["sealed16", "grow128", "grow16"],
    ["grow16", "grow128", "sealed16"],
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@functools.cache
def repeated(byte, size):
    value = hashlib.sha256()
    block = bytes([byte]) * MIB
    while size:
        count = min(MIB, size)
        value.update(block[:count])
        size -= count
    return value.hexdigest()


def inspect(work, mode, state):
    data_hash, files = hashlib.sha256(), []
    data = (
        sorted(work.glob("part-*"), key=lambda p: int(p.name.split("-")[1]))
        if mode == "sealed16"
        else [work / "payload"]
    )
    expected_old = [repeated(0x5A if i % 2 == 0 else 0, 16 * MIB) for i in range(12)]
    payload_bytes = 0
    for path in sorted(work.iterdir()):
        file_hash = hashlib.sha256()
        segments = []
        with path.open("rb") as stream:
            while block := stream.read(16 * MIB):
                file_hash.update(block)
                segments.append(hashlib.sha256(block).hexdigest())
                if path in data and (mode == "sealed16" or state == "fresh" or stream.tell() > 192 * MIB):
                    data_hash.update(block)
                    payload_bytes += len(block)
        stat = path.stat()
        files.append(
            dict(path=str(path), bytes=stat.st_size, allocated_bytes=stat.st_blocks * 512, sha256=file_hash.hexdigest())
        )
        if path.name == "marker":
            assert stat.st_size == 4096 and file_hash.hexdigest() == repeated(0x37, 4096)
        elif path.name.startswith("old-"):
            assert int(path.name.split("-")[1]) % 2 == 0
            assert stat.st_size == 16 * MIB and segments == [repeated(0x5A, 16 * MIB)]
        elif path in data:
            old = expected_old if mode != "sealed16" and state == "after_reclamation" else []
            new_count = 1 if mode == "sealed16" else 20
            assert segments == old + [repeated(0x37, 16 * MIB)] * new_count
            assert stat.st_size == (len(old) + new_count) * 16 * MIB
        else:
            raise AssertionError(path)
    assert len(data) == (20 if mode == "sealed16" else 1)
    if mode == "sealed16" and state == "after_reclamation":
        assert len(list(work.glob("old-*"))) == 6
    assert payload_bytes == 320 * MIB and data_hash.hexdigest() == repeated(0x37, 320 * MIB)
    return dict(bytes=payload_bytes, sha256=data_hash.hexdigest(), retained_guard_valid=True, files=files)


def distribution(values):
    values = sorted(values)
    return dict(
        count=len(values),
        sum=sum(values),
        median=statistics.median(values),
        max=max(values),
        p99=values[min(len(values) - 1, int(len(values) * 0.99))],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve() / "raw-batches"
    out.mkdir()
    here = Path(__file__).resolve().parent
    monitor_source = here.parents[1] / "io_interference/monitor.py"
    binary = out / "probe-batches"
    argv = [
        "c++",
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-Wall",
        "-Wextra",
        "-Werror",
        str(here / "probe_batches.cpp"),
        "-o",
        str(binary),
    ]
    with (out / "build.log").open("w") as log:
        subprocess.run(argv, check=True, stdout=log, stderr=log, timeout=120)
    actual = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(out)], text=True).strip()
    assert actual == "ext4"
    plan = dict(
        status="FROZEN_BEFORE_MEASUREMENT",
        frozen_epoch=time.time(),
        medium=actual,
        scope="Raw-file hypothesis only, no SQLite, FUSE, VaneFS format or adoption decision",
        states=["fresh", "after_reclamation"],
        orders=ORDERS,
        runs=36,
        variants=dict(
            grow128="one growing file, sync every 128 MiB",
            grow16="same growing file, sync every 16 MiB",
            sealed16="independent sealed 16 MiB files",
        ),
        bytes=320 * MIB,
        writes=[[256, 8], [64, 1]],
        buffer_page_offset=48,
        reclamation="192 MiB initial data, reclaim alternate 16 MiB ranges; holes for single file, unlink old segments for sealed files; 96 MiB guard retained",
        measurement="all writes, payload syncs, file create/close, one 4 KiB marker sync, final directory fsync; preparation, hashes and cleanup outside each measured phase",
        all_slow_samples_retained=True,
        global_cache_eviction=False,
        cpu_or_kernel_setting_changes=False,
        owned_build_or_test_overlap=False,
        final_barrier_preserved=True,
        inputs={
            str(p): digest(p) for p in (here / "probe_batches.py", here / "probe_batches.cpp", monitor_source, binary)
        },
        build_argv=argv,
        free_bytes_before=shutil.disk_usage(out).free,
    )
    assert plan["free_bytes_before"] > 4 * 1024**3
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    result = dict(status="RUNNING", samples=[], cleanup=[], groups=[], monitor_stopped=False, started_epoch=time.time())
    monitor = subprocess.Popen([sys.executable, str(monitor_source), str(out / "host.jsonl")])
    try:
        for state in plan["states"]:
            for rep, order in enumerate(ORDERS):
                for mode in order:
                    assert shutil.disk_usage(out).free > 2 * 1024**3
                    work = Path(tempfile.mkdtemp(prefix="owned-vane-batch-", dir=out))
                    try:
                        name = f"{state}-{rep}-{mode}"
                        path = out / (name + ".json")
                        with path.open("w") as log, (out / (name + ".log")).open("w") as error:
                            subprocess.run(
                                [str(binary), str(work), mode, state], check=True, stdout=log, stderr=error, timeout=120
                            )
                        sample = json.loads(path.read_text())
                        assert sample["status"] == "PASS" and sample["bytes"] == 320 * MIB
                        calls = [dict(zip(sample["call_columns"], row, strict=True)) for row in sample["calls"]]
                        syncs = [c for c in calls if c["kind"] == "payload_sync"]
                        assert len([c for c in calls if c["kind"] == "write"]) == 96
                        assert len(syncs) == (3 if mode == "grow128" else 20)
                        assert sum(c["dirty_bytes"] for c in syncs) == 320 * MIB
                        assert max(c["dirty_bytes"] for c in syncs) <= (128 if mode == "grow128" else 16) * MIB
                        assert all(0 <= c["begin_ns"] <= c["end_ns"] <= sample["complete_ns"] for c in calls)
                        sample.update(state=state, variant=mode, rep=rep, validation=inspect(work, mode, state))
                        result["samples"].append(sample)
                        print(state, mode, rep, round(320e9 / sample["complete_ns"], 3), "MiB/s", flush=True)
                    finally:
                        assert str(work) not in Path("/proc/mounts").read_text()
                        shutil.rmtree(work)
                        result["cleanup"].append(dict(path=str(work), removed=not work.exists(), process_stopped=True))
                    (out / "results.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
        for state in plan["states"]:
            for mode in MODES:
                samples = [s for s in result["samples"] if (s["state"], s["variant"]) == (state, mode)]
                syncs = [
                    dict(zip(s["call_columns"], c, strict=True))
                    for s in samples
                    for c in s["calls"]
                    if c[0] == "payload_sync"
                ]
                result["groups"].append(
                    dict(
                        state=state,
                        variant=mode,
                        complete_ns=distribution([s["complete_ns"] for s in samples]),
                        median_mib_per_second=statistics.median(320e9 / s["complete_ns"] for s in samples),
                        aggregate_mib_per_second=320e9 * len(samples) / sum(s["complete_ns"] for s in samples),
                        payload_sync_ns=distribution([c["end_ns"] - c["begin_ns"] for c in syncs]),
                        at_least_1s=sum(c["end_ns"] - c["begin_ns"] >= 1e9 for c in syncs),
                    )
                )
        assert all(digest(Path(p)) == h for p, h in plan["inputs"].items())
        result["status"] = "PASS"
    except BaseException as error:
        result.update(status="FAIL", error=repr(error))
        raise
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        result.update(ended_epoch=time.time(), monitor_stopped=True, free_bytes_after=shutil.disk_usage(out).free)
        (out / "results.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
