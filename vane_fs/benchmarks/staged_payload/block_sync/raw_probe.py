"""Separate writeback and durable sync; never substitute one for the other."""

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path


def disk_stats(path):
    before = time.time_ns()
    values = [int(v) for v in path.read_text().split()]
    assert len(values) == 17, "Whole-device counters with flush statistics required"
    return {"begin_epoch_ns": before, "end_epoch_ns": time.time_ns(), "values": values}


def timed(call, stat):
    before = disk_stats(stat)
    start = time.time_ns()
    monotonic = time.perf_counter_ns()
    rc = call()
    elapsed = time.perf_counter_ns() - monotonic
    end = time.time_ns()
    return {
        "begin_epoch_ns": start,
        "end_epoch_ns": end,
        "elapsed_ns": elapsed,
        "return_value": rc,
        "disk_before": before,
        "disk_after": disk_stats(stat),
    }


def sample_waits(pid, stopped, rows):
    while not stopped.is_set():
        for task in Path(f"/proc/{pid}/task").glob("*"):
            row = {"epoch_ns": time.time_ns(), "tid": int(task.name)}
            try:
                row["wchan"] = (task / "wchan").read_text().strip()
                row["syscall"] = (task / "syscall").read_text().strip()
                parts = row["syscall"].split()
                if parts and parts[0] in ["17", "18", "74", "75", "277"]:
                    row["file"] = os.readlink(task / "fd" / str(int(parts[1], 16)))
                if row["wchan"] != "0":
                    rows.append(row)
            except (FileNotFoundError, ProcessLookupError):
                pass
            except PermissionError as error:
                rows.append({**row, "error": str(error)})
        stopped.wait(0.002)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stat", type=Path, default=Path("/sys/block/sda/stat"))
    args = parser.parse_args()
    args.output.mkdir()  # Refuse to overwrite preceding evidence.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.sync_file_range.argtypes = [ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_uint]
    libc.sync_file_range.restype = ctypes.c_int

    def drain(fd):
        rc = libc.sync_file_range(fd, 0, 0, 1 | 2 | 4)
        if rc != 0:
            raise OSError(ctypes.get_errno(), "sync_file_range")
        return rc

    variants = ["ordinary", "split", "clean"]
    rotations = [variants[i:] + variants[:i] for i in range(3)]
    orders = rotations + [list(reversed(v)) for v in rotations]
    report = {
        "status": "RUNNING",
        "started_epoch": time.time(),
        "config": {
            "orders": orders,
            "operations": 128,
            "write_bytes": 32768,
            "initial_bytes": 16 * 1024**2,
            "stat": str(args.stat),
            "split_flags": ["WAIT_BEFORE", "WRITE", "WAIT_AFTER"],
            "final_durable_sync": "fdatasync in every variant, including clean",
            "counter_scope": "whole device; other processes can contribute",
        },
        "samples": [],
        "cleanup": [],
    }
    stopped = threading.Event()
    waits = []
    sampler = threading.Thread(target=sample_waits, args=(os.getpid(), stopped, waits))
    sampler.start()
    try:
        for rep, order in enumerate(orders):
            for variant in order:
                assert shutil.disk_usage(args.output).free > 8 * 1024**3
                path = args.output / f"owned-{rep}-{variant}.data"
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                expected = bytearray(b"a" * report["config"]["initial_bytes"])
                try:
                    assert os.write(fd, expected) == len(expected)
                    os.fsync(fd)
                    directory = os.open(args.output, os.O_DIRECTORY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                    row = {"rep": rep, "variant": variant, "started_epoch": time.time(), "operations": []}
                    report["samples"].append(row)
                    for i in range(128):
                        operation = {}
                        if variant != "clean":
                            data = bytes([i % 251 + 1]) * 32768
                            offset = i * len(data)
                            operation["write"] = timed(lambda: os.pwrite(fd, data, offset), args.stat)
                            assert operation["write"]["return_value"] == len(data)
                            expected[offset : offset + len(data)] = data
                        if variant == "split":
                            operation["writeback"] = timed(lambda: drain(fd), args.stat)
                        operation["barrier"] = timed(lambda: os.fdatasync(fd), args.stat)
                        info = os.fstat(fd)
                        operation.update(size=info.st_size, allocated_bytes=info.st_blocks * 512)
                        row["operations"].append(operation)
                    row["ended_epoch"] = time.time()
                    actual = path.read_bytes()
                    assert actual == expected
                    row["validation"] = {"bytes": len(actual), "sha256": hashlib.sha256(actual).hexdigest()}
                    print(
                        rep,
                        variant,
                        round(sum(v["barrier"]["elapsed_ns"] for v in row["operations"]) / 1e6, 3),
                        flush=True,
                    )
                finally:
                    os.close(fd)
                    report["cleanup"].append({"path": str(path), "bytes": path.stat().st_size})
                    path.unlink()
                    report["cleanup"][-1]["removed"] = not path.exists()
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        stopped.set()
        sampler.join(timeout=10)
        assert not sampler.is_alive()
        report.update(ended_epoch=time.time(), free_bytes_after=shutil.disk_usage(args.output).free)
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output / "waits.jsonl").write_text("".join(json.dumps(v) + "\n" for v in waits))


if __name__ == "__main__":
    main()
