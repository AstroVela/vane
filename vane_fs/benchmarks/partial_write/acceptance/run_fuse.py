# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run fixed-buffer FUSE comparisons on tmpfs or ext4, with separate diagnostics."""

import argparse
import contextlib
import ctypes
import hashlib
import importlib.util
import json
import mmap
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--medium", choices=("tmpfs", "ext4"), required=True)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    out = args.artifacts.resolve()
    assert json.loads((out / "tests.json").read_text())["status"] == "PASS"
    assert json.loads((out / "probe-build.json").read_text())["status"] == "PASS"
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("workload", here / "workload.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    kind = "diagnostic" if args.diagnostic else "plain"
    dest = out / f"{args.medium}-{kind}"
    dest.mkdir()
    data_root = Path("/dev/shm") if args.medium == "tmpfs" else dest
    actual = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(data_root)], text=True).strip()
    assert actual == args.medium, actual
    m.OUT = dest
    m.RESULT = dest / "results.json"
    m.VARIANTS = ["baseline", "updated"]
    m.ORDERS = m.FIRST = [["baseline", "updated"], ["updated", "baseline"]] * (1 if args.diagnostic else 3)
    m.DURABILITY = "fsync"
    m.SEQUENTIAL_SIZES = [(256, 8), (64, 1)]
    m.REPORT["config"].update(
        orders=m.ORDERS,
        medium=args.medium,
        instrumented=args.diagnostic,
        external_threshold_bytes=None,
        format_version=2,
        sqlite_primitive="original SDK fsync",
        fresh_fuse_rounds=len(m.ORDERS),
        aged_fuse_rounds=len(m.ORDERS),
        native_rounds=0,
        application_write_bytes=[8388608, 1048576],
        buffer_page_offset=48,
        sequence_mib_and_request_mib=m.SEQUENTIAL_SIZES,
        first_workload="256 MiB write into a fresh or independently aged workspace",
        interpretation="tmpfs isolates software costs, not durable-storage throughput",
    )
    for variant in m.VARIANTS:
        alias = dest / variant
        alias.mkdir()
        (alias / "build").symlink_to(
            out / "diagnostic" / variant / "build" if args.diagnostic else out / variant / "build/python-release",
            target_is_directory=True,
        )
        (alias / "venv").symlink_to(out / variant / "venv", target_is_directory=True)
    m.tempfile = types.SimpleNamespace(
        mkdtemp=lambda prefix, dir: tempfile.mkdtemp(prefix="vane-partial-" + prefix, dir=data_root)
    )
    owned = []
    active_pid = None
    state = "fresh"

    def start(command, **kwargs):
        nonlocal active_pid
        process = subprocess.Popen(command, **kwargs)
        active_pid = process.pid
        owned.append((process, Path(command[4]), Path(command[1]).parent))
        return process

    m.subprocess = types.SimpleNamespace(Popen=start, run=subprocess.run, TimeoutExpired=subprocess.TimeoutExpired)

    def cpu():
        values = Path(f"/proc/{active_pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return (int(values[11]) + int(values[12])) / os.sysconf("SC_CLK_TCK")

    def sequential(point, db, dimensions):
        mib, chunk_mib = dimensions
        buffer = mmap.mmap(-1, chunk_mib * m.MIB + 4096)
        base = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
        shift = (48 - base % 4096) % 4096
        data = memoryview(buffer)[shift : shift + chunk_mib * m.MIB]
        data[:] = m.BLOCK * chunk_mib
        assert (base + shift) % os.sysconf("SC_PAGE_SIZE") == 48
        expected = hashlib.sha256()
        for _ in range(mib):
            expected.update(m.BLOCK)
        path = point / f"write-{mib}"
        latencies, wal = [], []
        row = {
            "workload": "stream_256" if mib == 256 else "write_64",
            "application_write_bytes": len(data),
            "buffer_page_offset": 48,
            "started_epoch": time.time(),
        }
        before_cpu = cpu()
        start_time = time.perf_counter()
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            request_start = time.perf_counter()
            for _ in range(mib // chunk_mib):
                t = time.perf_counter()
                assert os.write(fd, data) == len(data)
                latencies.append(time.perf_counter() - t)
                wal.append(Path(str(db) + "-wal").stat().st_size)
            row["requests_seconds"] = time.perf_counter() - request_start
            t = time.perf_counter()
            os.fsync(fd)
            row["fsync_seconds"] = time.perf_counter() - t
        finally:
            t = time.perf_counter()
            os.close(fd)
            row["close_seconds"] = time.perf_counter() - t
            data.release()
            buffer.close()
        row.update(
            seconds=time.perf_counter() - start_time,
            ended_epoch=time.time(),
            daemon_cpu_seconds=cpu() - before_cpu,
            write_latencies=latencies,
            wal_bytes=wal,
        )
        row["mib_per_second"] = mib / row["seconds"]
        row["space_after_timer"] = m.allocation(db)
        digest = m.digest(path)
        assert digest == expected.hexdigest() and path.stat().st_size == mib * m.MIB
        row["validation"] = {"bytes": mib * m.MIB, "sha256": digest}
        path.unlink()
        return row

    m.sequential = sequential
    original_reads = m.reads_and_overwrites

    def reads(point):
        rows = original_reads(point)
        if state == "fresh":
            rows += m.small_and_metadata(point)
        return rows

    m.reads_and_overwrites = reads

    def inspect(db):
        with contextlib.closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as conn:
            quick = conn.execute("PRAGMA quick_check").fetchall()
            schema = conn.execute("SELECT version,block_size FROM format").fetchall()
            lengths = conn.execute("SELECT length(data),count(*) FROM block_payloads GROUP BY length(data)").fetchall()
            assert quick == [("ok",)] and schema == [(2, 4096)] and all(n == 4096 for n, _ in lengths)
            return {
                "quick_check": quick,
                "format": schema,
                "payload_lengths": lengths,
                "page_count": conn.execute("PRAGMA page_count").fetchone()[0],
                "freelist_count": conn.execute("PRAGMA freelist_count").fetchone()[0],
            }

    m.inspect = inspect
    original_cleanup = m.cleanup

    def cleanup(work, extra=None):
        if args.diagnostic and (work / "db.sqlite.timeline.json").exists():
            target = dest / (work.name + "-timeline.json")
            shutil.copy2(work / "db.sqlite.timeline.json", target)
            row = (m.REPORT["fuse"] if state == "fresh" else m.REPORT["aged_fuse"])[-1]
            row["timeline_file"] = str(target)
        original_cleanup(work, extra)

    m.cleanup = cleanup
    monitor = subprocess.Popen(
        [sys.executable, str(here.parents[1] / "io_interference/monitor.py"), str(dest / "host.jsonl")]
    )
    try:
        m.REPORT["started_epoch"] = time.time()
        m.save()
        time.sleep(20)
        assert monitor.poll() is None
        state = "fresh"
        m.fuse()
        state = "post_gc"
        m.aged_fuse()
        m.REPORT["status"] = "PASS"
    except BaseException as error:
        m.REPORT.update(status="FAIL", error=repr(error))
        raise
    finally:
        for process, point, work in owned:
            if m.mounted(point):
                subprocess.run(["fusermount3", "-u", str(point)], check=True, capture_output=True, timeout=30)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            assert not m.mounted(point)
            if work.exists():
                cleanup(
                    work, {"pid": process.pid, "process_stopped": True, "mount_absent": True, "emergency_cleanup": True}
                )
        monitor.terminate()
        monitor.wait(timeout=10)
        m.REPORT.update(
            ended_epoch=time.time(), monitor_stopped=True, free_bytes_after=shutil.disk_usage(data_root).free
        )
        m.save()


if __name__ == "__main__":
    main()
