# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Serial production-format BLOB comparisons; workload settings overridden by measure.py."""

import contextlib
import ctypes
import hashlib
import json
import os
import random
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
DURABILITY = "fsync"
SEQUENTIAL_SIZES = [(256, 8), (64, 1)]
VARIANTS = ["baseline", "cached", "updated"]
MIB = 1024**2
# Rotate three positions, then reverse to balance adjacent order pairs.
FIRST = [VARIANTS[i:] + VARIANTS[:i] for i in range(3)]
ORDERS = FIRST + [list(reversed(order)) for order in FIRST]
# Avoid all-zero compression/sparsity; distinct data at every 4 KiB position.
BLOCK = b"".join(bytes([i % 251 + 1]) * 4096 for i in range(256))
REPORT = {
    "status": "RUNNING",
    "config": {
        "orders": ORDERS,
        "durability": "fsync",
        "application_write_bytes": MIB,
        "read_fixture_application_write_bytes": MIB,
        "logical_block_bytes": 4096,
        "page_bytes": 4096,
        "cache_configuration": "production defaults",
        "strict_wal_trigger_frames": 4096,
        "external_threshold_bytes": 262144,
        "wake_mib": 16,
        "admission_mib": 64,
        "retained_wal_mib": 16,
        "global_drop_caches": False,
        "owned_concurrent_tests_or_builds": False,
        "fresh_fuse_rounds": 6,
        "aged_fuse_rounds": 3,
        "native_rounds": 0,
        "aged_guard_bytes": 33554432,
        "aged_guard_write_bytes": MIB,
        "aged_guard_zero_stride": 8192,
        "aged_fuse_read_fixture_bytes": 67108864,
        "aged_native_read_fixture_bytes": 16777216,
    },
    "native": [],
    "fuse": [],
    "aged_fuse": [],
    "space": [],
    "cleanup": [],
}
RESULT = OUT / "measurements.json"


def save():
    RESULT.write_text(json.dumps(REPORT, indent=2) + "\n")


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(MIB), b""):
            h.update(b)
    return h.hexdigest()


def file_state(p):
    return {"size": p.stat().st_size, "allocated_bytes": p.stat().st_blocks * 512} if p.exists() else None


def inspect(db):
    with contextlib.closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as d:
        quick = d.execute("PRAGMA quick_check").fetchall()
        assert quick == [("ok",)]
        return {
            "payload_file": file_state(Path(str(db) + ".payload")),
            "logical_payload_bytes": d.execute(
                "SELECT coalesce(sum(CASE WHEN length(data)=8 THEN 4096 ELSE length(data) END),0) FROM block_payloads"
            ).fetchone()[0],
            "payload_lengths": d.execute(
                "SELECT length(data),count(*) FROM block_payloads GROUP BY length(data)"
            ).fetchall(),
            "quick_check": quick,
            "page_size": d.execute("PRAGMA page_size").fetchone()[0],
            "freelist_count": d.execute("PRAGMA freelist_count").fetchone()[0],
            "page_count": d.execute("PRAGMA page_count").fetchone()[0],
            "payloads": d.execute("SELECT count(*),sum(length(data)),max(length(data)) FROM block_payloads").fetchone(),
            "block_versions": d.execute("SELECT count(*) FROM block_versions").fetchone()[0],
            "page_usage": d.execute(
                "SELECT name,pagetype,count(*),sum(payload),sum(unused),sum(pgsize) FROM dbstat GROUP BY name,pagetype"
            ).fetchall(),
        }


def cleanup(work, extra=None):
    files = [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in work.rglob("*") if p.is_file()]
    shutil.rmtree(work)
    REPORT["cleanup"].append({"path": str(work), "removed": not work.exists(), "files": files, **(extra or {})})
    save()


def mounted(point):
    return any(line.split()[1] == str(point) for line in Path("/proc/mounts").read_text().splitlines())


def stop(sig, frame):
    raise InterruptedError(sig)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


def sequential(point, db, spec):
    mib, chunk_mib = spec
    data = BLOCK * chunk_mib
    address = ctypes.pythonapi.PyBytes_AsString
    address.argtypes = [ctypes.py_object]
    address.restype = ctypes.c_void_p
    buffer_page_offset = address(data) % os.sysconf("SC_PAGE_SIZE")
    p = point / f"write-{mib}"
    times = []
    wal = []
    expected = hashlib.sha256(BLOCK * mib).hexdigest()
    row = {
        "workload": f"write_{mib}" if chunk_mib == 1 else f"stream_{mib}",
        "application_write_bytes": len(data),
        "buffer_page_offset": buffer_page_offset,
        "started_epoch": time.time(),
    }
    start = time.perf_counter()
    fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        for _ in range(mib // chunk_mib):
            t = time.perf_counter()
            n = os.write(fd, data)
            times.append(time.perf_counter() - t)
            assert n == len(data)
            wal.append(Path(str(db) + "-wal").stat().st_size)
        t = time.perf_counter()
        os.fsync(fd)
        row["fsync_seconds"] = time.perf_counter() - t
    finally:
        t = time.perf_counter()
        os.close(fd)
        row["close_seconds"] = time.perf_counter() - t
    row.update(seconds=time.perf_counter() - start, ended_epoch=time.time(), write_latencies=times, wal_bytes=wal)
    row["space_after_timer"] = allocation(db)
    row["mib_per_second"] = mib / row["seconds"]
    actual = digest(p)
    assert actual == expected
    assert p.stat().st_size == mib * MIB
    row["validation"] = {"sha256": actual, "bytes": mib * MIB}
    p.unlink()
    return row


def reads_and_overwrites(point):
    p = point / "read-fixture"
    data = BLOCK * 64
    with p.open("wb", buffering=0) as f:
        for _ in range(64):
            assert f.write(BLOCK) == len(BLOCK)
        os.fsync(f.fileno())
    expected = hashlib.sha256(data).hexdigest()
    assert digest(p) == expected  # complete warm pass, outside timer
    rows = []
    fd = os.open(p, os.O_RDWR)
    try:
        t = time.perf_counter()
        epoch = time.time()
        count = 0
        while b := os.read(fd, MIB):
            count += len(b)
        seconds = time.perf_counter() - t
        assert count == len(data)
        rows.append(
            {
                "workload": "read_64",
                "seconds": seconds,
                "mib_per_second": 64 / seconds,
                "started_epoch": epoch,
                "ended_epoch": time.time(),
                "bytes": count,
            }
        )
        rng = random.Random(4319)
        offsets = [rng.randrange(len(data) // 4096) * 4096 for _ in range(4096)]
        t = time.perf_counter()
        epoch = time.time()
        answers = [os.pread(fd, 4096, off) for off in offsets]
        seconds = time.perf_counter() - t
        assert all(b == data[off : off + 4096] for b, off in zip(answers, offsets))
        rows.append(
            {
                "workload": "random_read_4k",
                "seconds": seconds,
                "mib_per_second": 16 / seconds,
                "started_epoch": epoch,
                "ended_epoch": time.time(),
                "requests": len(offsets),
                "seed": 4319,
                "validation": "all requested slices compared",
            }
        )
        rng = random.Random(892)
        offsets = [v * 4096 for v in rng.sample(range(len(data) // 4096), 512)]
        replacement = b"z" * 4096
        expected_data = bytearray(data)
        for off in offsets:
            expected_data[off : off + 4096] = replacement
        epoch = time.time()
        t = time.perf_counter()
        latencies = []
        for off in offsets:
            t2 = time.perf_counter()
            assert os.pwrite(fd, replacement, off) == 4096
            latencies.append(time.perf_counter() - t2)
        requests_seconds = time.perf_counter() - t
        t2 = time.perf_counter()
        os.fsync(fd)
        sync = time.perf_counter() - t2
        seconds = time.perf_counter() - t
        row = {
            "workload": "random_write_4k",
            "seconds": seconds,
            "mib_per_second": 2 / seconds,
            "fsync_seconds": sync,
            "requests_seconds": requests_seconds,
            "started_epoch": epoch,
            "ended_epoch": time.time(),
            "requests": len(offsets),
            "seed": 892,
            "write_latencies": latencies,
        }
        rows.append(row)
    finally:
        os.close(fd)
    actual = digest(p)
    assert actual == hashlib.sha256(expected_data).hexdigest()
    row["validation"] = {"bytes": len(expected_data), "sha256": actual}
    p.unlink()
    return rows


def allocation(db):
    return {
        name: file_state(Path(str(db) + suffix))
        for name, suffix in [("database", ""), ("wal", "-wal"), ("shm", "-shm")]
    }


def fuse():
    for rep, order in enumerate(ORDERS):
        for variant in order:
            free = shutil.disk_usage(OUT).free
            assert free > 8 * 1024**3
            work = Path(tempfile.mkdtemp(prefix="owned-fuse-", dir=OUT))
            point = work / "mount"
            point.mkdir()
            db = work / "db.sqlite"
            proc = None
            row = {"variant": variant, "rep": rep, "free_bytes": free, "samples": [], "space": {}}
            REPORT["fuse"].append(row)
            logpath = OUT / f"fuse-{variant}-{rep}.log"
            with logpath.open("w") as log:
                start = time.perf_counter()
                row["started_epoch"] = time.time()
                try:
                    t = time.perf_counter()
                    subprocess.run(
                        [str(OUT / variant / "build/vane-fs"), str(db), "init"],
                        check=True,
                        stdout=log,
                        stderr=log,
                        timeout=20,
                    )
                    row["initialization_seconds"] = time.perf_counter() - t
                    row["space"]["after_init"] = allocation(db)
                    command = [
                        str(OUT / variant / "build/vane-fs-mount"),
                        str(db),
                        "--branch",
                        "main",
                        str(point),
                        "--durability=" + DURABILITY,
                    ]
                    row["command"] = command
                    t = time.perf_counter()
                    proc = subprocess.Popen(command, stdout=log, stderr=log)
                    deadline = time.monotonic() + 20
                    while not mounted(point):
                        assert proc.poll() is None
                        assert time.monotonic() < deadline
                        time.sleep(0.05)
                    row["mount_ready_seconds"] = time.perf_counter() - t
                    t = time.perf_counter()
                    time.sleep(0.25)
                    row["settle_seconds"] = time.perf_counter() - t
                    row["space"]["ready"] = allocation(db)
                    t = time.perf_counter()
                    for size in SEQUENTIAL_SIZES:
                        row["samples"].append(sequential(point, db, size))
                        row["space"]["after_write_" + str(size)] = allocation(db)
                    row["samples"] += reads_and_overwrites(point)
                    row["workload_seconds"] = time.perf_counter() - t
                    print(
                        "fuse",
                        variant,
                        rep,
                        {
                            s["workload"]: round(s.get("mib_per_second", s.get("ops_per_second", 0)), 3)
                            for s in row["samples"]
                        },
                        flush=True,
                    )
                finally:
                    row["space"]["before_shutdown"] = allocation(db)
                    if mounted(point):
                        t = time.perf_counter()
                        subprocess.run(["fusermount3", "-u", str(point)], check=True, capture_output=True, timeout=30)
                        if proc:
                            proc.wait(timeout=60)
                        row["unmount_seconds"] = time.perf_counter() - t
                    if proc and proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=10)
                    row["lifecycle_seconds"] = time.perf_counter() - start
                    row["ended_epoch"] = time.time()
                    row["space"]["after_shutdown"] = allocation(db)
                    assert not mounted(point)
                    assert proc is None or proc.returncode == 0
                    assert not Path(str(db) + "-wal").exists() and not Path(str(db) + "-shm").exists()
                    if db.exists():
                        row["inspection"] = inspect(db)
                    row["log_sha256"] = digest(logpath)
                    cleanup(
                        work,
                        {
                            "pid": proc.pid if proc else None,
                            "process_stopped": proc is None or proc.poll() is not None,
                            "mount_absent": True,
                        },
                    )


def collect(variant, db):
    command = [
        str(OUT / variant / "venv/bin/python"),
        "-I",
        "-c",
        "import sys,time,json; from vane_fs import Workspace; w=Workspace(sys.argv[1]); t=time.perf_counter(); r=w.collect_garbage(); s=time.perf_counter()-t; w.close(); print(json.dumps({'seconds':s,'versions':r.versions,'payloads':r.payloads}))",
        str(db),
    ]
    p = subprocess.run(command, text=True, capture_output=True, timeout=90)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def small_and_metadata(point):
    rows = []
    root = os.open(point, os.O_RDONLY | os.O_DIRECTORY)
    try:
        samples = []
        epoch = time.time()
        start = time.perf_counter()
        for i in range(128):
            t = time.perf_counter()
            fd = os.open(point / f"small-{i}", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                assert os.write(fd, b"s" * 4096) == 4096
                os.fsync(fd)
            finally:
                os.close(fd)
            samples.append(time.perf_counter() - t)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "workload": "small_file_fsync",
                "seconds": elapsed,
                "ops_per_second": 128 / elapsed,
                "requests": 128,
                "started_epoch": epoch,
                "ended_epoch": time.time(),
                "operation_latencies": samples,
            }
        )
        for i in range(128):
            p = point / f"small-{i}"
            assert p.read_bytes() == b"s" * 4096
            p.unlink()
        os.fsync(root)
        for op in ["mkdir", "rename", "rmdir"]:
            samples = []
            epoch = time.time()
            start = time.perf_counter()
            for i in range(128):
                t = time.perf_counter()
                if op == "mkdir":
                    os.mkdir(point / f"meta-{i}")
                elif op == "rename":
                    os.rename(point / f"meta-{i}", point / f"renamed-{i}")
                else:
                    os.rmdir(point / f"renamed-{i}")
                samples.append(time.perf_counter() - t)
            t = time.perf_counter()
            os.fsync(root)
            sync = time.perf_counter() - t
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "workload": op,
                    "seconds": elapsed,
                    "ops_per_second": 128 / elapsed,
                    "requests": 128,
                    "final_directory_fsync_seconds": sync,
                    "started_epoch": epoch,
                    "ended_epoch": time.time(),
                    "operation_latencies": samples,
                }
            )
        assert not [p.name for p in point.iterdir() if p.name.startswith(("small-", "meta-", "renamed-"))]
    finally:
        os.close(root)
    return rows


def aged_fuse():
    # Independent fresh mounts: no free pages from the large-write comparison.
    # A retained guard with alternate 4 KiB payloads removed creates reusable holes.
    for rep, order in enumerate(FIRST):
        for variant in order:
            assert shutil.disk_usage(OUT).free > 8 * 1024**3
            work = Path(tempfile.mkdtemp(prefix="owned-aged-", dir=OUT))
            point = work / "mount"
            point.mkdir()
            db = work / "db.sqlite"
            proc = None
            row = {
                "variant": variant,
                "rep": rep,
                "samples": [],
                "free_bytes": shutil.disk_usage(OUT).free,
                "started_epoch": time.time(),
            }
            REPORT["aged_fuse"].append(row)
            logpath = OUT / f"aged-fuse-{variant}-{rep}.log"
            with logpath.open("w") as log:
                try:
                    command = [
                        str(OUT / variant / "build/vane-fs-mount"),
                        str(db),
                        "--branch",
                        "main",
                        str(point),
                        "--durability=" + DURABILITY,
                    ]
                    row["command"] = command
                    proc = subprocess.Popen(command, stdout=log, stderr=log)
                    deadline = time.monotonic() + 20
                    while not mounted(point):
                        assert proc.poll() is None
                        assert time.monotonic() < deadline
                        time.sleep(0.05)
                    guard = point / "guard"
                    fd = os.open(guard, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                    try:
                        for chunk in range(32):
                            data = b"".join(bytes([i % 251 + 1]) * 4096 for i in range(chunk * 256, (chunk + 1) * 256))
                            assert os.write(fd, data) == MIB
                        os.fsync(fd)
                        for i in range(0, 8192, 2):
                            assert os.pwrite(fd, bytes(4096), i * 4096) == 4096
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                    expected = b"".join(bytes([i % 251 + 1] if i % 2 else [0]) * 4096 for i in range(8192))
                    h = hashlib.sha256(expected).hexdigest()
                    assert digest(guard) == h
                    t = time.perf_counter()
                    row["gc"] = collect(variant, db)
                    row["gc_wall_seconds"] = time.perf_counter() - t
                    row["after_gc_inspection"] = inspect(db)
                    row["guard_validation"] = {"bytes": len(expected), "sha256": h}
                    row["preparation_seconds"] = time.time() - row["started_epoch"]
                    for size in SEQUENTIAL_SIZES:
                        row["samples"].append(sequential(point, db, size))
                    row["samples"] += reads_and_overwrites(point)
                    row["samples"] += small_and_metadata(point)
                    assert digest(guard) == h
                    guard.unlink()
                    print(
                        "aged",
                        variant,
                        rep,
                        {
                            s["workload"]: round(s.get("mib_per_second", s.get("ops_per_second", 0)), 3)
                            for s in row["samples"]
                        },
                        flush=True,
                    )
                finally:
                    if mounted(point):
                        subprocess.run(["fusermount3", "-u", str(point)], check=True, capture_output=True, timeout=30)
                        if proc:
                            proc.wait(timeout=60)
                    if proc and proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=10)
                    assert not mounted(point)
                    row["process_returncode"] = proc.returncode if proc else None
                    assert proc is None or proc.returncode == 0
                    assert not Path(str(db) + "-wal").exists() and not Path(str(db) + "-shm").exists()
                    row["ended_epoch"] = time.time()
                    row["lifecycle_seconds"] = row["ended_epoch"] - row["started_epoch"]
                    if db.exists():
                        row["inspection"] = inspect(db)
                    cleanup(
                        work,
                        {
                            "pid": proc.pid if proc else None,
                            "process_stopped": proc is None or proc.poll() is not None,
                            "mount_absent": True,
                        },
                    )
