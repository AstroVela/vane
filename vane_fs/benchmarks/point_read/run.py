# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Serial read comparisons with separate tmpfs/native and ext4/FUSE windows."""

import argparse
import contextlib
import ctypes
import hashlib
import json
import mmap
import os
import random
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path

MIB = 1024**2
BLOCK = b"".join(bytes([i % 251 + 1]) * 4096 for i in range(256))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(MIB), b""):
            h.update(part)
    return h.hexdigest()


def mounted(path):
    return any(line.split()[1] == str(path) for line in Path("/proc/mounts").read_text().splitlines())


def inspect(path):
    with contextlib.closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        check = db.execute("PRAGMA quick_check").fetchall()
        assert check == [("ok",)]
        return dict(
            quick_check=check,
            version=db.execute("SELECT version FROM format").fetchone()[0],
            pages=db.execute("PRAGMA page_count").fetchone()[0],
            free_pages=db.execute("PRAGMA freelist_count").fetchone()[0],
            versions=db.execute("SELECT count(*) FROM block_versions").fetchone()[0],
        )


def host():
    return {name: Path("/proc/" + name).read_text() for name in ("stat", "diskstats", "pressure/io")}


def main():
    def interrupted(number, frame):
        raise InterruptedError(number)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--kind", choices=("native", "fuse"), required=True)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    assert json.loads((out / "tests.json").read_text())["status"] == "PASS"
    plan = json.loads((out / "plan.json").read_text())
    dest = out / args.kind
    dest.mkdir()
    report = dict(status="RUNNING", kind=args.kind, config=plan, samples=[], cleanup=[], started_epoch=time.time())
    medium = Path("/dev/shm") if args.kind == "native" else dest
    assert subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(medium)], text=True).strip() == (
        "tmpfs" if args.kind == "native" else "ext4"
    )

    def save():
        (dest / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    def cleanup(work, pid=None):
        assert not mounted(work / "mount")
        files = [dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p)) for p in work.rglob("*") if p.is_file()]
        shutil.rmtree(work)
        report["cleanup"].append(
            dict(path=str(work), files=files, removed=not work.exists(), mount_absent=True, pid=pid)
        )
        save()

    def write(fd, count):
        # Freeze buffer alignment, including preparation; no allocator-dependent split.
        with mmap.mmap(-1, MIB + 4096) as buffer:
            address = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
            shift = (48 - address % 4096) % 4096
            view = memoryview(buffer)[shift : shift + MIB]
            try:
                view[:] = BLOCK
                assert (address + shift) % 4096 == 48
                for _ in range(count):
                    assert os.write(fd, view) == len(view)
                os.fsync(fd)
            finally:
                view.release()

    def fuse(component, work, row, tag):
        db, point = work / "db.sqlite", work / "mount"
        point.mkdir()
        build = component / "build/python-release"
        subprocess.run([str(build / "vane-fs"), str(db), "init"], check=True, capture_output=True, timeout=30)
        logpath = dest / (tag + ".log")
        process = None
        try:
            with logpath.open("w") as log:
                command = [str(build / "vane-fs-mount"), str(db), "--branch", "main", str(point), "--durability=fsync"]
                process = subprocess.Popen(command, stdout=log, stderr=log)
                row.update(argv=command, pid=process.pid)
                deadline = time.monotonic() + 20
                while not mounted(point):
                    assert process.poll() is None and time.monotonic() < deadline
                    time.sleep(0.05)
                if row["state"] == "aged":
                    fd = os.open(point / "guard", os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                    try:
                        write(fd, 32)
                        for i in range(0, 8192, 2):
                            assert os.pwrite(fd, bytes(4096), i * 4096) == 4096
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                    # This pair has identical GC code/format; the candidate wheel
                    # supplies the native GC entry point for both plain daemons.
                    gc_python = out / (row["variant"].removesuffix("_point") + "_point/venv/bin/python")
                    command = [
                        str(gc_python),
                        "-I",
                        "-c",
                        "import sys;from vane_fs import Workspace;w=Workspace(sys.argv[1]);w.collect_garbage();w.close()",
                        str(db),
                    ]
                    subprocess.run(command, check=True, capture_output=True, timeout=90)
                    expected = b"".join(bytes([0 if i % 2 == 0 else i % 256 % 251 + 1]) * 4096 for i in range(8192))
                    assert digest(point / "guard") == hashlib.sha256(expected).hexdigest()
                    row["guard_sha256"] = hashlib.sha256(expected).hexdigest()
                    row["after_gc"] = inspect(db)
                path = point / "fixture"
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                try:
                    write(fd, 64)
                    assert digest(path) == hashlib.sha256(BLOCK * 64).hexdigest()
                    for i in range(128):
                        assert (
                            os.pread(fd, 512 * 1024, i * 512 * 1024)
                            == (BLOCK * 2)[i * 512 * 1024 % MIB : i * 512 * 1024 % MIB + 512 * 1024]
                        )
                    time.sleep(0.5)
                    row["samples"] = []
                    for workload in ("sequential", "point", "partial", "cross"):
                        size = {"sequential": 512 * 1024, "point": 4096, "partial": 128, "cross": 4096}[workload]
                        count = 128 if workload == "sequential" else 8192
                        rng = random.Random(4319)
                        offsets = [
                            i * size
                            if workload == "sequential"
                            else rng.randrange(16383) * 4096
                            + (51 if workload == "partial" else 13 if workload == "cross" else 0)
                            for i in range(count)
                        ]
                        reference = BLOCK * 2
                        durations = []
                        before = host()
                        epoch = time.time()
                        start = time.perf_counter()
                        for offset in offsets:
                            t = time.perf_counter()
                            answer = os.pread(fd, size, offset)
                            durations.append(time.perf_counter() - t)
                            assert answer == reference[offset % MIB : offset % MIB + size]
                        seconds = time.perf_counter() - start
                        after = host()
                        latency_file = tag + "-" + workload + ".json"
                        (dest / latency_file).write_text(
                            json.dumps(dict(latencies=durations, host_before=before, host_after=after), indent=2) + "\n"
                        )
                        row["samples"].append(
                            dict(
                                workload=workload,
                                requests=count,
                                bytes=count * size,
                                seconds=seconds,
                                started_epoch=epoch,
                                ended_epoch=time.time(),
                                latencies_file=latency_file,
                            )
                        )
                    row["content_verified"] = True
                finally:
                    os.close(fd)
        finally:
            if mounted(point):
                unmount = subprocess.run(["fusermount3", "-u", str(point)], capture_output=True, timeout=20)
                row["unmount_returncode"] = unmount.returncode
            if process is not None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                row["process_returncode"] = process.returncode
            if mounted(point):
                subprocess.run(["fusermount3", "-uz", str(point)], check=True, capture_output=True, timeout=20)
            assert not mounted(point)
            try:
                row.update(inspection=inspect(db), log_sha256=digest(logpath), ended_epoch=time.time())
            finally:
                cleanup(work, process.pid if process else None)
            assert process is None or process.returncode == 0

    try:
        states = plan["states"] if args.kind == "native" else plan["fuse_states"]
        for state in states:
            for rep, order in enumerate(plan[args.kind + "_orders"]):
                for variant in order:
                    assert shutil.disk_usage(out).free > 8 * 1024**3
                    work = Path(tempfile.mkdtemp(prefix="owned-point-", dir=medium))
                    tag = f"{state}-{rep}-{variant}"
                    row = dict(state=state, rep=rep, variant=variant, started_epoch=time.time())
                    report["samples"].append(row)
                    save()
                    if args.kind == "native":
                        try:
                            command = [str(out / variant / "probe"), str(work / "db.sqlite"), state]
                            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
                            (dest / (tag + ".stdout")).write_text(result.stdout)
                            (dest / (tag + ".stderr")).write_text(result.stderr)
                            result.check_returncode()
                            row.update(
                                argv=command,
                                metrics=json.loads(result.stdout),
                                inspection=inspect(work / "db.sqlite"),
                                ended_epoch=time.time(),
                            )
                            assert row["metrics"]["content_verified"]
                        finally:
                            cleanup(work)
                    else:
                        fuse(out / variant, work, row, tag)
                    print(args.kind, state, rep, variant, "PASS", flush=True)
                    save()
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_epoch"] = time.time()
        save()


if __name__ == "__main__":
    main()
