"""Compare equal-byte/equal-barrier owned-file workloads; never tune the host."""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
if len(sys.argv) != 3:
    raise SystemExit("usage: run.py OUTPUT_DIRECTORY direct|bounded|smoke")
OUT = Path(sys.argv[1]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
KIND = sys.argv[2]
assert KIND in {"direct", "bounded", "smoke"}
DEST = OUT / KIND
DEST.mkdir()
# Freeze source inputs and compiler arguments before the measurement window.
root = SOURCE.parents[2]
sdk = root / "vane_fs/vcpkg_installed/x64-linux-release"
shutil.copy2(SOURCE / "io_probe.cpp", DEST / "io_probe.cpp")
timeline = (SOURCE.parent / "staged_payload/block_sync/timeline.cpp").read_text()
timeline = timeline.replace("threads[8]", "threads[64]").replace("thread >= 8", "thread >= 64")
(DEST / "timeline.cpp").write_text(timeline)
command = [
    "g++",
    "-O3",
    "-DNDEBUG",
    "-std=c++20",
    "-pthread",
    "-DVANE_FS_SPLIT_WAL_SYNC=0",
    "-I" + str(sdk / "include"),
    str(DEST / "io_probe.cpp"),
    str(DEST / "timeline.cpp"),
    str(sdk / "lib/libsqlite3.a"),
    "-ldl",
    "-o",
    str(DEST / "io-probe"),
]
command += [
    "-Wl,--wrap=" + name
    for name in [
        "fsync",
        "fdatasync",
        "sqlite3_step",
        "sqlite3_wal_checkpoint_v2",
        "pread",
        "pwrite",
        "pread64",
        "pwrite64",
        "ftruncate",
        "ftruncate64",
    ]
]
compiled = subprocess.run(command, text=True, capture_output=True)
(DEST / "build.log").write_text(compiled.stdout + compiled.stderr)
(DEST / "build.json").write_text(
    json.dumps(
        {
            "argv": command,
            "exit_code": compiled.returncode,
            "compiler": subprocess.check_output(["g++", "--version"], text=True),
            "source_sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    DEST / "io_probe.cpp",
                    DEST / "timeline.cpp",
                    SOURCE / "run.py",
                    SOURCE / "monitor.py",
                    sdk / "lib/libsqlite3.a",
                ]
            },
        },
        indent=2,
    )
    + "\n"
)
compiled.check_returncode()
build = json.loads((DEST / "build.json").read_text())
build["binary_sha256"] = hashlib.sha256((DEST / "io-probe").read_bytes()).hexdigest()
(DEST / "build.json").write_text(json.dumps(build, indent=2) + "\n")
VARIANTS = [
    ("large-serial", 0, 0, 0, 64),
    ("large-concurrent", 0, 1, 0, 64),
    ("framed-serial", 1, 0, 0, 64),
    ("framed-concurrent", 1, 1, 0, 64),
    ("paced-serial", 1, 0, 100, 64),
    ("paced-concurrent", 1, 1, 100, 64),
]
ORDERS = [list(range(6)), list(reversed(range(6)))] * 3
if KIND == "bounded":
    VARIANTS = [
        ("serial-64", 1, 0, 100, 64),
        ("serial-16", 1, 0, 100, 16),
        ("concurrent-64", 1, 1, 100, 64),
        ("concurrent-16", 1, 1, 100, 16),
    ]
    ORDERS = [
        [0, 1, 3, 2],
        [1, 2, 0, 3],
        [2, 3, 1, 0],
        [3, 0, 2, 1],
        [3, 0, 2, 1],
        [2, 3, 1, 0],
        [1, 2, 0, 3],
        [0, 1, 3, 2],
    ]
if KIND == "smoke":
    VARIANTS = [
        ("large-serial", 0, 0, 0, 64),
        ("large-concurrent", 0, 1, 0, 64),
        ("framed-serial-16", 1, 0, 0, 16),
        ("framed-concurrent-16", 1, 1, 0, 16),
    ]
    ORDERS = [[0, 1, 2, 3]]
REPORT = {
    "status": "RUNNING",
    "started_epoch": time.time(),
    "config": {
        "measurement": KIND != "smoke",
        "variants": VARIANTS,
        "orders": ORDERS,
        "logical_bytes_per_trial": 256 * 2**20,
        "wal_capacity_logical_bytes": sorted({variant[4] * 2**20 for variant in VARIANTS}),
        "barriers_per_group": ["WAL after quarter group", "WAL after complete group", "database after complete group"],
        "host_changes": False,
        "scope": "Instrumented direct-file diagnostic; no SQLite/FUSE. Within each granularity/group size, equal bytes and barriers. Files start fresh; WAL wraps after each group. Smaller groups intentionally add barriers. All rounds retained.",
    },
    "runs": [],
    "cleanup": [],
}
done = threading.Event()
active = []


def save():
    (DEST / "results.json").write_text(json.dumps(REPORT, indent=2) + "\n")


def watch():
    with (DEST / "waits.jsonl").open("x", buffering=1) as log:
        while not done.wait(0.02):
            for process in list(active):
                if process.poll() is not None:
                    continue
                try:
                    tasks = list(Path(f"/proc/{process.pid}/task").iterdir())
                except OSError:
                    continue
                for task in tasks:
                    try:
                        stat = (task / "stat").read_text()
                        state = stat[stat.rfind(")") + 2 :].split()[0]
                        if state == "R":
                            continue
                        syscall = (task / "syscall").read_text().split()
                        log.write(
                            json.dumps(
                                {
                                    "epoch": time.time(),
                                    "pid": process.pid,
                                    "tid": int(task.name),
                                    "state": state,
                                    "wchan": (task / "wchan").read_text().strip(),
                                    "syscall": syscall[0] if syscall else "",
                                }
                            )
                            + "\n"
                        )
                    except OSError:
                        pass


monitor = subprocess.Popen([sys.executable, str(SOURCE / "monitor.py"), str(DEST / "host.jsonl")])
observer = threading.Thread(target=watch, daemon=True)
observer.start()
try:
    save()
    time.sleep(10)
    for rep, order in enumerate(ORDERS):
        for index in order:
            name, framed, concurrent, pace, group_mib = VARIANTS[index]
            assert shutil.disk_usage(DEST).free > 4 * 2**30
            work = Path(tempfile.mkdtemp(prefix="owned-", dir=DEST))
            command = [str(DEST / "io-probe"), str(work), str(framed), str(concurrent), str(pace), str(group_mib)]
            row = {"name": name, "rep": rep, "command": command, "started_epoch": time.time()}
            process = None
            try:
                process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                active.append(process)
                stdout, stderr = process.communicate(timeout=180)
                row.update(pid=process.pid, exit_code=process.returncode, ended_epoch=time.time())
                (DEST / f"{rep}-{name}.stdout").write_text(stdout)
                (DEST / f"{rep}-{name}.stderr").write_text(stderr)
                assert process.returncode == 0, stderr
                result = json.loads(stdout)
                assert result["status"] == "PASS" and result["full_content_validation"]
                assert len(result["groups"]) == 256 // group_mib and len(result["write_requests"]) == 256
                if concurrent:
                    assert all(group["overlap_seconds"] > 0 for group in result["groups"])
                row["result"] = result
                row["files"] = {}
                for name_file in ["wal.raw", "db.raw"]:
                    path = work / name_file
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    row["files"][name_file] = {
                        "bytes": path.stat().st_size,
                        "allocated_bytes": path.stat().st_blocks * 512,
                        "sha256": digest,
                    }
                shutil.copy2(work / "io.timeline.json", DEST / f"{rep}-{name}.timeline.json")
                REPORT["runs"].append(row)
                print(
                    rep,
                    name,
                    round(result["seconds"], 3),
                    "sync max",
                    round(max(g[k] for g in result["groups"] for k in ["first_sync", "remaining_sync", "db_sync"]), 3),
                    flush=True,
                )
            finally:
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
                if process is not None:
                    active.remove(process)
                for path in work.glob("*.json"):
                    if not (DEST / f"{rep}-{name}.timeline.json").exists():
                        shutil.copy2(path, DEST / f"{rep}-{name}.failed.{path.name}")
                shutil.rmtree(work)
                REPORT["cleanup"].append(
                    {
                        "path": str(work),
                        "removed": not work.exists(),
                        "process_stopped": process is None or process.poll() is not None,
                    }
                )
                save()
    REPORT["status"] = "PASS"
except BaseException as error:
    REPORT.update(status="FAIL", error=repr(error))
    raise
finally:
    done.set()
    observer.join(timeout=5)
    monitor.terminate()
    monitor.wait(timeout=10)
    REPORT.update(
        ended_epoch=time.time(),
        monitor_stopped=True,
        observer_stopped=not observer.is_alive(),
        free_bytes_after=shutil.disk_usage(DEST).free,
    )
    save()
