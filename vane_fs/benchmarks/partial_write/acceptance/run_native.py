# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Measure plain native calls separately from SQL counters, using only owned tmpfs files."""

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("/dev/shm"))
    args = parser.parse_args()
    out = args.artifacts.resolve()
    assert json.loads((out / "tests.json").read_text())["status"] == "PASS"
    assert json.loads((out / "probe-build.json").read_text())["status"] == "PASS"
    filesystem = subprocess.check_output(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(args.data_root)], text=True
    ).strip()
    assert filesystem == "tmpfs", filesystem
    report = {
        "status": "RUNNING",
        "filesystem": filesystem,
        "started_epoch": time.time(),
        "plain": [],
        "diagnostic": [],
        "cleanup": [],
    }
    monitor_source = Path(__file__).resolve().parents[2] / "io_interference/monitor.py"
    monitor = subprocess.Popen([sys.executable, str(monitor_source), str(out / "native-host.jsonl")])

    def save():
        (out / "native-results.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        time.sleep(10)
        assert monitor.poll() is None
        for kind, shifts, repeats in (("plain", (0, 1, 4095), 8), ("diagnostic", (0, 1), 3)):
            for shift in shifts:
                for rep in range(repeats):
                    for variant in ("baseline", "updated") if rep % 2 == 0 else ("updated", "baseline"):
                        free = shutil.disk_usage(args.data_root).free
                        assert free > 2 * 1024**3
                        work = Path(tempfile.mkdtemp(prefix="vane-partial-native-", dir=args.data_root))
                        db = work / "db.sqlite"
                        command = [
                            str(out / (variant + ("-native-probe" if kind == "plain" else "-alignment-probe"))),
                            str(db),
                            str(shift),
                        ]
                        if kind == "plain":
                            command.append("64")
                        row = {
                            "variant": variant,
                            "rep": rep,
                            "shift": shift,
                            "command": command,
                            "started_epoch": time.time(),
                            "free_bytes": free,
                        }
                        report[kind].append(row)
                        try:
                            result = subprocess.run(command, text=True, capture_output=True, timeout=90)
                            name = f"{kind}-{shift}-{rep}-{variant}"
                            (out / (name + ".jsonl")).write_text(result.stdout)
                            (out / (name + ".stderr")).write_text(result.stderr)
                            row.update(returncode=result.returncode, ended_epoch=time.time())
                            result.check_returncode()
                            row["metrics"] = [json.loads(line) for line in result.stdout.splitlines()]
                            assert row["metrics"][-1]["status"] == "PASS"
                            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
                                assert conn.execute("PRAGMA quick_check").fetchall() == [("ok",)]
                                assert conn.execute("SELECT version,block_size FROM format").fetchall() == [(2, 4096)]
                            conn.close()
                            row["quick_check"] = "ok"
                            phase = row["metrics"][0]
                            print(kind, shift, rep, variant, phase.get("cpu_seconds"), flush=True)
                        finally:
                            files = [
                                {
                                    "path": str(p),
                                    "bytes": p.stat().st_size,
                                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                }
                                for p in work.rglob("*")
                                if p.is_file()
                            ]
                            shutil.rmtree(work)
                            report["cleanup"].append({"path": str(work), "files": files, "removed": not work.exists()})
                            save()
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        report["monitor_stopped"] = True
        report["ended_epoch"] = time.time()
        save()


if __name__ == "__main__":
    main()
