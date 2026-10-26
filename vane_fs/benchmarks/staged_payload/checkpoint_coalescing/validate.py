# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Check actual WAL generations, small CLEAN commits, fault fallbacks and related suites."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CASES = ("small", "threshold", "reset", "busy", "io_status", "unknown", "invalid_progress")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    assert build["status"] == "PASS"
    report = dict(status="RUNNING", runs=[], cleanup=[])

    def run(label, binary, case, expected=0, env=None):
        work = Path(tempfile.mkdtemp(prefix="owned-vane-coalesce-", dir="/dev/shm"))
        start = time.time()
        try:
            with (out / (label + ".log")).open("w") as log:
                completed = subprocess.run(
                    [str(binary), str(work / "db.sqlite"), case], stdout=log, stderr=log, env=env, timeout=120
                )
            report["runs"].append(
                dict(
                    label=label,
                    case=case,
                    expected=expected,
                    returncode=completed.returncode,
                    started_epoch=start,
                    ended_epoch=time.time(),
                )
            )
            assert completed.returncode == expected, label
            print(label, "PASS", flush=True)
        finally:
            assert str(work) not in Path("/proc/mounts").read_text()
            files = [
                dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                for p in work.rglob("*")
                if p.is_file()
            ]
            shutil.rmtree(work)
            report["cleanup"].append(dict(path=str(work), removed=not work.exists(), files=files))

    def compile_helper(label, source, binary, wrappers):
        argv = [
            "c++",
            "-std=c++17",
            "-O2",
            "-DNDEBUG",
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-I" + str(out / "updated/include"),
            "-I" + str(Path(build["sdk"]["sqlite"]) / "include"),
            str(source),
            str(out / "updated/sanitize/libvane_fs.a"),
            str(Path(build["sdk"]["sqlite"]) / "lib/libsqlite3.a"),
            "-ldl",
            "-pthread",
        ]
        argv += ["-Wl,--wrap=" + s for s in wrappers] + ["-o", str(binary)]
        start = time.time()
        with (out / (label + ".log")).open("w") as log:
            completed = subprocess.run(argv, stdout=log, stderr=log, timeout=120)
        report["runs"].append(
            dict(label=label, argv=argv, returncode=completed.returncode, started_epoch=start, ended_epoch=time.time())
        )
        completed.check_returncode()

    try:
        for case in ("small", "threshold"):
            run("old-budget-" + case, out / "budget-control/coalesce-test", case, expected=1)
        for case in CASES:
            run("updated-coalesce-" + case, out / "updated/coalesce-test", case)
        spec = importlib.util.spec_from_file_location("budget_validation", here.parent / "writeback_budget/validate.py")
        budget = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(budget)
        for variant, cases in (("baseline", ("normal", "small")), ("updated", budget.CASES)):
            for case in cases:
                run(
                    variant + "-writeback-" + case,
                    out / variant / "writeback-test",
                    case,
                    expected=1 if variant == "baseline" and case == "normal" else 0,
                )
        subprocess.run([sys.executable, str(here.parent / "partial_edges/validate.py"), str(out)], check=True)
        sanitized = out / "updated/sanitize"
        compile_helper(
            "sanitize-coalesce-build",
            here / "test_coalesce.cpp",
            sanitized / "coalesce-test",
            ("sqlite3_step", "sqlite3_wal_checkpoint_v2"),
        )
        compile_helper(
            "sanitize-writeback-build",
            here.parent / "writeback_budget/test_budget.cpp",
            sanitized / "writeback-test",
            ("sync_file_range", "fdatasync", "fsync", "close"),
        )
        env = dict(os.environ, ASAN_OPTIONS="abort_on_error=1", UBSAN_OPTIONS="halt_on_error=1")
        for case in CASES:
            run("sanitize-coalesce-" + case, sanitized / "coalesce-test", case, env=env)
        for case in budget.CASES:
            run("sanitize-writeback-" + case, sanitized / "writeback-test", case, env=env)
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        (out / "coalesce-tests.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
