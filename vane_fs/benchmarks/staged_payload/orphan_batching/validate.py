# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Check cleanup intervals, live handles, rollback, recovery and SQL work."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CASES = ("volume", "fast", "snapshot", "future", "rollback", "lease", "recovery")


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
        work = Path(tempfile.mkdtemp(prefix="owned-vane-orphan-", dir="/dev/shm"))
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

    try:
        for variant in ("baseline", "updated"):
            for case in CASES:
                run(
                    variant + "-" + case,
                    out / variant / "orphan-test",
                    case,
                    expected=1 if variant == "baseline" and case == "volume" else 0,
                )
        subprocess.run([sys.executable, str(here.parent / "partial_edges/validate.py"), str(out)], check=True)
        binary = out / "updated/sanitize/orphan-test"
        argv = [
            "c++",
            "-std=c++17",
            "-O2",
            "-DNDEBUG",
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-I" + str(out / "updated/include"),
            "-I" + str(Path(build["sdk"]["sqlite"]) / "include"),
            str(here / "test_orphans.cpp"),
            str(out / "updated/sanitize/libvane_fs.a"),
            str(Path(build["sdk"]["sqlite"]) / "lib/libsqlite3.a"),
            "-ldl",
            "-pthread",
            "-Wl,--wrap=sqlite3_step",
            "-o",
            str(binary),
        ]
        start = time.time()
        with (out / "sanitize-helper-build.log").open("w") as log:
            completed = subprocess.run(argv, stdout=log, stderr=log, timeout=120)
        report["runs"].append(
            dict(
                label="sanitize-helper-build",
                argv=argv,
                returncode=completed.returncode,
                started_epoch=start,
                ended_epoch=time.time(),
            )
        )
        completed.check_returncode()
        env = dict(os.environ, ASAN_OPTIONS="abort_on_error=1", UBSAN_OPTIONS="halt_on_error=1")
        for case in CASES:
            run("sanitize-" + case, binary, case, env=env)
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        (out / "orphan-tests.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
