# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run inherited restart/publication tests and real post-reuse progress cases."""

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

CASES = ("cold", "reset", "live", "reader", "stream")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    assert build["status"] == "PASS"
    subprocess.run([sys.executable, str(here.parent / "checkpoint_restart/validate.py"), str(out)], check=True)
    result = dict(
        status="RUNNING",
        runs=[],
        cleanup=[],
        inputs={str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    )

    def save():
        (out / "progress-tests.json").write_text(json.dumps(result, indent=2) + "\n")

    def run(label, binary, case, expected=0, env=None):
        work = Path(tempfile.mkdtemp(prefix="owned-vane-clean-progress-", dir="/dev/shm"))
        started = time.time()
        log = out / (label + ".log")
        try:
            with log.open("w") as stream:
                done = subprocess.run(
                    [str(binary), str(work / "db.sqlite"), case], stdout=stream, stderr=stream, env=env, timeout=30
                )
            result["runs"].append(
                dict(
                    label=label,
                    case=case,
                    returncode=done.returncode,
                    expected=expected,
                    started_epoch=started,
                    ended_epoch=time.time(),
                )
            )
            assert done.returncode == expected, label
            if expected:
                assert "Small post-reuse commit was not backfilled without further commits" in log.read_text(), label
            print(label, "PASS", flush=True)
        finally:
            assert str(work) not in Path("/proc/mounts").read_text()
            shutil.rmtree(work)
            result["cleanup"].append(dict(path=str(work), removed=not work.exists()))
            save()

    try:
        for variant in ("orphan-control", "restart-control"):
            run(variant + "-progress-cold", out / variant / "test_progress", "cold")
            run(variant + "-progress-reset", out / variant / "test_progress", "reset", expected=1)
        for case in CASES:
            run("updated-progress-" + case, out / "updated/test_progress", case)
        sdk = Path(build["sdk"]["sqlite"])
        binary = out / "updated/sanitize/test_progress"
        argv = [
            "c++",
            "-std=c++17",
            "-O2",
            "-DNDEBUG",
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-I" + str(out / "updated/include"),
            "-I" + str(out / "updated/src"),
            "-I" + str(sdk / "include"),
            str(here / "test_progress.cpp"),
            str(out / "updated/sanitize/libvane_fs.a"),
            str(sdk / "lib/libsqlite3.a"),
            "-ldl",
            "-pthread",
            "-Wl,--wrap=sqlite3_wal_checkpoint_v2",
            "-Wl,--wrap=fsync",
            "-Wl,--wrap=fdatasync",
            "-o",
            str(binary),
        ]
        started = time.time()
        with (out / "sanitize-progress-build.log").open("w") as log:
            done = subprocess.run(argv, stdout=log, stderr=log, timeout=120)
        result["runs"].append(
            dict(
                label="sanitize-progress-build",
                argv=argv,
                returncode=done.returncode,
                started_epoch=started,
                ended_epoch=time.time(),
            )
        )
        done.check_returncode()
        result["inputs"][str(binary)] = hashlib.sha256(binary.read_bytes()).hexdigest()
        env = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1")
        for case in CASES:
            run("sanitize-progress-" + case, binary, case, env=env)
        result["status"] = "PASS"
    except BaseException as error:
        result.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
