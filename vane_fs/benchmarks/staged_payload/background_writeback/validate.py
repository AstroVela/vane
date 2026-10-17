# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Check hint fallbacks, error propagation and ordering, then related suites."""

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

CASES = (
    "normal",
    "small",
    "unsupported",
    "invalid",
    "not_supported",
    "interrupt",
    "io_error",
    "no_space",
    "barrier_error",
    "blocked",
    "coalesce",
    "close_blocked",
    "close_error",
    "fork",
    "gc",
    "post_barrier_error",
    "crash",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--helpers-only", action="store_true", help="Reuse already passed unchanged component suites")
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    assert build["status"] == "PASS"
    report = dict(status="RUNNING", runs=[], cleanup=[])

    def run(label, binary, mode, expected=0, env=None):
        work = Path(tempfile.mkdtemp(prefix="owned-vane-writeback-", dir="/dev/shm"))
        try:
            start = time.time()
            with (out / (label + ".log")).open("w") as log:
                result = subprocess.run(
                    [str(binary), str(work / "db.sqlite"), mode], stdout=log, stderr=log, env=env, timeout=120
                )
            report["runs"].append(
                dict(
                    label=label,
                    returncode=result.returncode,
                    expected=expected,
                    started_epoch=start,
                    ended_epoch=time.time(),
                )
            )
            assert result.returncode == expected, label
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
        for variant, cases in (("baseline", ("normal", "small")), ("updated", CASES)):
            for mode in cases:
                run(
                    variant + "-" + mode,
                    out / variant / "writeback-test",
                    mode,
                    expected=1 if variant == "baseline" and mode == "normal" else 0,
                )
        if args.helpers_only:
            assert json.loads((out / "tests.json").read_text())["status"] == "PASS"
            assert json.loads((out / "sanitize.json").read_text())["status"] == "PASS"
        else:
            subprocess.run([sys.executable, str(here.parent / "partial_edges/validate.py"), str(out)], check=True)
        sdk = Path(build["sdk"]["sqlite"])
        sanitized = out / "updated/sanitize"
        binary = sanitized / "writeback-test"
        argv = [
            "c++",
            "-std=c++17",
            "-O2",
            "-DNDEBUG",
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-I" + str(out / "updated/include"),
            str(here / "test_background.cpp"),
            str(sanitized / "libvane_fs.a"),
            str(sdk / "lib/libsqlite3.a"),
            "-ldl",
            "-pthread",
            "-Wl,--wrap=sync_file_range",
            "-Wl,--wrap=fdatasync",
            "-Wl,--wrap=fsync",
            "-Wl,--wrap=close",
            "-o",
            str(binary),
        ]
        with (out / "sanitize-writeback-build.log").open("w") as log:
            subprocess.run(argv, stdout=log, stderr=log, check=True)
        env = dict(os.environ, ASAN_OPTIONS="abort_on_error=1", UBSAN_OPTIONS="halt_on_error=1")
        for mode in CASES:
            run("sanitize-" + mode, binary, mode, env=env)
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        (out / "writeback-tests.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
