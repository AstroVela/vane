# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Verify atomic index removal, history/GC, and unchanged checkpoint/publication policy."""

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

INDEX = ("upgrade", "rollback", "history")
POLICY = ("clean", "continuous", "no_query", "stop")
ORPHANS = ("volume", "fast", "snapshot", "future", "rollback", "lease", "recovery")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    assert build["status"] == "PASS"
    report = dict(status="RUNNING", runs=[], cleanup=[])

    def run(label, binary, case, expected=0, marker=None, env=None):
        work = Path(tempfile.mkdtemp(prefix="owned-vane-payload-index-", dir="/dev/shm"))
        start = time.time()
        try:
            path = out / (label + ".log")
            with path.open("w") as log:
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
            if marker:
                assert marker in path.read_text(), label
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

    def compile_helper(label, source, name, wrappers):
        sdk = Path(build["sdk"]["sqlite"])
        argv = [
            "c++",
            "-std=c++17",
            "-O2",
            "-DNDEBUG",
            *(["-DVANE_TEST_INDEX_REMOVED"] if name == "index-test" else []),
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-I" + str(out / "updated/include"),
            "-I" + str(out / "updated/src"),
            "-I" + str(sdk / "include"),
            str(source),
            str(out / "updated/sanitize/libvane_fs.a"),
            str(sdk / "lib/libsqlite3.a"),
            "-ldl",
            "-pthread",
        ]
        argv += ["-Wl,--wrap=" + symbol for symbol in wrappers] + ["-o", str(out / "updated/sanitize" / name)]
        start = time.time()
        with (out / (label + ".log")).open("w") as log:
            result = subprocess.run(argv, stdout=log, stderr=log, timeout=120)
        report["runs"].append(
            dict(label=label, argv=argv, returncode=result.returncode, started_epoch=start, ended_epoch=time.time())
        )
        result.check_returncode()

    try:
        run(
            "old-orphan-upgrade",
            out / "orphan-control/index-test",
            "upgrade",
            expected=1,
            marker="Old block high index retained",
        )
        run("old-orphan-history", out / "orphan-control/index-test", "history")
        for case in INDEX:
            run("updated-index-" + case, out / "updated/index-test", case)
        for variant in ("baseline", "orphan-control", "updated"):
            for case in POLICY:
                run(variant + "-policy-" + case, out / variant / "policy-test", case)
        for variant in ("orphan-control", "updated"):
            for case in ORPHANS:
                run(variant + "-orphan-" + case, out / variant / "orphan-test", case)
        subprocess.run([sys.executable, str(here.parent / "partial_edges/validate.py"), str(out)], check=True)
        compile_helper(
            "sanitize-index-build",
            here / "test_index.cpp",
            "index-test",
            ("sqlite3_step",),
        )
        compile_helper(
            "sanitize-policy-build",
            here.parent / "checkpoint_batching/test_batch.cpp",
            "policy-test",
            (
                "sqlite3_step",
                "sqlite3_wal_checkpoint_v2",
                "_ZNSt18condition_variable10notify_oneEv",
                "_ZNSt18condition_variable10notify_allEv",
            ),
        )
        compile_helper(
            "sanitize-orphan-build", here.parent / "orphan_batching/test_orphans.cpp", "orphan-test", ("sqlite3_step",)
        )
        env = dict(os.environ, ASAN_OPTIONS="abort_on_error=1", UBSAN_OPTIONS="halt_on_error=1")
        for name, cases in (("index", INDEX), ("policy", POLICY), ("orphan", ORPHANS)):
            for case in cases:
                run("sanitize-" + name + "-" + case, out / "updated/sanitize" / (name + "-test"), case, env=env)
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        (out / "index-tests.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
