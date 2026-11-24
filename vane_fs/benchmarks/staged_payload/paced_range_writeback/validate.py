# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate actual FULL pauses and retirement using the frozen native libraries."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from generate import digest, save


def main():
    out, here = Path(sys.argv[1]).resolve(), Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    source = json.loads((out / "source.json").read_text())
    assert build["status"] == "PASS" and build["stages"]["paced_range_writeback"] == "PASS"
    assert all(digest(Path(p)) == sha for p, sha in source["sources"].items())
    report = dict(
        status="RUNNING",
        started_epoch=time.time(),
        commands=[],
        cleanup=[],
        binaries={},
        inputs={
            str(p): digest(p)
            for p in [
                Path(__file__),
                here / "test_full_pause.cpp",
                here.parent / "payload_range_writeback/test_prewriteback.cpp",
                out / "build.json",
                out / "source.json",
            ]
        },
    )
    target, sdk = Path(source["candidate"]), Path(source["sdk"])

    def run(label, argv, env=None):
        log = out / (label + ".log")
        start = time.time()
        with log.open("w") as stream:
            r = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=180)
        report["commands"].append(
            dict(
                label=label,
                argv=argv,
                started_epoch=start,
                ended_epoch=time.time(),
                returncode=r.returncode,
                log=str(log),
                sha256=digest(log),
            )
        )
        save(out / "full-pauses.json", report)
        assert r.returncode == 0, log
        print(label, "PASS", flush=True)

    try:
        for mode, directory in (("release", target / "build/python-release"), ("asan", target / "sanitize")):
            binary = out / (mode + "-full-pauses")
            flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if mode == "asan" else []
            run(
                mode + "-full-pauses-compile",
                [
                    "c++",
                    "-std=c++17",
                    "-O2",
                    "-I" + str(target / "src"),
                    "-I" + str(target / "include"),
                    "-I" + str(sdk / "include"),
                    str(here / "test_full_pause.cpp"),
                    str(directory / "libvane_fs.a"),
                    str(sdk / "lib/libsqlite3.a"),
                    "-ldl",
                    "-pthread",
                    *flags,
                    *["-Wl,--wrap=" + name for name in ("fsync", "fdatasync", "flock", "sync_file_range")],
                    "-o",
                    str(binary),
                ],
            )
            owned = Path(tempfile.mkdtemp(prefix="owned-paced-full-", dir="/dev/shm"))
            try:
                run(
                    mode + "-full-pauses",
                    [str(binary), str(owned)],
                    dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1"),
                )
            finally:
                assert str(owned) not in Path("/proc/mounts").read_text()
                files = {
                    str(p.relative_to(owned)): dict(bytes=p.stat().st_size, sha256=digest(p))
                    for p in owned.rglob("*")
                    if p.is_file()
                }
                shutil.rmtree(owned)
                report["cleanup"].append(
                    dict(
                        path=str(owned),
                        files=files,
                        removed=not owned.exists(),
                        processes_stopped=True,
                        mount_absent=True,
                    )
                )
                save(out / "full-pauses.json", report)
            report["binaries"][str(binary)] = digest(binary)
        report["status"] = "PASS"
        report["cases_per_mode"] = 8
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_epoch"] = time.time()
        save(out / "full-pauses.json", report)


if __name__ == "__main__":
    main()
