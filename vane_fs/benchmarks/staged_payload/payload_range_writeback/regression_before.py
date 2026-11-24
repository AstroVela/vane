# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run the same public-API concurrency regression on the frozen extent control."""

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from generate import digest, save


def main():
    out, here = Path(sys.argv[1]).resolve(), Path(__file__).resolve().parent
    source = json.loads((out / "source.json").read_text())
    target, sdk = Path(source["mechanism_parent"]), Path(source["sdk"])
    helper, binary = here / "test_prewriteback.cpp", out / "test-range-writeback-before"
    argv = [
        "c++",
        "-std=c++17",
        "-O2",
        "-I" + str(target / "include"),
        "-I" + str(target / "src"),
        "-I" + str(sdk / "include"),
        str(helper),
        str(target / "build/python-release/libvane_fs.a"),
        str(sdk / "lib/libsqlite3.a"),
        "-ldl",
        "-pthread",
        "-Wl,--wrap=fdatasync",
        "-Wl,--wrap=sync_file_range",
        "-Wl,--wrap=flock",
        "-o",
        str(binary),
    ]
    report = dict(status="RUNNING", started_epoch=time.time(), commands=[])

    def run(label, argv):
        log = out / (label + ".log")
        begin = time.time()
        with log.open("w") as stream:
            process = subprocess.run(argv, stdout=stream, stderr=stream, timeout=180)
        report["commands"].append(
            dict(
                label=label,
                argv=argv,
                returncode=process.returncode,
                started_epoch=begin,
                ended_epoch=time.time(),
                log=str(log),
                log_sha256=digest(log),
            )
        )
        return process.returncode, log

    try:
        code, log = run("regression-before-compile", argv)
        assert code == 0, log
        owned = Path(tempfile.mkdtemp(prefix="owned-range-writeback-before-", dir="/dev/shm"))
        try:
            code, log = run("regression-before", [str(binary), str(owned)])
            assert code == 1 and "Background prewriteback still performs fdatasync" in log.read_text(), log
        finally:
            assert str(owned) not in Path("/proc/mounts").read_text()
            files = {
                str(p.relative_to(owned)): dict(bytes=p.stat().st_size, sha256=digest(p))
                for p in owned.rglob("*")
                if p.is_file()
            }
            shutil.rmtree(owned)
            report["cleanup"] = dict(
                path=str(owned),
                files=files,
                removed=not owned.exists(),
                processes_stopped=True,
                mount_absent=True,
            )
        report.update(
            status="PASS",
            expected_failure=True,
            inputs={
                str(p): digest(p)
                for p in [
                    Path(__file__),
                    helper,
                    target / "build/python-release/libvane_fs.a",
                    sdk / "lib/libsqlite3.a",
                ]
            },
            binary=dict(path=str(binary), sha256=digest(binary)),
        )
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_epoch"] = time.time()
        save(out / "regression-before.json", report)
    print("PASS: fdatasync prewriteback control fails the same prewriteback regression as expected", flush=True)


if __name__ == "__main__":
    main()
