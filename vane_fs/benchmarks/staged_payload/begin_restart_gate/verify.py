# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate real SQLite lock contention, negative controls, gate lifetime and progress."""

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
    source = json.loads((out / "source.json").read_text())
    build = json.loads((out / "build.json").read_text())
    assert source["status"] == build["status"] == "PASS"
    assert all(digest(Path(p)) == sha for p, sha in (source["inputs"] | source["sources"]).items())
    prefix = ""
    cleanup = []
    preflights = []
    if (out / "gate-tests.json").exists():
        previous = json.loads((out / "gate-tests.json").read_text())
        assert previous["status"] == "FAIL"
        saved = out / "gate-tests-preflight.json"
        assert not saved.exists()
        shutil.copy2(out / "gate-tests.json", saved)
        preflights.append(str(saved))
        cleanup.extend(previous["cleanup"])
        prefix = "retry-"
    if (out / "progress-probe.json").exists():
        probe = json.loads((out / "progress-probe.json").read_text())
        cleanup.extend(probe["cleanup"])
        preflights.append(str(out / "progress-probe.json"))
    report = dict(
        status="RUNNING",
        started_epoch=time.time(),
        commands=[],
        cleanup=cleanup,
        binaries={},
        inputs={},
        preflights=preflights,
    )
    for path in [Path(__file__), here / "test_gate.cpp", out / "source.json", out / "build.json"]:
        report["inputs"][str(path)] = digest(path)

    def run(label, argv, env=None):
        log, started = out / (prefix + label + ".log"), time.time()
        code, failure = None, None
        try:
            with log.open("w") as stream:
                code = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=180).returncode
        except BaseException as error:
            failure = repr(error)
        report["commands"].append(
            dict(
                label=label,
                argv=argv,
                started_epoch=started,
                ended_epoch=time.time(),
                returncode=code,
                failure=failure,
                log=str(log),
                sha256=digest(log),
            )
        )
        save(out / "gate-tests.json", report)
        assert code == 0 and failure is None, log
        print(label, "PASS", flush=True)

    try:
        sdk, target, parent = Path(source["sdk"]), Path(source["candidate"]), Path(source["mechanism_parent"])
        for mode, directory in (("release", "build/python-release"), ("asan", "sanitize")):
            flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if mode == "asan" else []
            for variant, root in (("parent", parent), ("candidate", target)):
                binary, library = out / (prefix + mode + "-gate-" + variant), root / directory / "libvane_fs.a"
                report["inputs"][str(library)] = digest(library)
                run(
                    mode + "-gate-" + variant + "-compile",
                    [
                        "c++",
                        "-std=c++17",
                        "-O2",
                        "-I" + str(root / "include"),
                        "-I" + str(sdk / "include"),
                        str(here / "test_gate.cpp"),
                        str(library),
                        str(sdk / "lib/libsqlite3.a"),
                        "-ldl",
                        "-pthread",
                        *flags,
                        *["-Wl,--wrap=" + name for name in ("sqlite3_step", "sqlite3_wal_checkpoint_v2", "nanosleep")],
                        "-o",
                        str(binary),
                    ],
                )
                report["binaries"][str(binary)] = digest(binary)
                owned = Path(tempfile.mkdtemp(prefix="owned-begin-gate-" + variant + "-", dir="/dev/shm"))
                try:
                    run(
                        mode + "-gate-" + variant,
                        [str(binary), str(owned), *(["--parent-control"] if variant == "parent" else [])],
                        dict(
                            os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1"
                        ),
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
                    save(out / "gate-tests.json", report)
        report.update(status="PASS", cases_per_mode=7, parent_cases_per_mode=2, parent_negative_control="PASS")
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_epoch"] = time.time()
        save(out / "gate-tests.json", report)


if __name__ == "__main__":
    main()
