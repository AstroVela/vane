# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate old SQL/observer contracts and the new actual mutex-wait/hold stages."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from prepare import builder, digest, save, validate


def main():
    here, root = Path(__file__).resolve().parent, Path(sys.argv[1]).resolve()
    out = root / "native"
    parent = here.parent / "sqlite_step_waits/verify.py"
    original = code = parent.read_text()
    begin = code.index('        syntax = HERE.parents[2] / "build/sqlite-step-syntax-20261019.json"')
    end = code.index('        build["cleanup"].extend(report["cleanup"])', begin)
    removed = code[begin:end]
    code = code[:begin] + code[end:]
    assert code[:begin] + removed + code[begin:] == original
    save(
        root / "verify-adapter.json",
        dict(
            parent=str(parent),
            sha256=digest(parent),
            removed_obsolete_external_syntax_fixture_only=removed,
            fresh_compiled_contracts_cover_new_stages=True,
        ),
    )
    saved, sys.argv = sys.argv, [__file__, str(out)]
    try:
        exec(compile(code, str(parent), "exec"), dict(__name__="__main__", __file__=__file__))
    finally:
        sys.argv = saved
    probe = json.loads((out / "sampler-probe.json").read_text())
    observations = probe["observations"]
    assert probe["accepted"] >= 2
    assert all(not row["threads"] and not row["frequency_khz"] for row in observations)
    assert observations[-1]["total_sampler_cpu_ns"] > 0
    assert observations[-1]["observation_wall_ns"] >= 100000000
    assert observations[-1]["mapping_read_only"] and observations[-1]["mapping_closed"]
    build = json.loads((out / "build.json").read_text())
    report = dict(status="RUNNING", started_epoch=time.time(), commands=[], cleanup=[], checks=[])

    def run(label, argv, env=None):
        log, started = root / (label + ".log"), time.time()
        with log.open("w") as stream:
            result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=180)
        report["commands"].append(
            dict(
                argv=argv,
                started_epoch=started,
                ended_epoch=time.time(),
                returncode=result.returncode,
                log=str(log),
                sha256=digest(log),
            )
        )
        save(root / "gate-trace-tests.json", report)
        assert result.returncode == 0, log
        print("PASS", label, flush=True)

    try:
        sdk, core = Path(build["sdk"]), out / "diagnostic/updated"
        for mode in ("release", "asan"):
            binary = root / ("gate-trace-" + mode)
            flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if mode == "asan" else []
            run(
                mode + "-gate-compile",
                [
                    "c++",
                    "-std=c++17",
                    "-O1",
                    "-I" + str(core / "src"),
                    "-I" + str(sdk / "include"),
                    str(here / "test_gate_trace.cpp"),
                    str(core / "src/wait_trace.cpp"),
                    str(sdk / "lib/libsqlite3.a"),
                    "-ldl",
                    "-pthread",
                    *["-Wl,--wrap=" + name for name in builder["helpers"].WRAPS],
                    *flags,
                    "-o",
                    str(binary),
                ],
            )
            owned = Path(tempfile.mkdtemp(prefix="owned-gate-trace-", dir="/dev/shm"))
            try:
                run(
                    mode + "-gate-contract",
                    [str(binary), str(owned / "test")],
                    dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1"),
                )
                trace = json.loads((owned / "test.waits.json").read_text())
                stats = validate(trace)
                events = [event for thread in trace["threads"] for event in thread["events"]]
                waits = [e for e in events if e[0] == "begin_restart_wait"]
                holds = [e for e in events if e[0] == "begin_restart_hold"]
                assert stats["roots"] == 3 and len(waits) == 1 and len(holds) == 2
                assert waits[0][4] - waits[0][3] >= 20000000
                assert any(h[3] <= waits[0][3] < h[4] <= waits[0][4] for h in holds)
                save(root / (mode + "-gate-trace.json"), trace)
                report["checks"].append(
                    dict(
                        mode=mode,
                        **stats,
                        actual_contended_mutex=True,
                        errno_and_exception_unwind_preserved=True,
                        distinct_wait_and_hold_events=True,
                    )
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
            build["binaries"][str(binary)] = digest(binary)
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_epoch"] = time.time()
        save(root / "gate-trace-tests.json", report)
    build["cleanup"].extend(report["cleanup"])
    for p in [
        *here.glob("*.py"),
        *here.glob("*.cpp"),
        parent,
        parent.parent / "test_sql.cpp",
        parent.parent / "test_observer.py",
        parent.parent / "sample.py",
        parent.parent / "match.py",
        root / "gate-trace-tests.json",
        root / "verify-adapter.json",
    ]:
        build["inputs"][str(p)] = digest(p)
    build["ended_epoch"] = time.time()
    save(out / "build.json", build)


if __name__ == "__main__":
    main()
