# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Check actual busy backoff, non-busy sleep and targeted read-only observation."""

import bisect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import sample
from match import rejection
from prepare import builder, digest, save, validate

HERE = Path(__file__).resolve().parent


def symbols(path):
    text = subprocess.check_output(["nm", "-anC", str(path)], text=True)
    rows = []
    for line in text.splitlines():
        match = re.fullmatch(r"([0-9a-f]+) [tTwW] (.+)", line)
        if match:
            rows.append((int(match[1], 16), match[2]))
    rows.sort()
    return text, rows


def decode(offsets, rows):
    starts = [row[0] for row in rows]
    return [rows[bisect.bisect_right(starts, offset - 1) - 1][1] for offset in offsets]


def main():
    out, started = Path(sys.argv[1]).resolve(), time.time()
    build = json.loads((out / "build.json").read_text())
    assert build["status"] == "PASS"
    report = dict(status="RUNNING", started_epoch=started, commands=[], cleanup=[], checks=[])

    def run(label, command, env=None):
        begin = time.time()
        with (out / f"{label}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=900)
        report["commands"].append(
            dict(label=label, argv=command, started_epoch=begin, ended_epoch=time.time(), returncode=result.returncode)
        )
        save(out / "observer-tests.json", report)
        assert result.returncode == 0, label
        print(label, "PASS", flush=True)

    def cleanup(work):
        files = {
            str(p.relative_to(work)): dict(bytes=p.stat().st_size, sha256=digest(p))
            for p in work.rglob("*")
            if p.is_file()
        }
        shutil.rmtree(work)
        report["cleanup"].append(
            dict(path=str(work), removed=not work.exists(), processes_stopped=True, mount_absent=True, files=files)
        )
        save(out / "observer-tests.json", report)

    try:
        run("observer-unit-tests", [sys.executable, str(HERE / "test_observer.py"), "-v"])
        for mode in ("release", "asan"):
            core, sdk = out / "diagnostic/updated", Path(build["sdk"])
            binary = out / f"sql-test-{mode}"
            flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if mode == "asan" else []
            run(
                f"sql-{mode}-compile",
                [
                    "c++",
                    "-std=c++17",
                    "-O1",
                    "-I" + str(core / "src"),
                    "-I" + str(sdk / "include"),
                    str(HERE / "test_sql.cpp"),
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
            work = Path(tempfile.mkdtemp(prefix="owned-sql-contract-", dir="/dev/shm"))
            try:
                run(
                    f"sql-{mode}-contract",
                    [str(binary), str(work / "test")],
                    dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1"),
                )
                trace = json.loads((work / "test.waits.json").read_text())
                stats = validate(trace)
                text, table = symbols(binary)
                (out / f"sql-{mode}-symbols.txt").write_text(text)
                roots = {(thread["tid"], root[1]): root for thread in trace["threads"] for root in thread["roots"]}
                for name in ("busy_test", "busy_timeout_test", "custom_sleep_test"):
                    calls = [
                        (thread, event)
                        for thread in trace["threads"]
                        for event in thread["events"]
                        if event[0] == "nanosleep" and event[2] >= 0 and roots[thread["tid"], event[2]][0] == name
                    ]
                    assert calls and all(event[14] >= 0 for _, event in calls)
                    busy = [
                        any("sqliteDefaultBusyCallback" in frame for frame in decode(event[18], table))
                        for _, event in calls
                    ]
                    assert all(busy) if name != "custom_sleep_test" else not any(busy)
                    if name == "custom_sleep_test":
                        assert all(
                            thread["statements"][event[14]] == "SELECT diagnostic_sleep(), '\n\"\\\t'"
                            for thread, event in calls
                        )
                errors = [
                    event
                    for thread in trace["threads"]
                    for event in thread["events"]
                    if event[0] == "nanosleep" and event[6] < 0
                ]
                assert sorted(event[7] for event in errors) == [4, 22]
                assert any(
                    event[0] == "sql_step" and event[6] == 5
                    for thread in trace["threads"]
                    for event in thread["events"]
                )
                save(out / f"sql-{mode}-trace.json", trace)
                report["checks"].append(
                    dict(mode=mode, **stats, busy_and_nonbusy_stacks=True, errno_and_remainder_preserved=True)
                )
            finally:
                cleanup(work)
            build["binaries"][str(binary)] = digest(binary)

        work = Path(tempfile.mkdtemp(prefix="owned-sql-sampler-", dir="/dev/shm"))
        process = None
        stop, observations = threading.Event(), []
        observer = None
        try:
            command = [str(out / "sql-test-release"), str(work / "probe"), "--observe"]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            assert process.stdout.readline().strip() == "ready"
            observer = threading.Thread(target=sample.sample, args=(process.pid, stop, observations))
            observer.start()
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stderr
            stop.set()
            observer.join(timeout=10)
            assert (
                not observer.is_alive() and observations[-1]["fatal"] is None and observations[-1]["marker"] is not None
            )
            trace = json.loads((work / "probe.waits.json").read_text())
            validate(trace)
            events = [
                (thread["tid"], event)
                for thread in trace["threads"]
                for event in thread["events"]
                if event[0] == "nanosleep"
            ]
            accepted = [
                point
                for row in observations
                for point in row["targeted_waits"]
                if any(rejection(point, event, tid) is None for tid, event in events)
            ]
            assert len(accepted) >= 2
            save(
                out / "sampler-probe.json",
                dict(
                    status="PASS",
                    argv=command,
                    stdout=stdout,
                    stderr=stderr,
                    observations=observations,
                    accepted=len(accepted),
                    trace=trace,
                ),
            )
            report["sampler_probe"] = dict(
                accepted=len(accepted), read_only=True, mapping_closed=observations[-1]["mapping_closed"]
            )
        finally:
            stop.set()
            if process is not None and process.poll() is None:
                process.terminate()
                process.communicate(timeout=10)
            if observer is not None:
                observer.join(timeout=10)
                assert not observer.is_alive()
            cleanup(work)
        syntax = HERE.parents[2] / "build/sqlite-step-syntax-20261019.json"
        # The syntax check uses the same generated recorder and removes its own fixture.
        assert syntax.is_file()
        syntax_report = json.loads(syntax.read_text())
        assert syntax_report["status"] == "PASS" and syntax_report["cleanup"]["removed"]
        save(out / "syntax-check.json", syntax_report)
        report["cleanup"].append(syntax_report["cleanup"])
        build["cleanup"].extend(report["cleanup"])
        for path in [
            *HERE.glob("*.py"),
            *HERE.glob("*.hpp"),
            *HERE.glob("*.cpp"),
            HERE.parent / "fuse_sync_waits/sample.py",
        ]:
            build["inputs"][str(path)] = digest(path)
        report.update(status="PASS", ended_epoch=time.time(), default_busy_handler_unchanged=True)
        build["ended_epoch"] = report["ended_epoch"]
        save(out / "build.json", build)
        print("PASS SQL/sleep and read-only observer contracts", flush=True)
    except BaseException as error:
        report.update(status="FAIL", error=repr(error), ended_epoch=time.time())
        raise
    finally:
        save(out / "observer-tests.json", report)


if __name__ == "__main__":
    main()
