# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Screen exact-range payload allocation before changing the storage component."""

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MIB = 1024**2
VARIANTS = ["delayed", "reserve256k", "reserve1m"]
ORDERS = [
    ["delayed", "reserve256k", "reserve1m"],
    ["reserve1m", "reserve256k", "delayed"],
    ["reserve256k", "reserve1m", "delayed"],
    ["delayed", "reserve1m", "reserve256k"],
    ["reserve1m", "delayed", "reserve256k"],
    ["reserve256k", "delayed", "reserve1m"],
]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(MIB):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def fingerprint(paths):
    return {str(path): digest(path) for path in paths}


def clean(owned, report):
    assert str(owned) not in Path("/proc/mounts").read_text()
    files = {
        str(p.relative_to(owned)): dict(bytes=p.stat().st_size, sha256=digest(p))
        for p in owned.rglob("*")
        if p.is_file()
    }
    shutil.rmtree(owned)
    report["cleanup"].append(dict(path=str(owned), files=files, removed=not owned.exists(), processes_stopped=True))


def execute(binary, owned, variant, state, size, output):
    argv = [str(binary), str(owned), variant, state, str(size)]
    process = None
    start = time.time()
    try:
        with output.open("w") as result, output.with_suffix(".log").open("w") as log:
            process = subprocess.Popen(
                argv,
                stdout=result,
                stderr=log,
                env=dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1"),
            )
            code = process.wait(timeout=180)
        assert code == 0, f"{output.name}: {output.with_suffix('.log').read_text()}"
        row = json.loads(output.read_text())
        assert row["status"] == "PASS" and row["content_and_guard_valid"]
        assert row["bytes"] == size * MIB
        assert row["reserve_calls"] == (0 if variant == "delayed" else size * (4 if variant == "reserve256k" else 1))
        assert row["logical_bytes"] == 4096 + (48 * MIB if state == "after_gc" else 0) + size * MIB
        assert 0 < row["phase_ns"] <= row["total_ns"]
        assert row["total_ns"] == sum(row[k] for k in ("initialization_ns", "write_ns", "final_sync_ns", "close_ns"))
        row.update(command=argv, pid=process.pid, start_epoch=start, end_epoch=time.time())
        return row
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=20)
        assert process is None or process.poll() is not None


def prepare(out, here):
    out.mkdir()
    repo = here.parents[3]
    monitor = here.parents[1] / "io_interference/monitor.py"
    # Hash the installed/default storage independently of all raw-file binaries.
    default = (
        [
            repo / "vane_fs/vcpkg_installed/x64-linux-release/lib/libsqlite3.a",
            repo / ".venv/lib/python3.12/site-packages/vane_fs/_native.cpython-312-x86_64-linux-gnu.so",
        ]
        + sorted((repo / "vane_fs/src").glob("*.[ch]pp"))
        + sorted((repo / "vane_fs/include/vane_fs").glob("*"))
    )
    build = dict(
        status="RUNNING",
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        inputs=fingerprint([Path(__file__), here / "probe.cpp", monitor]),
        default=fingerprint(default),
        binaries={},
        commands=[],
        tests=[],
        cleanup=[],
        started_epoch=time.time(),
        compiler=subprocess.check_output(["c++", "--version"], text=True),
        host=subprocess.check_output(["uname", "-a"], text=True).strip(),
    )
    try:
        for mode in ("plain", "diagnostic", "asan"):
            binary = out / ("probe-" + mode)
            argv = [
                "c++",
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(here / "probe.cpp"),
                "-o",
                str(binary),
            ]
            if mode != "plain":
                argv.append("-DVANE_ALLOCATION_TRACE")
            if mode == "asan":
                argv += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
            start = time.time()
            with (out / ("build-" + mode + ".log")).open("w") as log:
                subprocess.run(argv, check=True, stdout=log, stderr=log, timeout=120)
            build["commands"].append(dict(argv=argv, start=start, end=time.time(), returncode=0))
            build["binaries"][str(binary)] = digest(binary)
            for state in ("fresh", "after_gc"):
                for variant in VARIANTS:
                    owned = Path(tempfile.mkdtemp(prefix="owned-payload-allocation-test-", dir="/dev/shm"))
                    try:
                        row = execute(binary, owned, variant, state, 4, out / f"test-{mode}-{state}-{variant}.json")
                        assert bool(row["calls"]) == (mode != "plain")
                        build["tests"].append(dict(mode=mode, state=state, variant=variant, result=row))
                    finally:
                        clean(owned, build)
                        save(out / "build.json", build)
        build.update(status="PASS", ended_epoch=time.time())
    except BaseException as error:
        build.update(status="FAIL", error=repr(error), ended_epoch=time.time())
        raise
    finally:
        save(out / "build.json", build)


def run(out, here, mode):
    build = json.loads((out / "build.json").read_text())
    assert build["status"] == "PASS" and len(build["tests"]) == 18
    dest = out / mode
    dest.mkdir()
    inputs = dict(build["inputs"], **build["binaries"], **build["default"])
    inputs[str(out / "build.json")] = digest(out / "build.json")
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    medium = subprocess.check_output(
        ["findmnt", "-n", "-o", "SOURCE,FSTYPE,OPTIONS", "-T", str(out)], text=True
    ).strip()
    assert medium.split()[1] == "ext4"
    plan = dict(
        status="FROZEN_BEFORE_COMPARISON",
        frozen_epoch=time.time(),
        scope="ordinary-file allocation hypothesis only; no SQLite/FUSE or component adoption",
        medium=medium,
        mode=mode,
        states=["fresh", "after_gc"],
        sizes_mib=[64, 256],
        orders=ORDERS if mode == "plain" else ORDERS[:2],
        append_bytes=MIB // 4,
        buffer_page_offset=48,
        final_barrier="fdatasync",
        allocation="exact remaining 256 KiB or 1 MiB; KEEP_SIZE; no reservation past final workload EOF",
        gc_fixture="64 MiB initialized; punch middle 16 MiB; truncate last 16 MiB; sync; retain 32 MiB guard",
        timing="open/header initialization, allocation, append, final fdatasync and close; GC fixture outside timing",
        screen_gates=dict(median_and_aggregate_rate_minimum=1.10, paired_wins_minimum=5, pooled_p99_ratio_maximum=1.20),
        no_owned_build_or_test_overlap=True,
        no_cache_eviction_or_host_setting_changes=True,
        all_samples_retained=True,
        inputs=inputs,
        free_bytes_before=shutil.disk_usage(out).free,
    )
    save(dest / "plan.json", plan)
    report = dict(status="RUNNING", started_epoch=time.time(), runs=[], cleanup=[])
    monitor = subprocess.Popen(
        [sys.executable, str(here.parents[1] / "io_interference/monitor.py"), str(dest / "host.jsonl")]
    )
    try:
        time.sleep(20)
        for state in plan["states"]:
            for size in plan["sizes_mib"]:
                for rep, order in enumerate(plan["orders"]):
                    for variant in order:
                        assert monitor.poll() is None
                        assert shutil.disk_usage(out).free >= 2 * 1024**3
                        label = f"{state}-{size}-{rep}-{variant}"
                        owned = Path(tempfile.mkdtemp(prefix="owned-payload-allocation-", dir=dest))
                        try:
                            row = execute(
                                out / ("probe-" + mode), owned, variant, state, size, dest / (label + ".json")
                            )
                            row["rep"] = rep
                            assert bool(row["calls"]) == (mode == "diagnostic")
                            assert row["fiemap_code"] == 0
                            if mode == "diagnostic":
                                assert sum(c[0] == "append" for c in row["calls"]) == size * 4
                                assert sum(c[0] == "reserve" for c in row["calls"]) == row["reserve_calls"]
                                assert sum(c[0] == "final_sync" for c in row["calls"]) == 1
                            row["payload_sha256"] = digest(owned / "payload")
                            report["runs"].append(row)
                            print(label, round(size * 1e9 / row["total_ns"], 3), "MiB/s", flush=True)
                        finally:
                            clean(owned, report)
                            save(dest / "results.json", report)
        assert all(digest(Path(p)) == sha for p, sha in inputs.items())
        report.update(status="PASS", ended_epoch=time.time(), inputs_unchanged=True)
    except BaseException as error:
        report.update(status="FAIL", ended_epoch=time.time(), error=repr(error))
        raise
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        report["monitor_stopped"] = True
        save(dest / "results.json", report)


def distribution(values):
    return dict(
        values=values,
        median=statistics.median(values),
        p99=sorted(values)[min(len(values) - 1, int(len(values) * 0.99))],
        max=max(values),
    )


def analyze(out):
    plain, diagnostic = [json.loads((out / mode / "results.json").read_text()) for mode in ("plain", "diagnostic")]
    assert plain["status"] == diagnostic["status"] == "PASS"
    assert len(plain["runs"]) == 72 and len(diagnostic["runs"]) == 24
    groups = []
    for state in ("fresh", "after_gc"):
        for size in (64, 256):
            rows = [r for r in plain["runs"] if r["state"] == state and r["bytes"] == size * MIB]
            base = [r for r in rows if r["variant"] == "delayed"]
            assert len({r["payload_sha256"] for r in rows}) == 1
            base_rates = [size * 1e9 / r["total_ns"] for r in base]
            base_aggregate = size * 1e9 * len(base) / sum(r["total_ns"] for r in base)
            for variant in VARIANTS:
                data = [r for r in rows if r["variant"] == variant]
                assert len(data) == 6
                rates = [size * 1e9 / r["total_ns"] for r in data]
                aggregate = size * 1e9 * len(data) / sum(r["total_ns"] for r in data)
                paired = [next(b for b in base if b["rep"] == r["rep"])["total_ns"] / r["total_ns"] for r in data]
                ratio = statistics.median(rates) / statistics.median(base_rates)
                tail = max(r["total_ns"] for r in data) / max(r["total_ns"] for r in base)
                checks = dict(
                    median=ratio >= 1.10,
                    aggregate=aggregate / base_aggregate >= 1.10,
                    pairs=sum(v > 1 for v in paired) >= 5,
                    p99=tail <= 1.20,
                )
                traced = [
                    r
                    for r in diagnostic["runs"]
                    if r["state"] == state and r["bytes"] == size * MIB and r["variant"] == variant
                ]
                groups.append(
                    dict(
                        state=state,
                        size_mib=size,
                        variant=variant,
                        median_mib_s=statistics.median(rates),
                        aggregate_mib_s=aggregate,
                        median_ratio=ratio,
                        aggregate_ratio=aggregate / base_aggregate,
                        paired_ratios=paired,
                        wins=sum(v > 1 for v in paired),
                        checks=checks,
                        screen_pass=all(checks.values()) if variant != "delayed" else None,
                        total_ns=distribution([r["total_ns"] for r in data]),
                        write_ns=distribution([r["write_ns"] for r in data]),
                        sync_ns=distribution([r["final_sync_ns"] for r in data]),
                        initialization_ns=distribution([r["initialization_ns"] for r in data]),
                        extents=distribution([len(r["extents"]) for r in data]),
                        allocated_bytes=distribution([r["allocated_bytes"] for r in data]),
                        diagnostic=[
                            dict(
                                rep=r["rep"],
                                reserve_ns=sum(c[2] - c[1] for c in r["calls"] if c[0] == "reserve"),
                                append_ns=sum(c[2] - c[1] for c in r["calls"] if c[0] == "append"),
                                sync_ns=r["final_sync_ns"],
                                extents=len(r["extents"]),
                            )
                            for r in traced
                        ],
                    )
                )
    summary = dict(
        status="PASS",
        plain_runs=72,
        diagnostic_runs=24,
        groups=groups,
        component_candidate_warranted={
            v: all(g["screen_pass"] for g in groups if g["variant"] == v) for v in VARIANTS[1:]
        },
    )
    save(out / "summary.json", summary)
    for group in groups:
        print(
            group["state"],
            group["size_mib"],
            group["variant"],
            round(group["median_mib_s"], 2),
            round(group["aggregate_mib_s"], 2),
            group["wins"],
            group["screen_pass"],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("stage", choices=["prepare", "plain", "diagnostic", "analyze"])
    args = parser.parse_args()
    out, here = args.artifacts.resolve(), Path(__file__).resolve().parent
    if args.stage == "prepare":
        prepare(out, here)
    elif args.stage == "analyze":
        analyze(out)
    else:
        run(out, here, args.stage)


if __name__ == "__main__":
    main()
