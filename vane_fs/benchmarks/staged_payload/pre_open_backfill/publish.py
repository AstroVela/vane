# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain the selective comparison, typed syncs and owned-data cleanup evidence."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out, here = args.artifacts.resolve(), Path(__file__).resolve().parent
    audit = dict(
        status="RUNNING", artifacts=str(out), identities={}, windows={}, cleanup=[], local_retained={}, retained={}
    )
    tails = []

    def verify(path, expected):
        path = Path(path)
        assert digest(path) == expected, path
        audit["identities"][str(path)] = expected

    def local(path):
        audit["local_retained"][str(path)] = dict(bytes=path.stat().st_size, sha256=digest(path))

    def copy(path, relative):
        target = here / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        assert digest(path) == digest(target)

    def cleanup(rows):
        for row in rows:
            assert row["removed"] and not Path(row["path"]).exists()
            if "process_stopped" in row:
                assert row["process_stopped"] and row["mount_absent"]
            audit["cleanup"].append(row)

    candidate, tests, comparisons = (read(out / n) for n in ("candidate.json", "tests.json", "comparison.json"))
    assert candidate["status"] == tests["status"] == "PASS"
    assert candidate["background_worker_identical"] and candidate["frozen_policy"]["skip_open"]
    for report in (candidate, tests, comparisons):
        for path, expected in report.get("inputs", {}).items():
            verify(path, expected)
    for path, expected in candidate["sources"].items():
        verify(out / "updated" / path, expected)
    for path, expected in candidate["binaries"].items():
        verify(path, expected)
    cleanup(tests["cleanup"])
    for path in out.glob("*.json"):
        if path.name.startswith("pr-"):
            continue
        copy(path, Path("validation") / path.name)
    for path in out.glob("*.log"):
        copy(path, Path("validation/logs") / path.name)
    for path in (out / "test-logs").rglob("*.log"):
        local(path)
    windows = []
    for name in ("plain", "trace"):
        source, target = out / name, Path(name)
        plan = read(source / "plan.json")
        for path, expected in plan["inputs"].items():
            verify(path, expected)
        data = source / ("ext4-plain" if name == "plain" else "ext4-diagnostic")
        raw = read(data / "results.json")
        assert raw["status"] == "PASS" and raw["monitor_stopped"]
        rows = [*raw["fuse"], *raw["aged_fuse"]]
        expected_mounts = 24 if name == "plain" else 16
        assert len(rows) == expected_mounts
        cleanup(raw["cleanup"])
        for path in source.glob("*.json"):
            copy(path, target / path.name)
        copy(data / "results.json", target / "results.json")
        for path in data.iterdir():
            if path.is_file() and path.name != "results.json":
                local(path)
        for state, key in (("fresh", "fuse"), ("post_gc", "aged_fuse")):
            for row in raw[key]:
                for sample in row["samples"]:
                    for field in (
                        "write_latencies",
                        "operation_latencies",
                        "fsync_seconds",
                        "final_directory_fsync_seconds",
                        "close_seconds",
                    ):
                        values = sample.get(field, [])
                        if isinstance(values, (float, int)):
                            values = [values]
                        for index, seconds in enumerate(values):
                            if seconds >= 1:
                                tails.append(
                                    dict(
                                        window=name,
                                        state=state,
                                        variant=row["variant"],
                                        rep=row["rep"],
                                        workload=sample["workload"],
                                        field=field,
                                        index=index,
                                        seconds=seconds,
                                    )
                                )
        record = dict(
            mounts=expected_mounts,
            start=raw["started_epoch"],
            end=raw["ended_epoch"],
            free_bytes_after=raw["free_bytes_after"],
        )
        windows.append(record)
        audit["windows"][name] = record
    assert windows[0]["end"] <= windows[1]["start"]
    trace_build = read(out / "trace/build.json")
    assert trace_build["status"] == "PASS"
    assert read(out / "trace/source-proof.json")["status"] == "PASS"
    for path, expected in trace_build["inputs"].items():
        verify(path, expected)
    for path, expected in trace_build["binaries"].items():
        verify(out / "trace" / path, expected)
    for variant, entries in trace_build["source_inputs"].items():
        for path, expected in entries.items():
            verify(out / "trace/diagnostic" / variant / path, expected)
    for path in (out / "trace-generator").iterdir():
        if path.is_file():
            copy(path, Path("executed") / (path.name + ".txt"))
    preflight = read(out / "preflight/identity.json")
    assert preflight["measured_mounts"] == 0 and not preflight["runtime_source_or_binary_changed"]
    for path, expected in preflight["files"].items():
        verify(path, expected)
    for path in (out / "preflight").iterdir():
        copy(path, Path("preflight") / path.name)
    for path in (out / "trace").glob("*.log"):
        copy(path, Path("trace/logs") / path.name)
    for name in ("findings.json", "preparation-findings.json"):
        for path, expected in read(out / "trace" / name)["inputs"].items():
            verify(path, expected)
    for report in (candidate, tests, trace_build):
        for run in report.get("runs", report.get("commands", [])):
            assert run.get("end", run.get("ended_epoch")) <= windows[0]["start"]
    audit["no_owned_build_or_test_overlap"] = True
    default = read(here.parent / "checkpoint_followthrough/validation/default-unchanged.json")
    repo = here.parents[3]
    for path, expected in default["sources"].items():
        verify(repo / "vane_fs" / path, expected)
    verify(default["module"]["path"], default["module"]["sha256"])
    audit["default_unchanged"] = True
    (here / "tails.json").write_text(json.dumps(tails, indent=2) + "\n")
    audit["request_or_barrier_tails_at_least_one_second"] = len(tails)
    audit["maximum_retained_tail_seconds"] = max((r["seconds"] for r in tails), default=0)
    audit["free_bytes"] = shutil.disk_usage(out).free
    for path in here.rglob("*"):
        if path.is_file() and path.name != "audit.json" and "__pycache__" not in path.parts:
            audit["retained"][str(path.relative_to(here))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    audit["status"] = "PASS"
    (here / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("PASS", len(audit["retained"]), "files; 40 mounts;", len(audit["cleanup"]), "cleaned paths", flush=True)


if __name__ == "__main__":
    main()
