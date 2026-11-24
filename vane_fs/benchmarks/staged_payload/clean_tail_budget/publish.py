# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain both causal comparisons, all latency samples and cleanup evidence."""

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

    candidate, tests, comparisons = (read(out / n) for n in ("candidate.json", "tests.json", "comparisons.json"))
    assert candidate["status"] == tests["status"] == "PASS"
    assert candidate["frozen_policy"]["clean_generation_below"] is None
    for report in (candidate, tests, comparisons):
        for path, expected in report.get("inputs", {}).items():
            verify(path, expected)
    for path, expected in candidate["sources"].items():
        verify(out / "updated" / path, expected)
    for path, expected in candidate["binaries"].items():
        verify(path, expected)
    cleanup(tests["cleanup"])
    for filename in ("wrapper-inputs.json", "validation-inputs.json"):
        for path, expected in read(out / filename).items():
            verify(path, expected)
    generator = out.with_name(out.name + "-generator")
    for path, expected in read(generator / "identity.json")["inputs"].items():
        verify(path, expected)
    for path in out.glob("*.json"):
        copy(path, Path("validation") / path.name)
    for path in out.glob("*.log"):
        copy(path, Path("validation/logs") / path.name)
    for path in (out / "test-logs").rglob("*.log"):
        local(path)
    for folder in (out / "preflight", generator, out.with_name(out.name + "-generator-preflight")):
        for path in folder.rglob("*"):
            if path.is_file():
                suffix = ".txt" if path.suffix == ".py" else ""
                copy(path, Path("executed") / folder.name / (path.name + suffix))
    windows = []
    for name in ("whole-generation", "tail"):
        source, target = out / name, Path(name)
        plan = read(source / "plan.json")
        for path, expected in plan["inputs"].items():
            verify(path, expected)
        data = source / "ext4-plain"
        raw = read(data / "results.json")
        assert raw["status"] == "PASS" and raw["monitor_stopped"]
        rows = [*raw["fuse"], *raw["aged_fuse"]]
        assert len(rows) == 24
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
            mounts=24, start=raw["started_epoch"], end=raw["ended_epoch"], free_bytes_after=raw["free_bytes_after"]
        )
        windows.append(record)
        audit["windows"][name] = record
    assert windows[0]["end"] <= windows[1]["start"]
    for report in (candidate, tests):
        for run in report.get("runs", report.get("commands", [])):
            assert run["end"] <= windows[0]["start"]
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
    print("PASS", len(audit["retained"]), "files; 48 mounts;", len(audit["cleanup"]), "cleaned paths", flush=True)


if __name__ == "__main__":
    main()
