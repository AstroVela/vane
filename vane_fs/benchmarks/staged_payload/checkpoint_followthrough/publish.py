# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Verify frozen identities, retain all samples and audit owned-data cleanup."""

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
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    audit = dict(
        status="RUNNING",
        local_artifacts=str(out),
        identities={},
        local_retained={},
        windows={},
        cleanup=[],
        retained={},
    )
    tails = []
    archives = [p for folder in (out / "executed", out / "preflight") for p in folder.rglob("*") if p.is_file()]

    def verify(path, expected):
        source = Path(path)
        if not source.exists() or digest(source) != expected:
            matches = [p for p in archives if digest(p) == expected]
            assert matches, (path, expected)
            source = matches[0]
        assert digest(source) == expected
        audit["identities"][str(path)] = dict(verified_path=str(source), sha256=expected)

    def copy(path, target):
        target = here / target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        assert digest(path) == digest(target)

    def local(path):
        audit["local_retained"][str(path)] = dict(bytes=path.stat().st_size, sha256=digest(path))

    def cleanup(rows):
        for row in rows:
            assert row["removed"] and not Path(row["path"]).exists()
            if "process_stopped" in row:
                assert row["process_stopped"] and row["mount_absent"]
            audit["cleanup"].append(row)

    unchanged = read(out / "default-unchanged.json")
    assert unchanged["status"] == "PASS"
    for path, expected in unchanged["sources"].items():
        verify(repo / "vane_fs" / path, expected)
    verify(unchanged["module"]["path"], unchanged["module"]["sha256"])
    copy(out / "default-unchanged.json", Path("validation/default-unchanged.json"))
    build = read(out / "build.json")
    assert build["status"] == "PASS"
    for path, expected in build["inputs"].items():
        verify(path, expected)
    for name, source in build["original_sources"].items():
        for path, expected in source["sha256"].items():
            verify(Path(source["path"]) / path, expected)
        for path, expected in build["source_inputs"][name].items():
            verify(out / "diagnostic" / name / path, expected)
    for path, expected in build["binaries"].items():
        verify(out / path, expected)
    for source, target in ((out, Path("validation/diagnostic")), (out / "budget", Path("validation/budget"))):
        for path in source.glob("*.json"):
            if path.name.startswith("pr-") or path.name in ("default-unchanged.json",):
                continue
            report = read(path)
            if path.name in ("tests.json", "sync-tests.json", "candidate.json"):
                assert report["status"] == "PASS"
                for original, expected in report.get("inputs", {}).items():
                    verify(original, expected)
                cleanup(report.get("cleanup", []))
            if path.name == "candidate.json":
                for original, expected in report["sources"].items():
                    verify(source / "updated" / original, expected)
                for original, expected in report["binaries"].items():
                    verify(original, expected)
            if path.name == "validation-inputs.json":
                for original, expected in report.items():
                    verify(original, expected)
            copy(path, target / path.name)
        for path in source.glob("*.log"):
            copy(path, target / "logs" / path.name)
        for path in (source / "test-logs").rglob("*.log"):
            if path.stat().st_size > 500 * 1024:
                local(path)
            else:
                copy(path, target / "test-logs" / path.relative_to(source / "test-logs"))
    for folder in ("preflight", "executed"):
        for path in (out / folder).rglob("*"):
            if path.is_file():
                # Executed files are evidence, not additional runnable source files.
                target = Path(folder) / path.relative_to(out / folder)
                if path.suffix in (".py", ".cpp"):
                    target = target.with_name(target.name + ".txt")
                copy(path, target)
                if path.name.startswith("sync-tests") and path.suffix == ".json":
                    cleanup(read(path).get("cleanup", []))
    for source, target, kind in (
        (out / "progress-lookup", Path("lookup-diagnostic"), "diagnostic"),
        (out / "lookup-coalesced", Path("clean-diagnostic"), "diagnostic"),
        (out / "budget/ablation", Path("budget-ablation"), "plain"),
        (out / "budget/original-control", Path("original-control"), "plain"),
    ):
        if not source.exists():
            continue
        plan = read(source / "plan.json")
        for path, expected in plan["inputs"].items():
            verify(path, expected)
        if kind == "diagnostic":
            for path, expected in read(source / "build.json")["inputs"].items():
                verify(path, expected)
            findings = read(source / "findings.json")
            assert findings["status"] == "PASS"
            for path, expected in findings["inputs"].items():
                verify(path, expected)
            assert read(source / "source-proof.json")["status"] == "PASS"
        data = source / ("ext4-" + kind)
        raw = read(data / "results.json")
        assert raw["status"] == "PASS" and raw["monitor_stopped"]
        rows = [*raw["fuse"], *raw["aged_fuse"]]
        assert len(rows) == (16 if kind == "diagnostic" else 24)
        cleanup(raw["cleanup"])
        for path in source.glob("*.json"):
            copy(path, target / path.name)
        for row in rows:
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
                                    window=str(target),
                                    variant=row["variant"],
                                    rep=row["rep"],
                                    workload=sample["workload"],
                                    field=field,
                                    index=index,
                                    seconds=seconds,
                                )
                            )
        # Keep the original compact report and all per-request samples losslessly.
        copy(data / "results.json", target / "results.json")
        for path in data.iterdir():
            if path.suffix == ".log":
                copy(path, target / "logs" / path.name)
            elif path.suffix in (".json", ".jsonl") and path.name != "results.json":
                local(path)
        audit["windows"][str(target)] = dict(
            mounts=len(rows),
            kind=kind,
            started_epoch=raw["started_epoch"],
            ended_epoch=raw["ended_epoch"],
            free_bytes_after=raw["free_bytes_after"],
        )
    windows = sorted(audit["windows"].values(), key=lambda r: r["started_epoch"])
    assert all(a["ended_epoch"] <= b["started_epoch"] for a, b in zip(windows, windows[1:]))
    for report_path in (out / "sync-tests.json", out / "budget/candidate.json", out / "budget/tests.json"):
        report = read(report_path)
        for run in report.get("runs", report.get("commands", [])):
            start, end = run.get("start", run.get("started_epoch")), run.get("end", run.get("ended_epoch"))
            if start is not None:
                assert all(end <= w["started_epoch"] or start >= w["ended_epoch"] for w in windows), run
    audit["no_owned_build_or_test_overlap"] = True
    audit["cpp_test_format_only"] = (here / "test_budget.cpp").read_text().split() == (
        out / "executed/test_budget.cpp"
    ).read_text().split()
    assert audit["cpp_test_format_only"]
    (here / "tails.json").write_text(json.dumps(tails, indent=2) + "\n")
    audit["request_or_barrier_tails_at_least_one_second"] = len(tails)
    audit["maximum_request_or_barrier_tail_seconds"] = max((r["seconds"] for r in tails), default=0)
    audit["free_bytes"] = shutil.disk_usage(out).free
    for path in here.rglob("*"):
        if path.is_file() and path.name != "audit.json" and "__pycache__" not in path.parts:
            audit["retained"][str(path.relative_to(here))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    audit["status"] = "PASS"
    (here / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        "PASS",
        len(audit["retained"]),
        "retained files;",
        sum(w["mounts"] for w in windows),
        "mounts;",
        len(audit["cleanup"]),
        "cleaned paths",
        flush=True,
    )


if __name__ == "__main__":
    main()
