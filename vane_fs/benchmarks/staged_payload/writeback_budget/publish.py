# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain complete samples, identities and cleanup evidence after measurement."""

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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value, compact=False):
    path.write_text(json.dumps(value, separators=(",", ":") if compact else None, indent=None if compact else 2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--diagnostic", type=Path, required=True)
    args = parser.parse_args()
    out, diagnostic = args.artifacts.resolve(), args.diagnostic.resolve()
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    for name in ("build.json", "tests.json", "sanitize.json", "writeback-tests.json", "negative-tests.json"):
        assert json.loads((out / name).read_text())["status"] == "PASS", name
    plan = json.loads((out / "plan.json").read_text())
    assert all(digest(Path(p)) == h for p, h in plan["inputs"].items())
    windows, raw_windows = {}, []
    for medium in ("tmpfs", "ext4"):
        raw = json.loads((out / (medium + "-plain/results.json")).read_text())
        assert raw["status"] == "PASS" and raw["monitor_stopped"]
        assert len(raw["fuse"]) == len(raw["aged_fuse"]) == 12
        raw_windows.append(raw)
        dest = here / "plain"
        dest.mkdir(exist_ok=True)
        for state, label in (("fuse", "fresh"), ("aged_fuse", "aged")):
            for variant in ("baseline", "updated"):
                rows = [r for r in raw[state] if r["variant"] == variant]
                assert len(rows) == 6
                write(dest / (medium + "-" + label + "-" + variant + ".json"), rows, True)
        windows[medium] = {k: v for k, v in raw.items() if k not in ("fuse", "aged_fuse")}
    write(here / "plain/windows.json", windows, True)
    shutil.copy2(out / "plan.json", here / "plain/plan.json")
    analyzer = here.parent / "partial_sync/analyze_writeback.py"
    subprocess.run([sys.executable, str(analyzer), str(out)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, str(analyzer), str(here / "plain")], check=True, stdout=subprocess.DEVNULL)
    summary = json.loads((out / "summary.json").read_text())
    assert json.loads((here / "plain/summary.json").read_text()) == summary

    dest = here / "diagnostic"
    dest.mkdir(exist_ok=True)
    raw = json.loads((diagnostic / "ext4-diagnostic/results.json").read_text())
    assert raw["status"] == "PASS" and raw["monitor_stopped"]
    assert len(raw["fuse"]) == len(raw["aged_fuse"]) == 4
    raw_windows.append(raw)
    for state, label in (("fuse", "fresh"), ("aged_fuse", "aged")):
        for variant in ("baseline", "updated"):
            write(dest / (label + "-" + variant + ".json"), [r for r in raw[state] if r["variant"] == variant], True)
    write(dest / "windows.json", {k: v for k, v in raw.items() if k not in ("fuse", "aged_fuse")}, True)
    for name in ("build.json", "plan.json", "diagnostic-summary.json", "background-diagnostic.json"):
        shutil.copy2(diagnostic / name, dest / name)

    empty_logs = []
    for root, target in ((out, here), (diagnostic, dest)):
        for p in sorted(root.glob("*.json")):
            if p.name not in ("plan.json", "summary.json"):
                shutil.copy2(p, target / p.name)
        for p in sorted(root.glob("*.log")):
            if p.stat().st_size:
                shutil.copy2(p, target / p.name)
            else:
                empty_logs.append(dict(path=str(p), sha256=digest(p)))
    for p in sorted((out / "test-logs").rglob("*.log")):
        if p.stat().st_size:
            target = here / "test-logs" / p.relative_to(out / "test-logs")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        else:
            empty_logs.append(dict(path=str(p), sha256=digest(p)))
    write(
        here / "empty-logs.json", dict(omitted_empty_published_copies=empty_logs, original_logs_retained_locally=True)
    )

    build = json.loads((out / "build.json").read_text())
    control = Path(build["parent"]) / "updated"
    with tempfile.TemporaryDirectory(prefix="budget-patch-proof-", dir=out) as temp:
        check = Path(temp) / "source"
        shutil.copytree(
            control, check, ignore=shutil.ignore_patterns("build", "venv", "sanitize", "__pycache__", "*.orig")
        )
        result = subprocess.run(
            ["patch", "--batch", "--fuzz=0", "--no-backup-if-mismatch", "-p1", "-i", str(here / "budget.patch")],
            cwd=check,
            capture_output=True,
            text=True,
            check=True,
        )
        expected = build["source_inputs"]["updated"]
        assert all(digest(check / p) == h for p, h in expected.items())
        changed = [p for p in expected if expected[p] != build["source_inputs"]["baseline"][p]]
        assert set(changed) == {"src/staging.hpp", "src/workspace.cpp"}
        proof_path = str(Path(temp))
    write(
        here / "reproducer-proof.json",
        dict(
            status="PASS",
            patch_fuzz=0,
            patch_sha256=digest(here / "budget.patch"),
            patch_stdout=result.stdout,
            matched_source_files=len(expected),
            modified_source_files=changed,
            split_sample_analysis_equals_original=True,
            frozen_measurement_input_hashes_match=True,
            all_raw_latency_vectors_retained=True,
            proof_temporary_directory=proof_path,
            removed=not Path(proof_path).exists(),
            analyzer_sha256=digest(analyzer),
        ),
    )

    commands = list(build["commands"]) + json.loads((out / "helper-builds.json").read_text())["commands"]
    commands += json.loads((out / "tests.json").read_text())["runs"]
    commands += json.loads((diagnostic / "build.json").read_text())["commands"]
    cleanup = []
    for p in sorted(out.glob("*.json")):
        value = json.loads(p.read_text())
        if isinstance(value, dict):
            for row in value.get("runs", []):
                if "started_epoch" in row and "ended_epoch" in row:
                    commands.append(row)
            for row in value.get("cleanup", []):
                cleanup.append(
                    dict(path=row["path"], removed=row["removed"], evidence=str(p), files=len(row.get("files", [])))
                )
    mounts = 0
    for window in raw_windows:
        assert all(
            c["ended_epoch"] <= window["started_epoch"] or c["started_epoch"] >= window["ended_epoch"] for c in commands
        ), "Owned build or correctness test overlaps a measurement window"
        for state in ("fuse", "aged_fuse"):
            mounts += len(window[state])
            assert all(
                row["inspection"]["quick_check"] == [["ok"]]
                and row["inspection"]["format"] == [[1004, 4096]]
                and all(size in (8, 4096) for size, _ in row["inspection"]["payload_lengths"])
                for row in window[state]
            )
        for row in window["cleanup"]:
            assert row.get("process_stopped", True) and row.get("mount_absent", True)
            if "pid" in row:
                assert not Path("/proc", str(row["pid"])).exists()
            cleanup.append(
                dict(
                    path=row["path"],
                    removed=row["removed"],
                    evidence="measurement windows",
                    files=len(row.get("files", [])),
                )
            )
    assert mounts == 56
    assert all(c["removed"] and not Path(c["path"]).exists() for c in cleanup)
    identity = json.loads((out / "source-identity.json").read_text())
    assert all(digest(repo / "vane_fs" / p) == h for p, h in identity["production"].items())
    assert digest(Path(identity["installed_component"]["path"])) == identity["installed_component"]["sha256"]
    assert digest(Path(build["sdk"]["sqlite"]) / "lib/libsqlite3.a") == identity["sqlite_sdk_sha256"]
    write(
        here / "audit.json",
        dict(
            status="PASS",
            completed_epoch=time.time(),
            artifact_roots=dict(plain=str(out), diagnostic=str(diagnostic)),
            successful_measurement_mounts=mounts,
            measurement_processes_stopped=True,
            measurement_data_removed=True,
            no_owned_build_or_correctness_test_overlap=True,
            production_sources_match_baseline=True,
            installed_component_matches_baseline=True,
            sqlite_sdk_matches_baseline=True,
            decision="ADOPT" if summary["pass_all"] else "REJECT",
            failed_frozen_checks=[c for c in summary["checks"] if not c["pass_gate"]],
            cleanup=cleanup,
            unique_removed_cleanup_paths=len({c["path"] for c in cleanup}),
            free_bytes={str(p): shutil.disk_usage(p).free for p in (repo, Path("/dev/shm"))},
            notes=[
                "Only isolated format-1004 candidates are measured; production format 2 is unchanged.",
                "Per-coordinator optional submission budget; already claimed calls and other coordinators remain independent.",
                "Software crash/fault checks do not certify physical power-loss behavior.",
            ],
        ),
    )

    paths = {}

    def retain(p, scope):
        if p.is_file():
            paths[str(p)] = dict(path=str(p), scope=scope, bytes=p.stat().st_size, sha256=digest(p))

    for p in here.rglob("*"):
        if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts:
            assert p.suffix != ".json" or p.stat().st_size < 500 * 1024, p
            retain(p, "published")
    for root in (out, diagnostic):
        for p in root.glob("*"):
            if p.suffix in (".json", ".log", ".cpp", ".py"):
                retain(p, "local-evidence")
        for name in ("tmpfs-plain", "ext4-plain", "ext4-diagnostic", "test-logs"):
            for p in (root / name).rglob("*"):
                if p.suffix in (".json", ".jsonl", ".log"):
                    retain(p, "local-evidence")
    for name, source in (
        ("baseline", control),
        ("updated", out / "updated"),
        ("unbudgeted", Path(build["unbudgeted_parent"]) / "updated"),
    ):
        for p in build["source_inputs"][name]:
            retain(source / p, "source")
        for p in (
            source / "CMakeLists.txt",
            source / "pyproject.toml",
            source / "build/python-release/libvane_fs.a",
            source / "build/python-release/vane-fs",
        ):
            retain(p, "dependency")
    for p in build["binaries"]:
        assert digest(out / p) == build["binaries"][p]
        retain(out / p, "binary")
    diagnostic_build = json.loads((diagnostic / "build.json").read_text())
    for p, h in diagnostic_build["binaries"].items():
        assert digest(diagnostic / p) == h
        retain(diagnostic / p, "binary")
    for name in ("baseline", "updated"):
        source = diagnostic / "diagnostic" / name
        for p, h in diagnostic_build["source_inputs"][name].items():
            assert digest(source / p) == h
            retain(source / p, "instrumented-source")
        for p in source.glob("src/*"):
            retain(p, "instrumented-source")
        retain(source / "build/libvane_fs.a", "binary")
    for p, h in diagnostic_build["inputs"].items():
        assert digest(Path(p)) == h
        retain(Path(p), "dependency")
    for p in (out / "updated/venv/lib/python3.12/site-packages/vane_fs").glob("_native*.so"):
        retain(p, "binary")
    for p in (out / "updated/sanitize").glob("*"):
        if p.is_file() and (p.suffix == ".a" or (os.access(p, os.X_OK) and p.suffix == "")):
            retain(p, "sanitized-binary")
    for p in (
        Path(identity["installed_component"]["path"]),
        Path(build["sdk"]["sqlite"]) / "lib/libsqlite3.a",
        Path(build["sdk"]["fuse"]) / "lib/x86_64-linux-gnu/libfuse3.so",
    ):
        retain(p, "dependency")
    write(here / "manifest.json", dict(status="PASS", entries=sorted(paths.values(), key=lambda e: e["path"])))
    assert (here / "manifest.json").stat().st_size < 500 * 1024
    print("Published", len(paths), "verified entries;", mounts, "mounts;", "ADOPT" if summary["pass_all"] else "REJECT")


if __name__ == "__main__":
    main()
