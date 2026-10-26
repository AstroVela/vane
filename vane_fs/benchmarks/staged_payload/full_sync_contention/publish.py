# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain both native windows, every barrier sample and the resolved harness preflights."""

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
    repo = here.parents[3]
    audit = dict(
        status="RUNNING", artifacts=str(out), identities={}, retained={}, local_retained={}, cleanup=[], windows={}
    )

    def verify(path, sha):
        path = Path(path)
        assert digest(path) == sha, path
        audit["identities"][str(path)] = sha

    def copy(path, relative):
        destination = here / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        assert digest(path) == digest(destination)

    def local(path):
        audit["local_retained"][str(path)] = dict(bytes=path.stat().st_size, sha256=digest(path))

    def cleanup(rows):
        for row in rows:
            assert row["removed"] and not Path(row["path"]).exists()
            if "process_stopped" in row:
                assert row["process_stopped"] and row["mount_absent"]
            audit["cleanup"].append(row)

    build = read(out / "build.json")
    assert build["status"] == "PASS" and len(build["tests"]) == 12
    for field in ("inputs", "runtime_source_inputs", "binaries"):
        for path, sha in build[field].items():
            verify(path, sha)
    cleanup(build["cleanup"])
    for path in out.glob("*.json"):
        if path.name in ("plan.json", "results.json", "findings.json") or path.name.startswith("pr-"):
            continue
        copy(path, Path("validation") / path.name)
    for path in out.glob("*.log"):
        copy(path, Path("validation/logs") / path.name)
    for identity in (out / "preflight").rglob("*identity.json"):
        for path, entry in read(identity)["files"].items():
            verify(path, entry["sha256"])
        if identity.name == "identity.json":
            assert read(identity)["measured_comparisons"] == 0
            failure = identity.parent / "build.json"
            if failure.exists():
                cleanup(read(failure)["cleanup"])
    for path in (out / "preflight").rglob("*"):
        if path.is_file():
            if path.name.startswith("probe-") or path.stat().st_size > 500_000:
                local(path)
            else:
                copy(path, Path("preflight") / path.relative_to(out / "preflight"))
    windows = []
    for name, source in (("screen", out), ("confirmation", out / "confirmation")):
        plan, raw, findings = (read(source / filename) for filename in ("plan.json", "results.json", "findings.json"))
        assert raw["status"] == findings["status"] == "PASS" and len(raw["runs"]) == 36
        for report in (plan, findings):
            for path, sha in report["inputs"].items():
                verify(path, sha)
        cleanup(raw["cleanup"])
        for filename in ("plan.json", "results.json", "findings.json"):
            copy(source / filename, Path(name) / filename)
        for path in (source / "runs").iterdir():
            if path.is_file():
                if path.name.endswith(("-trace.json", "-cpu.jsonl")):
                    local(path)
                else:
                    copy(path, Path(name) / "runs" / path.name)
        record = dict(
            runs=36,
            start=raw["started_epoch"],
            end=raw["ended_epoch"],
            full_wal_syncs=4608,
            free_bytes_after=raw["free_bytes_after"],
            syncs_at_least_100ms=len(findings["sync_tails"]),
        )
        windows.append(record)
        audit["windows"][name] = record
    assert windows[0]["end"] <= windows[1]["start"]
    assert all(command["end"] <= windows[0]["start"] for command in build["commands"])
    audit["no_owned_build_or_test_overlap"] = True
    default = read(here.parent / "checkpoint_followthrough/validation/default-unchanged.json")
    for path, sha in default["sources"].items():
        verify(repo / "vane_fs" / path, sha)
    verify(default["module"]["path"], default["module"]["sha256"])
    audit["default_unchanged"] = True
    assert len({row["path"] for row in audit["cleanup"]}) == len(audit["cleanup"])
    audit["free_bytes"] = shutil.disk_usage(out).free
    for path in here.rglob("*"):
        if path.is_file() and path.name != "audit.json" and "__pycache__" not in path.parts:
            audit["retained"][str(path.relative_to(here))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    audit["status"] = "PASS"
    (here / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        "PASS",
        len(audit["retained"]),
        "files; 72 native runs; 9,216 FULL WAL syncs;",
        len(audit["cleanup"]),
        "cleaned paths",
        flush=True,
    )


if __name__ == "__main__":
    main()
