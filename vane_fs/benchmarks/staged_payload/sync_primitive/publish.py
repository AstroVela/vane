# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain the complete sync comparison, corrected loss fixture and cleanup evidence."""

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
    audit = dict(status="RUNNING", artifacts=str(out), identities={}, retained={}, local_retained={}, cleanup=[])

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
    assert build["core_static_libraries_identical"] and build["worker_active_in_both_variants"]
    for field in ("inputs", "probes"):
        for path, sha in build[field].items():
            verify(path, sha)
    for component in build["components"].values():
        for field in ("sources", "binaries"):
            for path, sha in component[field].items():
                verify(path, sha)
    for variant in ("fsync", "fdatasync"):
        assert "154 passed, 1 skipped" in (out / (variant + "-python.log")).read_text()
        assert "100% tests passed, 0 tests failed out of 12" in (out / (variant + "-native.log")).read_text()
        assert "100% tests passed, 0 tests failed out of 12" in (out / (variant + "-asan-tests.log")).read_text()
        assert "SQLite WAL used unexpected syscall" in (out / (variant + "-wrong-primitive.log")).read_text()
    cleanup(build["cleanup"])
    for path, sha in read(out / "report.json")["inputs"].items():
        verify(path, sha)
    for path in out.iterdir():
        if path.is_file() and not path.name.startswith("pr-"):
            if path.name.endswith("-trace.json") or path.name.startswith("probe-"):
                local(path)
            elif path.suffix in (".json", ".log") and path.name not in ("plan.json", "results.json", "findings.json"):
                copy(path, Path("validation") / path.name)
    for directory in ("preflight", "preflight-loss-model"):
        root = out / directory
        identity = read(root / "identity.json")
        assert identity["measured_comparisons"] == 0
        for path, entry in identity["files"].items():
            verify(path, entry["sha256"])
        original = read(root / "build.json")
        assert digest(root / "build.py.txt") == original["inputs"][str(here / "build.py")]
        for path in root.iterdir():
            if path.is_file():
                if path.name.endswith("-trace.json") or path.name in ("probe-fsync-release", "loss-probe"):
                    local(path)
                else:
                    copy(path, Path(directory) / path.name)
    native = read(out / "results.json")
    findings = read(out / "findings.json")
    assert native["status"] == findings["status"] == "PASS" and len(native["runs"]) == 36
    assert native["monitor_stopped"]
    for report in (read(out / "plan.json"), findings):
        for path, sha in report["inputs"].items():
            verify(path, sha)
    cleanup(native["cleanup"])
    for filename in ("plan.json", "results.json", "findings.json", "host.jsonl"):
        copy(out / filename, Path("native") / filename)
    for path in (out / "runs").iterdir():
        if path.is_file():
            if path.name.endswith(("-trace.json", "-cpu.jsonl")):
                local(path)
            else:
                copy(path, Path("native/runs") / path.name)
    plain = out / "plain"
    data = read(plain / "ext4-plain/results.json")
    summary = read(plain / "summary.json")
    assert data["status"] == summary["status"] == "PASS" and data["monitor_stopped"]
    assert len(data["fuse"]) == len(data["aged_fuse"]) == 12
    for path, sha in read(plain / "plan.json")["inputs"].items():
        verify(path, sha)
    cleanup(data["cleanup"])
    for filename in ("plan.json", "build.json", "tests.json", "input-check.json", "summary.json"):
        copy(plain / filename, Path("plain") / filename)
    for filename in ("results.json", "host.jsonl"):
        copy(plain / "ext4-plain" / filename, Path("plain") / filename)
    for filename in ("run_fuse.py", "workload.py"):
        copy(plain / filename, Path("executed") / (filename + ".txt"))
    assert all(row["end"] <= native["started_epoch"] for row in build["commands"])
    assert native["ended_epoch"] <= data["started_epoch"]
    audit["no_owned_build_or_test_overlap"] = True
    audit["windows"] = dict(
        native_runs=36,
        full_wal_barriers=4608,
        plain_fuse_mounts=24,
        native_start=native["started_epoch"],
        native_end=native["ended_epoch"],
        plain_start=data["started_epoch"],
        plain_end=data["ended_epoch"],
    )
    default = read(here.parent / "checkpoint_followthrough/validation/default-unchanged.json")
    for path, sha in default["sources"].items():
        verify(repo / "vane_fs" / path, sha)
    verify(default["module"]["path"], default["module"]["sha256"])
    verify(
        repo / "vane_fs/vcpkg_installed/x64-linux-release/lib/libsqlite3.a",
        digest(Path(build["components"]["fsync"]["sdk"]) / "lib/libsqlite3.a"),
    )
    audit["default_unchanged"] = True
    audit["default_sdk_unchanged"] = True
    assert len({row["path"] for row in audit["cleanup"]}) == len(audit["cleanup"])
    audit["free_bytes"] = shutil.disk_usage(out).free
    for path in here.rglob("*"):
        if path.is_file() and path.name != "audit.json" and "__pycache__" not in path.parts:
            audit["retained"][str(path.relative_to(here))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    audit["status"] = "PASS"
    (here / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("PASS", len(audit["retained"]), "retained files;", len(audit["cleanup"]), "cleaned owned paths", flush=True)


if __name__ == "__main__":
    main()
