# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain cold-start evidence and verify the default and frozen controls remain unchanged."""

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
    out, here = parser.parse_args().artifacts.resolve(), Path(__file__).resolve().parent
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
                assert row["process_stopped"]
            assert row["path"] not in Path("/proc/mounts").read_text()
            audit["cleanup"].append(row)

    build = read(out / "build.json")
    assert build["status"] == "PASS" and len(build["tests"]) == 36 and build["candidate_core_libraries_identical"]
    for field in ("inputs", "probes"):
        for path, sha in build[field].items():
            verify(path, sha)
    for sdk, variant in build["variants"].items():
        for field in ("sources", "binaries"):
            for path, sha in variant[field].items():
                verify(path, sha)
        assert "154 passed, 1 skipped" in (out / (sdk + "-python.log")).read_text()
        for label in ("native", "asan-tests"):
            assert "100% tests passed, 0 tests failed out of 12" in (out / (sdk + "-" + label + ".log")).read_text()
        assert "New filesystem omitted WAL reservation" in (out / (sdk + "-negative.log")).read_text()
    cleanup(build["cleanup"])
    feasibility = read(out / "feasibility.json")
    assert feasibility["status"] == "PASS" and len(feasibility["cases"]) == 2
    for path, sha in feasibility["inputs"].items():
        verify(path, sha)
    cleanup(feasibility["cleanup"])
    for path in out.iterdir():
        if path.is_file() and not path.name.startswith("pr-"):
            if path.name.endswith("-trace.json"):
                local(path)
            elif path.suffix in (".json", ".log") and path.name != "findings.json":
                copy(path, Path("validation") / path.name)
    for directory in ("preflight", "initial-analysis"):
        root = out / directory
        identity = read(root / "identity.json")
        for path, entry in identity["files"].items():
            verify(path, entry["sha256"])
        for path in root.iterdir():
            if path.is_file():
                if path.name == "probe-fsync-baseline-plain" or path.stat().st_size > 500_000:
                    local(path)
                else:
                    copy(path, Path(directory) / path.name)
    assert read(out / "preflight/identity.json")["measured_comparisons"] == 0
    old = read(out / "initial-analysis/findings.json")
    findings = read(out / "findings.json")
    for field in ("plain", "checks", "native_gates_pass", "tails"):
        assert old[field] == findings[field]
    for a, b in zip(old["diagnostic"], findings["diagnostic"], strict=True):
        assert a["label"] == b["label"] and a["measured_syncs"] == b["measured_syncs"]
    for path, sha in findings["inputs"].items():
        verify(path, sha)
    copy(out / "findings.json", Path("findings.json"))
    windows = []
    for name, count in (("plain", 48), ("diagnostic", 16)):
        source = out / name
        data, plan = read(source / "results.json"), read(source / "plan.json")
        assert data["status"] == "PASS" and len(data["runs"]) == count and data["monitor_stopped"]
        for path, sha in plan["inputs"].items():
            verify(path, sha)
        cleanup(data["cleanup"])
        for path in source.iterdir():
            if path.is_file():
                if path.name.endswith("-trace.json"):
                    local(path)
                else:
                    copy(path, Path(name) / path.name)
        windows.append(dict(start=data["started_epoch"], end=data["ended_epoch"], runs=count))
        audit["windows"][name] = windows[-1]
    assert all(c["end"] <= windows[0]["start"] for c in build["commands"] + feasibility["commands"])
    assert windows[0]["end"] <= windows[1]["start"]
    audit["no_owned_build_or_test_overlap"] = True
    default = read(here.parent / "checkpoint_followthrough/validation/default-unchanged.json")
    for path, sha in default["sources"].items():
        verify(repo / "vane_fs" / path, sha)
    verify(default["module"]["path"], default["module"]["sha256"])
    verify(
        repo / "vane_fs/vcpkg_installed/x64-linux-release/lib/libsqlite3.a",
        digest(Path(build["variants"]["fsync"]["sdk"]) / "lib/libsqlite3.a"),
    )
    audit["default_unchanged"] = True
    assert len({r["path"] for r in audit["cleanup"]}) == len(audit["cleanup"])
    audit["free_bytes"] = shutil.disk_usage(out).free
    for path in here.rglob("*"):
        if path.is_file() and path.name != "audit.json" and "__pycache__" not in path.parts:
            audit["retained"][str(path.relative_to(here))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    audit["status"] = "PASS"
    (here / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("PASS", len(audit["retained"]), "retained files;", len(audit["cleanup"]), "cleaned owned directories")


if __name__ == "__main__":
    main()
