# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Publish lossless index comparison results, source proofs and owned-data cleanup."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, nargs="*", default=[])
    parser.add_argument("--shared-gc-preflight", type=Path, required=True)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    retained, commands = set(), []
    audit = dict(
        status="PASS",
        mounts=0,
        windows=[],
        cleanup_paths=[],
        production_unchanged=True,
        native_changed_files=["src/workspace.cpp"],
        publication_checkpoint_payload_sources_byte_identical=True,
        format_and_row_layout_unchanged=True,
        original_gates_and_timing_unchanged=True,
        slow_samples_retained=True,
        writeback_hints=False,
        payload_budget=False,
    )

    def retain(path):
        path = path.resolve()
        assert path.is_file(), path
        retained.add(path)

    def write(name, value):
        path = here / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, separators=(",", ":")) + "\n")
        assert path.stat().st_size < 500 * 1024, path

    def copy(path, name):
        retain(path)
        write(name, read(path))

    for name in ("build.json", "tests.json", "sanitize.json", "index-tests.json", "binding-tests.json"):
        value = read(out / name)
        assert value["status"] == "PASS"
        copy(out / name, name)
        commands.extend(value.get("commands", value.get("runs", [])))
        for entry in value.get("cleanup", []):
            assert entry["removed"] and not Path(entry["path"]).exists()
            audit["cleanup_paths"].append(entry["path"])
    build = read(out / "build.json")
    sources = {
        "baseline": Path(build["parent"]) / "updated",
        "orphan-control": Path(build["orphan_parent"]) / "updated",
        "updated": out / "updated",
    }
    for name, entries in build["source_inputs"].items():
        for relative, expected in entries.items():
            assert digest(sources[name] / relative) == expected
            retain(sources[name] / relative)
    for relative, expected in build["binaries"].items():
        assert digest(out / relative) == expected
        retain(out / relative)
    for directory in (
        "baseline/build/python-release",
        "orphan-control/build/python-release",
        "updated/build/python-release",
        "updated/sanitize",
    ):
        retain(out / directory / "libvane_fs.a")
        for path in (out / directory).glob("vane_fs_*test"):
            retain(path)
    for binary in ("index-test", "policy-test", "orphan-test"):
        retain(out / "updated/sanitize" / binary)
    retain(out / "updated/venv/lib/python3.12/site-packages/vane_fs/_native.cpython-312-x86_64-linux-gnu.so")
    for p, expected in build["inputs"].items():
        assert digest(Path(p)) == expected
        retain(Path(p))
    with tempfile.TemporaryDirectory(prefix="owned-vane-index-proof-") as work:
        target = Path(work)
        for directory in ("src", "include", "python", "tests"):
            shutil.copytree(
                sources["baseline"] / directory, target / directory, ignore=shutil.ignore_patterns("__pycache__")
            )
        for patch in (here.parent / "orphan_batching/batch.patch", here / "index.patch"):
            subprocess.run(
                ["patch", "--batch", "--fuzz=0", "--no-backup-if-mismatch", "-p1", "-i", str(patch)],
                cwd=target,
                check=True,
                capture_output=True,
            )
        assert all(digest(target / p) == h for p, h in build["source_inputs"]["updated"].items())
    write(
        "reproducer-proof.json", dict(status="PASS", reconstructed_files=len(build["source_inputs"]["updated"]), fuzz=0)
    )
    windows = {}
    for root, role, media in ((out, "plain", ("tmpfs", "ext4")), (out / "causal", "causal", ("ext4",))):
        for medium in media:
            path = root / (medium + "-plain/results.json")
            raw = read(path)
            assert raw["status"] == "PASS" and raw["monitor_stopped"]
            retain(path)
            prefix = "" if role == "plain" else "causal/"
            for key, state in (("fuse", "fresh"), ("aged_fuse", "aged")):
                restored = []
                for variant in ("baseline", "updated"):
                    selected = [m for m in raw[key] if m["variant"] == variant]
                    name = prefix + f"{medium}-{state}-{variant}.json"
                    write(name, selected)
                    restored.extend(read(here / name))
                assert sorted(restored, key=lambda m: (m["rep"], m["variant"])) == sorted(
                    raw[key], key=lambda m: (m["rep"], m["variant"])
                )
                for mount in raw[key]:
                    audit["mounts"] += 1
                    assert mount["inspection"]["block_high_index_present"] == (mount["variant"] == "baseline")
                    if key == "aged_fuse":
                        assert mount["after_gc_inspection"]["block_high_index_present"] == (
                            mount["variant"] == "baseline"
                        )
                    assert mount["inspection"]["quick_check"] == [["ok"]]
                    assert mount["inspection"]["format"] == [[1004, 4096]]
                    assert all(length in (8, 4096) for length, _ in mount["inspection"]["payload_lengths"])
                    assert all(
                        s.get("validation")
                        for s in mount["samples"]
                        if s["workload"] in ("stream_256", "write_64", "random_write_4k", "random_read_4k")
                    )
            for entry in raw["cleanup"]:
                assert entry["removed"] and not Path(entry["path"]).exists()
                assert entry["process_stopped"] and entry["mount_absent"]
                assert not Path(f"/proc/{entry['pid']}").exists()
                audit["cleanup_paths"].append(entry["path"])
            for log in path.parent.glob("*.log"):
                retain(log)
                if log.stat().st_size:
                    shutil.copy2(log, here / (prefix + medium + "-" + log.name))
            retain(path.parent / "host.jsonl")
            record = {k: v for k, v in raw.items() if k not in ("fuse", "aged_fuse")}
            audit["windows"].append(
                dict(kind=role, medium=medium, started_epoch=raw["started_epoch"], ended_epoch=raw["ended_epoch"])
            )
            if role == "plain":
                windows[medium] = record
                copy(root / (medium + "-input-check.json"), medium + "-input-check.json")
            else:
                write("causal/windows.json", {medium: record})
                copy(root / "input-check.json", "causal/input-check.json")
        for name in ("plan.json", "summary.json"):
            copy(root / name, ("" if role == "plain" else "causal/") + name)
        original = read(here / ("summary.json" if role == "plain" else "causal/summary.json"))
        if role == "plain":
            write("windows.json", windows)
        subprocess.run(
            [
                sys.executable,
                str(here.parent / "partial_sync/analyze_writeback.py"),
                str(here if role == "plain" else here / "causal"),
            ],
            check=True,
            capture_output=True,
        )
        assert read(here / ("summary.json" if role == "plain" else "causal/summary.json")) == original
        frozen = read(root / "plan.json")
        for p, expected in frozen["inputs"].items():
            assert digest(Path(p)) == expected
            retain(Path(p))
        for name, key in (("run_fuse.py", "prepared_runner_sha256"), ("workload.py", "prepared_workload_sha256")):
            assert digest(root / name) == frozen[key]
            retain(root / name)
    assert audit["mounts"] == 72
    legacy = args.shared_gc_preflight.resolve()
    legacy_manifest = {}
    for root, label, media in ((legacy, "plain", ("tmpfs", "ext4")), (legacy / "causal", "causal", ("ext4",))):
        for medium in media:
            path = root / (medium + "-plain/results.json")
            raw = read(path)
            assert raw["status"] == "PASS" and raw["monitor_stopped"]
            retain(path)
            prefix = f"shared-gc-preflight/{label}/"
            record = {k: v for k, v in raw.items() if k not in ("fuse", "aged_fuse")}
            legacy_manifest[label + "/" + medium] = record
            for key, state in (("fuse", "fresh"), ("aged_fuse", "aged")):
                restored = []
                for variant in ("baseline", "updated"):
                    selected = [m for m in raw[key] if m["variant"] == variant]
                    name = prefix + f"{medium}-{state}-{variant}.json"
                    write(name, selected)
                    restored.extend(read(here / name))
                assert sorted(restored, key=lambda m: (m["rep"], m["variant"])) == sorted(
                    raw[key], key=lambda m: (m["rep"], m["variant"])
                )
                audit["mounts"] += len(raw[key])
            for entry in raw["cleanup"]:
                assert entry["removed"] and entry["process_stopped"] and entry["mount_absent"]
                assert not Path(entry["path"]).exists() and not Path(f"/proc/{entry['pid']}").exists()
                audit["cleanup_paths"].append(entry["path"])
            retain(path.parent / "host.jsonl")
            for log in path.parent.glob("*.log"):
                retain(log)
                if log.stat().st_size:
                    shutil.copy2(log, here / (prefix + medium + "-" + log.name))
            audit["windows"].append(
                dict(
                    kind="shared_gc_preflight_" + label,
                    medium=medium,
                    started_epoch=raw["started_epoch"],
                    ended_epoch=raw["ended_epoch"],
                )
            )
        for name in ("plan.json", "build.json"):
            copy(root / name, f"shared-gc-preflight/{label}/" + name)
        frozen = read(root / "plan.json")
        for name, key in (("run_fuse.py", "prepared_runner_sha256"), ("workload.py", "prepared_workload_sha256")):
            assert digest(root / name) == frozen[key]
            retain(root / name)
    assert audit["mounts"] == 144
    write("shared-gc-preflight/windows.json", legacy_manifest)
    copy(legacy / "summary.json", "shared-gc-preflight/summary.json")
    mapping = read(legacy / "frozen-recipes/mapping.json")
    copy(legacy / "frozen-recipes/mapping.json", "shared-gc-preflight/recipe-mapping.json")
    for old_path, frozen_path in mapping.items():
        retain(Path(frozen_path))
        destination = here / "shared-gc-preflight/recipes" / Path(frozen_path).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(frozen_path, destination)
    for path in (legacy / "plan.json", legacy / "build.json", legacy / "causal/plan.json"):
        frozen = read(path)
        for old_path, expected in frozen["inputs"].items():
            assert digest(Path(mapping.get(old_path, old_path))) == expected
    write(
        "shared-gc-preflight/scope.json",
        dict(
            status="RETAINED",
            adoption_eligible=False,
            mounts=72,
            reason="GC helper Python aliases loaded the candidate module for the controls and dropped the control index. Corrected module bindings and schema checks require a new comparison; no slow samples removed.",
        ),
    )
    for path in legacy.glob("*.log"):
        retain(path)
        if path.stat().st_size:
            destination = here / "logs" / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    for path in (legacy / "test-logs").rglob("*.log"):
        retain(path)
        destination = here / "test-logs" / path.relative_to(legacy / "test-logs")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    write("windows.json", windows)
    copy(out / "causal/build.json", "causal/build.json")
    copy(out / "index-analysis.json", "index-analysis.json")
    analysis = read(out / "index-analysis.json")
    for p, expected in analysis["inputs"].items():
        assert digest(Path(p)) == expected
        retain(Path(p))
    for root in (out, out / "causal"):
        for path in root.glob("*.log"):
            retain(path)
            if path.stat().st_size:
                target = here / ("logs" if root == out else "causal/logs") / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    for number, root in enumerate([args.probe.resolve(), *[p.resolve() for p in args.preflight]]):
        value = read(root / "report.json")
        if number == 0:
            assert value["status"] == "PASS"
            for p, expected in value["inputs"].items():
                assert digest(Path(p)) == expected
                retain(Path(p))
        for entry in value["cleanup"]:
            assert entry["removed"] and not Path(entry["path"]).exists()
            audit["cleanup_paths"].append(entry["path"])
        prefix = "probe" if number == 0 else f"probe-preflight-{number}"
        copy(root / "report.json", prefix + "/report.json")
        for path in root.glob("*.log"):
            retain(path)
            destination = here / prefix / path.name
            shutil.copy2(path, destination)
    proof = read(out / "probe-preflight-sources/identity.json")
    assert proof["status"] == "PASS"
    copy(out / "probe-preflight-sources/identity.json", "probe-preflight-sources/identity.json")
    for record in proof["records"]:
        value = read(Path(record["report"]))
        expected = value.get("inputs", value.get("identities"))
        source, binary = Path(record["source"]), Path(record["binary"])
        assert digest(source) == record["source_sha256"]
        assert digest(binary) == record["binary_sha256"]
        for old_path, old_hash in expected.items():
            path = (
                source
                if old_path.endswith("metadata-index-probe.cpp")
                else binary
                if old_path.endswith("metadata-index-probe")
                else Path(old_path)
            )
            assert digest(path) == old_hash
            retain(path)
        shutil.copy2(source, here / "probe-preflight-sources" / (source.parent.name + ".cpp.txt"))
    ordered = sorted(audit["windows"], key=lambda w: w["started_epoch"])
    assert all(a["ended_epoch"] <= b["started_epoch"] for a, b in zip(ordered, ordered[1:], strict=False))
    assert all(
        c["ended_epoch"] <= w["started_epoch"] or c["started_epoch"] >= w["ended_epoch"]
        for c in commands
        for w in ordered
    )
    native = read(here.parent / "writeback_budget/source-identity.json")
    for relative, expected in native["production"].items():
        assert digest(repo / "vane_fs" / relative) == expected
    module = Path(native["installed_component"]["path"])
    assert digest(module) == native["installed_component"]["sha256"]
    retain(module)
    assert build["sdk"]["sqlite_sha256"] == native["sqlite_sdk_sha256"]
    retain(Path(build["sdk"]["sqlite"]) / "lib/libsqlite3.a")
    fuse = Path(build["sdk"]["fuse"]) / "lib/x86_64-linux-gnu/libfuse3.so"
    retain(fuse)
    copy(here.parent / "writeback_budget/source-identity.json", "source-identity.json")
    audit["cleanup_paths"] = sorted(set(audit["cleanup_paths"]))
    audit["free_bytes_after"] = shutil.disk_usage(repo).free
    audit["shm_free_bytes_after"] = shutil.disk_usage("/dev/shm").free
    audit["decision"] = "PASS" if read(here / "summary.json")["pass_all"] else "REJECT"
    write("audit.json", audit)
    files = [p for p in here.rglob("*") if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts]
    records = [dict(path=str(p.relative_to(here)), bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(files)]
    records += [dict(local_path=str(p), bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(retained)]
    assert all(digest(Path(r["local_path"]) if "local_path" in r else here / r["path"]) == r["sha256"] for r in records)
    write("manifest.json", dict(status="PASS", entries=records))
    print(
        "publication PASS",
        audit["mounts"],
        "mounts",
        len(audit["cleanup_paths"]),
        "cleanup paths",
        len(records),
        "hashes",
    )


if __name__ == "__main__":
    main()
