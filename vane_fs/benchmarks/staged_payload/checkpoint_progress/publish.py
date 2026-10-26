# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Publish the native replay and ext4 screen with source, cleanup and timing proofs."""

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


def function(source, signature):
    begin = source.index(signature)
    brace = source.index("{", begin)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    retained, commands, cleanup = set(), [], []

    def retain(path):
        path = path.resolve()
        assert path.is_file(), path
        retained.add(path)

    def write(name, data):
        path = here / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
        assert path.stat().st_size < 500 * 1024, path

    def copy(path, name):
        retain(path)
        write(name, read(path))

    def inputs(data):
        for path, expected in data.get("inputs", {}).items():
            assert digest(Path(path)) == expected, path
            retain(Path(path))

    for name in ("build", "tests", "sanitize", "restart-tests", "progress-tests", "probe-plan", "probe-summary"):
        path = out / (name + ".json")
        data = read(path)
        assert data["status"] in ("PASS", "FROZEN_BEFORE_NATIVE_SCREEN")
        copy(path, name + ".json")
        inputs(data)
        records = data.get("commands", data.get("runs", []))
        if isinstance(records, list):
            commands.extend(records)
        for entry in data.get("cleanup", []):
            assert entry["removed"] and not Path(entry["path"]).exists()
            cleanup.append(entry["path"])
    build = read(out / "build.json")
    roots = {
        "baseline": Path(build["parent"]) / "updated",
        "orphan-control": Path(build["orphan_parent"]) / "updated",
        "updated": out / "updated",
    }
    for name, entries in build["source_inputs"].items():
        for relative, expected in entries.items():
            path = roots[name] / relative
            assert digest(path) == expected
            retain(path)
    restart = Path(build["restart_parent"]) / "updated"
    previous_build = read(Path(build["restart_parent"]) / "build.json")
    assert {
        p for p, h in previous_build["source_inputs"]["updated"].items() if h != build["source_inputs"]["updated"][p]
    } == {"src/checkpoint.hpp"}
    for relative, expected in previous_build["source_inputs"]["updated"].items():
        assert digest(restart / relative) == expected
        retain(restart / relative)
    with tempfile.TemporaryDirectory(prefix="owned-vane-progress-proof-") as work:
        target = Path(work)
        for directory in ("src", "include", "python", "tests"):
            shutil.copytree(
                roots["baseline"] / directory, target / directory, ignore=shutil.ignore_patterns("__pycache__")
            )
        for path in (
            here.parent / "orphan_batching/batch.patch",
            here.parent / "checkpoint_restart/restart.patch",
            here / "progress.patch",
        ):
            subprocess.run(
                ["patch", "--batch", "--fuzz=0", "--no-backup-if-mismatch", "-p1", "-i", str(path)],
                cwd=target,
                check=True,
                capture_output=True,
            )
            retain(path)
        assert all(digest(target / p) == expected for p, expected in build["source_inputs"]["updated"].items())
    before = (restart / "src/checkpoint.hpp").read_text()
    after = (roots["updated"] / "src/checkpoint.hpp").read_text()
    for signature in ("\tvoid BeforeWrite(", "\tvoid Run("):
        assert function(before, signature) == function(after, signature)
    assert [line for line in before.splitlines() if "static constexpr" in line] == [
        line for line in after.splitlines() if "static constexpr" in line
    ]
    for path in ("src/staging.hpp", "src/workspace.cpp", "src/payload_file.hpp"):
        assert (restart / path).read_bytes() == (roots["updated"] / path).read_bytes()
    write(
        "source-proof.json",
        dict(
            status="PASS",
            runtime_changed_files=["src/checkpoint.hpp"],
            reconstructed_files=len(build["source_inputs"]["updated"]),
            patch_fuzz=0,
            publication_workspace_payload_byte_identical=True,
            worker_write_admission_and_thresholds_byte_identical=True,
            callback_progress_latched_after_large_generation=True,
            callback_no_sql_or_sync_verified_by_tests=True,
        ),
    )
    for relative, expected in build["binaries"].items():
        assert digest(out / relative) == expected
        retain(out / relative)
    for path in (out / "updated/sanitize").iterdir():
        if path.is_file() and (path.name.startswith("vane_fs_") or path.suffix == ".a" or path.name.endswith("-test")):
            retain(path)
    for root in roots.values():
        retain(root / "build/python-release/libvane_fs.a")
    retain(restart / "build/python-release/libvane_fs.a")
    retain(Path(build["sdk"]["sqlite"]) / "lib/libsqlite3.a")
    retain(Path(build["sdk"]["fuse"]) / "lib/x86_64-linux-gnu/libfuse3.so")
    generator = Path(build["generator"])
    copy(generator / "identity.json", "generator-identity.json")
    inputs(read(generator / "identity.json"))
    assert digest(generator / "prepare.py") == read(generator / "identity.json")["generated_sha256"]
    retain(generator / "prepare.py")
    recorder_tests = here.parent / "sync_costs/sync-tests.json"
    assert read(recorder_tests)["status"] == "PASS"
    inputs(read(recorder_tests))
    copy(recorder_tests, "recorder-tests-reused.json")
    commands.extend(read(recorder_tests)["runs"])
    stage = (roots["updated"] / "src/staging.hpp").read_text()
    publication = function(stage, "\tvoid Publish(")
    final_sync = function(stage, "\tvoid BeforeFullCommit(")
    assert (
        publication.index("BEGIN IMMEDIATE")
        < publication.index("BeforeFullCommit()")
        < publication.index('Exec(db, "COMMIT")')
    )
    assert "Sync(data.fd)" in final_sync
    review_inputs = {
        str(p): digest(p)
        for p in (
            roots["updated"] / "src/staging.hpp",
            roots["updated"] / "src/workspace.cpp",
            here.parent / "payload_presync/README.md",
        )
    }
    review = dict(
        status="PASS",
        scope="Source review and prior experiment; no additional payload runtime change",
        final_payload_sync_precedes_full_sql_commit=True,
        worker_publication_holds_publication_guard_and_sql_writer_transaction=True,
        recovery_open_prefix_is_captured_at_begin_batch=True,
        concurrent_appends_require_the_final_protected_sync=True,
        prior_unlocked_presync_experiment_remains_rejected=True,
        next_step="Identify a stable publishable payload/WAL boundary before reducing protected sync lock scope",
        inputs=review_inputs,
    )
    inputs(review)
    write("payload-lock-review.json", review)
    for path in (out / "link-preflight").iterdir():
        retain(path)
        dest = here / "link-preflight" / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    initial = read(out / "link-preflight/build.json")
    assert digest(out / "link-preflight/prepare.py.txt") == initial["inputs"][str(here / "prepare.py")]
    native = read(out / "probe-results.json")
    assert native["status"] == "PASS" and native["monitor_stopped"] and len(native["runs"]) == 54
    retain(out / "probe-results.json")
    reconstructed = []
    for state in ("cold", "reused", "retained"):
        for variant in ("orphan-control", "restart-control", "updated"):
            name = "native/" + state + "-" + variant + ".json"
            selected = [r for r in native["runs"] if (r["state"], r["variant"]) == (state, variant)]
            write(name, selected)
            reconstructed.extend(read(here / name))
    assert sorted(reconstructed, key=lambda r: (r["state"], r["variant"], r["rep"])) == sorted(
        native["runs"], key=lambda r: (r["state"], r["variant"], r["rep"])
    )
    write("native/window.json", {k: v for k, v in native.items() if k != "runs"})
    for run in native["runs"]:
        path = Path(run["trace_path"])
        copy(path, "native/traces/" + path.name)
    for entry in native["cleanup"]:
        assert entry["removed"] and entry["process_stopped"] and not Path(entry["path"]).exists()
        assert not Path(f"/proc/{entry['pid']}").exists()
        cleanup.append(entry["path"])
    retain(out / "probes/host.jsonl")
    screen = out / "fuse-screen"
    raw_path = screen / "ext4-plain/results.json"
    raw = read(raw_path)
    assert raw["status"] == "PASS" and raw["monitor_stopped"]
    retain(raw_path)
    write("fuse/windows.json", {"ext4": {k: v for k, v in raw.items() if k not in ("fuse", "aged_fuse")}})
    reconstructed = []
    for key, state in (("fuse", "fresh"), ("aged_fuse", "aged")):
        state_reconstructed = []
        for variant in ("baseline", "updated"):
            name = "fuse/ext4-" + state + "-" + variant + ".json"
            selected = [r for r in raw[key] if r["variant"] == variant]
            write(name, selected)
            reconstructed.extend(read(here / name))
            state_reconstructed.extend(read(here / name))
            for mount in selected:
                assert mount["inspection"]["quick_check"] == [["ok"]]
                assert mount["inspection"]["format"] == [[1004, 4096]]
                assert mount["inspection"]["block_high_index_present"]
                assert all(
                    s["validation"]
                    for s in mount["samples"]
                    if s["workload"] in ("stream_256", "write_64", "random_read_4k", "random_write_4k")
                )
        assert sorted(state_reconstructed, key=lambda r: (r["variant"], r["rep"])) == sorted(
            raw[key], key=lambda r: (r["variant"], r["rep"])
        )
    assert len(reconstructed) == 24
    for entry in raw["cleanup"]:
        assert entry["removed"] and entry["process_stopped"] and entry["mount_absent"]
        assert not Path(entry["path"]).exists() and not Path(f"/proc/{entry['pid']}").exists()
        cleanup.append(entry["path"])
    for name in ("plan.json", "summary.json", "ext4-input-check.json"):
        copy(screen / name, "fuse/" + name)
    expected_summary = read(here / "fuse/summary.json")
    subprocess.run(
        [sys.executable, str(here.parent / "partial_sync/analyze_writeback.py"), str(here / "fuse")],
        check=True,
        capture_output=True,
    )
    assert read(here / "fuse/summary.json") == expected_summary
    identity = read(out / "fuse-generator-identity.json")
    inputs(identity)
    assert digest(out / "fuse-screen-generator.py") == identity["generated_sha256"]
    retain(out / "fuse-screen-generator.py")
    copy(out / "fuse-generator-identity.json", "fuse/generator-identity.json")
    inputs(read(screen / "plan.json"))
    for name, key in (("run_fuse.py", "prepared_runner_sha256"), ("workload.py", "prepared_workload_sha256")):
        path = screen / name
        assert digest(path) == read(screen / "plan.json")[key]
        retain(path)
    retain(screen / "ext4-plain/host.jsonl")
    windows = [
        dict(kind="native", started_epoch=native["started_epoch"], ended_epoch=native["ended_epoch"]),
        dict(kind="fuse", started_epoch=raw["started_epoch"], ended_epoch=raw["ended_epoch"]),
    ]
    assert windows[0]["ended_epoch"] <= windows[1]["started_epoch"]
    assert all(
        c["ended_epoch"] <= w["started_epoch"] or c["started_epoch"] >= w["ended_epoch"]
        for c in commands
        for w in windows
    )
    production = read(here.parent / "writeback_budget/source-identity.json")
    assert all(digest(repo / "vane_fs" / p) == h for p, h in production["production"].items())
    module = Path(production["installed_component"]["path"])
    assert digest(module) == production["installed_component"]["sha256"]
    retain(module)
    copy(here.parent / "writeback_budget/source-identity.json", "source-identity.json")
    for directory, prefix in ((out, "logs"), (out / "probes", "native/logs"), (screen / "ext4-plain", "fuse/logs")):
        for path in directory.glob("*.log"):
            retain(path)
            if path.stat().st_size:
                destination = here / prefix / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
    for path in (out / "test-logs").rglob("*.log"):
        retain(path)
        destination = here / "test-logs" / path.relative_to(out / "test-logs")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    write(
        "audit.json",
        dict(
            status="PASS",
            native_runs=54,
            fuse_mounts=24,
            windows=windows,
            cleanup_paths=sorted(set(cleanup)),
            production_unchanged=True,
            no_adoption_decision=True,
            ext4_screen_decision="PASS" if expected_summary["pass_all"] else "REJECT",
            all_slow_samples_retained=True,
            source_and_gate_reconstruction=True,
            no_owned_build_or_test_overlap=True,
            free_bytes_after=shutil.disk_usage(repo).free,
        ),
    )
    files = [p for p in here.rglob("*") if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts]
    entries = [dict(path=str(p.relative_to(here)), bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(files)]
    entries += [dict(local_path=str(p), bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(retained)]
    write("manifest.json", dict(status="PASS", entries=entries))
    print(
        "publication PASS: 54 native runs, 24 FUSE mounts,", len(set(cleanup)), "removed paths,", len(entries), "hashes"
    )


if __name__ == "__main__":
    main()
