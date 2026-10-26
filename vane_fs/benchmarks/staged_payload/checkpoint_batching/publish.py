# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Publish lossless arrays, failed preparations and identity/cleanup audits."""

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
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--failed-preparations", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    out, diagnostic = args.artifacts.resolve(), args.diagnostic.resolve()
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    retained = set()
    audit = dict(
        status="PASS",
        mounts=0,
        windows=[],
        cleanup_paths=[],
        production_unchanged=True,
        writeback_budget_unchanged=True,
        native_changed_files=["src/checkpoint.hpp"],
        policy_tests_unchanged=True,
        measured_slow_samples_retained=True,
        initial_failures_retained=[],
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

    commands = []
    for filename in ("build.json", "tests.json", "sanitize.json", "batch-tests.json"):
        value = read(out / filename)
        assert value["status"] == "PASS"
        copy(out / filename, filename)
        commands.extend(value.get("commands", value.get("runs", [])))
        for entry in value.get("cleanup", []):
            assert entry["removed"] and not Path(entry["path"]).exists()
            audit["cleanup_paths"].append(entry["path"])
    build = read(out / "build.json")
    sources = {
        "baseline": Path(build["parent"]) / "updated",
        "budget-control": Path(build["budget_parent"]) / "updated",
        "rejected-control": Path(build["rejected_parent"]) / "updated",
        "updated": out / "updated",
    }
    for name, entries in build["source_inputs"].items():
        for relative, expected in entries.items():
            path = sources[name] / relative
            assert digest(path) == expected
            retain(path)
    assert {
        p for p, h in build["source_inputs"]["budget-control"].items() if h != build["source_inputs"]["updated"][p]
    } == {"src/checkpoint.hpp"}
    for relative, expected in build["binaries"].items():
        path = out / relative
        assert digest(path) == expected
        retain(path)
    for directory in (
        "baseline/build/python-release",
        "budget-control/build/python-release",
        "rejected-control/build/python-release",
        "updated/build/python-release",
        "updated/sanitize",
    ):
        retain(out / directory / "libvane_fs.a")
        for binary in (out / directory).glob("vane_fs_*test"):
            retain(binary)
    for binary in ("batch-test", "writeback-test"):
        retain(out / "updated/sanitize" / binary)
    retain(out / "updated/venv/lib/python3.12/site-packages/vane_fs/_native.cpython-312-x86_64-linux-gnu.so")
    for path, expected in build["inputs"].items():
        assert digest(Path(path)) == expected
        retain(Path(path))
    # Reconstruct both the inherited budget and this change from the exact
    # adoption control. The original component tests are byte-identical.
    with tempfile.TemporaryDirectory(prefix="owned-vane-batch-proof-") as work:
        target = Path(work)
        for directory in ("src", "include", "python", "tests"):
            shutil.copytree(
                sources["baseline"] / directory, target / directory, ignore=shutil.ignore_patterns("__pycache__")
            )
        for patch in (here.parent / "writeback_budget/budget.patch", here / "batch.patch"):
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
    for directory, label, media in (
        (out, "plain", ("tmpfs", "ext4")),
        (out / "causal", "causal", ("ext4",)),
        (diagnostic, "diagnostic", ("ext4",)),
    ):
        for medium in media:
            path = directory / (medium + "-" + ("plain" if label == "causal" else label) + "/results.json")
            raw = read(path)
            assert raw["status"] == "PASS" and raw["monitor_stopped"]
            retain(path)
            root = here if label == "plain" else here / label
            root.mkdir(exist_ok=True)
            record = {k: v for k, v in raw.items() if k not in ("fuse", "aged_fuse")}
            for key, state in (("fuse", "fresh"), ("aged_fuse", "aged")):
                reconstructed = []
                for variant in ("baseline", "updated"):
                    selected = [m for m in raw[key] if m["variant"] == variant]
                    name = f"{medium}-{state}-{variant}.json"
                    write(("" if label == "plain" else label + "/") + name, selected)
                    reconstructed.extend(read(root / name))
                assert sorted(reconstructed, key=lambda m: (m["rep"], m["variant"])) == sorted(
                    raw[key], key=lambda m: (m["rep"], m["variant"])
                )
                for mount in raw[key]:
                    audit["mounts"] += 1
                    assert mount["inspection"]["quick_check"] == [["ok"]] and mount["inspection"]["format"] == [
                        [1004, 4096]
                    ]
                    assert all(length in (8, 4096) for length, _ in mount["inspection"]["payload_lengths"])
                    assert all(
                        s.get("validation")
                        for s in mount["samples"]
                        if s["workload"] in ("stream_256", "write_64", "random_write_4k", "random_read_4k")
                    )
                    for name in ("request_costs_file", "cpu_samples_file"):
                        if name in mount:
                            retain(Path(mount[name]))
            for entry in raw["cleanup"]:
                assert entry["removed"] and not Path(entry["path"]).exists()
                assert entry["process_stopped"] and entry["mount_absent"]
                assert not Path(f"/proc/{entry['pid']}").exists()
                audit["cleanup_paths"].append(entry["path"])
            for log in path.parent.glob("*.log"):
                retain(log)
                if log.stat().st_size:
                    destination = root / (medium + "-" + log.name)
                    shutil.copy2(log, destination)
                    assert digest(destination) == digest(log)
            retain(path.parent / "host.jsonl")
            audit["windows"].append(
                dict(kind=label, medium=medium, started_epoch=raw["started_epoch"], ended_epoch=raw["ended_epoch"])
            )
            if label == "plain":
                windows[medium] = record
                copy(directory / (medium + "-input-check.json"), medium + "-input-check.json")
            elif label == "causal":
                write("causal/windows.json", {medium: record})
                copy(directory / "input-check.json", "causal/input-check.json")
            else:
                write("diagnostic/window.json", record)
    assert audit["mounts"] == 88
    write("windows.json", windows)
    for name in ("plan.json", "summary.json"):
        copy(out / name, name)
    # The analyzer consumes the split arrays when no original results path is
    # present. Preserve its existing output while checking identical decisions.
    original_summary = read(here / "summary.json")
    subprocess.run(
        [sys.executable, str(here.parent / "partial_sync/analyze_writeback.py"), str(here)],
        check=True,
        capture_output=True,
    )
    assert read(here / "summary.json") == original_summary
    write("summary.json", original_summary)
    for name in ("build.json", "plan.json", "summary.json"):
        copy(out / "causal" / name, "causal/" + name)
    causal_summary = read(here / "causal/summary.json")
    subprocess.run(
        [sys.executable, str(here.parent / "partial_sync/analyze_writeback.py"), str(here / "causal")],
        check=True,
        capture_output=True,
    )
    assert read(here / "causal/summary.json") == causal_summary
    write("causal/summary.json", causal_summary)
    for name in (
        "build.json",
        "plan.json",
        "input-check.json",
        "request-costs-summary.json",
        "cpu-summary.json",
    ):
        value = read(diagnostic / name)
        assert value.get("status", "PASS") == "PASS" or value["status"].startswith("FROZEN")
        copy(diagnostic / name, "diagnostic/" + name)
        for p, expected in value.get("inputs", {}).items():
            assert digest(Path(p)) == expected
            retain(Path(p))
    maintenance = read(diagnostic / "maintenance-analysis.json")
    assert maintenance["status"] == "PASS"
    retain(diagnostic / "maintenance-analysis.json")
    for p, expected in maintenance["inputs"].items():
        assert digest(Path(p)) == expected
        retain(Path(p))
    write(
        "diagnostic/slow-spans.json",
        dict(
            status="PASS",
            minimum_duration_ns=100000000,
            stages=list(maintenance["phases"][0]["foreground"]["stage_ns"]),
            note="Whole spans overlapping a phase, including earlier foreground callbacks; inclusive times are not additive",
            phases=maintenance["slow_overlapping_spans"],
        ),
    )
    phase_files, reconstructed = [], []
    for state in ("fresh", "post_gc"):
        for variant in ("baseline", "updated"):
            name = f"maintenance-{state}-{variant}.json"
            selected = [p for p in maintenance["phases"] if (p["state"], p["variant"]) == (state, variant)]
            write("diagnostic/" + name, selected)
            phase_files.append(name)
            reconstructed.extend(read(here / "diagnostic" / name))

    def order(phase):
        return (phase["state"], phase["variant"], phase["rep"], phase["workload"])

    assert sorted(reconstructed, key=order) == sorted(maintenance["phases"], key=order)
    write(
        "diagnostic/maintenance-analysis.json",
        {
            **{k: v for k, v in maintenance.items() if k not in ("phases", "slow_overlapping_spans")},
            "phase_files": phase_files,
            "slow_spans_file": "slow-spans.json",
        },
    )
    diagnostic_build = read(diagnostic / "build.json")
    commands.extend(diagnostic_build["commands"])
    for source in diagnostic_build["original_sources"].values():
        for relative, expected in source["sha256"].items():
            path = Path(source["path"]) / relative
            assert digest(path) == expected
            retain(path)
    for name, entries in diagnostic_build["source_inputs"].items():
        for relative, expected in entries.items():
            path = diagnostic / "diagnostic" / name / relative
            assert digest(path) == expected
            retain(path)
    for relative, expected in diagnostic_build["binaries"].items():
        assert digest(diagnostic / relative) == expected
        retain(diagnostic / relative)
        retain((diagnostic / relative).parent / "libvane_fs.a")
    for p, expected in diagnostic_build["inputs"].items():
        assert digest(Path(p)) == expected
        retain(Path(p))
    generator = out / "diagnostic-generator"
    identity = read(generator / "identity.json")
    assert identity["status"] == "PASS"
    for p, expected in identity["inputs"].items():
        assert digest(Path(p)) == expected
        retain(Path(p))
    copy(generator / "identity.json", "diagnostic/generator-identity.json")
    for path in generator.iterdir():
        if path.is_file():
            retain(path)
    for directory in (out, out / "causal", diagnostic):
        frozen = read(directory / "plan.json")
        for p, expected in frozen["inputs"].items():
            assert digest(Path(p)) == expected
            retain(Path(p))
        for name, key in (("run_fuse.py", "prepared_runner_sha256"), ("workload.py", "prepared_workload_sha256")):
            assert digest(directory / name) == frozen[key]
            retain(directory / name)
        for log in directory.glob("*.log"):
            retain(log)
            if log.stat().st_size:
                target = here / (
                    ("diagnostic/" if directory == diagnostic else "causal/" if directory.name == "causal" else "logs/")
                    + log.name
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(log, target)
    # Keep the actual initial failures and the code that produced them.
    for index, directory in enumerate(args.failed_preparations):
        directory = directory.resolve()
        value = read(directory / "build.json")
        reports = {}
        for path in directory.glob("*.json"):
            report = read(path)
            reports[path.name] = report.get("status")
            commands.extend(report.get("commands", report.get("runs", [])))
            for entry in report.get("cleanup", []):
                assert entry["removed"] and not Path(entry["path"]).exists()
                audit["cleanup_paths"].append(entry["path"])
            copy(path, f"initial-{index}/" + path.name)
        changed = sorted(
            p
            for p, expected in value["source_inputs"]["updated"].items()
            if expected != build["source_inputs"]["updated"][p]
        )
        audit["initial_failures_retained"].append(
            dict(path=str(directory), reports=reports, source_differences_from_final=changed)
        )
        for pattern in ("*.log", "initial-*", "initial-validation/*"):
            for path in directory.glob(pattern):
                if path.is_file():
                    retain(path)
                    # Historical code is a byte-preserved artifact. Formatting
                    # hooks should only process the current reproducer sources.
                    suffix = ".txt" if path.suffix in (".py", ".cpp", ".hpp") else ""
                    destination = here / f"initial-{index}" / (path.name + suffix)
                    destination.parent.mkdir(exist_ok=True, parents=True)
                    shutil.copy2(path, destination)
        for name, entries in value.get("source_inputs", {}).items():
            source = (
                directory / "updated"
                if name == "updated"
                else Path(value["parent" if name == "baseline" else "budget_parent"]) / "updated"
            )
            for relative, expected in entries.items():
                path = source / relative
                assert digest(path) == expected
                retain(path)
        for relative, expected in value.get("binaries", {}).items():
            path = directory / relative
            assert digest(path) == expected
            retain(path)
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
    audit["production_identity"] = native["production"]
    audit["cleanup_paths"] = sorted(set(audit["cleanup_paths"]))
    audit["free_bytes_after"] = shutil.disk_usage(repo).free
    audit["shm_free_bytes_after"] = shutil.disk_usage("/dev/shm").free
    audit["decision"] = "PASS" if original_summary["pass_all"] else "REJECT"
    write("audit.json", audit)
    files = [p for p in here.rglob("*") if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts]
    assert all(p.stat().st_size < 500 * 1024 for p in files if p.suffix == ".json")
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
