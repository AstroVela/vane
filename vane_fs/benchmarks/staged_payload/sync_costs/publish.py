# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Publish lossless sync evidence, raw-file probes, identities and cleanup audit."""

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
    retained = set()
    audit = dict(
        status="PASS",
        production_unchanged=True,
        source_logic_unchanged=True,
        adoption_decision="NONE: diagnostic and raw-file hypotheses only",
        mounts=0,
        native_probe_runs=0,
        sync_calls=0,
        cleanup_paths=[],
        windows=[],
        all_sync_records_published_losslessly=True,
        raw_request_traces_retained_locally=True,
        slow_samples_retained=True,
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
        return path

    def copy(path, name):
        retain(path)
        return write(name, read(path))

    def split(name, rows, count=400):
        files = []
        for index in range(0, len(rows), count):
            relative = f"{name}-{index // count:03d}.json"
            write(relative, rows[index : index + count])
            files.append(relative)
        assert [r for f in files for r in read(here / f)] == rows
        return files

    for name in (
        "build.json",
        "generator-identity.json",
        "sync-tests.json",
        "source-proof.json",
        "plan.json",
        "input-check.json",
        "sync-analysis.json",
        "cpu-summary.json",
        "request-costs-summary.json",
        "release-test-trace.json",
        "sanitize-test-trace.json",
    ):
        value = read(out / name)
        if name not in ("plan.json", "release-test-trace.json", "sanitize-test-trace.json"):
            assert value["status"] == "PASS"
        copy(out / name, name)
    for name in ("sync-phases", "slow-syncs"):
        retain(out / (name + ".json"))
        files = split("analysis/" + name, read(out / (name + ".json")), count=12 if name == "sync-phases" else 32)
        write(name + ".json", dict(status="PASS", files=files, order_preserved=True))
    maintenance = read(out / "maintenance-analysis.json")
    retain(out / "maintenance-analysis.json")
    phases = split("analysis/maintenance-phases", maintenance["phases"], count=12)
    slow = split("analysis/maintenance-slow-spans", maintenance["slow_overlapping_spans"], count=12)
    write(
        "maintenance-analysis.json",
        {
            **{k: v for k, v in maintenance.items() if k not in ("phases", "slow_overlapping_spans")},
            "phase_files": phases,
            "slow_span_files": slow,
        },
    )

    build = read(out / "build.json")
    for record in build["original_sources"].values():
        for relative, expected in record["sha256"].items():
            path = Path(record["path"]) / relative
            assert digest(path) == expected
            retain(path)
    for name, entries in build["source_inputs"].items():
        for relative, expected in entries.items():
            path = out / "diagnostic" / name / relative
            assert digest(path) == expected
            retain(path)
    for relative, expected in build["binaries"].items():
        assert digest(out / relative) == expected
        retain(out / relative)
        retain((out / relative).parent / "libvane_fs.a")
    identities = [build, read(out / "generator-identity.json"), read(out / "sync-tests.json"), read(out / "plan.json")]
    for value in identities:
        for path, expected in value["inputs"].items():
            assert digest(Path(path)) == expected
            retain(Path(path))
    for path in (out / "generator").iterdir():
        retain(path)
        if path.suffix in (".cpp", ".hpp", ".py"):
            destination = here / "generator" / (path.name + ".txt")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    for name, key in (("run_fuse.py", "prepared_runner_sha256"), ("workload.py", "prepared_workload_sha256")):
        assert digest(out / name) == read(out / "plan.json")[key]
        retain(out / name)
    for entry in read(out / "sync-tests.json")["cleanup"]:
        assert entry["removed"] and not Path(entry["path"]).exists()
        audit["cleanup_paths"].append(entry["path"])

    raw = read(out / "ext4-diagnostic/results.json")
    assert raw["status"] == "PASS" and raw["monitor_stopped"]
    retain(out / "ext4-diagnostic/results.json")
    audit["windows"].append(
        dict(kind="fuse_diagnostic", started_epoch=raw["started_epoch"], ended_epoch=raw["ended_epoch"])
    )
    for state, key in (("fresh", "fuse"), ("post_gc", "aged_fuse")):
        for variant in ("baseline", "updated"):
            write(f"mounts/{state}-{variant}.json", [m for m in raw[key] if m["variant"] == variant])
        for mount in raw[key]:
            audit["mounts"] += 1
            path = Path(mount["request_costs_file"])
            retain(path)
            retain(Path(mount["cpu_samples_file"]))
            trace = read(path)
            prefix = f"traces/{state}-{mount['rep']}-{mount['variant']}"
            index = {k: v for k, v in trace.items() if k != "threads"}
            index.update(status="PASS", source_path=str(path), source_sha256=digest(path), threads=[])
            for thread in trace["threads"]:
                tid = thread["tid"]
                selected = sorted({c[0] for c in thread["syncs"] if c[0] >= 0})
                parents = [dict(index=i, row=thread["rows"][i]) for i in selected]
                row_files = split(prefix + f"-{tid}-parent-rows", parents)
                sync_files = split(prefix + f"-{tid}-syncs", thread["syncs"])
                audit["sync_calls"] += len(thread["syncs"])
                index["threads"].append(
                    dict(
                        tid=tid,
                        parent_row_files=row_files,
                        sync_files=sync_files,
                        original_request_indices_preserved=True,
                    )
                )
                assert [c for f in sync_files for c in read(here / f)] == thread["syncs"]
                assert [r for f in row_files for r in read(here / f)] == parents
            index["non_sync_request_rows"] = "Retained in the original local trace; source path and hash above"
            write(prefix + "-index.json", index)
    write("window.json", {k: v for k, v in raw.items() if k not in ("fuse", "aged_fuse")})
    for entry in raw["cleanup"]:
        assert entry["removed"] and entry["process_stopped"] and entry["mount_absent"]
        assert not Path(entry["path"]).exists() and not Path(f"/proc/{entry['pid']}").exists()
        audit["cleanup_paths"].append(entry["path"])
    assert audit["mounts"] == 16 and audit["sync_calls"] == read(out / "sync-analysis.json")["total_calls"]

    batches = out / "raw-batches"
    plan, raw_batches = read(batches / "plan.json"), read(batches / "results.json")
    assert raw_batches["status"] == "PASS" and raw_batches["monitor_stopped"]
    assert len(raw_batches["samples"]) == 36
    copy(batches / "plan.json", "raw-batches/plan.json")
    retain(batches / "results.json")
    for state in plan["states"]:
        for variant in plan["variants"]:
            selected = [s for s in raw_batches["samples"] if (s["state"], s["variant"]) == (state, variant)]
            assert len(selected) == 6 and all(s["validation"]["retained_guard_valid"] for s in selected)
            write(f"raw-batches/{state}-{variant}.json", selected)
    write("raw-batches/summary.json", {k: v for k, v in raw_batches.items() if k != "samples"})
    audit["native_probe_runs"] = len(raw_batches["samples"])
    for entry in raw_batches["cleanup"]:
        assert entry["removed"] and entry["process_stopped"] and not Path(entry["path"]).exists()
        audit["cleanup_paths"].append(entry["path"])
    for path, expected in plan["inputs"].items():
        assert digest(Path(path)) == expected
        retain(Path(path))
    assert plan["frozen_epoch"] <= min(s["begin_epoch_ns"] for s in raw_batches["samples"]) / 1e9
    assert raw["ended_epoch"] < raw_batches["started_epoch"]
    audit["windows"].extend(
        dict(
            kind="raw_file_probe",
            started_epoch=s["begin_epoch_ns"] / 1e9,
            ended_epoch=s["end_epoch_ns"] / 1e9,
            state=s["state"],
            variant=s["variant"],
            rep=s["rep"],
        )
        for s in raw_batches["samples"]
    )
    for directory in (out, out / "ext4-diagnostic", batches, out / "raw-batches-preflight"):
        for path in directory.iterdir():
            if path.is_file():
                retain(path)
    copy(out / "raw-batches-preflight/preflight.json", "raw-batches/preflight.json")
    for window in audit["windows"]:
        for command in build["commands"] + read(out / "sync-tests.json")["runs"]:
            assert (
                command["ended_epoch"] <= window["started_epoch"] or command["started_epoch"] >= window["ended_epoch"]
            )

    production = read(here.parent / "writeback_budget/source-identity.json")
    for relative, expected in production["production"].items():
        path = repo / "vane_fs" / relative
        assert digest(path) == expected
        retain(path)
    module = production["installed_component"]
    assert digest(Path(module["path"])) == module["sha256"]
    retain(Path(module["path"]))
    assert not (repo / "vane_fs/src/staging.hpp").exists()
    copy(here.parent / "payload_presync/summary.json", "reference-adoption.json")
    assert not read(here / "reference-adoption.json")["pass_all"]
    write(
        "production-identity.json",
        {k: production[k] for k in ("production", "installed_component", "sqlite_sdk_sha256")},
    )
    write("audit.json", audit)
    files = [dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(retained)]
    write(
        "retention.json", dict(status="PASS", files=files, raw_measurements_retained=True, temporary_data_removed=True)
    )
    published = [
        dict(path=str(p.relative_to(here)), bytes=p.stat().st_size, sha256=digest(p))
        for p in sorted(here.rglob("*"))
        if p.is_file() and p.name not in ("manifest.json", "README.md") and "__pycache__" not in p.parts
    ]
    assert all(p["bytes"] < 500 * 1024 for p in published if p["path"].endswith(".json"))
    write("manifest.json", dict(status="PASS", files=published))
    print(
        "sync evidence publication PASS",
        len(published),
        "files",
        len(files),
        "retained identities",
        len(audit["cleanup_paths"]),
        "removed paths",
        flush=True,
    )


if __name__ == "__main__":
    main()
