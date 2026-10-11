# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain every diagnostic sample and audit source identity and owned-data cleanup."""

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
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--controlled", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    requests, cpu, controlled = (p.resolve() for p in (args.requests, args.cpu, args.controlled))
    local = set()
    audit = dict(
        status="PASS",
        production_algorithm_changed=False,
        writeback_parameters_changed=False,
        original_gate_decision="REJECT; nine failed checks retained",
        diagnostic_timings_decide_adoption=False,
        windows=[],
        mounts=0,
        cleanup_paths=[],
        retained_inputs={},
        source_verification={},
    )

    def add_local(path):
        path = path.resolve()
        assert path.is_file(), path
        local.add(path)

    def write(relative, value):
        path = here / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, separators=(",", ":")) + "\n")
        assert path.stat().st_size < 500 * 1024, path

    def copy(path, relative):
        add_local(path)
        write(relative, read(path))

    cost_summary = read(requests / "request-costs-summary.json")
    assert cost_summary["status"] == "PASS" and len(cost_summary["mounts"]) == 32
    add_local(requests / "request-costs-summary.json")
    for medium in ("tmpfs", "ext4"):
        value = {
            **cost_summary,
            "mounts": [m for m in cost_summary["mounts"] if m["medium"] == medium],
            "groups": [g for g in cost_summary["groups"] if g["medium"] == medium],
        }
        write("request-summary-" + medium + ".json", value)
    assert (
        read(here / "request-summary-tmpfs.json")["mounts"] + read(here / "request-summary-ext4.json")["mounts"]
        == cost_summary["mounts"]
    )
    assert (
        sorted(
            read(here / "request-summary-tmpfs.json")["groups"] + read(here / "request-summary-ext4.json")["groups"],
            key=lambda g: (g["medium"], g["state"], g["variant"], g["workload"]),
        )
        == cost_summary["groups"]
    )
    for directory, label, media in [
        (requests, "requests", ["tmpfs", "ext4"]),
        (cpu, "cpu", ["ext4"]),
        (controlled, "controlled", ["ext4"]),
    ]:
        build = read(directory / "build.json")
        assert build["status"] == "PASS"
        copy(directory / "build.json", label + "/build.json")
        if label == "requests":
            for name, record in build["original_sources"].items():
                for relative, expected in record["sha256"].items():
                    path = Path(record["path"]) / relative
                    assert digest(path) == expected
                    add_local(path)
            for name, entries in build["source_inputs"].items():
                for relative, expected in entries.items():
                    path = directory / "diagnostic" / name / relative
                    assert digest(path) == expected
                    add_local(path)
            for relative, expected in build["binaries"].items():
                path = directory / relative
                assert digest(path) == expected
                add_local(path)
                add_local(path.parent / "libvane_fs.a")
        for path, expected in build["inputs"].items():
            assert digest(Path(path)) == expected
            add_local(Path(path))
        for medium in media:
            check = directory / (medium + "-input-check.json" if label == "requests" else "input-check.json")
            assert read(check)["status"] == "PASS"
            copy(check, label + "/" + check.name)
            plan = directory / (medium + "-plan.json" if label == "requests" else "plan.json")
            copy(plan, label + "/" + plan.name)
            data = directory / (medium + "-diagnostic/results.json")
            raw = read(data)
            assert raw["status"] == "PASS" and raw["monitor_stopped"]
            add_local(data)
            assert all(digest(Path(path)) == expected for path, expected in read(plan)["inputs"].items())
            runner = directory / "run_fuse.py"
            assert digest(runner) == read(plan)["prepared_runner_sha256"]
            add_local(runner)
            workload = directory / "workload.py"
            if "prepared_workload_sha256" in read(plan):
                assert digest(workload) == read(plan)["prepared_workload_sha256"]
            add_local(workload)
            assert raw["config"]["durability"] == "fsync" and raw["config"]["format_version"] == 1004
            assert not raw["config"]["global_drop_caches"]
            window = {k: v for k, v in raw.items() if k not in ("fuse", "aged_fuse")}
            copy_paths = []
            for state, key in [("fresh", "fuse"), ("post_gc", "aged_fuse")]:
                arrays = []
                for variant in raw["config"]["orders"][0]:
                    rows = [m for m in raw[key] if m["variant"] == variant]
                    relative = f"{label}/{medium}-{state}-{variant}.json"
                    write(relative, rows)
                    arrays.extend(read(here / relative))
                    copy_paths.append(relative)
                assert sorted(arrays, key=lambda m: (m["rep"], m["variant"])) == sorted(
                    raw[key], key=lambda m: (m["rep"], m["variant"])
                )
                for mount in raw[key]:
                    audit["mounts"] += 1
                    assert mount["inspection"]["quick_check"] == [["ok"]]
                    assert mount["inspection"]["format"] == [[1004, 4096]]
                    assert all(length in (8, 4096) for length, _ in mount["inspection"]["payload_lengths"])
                    assert all(
                        s.get("validation")
                        for s in mount["samples"]
                        if s["workload"] in ("stream_256", "write_64", "random_write_4k", "random_read_4k")
                    )
                    for name in ("request_costs_file", "cpu_samples_file"):
                        if name in mount:
                            add_local(Path(mount[name]))
            for entry in raw["cleanup"]:
                assert entry["removed"] and not Path(entry["path"]).exists()
                if "pid" in entry:
                    assert entry["process_stopped"] and entry["mount_absent"]
                    assert entry["pid"] is None or not Path(f"/proc/{entry['pid']}").exists()
                audit["cleanup_paths"].append(entry["path"])
            for path in data.parent.glob("*.log"):
                add_local(path)
                if path.stat().st_size:
                    target = here / label / path.name
                    shutil.copy2(path, target)
            add_local(data.parent / "host.jsonl")
            write(label + "/" + medium + "-window.json", window)
            audit["windows"].append(
                dict(
                    kind=label,
                    medium=medium,
                    started_epoch=raw["started_epoch"],
                    ended_epoch=raw["ended_epoch"],
                    mounts=len(raw["fuse"]) + len(raw["aged_fuse"]),
                    arrays=copy_paths,
                )
            )
    assert audit["mounts"] == 64
    commands = read(requests / "build.json")["commands"]
    assert all(c["returncode"] == 0 for c in commands)
    assert all(
        c["ended_epoch"] <= w["started_epoch"] or c["started_epoch"] >= w["ended_epoch"]
        for c in commands
        for w in audit["windows"]
    )
    for directory in (requests, cpu, controlled):
        for path in directory.glob("*.log"):
            add_local(path)
            if path.stat().st_size:
                shutil.copy2(path, here / (directory.name + "-" + path.name))
    for directory, label in [(cpu, "cpu"), (controlled, "controlled")]:
        summary = read(directory / "cpu-summary.json")
        assert summary["status"] == "PASS" and len(summary["mounts"]) == 16
        copy(directory / "cpu-summary.json", label + "-summary.json")
    native = read(here.parent / "writeback_budget/source-identity.json")
    for relative, expected in native["production"].items():
        assert digest(repo / "vane_fs" / relative) == expected
    module = Path(native["installed_component"]["path"])
    assert digest(module) == native["installed_component"]["sha256"]
    add_local(module)
    assert (
        read(requests / "build.json")["inputs"][
            str(repo / "vane_fs/vcpkg_installed/x64-linux-release/lib/libsqlite3.a")
        ]
        == native["sqlite_sdk_sha256"]
    )
    audit["source_verification"] = dict(
        production=native["production"],
        installed_component=native["installed_component"],
        sqlite_sha256=native["sqlite_sdk_sha256"],
    )
    audit["cleanup_paths"] = sorted(set(audit["cleanup_paths"]))
    audit["free_bytes_after"] = shutil.disk_usage(repo).free
    audit["shm_free_bytes_after"] = shutil.disk_usage("/dev/shm").free
    mounts = Path("/proc/mounts").read_text()
    assert all(path + "/mount " not in mounts for path in audit["cleanup_paths"])
    audit["retained_inputs"] = {str(path): digest(path) for path in sorted(local)}
    write("audit.json", audit)
    files = [p for p in here.rglob("*") if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts]
    assert all(p.stat().st_size < 500 * 1024 for p in files if p.suffix == ".json")
    records = [dict(path=str(p.relative_to(here)), bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(files)]
    records += [dict(local_path=str(p), bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(local)]
    for record in records:
        path = Path(record["local_path"]) if "local_path" in record else here / record["path"]
        assert digest(path) == record["sha256"]
    write("manifest.json", dict(status="PASS", entries=records))
    print(
        "retention audit PASS",
        audit["mounts"],
        "mounts",
        len(audit["cleanup_paths"]),
        "cleaned data paths",
        len(records),
        "verified hashes",
        flush=True,
    )


if __name__ == "__main__":
    main()
