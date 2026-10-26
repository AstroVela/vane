# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain all samples, prove archive identities after restoration, and split raw FUSE reports."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    lookup = here.parents[1] / "lookup_reads"
    archive = json.loads((out / "archive-map.json").read_text())
    audit = dict(status="RUNNING", local_artifacts=str(out), windows={}, local_retained={}, identities={}, cleanup=[])

    def verify(path, expected):
        source = Path(path)
        if not source.exists() or digest(source) != expected:
            record = archive[str(source)]
            assert record["sha256"] == expected
            source = Path(record["path"])
        assert digest(source) == expected, str(source)
        audit["identities"][path] = dict(verified_path=str(source), sha256=expected)

    def copy(source, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        assert digest(source) == digest(dest)

    for name in (
        "candidates.json",
        "tests.json",
        "patch-proof.json",
        "archive-map.json",
        "production-restored.json",
        "production-test-cleanup.json",
        "restore-python-cleanup.json",
        "installs.json",
    ):
        copy(out / name, here / "validation" / name)
    restored_default = json.loads((out / "production-restored.json").read_text())
    assert restored_default["status"] == "PASS"
    repo = here.parents[3]
    for path, expected in restored_default["sources"].items():
        assert digest(repo / "vane_fs" / path) == expected
    assert digest(Path(restored_default["module"]["path"])) == restored_default["module"]["sha256"]
    audit["restored_default_verified"] = True
    for name in ("production-test-cleanup.json", "restore-python-cleanup.json"):
        row = json.loads((out / name).read_text())
        assert row["removed"] and not Path(row["path"]).exists()
        audit["cleanup"].append(row)
    build = json.loads((out / "candidates.json").read_text())
    assert build["status"] == "PASS"
    for path, expected in build["inputs"].items():
        verify(path, expected)
    for variant in build["variants"].values():
        for group in ("sources", "binaries"):
            for path, expected in variant[group].items():
                verify(str(Path(variant["path"]) / path), expected)
    for name in ("prepare.py", "lookup.patch"):
        copy(out / "executed" / name, here / "executed" / (name + ".txt" if name.endswith(".py") else name))
    for path in out.glob("*.log"):
        copy(path, here / "logs" / path.name)
    for path in (out / "test-logs").rglob("*.log"):
        dest = here / "test-logs" / path.relative_to(out / "test-logs")
        if path.stat().st_size > 500 * 1024:
            audit["local_retained"][str(path)] = dict(bytes=path.stat().st_size, sha256=digest(path))
            if dest.exists():
                dest.unlink()
        else:
            copy(path, dest)
    native = json.loads((out / "native/summary.json").read_text())
    assert native["status"] == "PASS" and len(native["samples"]) == 48
    for path in (out / "native").iterdir():
        if path.suffix in (".json", ".log"):
            copy(path, lookup / "native" / path.name)
    for path, expected in json.loads((out / "native/plan.json").read_text())["identities"].items():
        verify(path, expected)
    audit["native_runs"] = 48
    for row in native["cleanup"]:
        assert row["removed"] and not Path(row["path"]).exists()
        audit["cleanup"].append(row)
    for name in ("pilot.json", "version-pilot.json"):
        path = out / name
        audit["local_retained"][str(path)] = dict(
            bytes=path.stat().st_size, sha256=digest(path), role="Pre-freeze exploration, not adoption evidence"
        )
    previous_end = native["ended_epoch"]
    tails = []
    for name in ("lookup-ablation", "clean-ablation", "original-control", "production-fuse"):
        source = out / name
        target = lookup / "production" if name == "production-fuse" else here / name
        plan = json.loads((source / "plan.json").read_text())
        for path, expected in plan["inputs"].items():
            verify(path, expected)
        for filename in ("plan.json", "summary.json", "decision.json", "input-check.json"):
            if (source / filename).exists():
                copy(source / filename, target / filename)
        path = source / "ext4-plain/results.json"
        raw = json.loads(path.read_text())
        assert raw["status"] == "PASS" and raw["monitor_stopped"]
        assert raw["started_epoch"] >= previous_end
        previous_end = raw["ended_epoch"]
        metadata = {k: v for k, v in raw.items() if k not in ("fuse", "aged_fuse")}
        write(target / "raw/metadata.json", metadata)
        index = {}
        restored = dict(metadata)
        for state in ("fuse", "aged_fuse"):
            assert len(raw[state]) == 12
            index[state] = []
            restored[state] = []
            for row in raw[state]:
                filename = f"{state}-{row['rep']}-{row['variant']}.json"
                write(target / "raw" / filename, row)
                index[state].append(filename)
                restored[state].append(json.loads((target / "raw" / filename).read_text()))
                for sample in row["samples"]:
                    for key in (
                        "write_latencies",
                        "operation_latencies",
                        "fsync_seconds",
                        "final_directory_fsync_seconds",
                        "close_seconds",
                    ):
                        values = sample.get(key, [])
                        if isinstance(values, (float, int)):
                            values = [values]
                        for i, seconds in enumerate(values):
                            if seconds >= 1:
                                tails.append(
                                    dict(
                                        window=name,
                                        state=state,
                                        variant=row["variant"],
                                        rep=row["rep"],
                                        workload=sample["workload"],
                                        field=key,
                                        index=i,
                                        seconds=seconds,
                                    )
                                )
        assert restored == raw
        write(target / "raw/index.json", index)
        for row in raw["cleanup"]:
            assert row["removed"] and row["process_stopped"] and row["mount_absent"] and not Path(row["path"]).exists()
            audit["cleanup"].append(row)
        audit["windows"][name] = dict(
            mounts=24,
            raw_sha256=digest(path),
            lossless_split=True,
            start=raw["started_epoch"],
            end=raw["ended_epoch"],
            cleanup=len(raw["cleanup"]),
        )
        for path in (source / "ext4-plain").iterdir():
            if path.is_file() and path.name != "results.json":
                audit["local_retained"][str(path)] = dict(bytes=path.stat().st_size, sha256=digest(path))
    tests = json.loads((out / "tests.json").read_text())
    assert tests["status"] == "PASS"
    assert max(row["end"] for row in tests["runs"]) < native["started_epoch"]
    for row in tests["cleanup"]:
        assert row["removed"] and not Path(row["path"]).exists()
        audit["cleanup"].append(row)
    write(here / "tails.json", tails)
    audit.update(
        status="PASS",
        fuse_mounts=96,
        request_or_barrier_tails_at_least_one_second=len(tails),
        max_retained_tail_seconds=max(row["seconds"] for row in tails),
        free_bytes=shutil.disk_usage(out).free,
    )
    audit["published"] = {
        str(p.relative_to(here.parents[1])): dict(bytes=p.stat().st_size, sha256=digest(p))
        for folder in (here, lookup)
        for p in folder.rglob("*")
        if p.is_file() and p.name != "audit.json" and "__pycache__" not in p.parts
    }
    write(here / "audit.json", audit)
    print(
        json.dumps(
            {
                k: audit[k]
                for k in (
                    "status",
                    "native_runs",
                    "fuse_mounts",
                    "request_or_barrier_tails_at_least_one_second",
                    "max_retained_tail_seconds",
                    "free_bytes",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
