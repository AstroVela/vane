# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain every allocation sample, physical map and cleanup record with identities."""

import argparse
import collections
import json
import shutil
import time
from pathlib import Path

from run import digest, distribution, save


def physical_runs(extents):
    # Merge adjacent mappings only if both logical and physical addresses agree.
    # Counting FIEMAP records alone could mistake a split mapping for fragmentation.
    return sum(
        i == 0 or extent[:2] != [extents[i - 1][j] + extents[i - 1][2] for j in (0, 1)]
        for i, extent in enumerate(extents)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out, here = args.artifacts.resolve(), Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    results = {mode: json.loads((out / mode / "results.json").read_text()) for mode in ("plain", "diagnostic")}
    plans = {mode: json.loads((out / mode / "plan.json").read_text()) for mode in results}
    assert build["status"] == summary["status"] == "PASS"
    assert len(build["tests"]) == 18
    assert all(
        digest(Path(p)) == sha for p, sha in dict(build["inputs"], **build["binaries"], **build["default"]).items()
    )
    assert build["ended_epoch"] < results["plain"]["started_epoch"]
    assert results["plain"]["ended_epoch"] < results["diagnostic"]["started_epoch"]
    for mode, report in results.items():
        assert report["status"] == "PASS" and report["monitor_stopped"] and report["inputs_unchanged"]
        assert plans[mode]["frozen_epoch"] <= report["started_epoch"]
        assert all(digest(Path(p)) == sha for p, sha in plans[mode]["inputs"].items())
        assert len(report["runs"]) == (72 if mode == "plain" else 24)
        for run in report["runs"]:
            output = out / mode / f"{run['state']}-{run['bytes'] // 1024**2}-{run['rep']}-{run['variant']}.json"
            original = json.loads(output.read_text())
            assert all(run[key] == value for key, value in original.items())
            assert run["content_and_guard_valid"] and run["fiemap_code"] == 0
            assert all(e[3] in (0, 1) for e in run["extents"]), "Unexpected unstable/unwritten FIEMAP flag"
            # The last extent has FIEMAP_EXTENT_LAST; no mapping is discarded.
            assert run["extents"][-1][3] == 1
    cleanup = build["cleanup"] + [row for report in results.values() for row in report["cleanup"]]
    assert len(cleanup) == len({r["path"] for r in cleanup}) == 114
    assert all(row["removed"] and row["processes_stopped"] and not Path(row["path"]).exists() for row in cleanup)
    binaries = {str(out / ("probe-" + mode)).encode() for mode in ("plain", "diagnostic", "asan")}
    monitor = str(here.parents[1] / "io_interference/monitor.py").encode()
    for process in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = process.read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        assert not argv or argv[0] not in binaries, f"Probe still running: {process}"
        assert not (len(argv) >= 3 and argv[1] == monitor and argv[2].startswith(str(out).encode())), process
    maps = []
    for mode, report in results.items():
        for state in ("fresh", "after_gc"):
            for size in (64, 256):
                for variant in ("delayed", "reserve256k", "reserve1m"):
                    rows = [
                        r
                        for r in report["runs"]
                        if r["state"] == state and r["bytes"] == size * 1024**2 and r["variant"] == variant
                    ]
                    flags = collections.Counter(e[3] for r in rows for e in r["extents"])
                    maps.append(
                        dict(
                            mode=mode,
                            state=state,
                            size_mib=size,
                            variant=variant,
                            physical_runs=distribution([physical_runs(r["extents"]) for r in rows]),
                            extent_flags=dict(flags),
                        )
                    )
    candidates = [g for g in summary["groups"] if g["variant"] != "delayed"]
    audit = dict(
        status="PASS",
        published_epoch=time.time(),
        plain_runs=72,
        diagnostic_runs=24,
        related_probe_validations=18,
        asan_ubsan_probe_validations=sum(t["mode"] == "asan" for t in build["tests"]),
        failed_checks=sum(not value for g in candidates for value in g["checks"].values()),
        passing_candidate_groups=sum(g["screen_pass"] for g in candidates),
        component_candidate_warranted=summary["component_candidate_warranted"],
        default_unchanged=build["default"],
        cleanup_directories=114,
        owned_processes_stopped=True,
        owned_build_test_measurement_windows_do_not_overlap=True,
        windows={mode: {k: r[k] for k in ("started_epoch", "ended_epoch")} for mode, r in results.items()},
        exact_original_files={},
        retained_files={},
    )
    # Preserve JSON values exactly while removing formatting-only repetition.
    for relative in ("build.json", "summary.json"):
        source = out / relative
        target = here / ("validation/build.json" if relative == "build.json" else relative)
        target.parent.mkdir(exist_ok=True, parents=True)
        original = json.loads(source.read_text())
        target.write_text(json.dumps(original, separators=(",", ":")) + "\n")
        assert json.loads(target.read_text()) == original
        audit["exact_original_files"][str(source)] = dict(
            sha256=digest(source), retained_as=str(target.relative_to(here)), values_unchanged=True
        )
    for mode in results:
        report = results[mode]
        index = dict(report, runs=[])
        chunks = {}
        for run in report["runs"]:
            name = f"{mode}/groups/{run['state']}-{run['bytes'] // 1024**2}-{run['variant']}.json"
            rows = chunks.setdefault(name, [])
            index["runs"].append(dict(file=name, index=len(rows)))
            rows.append(run)
        for name, rows in chunks.items():
            target = here / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(rows, separators=(",", ":")) + "\n")
            assert target.stat().st_size < 500 * 1024
        save(here / mode / "index.json", index)
        reconstructed = dict(
            index, runs=[json.loads((here / r["file"]).read_text())[r["index"]] for r in index["runs"]]
        )
        assert reconstructed == report
        source = out / mode / "results.json"
        audit["exact_original_files"][str(source)] = dict(
            sha256=digest(source), retained_as=[f"{mode}/index.json", *chunks], values_unchanged=True
        )
        # Replace only this publisher's earlier untracked aggregate output.
        old_aggregate = here / mode / "results.json"
        if old_aggregate.exists():
            assert json.loads(old_aggregate.read_text()) == reconstructed
            old_aggregate.unlink()
        for name in ("plan.json", "host.jsonl"):
            source, target = out / mode / name, here / mode / name
            shutil.copy2(source, target)
            assert digest(source) == digest(target)
    save(here / "physical-runs.json", dict(status="PASS", groups=maps))
    save(here / "cleanup.json", dict(status="PASS", directories=cleanup))
    logs = {
        str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=digest(p), text=p.read_text())
        for p in out.rglob("*.log")
    }
    save(here / "validation/logs.json", logs)
    for path in sorted(here.rglob("*")):
        if path.is_file() and path.name != "audit.json" and "__pycache__" not in path.parts:
            audit["retained_files"][str(path.relative_to(here))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    save(here / "audit.json", audit)
    print(
        json.dumps(
            {
                k: audit[k]
                for k in (
                    "status",
                    "plain_runs",
                    "diagnostic_runs",
                    "cleanup_directories",
                    "failed_checks",
                    "component_candidate_warranted",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
