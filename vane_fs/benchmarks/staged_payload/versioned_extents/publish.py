# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain all extent validation/count evidence, excluding generated workspace data."""

import argparse
import json
import shutil
import time
from pathlib import Path

from generate import digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out, here = args.artifacts.resolve(), Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    count_build = json.loads((out / "count-build.json").read_text())
    plan = json.loads((out / "count-plan.json").read_text())
    counts = json.loads((out / "counts.json").read_text())
    analysis = json.loads((out / "analysis.json").read_text())
    assert all(item["status"] == "PASS" for item in (build, count_build, plan, counts, analysis))
    assert counts["host_monitor_stopped"]
    assert max(c["ended_epoch"] for c in build["commands"] + count_build["commands"]) < plan["frozen_epoch"]
    assert build["ended_epoch"] <= count_build["ended_epoch"] < plan["frozen_epoch"] <= counts["started_epoch"]
    assert max(run["ended_epoch"] for run in counts["runs"]) < counts["ended_epoch"] <= analysis["started_epoch"]
    inputs = dict(plan["inputs"], **analysis["inputs"])
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    preflight_cleanup = [
        row for path in (out / "preflight").rglob("count-build.json") for row in json.loads(path.read_text())["cleanup"]
    ]
    cleanup = build["cleanup"] + count_build["cleanup"] + counts["cleanup"] + preflight_cleanup
    assert len(cleanup) == len({row["path"] for row in cleanup})
    assert all(
        row["removed"] and row["processes_stopped"] and row["mount_absent"] and not Path(row["path"]).exists()
        for row in cleanup
    )
    binaries = set(build["binaries"]) | set(count_build["binaries"])
    for entry in Path("/proc").glob("[0-9]*/exe"):
        try:
            executable = str(entry.resolve(strict=True))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        assert executable not in binaries, entry
    audit = dict(
        status="PASS",
        published_epoch=time.time(),
        inputs=inputs,
        count_runs=len(counts["runs"]),
        full_content_checks=analysis["full_content_checks"],
        cleanup_directories=len(cleanup),
        all_owned_processes_stopped=True,
        build_test_and_count_windows_do_not_overlap=True,
        all_count_samples_retained=True,
        default_format_2_unchanged=True,
        original_files={},
        retained_files={},
    )

    def compact(relative, value):
        path = here / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, separators=(",", ":")) + "\n")
        assert path.stat().st_size < 500 * 1024, path
        assert json.loads(path.read_text()) == value

    def parts(relative, values):
        names, chunk, size = [], [], 2

        def flush():
            name = relative + f"/{len(names):03}.json"
            compact(name, chunk)
            names.append(name)

        for value in values:
            length = len(json.dumps(value, separators=(",", ":"))) + 1
            if chunk and size + length > 300000:
                flush()
                chunk, size = [], 2
            chunk.append(value)
            size += length
        if chunk:
            flush()
        assert [row for name in names for row in json.loads((here / name).read_text())] == values
        return names

    def retain(path, name):
        value = json.loads(path.read_text())
        compact(name, value)
        audit["original_files"][str(path)] = dict(sha256=digest(path), retained_as=name, values_unchanged=True)

    index = dict(counts)
    index["runs"] = []
    for i, row in enumerate(counts["runs"]):
        name = f"runs/{i:02}-{row['variant']}-{row['mib']}-{row['shape']}.json"
        compact(name, row)
        index["runs"].append(name)
    index["host"] = parts("host", counts["host"])
    index["cleanup"] = parts("cleanup/counts", counts["cleanup"])
    reconstructed = dict(index)
    reconstructed["runs"] = [json.loads((here / name).read_text()) for name in index["runs"]]
    for key in ("host", "cleanup"):
        reconstructed[key] = [row for name in index[key] for row in json.loads((here / name).read_text())]
    assert reconstructed == counts
    compact("counts.json", index)
    audit["original_files"][str(out / "counts.json")] = dict(
        sha256=digest(out / "counts.json"), retained_as="counts.json", values_unchanged=True
    )
    for name in (
        "source.json",
        "build.json",
        "count-build.json",
        "count-plan.json",
        "analysis.json",
        "final-checks.json",
    ):
        retain(out / name, "validation/" + name)
    compact("cleanup.json", dict(status="PASS", directories=len(cleanup), parts=parts("cleanup/all", cleanup)))
    for pattern in ("build-before-*.json", "build-source-*.json", "source-history/*.json"):
        for path in sorted(out.glob(pattern)):
            retain(path, "validation/history/" + str(path.relative_to(out)))
    for path in sorted(out.glob("build-harness-*.py")):
        target = here / "validation/history" / (path.name + ".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    logs = {p.name: dict(sha256=digest(p), text=p.read_text()) for p in out.glob("*.log")}
    compact("validation/logs.json", logs)
    for path in sorted((out / "preflight").rglob("*")):
        if path.is_file():
            relative = "preflight/" + str(path.relative_to(out / "preflight"))
            target = here / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    for path in sorted(here.rglob("*")):
        if path.is_file() and path.name != "audit.json" and "__pycache__" not in path.parts:
            assert path.stat().st_size < 500 * 1024, path
            audit["retained_files"][str(path.relative_to(here))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    save(here / "audit.json", audit)
    print("PASS retained all", len(counts["runs"]), "count runs; cleaned", len(cleanup), "owned directories")


if __name__ == "__main__":
    main()
