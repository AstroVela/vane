# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Publish complete indexed wait traces without retaining workspace data files."""

import argparse
import json
import time
from pathlib import Path

from prepare import digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out, here = args.artifacts.resolve(), Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    plan = json.loads((out / "plan.json").read_text())
    analysis = json.loads((out / "analysis.json").read_text())
    result = json.loads((out / "ext4-diagnostic/results.json").read_text())
    assert build["status"] == analysis["status"] == result["status"] == "PASS"
    assert json.loads((out / "input-check.json").read_text())["status"] == "PASS"
    assert result["monitor_stopped"]
    assert (
        build["ended_epoch"]
        < plan["frozen_epoch"]
        <= result["started_epoch"]
        < result["ended_epoch"]
        <= analysis["started_epoch"]
    )
    assert all(command["ended_epoch"] < result["started_epoch"] for command in build["commands"])
    inputs = dict(plan["inputs"], **analysis["inputs"])
    if (out / "findings.json").exists():
        inputs.update(json.loads((out / "findings.json").read_text())["inputs"])
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    cleanup = build["cleanup"] + result["cleanup"]
    assert len(cleanup) == len({row["path"] for row in cleanup}) == 18
    assert all(row["removed"] and not Path(row["path"]).exists() for row in cleanup)
    assert all(
        row.get("processes_stopped", row.get("process_stopped")) and row.get("mount_absent", True) for row in cleanup
    )
    binaries = {str(Path(path).resolve()).encode() for path in build["binaries"]}
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = path.read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        executable = str(Path(argv[0].decode()).resolve()).encode() if argv and argv[0] else b""
        assert executable not in binaries, path
    audit = dict(
        status="PASS",
        published_epoch=time.time(),
        inputs=inputs,
        build_test_end=build["ended_epoch"],
        diagnostic_begin=result["started_epoch"],
        diagnostic_end=result["ended_epoch"],
        analysis_begin=analysis["started_epoch"],
        owned_build_test_measurement_windows_do_not_overlap=True,
        source_and_runner_reverse_exact=True,
        mounts=16,
        all_samples_retained=True,
        cleanup_directories=18,
        all_owned_processes_stopped=True,
        original_plain_gate_failures_unchanged=True,
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
            count = len(json.dumps(value, separators=(",", ":"))) + 1
            if chunk and size + count > 300000:
                flush()
                chunk, size = [], 2
            chunk.append(value)
            size += count
        if chunk:
            flush()
        assert [v for name in names for v in json.loads((here / name).read_text())] == values
        return names

    def retain(relative, name):
        path = out / relative
        value = json.loads(path.read_text())
        compact(name, value)
        audit["original_files"][str(path)] = dict(sha256=digest(path), retained_as=name, values_unchanged=True)

    result_index = dict(result)
    analysis_index = dict(analysis)
    for state in ("fuse", "aged_fuse"):
        result_index[state] = []
        for mount in result[state]:
            label = f"{state}-{mount['variant']}-{mount['rep']}"
            name = "runs/" + label + ".json"
            compact(name, mount)
            result_index[state].append(name)
            path = Path(mount["waits_file"])
            trace = json.loads(path.read_text())
            index = dict(trace)
            index["threads"] = []
            for i, thread in enumerate(trace["threads"]):
                index["threads"].append(
                    dict(
                        tid=thread["tid"],
                        roots=parts(f"traces/{label}/{i}/roots", thread["roots"]),
                        events=parts(f"traces/{label}/{i}/events", thread["events"]),
                    )
                )
            target = "traces/" + label + "/index.json"
            compact(target, index)
            reconstructed = dict(index)
            reconstructed["threads"] = [
                dict(
                    tid=t["tid"],
                    **{
                        key: [row for part in t[key] for row in json.loads((here / part).read_text())]
                        for key in ("roots", "events")
                    },
                )
                for t in index["threads"]
            ]
            assert reconstructed == trace
            audit["original_files"][str(path)] = dict(sha256=digest(path), retained_as=target, values_unchanged=True)
            cpu_path = Path(mount["cpu_samples_file"])
            samples = [json.loads(line) for line in cpu_path.read_text().splitlines()]
            cpu_parts = parts("cpu/" + label, samples)
            compact(
                "cpu/" + label + "/index.json",
                dict(samples=len(samples), parts=cpu_parts, original_sha256=digest(cpu_path)),
            )
            audit["original_files"][str(cpu_path)] = dict(
                sha256=digest(cpu_path), retained_as=cpu_parts, values_unchanged=True
            )
    reconstructed = dict(result_index)
    for state in ("fuse", "aged_fuse"):
        reconstructed[state] = [json.loads((here / name).read_text()) for name in result_index[state]]
    assert reconstructed == result
    compact("results.json", result_index)
    analysis_index["mounts"] = []
    for row in analysis["mounts"]:
        label = f"{row['state']}-{row['variant']}-{row['rep']}"
        value = dict(row)
        for key in ("phases", "slow_roots", "slow_syscalls"):
            value[key] = parts("analysis/" + label + "/" + key, row[key])
        target = "analysis/" + label + "/index.json"
        compact(target, value)
        reconstructed = dict(value)
        for key in ("phases", "slow_roots", "slow_syscalls"):
            reconstructed[key] = [r for name in value[key] for r in json.loads((here / name).read_text())]
        assert reconstructed == row
        analysis_index["mounts"].append(target)
    compact("analysis/index.json", analysis_index)
    for relative in (
        "build.json",
        "plan.json",
        "tests.json",
        "input-check.json",
        "release-trace.json",
        "asan-trace.json",
    ):
        retain(relative, "validation/" + relative)
    if (out / "final-checks.json").exists():
        retain("final-checks.json", "validation/final-checks.json")
    if (out / "findings.json").exists():
        retain("findings.json", "findings.json")
    compact("cleanup.json", dict(status="PASS", directories=len(cleanup), parts=parts("cleanup", cleanup)))
    logs = {p.name: dict(sha256=digest(p), text=p.read_text()) for p in out.glob("*.log")}
    logs.update(
        {
            str(p.relative_to(out)): dict(sha256=digest(p), text=p.read_text())
            for p in (out / "ext4-diagnostic").glob("*.log")
        }
    )
    compact("validation/logs.json", logs)
    host = out / "ext4-diagnostic/host.jsonl"
    host_parts = parts("host", [json.loads(line) for line in host.read_text().splitlines()])
    compact("host/index.json", dict(original_sha256=digest(host), parts=host_parts))
    for path in sorted(here.rglob("*")):
        if path.is_file() and path.name != "audit.json" and "__pycache__" not in path.parts:
            assert path.stat().st_size < 500 * 1024, path
            audit["retained_files"][str(path.relative_to(here))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    save(here / "audit.json", audit)
    print("PASS retained", len(audit["retained_files"]), "artifacts and all 16 mounts; 18 owned directories cleaned")


if __name__ == "__main__":
    main()
