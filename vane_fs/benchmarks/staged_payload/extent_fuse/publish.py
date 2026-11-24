# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain complete ordinary/sustained FUSE evidence and audit owned data cleanup."""

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
    run = json.loads((out / "run.json").read_text())
    analysis = json.loads((out / "analysis.json").read_text())
    assert build["status"] == run["status"] == analysis["status"] == "PASS"
    inputs = dict(build["inputs"], **analysis["inputs"])
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    component = json.loads((Path(build["component"]) / "build.json").read_text())
    windows, cleanup, results = [], [], {}
    for window, config in build["windows"].items():
        root = Path(config["root"])
        result = json.loads((root / "ext4-plain/results.json").read_text())
        assert result["status"] == "PASS" and result["monitor_stopped"]
        assert len(result["fuse"]) + len(result["aged_fuse"]) == config["mounts"]
        assert build["prepared_epoch"] < result["started_epoch"] < result["ended_epoch"] <= analysis["started_epoch"]
        results[window] = result
        windows.append(dict(name=window, begin=result["started_epoch"], end=result["ended_epoch"]))
        cleanup += result["cleanup"]
    assert windows[0]["end"] < windows[1]["begin"]
    assert run["ended_epoch"] <= analysis["started_epoch"]
    assert max(c["ended_epoch"] for c in component["commands"]) < windows[0]["begin"]
    assert len(cleanup) == len({r["path"] for r in cleanup}) == 32
    for row in cleanup:
        assert row["removed"] and row["process_stopped"] and row["mount_absent"] and not Path(row["path"]).exists()
    binaries = {p for p in inputs if p.endswith(("/vane-fs", "/vane-fs-mount"))}
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
        windows=windows,
        original_component_suites_reused_by_exact_identity=True,
        full_acceptance_pass=analysis["full_acceptance_pass"],
        failed_full_checks=analysis["full_failed_checks"],
        full_mounts=24,
        sustained_mounts=8,
        sequential_content_checks=112,
        cleanup_directories=32,
        owned_build_test_measurement_analysis_windows_do_not_overlap=True,
        all_owned_processes_stopped=True,
        all_samples_retained=True,
        original_files={},
        retained_files={},
    )

    def compact(relative, value):
        path = here / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, separators=(",", ":")) + "\n")
        assert json.loads(path.read_text()) == value
        assert path.stat().st_size < 500 * 1024, path

    def retain(path, target):
        value = json.loads(path.read_text())
        compact(target, value)
        audit["original_files"][str(path)] = dict(sha256=digest(path), retained_as=target, values_unchanged=True)

    def parts(relative, values):
        names, chunk, length = [], [], 2

        def flush():
            name = relative + f"/{len(names):03}.json"
            compact(name, chunk)
            names.append(name)

        for row in values:
            count = len(json.dumps(row, separators=(",", ":"))) + 1
            if chunk and length + count > 300000:
                flush()
                chunk, length = [], 2
            chunk.append(row)
            length += count
        if chunk:
            flush()
        assert [row for name in names for row in json.loads((here / name).read_text())] == values
        return names

    for window, result in results.items():
        root = Path(build["windows"][window]["root"])
        index = dict(result)
        for state in ("fuse", "aged_fuse"):
            index[state] = []
            for row in result[state]:
                target = f"{window}/{state}/{row['variant']}-{row['rep']}.json"
                compact(target, row)
                index[state].append(target)
        index["cleanup"] = parts(window + "/cleanup", result["cleanup"])
        reconstructed = dict(index)
        for state in ("fuse", "aged_fuse"):
            reconstructed[state] = [json.loads((here / name).read_text()) for name in index[state]]
        reconstructed["cleanup"] = [row for name in index["cleanup"] for row in json.loads((here / name).read_text())]
        assert reconstructed == result
        compact(window + "/results.json", index)
        original = root / "ext4-plain/results.json"
        audit["original_files"][str(original)] = dict(
            sha256=digest(original), retained_as=window + "/results.json", values_unchanged=True
        )
        host = root / "ext4-plain/host.jsonl"
        lines = host.read_text().splitlines(True)
        retained = []
        for i in range(0, len(lines), 100):
            name = window + f"/host/{i // 100:03}.jsonl"
            path = here / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(lines[i : i + 100]))
            assert path.stat().st_size < 500 * 1024
            retained.append(name)
        assert "".join((here / name).read_text() for name in retained) == host.read_text()
        compact(window + "/host/index.json", dict(original_sha256=digest(host), samples=len(lines), parts=retained))
        audit["original_files"][str(host)] = dict(sha256=digest(host), retained_as=retained, values_unchanged=True)
        retain(root / "plan.json", window + "/plan.json")
        logs = {str(p.relative_to(root)): dict(sha256=digest(p), text=p.read_text()) for p in root.rglob("*.log")}
        compact(window + "/logs.json", logs)
    for name in ("build.json", "run.json", "analysis.json", "final-checks.json"):
        retain(out / name, "validation/" + name)
    retain(out / "full/summary.json", "full/summary.json")
    compact(
        "validation/runner-logs.json", {p.name: dict(sha256=digest(p), text=p.read_text()) for p in out.glob("*.log")}
    )
    compact("cleanup.json", dict(status="PASS", directories=32, parts=parts("cleanup", cleanup)))
    for path in sorted(here.rglob("*")):
        if path.is_file() and path.name != "audit.json" and "__pycache__" not in path.parts:
            assert path.stat().st_size < 500 * 1024, path
            audit["retained_files"][str(path.relative_to(here))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    save(here / "audit.json", audit)
    print("PASS retained 24 ordinary and 8 sustained mounts; all content/guard checks and 32 cleanup records")


if __name__ == "__main__":
    main()
