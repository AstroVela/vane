# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain complete native attribution, buffer comparison and ordinary FUSE evidence."""

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
    borrow = json.loads((out / "borrow-build.json").read_text())
    comparison = json.loads((out / "compare-summary.json").read_text())
    fuse = json.loads((out / "fuse-screen/summary.json").read_text())
    assert build["status"] == borrow["status"] == comparison["status"] == fuse["status"] == "PASS"
    assert comparison["screen_pass"] and not fuse["pass_all"]
    inputs = dict(
        build["inputs"],
        **build["binaries"],
        **build["default"],
        **build["diagnostic_sources"],
        **borrow["inputs"],
        **borrow["sources"],
        **borrow["binaries"],
    )
    inputs.update(build["component"]["sources"])
    inputs.update(build["component"]["binaries"])
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    reports = {
        name: json.loads((out / path).read_text())
        for name, path in [
            ("plain", "plain/results.json"),
            ("diagnostic", "diagnostic/results.json"),
            ("compare", "compare/results.json"),
            ("fuse", "fuse-screen/ext4-plain/results.json"),
        ]
    }
    for name, count in [("plain", 24), ("diagnostic", 8), ("compare", 48)]:
        assert reports[name]["status"] == "PASS" and reports[name]["monitor_stopped"]
        assert reports[name]["inputs_unchanged"] and len(reports[name]["runs"]) == count
    assert reports["fuse"]["status"] == "PASS" and reports["fuse"]["monitor_stopped"]
    assert len(reports["fuse"]["fuse"]) == len(reports["fuse"]["aged_fuse"]) == 12
    windows = [dict(name=name, begin=r["started_epoch"], end=r["ended_epoch"]) for name, r in reports.items()]
    assert all(a["end"] < b["begin"] for a, b in zip(windows, windows[1:]))
    for command in build["commands"] + borrow["commands"]:
        begin = command.get("started_epoch", command.get("start"))
        end = command.get("ended_epoch", command.get("end"))
        assert all(end <= w["begin"] or begin >= w["end"] for w in windows)
    for mode in ("plain", "diagnostic", "compare"):
        plan = json.loads((out / mode / "plan.json").read_text())
        assert all(digest(Path(p)) == sha for p, sha in plan["inputs"].items())
    plan = json.loads((out / "fuse-screen/plan.json").read_text())
    assert all(digest(Path(p)) == sha for p, sha in plan["inputs"].items())
    cleanup = build["cleanup"] + borrow["cleanup"] + [c for r in reports.values() for c in r["cleanup"]]
    assert len(cleanup) == len({c["path"] for c in cleanup}) == 116
    for row in cleanup:
        assert row["removed"] and not Path(row["path"]).exists()
        assert row.get("processes_stopped", row.get("process_stopped"))
        assert row.get("mount_absent", True)
    binaries = {str(Path(p).resolve()).encode() for p in dict(build["binaries"], **borrow["binaries"])}
    for proc in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = proc.read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        executable = str(Path(argv[0].decode()).resolve()).encode() if argv and argv[0] else b""
        assert executable not in binaries, proc
    audit = dict(
        status="PASS",
        published_epoch=time.time(),
        windows=windows,
        default_unchanged=build["default"],
        native_attribution_processes=32,
        paired_native_processes=48,
        fuse_mounts=24,
        native_screen_pass=True,
        fuse_screen_pass=False,
        fuse_failed_checks=sum(not c["pass_gate"] for c in fuse["checks"]),
        cleanup_directories=116,
        all_owned_processes_stopped=True,
        owned_build_test_measurement_windows_do_not_overlap=True,
        inputs=inputs,
        original_files={},
        retained_files={},
    )

    def compact(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, separators=(",", ":")) + "\n")
        assert json.loads(path.read_text()) == value
        assert path.stat().st_size < 500 * 1024, path

    def copy_json(relative, destination, split=()):
        source = out / relative
        value = json.loads(source.read_text())
        index = dict(value)
        for key in split:
            index[key] = []
            for i, row in enumerate(value[key]):
                relative_row = f"{destination}/{key}/{i:03}.json"
                compact(here / relative_row, row)
                index[key].append(dict(file=relative_row))
        target = here / (destination + "/index.json" if split else destination)
        compact(target, index)
        reconstructed = dict(index)
        for key in split:
            reconstructed[key] = [json.loads((here / r["file"]).read_text()) for r in index[key]]
        assert reconstructed == value
        audit["original_files"][str(source)] = dict(
            sha256=digest(source), retained_as=str(target.relative_to(here)), values_unchanged=True
        )

    def copy_lines(source, label):
        lines = source.read_text().splitlines(True)
        retained = []
        for i in range(0, len(lines), 100):
            path = here / (label + f"-{i // 100:03}.jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(lines[i : i + 100]))
            assert path.stat().st_size < 500 * 1024
            retained.append(str(path.relative_to(here)))
        assert "".join((here / p).read_text() for p in retained) == source.read_text()
        audit["original_files"][str(source)] = dict(sha256=digest(source), retained_as=retained, values_unchanged=True)

    for name in ("plain", "diagnostic", "compare"):
        copy_json(f"{name}/results.json", name, ("runs",))
        copy_json(f"{name}/plan.json", f"{name}/plan.json")
        copy_lines(out / name / "host.jsonl", f"{name}/host")
        for row in reports[name]["runs"]:
            source = Path(row["cpu_samples_file"])
            copy_lines(source, f"cpu/{name}-{source.stem}")
    copy_json("analysis.json", "attribution", ("runs", "groups"))
    copy_json("compare-summary.json", "compare-summary.json")
    copy_json("fuse-screen/ext4-plain/results.json", "fuse", ("fuse", "aged_fuse"))
    copy_json("fuse-screen/summary.json", "fuse-summary.json")
    copy_json("fuse-screen/plan.json", "fuse/plan.json")
    copy_json("fuse-screen/input-check.json", "fuse/input-check.json")
    copy_lines(out / "fuse-screen/ext4-plain/host.jsonl", "fuse/host")
    for name in ("build.json", "borrow-build.json", "source-proof.json", "compare-runner-proof.json"):
        copy_json(name, "validation/" + name)
    if (out / "final-checks.json").exists():
        copy_json("final-checks.json", "validation/final-checks.json")
    for folder in ("borrow-preflight", "borrow-preflight-arrow22", "borrow-preflight-runtime"):
        for source in sorted((out / folder).iterdir()):
            target = here / "preflight" / folder / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            assert target.stat().st_size < 500 * 1024
    environment = out / "test-environment-freeze.txt"
    (here / "validation/test-environment-freeze.txt").write_bytes(environment.read_bytes())
    logs = {
        str(p.relative_to(out)): dict(sha256=digest(p), bytes=p.stat().st_size, text=p.read_text())
        for p in out.glob("*.log")
    }
    compact(here / "validation/logs.json", logs)
    parts, chunk = [], []
    for row in cleanup:
        if chunk and len(json.dumps(chunk + [row], separators=(",", ":"))) > 350000:
            name = f"cleanup/{len(parts):03}.json"
            compact(here / name, chunk)
            parts.append(name)
            chunk = []
        chunk.append(row)
    if chunk:
        name = f"cleanup/{len(parts):03}.json"
        compact(here / name, chunk)
        parts.append(name)
    assert [row for name in parts for row in json.loads((here / name).read_text())] == cleanup
    save(here / "cleanup.json", dict(status="PASS", directory_count=len(cleanup), chunks=parts))
    for path in sorted(here.rglob("*")):
        if path.is_file() and path.name != "audit.json" and "__pycache__" not in path.parts:
            assert path.stat().st_size < 500 * 1024, path
            audit["retained_files"][str(path.relative_to(here))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    save(here / "audit.json", audit)
    print("PASS retained", len(audit["retained_files"]), "artifacts;", len(cleanup), "cleaned owned directories")


if __name__ == "__main__":
    main()
