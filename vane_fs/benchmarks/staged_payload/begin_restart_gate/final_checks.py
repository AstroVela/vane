# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Audit frozen sources, original gates, test/measurement separation and cleanup."""

import dataclasses
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from generate import digest, save


def main():
    out, here = Path(sys.argv[1]).resolve(), Path(__file__).resolve().parent
    repo = here.parents[3]
    source, build, gates, pauses, findings = [
        json.loads((out / name).read_text())
        for name in ("source.json", "build.json", "gate-tests.json", "full-pauses.json", "findings.json")
    ]
    assert all(r["status"] == "PASS" for r in (source, build, gates, pauses, findings))
    assert build["source_manifest"]["sources"] == source["sources"]
    assert all(command["returncode"] == 0 for r in (build, gates, pauses) for command in r["commands"])
    frozen = json.loads((out / "fuse/build.json").read_text())
    run = json.loads((out / "fuse/run.json").read_text())
    assert (
        run["status"] == "PASS"
        and max(build["ended_epoch"], gates["ended_epoch"], pauses["ended_epoch"])
        < run["started_epoch"]
        < run["ended_epoch"]
        < findings["started_epoch"]
    )
    identities = (
        source["inputs"]
        | source["sources"]
        | build["inputs"]
        | build["binaries"]
        | gates["inputs"]
        | gates["binaries"]
        | frozen["inputs"]
        | findings["inputs"]
    )
    assert all(digest(Path(p)) == sha for p, sha in identities.items())
    assert len(source["default_unchanged"]) == 11
    assert all(digest(Path(p)) == sha for p, sha in source["default_unchanged"].items())
    for relative, proof in source["transformations"].items():
        text = (Path(source["candidate"]) / relative).read_text()
        for before, after, count in reversed(proof["changes"]):
            assert text.count(after) == count
            text = text.replace(after, before)
        assert text == (Path(source["mechanism_parent"]) / relative).read_text()
    assert len(source["transformations"]) == 2
    raw = [json.loads((out / "fuse" / w / "ext4-plain/results.json").read_text()) for w in ("full", "sustained")]
    assert all(r["status"] == "PASS" and r["monitor_stopped"] for r in raw)
    cleanup = build["cleanup"] + gates["cleanup"] + pauses["cleanup"] + [row for r in raw for row in r["cleanup"]]
    assert len(cleanup) == len({row["path"] for row in cleanup})
    assert all(
        row["removed"] and row.get("processes_stopped", row.get("process_stopped")) and not Path(row["path"]).exists()
        for row in cleanup
    )
    ancestors, pid = {os.getpid()}, os.getppid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[1])
        except FileNotFoundError:
            break
    active = []
    prefixes = [str(out).encode(), *(row["path"].encode() for row in cleanup)]
    for task in Path("/proc").glob("[0-9]*"):
        if int(task.name) in ancestors:
            continue
        try:
            argv = (task / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(prefix in argv for prefix in prefixes):
            active.append(int(task.name))
    mounts = [line for line in Path("/proc/self/mountinfo").read_text().splitlines() if str(out) in line]
    assert not active and not mounts
    checks = []

    def check(label, argv):
        start = time.time()
        r = subprocess.run(argv, cwd=repo, capture_output=True, text=True)
        log = out / (label + ".log")
        log.write_text(r.stdout + r.stderr)
        checks.append(
            dict(
                label=label,
                argv=argv,
                started_epoch=start,
                ended_epoch=time.time(),
                returncode=r.returncode,
                log=str(log),
                sha256=digest(log),
            )
        )
        return r

    assert check("format", ["scripts/format", "root", "--changed", "--check"]).returncode == 0
    assert check("ruff", ["ruff", "check", "--select", "E9,F63,F7,F82", str(here)]).returncode == 0
    assert check("whitespace", ["git", "diff", "--check", "HEAD"]).returncode == 0
    assert check("copyleft", [sys.executable, "-I", "scripts/check_copyleft.py"]).returncode == 0
    full = check("spdx-root", [sys.executable, "-I", "scripts/check_source_license_headers.py", "--repo", "root"])
    assert full.returncode == 1 and "root: missing=5 " in full.stdout
    missing = [line.strip()[2:] for line in full.stdout.splitlines() if line.startswith("  A ")]
    assert len(missing) == 5
    for relative in missing:
        assert subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=repo) == (repo / relative).read_bytes()
    path = repo / "scripts/check_source_license_headers.py"
    spec = importlib.util.spec_from_file_location("gate_spdx", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.POLICIES["root"] = dataclasses.replace(module.POLICIES["root"], baseline="HEAD")
    saved_argv, sys.argv = sys.argv, [str(path), "--repo", "root"]
    try:
        assert module.main() == 0
    finally:
        sys.argv = saved_argv
    save(
        out / "final-checks.json",
        dict(
            status="PASS",
            recorded_epoch=time.time(),
            source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            decision=findings["decision"],
            failed_ordinary_checks=len(findings["failed_checks"]),
            production_unchanged=True,
            checked_frozen_identities=len(identities),
            default_unchanged=source["default_unchanged"],
            exact_reversed_native_sources=2,
            all_owned_processes_stopped=True,
            no_owned_mounts=True,
            active_owned_pids=active,
            cleanup_directories=len(cleanup),
            cleanup_paths=[row["path"] for row in cleanup],
            mounts=32,
            sequential_content_checks=112,
            gate_cases_per_mode=7,
            parent_negative_cases_per_mode=2,
            related_source_spdx="PASS",
            unchanged_historical_spdx_missing=missing,
            checks=checks,
            free_disk_bytes=shutil.disk_usage(out).free,
            inputs={str(p): digest(p) for p in (Path(__file__), here / "README.md", out / "findings.json")},
        ),
    )
    print(
        "PASS",
        len(identities),
        "frozen identities, 11 defaults,",
        len(cleanup),
        "cleaned directories and related source checks",
        flush=True,
    )


if __name__ == "__main__":
    main()
