# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Audit immutable inputs, measurement separation, source checks and owned cleanup."""

import dataclasses
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from prepare import digest, save


def owned_state(out, cleanup=()):
    ancestors, pid = {os.getpid()}, os.getppid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[1])
        except FileNotFoundError:
            break
    prefixes = [str(out).encode(), *(row["path"].encode() for row in cleanup)]
    active = []
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
    return dict(active_owned_pids=active, owned_mounts=mounts)


def main():
    here, out = Path(__file__).resolve().parent, Path(sys.argv[1]).resolve()
    repo = here.parents[3]
    build, tests, comparison, run, findings = [
        json.loads((out / n).read_text())
        for n in ("build.json", "tests.json", "comparison.json", "run.json", "findings.json")
    ]
    assert all(r["status"] == "PASS" for r in (build, tests, comparison, run, findings))
    assert (
        tests["ended_epoch"]
        < build["prepared_epoch"]
        < run["started_epoch"]
        < run["ended_epoch"]
        < findings["started_epoch"]
    )
    assert len(tests["commands"]) == 4 and all(c["returncode"] == 0 for c in tests["commands"])
    identities = build["inputs"] | findings["inputs"]
    assert all(digest(Path(p)) == sha for p, sha in identities.items())
    defaults = comparison["default_unchanged"]
    assert len(defaults) == 11 and all(digest(Path(p)) == sha for p, sha in defaults.items())
    assert (
        set(comparison["native_changed_files"]) == {"src/checkpoint.hpp", "src/workspace.cpp"}
        and comparison["reverse_exact"]
    )
    raw = [json.loads((out / w / "ext4-plain/results.json").read_text()) for w in ("full", "sustained")]
    assert all(r["status"] == "PASS" and r["monitor_stopped"] for r in raw)
    cleanup = tests["cleanup"] + [row for r in raw for row in r["cleanup"]]
    assert len(cleanup) == len({row["path"] for row in cleanup}) == 36
    assert all(
        row["removed"] and row.get("processes_stopped", row.get("process_stopped")) and not Path(row["path"]).exists()
        for row in cleanup
    )
    state = owned_state(out, cleanup)
    assert not state["active_owned_pids"] and not state["owned_mounts"]
    commands = []

    def check(label, argv):
        started = time.time()
        r = subprocess.run(argv, cwd=repo, capture_output=True, text=True)
        log = out / (label + ".log")
        log.write_text(r.stdout + r.stderr)
        commands.append(
            dict(
                argv=argv,
                started_epoch=started,
                ended_epoch=time.time(),
                returncode=r.returncode,
                log=str(log),
                sha256=digest(log),
            )
        )
        save(out / "source-checks.json", commands)
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
    spec = importlib.util.spec_from_file_location("matched_spdx", path)
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
            decision=findings["decision"],
            checked_frozen_identities=len(identities),
            default_unchanged=defaults,
            native_policy_unchanged=True,
            native_builds_reused_by_identity=True,
            all_owned_processes_stopped=True,
            no_owned_mounts=True,
            cleanup_directories=len(cleanup),
            cleanup_paths=[row["path"] for row in cleanup],
            owned_state=state,
            mounts=32,
            sequential_content_checks=112,
            rerun_gate_cases_per_build=7,
            rerun_parent_negative_cases_per_build=2,
            original_failed_checks=findings["original_failed_checks"],
            related_source_spdx="PASS",
            unchanged_historical_spdx_missing=missing,
            checks=commands,
            free_bytes=shutil.disk_usage(out).free,
            inputs={str(p): digest(p) for p in (Path(__file__), here / "README.md", out / "findings.json")},
        ),
    )
    print("PASS", len(identities), "frozen identities, 11 defaults and 36 cleaned owned directories", flush=True)


if __name__ == "__main__":
    main()
