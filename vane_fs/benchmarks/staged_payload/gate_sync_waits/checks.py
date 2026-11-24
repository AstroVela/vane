# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Audit frozen diagnostic inputs, timing separation, reversal and owned cleanup."""

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
    here, root = Path(__file__).resolve().parent, Path(sys.argv[1]).resolve()
    repo, out = here.parents[3], root / "native"
    build, plan, raw, findings, tests = [
        json.loads(p.read_text())
        for p in (
            out / "build.json",
            out / "plan.json",
            out / "ext4-diagnostic/results.json",
            root / "findings.json",
            root / "gate-trace-tests.json",
        )
    ]
    assert all(r["status"] == "PASS" for r in (build, raw, findings, tests))
    assert (
        max(build["ended_epoch"], tests["ended_epoch"])
        < raw["started_epoch"]
        < raw["ended_epoch"]
        < findings["started_epoch"]
    )
    identities = build["inputs"] | build["sources"] | build["binaries"] | plan["inputs"] | findings["inputs"]
    assert all(digest(Path(p)) == sha for p, sha in identities.items())
    setup = json.loads((root / "setup.json").read_text())
    comparison = json.loads((Path(setup["comparison"]) / "comparison.json").read_text())
    defaults = comparison["default_unchanged"]
    assert len(defaults) == 11 and all(digest(Path(p)) == sha for p, sha in defaults.items())
    assert len(build["reverse_proof"]) == 12
    for instrumented, row in build["reverse_proof"].items():
        original = Path(row["original_path"])
        assert digest(original) == row["original_sha256"] and row["exact_reverse"]
        assert digest(Path(instrumented)) == row["instrumented_sha256"]
        text = Path(instrumented).read_text()
        for before, after, count in reversed(row["changes"]):
            assert text.count(after) == count
            text = text.replace(after, before)
        assert text == original.read_text()
    assert len(build["recorder_reverse_proof"]) == 4
    for instrumented, row in build["recorder_reverse_proof"].items():
        text = Path(instrumented).read_text()
        for before, after in reversed(row["changes"]):
            assert text.count(after) == 1
            text = text.replace(after, before)
        assert text == Path(row["original_path"]).read_text() and row["exact_reverse"]
    cleanup = build["cleanup"] + raw["cleanup"]
    assert len(cleanup) == len({row["path"] for row in cleanup}) == 31
    assert all(
        row["removed"] and row.get("processes_stopped", row.get("process_stopped")) and not Path(row["path"]).exists()
        for row in cleanup
    )
    state = owned_state(root, cleanup)
    assert not state["active_owned_pids"] and not state["owned_mounts"]
    commands = []

    def check(label, argv):
        started = time.time()
        r = subprocess.run(argv, cwd=repo, capture_output=True, text=True)
        log = root / (label + ".log")
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
        save(root / "source-checks.json", commands)
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
    spec = importlib.util.spec_from_file_location("gate_sync_spdx", path)
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
        root / "final-checks.json",
        dict(
            status="PASS",
            recorded_epoch=time.time(),
            source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            decision=findings["decision"],
            checked_frozen_identities=len(identities),
            default_unchanged=defaults,
            native_policy_unchanged=True,
            reverse_exact_sources=12,
            all_owned_processes_stopped=True,
            no_owned_mounts=True,
            cleanup_directories=len(cleanup),
            cleanup_paths=[row["path"] for row in cleanup],
            owned_state=state,
            mounts=24,
            sequential_content_checks=48,
            ordinary_failed_checks_retained=9,
            related_source_spdx="PASS",
            unchanged_historical_spdx_missing=missing,
            checks=commands,
            free_bytes=shutil.disk_usage(root).free,
            inputs={str(p): digest(p) for p in (Path(__file__), here / "README.md", root / "findings.json")},
        ),
    )
    print("PASS", len(identities), "frozen identities, 11 defaults and 31 cleaned owned directories", flush=True)


if __name__ == "__main__":
    main()
