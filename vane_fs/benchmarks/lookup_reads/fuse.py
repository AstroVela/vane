# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate production lookup changes with the original plain FUSE workload."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    out = args.output.resolve()
    out.mkdir()
    assert json.loads(args.validation.read_text())["status"] == "PASS"
    runner = here.parent / "partial_write/acceptance/run_fuse.py"
    analyzer = here.parent / "staged_payload/partial_sync/analyze_writeback.py"
    original = json.loads((here.parent / "staged_payload/writeback_budget/writeback-plan.json").read_text())
    identities = {
        str(p): digest(p) for p in (Path(__file__).resolve(), runner, runner.with_name("workload.py"), analyzer)
    }
    for name, component, venv in (
        ("baseline", args.baseline.resolve(), args.baseline.resolve() / "venv"),
        ("updated", repo / "vane_fs", repo / ".venv"),
    ):
        alias = out / name
        (alias / "build").mkdir(parents=True)
        (alias / "build/python-release").symlink_to(component / "build/python-release", target_is_directory=True)
        (alias / "venv").symlink_to(venv, target_is_directory=True)
        for directory in ("src", "include"):
            for path in (component / directory).rglob("*"):
                if path.is_file():
                    identities[str(path)] = digest(path)
        for path in (
            component / "build/python-release/vane-fs-mount",
            venv / "lib/python3.12/site-packages/vane_fs/_native.cpython-312-x86_64-linux-gnu.so",
        ):
            identities[str(path)] = digest(path)
        observed = subprocess.check_output(
            [str(venv / "bin/python"), "-I", "-c", "import vane_fs._native as n;print(n.__file__)"], text=True
        ).strip()
        assert Path(observed).resolve().is_relative_to(venv.resolve())
    plan = dict(
        status="FROZEN_BEFORE_PLAIN_COMPARISON",
        frozen_epoch=time.time(),
        media=["ext4"],
        orders=original["orders"],
        regression_gates=original["regression_gates"],
        primary_gates=original["primary_gates"],
        decision="All original workload non-regression and lifecycle gates; no new large-write speedup target for a lookup change",
        source_format=2,
        payload="4 KiB SQLite BLOB",
        owned_build_test_overlap=False,
        global_cache_eviction=False,
        frequency_affinity_changes=False,
        inputs=identities,
    )
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (out / "tests.json").write_bytes(args.validation.read_bytes())
    (out / "probe-build.json").write_text(json.dumps(dict(status="PASS", identities=identities), indent=2) + "\n")
    subprocess.run([sys.executable, str(runner), str(out), "--medium", "ext4"], check=True)
    assert all(digest(Path(p)) == h for p, h in identities.items())
    subprocess.run([sys.executable, str(analyzer), str(out)], check=True)
    result = json.loads((out / "summary.json").read_text())
    checks = [c for c in result["checks"] if c["workload"] != "combined_large_writes"]
    decision = dict(
        status="PASS" if all(c["pass_gate"] for c in checks) else "REJECT",
        checks=checks,
        excluded_check="The external-payload 10% large-write improvement target is not a lookup acceptance condition",
        identities_verified=True,
    )
    (out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print("Production lookup decision:", decision["status"], flush=True)


if __name__ == "__main__":
    main()
