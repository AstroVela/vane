# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Use the original workload, checking each variant's schema outside measured phases."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def prepared(plan):
    acceptance = Path(__file__).resolve().parents[2] / "partial_write/acceptance"
    runner_source, workload_source = acceptance / "run_fuse.py", acceptance / "workload.py"
    assert hashlib.sha256(runner_source.read_bytes()).hexdigest() == plan["runner_source_sha256"]
    assert hashlib.sha256(workload_source.read_bytes()).hexdigest() == plan["workload_source_sha256"]
    runner = runner_source.read_text().replace(
        'assert json.loads((out / "probe-build.json").read_text())["status"] == "PASS"',
        'assert json.loads((out / "build.json").read_text())["status"] == "PASS"',
    )
    runner = runner.replace("here = Path(__file__).resolve().parent", "here = Path(" + repr(str(acceptance)) + ")")
    runner = runner.replace('here / "workload.py"', 'out / "workload.py"')
    runner = runner.replace("external_threshold_bytes=None,", "external_threshold_bytes=262144,")
    runner = runner.replace("format_version=2,", "format_version=1004,")
    runner = runner.replace(
        "schema == [(2, 4096)] and all(n == 4096 for n, _ in lengths)",
        "schema == [(1004, 4096)] and all(n in (8, 4096) for n, _ in lengths)",
    )
    original_hash = hashlib.sha256(runner.encode()).hexdigest()
    needle = '            return {\n                "quick_check": quick,'
    assert runner.count(needle) == 1
    runner = runner.replace(
        needle,
        """            rows = m.REPORT["fuse"] if state == "fresh" else m.REPORT["aged_fuse"]
            present = bool(conn.execute("SELECT count(*) FROM sqlite_schema WHERE type='index' AND name='block_versions_high'").fetchone()[0])
            assert present, "Original block index missing during GC/inspection"
            return {
                "block_high_index_present": present,
                "quick_check": quick,""",
    )
    marker = "    owned = []\n"
    assert runner.count(marker) == 1
    runner = runner.replace(
        marker,
        """    binding_plan = json.loads((out / "plan.json").read_text())["variant_gc_python_modules"]
    observed_modules = {}
    for name in m.VARIANTS:
        code = "import json,hashlib,vane_fs._native as n;print(json.dumps({'path':n.__file__,'sha256':hashlib.sha256(open(n.__file__,'rb').read()).hexdigest()}))"
        result = subprocess.check_output([str(out / name / "venv/bin/python"), "-I", "-c", code], text=True)
        observed = json.loads(result)
        assert observed["sha256"] == binding_plan[name]["sha256"]
        assert Path(observed["path"]).resolve() == Path(binding_plan[name]["path"]).resolve()
        observed_modules[name] = observed
    m.REPORT["config"]["gc_python_modules"] = observed_modules
"""
        + marker,
    )
    workload = workload_source.read_text().replace(
        'assert not Path(str(db) + "-wal").exists() and not Path(str(db) + "-shm").exists()',
        'assert Path(str(db) + ".stage-fence").read_bytes()[40:48] == bytes(8)',
    )
    return runner, workload, original_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--medium", choices=("tmpfs", "ext4"), required=True)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    plan = json.loads((out / "plan.json").read_text())
    runner, workload, original = prepared(plan)
    assert original == plan["original_prepared_runner_sha256"]
    assert hashlib.sha256(runner.encode()).hexdigest() == plan["prepared_runner_sha256"]
    assert hashlib.sha256(workload.encode()).hexdigest() == plan["prepared_workload_sha256"]
    (out / "run_fuse.py").write_text(runner)
    (out / "workload.py").write_text(workload)
    subprocess.run([sys.executable, str(out / "run_fuse.py"), str(out), "--medium", args.medium], check=True)


if __name__ == "__main__":
    main()
