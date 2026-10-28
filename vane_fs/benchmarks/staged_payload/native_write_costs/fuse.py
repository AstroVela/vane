# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Screen borrowed buffers with the unchanged ordinary FUSE workload and gates."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from prepare import digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--template", type=Path, required=True)
    args = parser.parse_args()
    out, template, here = args.artifacts.resolve(), args.template.resolve(), Path(__file__).resolve().parent
    validation = json.loads((out / "borrow-build.json").read_text())
    comparison = json.loads((out / "compare-summary.json").read_text())
    assert validation["status"] == "PASS" and comparison["screen_pass"]
    old = json.loads((template / "plan.json").read_text())
    assert digest(template / "run_fuse.py") == old["prepared_runner_sha256"]
    assert digest(template / "workload.py") == old["prepared_workload_sha256"]
    root = out / "fuse-screen"
    root.mkdir()
    base = json.loads((out / "build.json").read_text())
    components = {"baseline": Path(base["component"]["path"]), "updated": out / "borrowed"}
    identities = dict(
        base["component"]["sources"], **base["component"]["binaries"], **validation["sources"], **validation["binaries"]
    )
    identities.update(
        {
            str(p): digest(p)
            for p in [
                Path(__file__),
                out / "borrow-build.json",
                out / "compare-summary.json",
                template / "run_fuse.py",
                template / "workload.py",
            ]
        }
    )
    modules = {}
    for name, component in components.items():
        (root / name).symlink_to(component, target_is_directory=True)
        native = component / "venv/lib/python3.12/site-packages/vane_fs/_native.cpython-312-x86_64-linux-gnu.so"
        modules[name] = dict(path=str(native), sha256=digest(native))
    workload = root / "workload.py"
    workload.write_bytes((template / "workload.py").read_bytes())
    runner = (template / "run_fuse.py").read_text()
    replacements = {
        'sqlite_primitive="original SDK fsync",': 'sqlite_primitive="unchanged default SDK fsync",',
        'interpretation="tmpfs isolates software costs, not durable-storage throughput",': 'interpretation="Borrowed input buffers in matched format-1004 FUSE components on ext4; native CPU screen is separate",',
    }
    for before, after in replacements.items():
        assert runner.count(before) == 1
        runner = runner.replace(before, after)
    reverse = runner
    for before, after in replacements.items():
        reverse = reverse.replace(after, before)
    assert reverse.encode() == (template / "run_fuse.py").read_bytes()
    (root / "run_fuse.py").write_text(runner)
    identities.update({str(p): digest(p) for p in [workload, root / "run_fuse.py"]})
    assert all(digest(Path(p)) == sha for p, sha in identities.items())
    plan = dict(old)
    plan.update(
        status="FROZEN_BEFORE_PLAIN_COMPARISON",
        frozen_epoch=time.time(),
        role="borrowed-input-buffers",
        baseline=str(components["baseline"]),
        candidate=str(components["updated"]),
        media=["ext4"],
        variant_gc_python_modules=modules,
        full_adoption_eligible=False,
        report_labels_only_changed=True,
        template_runner_sha256=old["prepared_runner_sha256"],
        prepared_runner_sha256=digest(root / "run_fuse.py"),
        interpretation="CPU optimization on the isolated format-1004 prototype; retain original complete FUSE gates",
        only_difference="call-scoped block views and direct append of a complete contiguous input span",
        batch_blocks=64,
        external_threshold_bytes=262144,
        lookup_reads_enabled=True,
        small_clean_fixed_deferral_ms=5,
        inputs=identities,
    )
    save(root / "plan.json", plan)
    save(root / "build.json", dict(status="PASS", validated_identities=identities))
    save(root / "tests.json", validation)
    subprocess.run([sys.executable, str(root / "run_fuse.py"), str(root), "--medium", "ext4"], check=True)
    assert all(digest(Path(p)) == sha for p, sha in identities.items())
    save(root / "input-check.json", dict(status="PASS", checked=len(identities)))
    subprocess.run([sys.executable, str(here.parent / "partial_sync/analyze_writeback.py"), str(root)], check=True)


if __name__ == "__main__":
    main()
