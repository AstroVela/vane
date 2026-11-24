# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Replay the frozen FUSE workload with matched SQLite sync variants on ext4."""

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
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--updated", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    out, template = args.output.resolve(), args.template.resolve()
    assert json.loads(args.validation.read_text())["status"] == "PASS"
    original = json.loads((template / "plan.json").read_text())
    assert digest(template / "run_fuse.py") == original["prepared_runner_sha256"]
    assert digest(template / "workload.py") == original["prepared_workload_sha256"]
    out.mkdir()
    paths = {"baseline": args.baseline.resolve(), "updated": args.updated.resolve()}
    identities = {str(p): digest(p) for p in (Path(__file__).resolve(), args.validation.resolve())}
    build_path = Path(json.loads(args.validation.read_text())["build"])
    build = json.loads(build_path.read_text())
    assert build["status"] == "PASS" and build["core_static_libraries_identical"]
    identities.update(build["inputs"])
    identities[str(build_path)] = digest(build_path)
    assert all(digest(Path(p)) == sha for p, sha in identities.items())
    modules = {}
    for name, component in paths.items():
        (out / name).symlink_to(component, target_is_directory=True)
        native = component / "venv/lib/python3.12/site-packages/vane_fs/_native.cpython-312-x86_64-linux-gnu.so"
        modules[name] = dict(path=str(native.resolve()), sha256=digest(native))
        for path in (component / "build/python-release/vane-fs-mount", native):
            identities[str(path)] = digest(path)
        for folder in ("src", "include"):
            for path in (component / folder).rglob("*"):
                if path.is_file():
                    identities[str(path)] = digest(path)
    for filename in ("run_fuse.py", "workload.py"):
        (out / filename).write_bytes((template / filename).read_bytes())
        identities[str(out / filename)] = digest(out / filename)
    # Only stale report labels change; every workload and scheduling statement stays exact.
    runner = out / "run_fuse.py"
    text = runner.read_text()
    metadata = {
        'sqlite_primitive="original SDK fsync",': 'sqlite_primitive={"baseline": "fsync", "updated": "fdatasync"},',
        'interpretation="tmpfs isolates software costs, not durable-storage throughput",': 'interpretation="Matched SQLite primitives on ext4 with unchanged FULL barriers",',
    }
    for before, after in metadata.items():
        assert text.count(before) == 1
        text = text.replace(before, after)
    reversed_text = text
    for before, after in metadata.items():
        reversed_text = reversed_text.replace(after, before)
    assert reversed_text.encode() == (template / "run_fuse.py").read_bytes()
    runner.write_text(text)
    identities[str(runner)] = digest(runner)
    plan = dict(original)
    plan.update(
        status="FROZEN_BEFORE_PLAIN_COMPARISON",
        template_runner_sha256=original["prepared_runner_sha256"],
        prepared_runner_sha256=digest(runner),
        report_labels_only_changed=True,
        frozen_epoch=time.time(),
        role="sqlite-sync-primitive",
        sqlite_primitives={"baseline": "fsync", "updated": "fdatasync"},
        only_difference="SQLite HAVE_FDATASYNC build flag",
        core_and_background_worker_identical=True,
        gc_preparation="Each daemon's own non-editable wheel collects its independently prepared GC workspace",
        causal_comparison=dict(medium="ext4", pairs_per_state=6, baseline="SQLite fsync", candidate="SQLite fdatasync"),
        baseline=str(paths["baseline"]),
        candidate=str(paths["updated"]),
        media=["ext4"],
        variant_gc_python_modules=modules,
        full_adoption_eligible=False,
        interpretation="Optional SQLite primitive comparison on the isolated format-1004 prototype, not production adoption",
        lookup_reads_enabled=True,
        small_clean_fixed_deferral_ms=5,
        inputs=identities,
    )
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (out / "build.json").write_text(json.dumps(dict(status="PASS", validated_identities=identities), indent=2) + "\n")
    (out / "tests.json").write_bytes(args.validation.read_bytes())
    subprocess.run([sys.executable, str(out / "run_fuse.py"), str(out), "--medium", "ext4"], check=True)
    assert all(digest(Path(p)) == h for p, h in identities.items())
    (out / "input-check.json").write_text(json.dumps(dict(status="PASS", checked=len(identities)), indent=2) + "\n")
    subprocess.run([sys.executable, str(here.parent / "partial_sync/analyze_writeback.py"), str(out)], check=True)


if __name__ == "__main__":
    main()
