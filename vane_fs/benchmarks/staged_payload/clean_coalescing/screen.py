# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Replay the unchanged plain ext4 FUSE workload for an ablation or the original control."""

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
    parser.add_argument("--role", choices=("lookup-ablation", "clean-ablation", "original-control"), required=True)
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
    plan = dict(original)
    plan.update(
        status="FROZEN_BEFORE_PLAIN_COMPARISON",
        frozen_epoch=time.time(),
        role=args.role,
        baseline=str(paths["baseline"]),
        candidate=str(paths["updated"]),
        media=["ext4"],
        variant_gc_python_modules=modules,
        full_adoption_eligible=False,
        interpretation="Original ext4 subset and unchanged gates; ablations isolate mechanisms, not overall adoption",
        lookup_reads_enabled=True,
        small_clean_fixed_deferral_ms=5 if args.role != "lookup-ablation" else 0,
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
