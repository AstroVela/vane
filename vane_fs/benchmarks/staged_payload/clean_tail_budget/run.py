# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Freeze both controls and run independent unchanged plain scheduling comparisons."""

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
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--generation-control", type=Path, required=True)
    parser.add_argument("--window", choices=("whole-generation", "tail"), required=True)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    candidate = json.loads((out / "candidate.json").read_text())
    assert candidate["status"] == "PASS"
    assert json.loads((out / "tests.json").read_text())["status"] == "PASS"
    components = dict(
        candidate=out / "updated",
        generation=args.generation_control.resolve(),
        original=Path(candidate["parent"]) / "coalesced",
    )
    identities = {
        str(p): digest(p) for p in (Path(__file__), here / "budget.patch", out / "candidate.json", out / "tests.json")
    }
    for component in components.values():
        for folder in ("src", "include"):
            for path in (component / folder).rglob("*"):
                if path.is_file():
                    identities[str(path)] = digest(path)
        for path in (
            component / "CMakeLists.txt",
            component / "pyproject.toml",
            component / "build/python-release/vane-fs-mount",
            component / "venv/lib/python3.12/site-packages/vane_fs/_native.cpython-312-x86_64-linux-gnu.so",
        ):
            identities[str(path)] = digest(path)
    frozen_path = out / "comparisons.json"
    frozen = dict(
        status="FROZEN_BEFORE_COMPARISONS",
        epoch=time.time(),
        components={n: str(p) for n, p in components.items()},
        windows={
            "whole-generation": dict(
                control="generation",
                difference="Remove the entire-generation restriction; retain the 256 KiB dirty-tail cap",
            ),
            "tail": dict(
                control="original",
                difference="Add only the 256 KiB dirty-tail cap; retain coalescing in large generations",
            ),
        },
        unchanged=dict(
            clean_fixed_ms=5,
            open_bytes=16 * 1024 * 1024,
            admission_bytes=64 * 1024 * 1024,
            original_workload=True,
            original_gates=True,
            pairs_per_state=6,
            medium="ext4",
        ),
        adoption_eligible=False,
        inputs=identities,
    )
    if frozen_path.exists():
        previous = json.loads(frozen_path.read_text())
        assert previous["components"] == frozen["components"] and previous["inputs"] == identities
    else:
        frozen_path.write_text(json.dumps(frozen, indent=2) + "\n")
    baseline = components["generation" if args.window == "whole-generation" else "original"]
    runner = here.parent / "clean_coalescing/screen.py"
    subprocess.run(
        [
            sys.executable,
            str(runner),
            str(out / args.window),
            "--template",
            str(args.template.resolve()),
            "--validation",
            str(out / "tests.json"),
            "--baseline",
            str(baseline),
            "--updated",
            str(components["candidate"]),
            "--role",
            "clean-ablation",
        ],
        check=True,
    )
    assert all(digest(Path(p)) == h for p, h in identities.items())


if __name__ == "__main__":
    main()
