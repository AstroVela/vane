# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare selective pre-OPEN backfill with the unchanged 5 ms CLEAN policy."""

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
    args = parser.parse_args()
    out, here = args.artifacts.resolve(), Path(__file__).resolve().parent
    candidate = json.loads((out / "candidate.json").read_text())
    assert candidate["status"] == json.loads((out / "tests.json").read_text())["status"] == "PASS"
    previous, updated = Path(candidate["parent"]) / "coalesced", out / "updated"
    identities = {
        str(path): digest(path)
        for path in (Path(__file__), here / "prepare.patch", out / "candidate.json", out / "tests.json")
    }
    for component in (previous, updated):
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
    frozen = dict(
        status="FROZEN_BEFORE_COMPARISON",
        epoch=time.time(),
        baseline=str(previous),
        candidate=str(updated),
        difference="One PASSIVE attempt for a large CLEAN prefix before a potentially external write",
        background_worker_identical=True,
        frozen_policy=candidate["frozen_policy"],
        unchanged=dict(original_workload=True, original_gates=True, pairs_per_state=6, medium="ext4"),
        adoption_eligible=False,
        inputs=identities,
    )
    (out / "comparison.json").write_text(json.dumps(frozen, indent=2) + "\n")
    runner = here.parent / "clean_coalescing/screen.py"
    subprocess.run(
        [
            sys.executable,
            str(runner),
            str(out / "plain"),
            "--template",
            str(args.template.resolve()),
            "--validation",
            str(out / "tests.json"),
            "--baseline",
            str(previous),
            "--updated",
            str(updated),
            "--role",
            "clean-ablation",
        ],
        check=True,
    )
    assert all(digest(Path(path)) == sha for path, sha in identities.items())


if __name__ == "__main__":
    main()
