# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare baseline, external, staged, and checkpoint-batched staged components."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New experiment directory")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    previous = here.parent / "external_payload"
    # Reuse the previous experiment's source-identity checks and exact inputs.
    # Its output must not exist; only its newly generated coalesced copy is
    # removed. That policy is not part of this experiment.
    subprocess.run([sys.executable, str(previous / "prepare.py"), str(args.output)], check=True)
    shutil.rmtree(args.output / "coalesced")
    target = args.output / "staged"
    shutil.copytree(args.output / "external", target)
    shutil.copy2(previous / "test_external.cpp", target / "tests/test_external_faults.cpp")
    for patch in ("storage.patch", "fixtures.patch"):
        subprocess.run(["patch", "--batch", "-p1", "-i", str(here / patch)], cwd=target, check=True)
    shutil.copy2(here / "staging.hpp", target / "src/staging.hpp")
    for name in ("test_staging.cpp", "test_staging.py"):
        shutil.copy2(here / name, target / "tests" / name)
    print(target)
    batched = args.output / "batched"
    shutil.copytree(target, batched)
    shutil.copy2(previous / "test_checkpoint_coalescing.cpp", batched / "tests/test_checkpoint_coalescing.cpp")
    subprocess.run(["patch", "--batch", "-p1", "-i", str(here / "batching.patch")], cwd=batched, check=True)
    print(batched)


if __name__ == "__main__":
    main()
