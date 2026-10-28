# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare isolated baseline and Linux external-payload prototype components."""

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New experiment directory")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    component = here.parents[1]
    expected = {
        "src/workspace.cpp": "cae97ec42ce136c902d6fda9b9d1c1cd9dc60844c1c6c993be11367d464fc02b",
        "src/checkpoint.hpp": "c662d344ec1c19c6daeed58d4e365241f2e6d237f6d7d3b6fca1554800e61159",
        "tests/test_checkpoint.cpp": "cd298919190e5e784c17e554dc7ed540dcc226ffe5d006a171f91156e908148e",
        "tests/test_durability.cpp": "9c3c0e30aaa4f8997ce8aaf995f0f32166f0efd3d54f38d2118f02d90f84ce80",
        "tests/test_recovery.py": "0f9dc3f5723fcd77b13a5c33e033fa24966dd6b1103914fd900c26a4870e76a0",
    }
    for name, digest in expected.items():
        if hashlib.sha256((component / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Review/rebase the experiment patches: {name} has changed")
    args.output.mkdir(parents=True, exist_ok=False)
    for variant in ("baseline", "external", "coalesced"):
        target = args.output / variant
        target.mkdir()
        for name in ("CMakeLists.txt", "pyproject.toml", "README.md", "LICENSE"):
            shutil.copy2(component / name, target / name)
        for name in ("src", "include", "python", "tests", "LICENSES"):
            shutil.copytree(component / name, target / name, ignore=shutil.ignore_patterns("__pycache__"))
        if variant != "baseline":
            for patch in ("storage.patch", "fixtures.patch"):
                subprocess.run(["patch", "--batch", "-p1", "-i", str(here / patch)], cwd=target, check=True)
            shutil.copy2(here / "payload_file.hpp", target / "src/payload_file.hpp")
            shutil.copy2(here / "test_external.py", target / "tests/test_external.py")
        if variant == "coalesced":
            subprocess.run(["patch", "--batch", "-p1", "-i", str(here / "coalescing.patch")], cwd=target, check=True)
            shutil.copy2(here / "test_checkpoint_coalescing.cpp", target / "tests/test_checkpoint_coalescing.cpp")
        print(target)


if __name__ == "__main__":
    main()
