# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare isolated component copies for the SQLite page-size experiment."""

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
        "checkpoint.hpp": "c662d344ec1c19c6daeed58d4e365241f2e6d237f6d7d3b6fca1554800e61159",
        "workspace.cpp": "cae97ec42ce136c902d6fda9b9d1c1cd9dc60844c1c6c993be11367d464fc02b",
    }
    for name, digest in expected.items():
        if hashlib.sha256((component / "src" / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Review/rebase the experiment patch: {name} has changed")
    args.output.mkdir(parents=True, exist_ok=False)
    for kib in (4, 8, 16, 32):
        target = args.output / f"page{kib}"
        target.mkdir()
        for name in ("CMakeLists.txt", "pyproject.toml", "README.md", "LICENSE"):
            shutil.copy2(component / name, target / name)
        for name in ("src", "include", "python", "tests", "LICENSES"):
            shutil.copytree(component / name, target / name, ignore=shutil.ignore_patterns("__pycache__"))
        subprocess.run(["patch", "--batch", "-p1", "-i", str(here / "layout.patch")], cwd=target, check=True)
        shutil.copy2(here / "test_layout.py", target / "tests/test_layout.py")
        print(target)


if __name__ == "__main__":
    main()
