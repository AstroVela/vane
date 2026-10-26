# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Create disposable component source trees; never patch the working source."""

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New, empty experiment directory")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    component = here.parents[1]
    expected = "cae97ec42ce136c902d6fda9b9d1c1cd9dc60844c1c6c993be11367d464fc02b"
    if hashlib.sha256((component / "src/workspace.cpp").read_bytes()).hexdigest() != expected:
        raise RuntimeError("Prototype expects workspace.cpp from 82ce177d10; review/rebase the patch first")
    args.output.mkdir(parents=True, exist_ok=False)
    for variant in ("baseline", "extent64", "extent256"):
        target = args.output / variant
        target.mkdir()
        for name in ("CMakeLists.txt", "pyproject.toml", "README.md", "LICENSE"):
            shutil.copy2(component / name, target / name)
        for name in ("src", "include", "python", "tests", "LICENSES"):
            shutil.copytree(component / name, target / name, ignore=shutil.ignore_patterns("__pycache__"))
        if variant != "baseline":
            for patch in ("workspace.patch", "test-fixtures.patch"):
                subprocess.run(["patch", "--batch", "-p1", "-i", str(here / patch)], cwd=target, check=True)
            shutil.copy2(here / "test_extents.py", target / "tests/test_extents.py")
        print(target)


if __name__ == "__main__":
    main()
