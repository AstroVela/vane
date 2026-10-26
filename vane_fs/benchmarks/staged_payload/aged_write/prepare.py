# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the cached prototype with exact-interval version updates."""

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
    subprocess.run([sys.executable, str(here.parent / "random_io/prepare.py"), str(args.output)], check=True)
    target = args.output / "updated"
    shutil.copytree(args.output / "cached", target)
    subprocess.run(["patch", "--batch", "-p1", "-i", str(here / "update.patch")], cwd=target, check=True)
    shutil.copy2(here / "test_version_update.cpp", target / "tests/test_version_update.cpp")
    print(target)


if __name__ == "__main__":
    main()
