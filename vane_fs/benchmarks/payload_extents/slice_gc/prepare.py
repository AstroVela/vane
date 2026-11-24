# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare isolated extent prototypes with pointer maps and live-slice GC."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New, empty experiment directory")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here.parent / "prepare.py"), str(args.output)], check=True)
    for variant in ("extent64", "extent256"):
        target = args.output / variant
        for patch in ("workspace.patch", "test-fixtures.patch"):
            subprocess.run(["patch", "--batch", "-p1", "-i", str(here / patch)], cwd=target, check=True)
        shutil.copy2(here / "test_slice_gc.py", target / "tests/test_slice_gc.py")


if __name__ == "__main__":
    main()
