# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare isolated OPEN-only checkpoint and selective-publication candidates."""

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
    subprocess.run([sys.executable, str(here.parent / "prepare.py"), str(args.output)], check=True)
    conditional = args.output / "conditional"
    shutil.copytree(args.output / "batched", conditional)
    subprocess.run(["patch", "--batch", "-p1", "-i", str(here / "checkpoint.patch")], cwd=conditional, check=True)
    shutil.copy2(here / "test_checkpoint_coalescing.cpp", conditional / "tests/test_checkpoint_coalescing.cpp")
    selective = args.output / "selective"
    shutil.copytree(conditional, selective)
    subprocess.run(["patch", "--batch", "-p1", "-i", str(here / "inline.patch")], cwd=selective, check=True)
    shutil.copy2(here / "test_inline_publication.cpp", selective / "tests/test_inline_publication.cpp")
    print(conditional)
    print(selective)


if __name__ == "__main__":
    main()
