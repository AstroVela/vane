# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the selective prototype with cached transaction-control statements."""

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
    subprocess.run([sys.executable, str(here.parent / "conditional/prepare.py"), str(args.output)], check=True)
    target = args.output / "cached"
    shutil.copytree(args.output / "selective", target)
    subprocess.run(["patch", "--batch", "-p1", "-i", str(here / "control.patch")], cwd=target, check=True)
    shutil.copy2(here / "test_transaction_cache.cpp", target / "tests/test_transaction_cache.cpp")
    print(target)


if __name__ == "__main__":
    main()
