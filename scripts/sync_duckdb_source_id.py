#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Synchronize DuckDB's recorded SourceID with its Git tree object."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = "external/duckdb"
SOURCE_ID_FILE = REPOSITORY_ROOT / "DUCKDB_SOURCE_ID"
GIT_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


def _git(*args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def source_tree_id() -> str:
    """Return the Git tree ID for the current DuckDB working tree."""
    top_level = Path(_git("rev-parse", "--show-toplevel")).resolve()
    if top_level != REPOSITORY_ROOT:
        raise RuntimeError(f"expected Git root {REPOSITORY_ROOT}, found {top_level}")

    # Use a temporary index so staged, unstaged, and untracked non-ignored
    # DuckDB files all contribute without changing the developer's real index.
    with tempfile.TemporaryDirectory(prefix="vane-duckdb-source-id-") as temporary_directory:
        temporary_index = Path(temporary_directory) / "index"
        environment = os.environ.copy()
        environment["GIT_INDEX_FILE"] = str(temporary_index)
        _git("read-tree", "HEAD", env=environment)
        _git("add", "-A", "--", SOURCE_DIRECTORY, env=environment)
        repository_tree = _git("write-tree", env=environment)
        tree_id = _git("rev-parse", f"{repository_tree}:{SOURCE_DIRECTORY}", env=environment)

    if GIT_OBJECT_ID.fullmatch(tree_id) is None:
        raise RuntimeError(f"Git returned an invalid DuckDB tree ID: {tree_id!r}")
    return tree_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail instead of rewriting an out-of-date SourceID")
    mode.add_argument("--print", action="store_true", dest="print_only", help="print the current tree ID")
    args = parser.parse_args()

    expected = source_tree_id() + "\n"
    if args.print_only:
        print(expected, end="")
        return 0

    actual = SOURCE_ID_FILE.read_text(encoding="utf-8") if SOURCE_ID_FILE.exists() else ""
    if args.check:
        if actual != expected:
            print(f"{SOURCE_ID_FILE} is missing or out of date")
            return 1
        print(f"{SOURCE_ID_FILE} is up to date")
        return 0

    SOURCE_ID_FILE.write_text(expected, encoding="utf-8")
    print(f"wrote {SOURCE_ID_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
