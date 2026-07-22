#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reject accidental compiled binaries and oversized files in the repository."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path, PurePosixPath

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_FILE_BYTES = 5 * 1024 * 1024

EXECUTABLE_BINARY_ALLOWLIST: frozenset[str] = frozenset()
LARGE_FILE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "external/duckdb/data/csv/sequences.csv.gz",
        "external/duckdb/data/parquet-testing/issue12621.parquet",
        "external/duckdb/data/parquet-testing/leftdate3_192_loop_1.parquet",
        "external/duckdb/extension/icu/third_party/icu/stubdata/stubdata.cpp",
        "external/duckdb/extension/tpcds/dsdgen/include/tpcds_constants.hpp",
        "external/duckdb/third_party/libpg_query/grammar/grammar_out.output",
    }
)

MACHO_MAGICS = {
    b"\xca\xfe\xba\xbe",
    b"\xca\xfe\xba\xbf",
    b"\xbe\xba\xfe\xca",
    b"\xbf\xba\xfe\xca",
    b"\xfe\xed\xfa\xce",
    b"\xfe\xed\xfa\xcf",
    b"\xce\xfa\xed\xfe",
    b"\xcf\xfa\xed\xfe",
}


def _normalize(path: str) -> str:
    return PurePosixPath(path).as_posix()


def tracked_files(repository_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return [_normalize(path.decode("utf-8", errors="surrogateescape")) for path in result.stdout.split(b"\0") if path]


def executable_binary_kind(data: bytes) -> str | None:
    if data.startswith(b"\x7fELF"):
        return "ELF"
    if data.startswith(b"MZ"):
        return "PE/COFF"
    if data[:4] in MACHO_MAGICS:
        return "Mach-O"
    return None


def check_repository(repository_root: Path = REPOSITORY_ROOT) -> list[str]:
    errors: list[str] = []
    for relative_path in tracked_files(repository_root):
        path = repository_root / relative_path
        if not path.is_file() or path.is_symlink():
            continue

        size = path.stat().st_size
        with path.open("rb") as source_file:
            kind = executable_binary_kind(source_file.read(4))
        if kind is not None and relative_path not in EXECUTABLE_BINARY_ALLOWLIST:
            errors.append(
                f"{relative_path}: tracked {kind} executable binary is not allowed; "
                "keep reproducible source/build instructions instead"
            )
        if size > MAX_TRACKED_FILE_BYTES and relative_path not in LARGE_FILE_ALLOWLIST:
            errors.append(
                f"{relative_path}: tracked file is {size} bytes, above {MAX_TRACKED_FILE_BYTES}; "
                "add a narrow allowlist entry only for intentional fixtures"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT, help="repository root to check")
    args = parser.parse_args()

    errors = check_repository(args.root.resolve())
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("repository binary and large-file checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
