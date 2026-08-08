#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Resolve Vane's human-readable DuckDB fork version."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = "external/duckdb"
UPSTREAM_VERSION_FILE = REPOSITORY_ROOT / "DUCKDB_UPSTREAM_VERSION"
FORK_REVISION_FILE = REPOSITORY_ROOT / "DUCKDB_FORK_REVISION"
GIT_OBJECT_ID = r"(?:[0-9a-f]{40}|[0-9a-f]{64})"
FORK_REVISION = re.compile(rf"({GIT_OBJECT_ID})(-dirty)?")
UPSTREAM_VERSION = re.compile(r"v[0-9]+\.[0-9]+\.[0-9]+")
SHORT_REVISION_LENGTH = 10


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_bytes(*args: str) -> bytes:
    result = subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _git_top_level() -> Path | None:
    try:
        result = subprocess.run(
            ("git", "rev-parse", "--show-toplevel"),
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def _git_is_shallow() -> bool:
    value = _git("rev-parse", "--is-shallow-repository")
    if value not in {"true", "false"}:
        raise RuntimeError(f"Git returned an invalid shallow-repository state: {value!r}")
    return value == "true"


def validate_revision(revision: str) -> str:
    """Validate and return a full Git commit ID with an optional dirty marker."""
    if FORK_REVISION.fullmatch(revision) is None:
        raise ValueError(
            "invalid DuckDB fork revision: "
            f"{revision!r}; expected a 40- or 64-character lowercase commit ID, optionally followed by '-dirty'"
        )
    return revision


def validate_upstream_version(version: str) -> str:
    """Validate and return an exact upstream DuckDB release version."""
    if UPSTREAM_VERSION.fullmatch(version) is None:
        raise ValueError(f"invalid DuckDB upstream version: {version!r}; expected 'vX.Y.Z'")
    return version


def upstream_version() -> str:
    """Return the reviewed official DuckDB release used as Vane's baseline."""
    try:
        version = UPSTREAM_VERSION_FILE.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise RuntimeError(f"DuckDB upstream version is unavailable: {UPSTREAM_VERSION_FILE}") from exc
    return validate_upstream_version(version)


def _git_revision() -> str:
    commit_id = _git("rev-list", "-1", "HEAD", "--", SOURCE_DIRECTORY)
    if not commit_id:
        raise RuntimeError(f"Git history contains no commit for {SOURCE_DIRECTORY}")
    if re.fullmatch(GIT_OBJECT_ID, commit_id) is None:
        raise RuntimeError(f"Git returned an invalid DuckDB fork commit ID: {commit_id!r}")

    status = _git_bytes("status", "--porcelain=v1", "-z", "--untracked-files=all", "--", SOURCE_DIRECTORY)
    return validate_revision(commit_id + ("-dirty" if status else ""))


def source_revision() -> str:
    """Return the last DuckDB-changing commit and current engine dirty state."""
    if _git_top_level() == REPOSITORY_ROOT:
        if _git_is_shallow():
            raise RuntimeError(
                "DuckDB fork revision requires complete Git history; shallow repositories are unsupported"
            )
        return _git_revision()

    try:
        revision = FORK_REVISION_FILE.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise RuntimeError(
            f"DuckDB fork revision is unavailable without Git metadata: expected {FORK_REVISION_FILE}"
        ) from exc
    return validate_revision(revision)


def version_from_revision(revision: str, base_version: str | None = None) -> str:
    """Return the Vane fork version for a validated full revision."""
    match = FORK_REVISION.fullmatch(validate_revision(revision))
    assert match is not None
    version = upstream_version() if base_version is None else validate_upstream_version(base_version)
    short_commit = match.group(1)[:SHORT_REVISION_LENGTH]
    dirty_suffix = match.group(2) or ""
    return f"{version}-vane.{short_commit}{dirty_suffix}"


def _write_if_changed(path: Path, contents: str) -> None:
    actual = path.read_text(encoding="ascii") if path.exists() else ""
    if actual == contents:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as temporary_file:
            temporary_file.write(contents)
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_version_header(path: Path, revision: str, base_version: str | None = None) -> None:
    """Write a generated header that overrides DuckDB's runtime version."""
    version = version_from_revision(revision, base_version)
    contents = "\n".join(
        (
            "// Generated by scripts/resolve_duckdb_fork_version.py. Do not edit.",
            "#pragma once",
            "",
            "#ifdef DUCKDB_VERSION",
            "#undef DUCKDB_VERSION",
            "#endif",
            f'#define DUCKDB_VERSION "{version}"',
            "",
        )
    )
    _write_if_changed(path, contents)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-revision", action="store_true", help="print the full fork revision")
    mode.add_argument("--print-version", action="store_true", help="print the human-readable fork version")
    mode.add_argument("--header", type=Path, help="write a C++ header containing the fork version")
    parser.add_argument("--revision", help="use this full fork revision instead of resolving the source tree")
    parser.add_argument("--base-version", help="use this exact upstream release instead of the tracked in-tree base")
    args = parser.parse_args()

    revision = validate_revision(args.revision) if args.revision is not None else source_revision()
    if args.print_revision:
        print(revision)
    elif args.print_version:
        print(version_from_revision(revision, args.base_version))
    else:
        assert args.header is not None
        write_version_header(args.header, revision, args.base_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
