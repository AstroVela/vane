#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Compute DuckDB's content-derived SourceID and write build outputs."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = "external/duckdb"
ARCHIVE_METADATA_PATH = ".git_archival.txt"
GIT_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
ARCHIVE_COMMIT_ID = re.compile(r"^commit: ([0-9a-f]{40}|[0-9a-f]{64})$", re.MULTILINE)
C_PREPROCESSOR_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


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


def _git_bytes(*args: str, env: dict[str, str] | None = None) -> bytes:
    result = subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        env=env,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _git_path(name: str) -> Path:
    path = Path(_git("rev-parse", "--git-path", name))
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _clear_index_suppression(environment: dict[str, str]) -> None:
    tracked_paths = _git_bytes("ls-files", "-z", "--cached", "--", SOURCE_DIRECTORY, env=environment)
    if not tracked_paths:
        return
    # update-index applies only one flag operation per invocation.
    for flag in ("--no-assume-unchanged", "--no-skip-worktree"):
        subprocess.run(
            ("git", "update-index", flag, "-z", "--stdin"),
            cwd=REPOSITORY_ROOT,
            env=environment,
            input=tracked_paths,
            check=True,
            capture_output=True,
        )


def _write_tree(environment: dict[str, str], temporary_index: Path) -> str:
    """Write the worktree through a disposable index, with a safe fallback."""
    copied_index = False
    try:
        shutil.copyfile(_git_path("index"), temporary_index)
        copied_index = True
    except OSError:
        _git("read-tree", "HEAD", env=environment)

    try:
        if copied_index:
            _clear_index_suppression(environment)
        _git("add", "-A", "--", SOURCE_DIRECTORY, env=environment)
        return _git("write-tree", env=environment)
    except subprocess.CalledProcessError:
        if not copied_index:
            raise

    # A split or conflicted real index might not be reusable outside its Git
    # directory. Starting from HEAD is slower, but remains independent of the
    # developer's index and resolves conflicts outside the DuckDB subtree.
    temporary_index.unlink(missing_ok=True)
    _git("read-tree", "HEAD", env=environment)
    _git("add", "-A", "--", SOURCE_DIRECTORY, env=environment)
    return _git("write-tree", env=environment)


def _git_top_level() -> Path | None:
    """Return this checkout's Git root, or None when metadata is unavailable."""
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


def _new_hasher(object_format: str):
    if object_format not in {"sha1", "sha256"}:
        raise ValueError(f"unsupported Git object format: {object_format!r}")
    return hashlib.new(object_format, usedforsecurity=False)


def _object_hash(object_format: str, object_type: bytes, contents: bytes) -> bytes:
    hasher = _new_hasher(object_format)
    hasher.update(object_type + b" " + str(len(contents)).encode("ascii") + b"\0")
    hasher.update(contents)
    return hasher.digest()


def _file_hash(path: Path, size: int, object_format: str) -> bytes:
    hasher = _new_hasher(object_format)
    hasher.update(b"blob " + str(size).encode("ascii") + b"\0")
    bytes_read = 0
    with path.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            hasher.update(chunk)
            bytes_read += len(chunk)
    if bytes_read != size:
        raise RuntimeError(f"{path} changed while computing the DuckDB source tree ID")
    return hasher.digest()


def _filesystem_tree_hash(directory: Path, object_format: str) -> bytes:
    entries: list[tuple[bytes, bytes]] = []
    with os.scandir(directory) as children:
        for child in children:
            name = os.fsencode(child.name)
            path = Path(child.path)
            metadata = child.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                mode = b"120000"
                object_id = _object_hash(object_format, b"blob", os.fsencode(os.readlink(child.path)))
                sort_name = name
            elif stat.S_ISDIR(metadata.st_mode):
                mode = b"40000"
                object_id = _filesystem_tree_hash(path, object_format)
                sort_name = name + b"/"
            elif stat.S_ISREG(metadata.st_mode):
                mode = b"100755" if metadata.st_mode & stat.S_IXUSR else b"100644"
                object_id = _file_hash(path, metadata.st_size, object_format)
                sort_name = name
            else:
                raise RuntimeError(f"unsupported file type in DuckDB source tree: {path}")
            entries.append((sort_name, mode + b" " + name + b"\0" + object_id))

    contents = b"".join(entry for _, entry in sorted(entries))
    return _object_hash(object_format, b"tree", contents)


def _archive_object_format() -> str:
    metadata_path = REPOSITORY_ROOT / ARCHIVE_METADATA_PATH
    try:
        metadata = metadata_path.read_text(encoding="ascii")
    except OSError:
        return "sha1"
    match = ARCHIVE_COMMIT_ID.search(metadata)
    return "sha256" if match is not None and len(match.group(1)) == 64 else "sha1"


def filesystem_tree_id(source_path: Path | None = None, *, object_format: str | None = None) -> str:
    """Return a Git-compatible tree ID when repository metadata is absent."""
    source_path = REPOSITORY_ROOT / SOURCE_DIRECTORY if source_path is None else source_path
    if not source_path.is_dir():
        raise RuntimeError(f"DuckDB source directory is unavailable: {source_path}")
    object_format = _archive_object_format() if object_format is None else object_format
    return _filesystem_tree_hash(source_path, object_format).hex()


def source_tree_id() -> str:
    """Return the Git-compatible ID for the current DuckDB source tree."""
    top_level = _git_top_level()
    if top_level != REPOSITORY_ROOT:
        return filesystem_tree_id()

    # Use temporary index and object stores so staged, unstaged, and untracked
    # non-ignored DuckDB files all contribute without writing the checkout or
    # its Git metadata. The real object store remains available read-only.
    with tempfile.TemporaryDirectory(prefix="vane-duckdb-source-id-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        temporary_index = temporary_root / "index"
        temporary_objects = temporary_root / "objects"
        temporary_objects.mkdir()
        environment = os.environ.copy()
        environment["GIT_INDEX_FILE"] = str(temporary_index)
        environment["GIT_OBJECT_DIRECTORY"] = str(temporary_objects)
        real_objects = str(_git_path("objects"))
        existing_alternates = environment.get("GIT_ALTERNATE_OBJECT_DIRECTORIES")
        environment["GIT_ALTERNATE_OBJECT_DIRECTORIES"] = (
            real_objects if existing_alternates is None else real_objects + os.pathsep + existing_alternates
        )
        environment["GIT_OPTIONAL_LOCKS"] = "0"
        repository_tree = _write_tree(environment, temporary_index)
        tree_id = _git("rev-parse", f"{repository_tree}:{SOURCE_DIRECTORY}", env=environment)

    if GIT_OBJECT_ID.fullmatch(tree_id) is None:
        raise RuntimeError(f"Git returned an invalid DuckDB tree ID: {tree_id!r}")
    return tree_id


def validate_source_id(source_id: str) -> str:
    """Validate and return a full Git object ID."""
    if GIT_OBJECT_ID.fullmatch(source_id) is None:
        raise ValueError(f"invalid DuckDB source tree ID: {source_id!r}")
    return source_id


def _write_if_changed(path: Path, contents: str) -> None:
    """Atomically replace a generated file only when its contents changed."""
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


def write_source_id(path: Path, source_id: str) -> None:
    """Write a full SourceID manifest to an arbitrary build output path."""
    _write_if_changed(path, validate_source_id(source_id) + "\n")


def write_source_id_header(path: Path, source_id: str, definitions: Iterable[str] = ()) -> None:
    """Write build-time macros containing the short SourceID."""
    short_source_id = validate_source_id(source_id)[:10]
    extra_definitions = sorted(set(definitions) - {"DUCKDB_SOURCE_ID"})
    invalid_definitions = [
        definition for definition in extra_definitions if C_PREPROCESSOR_IDENTIFIER.fullmatch(definition) is None
    ]
    if invalid_definitions:
        raise ValueError(f"invalid C preprocessor definitions: {invalid_definitions!r}")

    lines = ["// Generated by scripts/sync_duckdb_source_id.py. Do not edit.", "#pragma once", ""]
    for definition in ("DUCKDB_SOURCE_ID", *extra_definitions):
        lines.extend(
            (
                f"#ifdef {definition}",
                f"#undef {definition}",
                "#endif",
                f'#define {definition} "{short_source_id}"',
                "",
            )
        )
    contents = "\n".join(lines)
    _write_if_changed(path, contents)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--print", action="store_true", dest="print_only", help="print the full tree ID")
    mode.add_argument("--output", type=Path, help="write the full tree ID to this build output")
    mode.add_argument("--header", type=Path, help="write a C++ header containing the short SourceID")
    parser.add_argument("--source-id", help="use this full tree ID instead of computing it from Git")
    parser.add_argument(
        "--define",
        action="append",
        default=[],
        dest="definitions",
        metavar="MACRO",
        help="also define this C preprocessor macro to the short SourceID",
    )
    args = parser.parse_args()

    if args.definitions and args.header is None:
        parser.error("--define requires --header")

    source_id = validate_source_id(args.source_id) if args.source_id is not None else source_tree_id()

    if args.output is not None:
        write_source_id(args.output, source_id)
    elif args.header is not None:
        write_source_id_header(args.header, source_id, args.definitions)
    else:
        print(source_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
