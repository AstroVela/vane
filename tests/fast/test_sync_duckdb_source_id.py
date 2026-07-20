# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import pytest

from scripts import sync_duckdb_source_id


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.mark.parametrize("object_format", ["sha1", "sha256"])
def test_filesystem_tree_id_matches_git_tree_encoding(tmp_path, monkeypatch, object_format):
    source = tmp_path / "external" / "duckdb"
    (source / "foo").mkdir(parents=True)
    (source / "foo" / "nested.txt").write_text("nested\n", encoding="ascii")
    (source / "foo.bar").write_text("prefix ordering\n", encoding="ascii")
    executable = source / "tool.sh"
    executable.write_text("#!/bin/sh\n", encoding="ascii")
    executable.chmod(0o755)
    group_executable = source / "group-executable.sh"
    group_executable.write_text("#!/bin/sh\n", encoding="ascii")
    group_executable.chmod(0o654)
    os.symlink("foo.bar", source / "link")

    _git(tmp_path, "init", "--quiet", f"--object-format={object_format}")
    _git(tmp_path, "config", "core.filemode", "true")
    _git(tmp_path, "add", "external/duckdb")
    repository_tree = _git(tmp_path, "write-tree")
    expected = _git(tmp_path, "rev-parse", f"{repository_tree}:external/duckdb")
    commit_length = 64 if object_format == "sha256" else 40
    (tmp_path / ".git_archival.txt").write_text(f"commit: {'a' * commit_length}\n", encoding="ascii")
    monkeypatch.setattr(sync_duckdb_source_id, "REPOSITORY_ROOT", tmp_path)

    assert sync_duckdb_source_id.filesystem_tree_id() == expected


def test_sha256_git_archive_preserves_object_format(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    source = repository / "external" / "duckdb"
    source.mkdir(parents=True)
    (source / "source.cpp").write_text("int answer = 42;\n", encoding="ascii")
    (repository / ".gitattributes").write_text("/.git_archival.txt export-subst\n", encoding="ascii")
    (repository / ".git_archival.txt").write_text("commit: $Format:%H$\n", encoding="ascii")
    _git(repository, "init", "--quiet", "--object-format=sha256")
    _git(repository, "add", ".")
    _git(
        repository,
        "-c",
        "user.name=Vane test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "archive fixture",
    )
    expected = _git(repository, "rev-parse", "HEAD:external/duckdb")
    exported = tmp_path / "exported"
    exported.mkdir()
    archive_path = tmp_path / "source.tar"
    with archive_path.open("wb") as archive_file:
        subprocess.run(
            ("git", "archive", "--format=tar", "HEAD"),
            cwd=repository,
            check=True,
            stdout=archive_file,
        )
    subprocess.run(("tar", "-xf", archive_path, "-C", exported), check=True)
    monkeypatch.setattr(sync_duckdb_source_id, "REPOSITORY_ROOT", exported)

    archive_metadata = (exported / ".git_archival.txt").read_text(encoding="ascii")
    assert archive_metadata.startswith("commit: ")
    assert len(archive_metadata.removeprefix("commit: ").strip()) == 64
    assert sync_duckdb_source_id.source_tree_id() == expected


@pytest.mark.parametrize(
    ("suppression_flag", "expected_prefix"),
    [("--assume-unchanged", "h "), ("--skip-worktree", "S ")],
)
def test_source_tree_id_clears_suppression_in_temporary_index(tmp_path, monkeypatch, suppression_flag, expected_prefix):
    source = tmp_path / "external" / "duckdb"
    source.mkdir(parents=True)
    tracked_file = source / "source.cpp"
    tracked_file.write_text("original\n", encoding="ascii")
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "add", "external/duckdb/source.cpp")
    _git(tmp_path, "update-index", suppression_flag, "external/duckdb/source.cpp")
    tracked_file.write_text("modified\n", encoding="ascii")
    monkeypatch.setattr(sync_duckdb_source_id, "REPOSITORY_ROOT", tmp_path)

    expected = sync_duckdb_source_id.filesystem_tree_id(source, object_format="sha1")

    assert sync_duckdb_source_id.source_tree_id() == expected
    assert _git(tmp_path, "ls-files", "-v", "external/duckdb/source.cpp").startswith(expected_prefix)


def test_source_id_header_overrides_engine_and_extension_versions(tmp_path):
    header = tmp_path / "generated" / "source_id.hpp"

    sync_duckdb_source_id.write_source_id_header(
        header,
        "a" * 40,
        ["EXT_VERSION_PARQUET", "EXT_VERSION_CORE_FUNCTIONS", "DUCKDB_SOURCE_ID"],
    )

    contents = header.read_text(encoding="ascii")
    assert contents.count('#define DUCKDB_SOURCE_ID "aaaaaaaaaa"') == 1
    assert '#define EXT_VERSION_CORE_FUNCTIONS "aaaaaaaaaa"' in contents
    assert '#define EXT_VERSION_PARQUET "aaaaaaaaaa"' in contents


def test_source_id_header_rejects_invalid_preprocessor_definitions(tmp_path):
    with pytest.raises(ValueError, match="invalid C preprocessor definitions"):
        sync_duckdb_source_id.write_source_id_header(tmp_path / "source_id.hpp", "a" * 40, ["INVALID-NAME"])
