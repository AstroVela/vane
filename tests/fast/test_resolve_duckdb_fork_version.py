# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import resolve_duckdb_fork_version


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(repository: Path, message: str) -> str:
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
        message,
    )
    return _git(repository, "rev-parse", "HEAD")


def _repository(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    source = tmp_path / "external" / "duckdb"
    source.mkdir(parents=True)
    (source / "source.cpp").write_text("int answer = 42;\n", encoding="ascii")
    upstream_version = tmp_path / "DUCKDB_UPSTREAM_VERSION"
    upstream_version.write_text("v1.5.0\n", encoding="ascii")
    _git(tmp_path, "init", "--quiet")
    _commit(tmp_path, "initial engine")
    monkeypatch.setattr(resolve_duckdb_fork_version, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(resolve_duckdb_fork_version, "UPSTREAM_VERSION_FILE", upstream_version)
    monkeypatch.setattr(resolve_duckdb_fork_version, "FORK_REVISION_FILE", tmp_path / "DUCKDB_FORK_REVISION")
    return tmp_path, source


def _custom_source_cmake_probe(tmp_path: Path, upstream_version: str | None) -> subprocess.CompletedProcess[str]:
    custom_source = tmp_path / "custom-duckdb"
    custom_source.mkdir()
    (custom_source / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.29)\n", encoding="ascii")
    probe = tmp_path / "probe.cmake"
    probe.write_text(
        """
set(PROJECT_SOURCE_DIR "${VANE_TEST_REPOSITORY}")
set(DUCKDB_SOURCE_PATH "${VANE_TEST_DUCKDB_SOURCE}")
set(VANE_DUCKDB_SOURCE_ID "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
set(VANE_DUCKDB_FORK_REVISION "ffffffffffffffffffffffffffffffffffffffff")
if(DEFINED VANE_TEST_UPSTREAM_VERSION)
  set(VANE_DUCKDB_UPSTREAM_VERSION "${VANE_TEST_UPSTREAM_VERSION}")
endif()
include("${PROJECT_SOURCE_DIR}/cmake/duckdb_loader.cmake")
_duckdb_validate_source_path()
_duckdb_resolve_source_id()
_duckdb_resolve_fork_version()
message(STATUS "fork-version=${VANE_DUCKDB_FORK_VERSION}")
""".lstrip(),
        encoding="ascii",
    )
    command = [
        "cmake",
        f"-DVANE_TEST_REPOSITORY={resolve_duckdb_fork_version.REPOSITORY_ROOT}",
        f"-DVANE_TEST_DUCKDB_SOURCE={custom_source}",
    ]
    if upstream_version is not None:
        command.append(f"-DVANE_TEST_UPSTREAM_VERSION={upstream_version}")
    command.extend(("-P", str(probe)))
    return subprocess.run(command, capture_output=True, text=True, check=False)


def test_revision_tracks_the_last_engine_commit_not_repository_head(tmp_path, monkeypatch):
    repository, _ = _repository(tmp_path, monkeypatch)
    engine_commit = _git(repository, "rev-parse", "HEAD")
    (repository / "README.md").write_text("root-only change\n", encoding="ascii")
    root_commit = _commit(repository, "root-only change")

    assert root_commit != engine_commit
    assert resolve_duckdb_fork_version.source_revision() == engine_commit
    assert resolve_duckdb_fork_version.version_from_revision(engine_commit) == f"v1.5.0-vane.{engine_commit[:10]}"


def test_revision_marks_tracked_and_untracked_engine_changes_dirty(tmp_path, monkeypatch):
    repository, source = _repository(tmp_path, monkeypatch)
    engine_commit = _git(repository, "rev-parse", "HEAD")
    source_file = source / "source.cpp"
    source_file.write_text("int answer = 43;\n", encoding="ascii")

    assert resolve_duckdb_fork_version.source_revision() == f"{engine_commit}-dirty"

    _git(repository, "checkout", "--", "external/duckdb/source.cpp")
    (source / "new.cpp").write_text("int added = 1;\n", encoding="ascii")

    assert resolve_duckdb_fork_version.source_revision() == f"{engine_commit}-dirty"


def test_revision_marks_mode_only_engine_changes_dirty(tmp_path, monkeypatch):
    repository, source = _repository(tmp_path, monkeypatch)
    engine_commit = _git(repository, "rev-parse", "HEAD")
    (source / "source.cpp").chmod(0o755)

    assert resolve_duckdb_fork_version.source_revision() == f"{engine_commit}-dirty"


def test_revision_changes_after_an_engine_commit(tmp_path, monkeypatch):
    repository, source = _repository(tmp_path, monkeypatch)
    (source / "source.cpp").write_text("int answer = 43;\n", encoding="ascii")
    engine_commit = _commit(repository, "change engine")

    assert resolve_duckdb_fork_version.source_revision() == engine_commit


def test_shallow_repository_is_rejected(tmp_path, monkeypatch):
    source_repository = tmp_path / "source"
    source_repository.mkdir()
    _repository(source_repository, monkeypatch)
    (source_repository / "README.md").write_text("root-only change\n", encoding="ascii")
    _commit(source_repository, "root-only change")
    shallow_repository = tmp_path / "shallow"
    subprocess.run(
        (
            "git",
            "clone",
            "--quiet",
            "--depth=1",
            "--no-local",
            str(source_repository),
            str(shallow_repository),
        ),
        check=True,
    )
    monkeypatch.setattr(resolve_duckdb_fork_version, "REPOSITORY_ROOT", shallow_repository)
    monkeypatch.setattr(
        resolve_duckdb_fork_version,
        "FORK_REVISION_FILE",
        shallow_repository / "DUCKDB_FORK_REVISION",
    )

    with pytest.raises(RuntimeError, match="complete Git history"):
        resolve_duckdb_fork_version.source_revision()


def test_non_engine_worktree_changes_do_not_mark_the_revision_dirty(tmp_path, monkeypatch):
    repository, _ = _repository(tmp_path, monkeypatch)
    engine_commit = _git(repository, "rev-parse", "HEAD")
    (repository / "untracked.txt").write_text("outside engine\n", encoding="ascii")

    assert resolve_duckdb_fork_version.source_revision() == engine_commit


def test_source_archive_requires_and_reuses_injected_revision(tmp_path, monkeypatch):
    source = tmp_path / "external" / "duckdb"
    source.mkdir(parents=True)
    revision = "a" * 40
    manifest = tmp_path / "DUCKDB_FORK_REVISION"
    manifest.write_text(revision + "-dirty\n", encoding="ascii")
    monkeypatch.setattr(resolve_duckdb_fork_version, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(resolve_duckdb_fork_version, "FORK_REVISION_FILE", manifest)

    assert resolve_duckdb_fork_version.source_revision() == revision + "-dirty"


def test_source_archive_without_revision_fails(tmp_path, monkeypatch):
    (tmp_path / "external" / "duckdb").mkdir(parents=True)
    monkeypatch.setattr(resolve_duckdb_fork_version, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(resolve_duckdb_fork_version, "FORK_REVISION_FILE", tmp_path / "DUCKDB_FORK_REVISION")

    with pytest.raises(RuntimeError, match="unavailable without Git metadata"):
        resolve_duckdb_fork_version.source_revision()


@pytest.mark.parametrize(
    "revision",
    ["abc", "A" * 40, "a" * 39, "a" * 41, "a" * 40 + "-modified"],
)
def test_invalid_revision_is_rejected(revision):
    with pytest.raises(ValueError, match="invalid DuckDB fork revision"):
        resolve_duckdb_fork_version.validate_revision(revision)


def test_version_header_overrides_duckdb_version(tmp_path, monkeypatch):
    upstream_version = tmp_path / "DUCKDB_UPSTREAM_VERSION"
    upstream_version.write_text("v1.5.0\n", encoding="ascii")
    monkeypatch.setattr(resolve_duckdb_fork_version, "UPSTREAM_VERSION_FILE", upstream_version)
    header = tmp_path / "generated" / "version.hpp"
    revision = "b" * 40 + "-dirty"

    resolve_duckdb_fork_version.write_version_header(header, revision)

    assert header.read_text(encoding="ascii").endswith('#define DUCKDB_VERSION "v1.5.0-vane.bbbbbbbbbb-dirty"\n')


def test_cli_prints_an_explicit_upstream_version():
    revision = "c" * 40

    result = subprocess.run(
        [
            sys.executable,
            str(Path(resolve_duckdb_fork_version.__file__).resolve()),
            "--print-version",
            "--revision",
            revision,
            "--base-version",
            "v1.6.0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "v1.6.0-vane.cccccccccc"


def test_cli_writes_a_header_with_an_explicit_upstream_version(tmp_path):
    revision = "d" * 40
    header = tmp_path / "version.hpp"

    subprocess.run(
        [
            sys.executable,
            str(Path(resolve_duckdb_fork_version.__file__).resolve()),
            "--header",
            str(header),
            "--revision",
            revision,
            "--base-version",
            "v1.6.0",
        ],
        check=True,
    )

    assert header.read_text(encoding="ascii").endswith('#define DUCKDB_VERSION "v1.6.0-vane.dddddddddd"\n')


def test_custom_cmake_source_uses_its_explicit_upstream_version(tmp_path):
    result = _custom_source_cmake_probe(tmp_path, "v1.6.0")

    assert result.returncode == 0, result.stderr
    assert "fork-version=v1.6.0-vane.ffffffffff" in result.stdout


def test_custom_cmake_source_requires_an_explicit_upstream_version(tmp_path):
    result = _custom_source_cmake_probe(tmp_path, None)

    assert result.returncode != 0
    assert "requires an explicit" in result.stderr
    assert "VANE_DUCKDB_UPSTREAM_VERSION" in result.stderr
