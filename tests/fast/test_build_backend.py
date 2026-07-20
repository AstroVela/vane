# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import gzip
import io
import tarfile
from pathlib import Path

import pytest

import build_backend

ARCHIVE_TIMESTAMP = 1_667_997_441


def _stub_backend(monkeypatch, expected_directory: str, expected_settings, result: str):
    def build_sdist(directory, settings):
        assert directory == expected_directory
        assert settings == expected_settings
        return result

    monkeypatch.setattr(build_backend._backend, "build_sdist", build_sdist)


def _write_sdist(path: Path, files: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_archive:
        with gzip.GzipFile(
            filename=path.name.removesuffix(".gz"),
            mode="wb",
            compresslevel=9,
            fileobj=raw_archive,
            mtime=ARCHIVE_TIMESTAMP,
        ) as gzip_archive:
            with tarfile.open(fileobj=gzip_archive, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for name, data in sorted(files.items()):
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    member.mode = 0o644
                    member.mtime = ARCHIVE_TIMESTAMP
                    archive.addfile(member, io.BytesIO(data))


def _read_sdist_file(path: Path, suffix: str) -> tuple[list[str], bytes, tarfile.TarInfo]:
    with tarfile.open(path, mode="r:gz") as archive:
        names = archive.getnames()
        matches = [member for member in archive.getmembers() if member.name.endswith(suffix)]
        assert len(matches) == 1
        source_file = archive.extractfile(matches[0])
        assert source_file is not None
        return names, source_file.read(), matches[0]


def test_build_sdist_injects_git_tree_id_without_writing_checkout(tmp_path, monkeypatch):
    source_root = tmp_path / "read-only-source"
    (source_root / ".git").mkdir(parents=True)
    source_root.chmod(0o555)
    dist = tmp_path / "dist"
    filename = "backend-produced.tar.gz"
    archive_path = dist / filename
    source_id = "a" * 40
    _write_sdist(archive_path, {"project-1.0/pyproject.toml": b"[build-system]\n"})
    _stub_backend(monkeypatch, str(dist), None, filename)
    monkeypatch.setattr(build_backend, "REPOSITORY_ROOT", source_root)
    monkeypatch.setattr(build_backend, "SOURCE_ID_FILE", source_root / "DUCKDB_SOURCE_ID")
    monkeypatch.setattr(build_backend, "source_tree_id", lambda: source_id)

    try:
        result = build_backend.build_sdist(str(dist))
    finally:
        source_root.chmod(0o755)

    assert result == filename
    assert not (source_root / "DUCKDB_SOURCE_ID").exists()
    _, contents, member = _read_sdist_file(archive_path, "/DUCKDB_SOURCE_ID")
    assert contents == (source_id + "\n").encode("ascii")
    assert member.mode == 0o644
    assert member.mtime == ARCHIVE_TIMESTAMP


def test_build_sdist_reuses_manifest_without_git_metadata(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_id = "b" * 40
    manifest = source_root / "DUCKDB_SOURCE_ID"
    manifest.write_text(source_id + "\n", encoding="ascii")
    dist = tmp_path / "dist"
    filename = "backend-produced.tar.gz"
    archive_path = dist / filename
    _write_sdist(
        archive_path,
        {
            "project-1.0/DUCKDB_SOURCE_ID": (source_id + "\n").encode("ascii"),
            "project-1.0/pyproject.toml": b"[build-system]\n",
        },
    )
    original_archive = archive_path.read_bytes()
    _stub_backend(monkeypatch, str(dist), None, filename)
    monkeypatch.setattr(build_backend, "REPOSITORY_ROOT", source_root)
    monkeypatch.setattr(build_backend, "SOURCE_ID_FILE", manifest)
    monkeypatch.setattr(
        build_backend,
        "source_tree_id",
        lambda: pytest.fail("an sdist without Git metadata must reuse its manifest"),
    )

    result = build_backend.build_sdist(str(dist))

    assert result == filename
    assert archive_path.read_bytes() == original_archive


def test_build_sdist_derives_tree_id_without_git_metadata_or_manifest(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source_root.mkdir()
    dist = tmp_path / "dist"
    filename = "backend-produced.tar.gz"
    archive_path = dist / filename
    source_id = "c" * 40
    _write_sdist(archive_path, {"project-1.0/pyproject.toml": b"[build-system]\n"})
    _stub_backend(monkeypatch, str(dist), None, filename)
    monkeypatch.setattr(build_backend, "REPOSITORY_ROOT", source_root)
    monkeypatch.setattr(build_backend, "SOURCE_ID_FILE", source_root / "DUCKDB_SOURCE_ID")
    monkeypatch.setattr(build_backend, "source_tree_id", lambda: source_id)

    result = build_backend.build_sdist(str(dist))

    assert result == filename
    _, contents, _ = _read_sdist_file(archive_path, "/DUCKDB_SOURCE_ID")
    assert contents == (source_id + "\n").encode("ascii")


def test_source_id_injection_replaces_stale_manifest_once(tmp_path):
    archive_path = tmp_path / "project.tar.gz"
    current_source_id = "c" * 40
    _write_sdist(
        archive_path,
        {
            "project-1.0/DUCKDB_SOURCE_ID": ("d" * 40 + "\n").encode("ascii"),
            "project-1.0/pyproject.toml": b"[build-system]\n",
        },
    )

    build_backend._inject_source_id(archive_path, current_source_id)

    names, contents, _ = _read_sdist_file(archive_path, "/DUCKDB_SOURCE_ID")
    assert names.count("project-1.0/DUCKDB_SOURCE_ID") == 1
    assert contents == (current_source_id + "\n").encode("ascii")
