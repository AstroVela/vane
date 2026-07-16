#!/usr/bin/env python3
"""Validate Vane source and wheel artifacts before publication."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import stat
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Protocol

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_METADATA = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
EXPECTED_NAME = str(PROJECT_METADATA["name"])
EXPECTED_VERSION = str(PROJECT_METADATA["version"])
MAX_ARTIFACT_BYTES = 100 * 1024 * 1024

BANNED_PATH_PARTS = (
    "/.git/",
    "/.venv",
    "/build/",
    "/dist/",
    "/external/duckdb/extension/tpcds/",
    "/external/duckdb/extension/tpch/",
    "/external/duckdb/third_party/tpce-tool/",
    "/vane/session_",
    "/vcpkg_installed/",
)
BANNED_PATH_SUFFIXES = (
    ".log",
    "/ray_data_main_old.py",
)
BANNED_CONTENT = (
    b"192.168." + b"1.28",
    b"StrongPass" + b"123!",
)
BANNED_TEXT_CONTENT = (b"/home/" + b"kaka/",)


class Artifact(Protocol):
    path: Path

    def path_names(self) -> list[str]: ...

    def names(self) -> list[str]: ...

    def read(self, name: str) -> bytes: ...


class WheelArtifact:
    def __init__(self, path: Path):
        self.path = path
        self.archive = zipfile.ZipFile(path)
        for info in self.archive.infolist():
            if stat.S_ISLNK(info.external_attr >> 16):
                self.archive.close()
                raise ValueError(f"{path}: symbolic links are not allowed in wheels: {info.filename}")

    def path_names(self) -> list[str]:
        return self.archive.namelist()

    def names(self) -> list[str]:
        return self.archive.namelist()

    def read(self, name: str) -> bytes:
        return self.archive.read(name)

    def close(self) -> None:
        self.archive.close()


class SdistArtifact:
    def __init__(self, path: Path):
        self.path = path
        self.archive = tarfile.open(path, mode="r:gz")
        self.all_members = self.archive.getmembers()
        for member in self.all_members:
            if not member.isfile() and not member.isdir():
                self.archive.close()
                raise ValueError(f"{path}: unsupported tar member type for {member.name}")
        self.members = {member.name: member for member in self.all_members if member.isfile()}

    def path_names(self) -> list[str]:
        return [member.name for member in self.all_members]

    def names(self) -> list[str]:
        return list(self.members)

    def read(self, name: str) -> bytes:
        extracted = self.archive.extractfile(self.members[name])
        if extracted is None:
            raise ValueError(f"could not read {name} from {self.path}")
        return extracted.read()

    def close(self) -> None:
        self.archive.close()


def _normalized(name: str) -> str:
    return "/" + name.replace("\\", "/").lstrip("/")


def _require_suffix(names: list[str], suffix: str, artifact: Path) -> str:
    matches = [name for name in names if _normalized(name).endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"{artifact}: expected one *{suffix}, found {matches}")
    return matches[0]


def _require_sdist_path(names: list[str], relative_path: str, artifact: Path) -> str:
    expected_parts = PurePosixPath(relative_path).parts
    matches = [name for name in names if PurePosixPath(name).parts[1:] == expected_parts]
    if len(matches) != 1:
        raise ValueError(f"{artifact}: expected one project file {relative_path!r}, found {matches}")
    return matches[0]


def _check_paths(artifact: Artifact) -> None:
    names = artifact.path_names()
    if len(names) != len(set(names)):
        raise ValueError(f"{artifact.path}: duplicate archive paths are not allowed")

    for name in names:
        normalized = _normalized(name)
        pure_path = PurePosixPath(normalized)
        if ".." in pure_path.parts:
            raise ValueError(f"{artifact.path}: unsafe archive path {name}")
        if any(part in normalized for part in BANNED_PATH_PARTS):
            raise ValueError(f"{artifact.path}: banned release path {name}")
        if normalized.endswith(BANNED_PATH_SUFFIXES):
            raise ValueError(f"{artifact.path}: stale release path {name}")


def _check_internal_content(artifact: Artifact) -> None:
    for name in artifact.names():
        data = artifact.read(name)
        markers = BANNED_CONTENT
        if b"\0" not in data[:8192]:
            markers += BANNED_TEXT_CONTENT
        for marker in markers:
            if marker in data:
                value = marker.decode("utf-8", errors="replace")
                raise ValueError(f"{artifact.path}: internal value {value!r} found in {name}")


def _metadata(artifact: Artifact, suffix: str):
    name = _require_suffix(artifact.names(), suffix, artifact.path)
    return BytesParser().parsebytes(artifact.read(name))


def _check_metadata(artifact: Artifact, suffix: str) -> None:
    metadata = _metadata(artifact, suffix)
    if metadata["Name"] != EXPECTED_NAME:
        raise ValueError(f"{artifact.path}: unexpected project name {metadata['Name']!r}")
    if metadata["License-Expression"] != "Apache-2.0":
        raise ValueError(f"{artifact.path}: missing Apache-2.0 License-Expression")
    if metadata["Version"] != EXPECTED_VERSION:
        raise ValueError(f"{artifact.path}: expected version {EXPECTED_VERSION!r}, found {metadata['Version']!r}")


def _check_sdist(artifact: SdistArtifact) -> None:
    names = artifact.names()
    required_paths = (
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY.md",
        "SOURCE_PROVENANCE.md",
        "LICENSES/DuckDB-MIT.txt",
        "LICENSES/vcpkg-binary-dependencies.txt",
        "external/duckdb/LICENSE",
        "scripts/run_release_tests.sh",
        "tests/fast/test_package_metadata.py",
    )
    for relative_path in required_paths:
        _require_sdist_path(names, relative_path, artifact.path)
    _check_metadata(artifact, "/PKG-INFO")


def _urlsafe_sha256(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _check_wheel_record(artifact: WheelArtifact) -> None:
    record_name = _require_suffix(artifact.names(), ".dist-info/RECORD", artifact.path)
    rows = csv.reader(io.StringIO(artifact.read(record_name).decode("utf-8")))
    recorded: set[str] = set()
    for name, digest, size in rows:
        recorded.add(name)
        if name == record_name:
            continue
        data = artifact.read(name)
        expected_digest = f"sha256={_urlsafe_sha256(data)}"
        if digest != expected_digest or size != str(len(data)):
            raise ValueError(f"{artifact.path}: invalid RECORD entry for {name}")
    missing = set(artifact.names()) - recorded
    if missing:
        raise ValueError(f"{artifact.path}: files missing from RECORD: {sorted(missing)}")


def _check_wheel(artifact: WheelArtifact) -> None:
    names = artifact.names()
    required_suffixes = (
        ".dist-info/licenses/LICENSE",
        ".dist-info/licenses/NOTICE",
        ".dist-info/licenses/LICENSES/DuckDB-MIT.txt",
        ".dist-info/licenses/LICENSES/vcpkg-binary-dependencies.txt",
        ".dist-info/licenses/duckdb/experimental/spark/LICENSE",
        ".dist-info/licenses/external/duckdb/LICENSE",
        ".dist-info/licenses/external/duckdb/src/include/duckdb/storage/compression/alp/algorithm/LICENSE",
        ".dist-info/licenses/external/duckdb/src/include/duckdb/storage/compression/alprd/algorithm/LICENSE",
    )
    for suffix in required_suffixes:
        _require_suffix(names, suffix, artifact.path)
    _check_metadata(artifact, ".dist-info/METADATA")
    _check_wheel_record(artifact)


def check_artifact(path: Path) -> None:
    """Validate one sdist or wheel."""
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError(f"{path}: artifact exceeds the project's 100 MiB publication limit")

    if path.name.endswith(".tar.gz"):
        artifact: SdistArtifact | WheelArtifact = SdistArtifact(path)
        specific_check = _check_sdist
    elif path.suffix == ".whl":
        artifact = WheelArtifact(path)
        specific_check = _check_wheel
    else:
        raise ValueError(f"unsupported artifact type: {path}")

    try:
        _check_paths(artifact)
        _check_internal_content(artifact)
        specific_check(artifact)
    finally:
        artifact.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()

    for artifact in args.artifacts:
        check_artifact(artifact)
        print(f"validated {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
