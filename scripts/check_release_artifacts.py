#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate Vane source and wheel artifacts before publication."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import re
import stat
import subprocess
import sys
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
GIT_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
PRIVATE_EXTENSION_NAME = re.compile(r"_vane_duckdb(?:[.].+)?[.](?:pyd|so)")
OFFICIAL_DUCKDB_TOP_LEVEL = frozenset(
    {
        "duckdb",
        "_duckdb-stubs",
        "adbc_driver_duckdb",
    }
)

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
        self.all_members = self.archive.infolist()
        for info in self.all_members:
            if stat.S_ISLNK(info.external_attr >> 16):
                self.archive.close()
                raise ValueError(f"{path}: symbolic links are not allowed in wheels: {info.filename}")
        self.members = [info.filename for info in self.all_members if not info.is_dir()]

    def path_names(self) -> list[str]:
        return [info.filename for info in self.all_members]

    def names(self) -> list[str]:
        return self.members

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


def _check_metadata(artifact: Artifact, suffix: str):
    metadata = _metadata(artifact, suffix)
    if metadata["Name"] != EXPECTED_NAME:
        raise ValueError(f"{artifact.path}: unexpected project name {metadata['Name']!r}")
    if metadata["License-Expression"] != "Apache-2.0":
        raise ValueError(f"{artifact.path}: missing Apache-2.0 License-Expression")
    if metadata["Version"] != EXPECTED_VERSION:
        raise ValueError(f"{artifact.path}: expected version {EXPECTED_VERSION!r}, found {metadata['Version']!r}")
    return metadata


def _check_sdist_license_files(artifact: SdistArtifact, metadata) -> None:
    names = artifact.names()
    for relative_path in metadata.get_all("License-File", []):
        _require_sdist_path(names, relative_path, artifact.path)


def _checkout_duckdb_source_id() -> str | None:
    """Return the current checkout identity when Git metadata is available."""
    if not (REPOSITORY_ROOT / ".git").exists():
        return None

    result = subprocess.run(
        [sys.executable, "scripts/sync_duckdb_source_id.py", "--print"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    source_id = result.stdout.strip()
    if GIT_OBJECT_ID.fullmatch(source_id) is None:
        raise ValueError(f"current checkout produced an invalid DuckDB source tree ID {source_id!r}")
    return source_id


def _check_wheel_license_files(artifact: WheelArtifact, metadata) -> None:
    metadata_name = _require_suffix(artifact.names(), ".dist-info/METADATA", artifact.path)
    license_root = PurePosixPath(metadata_name).parent / "licenses"
    names = artifact.names()
    for relative_path in metadata.get_all("License-File", []):
        expected = str(license_root / PurePosixPath(relative_path))
        matches = [name for name in names if name == expected]
        if len(matches) != 1:
            raise ValueError(
                f"{artifact.path}: metadata declares missing license file {relative_path!r}; "
                f"expected wheel path {expected!r}"
            )


def _check_sdist(artifact: SdistArtifact) -> None:
    names = artifact.names()
    required_paths = (
        "DUCKDB_SOURCE_ID",
        "COMPATIBILITY.md",
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY.md",
        "SOURCE_PROVENANCE.md",
        "LICENSES/DuckDB-MIT.txt",
        "LICENSES/vcpkg-binary-dependencies.txt",
        "external/duckdb/LICENSE",
        "build_backend.py",
        "scripts/run_release_tests.sh",
        "scripts/sync_duckdb_source_id.py",
        "tests/ray_test_profile.py",
        "tests/fast/test_package_metadata.py",
        "tests/fast/test_ray_test_profile.py",
        "tests/packaging/test_duckdb_distribution_coexistence.py",
    )
    for relative_path in required_paths:
        _require_sdist_path(names, relative_path, artifact.path)

    source_id_name = _require_sdist_path(names, "DUCKDB_SOURCE_ID", artifact.path)
    source_id = artifact.read(source_id_name).decode("ascii").strip()
    if GIT_OBJECT_ID.fullmatch(source_id) is None:
        raise ValueError(f"{artifact.path}: invalid DuckDB source tree ID {source_id!r}")
    checkout_source_id = _checkout_duckdb_source_id()
    if checkout_source_id is not None and source_id != checkout_source_id:
        raise ValueError(
            f"{artifact.path}: DuckDB source tree ID {source_id!r} does not match checkout {checkout_source_id!r}"
        )

    metadata = _check_metadata(artifact, "/PKG-INFO")
    _check_sdist_license_files(artifact, metadata)


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


def _check_wheel_namespace(artifact: WheelArtifact) -> None:
    top_level = {PurePosixPath(name).parts[0] for name in artifact.names() if PurePosixPath(name).parts}
    conflicting = sorted(
        name
        for name in top_level
        if name in OFFICIAL_DUCKDB_TOP_LEVEL or name == "_duckdb" or name.startswith("_duckdb.")
    )
    if conflicting:
        raise ValueError(f"{artifact.path}: wheel claims official DuckDB paths: {conflicting}")

    required = {"vane", "_vane_duckdb-stubs", "vane_adbc_driver_duckdb"}
    missing = required - top_level
    if missing:
        raise ValueError(f"{artifact.path}: wheel is missing private Vane package paths: {sorted(missing)}")

    extensions = sorted(name for name in top_level if PRIVATE_EXTENSION_NAME.fullmatch(name))
    if len(extensions) != 1:
        raise ValueError(f"{artifact.path}: expected one private _vane_duckdb extension, found {extensions}")


def _check_wheel(artifact: WheelArtifact) -> None:
    names = artifact.names()
    required_suffixes = (
        ".dist-info/licenses/LICENSE",
        ".dist-info/licenses/NOTICE",
        ".dist-info/licenses/LICENSES/DuckDB-MIT.txt",
        ".dist-info/licenses/LICENSES/vcpkg-binary-dependencies.txt",
        ".dist-info/licenses/vane/experimental/spark/LICENSE",
        ".dist-info/licenses/external/duckdb/LICENSE",
        ".dist-info/licenses/external/duckdb/src/include/duckdb/storage/compression/alp/algorithm/LICENSE",
        ".dist-info/licenses/external/duckdb/src/include/duckdb/storage/compression/alprd/algorithm/LICENSE",
    )
    for suffix in required_suffixes:
        _require_suffix(names, suffix, artifact.path)
    metadata = _check_metadata(artifact, ".dist-info/METADATA")
    _check_wheel_license_files(artifact, metadata)
    _check_wheel_namespace(artifact)
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
