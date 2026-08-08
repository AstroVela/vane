#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate Vane source and wheel artifacts before publication."""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import io
import json
import re
import stat
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass, field
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
DUCKDB_FORK_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})(?:-dirty)?")
DUCKDB_UPSTREAM_VERSION = re.compile(r"v[0-9]+\.[0-9]+\.[0-9]+")
CONTENT_RULE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
FORBIDDEN_DISTRIBUTION_ROOTS = {"adbc_driver_duckdb", "duckdb"}

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


class Artifact(Protocol):
    path: Path

    def path_names(self) -> list[str]: ...

    def names(self) -> list[str]: ...

    def read(self, name: str) -> bytes: ...


class ContentRule(Protocol):
    rule_id: str

    def matches(self, data: bytes) -> bool: ...


@dataclass(frozen=True)
class ContentMatch:
    rule_id: str
    member_name: str


@dataclass(frozen=True)
class LiteralContentRule:
    rule_id: str
    value: bytes = field(repr=False)

    def matches(self, data: bytes) -> bool:
        return self.value in data


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


def _is_forbidden_import_root(path_component: str) -> bool:
    import_name = path_component.partition(".")[0]
    return import_name in FORBIDDEN_DISTRIBUTION_ROOTS or import_name.startswith("_duckdb")


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


def _detect_internal_content(
    artifact: Artifact,
    content_rules: tuple[ContentRule, ...],
    text_content_rules: tuple[ContentRule, ...],
) -> ContentMatch | None:
    for name in artifact.names():
        data = artifact.read(name)
        for rule in content_rules:
            if rule.matches(data):
                return ContentMatch(rule_id=rule.rule_id, member_name=name)
        if b"\0" in data[:8192]:
            continue
        for rule in text_content_rules:
            if rule.matches(data):
                return ContentMatch(rule_id=rule.rule_id, member_name=name)
    return None


def _check_internal_content(
    artifact: Artifact,
    content_rules: tuple[ContentRule, ...],
    text_content_rules: tuple[ContentRule, ...],
) -> None:
    match = _detect_internal_content(artifact, content_rules, text_content_rules)
    if match is not None:
        raise ValueError(
            f"{artifact.path}: content rule {match.rule_id!r} matched archive member {match.member_name!r}"
        )


def _parse_content_rule_manifest(
    raw_manifest: str,
    *,
    source: str,
) -> tuple[tuple[ContentRule, ...], tuple[ContentRule, ...]]:
    try:
        manifest = json.loads(raw_manifest)
    except json.JSONDecodeError:
        raise ValueError(f"{source}: invalid content rule manifest JSON") from None

    if not isinstance(manifest, dict) or set(manifest) != {"version", "rules"}:
        raise ValueError(f"{source}: invalid content rule manifest structure")
    if manifest["version"] != 1:
        raise ValueError(f"{source}: unsupported content rule manifest version")

    entries = manifest["rules"]
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{source}: content rule manifest must contain at least one rule")

    content_rules: list[ContentRule] = []
    text_content_rules: list[ContentRule] = []
    rule_ids: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {"id", "scope", "value_base64"}:
            raise ValueError(f"{source}: invalid content rule at index {index}")

        rule_id = entry["id"]
        if not isinstance(rule_id, str) or CONTENT_RULE_ID.fullmatch(rule_id) is None:
            raise ValueError(f"{source}: invalid content rule id at index {index}")
        if rule_id in rule_ids:
            raise ValueError(f"{source}: duplicate content rule id {rule_id!r}")
        rule_ids.add(rule_id)

        scope = entry["scope"]
        if not isinstance(scope, str) or scope not in {"all", "text"}:
            raise ValueError(f"{source}: invalid content rule scope at index {index}")

        encoded_value = entry["value_base64"]
        if not isinstance(encoded_value, str):
            raise ValueError(f"{source}: invalid content rule value at index {index}")
        try:
            value = base64.b64decode(encoded_value, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError(f"{source}: invalid content rule value at index {index}") from None
        if not value:
            raise ValueError(f"{source}: empty content rule value at index {index}")

        rule = LiteralContentRule(rule_id=rule_id, value=value)
        if scope == "text":
            text_content_rules.append(rule)
        else:
            content_rules.append(rule)

    return tuple(content_rules), tuple(text_content_rules)


def _load_content_rule_manifest(
    source: str,
) -> tuple[tuple[ContentRule, ...], tuple[ContentRule, ...]]:
    if source == "-":
        raw_manifest = sys.stdin.read()
        source_name = "stdin"
    else:
        path = Path(source)
        try:
            raw_manifest = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            raise ValueError(f"{path}: could not read content rule manifest") from None
        source_name = str(path)
    return _parse_content_rule_manifest(raw_manifest, source=source_name)


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


def _checkout_duckdb_fork_revision() -> str | None:
    """Return the checkout's last DuckDB-changing commit when Git is available."""
    if not (REPOSITORY_ROOT / ".git").exists():
        return None

    result = subprocess.run(
        [sys.executable, "scripts/resolve_duckdb_fork_version.py", "--print-revision"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if DUCKDB_FORK_REVISION.fullmatch(revision) is None:
        raise ValueError(f"current checkout produced an invalid DuckDB fork revision {revision!r}")
    return revision


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
    for name in names:
        parts = PurePosixPath(name).parts
        if len(parts) < 2:
            continue
        source_root = parts[1]
        if _is_forbidden_import_root(source_root):
            raise ValueError(f"{artifact.path}: Vane sdist contains conflicting Python package path {name!r}")

    required_paths = (
        "DUCKDB_FORK_REVISION",
        "DUCKDB_SOURCE_ID",
        "DUCKDB_UPSTREAM_VERSION",
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY.md",
        "SOURCE_PROVENANCE.md",
        "LICENSES/DuckDB-MIT.txt",
        "LICENSES/vcpkg-binary-dependencies.txt",
        "external/duckdb/LICENSE",
        "build_backend.py",
        "scripts/resolve_duckdb_fork_version.py",
        "scripts/run_installed_pytest.sh",
        "scripts/run_release_tests.sh",
        "scripts/sync_duckdb_source_id.py",
        "scripts/verify_duckdb_coexistence.py",
        "tests/ray_test_profile.py",
        "tests/fast/test_package_metadata.py",
        "tests/fast/test_ray_test_profile.py",
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

    fork_revision_name = _require_sdist_path(names, "DUCKDB_FORK_REVISION", artifact.path)
    fork_revision = artifact.read(fork_revision_name).decode("ascii").strip()
    if DUCKDB_FORK_REVISION.fullmatch(fork_revision) is None:
        raise ValueError(f"{artifact.path}: invalid DuckDB fork revision {fork_revision!r}")
    checkout_fork_revision = _checkout_duckdb_fork_revision()
    if checkout_fork_revision is not None and fork_revision != checkout_fork_revision:
        raise ValueError(
            f"{artifact.path}: DuckDB fork revision {fork_revision!r} "
            f"does not match checkout {checkout_fork_revision!r}"
        )

    upstream_version_name = _require_sdist_path(names, "DUCKDB_UPSTREAM_VERSION", artifact.path)
    upstream_version = artifact.read(upstream_version_name).decode("ascii").strip()
    if DUCKDB_UPSTREAM_VERSION.fullmatch(upstream_version) is None:
        raise ValueError(f"{artifact.path}: invalid DuckDB upstream version {upstream_version!r}")
    checkout_upstream_version = (REPOSITORY_ROOT / "DUCKDB_UPSTREAM_VERSION").read_text(encoding="ascii").strip()
    if upstream_version != checkout_upstream_version:
        raise ValueError(
            f"{artifact.path}: DuckDB upstream version {upstream_version!r} "
            f"does not match checkout {checkout_upstream_version!r}"
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


def _check_wheel(artifact: WheelArtifact) -> None:
    names = artifact.names()
    for name in names:
        parts = PurePosixPath(name).parts
        import_parts = parts
        if len(parts) >= 3 and parts[0].endswith(".data") and parts[1] in {"platlib", "purelib"}:
            import_parts = parts[2:]
        import_root = import_parts[0]
        if _is_forbidden_import_root(import_root):
            raise ValueError(f"{artifact.path}: Vane wheel contains conflicting Python package path {name!r}")

    native_extensions = [
        name
        for name in names
        if PurePosixPath(name).parent == PurePosixPath("vane")
        and PurePosixPath(name).name.startswith("_native.")
        and PurePosixPath(name).suffix in {".pyd", ".so"}
    ]
    if len(native_extensions) != 1:
        raise ValueError(
            f"{artifact.path}: expected one platform extension under vane/_native.*, found {native_extensions}"
        )

    required_suffixes = (
        "/vane/py.typed",
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
    for requirement in metadata.get_all("Requires-Dist", []):
        requirement_name = re.split(r"[ ;(<=>!~\[]", requirement, maxsplit=1)[0].lower().replace("_", "-")
        if requirement_name == "duckdb":
            raise ValueError(f"{artifact.path}: vane-ai must not depend on the official duckdb distribution")
    _check_wheel_license_files(artifact, metadata)
    _check_wheel_record(artifact)


def check_artifact(
    path: Path,
    *,
    content_rules: tuple[ContentRule, ...] = (),
    text_content_rules: tuple[ContentRule, ...] = (),
) -> None:
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
        _check_internal_content(artifact, content_rules, text_content_rules)
        specific_check(artifact)
    finally:
        artifact.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--content-rules-manifest",
        metavar="PATH",
        help="load private exact-content rules from PATH, or from standard input with '-'",
    )
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()

    content_rules: tuple[ContentRule, ...] = ()
    text_content_rules: tuple[ContentRule, ...] = ()
    if args.content_rules_manifest is not None:
        content_rules, text_content_rules = _load_content_rule_manifest(args.content_rules_manifest)

    for artifact in args.artifacts:
        check_artifact(
            artifact,
            content_rules=content_rules,
            text_content_rules=text_content_rules,
        )
        print(f"validated {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
