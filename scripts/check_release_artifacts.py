#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate Vane source and wheel artifacts before publication."""

from __future__ import annotations

import argparse
import base64
import binascii
import configparser
import csv
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass, field
from email.parser import BytesParser
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol

from packaging.markers import Marker
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ORIGINAL_SYS_PATH = sys.path.copy()
try:
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from vane_packaging.archive_safety import (
        ArchiveSnapshot,
        open_tar_snapshot,
        open_zip_snapshot,
        snapshot_archive,
    )
finally:
    sys.path[:] = _ORIGINAL_SYS_PATH
    del _ORIGINAL_SYS_PATH

PROJECT_METADATA = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
EXPECTED_NAME = str(PROJECT_METADATA["name"])
MAX_ARTIFACT_BYTES = 100 * 1024 * 1024
MAX_ARTIFACT_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
MAX_ARTIFACT_MEMBERS = 10_000
MAX_CORE_METADATA_BYTES = 1024 * 1024
MAX_CORE_METADATA_HEADERS = 1024
MAX_CORE_METADATA_LINES = 10_000
MAX_CORE_METADATA_LINE_BYTES = 64 * 1024
_RAW_ARCHIVE_SCAN_CHUNK_BYTES = 1024 * 1024
GIT_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
DUCKDB_FORK_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})(?:-dirty)?")
DUCKDB_UPSTREAM_VERSION = re.compile(r"v[0-9]+\.[0-9]+\.[0-9]+")
CONTENT_RULE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
FORBIDDEN_DISTRIBUTION_ROOTS = {"adbc_driver_duckdb", "duckdb"}
WHEEL_DISTRIBUTION = re.sub(r"[-_.]+", "_", EXPECTED_NAME)
EXPECTED_LIBRARY_ROOT = f"{WHEEL_DISTRIBUTION}.libs"


def _expected_project_requires_dist() -> tuple[str, ...]:
    requirements = [Requirement(str(value)) for value in PROJECT_METADATA.get("dependencies", [])]
    for raw_extra, values in PROJECT_METADATA.get("optional-dependencies", {}).items():
        extra = canonicalize_name(str(raw_extra))
        for value in values:
            requirement = Requirement(str(value))
            extra_marker = f'extra == "{extra}"'
            requirement.marker = Marker(
                extra_marker if requirement.marker is None else f"{requirement.marker} and {extra_marker}"
            )
            requirements.append(requirement)
    return tuple(str(requirement) for requirement in requirements)


EXPECTED_REQUIRES_DIST = _expected_project_requires_dist()
EXPECTED_PROVIDES_EXTRA = tuple(
    canonicalize_name(str(extra)) for extra in PROJECT_METADATA.get("optional-dependencies", {})
)


def _expected_project_entry_points() -> dict[str, dict[str, str]]:
    groups: dict[str, dict[str, str]] = {}
    for project_key, group in (("scripts", "console_scripts"), ("gui-scripts", "gui_scripts")):
        values = PROJECT_METADATA.get(project_key, {})
        if values:
            groups[group] = {str(name): str(reference) for name, reference in values.items()}
    for group, values in PROJECT_METADATA.get("entry-points", {}).items():
        group_name = str(group)
        if group_name in groups:
            raise ValueError(f"project metadata declares entry-point group {group_name!r} more than once")
        groups[group_name] = {str(name): str(reference) for name, reference in values.items()}
    return groups


EXPECTED_ENTRY_POINTS = _expected_project_entry_points()

BANNED_PATH_PARTS = (
    "/.git/",
    "/.venv",
    "/build/",
    "/dist/",
    "/external/duckdb/extension/tpcds/",
    "/external/duckdb/extension/tpch/",
    "/external/duckdb/third_party/tpce-tool/",
    "/vane/_native.pyi",
    "/vane/session_",
    "/vcpkg_installed/",
)
BANNED_PATH_SUFFIXES = (
    ".log",
    "/ray_data_main_old.py",
)


class Artifact(Protocol):
    path: Path
    raw_content_match: ContentMatch | None

    def path_names(self) -> list[str]: ...

    def names(self) -> list[str]: ...

    def member_sizes(self) -> Iterable[tuple[str, int]]: ...

    def read(self, name: str) -> bytes: ...


class ContentRule(Protocol):
    rule_id: str

    def matches(self, data: bytes) -> bool: ...

    def overlap_bytes(self) -> int: ...


@dataclass(frozen=True)
class ContentMatch:
    rule_id: str
    member_name: str


@dataclass(frozen=True)
class DistributionLayout:
    version: Version
    archive_root: str
    dist_info_root: str


def distribution_layout(version: Version | str) -> DistributionLayout:
    parsed_version = version if isinstance(version, Version) else Version(version)
    wheel_version = re.sub(r"[^A-Za-z0-9.]+", "_", str(parsed_version))
    archive_root = f"{WHEEL_DISTRIBUTION}-{wheel_version}"
    return DistributionLayout(
        version=parsed_version,
        archive_root=archive_root,
        dist_info_root=f"{archive_root}.dist-info",
    )


@dataclass(frozen=True)
class LiteralContentRule:
    rule_id: str
    value: bytes = field(repr=False)

    def matches(self, data: bytes) -> bool:
        return self.value in data

    def overlap_bytes(self) -> int:
        return max(len(self.value) - 1, 0)


class _TarMetadataContentScanner:
    def __init__(self, path: Path, rules: tuple[ContentRule, ...]):
        self.path = path
        self.rules = rules
        self.overlap_bytes = max((rule.overlap_bytes() for rule in rules), default=0)
        self.overlap = b""

    def __call__(self, chunk: bytes, new_region: bool) -> None:
        if new_region:
            self.overlap = b""
        window = self.overlap + chunk
        for rule in self.rules:
            if rule.matches(window):
                raise ValueError(f"{self.path}: content rule {rule.rule_id!r} matched decompressed TAR metadata")
        self.overlap = window[-self.overlap_bytes :] if self.overlap_bytes else b""


class WheelArtifact:
    def __init__(self, archive_input: Path | ArchiveSnapshot, *, content_rules: tuple[ContentRule, ...] = ()):
        self.path = archive_input.source_path if isinstance(archive_input, ArchiveSnapshot) else archive_input
        self._resources = ExitStack()
        try:
            if isinstance(archive_input, ArchiveSnapshot):
                self.snapshot = archive_input
            else:
                self.snapshot = self._resources.enter_context(
                    snapshot_archive(
                        archive_input,
                        max_bytes=MAX_ARTIFACT_BYTES,
                        description="artifact",
                        size_limit_description="the project's 100 MiB publication limit",
                    )
                )
            self.raw_content_match = _detect_raw_archive_content(self.snapshot, content_rules)
            self.archive = self._resources.enter_context(
                open_zip_snapshot(
                    self.snapshot,
                    max_members=MAX_ARTIFACT_MEMBERS,
                    description="wheel",
                )
            )
            self.all_members = self.archive.infolist()
            for info in self.all_members:
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError(f"{self.path}: symbolic links are not allowed in wheels: {info.filename}")
            self.members = [info.filename for info in self.all_members if not info.is_dir()]
        except BaseException:
            self._resources.close()
            raise

    def path_names(self) -> list[str]:
        return [info.filename for info in self.all_members]

    def names(self) -> list[str]:
        return self.members

    def member_sizes(self) -> Iterable[tuple[str, int]]:
        return ((info.filename, info.file_size) for info in self.all_members)

    def read(self, name: str) -> bytes:
        return self.archive.read(name)

    def close(self) -> None:
        self._resources.close()


class SdistArtifact:
    def __init__(self, archive_input: Path | ArchiveSnapshot, *, content_rules: tuple[ContentRule, ...] = ()):
        self.path = archive_input.source_path if isinstance(archive_input, ArchiveSnapshot) else archive_input
        self._resources = ExitStack()
        try:
            if isinstance(archive_input, ArchiveSnapshot):
                self.snapshot = archive_input
            else:
                self.snapshot = self._resources.enter_context(
                    snapshot_archive(
                        archive_input,
                        max_bytes=MAX_ARTIFACT_BYTES,
                        description="artifact",
                        size_limit_description="the project's 100 MiB publication limit",
                    )
                )
            self.raw_content_match = _detect_raw_archive_content(self.snapshot, content_rules)
            metadata_scanner = _TarMetadataContentScanner(self.path, content_rules)
            self.archive = self._resources.enter_context(
                open_tar_snapshot(
                    self.snapshot,
                    max_members=MAX_ARTIFACT_MEMBERS,
                    max_member_bytes=MAX_ARTIFACT_UNCOMPRESSED_BYTES,
                    max_total_bytes=MAX_ARTIFACT_UNCOMPRESSED_BYTES,
                    uncompressed_limit_description="the project's 100 MiB uncompressed limit",
                    description="sdist",
                    metadata_chunk_callback=metadata_scanner if content_rules else None,
                    mode="r:gz",
                )
            )
            self.all_members = self.archive.getmembers()
            for member in self.all_members:
                if not member.isfile() and not member.isdir():
                    raise ValueError(f"{self.path}: unsupported tar member type for {member.name}")
            self.members = {member.name: member for member in self.all_members if member.isfile()}
        except BaseException:
            self._resources.close()
            raise

    def path_names(self) -> list[str]:
        return [member.name for member in self.all_members]

    def names(self) -> list[str]:
        return list(self.members)

    def member_sizes(self) -> Iterable[tuple[str, int]]:
        return ((member.name, member.size) for member in self.all_members)

    def read(self, name: str) -> bytes:
        extracted = self.archive.extractfile(self.members[name])
        if extracted is None:
            raise ValueError(f"could not read {name} from {self.path}")
        return extracted.read()

    def close(self) -> None:
        self._resources.close()


def _artifact_version(path: Path) -> Version:
    try:
        if path.name.endswith(".tar.gz"):
            name, version = parse_sdist_filename(path.name)
        elif path.suffix == ".whl":
            name, version, _, _ = parse_wheel_filename(path.name)
        else:
            raise ValueError(f"unsupported artifact type: {path}")
    except (InvalidSdistFilename, InvalidWheelFilename):
        raise ValueError(f"{path}: invalid Python distribution filename") from None

    if canonicalize_name(name) != canonicalize_name(EXPECTED_NAME):
        raise ValueError(f"{path}: unexpected project name {name!r}")
    return version


def _normalized(name: str) -> str:
    return "/" + name.rstrip("/")


def _archive_path_key(name: str, artifact: Path) -> str:
    if not name or "\\" in name or name.startswith("/") or PureWindowsPath(name).drive:
        raise ValueError(f"{artifact}: unsafe archive path {name!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise ValueError(f"{artifact}: unsafe archive path {name!r}")

    path_without_directory_marker = name[:-1] if name.endswith("/") else name
    parts = path_without_directory_marker.split("/")
    if not path_without_directory_marker or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{artifact}: unsafe archive path {name!r}")
    return unicodedata.normalize("NFC", path_without_directory_marker).casefold()


def _is_forbidden_import_root(path_component: str) -> bool:
    import_name = path_component.partition(".")[0].lower()
    return import_name in FORBIDDEN_DISTRIBUTION_ROOTS or import_name.startswith("_duckdb")


def _require_exact_path(names: list[str], expected: str, artifact: Path) -> str:
    matches = [name for name in names if name == expected]
    if len(matches) != 1:
        raise ValueError(f"{artifact}: expected one archive member {expected!r}, found {matches}")
    return matches[0]


def _require_sdist_path(names: list[str], relative_path: str, artifact: Path) -> str:
    expected_parts = PurePosixPath(relative_path).parts
    matches = [name for name in names if PurePosixPath(name).parts[1:] == expected_parts]
    if len(matches) != 1:
        raise ValueError(f"{artifact}: expected one project file {relative_path!r}, found {matches}")
    return matches[0]


def _check_paths(artifact: Artifact) -> None:
    names = artifact.path_names()
    canonical_paths: dict[str, str] = {}
    canonical_files = {_archive_path_key(name, artifact.path): name for name in artifact.names()}

    for name in names:
        canonical_path = _archive_path_key(name, artifact.path)
        previous = canonical_paths.get(canonical_path)
        if previous is not None:
            raise ValueError(
                f"{artifact.path}: duplicate or cross-platform-colliding archive paths are not allowed: "
                f"{previous!r}, {name!r}"
            )
        canonical_paths[canonical_path] = name

        path_parts = canonical_path.split("/")
        for depth in range(1, len(path_parts)):
            ancestor = "/".join(path_parts[:depth])
            ancestor_file = canonical_files.get(ancestor)
            if ancestor_file is not None:
                raise ValueError(
                    f"{artifact.path}: archive file cannot be the parent of another member: {ancestor_file!r}, {name!r}"
                )

        normalized = _normalized(name).casefold()
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


def _detect_raw_archive_content(
    snapshot: ArchiveSnapshot,
    content_rules: tuple[ContentRule, ...],
) -> ContentMatch | None:
    if not content_rules:
        return None
    path = snapshot.source_path
    overlap_bytes = max(rule.overlap_bytes() for rule in content_rules)
    try:
        metadata = os.fstat(snapshot.file.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != snapshot.size:
            raise ValueError(f"{path}: private artifact snapshot changed before content scanning")
        snapshot.file.seek(0)
        overlap = b""
        total_size = 0
        while chunk := snapshot.file.read(_RAW_ARCHIVE_SCAN_CHUNK_BYTES):
            total_size += len(chunk)
            if total_size > MAX_ARTIFACT_BYTES:
                raise ValueError(f"{path}: artifact exceeds the project's 100 MiB publication limit")
            window = overlap + chunk
            for rule in content_rules:
                if rule.matches(window):
                    return ContentMatch(rule_id=rule.rule_id, member_name="raw archive bytes")
            overlap = window[-overlap_bytes:] if overlap_bytes else b""
    except OSError as exception:
        raise ValueError(f"could not scan artifact contents: {path}") from exception
    return None


def _check_raw_archive_content(path: Path, match: ContentMatch | None) -> None:
    if match is not None:
        raise ValueError(f"{path}: content rule {match.rule_id!r} matched {match.member_name}")


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


def _metadata(artifact: Artifact, path: str):
    name = _require_exact_path(artifact.names(), path, artifact.path)
    member_sizes = [size for member_name, size in artifact.member_sizes() if member_name == name]
    if len(member_sizes) != 1 or member_sizes[0] > MAX_CORE_METADATA_BYTES:
        raise ValueError(f"{artifact.path}: core metadata exceeds its bounded 1 MiB limit")
    contents = artifact.read(name)
    _validate_core_metadata_shape(contents, artifact=artifact.path)
    return BytesParser().parsebytes(contents)


def _validate_core_metadata_shape(contents: bytes, *, artifact: Path) -> None:
    line_count = 0
    header_count = 0
    in_headers = True
    have_header = False
    for line in _core_metadata_lines(contents):
        line_count += 1
        if line_count > MAX_CORE_METADATA_LINES:
            raise ValueError(f"{artifact}: core metadata contains more than {MAX_CORE_METADATA_LINES} lines")
        if len(line) > MAX_CORE_METADATA_LINE_BYTES:
            raise ValueError(f"{artifact}: core metadata contains an overlong line")
        if not in_headers:
            continue
        if not line:
            in_headers = False
        elif line.startswith((b" ", b"\t")):
            if not have_header:
                raise ValueError(f"{artifact}: core metadata starts with an invalid continuation line")
        else:
            have_header = True
            header_count += 1
            if header_count > MAX_CORE_METADATA_HEADERS:
                raise ValueError(f"{artifact}: core metadata contains more than {MAX_CORE_METADATA_HEADERS} headers")


def _core_metadata_lines(contents: bytes) -> Iterable[bytes]:
    line_start = 0
    offset = 0
    while offset < len(contents):
        if contents[offset] not in (10, 13):
            offset += 1
            continue
        yield contents[line_start:offset]
        if contents[offset] == 13 and offset + 1 < len(contents) and contents[offset + 1] == 10:
            offset += 1
        offset += 1
        line_start = offset
    if line_start < len(contents):
        yield contents[line_start:]


def _check_metadata(artifact: Artifact, path: str, layout: DistributionLayout):
    metadata = _metadata(artifact, path)
    if metadata["Name"] != EXPECTED_NAME:
        raise ValueError(f"{artifact.path}: unexpected project name {metadata['Name']!r}")
    if metadata["License-Expression"] != "Apache-2.0":
        raise ValueError(f"{artifact.path}: missing Apache-2.0 License-Expression")
    expected_version = str(layout.version)
    if metadata["Version"] != expected_version:
        raise ValueError(f"{artifact.path}: expected version {expected_version!r}, found {metadata['Version']!r}")
    return metadata


def _check_no_official_duckdb_dependency(artifact: Artifact, metadata) -> None:
    for raw_requirement in metadata.get_all("Requires-Dist", []):
        try:
            requirement = Requirement(raw_requirement)
        except InvalidRequirement:
            raise ValueError(f"{artifact.path}: invalid Requires-Dist metadata {raw_requirement!r}") from None
        if canonicalize_name(requirement.name) == "duckdb":
            raise ValueError(f"{artifact.path}: vane-ai must not depend on the official duckdb distribution")


def _check_project_dependency_metadata(artifact: Artifact, metadata) -> None:
    raw_requirements = tuple(str(value) for value in metadata.get_all("Requires-Dist", []))
    try:
        requirements = tuple(str(Requirement(value)) for value in raw_requirements)
    except InvalidRequirement:
        raise ValueError(f"{artifact.path}: invalid Requires-Dist metadata") from None
    actual_requirements = Counter(requirements)
    expected_requirements = Counter(EXPECTED_REQUIRES_DIST)
    if actual_requirements != expected_requirements:
        missing = sorted((expected_requirements - actual_requirements).elements())
        unexpected = sorted((actual_requirements - expected_requirements).elements())
        raise ValueError(
            f"{artifact.path}: Requires-Dist must match project metadata exactly: "
            f"missing={missing}, unexpected={unexpected}"
        )

    actual_extras = Counter(str(value) for value in metadata.get_all("Provides-Extra", []))
    expected_extras = Counter(EXPECTED_PROVIDES_EXTRA)
    if actual_extras != expected_extras:
        missing = sorted((expected_extras - actual_extras).elements())
        unexpected = sorted((actual_extras - expected_extras).elements())
        raise ValueError(
            f"{artifact.path}: Provides-Extra must match project metadata exactly: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _check_project_entry_points(artifact: WheelArtifact, layout: DistributionLayout) -> None:
    entry_points_path = f"{layout.dist_info_root}/entry_points.txt"
    matches = [name for name in artifact.names() if name.casefold() == entry_points_path.casefold()]
    if not EXPECTED_ENTRY_POINTS:
        if matches:
            raise ValueError(
                f"{artifact.path}: wheel must not contain entry_points.txt because project metadata "
                "declares no entry points"
            )
        return
    if matches != [entry_points_path]:
        raise ValueError(
            f"{artifact.path}: expected one exact entry-point metadata member {entry_points_path!r}, found {matches}"
        )

    try:
        contents = artifact.read(entry_points_path).decode("utf-8")
    except UnicodeError:
        raise ValueError(f"{artifact.path}: entry_points.txt is not valid UTF-8") from None
    parser = configparser.ConfigParser(
        interpolation=None,
        delimiters=("=",),
        strict=True,
        empty_lines_in_values=False,
    )
    parser.optionxform = str
    try:
        parser.read_string(contents)
    except configparser.Error:
        raise ValueError(f"{artifact.path}: entry_points.txt is not valid entry-point metadata") from None
    if parser.defaults():
        raise ValueError(f"{artifact.path}: entry_points.txt must not contain default entries")
    actual = {section: dict(parser.items(section, raw=True)) for section in parser.sections()}
    if actual != EXPECTED_ENTRY_POINTS:
        raise ValueError(
            f"{artifact.path}: entry_points.txt must match project metadata exactly: "
            f"expected={EXPECTED_ENTRY_POINTS}, actual={actual}"
        )


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


def _check_wheel_license_files(
    artifact: WheelArtifact,
    metadata,
    layout: DistributionLayout,
) -> None:
    metadata_name = _require_exact_path(artifact.names(), f"{layout.dist_info_root}/METADATA", artifact.path)
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


def _check_sdist(artifact: SdistArtifact, layout: DistributionLayout) -> None:
    names = artifact.names()
    for name in names:
        parts = PurePosixPath(name).parts
        possible_source_roots = parts[:2]
        if any(_is_forbidden_import_root(source_root) for source_root in possible_source_roots):
            raise ValueError(f"{artifact.path}: Vane sdist contains conflicting Python package path {name!r}")
        if not parts or parts[0] != layout.archive_root:
            raise ValueError(f"{artifact.path}: Vane sdist contains an unexpected archive root: {name!r}")

    required_paths = (
        "DUCKDB_FORK_REVISION",
        "DUCKDB_SOURCE_ID",
        "DUCKDB_UPSTREAM_VERSION",
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY.md",
        "SOURCE_PROVENANCE.md",
        "LICENSES/DuckDB-MIT.txt",
        "LICENSES/auditwheel-LICENSE.txt",
        "LICENSES/vcpkg-binary-dependencies.txt",
        "external/duckdb/LICENSE",
        "build_backend.py",
        "vane_packaging/__init__.py",
        "vane_packaging/archive_safety.py",
        "vane_packaging/extension_wheel.py",
        "vane_packaging/manylinux_policy.py",
        "vane_packaging/setuptools_scm_version.py",
        "vane_packaging/_vendor/auditwheel/manylinux-policy.json",
        "scripts/build_extension_wheel.py",
        "scripts/check_release_artifacts.py",
        "scripts/resolve_duckdb_fork_version.py",
        "scripts/run_installed_pytest.sh",
        "scripts/run_release_tests.sh",
        "scripts/sync_duckdb_source_id.py",
        "scripts/verify_duckdb_coexistence.py",
        "scripts/verify_extension_wheel.py",
        "tests/ray_test_profile.py",
        "tests/fast/test_package_metadata.py",
        "tests/fast/test_ray_test_profile.py",
        "vane/_native/__init__.pyi",
        "vane/_native/_func.pyi",
        "vane/_native/_sqltypes.pyi",
        "vane/_native/ray_cxx.pyi",
        "vane/sqltypes/__init__.pyi",
        "vane/udf.pyi",
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

    metadata = _check_metadata(artifact, f"{layout.archive_root}/PKG-INFO", layout)
    _check_no_official_duckdb_dependency(artifact, metadata)
    _check_project_dependency_metadata(artifact, metadata)
    _check_sdist_license_files(artifact, metadata)


def _urlsafe_sha256(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _check_wheel_record(artifact: WheelArtifact, layout: DistributionLayout) -> None:
    record_name = _require_exact_path(artifact.names(), f"{layout.dist_info_root}/RECORD", artifact.path)
    try:
        rows = csv.reader(io.StringIO(artifact.read(record_name).decode("utf-8")))
    except UnicodeError:
        raise ValueError(f"{artifact.path}: RECORD is not valid UTF-8") from None
    recorded: set[str] = set()
    artifact_names = set(artifact.names())
    for row in rows:
        if len(row) != 3:
            raise ValueError(f"{artifact.path}: invalid RECORD row {row!r}")
        name, digest, size = row
        if name in recorded:
            raise ValueError(f"{artifact.path}: duplicate RECORD entry for {name!r}")
        if name not in artifact_names:
            raise ValueError(f"{artifact.path}: RECORD names a missing file {name!r}")
        recorded.add(name)
        if name == record_name:
            if digest or size:
                raise ValueError(f"{artifact.path}: RECORD must not hash or size itself")
            continue
        data = artifact.read(name)
        expected_digest = f"sha256={_urlsafe_sha256(data)}"
        if digest != expected_digest or size != str(len(data)):
            raise ValueError(f"{artifact.path}: invalid RECORD entry for {name}")
    missing = artifact_names - recorded
    if missing:
        raise ValueError(f"{artifact.path}: files missing from RECORD: {sorted(missing)}")


def _check_wheel(artifact: WheelArtifact, layout: DistributionLayout) -> None:
    names = artifact.names()
    for name in names:
        parts = PurePosixPath(name).parts
        root = parts[0]
        if root in {"vane", layout.dist_info_root, EXPECTED_LIBRARY_ROOT}:
            continue
        raise ValueError(f"{artifact.path}: Vane wheel contains conflicting Python package path {name!r}")

    optional_artifacts = [name for name in names if name.casefold().endswith(".duckdb_extension")]
    if optional_artifacts:
        raise ValueError(
            f"{artifact.path}: base Vane wheel must not contain optional extension artifacts: {optional_artifacts}"
        )

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

    required_paths = (
        "vane/py.typed",
        "vane/_native/__init__.pyi",
        "vane/_native/_func.pyi",
        "vane/_native/_sqltypes.pyi",
        "vane/_native/ray_cxx.pyi",
        "vane/sqltypes/__init__.pyi",
        "vane/udf.pyi",
        f"{layout.dist_info_root}/METADATA",
        f"{layout.dist_info_root}/WHEEL",
        f"{layout.dist_info_root}/RECORD",
    )
    for required_path in required_paths:
        _require_exact_path(names, required_path, artifact.path)
    metadata = _check_metadata(artifact, f"{layout.dist_info_root}/METADATA", layout)
    _check_no_official_duckdb_dependency(artifact, metadata)
    _check_project_dependency_metadata(artifact, metadata)
    _check_project_entry_points(artifact, layout)
    _check_wheel_license_files(artifact, metadata, layout)
    _check_wheel_record(artifact, layout)


def _validate_artifact_contents_size(artifact: Artifact) -> None:
    total_size = 0
    for member_name, member_size in artifact.member_sizes():
        if member_size < 0:
            raise ValueError(f"{artifact.path}: archive member {member_name!r} has an invalid negative size")
        if member_size > MAX_ARTIFACT_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"{artifact.path}: archive member {member_name!r} exceeds the project's 100 MiB uncompressed limit"
            )
        total_size += member_size
        if total_size > MAX_ARTIFACT_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"{artifact.path}: archive decompressed contents exceed the project's 100 MiB uncompressed limit"
            )


def _open_artifact(
    archive_input: Path | ArchiveSnapshot,
    *,
    content_rules: tuple[ContentRule, ...] = (),
) -> SdistArtifact | WheelArtifact:
    path = archive_input.source_path if isinstance(archive_input, ArchiveSnapshot) else archive_input
    if path.name.endswith(".tar.gz"):
        return SdistArtifact(archive_input, content_rules=content_rules)
    if path.suffix == ".whl":
        return WheelArtifact(archive_input, content_rules=content_rules)
    raise ValueError(f"unsupported artifact type: {path}")


def check_artifact_contents(
    archive_input: Path | ArchiveSnapshot,
    *,
    content_rules: tuple[ContentRule, ...] = (),
    text_content_rules: tuple[ContentRule, ...] = (),
) -> None:
    """Validate generic publication limits, paths, and private-content rules."""
    path = archive_input.source_path if isinstance(archive_input, ArchiveSnapshot) else archive_input
    artifact = _open_artifact(archive_input, content_rules=content_rules)
    try:
        _validate_artifact_contents_size(artifact)
        _check_paths(artifact)
        _check_internal_content(artifact, content_rules, text_content_rules)
        _check_raw_archive_content(path, artifact.raw_content_match)
    finally:
        artifact.close()


def check_artifact(
    archive_input: Path | ArchiveSnapshot,
    *,
    expected_version: Version | str,
    content_rules: tuple[ContentRule, ...] = (),
    text_content_rules: tuple[ContentRule, ...] = (),
) -> None:
    """Validate one sdist or wheel."""
    path = archive_input.source_path if isinstance(archive_input, ArchiveSnapshot) else archive_input
    layout = distribution_layout(expected_version)
    artifact_version = _artifact_version(path)
    if artifact_version != layout.version:
        raise ValueError(
            f"{path}: expected version {layout.version}, found {artifact_version} in distribution filename"
        )
    artifact = _open_artifact(archive_input, content_rules=content_rules)

    try:
        _validate_artifact_contents_size(artifact)
        _check_paths(artifact)
        _check_internal_content(artifact, content_rules, text_content_rules)
        _check_raw_archive_content(path, artifact.raw_content_match)
        if isinstance(artifact, SdistArtifact):
            _check_sdist(artifact, layout)
        else:
            _check_wheel(artifact, layout)
    finally:
        artifact.close()


def _canonical_version_argument(value: str) -> Version:
    try:
        version = Version(value)
    except InvalidVersion:
        raise argparse.ArgumentTypeError(f"invalid PEP 440 version: {value!r}") from None
    if str(version) != value:
        raise argparse.ArgumentTypeError(f"version must use canonical PEP 440 spelling: {version}")
    return version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--content-rules-manifest",
        metavar="PATH",
        help="load private exact-content rules from PATH, or from standard input with '-'",
    )
    parser.add_argument(
        "--expected-version",
        type=_canonical_version_argument,
        help="require every artifact to carry this canonical PEP 440 version",
    )
    parser.add_argument(
        "--scan-contents-only",
        action="store_true",
        help="check generic publication limits, archive paths, and private-content rules only",
    )
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    if args.scan_contents_only and args.expected_version is not None:
        parser.error("--scan-contents-only cannot be combined with --expected-version")

    content_rules: tuple[ContentRule, ...] = ()
    text_content_rules: tuple[ContentRule, ...] = ()
    if args.content_rules_manifest is not None:
        content_rules, text_content_rules = _load_content_rule_manifest(args.content_rules_manifest)

    expected_version = (
        None if args.scan_contents_only else args.expected_version or _artifact_version(args.artifacts[0])
    )
    for artifact in args.artifacts:
        if expected_version is None:
            check_artifact_contents(
                artifact,
                content_rules=content_rules,
                text_content_rules=text_content_rules,
            )
        else:
            check_artifact(
                artifact,
                expected_version=expected_version,
                content_rules=content_rules,
                text_content_rules=text_content_rules,
            )
        print(f"validated {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
