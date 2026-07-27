#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate Vane source and wheel artifacts before publication."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import hmac
import io
import ipaddress
import re
import stat
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from functools import cache
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


# Exact local matching is deterministic; scrypt raises the offline guessing cost
# for low-entropy rules but does not replace rotation or history review.
FINGERPRINT_BYTES = hashlib.sha256().digest_size
CANDIDATE_TAG_BYTES = 2
SCRYPT_N = 1 << 14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_MAX_MEMORY = 32 * 1024 * 1024
SCRYPT_SALT_PREFIX = b"vane-release-content-rule-v1\0"


@cache
def _credential_candidate_pattern(length: int) -> re.Pattern[bytes]:
    prefix = f"[!-~]{{0,{length - 1}}}"
    lookaheads = "".join(
        (
            f"(?={prefix}[a-z])",
            f"(?={prefix}[A-Z])",
            f"(?={prefix}[0-9])",
            f"(?={prefix}[^A-Za-z0-9])",
        )
    )
    window = f"[!-~]{{{length}}}"
    return re.compile(f"(?={lookaheads}({window}))".encode("ascii"))


@cache
def _ipv4_candidate_pattern(length: int) -> re.Pattern[bytes]:
    return re.compile(f"(?=([0-9.]{{{length}}}))".encode("ascii"))


@cache
def _posix_path_candidate_pattern(
    part_lengths: tuple[int, ...],
    trailing_slash: bool,
) -> re.Pattern[bytes]:
    part_patterns = [f"[\\x20-\\x2e\\x30-\\x7e]{{{length}}}" for length in part_lengths]
    candidate = "/" + "/".join(part_patterns)
    if trailing_slash:
        candidate += "/"
    return re.compile(f"(?=({candidate}))".encode("ascii"))


def _is_credential_shaped(value: bytes) -> bool:
    return (
        bool(value)
        and all(0x21 <= byte <= 0x7E for byte in value)
        and any(0x61 <= byte <= 0x7A for byte in value)
        and any(0x41 <= byte <= 0x5A for byte in value)
        and any(0x30 <= byte <= 0x39 for byte in value)
        and any(not (0x61 <= byte <= 0x7A or 0x41 <= byte <= 0x5A or 0x30 <= byte <= 0x39) for byte in value)
    )


def _is_ipv4(value: bytes) -> bool:
    try:
        ipaddress.IPv4Address(value.decode("ascii"))
    except (UnicodeDecodeError, ipaddress.AddressValueError):
        return False
    return True


def _posix_path_shape(value: bytes) -> tuple[tuple[int, ...], bool]:
    if not value.startswith(b"/") or value == b"/":
        raise ValueError("POSIX path fingerprint rule requires an absolute path")
    if any(byte < 0x20 or byte > 0x7E for byte in value):
        raise ValueError("POSIX path fingerprint rule requires printable ASCII")

    trailing_slash = value.endswith(b"/")
    body = value[1:-1] if trailing_slash else value[1:]
    parts = body.split(b"/")
    if not parts or any(not part for part in parts):
        raise ValueError("POSIX path fingerprint rule requires non-empty path parts")
    return tuple(len(part) for part in parts), trailing_slash


def _memory_hard_fingerprint(rule_id: str, value: bytes) -> bytes:
    return hashlib.scrypt(
        value,
        salt=SCRYPT_SALT_PREFIX + rule_id.encode("utf-8"),
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        maxmem=SCRYPT_MAX_MEMORY,
        dklen=FINGERPRINT_BYTES,
    )


def _candidate_tag(value: bytes) -> bytes:
    """Return a short scan prefilter; scrypt remains the authoritative check."""
    return hashlib.sha256(value).digest()[:CANDIDATE_TAG_BYTES]


@dataclass(frozen=True)
class _FingerprintRule:
    rule_id: str
    fingerprint: bytes
    candidate_tag: bytes
    length: int

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("content rule ID must not be empty")
        if self.length <= 0:
            raise ValueError("content rule length must be positive")
        if len(self.fingerprint) != FINGERPRINT_BYTES:
            raise ValueError(f"content rule fingerprint must be {FINGERPRINT_BYTES} bytes")
        if len(self.candidate_tag) != CANDIDATE_TAG_BYTES:
            raise ValueError(f"content rule candidate tag must be {CANDIDATE_TAG_BYTES} bytes")

    def _matches_candidate(self, value: bytes) -> bool:
        if not hmac.compare_digest(_candidate_tag(value), self.candidate_tag):
            return False
        return hmac.compare_digest(
            _memory_hard_fingerprint(self.rule_id, value),
            self.fingerprint,
        )


@dataclass(frozen=True)
class CredentialFingerprintRule(_FingerprintRule):
    """Match printable credential-shaped windows by memory-hard fingerprint."""

    @classmethod
    def from_value(cls, rule_id: str, value: bytes) -> CredentialFingerprintRule:
        if not _is_credential_shaped(value):
            raise ValueError("credential fingerprint rule requires a credential-shaped value")
        return cls(
            rule_id=rule_id,
            fingerprint=_memory_hard_fingerprint(rule_id, value),
            candidate_tag=_candidate_tag(value),
            length=len(value),
        )

    def matches(self, data: bytes) -> bool:
        for candidate in _credential_candidate_pattern(self.length).finditer(data):
            if self._matches_candidate(candidate.group(1)):
                return True
        return False


@dataclass(frozen=True)
class IPv4FingerprintRule(_FingerprintRule):
    """Match IPv4 candidates using a memory-hard fingerprint."""

    @classmethod
    def from_value(cls, rule_id: str, value: bytes) -> IPv4FingerprintRule:
        if not _is_ipv4(value):
            raise ValueError("IPv4 fingerprint rule requires an IPv4 address")
        return cls(
            rule_id=rule_id,
            fingerprint=_memory_hard_fingerprint(rule_id, value),
            candidate_tag=_candidate_tag(value),
            length=len(value),
        )

    def matches(self, data: bytes) -> bool:
        for match in _ipv4_candidate_pattern(self.length).finditer(data):
            candidate = match.group(1)
            if _is_ipv4(candidate) and self._matches_candidate(candidate):
                return True
        return False


@dataclass(frozen=True)
class PosixPathFingerprintRule(_FingerprintRule):
    """Match printable absolute POSIX paths using a memory-hard fingerprint."""

    part_lengths: tuple[int, ...]
    trailing_slash: bool

    @classmethod
    def from_value(cls, rule_id: str, value: bytes) -> PosixPathFingerprintRule:
        part_lengths, trailing_slash = _posix_path_shape(value)
        return cls(
            rule_id=rule_id,
            fingerprint=_memory_hard_fingerprint(rule_id, value),
            candidate_tag=_candidate_tag(value),
            length=len(value),
            part_lengths=part_lengths,
            trailing_slash=trailing_slash,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.part_lengths or any(length <= 0 for length in self.part_lengths):
            raise ValueError("POSIX path fingerprint rule requires positive part lengths")
        expected_length = sum(self.part_lengths) + len(self.part_lengths) + self.trailing_slash
        if self.length != expected_length:
            raise ValueError("POSIX path fingerprint rule length does not match its path shape")

    def matches(self, data: bytes) -> bool:
        pattern = _posix_path_candidate_pattern(self.part_lengths, self.trailing_slash)
        for match in pattern.finditer(data):
            if self._matches_candidate(match.group(1)):
                return True
        return False


@dataclass(frozen=True)
class ContentMatch:
    rule_id: str
    member_name: str


CONTENT_RULES: tuple[ContentRule, ...] = (
    IPv4FingerprintRule(
        rule_id="release-internal-ip",
        fingerprint=bytes.fromhex("ea0958373ff59a2b35ce264fda1b50cf0f9fed173878bbff134729f6338c9edd"),
        candidate_tag=bytes.fromhex("7628"),
        length=12,
    ),
    CredentialFingerprintRule(
        rule_id="release-credential",
        fingerprint=bytes.fromhex("e950be9f297db2c9f6d3a4878256321972b47bcb0540fbb6b66610b339c35fbe"),
        candidate_tag=bytes.fromhex("805b"),
        length=14,
    ),
)
TEXT_CONTENT_RULES: tuple[ContentRule, ...] = (
    PosixPathFingerprintRule(
        rule_id="release-local-path",
        fingerprint=bytes.fromhex("ba09f8cff6db233c4dd270f695d4c75dc9afd939826d017c3ffe27aaaa11be5f"),
        candidate_tag=bytes.fromhex("146b"),
        length=11,
        part_lengths=(4, 4),
        trailing_slash=True,
    ),
)


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
    metadata = _check_metadata(artifact, ".dist-info/METADATA")
    _check_wheel_license_files(artifact, metadata)
    _check_wheel_record(artifact)


def check_artifact(
    path: Path,
    *,
    content_rules: tuple[ContentRule, ...] | None = None,
    text_content_rules: tuple[ContentRule, ...] | None = None,
) -> None:
    """Validate one sdist or wheel."""
    if content_rules is None:
        content_rules = CONTENT_RULES
    if text_content_rules is None:
        text_content_rules = TEXT_CONTENT_RULES
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
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()

    for artifact in args.artifacts:
        check_artifact(artifact)
        print(f"validated {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
