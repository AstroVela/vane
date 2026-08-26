# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0
"""Trusted local resolution for loadable DuckDB extensions.

This module deliberately has no repository, download, or implicit-directory
lookup. A caller supplies an explicit artifact or an installed local provider,
and a descriptor pins the exact bytes that may be loaded into a connection.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from vane import DuckDBPyConnection

_DESCRIPTOR_FORMAT_VERSION = 1
_VALID_ABI_TYPES = frozenset({"CPP", "C_STRUCT", "C_STRUCT_UNSTABLE"})
_EXTENSION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_HEX_RE = re.compile(r"^[0-9a-f]{7,64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PLATFORM_RE = re.compile(r"^[a-z0-9_]+$")
_TRUST_IDENTITY_RE = re.compile(r"^[A-Za-z0-9._:/-]+$")
_CAPI_VERSION_RE = re.compile(r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_WINDOWS_PERMISSION_MODEL = os.name == "nt"
_WRITE_PERMISSION_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def _snapshot_mode_is_read_only(mode: int) -> bool:
    permissions = stat.S_IMODE(mode)
    if _WINDOWS_PERMISSION_MODEL:
        # Windows exposes its read-only file attribute as 0o444 through
        # st_mode; the individual POSIX read bits are not meaningful there.
        return permissions & _WRITE_PERMISSION_BITS == 0
    return permissions == 0o400


class DynamicExtensionError(RuntimeError):
    """A deterministic failure while resolving or loading an extension."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"VANE_DYNAMIC_EXTENSION_{code}: {message}")


def _fail(code: str, message: str) -> NoReturn:
    raise DynamicExtensionError(code, message)


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("DESCRIPTOR_INVALID", f"{field_name} must be a non-empty string")
    if any(character.isspace() or ord(character) < 32 for character in value):
        _fail("DESCRIPTOR_INVALID", f"{field_name} must not contain whitespace or control characters")
    return value


def _native_canonical_extension_name(extension: str) -> str:
    from vane import _native

    try:
        canonical_name = _native._dynamic_extension_canonical_name(extension)
    except Exception as exception:
        raise DynamicExtensionError(
            "RUNTIME_IDENTITY_UNAVAILABLE", "DuckDB could not canonicalize an extension name"
        ) from exception
    if not isinstance(canonical_name, str) or not canonical_name:
        _fail("RUNTIME_IDENTITY_UNAVAILABLE", "DuckDB returned an invalid canonical extension name")
    return canonical_name


def _validate_extension_name(name: object, field_name: str = "name") -> str:
    value = _require_string(name, field_name)
    if not _EXTENSION_NAME_RE.fullmatch(value):
        _fail("DESCRIPTOR_INVALID", f"{field_name} must use lowercase ASCII extension-name syntax")
    canonical_name = _native_canonical_extension_name(value)
    if canonical_name != value:
        _fail(
            "DESCRIPTOR_INVALID",
            f"{field_name} must use canonical DuckDB extension name {canonical_name!r}, not alias {value!r}",
        )
    return value


def _validate_sha256(value: object, field_name: str = "sha256") -> str:
    digest = _require_string(value, field_name)
    if not _SHA256_RE.fullmatch(digest):
        _fail("DESCRIPTOR_INVALID", f"{field_name} must be a lowercase SHA-256 digest")
    return digest


def _validate_source_id(value: object, field_name: str = "duckdb_source_id") -> str:
    source_id = _require_string(value, field_name)
    if not _HEX_RE.fullmatch(source_id):
        _fail("DESCRIPTOR_INVALID", f"{field_name} must be a lowercase Git-compatible object id")
    return source_id


def _validate_platform(value: object) -> str:
    platform = _require_string(value, "platform")
    if not _PLATFORM_RE.fullmatch(platform):
        _fail("DESCRIPTOR_INVALID", "platform must contain lowercase ASCII letters, digits, and underscores")
    return platform


def _validate_trust_identity(value: object) -> str:
    trust_identity = _require_string(value, "trust_identity")
    if not _TRUST_IDENTITY_RE.fullmatch(trust_identity):
        _fail("DESCRIPTOR_INVALID", "trust_identity contains unsupported characters")
    return trust_identity


def _validate_abi_type(value: object) -> str:
    abi_type = _require_string(value, "abi_type")
    if abi_type not in _VALID_ABI_TYPES:
        _fail("DESCRIPTOR_INVALID", f"abi_type must be one of {sorted(_VALID_ABI_TYPES)}")
    return abi_type


def _validate_extension_version(value: object) -> str:
    return _require_string(value, "extension_version")


def _parse_capi_version(value: object, field_name: str) -> tuple[int, int, int]:
    capi_version = _require_string(value, field_name)
    match = _CAPI_VERSION_RE.fullmatch(capi_version)
    if match is None:
        _fail("DESCRIPTOR_INVALID", f"{field_name} must use v<major>.<minor>.<patch> syntax")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _runtime_identity() -> tuple[str, str]:
    # Import lazily so this module can be imported from vane.__init__ while that
    # package is still being initialized.
    import vane

    source_id = _validate_source_id(getattr(vane, "__git_revision__", ""), "runtime DuckDB SourceID")
    vane_version = _require_string(getattr(vane, "__version__", ""), "runtime Vane version")
    return source_id, vane_version


@dataclass(frozen=True)
class DynamicExtensionDependency:
    """An exact dependency identity, preserved in descriptor order."""

    name: str
    extension_version: str
    sha256: str

    def __post_init__(self) -> None:
        _validate_extension_name(self.name)
        _validate_extension_version(self.extension_version)
        _validate_sha256(self.sha256)

    @property
    def identity(self) -> str:
        """Return the immutable dependency identity."""
        return f"{self.name}@{self.extension_version}#sha256:{self.sha256}"

    def to_dict(self) -> dict[str, str]:
        """Serialize this dependency for a descriptor document."""
        return {
            "name": self.name,
            "extension_version": self.extension_version,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DynamicExtensionDependency:
        """Deserialize and validate one dependency document."""
        expected_keys = {"name", "extension_version", "sha256"}
        if set(value) != expected_keys:
            _fail("DESCRIPTOR_INVALID", "dependency must contain only name, extension_version, and sha256")
        return cls(
            name=_validate_extension_name(value["name"]),
            extension_version=_validate_extension_version(value["extension_version"]),
            sha256=_validate_sha256(value["sha256"]),
        )


@dataclass(frozen=True)
class DynamicExtensionDescriptor:
    """Versioned, immutable identity of one loadable extension artifact."""

    name: str
    extension_version: str
    abi_type: str
    duckdb_source_id: str
    vane_version: str
    platform: str
    sha256: str
    trust_identity: str
    dependencies: tuple[DynamicExtensionDependency, ...] = ()
    duckdb_capi_version: str | None = None
    format_version: int = _DESCRIPTOR_FORMAT_VERSION

    def __post_init__(self) -> None:
        if type(self.format_version) is not int or self.format_version != _DESCRIPTOR_FORMAT_VERSION:
            _fail("DESCRIPTOR_INVALID", f"format_version must be {_DESCRIPTOR_FORMAT_VERSION}")
        _validate_extension_name(self.name)
        _validate_extension_version(self.extension_version)
        abi_type = _validate_abi_type(self.abi_type)
        _validate_source_id(self.duckdb_source_id)
        _require_string(self.vane_version, "vane_version")
        _validate_platform(self.platform)
        _validate_sha256(self.sha256)
        _validate_trust_identity(self.trust_identity)

        try:
            dependencies = tuple(self.dependencies)
        except TypeError as exception:
            raise DynamicExtensionError(
                "DESCRIPTOR_INVALID", "dependencies must be an iterable of DynamicExtensionDependency values"
            ) from exception
        if any(not isinstance(dependency, DynamicExtensionDependency) for dependency in dependencies):
            _fail("DESCRIPTOR_INVALID", "dependencies must contain DynamicExtensionDependency values")
        dependency_identities = [dependency.identity for dependency in dependencies]
        if len(set(dependency_identities)) != len(dependency_identities):
            _fail("DESCRIPTOR_INVALID", "dependencies must be unique and preserve declaration order")
        object.__setattr__(self, "dependencies", dependencies)

        if abi_type == "C_STRUCT":
            _parse_capi_version(self.duckdb_capi_version, "duckdb_capi_version")
        elif self.duckdb_capi_version is not None:
            _fail("DESCRIPTOR_INVALID", "duckdb_capi_version is valid only for C_STRUCT extensions")

    @property
    def identity(self) -> str:
        """Return the immutable descriptor identity."""
        return f"{self.name}@{self.extension_version}#sha256:{self.sha256}"

    def to_dict(self) -> dict[str, object]:
        """Return the canonical descriptor mapping."""
        result: dict[str, object] = {
            "format_version": self.format_version,
            "name": self.name,
            "extension_version": self.extension_version,
            "abi_type": self.abi_type,
            "duckdb_source_id": self.duckdb_source_id,
            "vane_version": self.vane_version,
            "platform": self.platform,
            "sha256": self.sha256,
            "trust_identity": self.trust_identity,
            "dependencies": [dependency.to_dict() for dependency in self.dependencies],
        }
        if self.duckdb_capi_version is not None:
            result["duckdb_capi_version"] = self.duckdb_capi_version
        return result

    def to_json(self) -> str:
        """Serialize the descriptor deterministically."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DynamicExtensionDescriptor:
        """Deserialize and validate a version-one descriptor mapping."""
        required_keys = {
            "format_version",
            "name",
            "extension_version",
            "abi_type",
            "duckdb_source_id",
            "vane_version",
            "platform",
            "sha256",
            "trust_identity",
            "dependencies",
        }
        optional_keys = {"duckdb_capi_version"}
        unknown_keys = set(value) - required_keys - optional_keys
        missing_keys = required_keys - set(value)
        if missing_keys or unknown_keys:
            details = []
            if missing_keys:
                details.append(f"missing {sorted(missing_keys)}")
            if unknown_keys:
                details.append(f"unknown {sorted(unknown_keys)}")
            _fail("DESCRIPTOR_INVALID", f"descriptor keys are invalid: {', '.join(details)}")

        dependencies_value = value["dependencies"]
        if not isinstance(dependencies_value, list):
            _fail("DESCRIPTOR_INVALID", "dependencies must be a list")
        dependencies: list[DynamicExtensionDependency] = []
        for dependency in dependencies_value:
            if not isinstance(dependency, Mapping):
                _fail("DESCRIPTOR_INVALID", "each dependency must be an object")
            dependencies.append(DynamicExtensionDependency.from_dict(dependency))

        format_version = value["format_version"]
        if type(format_version) is not int:
            _fail("DESCRIPTOR_INVALID", "format_version must be an integer")
        capi_version = value.get("duckdb_capi_version")
        if capi_version is not None and not isinstance(capi_version, str):
            _fail("DESCRIPTOR_INVALID", "duckdb_capi_version must be a string")
        return cls(
            format_version=format_version,
            name=_validate_extension_name(value["name"]),
            extension_version=_validate_extension_version(value["extension_version"]),
            abi_type=_validate_abi_type(value["abi_type"]),
            duckdb_source_id=_validate_source_id(value["duckdb_source_id"]),
            vane_version=_require_string(value["vane_version"], "vane_version"),
            platform=_validate_platform(value["platform"]),
            sha256=_validate_sha256(value["sha256"]),
            trust_identity=_validate_trust_identity(value["trust_identity"]),
            dependencies=tuple(dependencies),
            duckdb_capi_version=capi_version,
        )

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> DynamicExtensionDescriptor:
        """Deserialize one JSON descriptor document."""
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError) as exception:
            raise DynamicExtensionError("DESCRIPTOR_INVALID", "descriptor is not valid JSON") from exception
        if not isinstance(parsed, Mapping):
            _fail("DESCRIPTOR_INVALID", "descriptor JSON must contain an object")
        return cls.from_dict(parsed)


@dataclass(frozen=True)
class LocalExtensionArtifact:
    """A descriptor paired with an explicit local binary path."""

    descriptor: DynamicExtensionDescriptor
    path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, DynamicExtensionDescriptor):
            _fail("DESCRIPTOR_INVALID", "artifact descriptor must be a DynamicExtensionDescriptor")
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())


@dataclass(frozen=True)
class ResolvedDynamicExtension:
    """A verified local artifact ready to be loaded into a connection."""

    descriptor: DynamicExtensionDescriptor
    path: Path

    @property
    def identity(self) -> str:
        """Return the verified descriptor identity."""
        return self.descriptor.identity


@dataclass(frozen=True)
class _NativeExtensionMetadata:
    canonical_name: str
    abi_type: str
    platform: str
    duckdb_version: str
    duckdb_capi_version: str
    extension_version: str
    compatibility_error: str


@dataclass(frozen=True)
class _NativeLoadedExtension:
    canonical_name: str
    full_path: str
    install_mode: str
    extension_version: str


class LocalExtensionProvider:
    """An installed, explicit provider of local extension artifacts.

    Packaging code can construct this provider from files installed by a
    platform wheel. It intentionally has no directory scan or network fallback.
    """

    def __init__(self, trust_identity: str, artifacts: Iterable[LocalExtensionArtifact]):
        self._trust_identity = _validate_trust_identity(trust_identity)
        artifact_by_identity: dict[str, LocalExtensionArtifact] = {}
        for artifact in artifacts:
            if not isinstance(artifact, LocalExtensionArtifact):
                _fail("DESCRIPTOR_INVALID", "provider artifacts must be LocalExtensionArtifact values")
            if artifact.descriptor.trust_identity != self._trust_identity:
                _fail(
                    "DESCRIPTOR_INVALID",
                    "provider trust_identity must equal each artifact descriptor trust_identity",
                )
            if artifact.descriptor.identity in artifact_by_identity:
                _fail("DESCRIPTOR_INVALID", f"provider declares {artifact.descriptor.identity} more than once")
            artifact_by_identity[artifact.descriptor.identity] = artifact
        self._artifact_by_identity = artifact_by_identity

    @property
    def trust_identity(self) -> str:
        """Return the local provider trust identity."""
        return self._trust_identity

    def find(self, identity: str) -> LocalExtensionArtifact | None:
        """Return one exact artifact identity, without any fallback lookup."""
        return self._artifact_by_identity.get(identity)


class DynamicExtensionResolver:
    """Resolve and load only explicitly trusted local extension artifacts.

    The cache isolates operating-system principals. As with DuckDB's native
    extension loader, code running as the same operating-system identity is
    inside the process trust boundary.
    """

    def __init__(
        self,
        *,
        trusted_identities: Iterable[str],
        providers: Iterable[LocalExtensionProvider] = (),
        cache_directory: str | Path | None = None,
    ):
        trusted = frozenset(_validate_trust_identity(identity) for identity in trusted_identities)
        if not trusted:
            _fail("DESCRIPTOR_INVALID", "trusted_identities must not be empty")
        provider_values = tuple(providers)
        if any(not isinstance(provider, LocalExtensionProvider) for provider in provider_values):
            _fail("DESCRIPTOR_INVALID", "providers must contain LocalExtensionProvider values")
        self._trusted_identities = trusted
        self._providers = provider_values
        self._cache_directory = None if cache_directory is None else Path(cache_directory).expanduser().resolve()
        self._known_artifacts: dict[tuple[str, str], ResolvedDynamicExtension] = {}
        self._lock = threading.RLock()

    def resolve(
        self,
        connection: DuckDBPyConnection,
        descriptor: DynamicExtensionDescriptor,
        *,
        artifact: str | Path | None = None,
    ) -> tuple[ResolvedDynamicExtension, ...]:
        """Verify dependencies first and return a deterministic load order."""
        if not isinstance(descriptor, DynamicExtensionDescriptor):
            _fail("DESCRIPTOR_INVALID", "descriptor must be a DynamicExtensionDescriptor")
        explicit_artifact = Path(artifact).expanduser().resolve() if artifact is not None else None
        load_plan = self._collect_load_plan(descriptor, explicit_artifact)
        self._validate_resolved_names(candidate for candidate, _ in load_plan)

        current_source_id, current_vane_version = _runtime_identity()
        current_platform = _connection_platform(connection)
        for candidate, _ in load_plan:
            self._validate_runtime_compatibility(
                candidate,
                current_source_id=current_source_id,
                current_vane_version=current_vane_version,
                current_platform=current_platform,
            )
        cache_root = self._cache_root(connection)
        resolved = [
            ResolvedDynamicExtension(
                candidate,
                self._verify_artifact(
                    candidate,
                    candidate_path,
                    cache_root=cache_root,
                    current_platform=current_platform,
                ),
            )
            for candidate, candidate_path in load_plan
        ]
        return tuple(resolved)

    def load(
        self,
        connection: DuckDBPyConnection,
        descriptor: DynamicExtensionDescriptor,
        *,
        artifact: str | Path | None = None,
    ) -> ResolvedDynamicExtension:
        """Resolve and load dependencies before the requested extension."""
        resolved = self.resolve(connection, descriptor, artifact=artifact)
        for candidate in resolved:
            # DuckDB retains its own read handle from footer validation through
            # dlopen. Cache publishers never replace an existing entry; a
            # process acting maliciously as the cache owner is inside the same
            # trust boundary as the process it could otherwise inject into.
            loaded = _load_native_extension(candidate.path, connection)
            self._validate_loaded_provenance(candidate, loaded)
            with self._lock:
                self._known_artifacts.setdefault((candidate.identity, str(candidate.path)), candidate)
        return resolved[-1]

    def loaded_identities(self, connection: DuckDBPyConnection) -> tuple[str, ...]:
        """Return resolver-known identities that DuckDB reports as loaded."""
        with self._lock:
            known_artifacts = tuple(self._known_artifacts.values())
        loaded_by_name: dict[str, _NativeLoadedExtension | None] = {}
        identities: list[str] = []
        seen_identities: set[str] = set()
        for candidate in known_artifacts:
            name = candidate.descriptor.name
            if name not in loaded_by_name:
                loaded_by_name[name] = _loaded_native_extension(name, connection)
            loaded = loaded_by_name[name]
            if (
                loaded is not None
                and self._provenance_matches(candidate, loaded)
                and candidate.identity not in seen_identities
            ):
                identities.append(candidate.identity)
                seen_identities.add(candidate.identity)
        return tuple(identities)

    def _collect_load_plan(
        self,
        descriptor: DynamicExtensionDescriptor,
        explicit_artifact: Path | None,
    ) -> tuple[tuple[DynamicExtensionDescriptor, Path], ...]:
        plan: list[tuple[DynamicExtensionDescriptor, Path]] = []
        resolved_by_identity: dict[str, DynamicExtensionDescriptor] = {}
        visiting: set[str] = set()

        def visit(candidate: DynamicExtensionDescriptor, candidate_path: Path | None) -> None:
            if candidate.identity in visiting:
                _fail("DEPENDENCY_CYCLE", f"dependency cycle contains {candidate.identity}")
            existing = resolved_by_identity.get(candidate.identity)
            if existing is not None:
                if existing != candidate:
                    _fail(
                        "RESOLVED_IDENTITY_CONFLICT",
                        f"{candidate.identity} resolves to conflicting descriptors",
                    )
                return
            if candidate_path is None:
                provider_artifact = self._provider_artifact(candidate.identity)
                if provider_artifact.descriptor != candidate:
                    _fail(
                        "PROVIDER_DESCRIPTOR_MISMATCH",
                        f"provider descriptor for {candidate.identity} does not match the requested descriptor",
                    )
                candidate_path = provider_artifact.path

            visiting.add(candidate.identity)
            try:
                for dependency in candidate.dependencies:
                    dependency_artifact = self._provider_artifact(dependency.identity)
                    if dependency_artifact.descriptor.identity != dependency.identity:
                        _fail("DEPENDENCY_NOT_FOUND", f"provider identity does not match {dependency.identity}")
                    visit(dependency_artifact.descriptor, dependency_artifact.path)
            finally:
                visiting.discard(candidate.identity)
            resolved_by_identity[candidate.identity] = candidate
            plan.append((candidate, candidate_path))

        visit(descriptor, explicit_artifact)
        return tuple(plan)

    @staticmethod
    def _validate_resolved_names(resolved: Iterable[DynamicExtensionDescriptor]) -> None:
        by_name: dict[str, DynamicExtensionDescriptor] = {}
        for candidate in resolved:
            existing = by_name.get(candidate.name)
            if existing is not None and existing.identity != candidate.identity:
                _fail(
                    "RESOLVED_NAME_CONFLICT",
                    f"dependency graph resolves {candidate.name} as both {existing.identity} and {candidate.identity}",
                )
            by_name[candidate.name] = candidate

    def _provider_artifact(self, identity: str) -> LocalExtensionArtifact:
        candidates = [artifact for provider in self._providers if (artifact := provider.find(identity)) is not None]
        if not candidates:
            _fail("DEPENDENCY_NOT_FOUND", f"no trusted local provider contains {identity}")
        if len(candidates) != 1:
            _fail("ARTIFACT_AMBIGUOUS", f"multiple local providers contain {identity}")
        return candidates[0]

    def _verify_artifact(
        self,
        descriptor: DynamicExtensionDescriptor,
        artifact_path: Path,
        *,
        cache_root: Path,
        current_platform: str,
    ) -> Path:
        expected_filename = f"{descriptor.name}.duckdb_extension"
        if artifact_path.name != expected_filename:
            _fail("NAME_MISMATCH", f"{descriptor.identity} must use artifact filename {expected_filename}")
        if not artifact_path.is_file():
            _fail("ARTIFACT_NOT_FOUND", f"artifact does not exist: {artifact_path}")

        digest_directory = cache_root / descriptor.sha256
        artifact_directory = digest_directory / descriptor.name
        snapshot_path = artifact_directory / expected_filename
        if snapshot_path.exists() or snapshot_path.is_symlink():
            source_digest = _sha256_file(artifact_path)
            self._validate_digest(descriptor, source_digest)
            self._validate_cached_snapshot(
                descriptor,
                snapshot_path,
                current_platform=current_platform,
            )
            return snapshot_path

        staging_directory = Path(tempfile.mkdtemp(prefix=f".{descriptor.name}-", dir=cache_root))
        staging_path = staging_directory / expected_filename
        try:
            actual_digest = _copy_and_hash_artifact(artifact_path, staging_path)
            self._validate_digest(descriptor, actual_digest)
            try:
                staging_path.chmod(0o400)
            except OSError as exception:
                raise DynamicExtensionError(
                    "ARTIFACT_SNAPSHOT_FAILED",
                    f"could not make verified artifact snapshot read-only: {artifact_path}",
                ) from exception
            try:
                staging_mode = staging_path.stat().st_mode
            except OSError as exception:
                raise DynamicExtensionError(
                    "ARTIFACT_SNAPSHOT_FAILED",
                    f"could not inspect verified artifact snapshot permissions: {staging_path}",
                ) from exception
            if not _snapshot_mode_is_read_only(staging_mode):
                _fail(
                    "ARTIFACT_SNAPSHOT_FAILED",
                    f"verified artifact snapshot is not read-only: {staging_path}",
                )
            metadata = _inspect_native_extension(staging_path)
            self._validate_native_metadata(descriptor, metadata, current_platform=current_platform)
            self._prepare_cache_directory(digest_directory)
            self._prepare_cache_directory(artifact_directory)
            try:
                os.link(staging_path, snapshot_path, follow_symlinks=False)
            except FileExistsError:
                self._validate_cached_snapshot(
                    descriptor,
                    snapshot_path,
                    current_platform=current_platform,
                )
            except OSError as exception:
                raise DynamicExtensionError(
                    "ARTIFACT_SNAPSHOT_FAILED",
                    f"could not publish verified artifact snapshot: {snapshot_path}",
                ) from exception
            return snapshot_path
        finally:
            shutil.rmtree(staging_directory, ignore_errors=True)

    def _validate_runtime_compatibility(
        self,
        descriptor: DynamicExtensionDescriptor,
        *,
        current_source_id: str,
        current_vane_version: str,
        current_platform: str,
    ) -> None:
        if descriptor.trust_identity not in self._trusted_identities:
            _fail("TRUST_IDENTITY_UNTRUSTED", f"{descriptor.trust_identity} is not in trusted_identities")
        if descriptor.duckdb_source_id != current_source_id:
            _fail(
                "SOURCE_ID_MISMATCH",
                f"{descriptor.identity} requires SourceID {descriptor.duckdb_source_id}, runtime has {current_source_id}",
            )
        if descriptor.vane_version != current_vane_version:
            _fail(
                "VANE_VERSION_MISMATCH",
                f"{descriptor.identity} requires Vane {descriptor.vane_version}, runtime has {current_vane_version}",
            )
        if descriptor.platform != current_platform:
            _fail(
                "PLATFORM_MISMATCH",
                f"{descriptor.identity} targets {descriptor.platform}, runtime is {current_platform}",
            )

    @staticmethod
    def _validate_digest(descriptor: DynamicExtensionDescriptor, actual_digest: str) -> None:
        if actual_digest != descriptor.sha256:
            _fail(
                "DIGEST_MISMATCH",
                f"{descriptor.identity} expected SHA-256 {descriptor.sha256}, got {actual_digest}",
            )

    def _validate_cached_snapshot(
        self,
        descriptor: DynamicExtensionDescriptor,
        snapshot_path: Path,
        *,
        current_platform: str,
    ) -> None:
        if snapshot_path.is_symlink() or not snapshot_path.is_file():
            _fail("ARTIFACT_CACHE_CORRUPT", f"verified artifact cache entry is invalid: {snapshot_path}")
        try:
            snapshot_mode = snapshot_path.stat().st_mode
        except OSError as exception:
            raise DynamicExtensionError(
                "ARTIFACT_CACHE_CORRUPT", f"could not inspect verified artifact cache entry: {snapshot_path}"
            ) from exception
        if not _snapshot_mode_is_read_only(snapshot_mode):
            _fail("ARTIFACT_CACHE_CORRUPT", f"verified artifact cache entry is not read-only: {snapshot_path}")
        if _sha256_file(snapshot_path) != descriptor.sha256:
            _fail("ARTIFACT_CACHE_CORRUPT", f"verified artifact cache digest is invalid: {snapshot_path}")
        metadata = _inspect_native_extension(snapshot_path)
        self._validate_native_metadata(descriptor, metadata, current_platform=current_platform)

    @staticmethod
    def _validate_native_metadata(
        descriptor: DynamicExtensionDescriptor,
        metadata: _NativeExtensionMetadata,
        *,
        current_platform: str,
    ) -> None:
        if metadata.canonical_name != descriptor.name:
            _fail(
                "NAME_MISMATCH",
                f"{descriptor.identity} resolves natively as {metadata.canonical_name}",
            )
        if metadata.abi_type != descriptor.abi_type:
            _fail(
                "ABI_MISMATCH",
                f"{descriptor.identity} declares ABI {descriptor.abi_type}, artifact footer has {metadata.abi_type}",
            )
        if metadata.platform != descriptor.platform:
            _fail(
                "PLATFORM_MISMATCH",
                f"{descriptor.identity} declares platform {descriptor.platform}, artifact footer has {metadata.platform}",
            )
        if metadata.platform != current_platform:
            _fail(
                "PLATFORM_MISMATCH",
                f"{descriptor.identity} targets {metadata.platform}, runtime is {current_platform}",
            )
        if metadata.extension_version != descriptor.extension_version:
            _fail(
                "EXTENSION_VERSION_MISMATCH",
                f"{descriptor.identity} footer version is {metadata.extension_version}",
            )
        if descriptor.abi_type == "C_STRUCT":
            if metadata.duckdb_capi_version != descriptor.duckdb_capi_version:
                _fail(
                    "CAPI_VERSION_MISMATCH",
                    f"{descriptor.identity} requires C API {descriptor.duckdb_capi_version}, "
                    f"artifact footer has {metadata.duckdb_capi_version}",
                )
        elif metadata.duckdb_version != descriptor.duckdb_source_id:
            _fail(
                "SOURCE_ID_MISMATCH",
                f"{descriptor.identity} footer SourceID is {metadata.duckdb_version}",
            )
        if metadata.compatibility_error:
            if descriptor.abi_type == "C_STRUCT":
                _fail(
                    "CAPI_VERSION_MISMATCH",
                    f"{descriptor.identity} is incompatible with DuckDB: {metadata.compatibility_error}",
                )
            _fail(
                "DUCKDB_INCOMPATIBLE",
                f"{descriptor.identity} is incompatible with DuckDB: {metadata.compatibility_error}",
            )

    def _cache_root(self, connection: DuckDBPyConnection) -> Path:
        root = self._cache_directory
        if root is None:
            vane_directory = self._prepare_cache_directory(_native_extension_directory(connection) / ".vane")
            root = vane_directory / "verified-v1"
        return self._prepare_cache_directory(root)

    @staticmethod
    def _prepare_cache_directory(path: Path) -> Path:
        try:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as exception:
            raise DynamicExtensionError(
                "ARTIFACT_SNAPSHOT_FAILED", f"could not create verified artifact cache directory: {path}"
            ) from exception
        if path.is_symlink() or not path.is_dir():
            _fail("ARTIFACT_CACHE_CORRUPT", f"verified artifact cache directory is invalid: {path}")
        if _WINDOWS_PERMISSION_MODEL:
            _secure_native_cache_directory(path)
            return path
        try:
            directory_mode = path.stat().st_mode
        except OSError as exception:
            raise DynamicExtensionError(
                "ARTIFACT_CACHE_CORRUPT", f"could not inspect verified artifact cache directory: {path}"
            ) from exception
        if stat.S_IMODE(directory_mode) != 0o700:
            _fail("ARTIFACT_CACHE_CORRUPT", f"verified artifact cache directory is not private: {path}")
        return path

    @staticmethod
    def _provenance_matches(
        candidate: ResolvedDynamicExtension,
        loaded: _NativeLoadedExtension,
    ) -> bool:
        return (
            loaded.canonical_name == candidate.descriptor.name
            and loaded.full_path == str(candidate.path)
            and loaded.install_mode == "NOT_INSTALLED"
            and loaded.extension_version == candidate.descriptor.extension_version
        )

    @classmethod
    def _validate_loaded_provenance(
        cls,
        candidate: ResolvedDynamicExtension,
        loaded: _NativeLoadedExtension,
    ) -> None:
        if cls._provenance_matches(candidate, loaded):
            return
        _fail(
            "LOADED_ARTIFACT_CONFLICT",
            f"DuckDB loaded {loaded.canonical_name!r} from {loaded.full_path or loaded.install_mode!r}, "
            f"not verified artifact {candidate.path}",
        )


def create_dynamic_extension_descriptor(
    artifact: str | Path,
    *,
    name: str,
    trust_identity: str,
    dependencies: Iterable[DynamicExtensionDependency] = (),
    vane_version: str | None = None,
) -> DynamicExtensionDescriptor:
    """Create a version-one descriptor directly from a built local artifact."""
    artifact_path = Path(artifact).expanduser().resolve()
    validated_name = _validate_extension_name(name)
    expected_filename = f"{validated_name}.duckdb_extension"
    if artifact_path.name != expected_filename:
        _fail("NAME_MISMATCH", f"descriptor name {validated_name} requires artifact filename {expected_filename}")
    if not artifact_path.is_file():
        _fail("ARTIFACT_NOT_FOUND", f"artifact does not exist: {artifact_path}")
    with tempfile.TemporaryDirectory(prefix="vane-extension-descriptor-") as snapshot_directory:
        snapshot_path = Path(snapshot_directory) / expected_filename
        artifact_digest = _copy_and_hash_artifact(artifact_path, snapshot_path)
        metadata = _inspect_native_extension(snapshot_path)
    if metadata.canonical_name != validated_name:
        _fail("NAME_MISMATCH", f"DuckDB canonicalizes {artifact_path.name} as {metadata.canonical_name}")
    if metadata.abi_type not in _VALID_ABI_TYPES:
        _fail("FOOTER_INVALID", f"artifact footer has unsupported ABI {metadata.abi_type!r}")
    _validate_platform(metadata.platform)
    _validate_extension_version(metadata.extension_version)
    source_id, runtime_vane_version = _runtime_identity()
    descriptor_vane_version = (
        runtime_vane_version if vane_version is None else _require_string(vane_version, "vane_version")
    )
    try:
        dependency_values = tuple(dependencies)
    except TypeError as exception:
        raise DynamicExtensionError(
            "DESCRIPTOR_INVALID", "dependencies must be an iterable of DynamicExtensionDependency values"
        ) from exception
    if metadata.abi_type == "C_STRUCT":
        _parse_capi_version(metadata.duckdb_capi_version, "artifact footer duckdb_capi_version")
        return DynamicExtensionDescriptor(
            name=validated_name,
            extension_version=metadata.extension_version,
            abi_type=metadata.abi_type,
            duckdb_source_id=source_id,
            vane_version=descriptor_vane_version,
            platform=metadata.platform,
            sha256=artifact_digest,
            trust_identity=trust_identity,
            dependencies=dependency_values,
            duckdb_capi_version=metadata.duckdb_capi_version,
        )
    return DynamicExtensionDescriptor(
        name=validated_name,
        extension_version=metadata.extension_version,
        abi_type=metadata.abi_type,
        duckdb_source_id=_validate_source_id(metadata.duckdb_version, "artifact footer SourceID"),
        vane_version=descriptor_vane_version,
        platform=metadata.platform,
        sha256=artifact_digest,
        trust_identity=trust_identity,
        dependencies=dependency_values,
    )


def _connection_platform(connection: DuckDBPyConnection) -> str:
    try:
        row = connection.execute("SELECT platform FROM pragma_platform()").fetchone()
    except Exception as exception:
        raise DynamicExtensionError(
            "RUNTIME_IDENTITY_UNAVAILABLE", "could not query the DuckDB platform"
        ) from exception
    if not isinstance(row, tuple) or len(row) != 1 or not isinstance(row[0], str):
        _fail("RUNTIME_IDENTITY_UNAVAILABLE", "pragma_platform() did not return one platform string")
    return _validate_platform(row[0])


def _native_extension_directory(connection: DuckDBPyConnection) -> Path:
    from vane import _native

    try:
        directory = _native._dynamic_extension_directory(connection=connection)
    except Exception as exception:
        raise DynamicExtensionError(
            "ARTIFACT_SNAPSHOT_FAILED", "DuckDB could not provide its configured extension directory"
        ) from exception
    if not isinstance(directory, str) or not directory or "\0" in directory:
        _fail("ARTIFACT_SNAPSHOT_FAILED", "DuckDB returned an invalid extension directory")
    return Path(directory).expanduser().resolve()


def _secure_native_cache_directory(path: Path) -> None:
    from vane import _native

    try:
        _native._secure_dynamic_extension_cache_directory(str(path))
    except Exception as exception:
        raise DynamicExtensionError(
            "ARTIFACT_CACHE_CORRUPT",
            f"could not apply a private Windows DACL to verified artifact cache directory: {path}",
        ) from exception


def _inspect_native_extension(path: Path) -> _NativeExtensionMetadata:
    from vane import _native

    try:
        value = _native._inspect_dynamic_extension(str(path))
    except Exception as exception:
        raise DynamicExtensionError(
            "FOOTER_INVALID", f"DuckDB could not inspect extension artifact: {path}"
        ) from exception
    expected_keys = {
        "canonical_name",
        "abi_type",
        "platform",
        "duckdb_version",
        "duckdb_capi_version",
        "extension_version",
        "compatibility_error",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        _fail("FOOTER_INVALID", "DuckDB returned invalid extension metadata")
    fields = {key: value[key] for key in expected_keys}
    if any(not isinstance(field, str) for field in fields.values()):
        _fail("FOOTER_INVALID", "DuckDB returned non-string extension metadata")
    return _NativeExtensionMetadata(**fields)


def _parse_native_loaded_extension(value: object) -> _NativeLoadedExtension:
    expected_keys = {"canonical_name", "full_path", "install_mode", "extension_version"}
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        _fail("LOADED_STATE_UNAVAILABLE", "DuckDB returned invalid loaded-extension provenance")
    fields = {key: value[key] for key in expected_keys}
    if any(not isinstance(field, str) for field in fields.values()):
        _fail("LOADED_STATE_UNAVAILABLE", "DuckDB returned non-string loaded-extension provenance")
    return _NativeLoadedExtension(**fields)


def _load_native_extension(path: Path, connection: DuckDBPyConnection) -> _NativeLoadedExtension:
    from vane import _native

    try:
        value = _native._load_dynamic_extension(str(path), connection=connection)
    except Exception as exception:
        raise DynamicExtensionError("LOAD_FAILED", f"DuckDB could not load verified artifact: {path}") from exception
    return _parse_native_loaded_extension(value)


def _loaded_native_extension(
    extension: str,
    connection: DuckDBPyConnection,
) -> _NativeLoadedExtension | None:
    from vane import _native

    try:
        value = _native._loaded_dynamic_extension(extension, connection=connection)
    except Exception as exception:
        raise DynamicExtensionError(
            "LOADED_STATE_UNAVAILABLE", f"DuckDB could not report loaded state for {extension}"
        ) from exception
    if value is None:
        return None
    return _parse_native_loaded_extension(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as artifact_file:
            for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exception:
        raise DynamicExtensionError("ARTIFACT_NOT_FOUND", f"could not read extension artifact: {path}") from exception
    return digest.hexdigest()


def _copy_and_hash_artifact(source: Path, destination: Path) -> str:
    digest = hashlib.sha256()
    try:
        with source.open("rb") as source_file, destination.open("xb") as destination_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
                destination_file.write(chunk)
            destination_file.flush()
            os.fsync(destination_file.fileno())
    except OSError as exception:
        raise DynamicExtensionError(
            "ARTIFACT_SNAPSHOT_FAILED", f"could not create verified artifact snapshot from {source}"
        ) from exception
    return digest.hexdigest()


__all__ = [
    "DynamicExtensionDependency",
    "DynamicExtensionDescriptor",
    "DynamicExtensionError",
    "DynamicExtensionResolver",
    "LocalExtensionArtifact",
    "LocalExtensionProvider",
    "ResolvedDynamicExtension",
    "create_dynamic_extension_descriptor",
]
