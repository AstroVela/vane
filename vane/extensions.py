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
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Protocol

_DESCRIPTOR_FORMAT_VERSION = 1
_EXTENSION_FOOTER_SIZE = 512
_EXTENSION_FOOTER_FIELD_SIZE = 32
_EXTENSION_FOOTER_FIELD_COUNT = 8
_VALID_ABI_TYPES = frozenset({"CPP", "C_STRUCT", "C_STRUCT_UNSTABLE"})
_EXTENSION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_HEX_RE = re.compile(r"^[0-9a-f]{7,64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PLATFORM_RE = re.compile(r"^[a-z0-9_]+$")
_TRUST_IDENTITY_RE = re.compile(r"^[A-Za-z0-9._:/-]+$")


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


def _validate_extension_name(name: object, field_name: str = "name") -> str:
    value = _require_string(name, field_name)
    if not _EXTENSION_NAME_RE.fullmatch(value):
        _fail("DESCRIPTOR_INVALID", f"{field_name} must use lowercase ASCII extension-name syntax")
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
            _require_string(self.duckdb_capi_version, "duckdb_capi_version")
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
class _ExtensionFooter:
    abi_type: str
    platform: str
    engine_identity: str
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


class _ExtensionConnection(Protocol):
    def execute(self, query: str) -> Any: ...

    def load_extension(self, extension: str) -> None: ...


class DynamicExtensionResolver:
    """Resolve and load only explicitly trusted local extension artifacts."""

    def __init__(
        self,
        *,
        trusted_identities: Iterable[str],
        providers: Iterable[LocalExtensionProvider] = (),
    ):
        trusted = frozenset(_validate_trust_identity(identity) for identity in trusted_identities)
        if not trusted:
            _fail("DESCRIPTOR_INVALID", "trusted_identities must not be empty")
        self._trusted_identities = trusted
        self._providers = tuple(providers)
        # Keep the connection beside its id. A bare id() cache can be reused by
        # a later connection after the original object is collected.
        self._loaded_by_connection: dict[int, tuple[_ExtensionConnection, dict[str, ResolvedDynamicExtension]]] = {}

    def resolve(
        self,
        connection: _ExtensionConnection,
        descriptor: DynamicExtensionDescriptor,
        *,
        artifact: str | Path | None = None,
    ) -> tuple[ResolvedDynamicExtension, ...]:
        """Verify dependencies first and return a deterministic load order."""
        current_source_id, current_vane_version = _runtime_identity()
        current_platform = _connection_platform(connection)
        resolved: list[ResolvedDynamicExtension] = []
        resolved_identities: set[str] = set()
        visiting: set[str] = set()

        def visit(candidate: DynamicExtensionDescriptor, candidate_path: Path | None) -> None:
            if candidate.identity in resolved_identities:
                return
            if candidate.identity in visiting:
                _fail("DEPENDENCY_CYCLE", f"dependency cycle contains {candidate.identity}")
            visiting.add(candidate.identity)
            try:
                if candidate_path is None:
                    candidate_path = self._provider_artifact(candidate.identity).path
                self._verify_artifact(
                    candidate,
                    candidate_path,
                    current_source_id=current_source_id,
                    current_vane_version=current_vane_version,
                    current_platform=current_platform,
                )
                for dependency in candidate.dependencies:
                    dependency_artifact = self._provider_artifact(dependency.identity)
                    if dependency_artifact.descriptor.identity != dependency.identity:
                        _fail("DEPENDENCY_NOT_FOUND", f"provider identity does not match {dependency.identity}")
                    visit(dependency_artifact.descriptor, dependency_artifact.path)
                resolved.append(ResolvedDynamicExtension(candidate, candidate_path))
                resolved_identities.add(candidate.identity)
            finally:
                visiting.discard(candidate.identity)

        explicit_artifact = Path(artifact).expanduser().resolve() if artifact is not None else None
        visit(descriptor, explicit_artifact)
        return tuple(resolved)

    def load(
        self,
        connection: _ExtensionConnection,
        descriptor: DynamicExtensionDescriptor,
        *,
        artifact: str | Path | None = None,
    ) -> ResolvedDynamicExtension:
        """Resolve and load dependencies before the requested extension."""
        resolved = self.resolve(connection, descriptor, artifact=artifact)
        loaded = self._loaded_for_connection(connection)
        for candidate in resolved:
            existing = loaded.get(candidate.descriptor.sha256)
            if existing is not None:
                if existing.identity != candidate.identity:
                    _fail(
                        "LOADED_IDENTITY_CONFLICT",
                        f"digest {candidate.descriptor.sha256} is already cached as {existing.identity}",
                    )
                continue
            for loaded_candidate in loaded.values():
                if loaded_candidate.descriptor.name == candidate.descriptor.name:
                    _fail(
                        "LOADED_NAME_CONFLICT",
                        f"{candidate.descriptor.name} is already loaded as {loaded_candidate.identity}",
                    )
            try:
                connection.load_extension(str(candidate.path))
            except Exception as exception:
                raise DynamicExtensionError(
                    "LOAD_FAILED",
                    f"failed to load {candidate.identity} from {candidate.path.name}: {exception}",
                ) from exception
            loaded[candidate.descriptor.sha256] = candidate
        return resolved[-1]

    def loaded_identities(self, connection: _ExtensionConnection) -> tuple[str, ...]:
        """Return cached artifact identities for one connection in load order."""
        cached = self._loaded_by_connection.get(id(connection))
        if cached is None or cached[0] is not connection:
            return ()
        return tuple(candidate.identity for candidate in cached[1].values())

    def _loaded_for_connection(self, connection: _ExtensionConnection) -> dict[str, ResolvedDynamicExtension]:
        connection_id = id(connection)
        cached = self._loaded_by_connection.get(connection_id)
        if cached is not None and cached[0] is connection:
            return cached[1]
        loaded: dict[str, ResolvedDynamicExtension] = {}
        self._loaded_by_connection[connection_id] = (connection, loaded)
        return loaded

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
        expected_filename = f"{descriptor.name}.duckdb_extension"
        if artifact_path.name != expected_filename:
            _fail("NAME_MISMATCH", f"{descriptor.identity} must use artifact filename {expected_filename}")
        if not artifact_path.is_file():
            _fail("ARTIFACT_NOT_FOUND", f"artifact does not exist: {artifact_path}")
        actual_digest = _sha256_file(artifact_path)
        if actual_digest != descriptor.sha256:
            _fail(
                "DIGEST_MISMATCH",
                f"{descriptor.identity} expected SHA-256 {descriptor.sha256}, got {actual_digest}",
            )
        footer = _parse_extension_footer(artifact_path)
        if footer.abi_type != descriptor.abi_type:
            _fail(
                "ABI_MISMATCH",
                f"{descriptor.identity} declares ABI {descriptor.abi_type}, artifact footer has {footer.abi_type}",
            )
        if footer.platform != descriptor.platform:
            _fail(
                "PLATFORM_MISMATCH",
                f"{descriptor.identity} declares platform {descriptor.platform}, artifact footer has {footer.platform}",
            )
        if footer.extension_version != descriptor.extension_version:
            _fail(
                "EXTENSION_VERSION_MISMATCH",
                f"{descriptor.identity} footer version is {footer.extension_version}",
            )
        if descriptor.abi_type == "C_STRUCT":
            if footer.engine_identity != descriptor.duckdb_capi_version:
                _fail(
                    "CAPI_VERSION_MISMATCH",
                    f"{descriptor.identity} requires C API {descriptor.duckdb_capi_version}, "
                    f"artifact footer has {footer.engine_identity}",
                )
        elif footer.engine_identity != descriptor.duckdb_source_id:
            _fail(
                "SOURCE_ID_MISMATCH",
                f"{descriptor.identity} footer SourceID is {footer.engine_identity}",
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
    footer = _parse_extension_footer(artifact_path)
    source_id, runtime_vane_version = _runtime_identity()
    descriptor_vane_version = (
        runtime_vane_version if vane_version is None else _require_string(vane_version, "vane_version")
    )
    if footer.abi_type == "C_STRUCT":
        return DynamicExtensionDescriptor(
            name=validated_name,
            extension_version=footer.extension_version,
            abi_type=footer.abi_type,
            duckdb_source_id=source_id,
            vane_version=descriptor_vane_version,
            platform=footer.platform,
            sha256=_sha256_file(artifact_path),
            trust_identity=trust_identity,
            dependencies=tuple(dependencies),
            duckdb_capi_version=footer.engine_identity,
        )
    return DynamicExtensionDescriptor(
        name=validated_name,
        extension_version=footer.extension_version,
        abi_type=footer.abi_type,
        duckdb_source_id=_validate_source_id(footer.engine_identity, "artifact footer SourceID"),
        vane_version=descriptor_vane_version,
        platform=footer.platform,
        sha256=_sha256_file(artifact_path),
        trust_identity=trust_identity,
        dependencies=tuple(dependencies),
    )


def _connection_platform(connection: _ExtensionConnection) -> str:
    try:
        row = connection.execute("SELECT platform FROM pragma_platform()").fetchone()
    except Exception as exception:
        raise DynamicExtensionError(
            "RUNTIME_IDENTITY_UNAVAILABLE", "could not query the DuckDB platform"
        ) from exception
    if not isinstance(row, tuple) or len(row) != 1 or not isinstance(row[0], str):
        _fail("RUNTIME_IDENTITY_UNAVAILABLE", "pragma_platform() did not return one platform string")
    return _validate_platform(row[0])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as artifact_file:
            for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exception:
        raise DynamicExtensionError("ARTIFACT_NOT_FOUND", f"could not read artifact: {path}") from exception
    return digest.hexdigest()


def _parse_extension_footer(path: Path) -> _ExtensionFooter:
    try:
        with path.open("rb") as artifact_file:
            artifact_file.seek(0, 2)
            if artifact_file.tell() < _EXTENSION_FOOTER_SIZE:
                _fail("FOOTER_INVALID", f"artifact is smaller than {_EXTENSION_FOOTER_SIZE} bytes: {path.name}")
            artifact_file.seek(-_EXTENSION_FOOTER_SIZE, 2)
            footer = artifact_file.read(_EXTENSION_FOOTER_SIZE)
    except OSError as exception:
        raise DynamicExtensionError("ARTIFACT_NOT_FOUND", f"could not read artifact: {path}") from exception
    if len(footer) != _EXTENSION_FOOTER_SIZE:
        _fail("FOOTER_INVALID", f"could not read a complete extension footer from {path.name}")

    fields = []
    for field_index in range(_EXTENSION_FOOTER_FIELD_COUNT):
        start = field_index * _EXTENSION_FOOTER_FIELD_SIZE
        raw_field = footer[start : start + _EXTENSION_FOOTER_FIELD_SIZE]
        try:
            field = raw_field.decode("ascii").rstrip("\0")
        except UnicodeDecodeError as exception:
            raise DynamicExtensionError(
                "FOOTER_INVALID", f"extension footer contains non-ASCII field {field_index}"
            ) from exception
        if "\0" in field:
            _fail("FOOTER_INVALID", f"extension footer contains an embedded NUL in field {field_index}")
        fields.append(field)

    if fields[7] != "4":
        _fail("FOOTER_INVALID", f"artifact footer magic is invalid: {path.name}")
    abi_type = fields[3] or "CPP"
    if abi_type not in _VALID_ABI_TYPES:
        _fail("FOOTER_INVALID", f"artifact footer has unsupported ABI {abi_type!r}")
    platform = fields[6]
    if _PLATFORM_RE.fullmatch(platform) is None:
        _fail("FOOTER_INVALID", "artifact footer platform is invalid")
    engine_identity = fields[5]
    if not engine_identity or any(character.isspace() or ord(character) < 32 for character in engine_identity):
        _fail("FOOTER_INVALID", "artifact footer engine identity is invalid")
    extension_version = fields[4]
    if not extension_version or any(character.isspace() or ord(character) < 32 for character in extension_version):
        _fail("FOOTER_INVALID", "artifact footer extension version is invalid")
    return _ExtensionFooter(
        abi_type=abi_type,
        platform=platform,
        engine_identity=engine_identity,
        extension_version=extension_version,
    )


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
