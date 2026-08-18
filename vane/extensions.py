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
import threading
import weakref
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
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
_DYNAMIC_EXTENSION_PROVIDER_ENTRY_POINT_GROUP = "vane.dynamic_extension_providers"
_DYNAMIC_EXTENSION_SNAPSHOT_ENTRY_KEYS = frozenset({"descriptor", "dependency_order"})


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


@dataclass(frozen=True)
class _DynamicExtensionSnapshotEntry:
    """One ordered dynamic artifact entry retained for distributed replay."""

    descriptor: DynamicExtensionDescriptor
    dependency_order: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "descriptor": self.descriptor.to_dict(),
            "dependency_order": list(self.dependency_order),
        }


# Vane's native connections retain these entries in their Vane session so
# cursors share the exact same state. Protocol-only connections use weak state
# solely so resolver unit doubles can exercise the same invariant without
# retaining a closed connection.
_protocol_snapshot_entries_lock = threading.Lock()
_protocol_snapshot_entries_by_connection: dict[
    int,
    tuple[weakref.ReferenceType[object], list[str]],
] = {}


def _discard_protocol_snapshot_entries(connection_id: int, reference: weakref.ReferenceType[object]) -> None:
    with _protocol_snapshot_entries_lock:
        cached = _protocol_snapshot_entries_by_connection.get(connection_id)
        if cached is not None and cached[0] is reference:
            _protocol_snapshot_entries_by_connection.pop(connection_id, None)


def _protocol_snapshot_entries(connection: _ExtensionConnection, *, create: bool) -> list[str]:
    connection_id = id(connection)
    with _protocol_snapshot_entries_lock:
        cached = _protocol_snapshot_entries_by_connection.get(connection_id)
        if cached is not None and cached[0]() is connection:
            return cached[1]
        if not create:
            return []
        try:
            reference = weakref.ref(
                connection,
                lambda reference: _discard_protocol_snapshot_entries(connection_id, reference),
            )
        except TypeError as exception:
            raise DynamicExtensionError(
                "SNAPSHOT_UNAVAILABLE",
                "dynamic extension snapshots require a weak-referenceable protocol connection",
            ) from exception
        entries: list[str] = []
        _protocol_snapshot_entries_by_connection[connection_id] = (reference, entries)
        return entries


def _dynamic_extension_snapshot_entry(candidate: ResolvedDynamicExtension) -> _DynamicExtensionSnapshotEntry:
    return _DynamicExtensionSnapshotEntry(
        descriptor=candidate.descriptor,
        dependency_order=tuple(dependency.identity for dependency in candidate.descriptor.dependencies),
    )


def _canonical_snapshot_entry(entry: _DynamicExtensionSnapshotEntry) -> str:
    return json.dumps(entry.to_dict(), sort_keys=True, separators=(",", ":"))


def _parse_dynamic_extension_snapshot(snapshot: object) -> tuple[_DynamicExtensionSnapshotEntry, ...]:
    """Validate an ordered dynamic-extension snapshot without loading anything."""
    if not isinstance(snapshot, list):
        _fail("SNAPSHOT_INVALID", "dynamic_extensions must be a list")

    parsed: list[_DynamicExtensionSnapshotEntry] = []
    seen_identities: set[str] = set()
    seen_names: set[str] = set()
    available_dependencies: set[str] = set()
    for entry_index, raw_entry in enumerate(snapshot):
        if not isinstance(raw_entry, Mapping):
            _fail("SNAPSHOT_INVALID", f"dynamic_extensions[{entry_index}] must be an object")
        if set(raw_entry) != _DYNAMIC_EXTENSION_SNAPSHOT_ENTRY_KEYS:
            _fail(
                "SNAPSHOT_INVALID",
                f"dynamic_extensions[{entry_index}] must contain only descriptor and dependency_order",
            )
        raw_descriptor = raw_entry["descriptor"]
        if not isinstance(raw_descriptor, Mapping):
            _fail("SNAPSHOT_INVALID", f"dynamic_extensions[{entry_index}].descriptor must be an object")
        descriptor = DynamicExtensionDescriptor.from_dict(raw_descriptor)

        raw_dependency_order = raw_entry["dependency_order"]
        if not isinstance(raw_dependency_order, list) or any(
            not isinstance(identity, str) for identity in raw_dependency_order
        ):
            _fail(
                "SNAPSHOT_INVALID",
                f"dynamic_extensions[{entry_index}].dependency_order must be a list of strings",
            )
        dependency_order = tuple(raw_dependency_order)
        expected_dependency_order = tuple(dependency.identity for dependency in descriptor.dependencies)
        if dependency_order != expected_dependency_order:
            _fail(
                "SNAPSHOT_DEPENDENCY_ORDER",
                f"dynamic_extensions[{entry_index}] dependency_order does not match its descriptor",
            )
        missing_dependencies = [identity for identity in dependency_order if identity not in available_dependencies]
        if missing_dependencies:
            _fail(
                "SNAPSHOT_DEPENDENCY_ORDER",
                f"dynamic_extensions[{entry_index}] declares dependencies before they are loaded: "
                f"{', '.join(missing_dependencies)}",
            )
        if descriptor.identity in seen_identities:
            _fail("SNAPSHOT_INVALID", f"dynamic_extensions contains duplicate {descriptor.identity}")
        if descriptor.name in seen_names:
            _fail("SNAPSHOT_INVALID", f"dynamic_extensions contains conflicting name {descriptor.name}")

        parsed_entry = _DynamicExtensionSnapshotEntry(descriptor, dependency_order)
        parsed.append(parsed_entry)
        seen_identities.add(descriptor.identity)
        seen_names.add(descriptor.name)
        available_dependencies.add(descriptor.identity)
    return tuple(parsed)


def _normalize_dynamic_extension_snapshot(snapshot: object) -> list[dict[str, object]]:
    """Return the canonical dynamic snapshot after strict structural validation."""
    return [entry.to_dict() for entry in _parse_dynamic_extension_snapshot(snapshot)]


def _dynamic_extension_snapshot_cache_identity(snapshot: object) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return an exact, hashable artifact identity for a worker DB cache key."""
    return tuple(
        (entry.descriptor.to_json(), entry.dependency_order) for entry in _parse_dynamic_extension_snapshot(snapshot)
    )


def _native_dynamic_extension_snapshot_entries(connection: _ExtensionConnection) -> list[object] | None:
    export_entries = getattr(connection, "_export_dynamic_extension_snapshot_entries", None)
    if not callable(export_entries):
        return None
    try:
        serialized_entries = export_entries()
    except Exception as exception:
        raise DynamicExtensionError(
            "SNAPSHOT_UNAVAILABLE", "could not read the connection's dynamic extension snapshot"
        ) from exception
    if not isinstance(serialized_entries, list) or any(not isinstance(entry, str) for entry in serialized_entries):
        _fail("SNAPSHOT_INVALID", "connection dynamic extension snapshot entries must be a list of JSON strings")
    parsed_entries: list[object] = []
    for entry_index, serialized_entry in enumerate(serialized_entries):
        try:
            parsed_entries.append(json.loads(serialized_entry))
        except ValueError as exception:
            raise DynamicExtensionError(
                "SNAPSHOT_INVALID",
                f"connection dynamic extension snapshot entry {entry_index} is not valid JSON",
            ) from exception
    return parsed_entries


def _capture_dynamic_extension_snapshot(connection: _ExtensionConnection) -> list[dict[str, object]]:
    """Capture resolver-owned dynamic artifacts without serializing local paths."""
    native_entries = _native_dynamic_extension_snapshot_entries(connection)
    if native_entries is not None:
        return _normalize_dynamic_extension_snapshot(native_entries)

    serialized_entries = list(_protocol_snapshot_entries(connection, create=False))
    parsed_entries: list[object] = []
    for entry_index, serialized_entry in enumerate(serialized_entries):
        try:
            parsed_entries.append(json.loads(serialized_entry))
        except ValueError as exception:  # pragma: no cover - entries are produced locally below.
            raise DynamicExtensionError(
                "SNAPSHOT_INVALID",
                f"connection dynamic extension snapshot entry {entry_index} is not valid JSON",
            ) from exception
    return _normalize_dynamic_extension_snapshot(parsed_entries)


def _assert_native_loaded_extensions_match_snapshot(connection: _ExtensionConnection) -> None:
    """Ensure a Vane connection has no dynamic binary outside resolver state."""
    if _native_dynamic_extension_snapshot_entries(connection) is None:
        return
    expected_names = sorted(
        entry.descriptor.name
        for entry in _parse_dynamic_extension_snapshot(_capture_dynamic_extension_snapshot(connection))
    )
    try:
        rows = connection.execute(
            """
            SELECT extension_name
            FROM duckdb_extensions()
            WHERE loaded AND install_mode <> 'STATICALLY_LINKED'
            ORDER BY extension_name
            """
        ).fetchall()
    except Exception as exception:
        raise DynamicExtensionError(
            "RUNTIME_IDENTITY_UNAVAILABLE", "could not query loaded dynamic extensions"
        ) from exception
    loaded_names: list[str] = []
    for row in rows:
        if not isinstance(row, tuple) or len(row) != 1 or not isinstance(row[0], str) or not row[0]:
            _fail("RUNTIME_IDENTITY_UNAVAILABLE", "duckdb_extensions() returned an invalid extension name")
        loaded_names.append(row[0])
    if loaded_names != expected_names:
        _fail(
            "LOADED_STATE_MISMATCH",
            "loaded dynamic extensions do not exactly match resolver-owned snapshot state: "
            f"loaded={loaded_names}, recorded={expected_names}",
        )


def _assert_dynamic_extension_snapshot_can_record(
    connection: _ExtensionConnection,
    candidate: ResolvedDynamicExtension,
) -> bool:
    """Reject cross-resolver identity conflicts before DuckDB loads another binary."""
    for existing in _parse_dynamic_extension_snapshot(_capture_dynamic_extension_snapshot(connection)):
        if existing.descriptor.identity == candidate.identity:
            if existing.descriptor != candidate.descriptor:
                _fail(
                    "LOADED_IDENTITY_CONFLICT",
                    f"{candidate.identity} has conflicting immutable descriptors on this connection",
                )
            return True
        if existing.descriptor.name == candidate.descriptor.name:
            _fail(
                "LOADED_NAME_CONFLICT",
                f"{candidate.descriptor.name} is already loaded as {existing.descriptor.identity}",
            )
    return False


def _record_dynamic_extension_snapshot_entry(
    connection: _ExtensionConnection, candidate: ResolvedDynamicExtension
) -> None:
    """Persist a verified loaded artifact for later distributed snapshot capture."""
    entry = _dynamic_extension_snapshot_entry(candidate)
    serialized_entry = _canonical_snapshot_entry(entry)
    record_entry = getattr(connection, "_record_dynamic_extension_snapshot_entry", None)
    if callable(record_entry):
        try:
            record_entry(serialized_entry)
        except Exception as exception:
            raise DynamicExtensionError(
                "SNAPSHOT_UNAVAILABLE", "could not record the connection's dynamic extension snapshot"
            ) from exception
        return

    cached_entries = _protocol_snapshot_entries(connection, create=True)
    with _protocol_snapshot_entries_lock:
        if serialized_entry not in cached_entries:
            cached_entries.append(serialized_entry)


def _load_installed_dynamic_extension_providers(
    extension_names: Iterable[str],
) -> tuple[LocalExtensionProvider, ...]:
    """Load only the entry points for artifacts explicitly named by a snapshot."""
    names = tuple(sorted({_validate_extension_name(name) for name in extension_names}))
    if not names:
        return ()
    try:
        available_entry_points = tuple(entry_points(group=_DYNAMIC_EXTENSION_PROVIDER_ENTRY_POINT_GROUP))
    except Exception as exception:
        raise DynamicExtensionError(
            "PROVIDER_DISCOVERY_FAILED", "could not enumerate installed dynamic extension providers"
        ) from exception

    providers: list[LocalExtensionProvider] = []
    for name in names:
        matches = [entry_point for entry_point in available_entry_points if entry_point.name == name]
        if not matches:
            _fail(
                "PROVIDER_NOT_FOUND",
                f"no installed local provider entry point exists for {name}",
            )
        if len(matches) != 1:
            _fail(
                "PROVIDER_AMBIGUOUS",
                f"multiple installed local provider entry points exist for {name}",
            )
        try:
            provider_factory = matches[0].load()
            provider = provider_factory()
        except DynamicExtensionError:
            raise
        except Exception as exception:
            raise DynamicExtensionError(
                "PROVIDER_INVALID", f"could not initialize installed local provider for {name}"
            ) from exception
        if not isinstance(provider, LocalExtensionProvider):
            _fail("PROVIDER_INVALID", f"installed local provider for {name} did not return LocalExtensionProvider")
        if all(existing is not provider for existing in providers):
            providers.append(provider)
    return tuple(providers)


def _replay_dynamic_extension_snapshot(
    connection: _ExtensionConnection,
    snapshot: object,
    verify_native_loaded_state: bool = True,
) -> None:
    """Verify and load an immutable dynamic-extension snapshot from local wheels only."""
    expected_entries = _parse_dynamic_extension_snapshot(snapshot)
    if verify_native_loaded_state:
        _assert_native_loaded_extensions_match_snapshot(connection)
    if not expected_entries:
        return

    existing_entries = _parse_dynamic_extension_snapshot(_capture_dynamic_extension_snapshot(connection))
    expected_prefix = expected_entries[: len(existing_entries)]
    if existing_entries != expected_prefix:
        _fail(
            "WORKER_DISAGREEMENT",
            "worker dynamic extension snapshot differs from the coordinator snapshot",
        )

    providers = _load_installed_dynamic_extension_providers(entry.descriptor.name for entry in expected_entries)
    resolver = DynamicExtensionResolver(
        trusted_identities={entry.descriptor.trust_identity for entry in expected_entries},
        providers=providers,
    )

    # Verify every byte and every dependency before modifying the worker's
    # DuckDB instance. The resolver is then primed with an interrupted replay's
    # verified prefix so a retry is deterministic and does not reload it.
    resolved_by_identity: dict[str, ResolvedDynamicExtension] = {}
    for entry in expected_entries:
        for candidate in resolver.resolve(connection, entry.descriptor):
            resolved_by_identity[candidate.identity] = candidate
    loaded = resolver._loaded_for_connection(connection)
    for entry in existing_entries:
        candidate = resolved_by_identity.get(entry.descriptor.identity)
        if candidate is None:  # pragma: no cover - resolve above is exhaustive.
            _fail("WORKER_DISAGREEMENT", f"worker cannot resolve {entry.descriptor.identity}")
        loaded[candidate.descriptor.sha256] = candidate

    for entry in expected_entries[len(existing_entries) :]:
        resolver.load(connection, entry.descriptor)

    replayed_entries = _parse_dynamic_extension_snapshot(_capture_dynamic_extension_snapshot(connection))
    if replayed_entries != expected_entries:
        _fail(
            "WORKER_DISAGREEMENT",
            "worker dynamic extension identities differ after local replay",
        )


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
                    candidate_path = self._provider_artifact(
                        candidate.identity,
                        expected_descriptor=candidate,
                    ).path
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
        _assert_native_loaded_extensions_match_snapshot(connection)
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
            if _assert_dynamic_extension_snapshot_can_record(connection, candidate):
                loaded[candidate.descriptor.sha256] = candidate
                continue
            try:
                connection.load_extension(str(candidate.path))
            except Exception as exception:
                raise DynamicExtensionError(
                    "LOAD_FAILED",
                    f"failed to load {candidate.identity} from {candidate.path.name}: {exception}",
                ) from exception
            loaded[candidate.descriptor.sha256] = candidate
            _record_dynamic_extension_snapshot_entry(connection, candidate)
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

    def _provider_artifact(
        self,
        identity: str,
        *,
        expected_descriptor: DynamicExtensionDescriptor | None = None,
    ) -> LocalExtensionArtifact:
        candidates = [artifact for provider in self._providers if (artifact := provider.find(identity)) is not None]
        if not candidates:
            _fail("DEPENDENCY_NOT_FOUND", f"no trusted local provider contains {identity}")
        if len(candidates) != 1:
            _fail("ARTIFACT_AMBIGUOUS", f"multiple local providers contain {identity}")
        artifact = candidates[0]
        if expected_descriptor is not None and artifact.descriptor != expected_descriptor:
            _fail(
                "PROVIDER_DESCRIPTOR_MISMATCH",
                f"local provider descriptor does not exactly match {expected_descriptor.identity}",
            )
        return artifact

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
