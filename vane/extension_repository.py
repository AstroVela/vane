# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Signed, explicit repositories for optional Vane extension artifacts.

The dynamic-extension resolver intentionally knows only about local files.
This module is the separate transport layer: a caller pins a repository URL,
repository identity, signer identity, and Ed25519 public key, then materializes
one verified dependency closure into an explicit local provider.  It never
consults DuckDB's public extension repository and has no implicit cache or
network fallback.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

from vane.extensions import (
    DynamicExtensionDescriptor,
    DynamicExtensionError,
    DynamicExtensionResolver,
    LocalExtensionArtifact,
    LocalExtensionProvider,
    _connection_platform,
    _runtime_identity,
    _validate_extension_name,
    _validate_trust_identity,
)

_REPOSITORY_FORMAT_VERSION = 1
_INDEX_FILE_NAME = "repository.json"
_SIGNATURE_FILE_NAME = "repository.sig.json"
_CACHE_FORMAT_DIRECTORY = "vane-extension-repository-v1"
_MAX_INDEX_BYTES = 10 * 1024 * 1024
_MAX_SIGNATURE_BYTES = 64 * 1024
_MAX_DESCRIPTOR_BYTES = 1024 * 1024
_MAX_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
_ED25519_PRIVATE_KEY_BYTES = 32
_ED25519_PUBLIC_KEY_BYTES = 32
_ED25519_SIGNATURE_BYTES = 64


def _fail(code: str, message: str) -> NoReturn:
    raise DynamicExtensionError(code, message)


def _require_string(value: object, field_name: str, *, code: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(code, f"{field_name} must be a non-empty string")
    if any(character.isspace() or ord(character) < 32 for character in value):
        _fail(code, f"{field_name} must not contain whitespace or control characters")
    return value


def _validate_repository_id(value: object, *, code: str) -> str:
    repository_id = _require_string(value, "repository_id", code=code)
    if not all(character.isascii() and (character.isalnum() or character in "._:/-") for character in repository_id):
        _fail(code, "repository_id contains unsupported characters")
    return repository_id


def _validate_signer_identity(value: object, *, code: str) -> str:
    try:
        return _validate_trust_identity(value)
    except DynamicExtensionError as exception:
        raise DynamicExtensionError(code, "signer_identity is invalid") from exception


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")


def _descriptor_key(descriptor: DynamicExtensionDescriptor) -> str:
    return hashlib.sha256(descriptor.to_json().encode("utf-8")).hexdigest()


def _artifact_path(descriptor: DynamicExtensionDescriptor) -> str:
    return f"artifacts/{_descriptor_key(descriptor)}/{descriptor.name}.duckdb_extension"


def _descriptor_path(descriptor: DynamicExtensionDescriptor) -> str:
    return f"artifacts/{_descriptor_key(descriptor)}/{descriptor.name}.dynamic-extension.json"


@dataclass(frozen=True)
class RepositoryArtifact:
    """One descriptor and the immutable repository paths that carry it."""

    descriptor: DynamicExtensionDescriptor
    artifact_path: str
    descriptor_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, DynamicExtensionDescriptor):
            _fail("REPOSITORY_MANIFEST_INVALID", "repository artifact descriptor is invalid")
        if self.artifact_path != _artifact_path(self.descriptor):
            _fail("REPOSITORY_MANIFEST_INVALID", "repository artifact path does not match its descriptor")
        if self.descriptor_path != _descriptor_path(self.descriptor):
            _fail("REPOSITORY_MANIFEST_INVALID", "repository descriptor path does not match its descriptor")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical signed index entry."""
        return {
            "artifact": self.artifact_path,
            "descriptor": self.descriptor.to_dict(),
            "descriptor_file": self.descriptor_path,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RepositoryArtifact:
        """Deserialize one strict repository-index entry."""
        expected_keys = {"artifact", "descriptor", "descriptor_file"}
        if set(value) != expected_keys:
            _fail("REPOSITORY_MANIFEST_INVALID", "repository artifact keys are invalid")
        descriptor_value = value["descriptor"]
        if not isinstance(descriptor_value, Mapping):
            _fail("REPOSITORY_MANIFEST_INVALID", "repository artifact descriptor must be an object")
        artifact_path = _require_string(value["artifact"], "artifact", code="REPOSITORY_MANIFEST_INVALID")
        descriptor_path = _require_string(
            value["descriptor_file"], "descriptor_file", code="REPOSITORY_MANIFEST_INVALID"
        )
        return cls(
            descriptor=DynamicExtensionDescriptor.from_dict(descriptor_value),
            artifact_path=artifact_path,
            descriptor_path=descriptor_path,
        )


@dataclass(frozen=True)
class InstalledExtension:
    """A verified, local dependency closure selected from a signed repository."""

    descriptor: DynamicExtensionDescriptor
    provider: LocalExtensionProvider
    artifacts: tuple[LocalExtensionArtifact, ...]

    def resolver(self) -> DynamicExtensionResolver:
        """Create a resolver over exactly this materialized local provider."""
        return DynamicExtensionResolver(
            trusted_identities=(self.provider.trust_identity,),
            providers=(self.provider,),
        )


@dataclass(frozen=True)
class PublishedExtensionRepository:
    """The immutable repository state written by :func:`publish_extension_repository`."""

    path: Path
    repository_id: str
    signer_identity: str
    public_key: bytes

    @property
    def public_key_base64(self) -> str:
        """Return the pinned Ed25519 public key in a CLI-friendly encoding."""
        return base64.b64encode(self.public_key).decode("ascii")


@dataclass(frozen=True)
class _RepositoryManifest:
    repository_id: str
    artifacts: tuple[RepositoryArtifact, ...]

    @property
    def by_identity(self) -> dict[str, RepositoryArtifact]:
        return {artifact.descriptor.identity: artifact for artifact in self.artifacts}

    def to_dict(self) -> dict[str, object]:
        return {
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "format_version": _REPOSITORY_FORMAT_VERSION,
            "repository_id": self.repository_id,
        }


@dataclass(frozen=True)
class _WheelArtifact:
    descriptor: DynamicExtensionDescriptor
    descriptor_bytes: bytes
    artifact_bytes: bytes


class SignedExtensionRepository:
    """Materialize one extension from an explicitly pinned signed repository.

    ``repository_url`` must use ``https://`` in production.  ``file://`` is
    supported for an explicitly mounted or locally produced repository; plain
    HTTP and default repository discovery are deliberately unsupported.
    """

    def __init__(
        self,
        *,
        repository_url: str,
        repository_id: str,
        signer_identity: str,
        trusted_public_key: bytes,
        cache_directory: str | Path,
        timeout_seconds: float = 30.0,
    ):
        self._repository_url = _normalize_repository_url(repository_url)
        self._repository_id = _validate_repository_id(repository_id, code="REPOSITORY_CONFIG_INVALID")
        self._signer_identity = _validate_signer_identity(signer_identity, code="REPOSITORY_CONFIG_INVALID")
        self._trusted_public_key = _validate_public_key(trusted_public_key, code="REPOSITORY_CONFIG_INVALID")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            _fail("REPOSITORY_CONFIG_INVALID", "timeout_seconds must be a positive number")
        self._timeout_seconds = float(timeout_seconds)
        self._cache_directory = Path(cache_directory).expanduser().resolve()

    @property
    def repository_id(self) -> str:
        """Return the configured immutable repository identity."""
        return self._repository_id

    @property
    def signer_identity(self) -> str:
        """Return the configured signing and descriptor trust identity."""
        return self._signer_identity

    def install(self, extension_name: str, *, extension_version: str | None = None) -> InstalledExtension:
        """Download, cache, and validate one exact local extension closure.

        The result does not load an extension or change DuckDB configuration.
        Call :meth:`InstalledExtension.resolver` and pass that local provider to
        the ordinary dynamic-extension resolver when a connection should load it.
        """
        requested_name = _validate_extension_name(extension_name)
        requested_version = _validate_requested_version(extension_version)
        manifest = self._read_signed_manifest()
        root = self._select_compatible_artifact(manifest, requested_name, requested_version)
        closure = self._dependency_closure(manifest, root)
        artifacts = tuple(self._materialize(artifact) for artifact in closure)
        provider = LocalExtensionProvider(self._signer_identity, artifacts)

        # The repository signature binds transport metadata.  The existing
        # resolver additionally binds the exact local bytes to this Vane/DuckDB
        # runtime, its footer, platform, descriptor trust identity, and closure.
        self._validate_local_closure(provider, root.descriptor)
        return InstalledExtension(descriptor=root.descriptor, provider=provider, artifacts=artifacts)

    def _read_signed_manifest(self) -> _RepositoryManifest:
        index = self._download_json(_INDEX_FILE_NAME, _MAX_INDEX_BYTES, "repository index")
        signature = self._download_json(_SIGNATURE_FILE_NAME, _MAX_SIGNATURE_BYTES, "repository signature")
        signature_bytes = _parse_signature(signature, expected_signer=self._signer_identity)
        _verify_ed25519(self._trusted_public_key, signature_bytes, _canonical_json_bytes(index))
        manifest = _parse_manifest(index)
        if manifest.repository_id != self._repository_id:
            _fail(
                "REPOSITORY_ID_MISMATCH",
                f"repository index declares {manifest.repository_id!r}, expected {self._repository_id!r}",
            )
        for artifact in manifest.artifacts:
            if artifact.descriptor.trust_identity != self._signer_identity:
                _fail(
                    "REPOSITORY_TRUST_IDENTITY_MISMATCH",
                    f"{artifact.descriptor.identity} is not trusted by signer {self._signer_identity}",
                )
        return manifest

    def _select_compatible_artifact(
        self,
        manifest: _RepositoryManifest,
        extension_name: str,
        extension_version: str | None,
    ) -> RepositoryArtifact:
        named = [artifact for artifact in manifest.artifacts if artifact.descriptor.name == extension_name]
        if extension_version is not None:
            named = [artifact for artifact in named if artifact.descriptor.extension_version == extension_version]
        if not named:
            requested = extension_name if extension_version is None else f"{extension_name}@{extension_version}"
            _fail("REPOSITORY_ARTIFACT_NOT_FOUND", f"repository has no artifact for {requested}")

        source_id, vane_version = _runtime_identity()
        import vane

        connection = vane.connect()
        try:
            platform = _connection_platform(connection)
        finally:
            connection.close()
        compatible = [
            artifact
            for artifact in named
            if artifact.descriptor.duckdb_source_id == source_id
            and artifact.descriptor.vane_version == vane_version
            and artifact.descriptor.platform == platform
        ]
        if not compatible:
            requested = extension_name if extension_version is None else f"{extension_name}@{extension_version}"
            _fail("REPOSITORY_ARTIFACT_INCOMPATIBLE", f"repository has no runtime-compatible artifact for {requested}")
        if len(compatible) != 1:
            identities = ", ".join(artifact.descriptor.identity for artifact in compatible)
            _fail(
                "REPOSITORY_ARTIFACT_AMBIGUOUS",
                f"repository has multiple runtime-compatible artifacts for {extension_name}: {identities}",
            )
        return compatible[0]

    def _dependency_closure(
        self,
        manifest: _RepositoryManifest,
        root: RepositoryArtifact,
    ) -> tuple[RepositoryArtifact, ...]:
        by_identity = manifest.by_identity
        resolved: list[RepositoryArtifact] = []
        resolved_identities: set[str] = set()
        visiting: set[str] = set()

        def visit(artifact: RepositoryArtifact) -> None:
            identity = artifact.descriptor.identity
            if identity in resolved_identities:
                return
            if identity in visiting:
                _fail("REPOSITORY_DEPENDENCY_CYCLE", f"repository dependency cycle contains {identity}")
            visiting.add(identity)
            try:
                for dependency in artifact.descriptor.dependencies:
                    candidate = by_identity.get(dependency.identity)
                    if candidate is None:
                        _fail(
                            "REPOSITORY_DEPENDENCY_NOT_FOUND",
                            f"repository does not retain dependency {dependency.identity}",
                        )
                    visit(candidate)
                resolved.append(artifact)
                resolved_identities.add(identity)
            finally:
                visiting.discard(identity)

        visit(root)
        return tuple(resolved)

    def _materialize(self, entry: RepositoryArtifact) -> LocalExtensionArtifact:
        cache_directory, artifact_cache_path, descriptor_cache_path = self._cache_paths(entry)
        if self._cached_entry_is_valid(entry, artifact_cache_path, descriptor_cache_path):
            return LocalExtensionArtifact(entry.descriptor, artifact_cache_path)

        self._remove_invalid_cache_files(artifact_cache_path, descriptor_cache_path)
        self._ensure_cache_directory(cache_directory)
        descriptor_bytes = self._download_bytes(entry.descriptor_path, _MAX_DESCRIPTOR_BYTES, "extension descriptor")
        try:
            downloaded_descriptor = DynamicExtensionDescriptor.from_json(descriptor_bytes)
        except DynamicExtensionError as exception:
            raise DynamicExtensionError(
                "REPOSITORY_DESCRIPTOR_INVALID",
                f"repository descriptor for {entry.descriptor.identity} is invalid: {exception}",
            ) from exception
        if downloaded_descriptor != entry.descriptor:
            _fail(
                "REPOSITORY_DESCRIPTOR_MISMATCH",
                f"repository descriptor file does not match signed index for {entry.descriptor.identity}",
            )
        _atomic_write(descriptor_cache_path, descriptor_bytes)
        self._download_artifact(entry, artifact_cache_path)
        return LocalExtensionArtifact(entry.descriptor, artifact_cache_path)

    def _cache_paths(self, entry: RepositoryArtifact) -> tuple[Path, Path, Path]:
        repository_key = hashlib.sha256(self._repository_id.encode("ascii")).hexdigest()
        descriptor_key = _descriptor_key(entry.descriptor)
        cache_directory = self._cache_directory / _CACHE_FORMAT_DIRECTORY / repository_key / descriptor_key
        return (
            cache_directory,
            cache_directory / f"{entry.descriptor.name}.duckdb_extension",
            cache_directory / f"{entry.descriptor.name}.dynamic-extension.json",
        )

    def _cached_entry_is_valid(
        self,
        entry: RepositoryArtifact,
        artifact_path: Path,
        descriptor_path: Path,
    ) -> bool:
        if artifact_path.is_symlink() or descriptor_path.is_symlink():
            return False
        try:
            cached_descriptor = DynamicExtensionDescriptor.from_json(descriptor_path.read_bytes())
        except (DynamicExtensionError, OSError):
            return False
        if cached_descriptor != entry.descriptor:
            return False
        try:
            return _sha256_file(artifact_path) == entry.descriptor.sha256
        except OSError:
            return False

    def _remove_invalid_cache_files(self, artifact_path: Path, descriptor_path: Path) -> None:
        for path in (artifact_path, descriptor_path):
            try:
                if path.is_symlink() or path.is_file():
                    path.unlink()
                elif path.exists():
                    _fail("REPOSITORY_CACHE_INVALID", f"cache path is not a file: {path}")
            except OSError as exception:
                raise DynamicExtensionError(
                    "REPOSITORY_CACHE_INVALID", f"could not clear cache file: {path}"
                ) from exception

    def _ensure_cache_directory(self, cache_directory: Path) -> None:
        try:
            cache_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exception:
            raise DynamicExtensionError(
                "REPOSITORY_CACHE_INVALID", f"could not create cache directory: {cache_directory}"
            ) from exception
        if cache_directory.is_symlink() or not cache_directory.is_dir():
            _fail("REPOSITORY_CACHE_INVALID", f"cache path is not a real directory: {cache_directory}")
        root = (self._cache_directory / _CACHE_FORMAT_DIRECTORY).resolve()
        resolved_cache = cache_directory.resolve()
        if not resolved_cache.is_relative_to(root):
            _fail("REPOSITORY_CACHE_INVALID", "cache path escaped the configured cache directory")

    def _download_artifact(self, entry: RepositoryArtifact, destination: Path) -> None:
        response = self._open_response(entry.artifact_path, "extension artifact")
        temporary_path: Path | None = None
        try:
            with response:
                _check_content_length(response, _MAX_ARTIFACT_BYTES, "extension artifact")
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=destination.parent,
                    prefix=f".{entry.descriptor.name}-",
                    suffix=".tmp",
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    digest = hashlib.sha256()
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > _MAX_ARTIFACT_BYTES:
                            _fail("REPOSITORY_DOWNLOAD_FAILED", "extension artifact exceeds the maximum allowed size")
                        digest.update(chunk)
                        temporary_file.write(chunk)
                if digest.hexdigest() != entry.descriptor.sha256:
                    _fail(
                        "REPOSITORY_DIGEST_MISMATCH",
                        f"downloaded bytes do not match signed descriptor for {entry.descriptor.identity}",
                    )
                os.replace(temporary_path, destination)
                temporary_path = None
        except DynamicExtensionError:
            raise
        except OSError as exception:
            raise DynamicExtensionError(
                "REPOSITORY_CACHE_INVALID", f"could not store {entry.descriptor.identity} in the local cache"
            ) from exception
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _download_json(self, relative_path: str, maximum_size: int, label: str) -> object:
        payload = self._download_bytes(relative_path, maximum_size, label)
        return _load_json_document(payload, label=label, code="REPOSITORY_MANIFEST_INVALID")

    def _download_bytes(self, relative_path: str, maximum_size: int, label: str) -> bytes:
        response = self._open_response(relative_path, label)
        try:
            with response:
                _check_content_length(response, maximum_size, label)
                chunks: list[bytes] = []
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > maximum_size:
                        _fail("REPOSITORY_DOWNLOAD_FAILED", f"{label} exceeds the maximum allowed size")
                    chunks.append(chunk)
                return b"".join(chunks)
        except DynamicExtensionError:
            raise
        except OSError as exception:
            raise DynamicExtensionError("REPOSITORY_DOWNLOAD_FAILED", f"could not read {label}") from exception

    def _open_response(self, relative_path: str, label: str) -> Any:
        url = self._url_for(relative_path)
        request = urllib.request.Request(url, headers={"Accept": "application/json, application/octet-stream"})
        try:
            response = urllib.request.urlopen(request, timeout=self._timeout_seconds)
        except (urllib.error.URLError, OSError) as exception:
            raise DynamicExtensionError(
                "REPOSITORY_DOWNLOAD_FAILED", f"could not download {label} from configured repository"
            ) from exception
        response_url = response.geturl()
        if not _url_is_within_repository(response_url, self._repository_url):
            response.close()
            _fail("REPOSITORY_REDIRECT_REJECTED", f"configured repository redirected {label} outside its pinned URL")
        return response

    def _url_for(self, relative_path: str) -> str:
        if not _is_safe_repository_path(relative_path):
            _fail("REPOSITORY_MANIFEST_INVALID", f"repository path is unsafe: {relative_path!r}")
        return self._repository_url + urllib.parse.quote(relative_path, safe="/")

    def _validate_local_closure(self, provider: LocalExtensionProvider, descriptor: DynamicExtensionDescriptor) -> None:
        import vane

        connection = vane.connect()
        try:
            DynamicExtensionResolver(
                trusted_identities=(self._signer_identity,),
                providers=(provider,),
            ).resolve(connection, descriptor)
        finally:
            connection.close()


def publish_extension_repository(
    *,
    extension_wheels: Iterable[str | Path],
    output_directory: str | Path,
    repository_id: str,
    signer_identity: str,
    private_key: bytes,
) -> PublishedExtensionRepository:
    """Publish immutable extension-wheel contents to a signed local repository.

    Every supplied wheel contributes the exact descriptor and extension bytes it
    already packages.  Existing signed entries are retained and cannot be
    replaced with new bytes, which keeps previously selected descriptors
    reproducible after later repository releases.
    """
    validated_repository_id = _validate_repository_id(repository_id, code="REPOSITORY_CONFIG_INVALID")
    validated_signer_identity = _validate_signer_identity(signer_identity, code="REPOSITORY_CONFIG_INVALID")
    signing_key = _ed25519_private_key(private_key)
    public_key = _ed25519_public_key_bytes(signing_key.public_key())
    destination = Path(output_directory).expanduser().resolve()
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exception:
        raise DynamicExtensionError(
            "REPOSITORY_PUBLISH_FAILED", f"could not create repository directory: {destination}"
        ) from exception
    if destination.is_symlink() or not destination.is_dir():
        _fail("REPOSITORY_PUBLISH_FAILED", f"repository output path is not a real directory: {destination}")

    existing = _read_existing_manifest(destination, validated_repository_id, validated_signer_identity, public_key)
    entries = existing.by_identity
    supplied_wheels = tuple(extension_wheels)
    if not supplied_wheels:
        _fail("REPOSITORY_CONFIG_INVALID", "extension_wheels must not be empty")
    pending: dict[str, tuple[RepositoryArtifact, _WheelArtifact]] = {}
    for extension_wheel in supplied_wheels:
        wheel_artifact = _read_extension_wheel(extension_wheel)
        descriptor = wheel_artifact.descriptor
        if descriptor.trust_identity != validated_signer_identity:
            _fail(
                "REPOSITORY_TRUST_IDENTITY_MISMATCH",
                f"{descriptor.identity} has trust identity {descriptor.trust_identity!r}, expected {validated_signer_identity!r}",
            )
        entry = RepositoryArtifact(
            descriptor=descriptor,
            artifact_path=_artifact_path(descriptor),
            descriptor_path=_descriptor_path(descriptor),
        )
        existing_entry = entries.get(descriptor.identity)
        if existing_entry is not None and existing_entry != entry:
            _fail("REPOSITORY_PUBLISH_FAILED", f"repository identity changed paths: {descriptor.identity}")
        prior_pending = pending.get(descriptor.identity)
        if prior_pending is not None and prior_pending[1] != wheel_artifact:
            _fail("REPOSITORY_PUBLISH_FAILED", f"supplied wheels disagree about {descriptor.identity}")
        entries[descriptor.identity] = entry
        pending[descriptor.identity] = (entry, wheel_artifact)

    artifacts = tuple(sorted(entries.values(), key=lambda artifact: artifact.descriptor.identity))
    manifest = _RepositoryManifest(repository_id=validated_repository_id, artifacts=artifacts)
    _validate_manifest_closure(manifest, validated_signer_identity)
    for entry, wheel_artifact in pending.values():
        _write_immutable_file(destination / entry.artifact_path, wheel_artifact.artifact_bytes)
        _write_immutable_file(destination / entry.descriptor_path, wheel_artifact.descriptor_bytes)
    index = manifest.to_dict()
    index_bytes = _canonical_json_bytes(index) + b"\n"
    signature = {
        "algorithm": "ed25519",
        "format_version": _REPOSITORY_FORMAT_VERSION,
        "signature": base64.b64encode(signing_key.sign(_canonical_json_bytes(index))).decode("ascii"),
        "signer_identity": validated_signer_identity,
    }
    _atomic_write(destination / _INDEX_FILE_NAME, index_bytes)
    _atomic_write(destination / _SIGNATURE_FILE_NAME, _canonical_json_bytes(signature) + b"\n")
    return PublishedExtensionRepository(
        path=destination,
        repository_id=validated_repository_id,
        signer_identity=validated_signer_identity,
        public_key=public_key,
    )


def read_ed25519_private_key(path: str | Path) -> bytes:
    """Read a raw, base64, or PEM-encoded Ed25519 private key file."""
    data = _read_key_file(path, "private")
    if len(data) == _ED25519_PRIVATE_KEY_BYTES:
        return data
    stripped = data.strip()
    if stripped.startswith(b"-----BEGIN"):
        serialization, ed25519 = _cryptography_serialization()
        try:
            key = serialization.load_pem_private_key(stripped, password=None)
        except (TypeError, ValueError) as exception:
            raise DynamicExtensionError("REPOSITORY_KEY_INVALID", "private key PEM is invalid") from exception
        if not isinstance(key, ed25519.Ed25519PrivateKey):
            _fail("REPOSITORY_KEY_INVALID", "private key must be an Ed25519 key")
        return key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
    return _decode_base64_key(stripped, "private")


def read_ed25519_public_key(path: str | Path) -> bytes:
    """Read a raw, base64, or PEM-encoded Ed25519 public key file."""
    data = _read_key_file(path, "public")
    if len(data) == _ED25519_PUBLIC_KEY_BYTES:
        return data
    stripped = data.strip()
    if stripped.startswith(b"-----BEGIN"):
        serialization, ed25519 = _cryptography_serialization()
        try:
            key = serialization.load_pem_public_key(stripped)
        except (TypeError, ValueError) as exception:
            raise DynamicExtensionError("REPOSITORY_KEY_INVALID", "public key PEM is invalid") from exception
        if not isinstance(key, ed25519.Ed25519PublicKey):
            _fail("REPOSITORY_KEY_INVALID", "public key must be an Ed25519 key")
        return key.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
    return _decode_base64_key(stripped, "public")


def _normalize_repository_url(value: object) -> str:
    url = _require_string(value, "repository_url", code="REPOSITORY_CONFIG_INVALID")
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError as exception:
        raise DynamicExtensionError("REPOSITORY_CONFIG_INVALID", "repository_url is invalid") from exception
    if parsed.scheme not in {"https", "file"}:
        _fail("REPOSITORY_CONFIG_INVALID", "repository_url must use https:// or file://")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        _fail("REPOSITORY_CONFIG_INVALID", "repository_url must not contain credentials, query, or fragment")
    if parsed.scheme == "https" and not parsed.netloc:
        _fail("REPOSITORY_CONFIG_INVALID", "https repository_url requires a host")
    if parsed.scheme == "file" and parsed.netloc not in {"", "localhost"}:
        _fail("REPOSITORY_CONFIG_INVALID", "file repository_url must not name a remote host")
    if not parsed.path.startswith("/"):
        _fail("REPOSITORY_CONFIG_INVALID", "repository_url must contain an absolute path")
    normalized_path = parsed.path.rstrip("/") + "/"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, normalized_path, "", ""))


def _url_is_within_repository(candidate: str, repository_url: str) -> bool:
    parsed_candidate = urllib.parse.urlsplit(candidate)
    parsed_repository = urllib.parse.urlsplit(repository_url)
    return (
        parsed_candidate.scheme == parsed_repository.scheme
        and parsed_candidate.netloc == parsed_repository.netloc
        and parsed_candidate.path.startswith(parsed_repository.path)
        and not parsed_candidate.query
        and not parsed_candidate.fragment
    )


def _is_safe_repository_path(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        bool(value)
        and "\\" not in value
        and not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts)
        and all(ord(character) >= 32 and ord(character) != 127 for character in value)
    )


def _validate_requested_version(value: str | None) -> str | None:
    if value is None:
        return None
    return _require_string(value, "extension_version", code="REPOSITORY_CONFIG_INVALID")


def _load_json_document(value: str | bytes, *, label: str, code: str) -> object:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        parsed: dict[str, object] = {}
        for key, member in pairs:
            if key in parsed:
                _fail(code, f"{label} contains a duplicate key: {key!r}")
            parsed[key] = member
        return parsed

    def reject_nonfinite_constant(constant: str) -> NoReturn:
        _fail(code, f"{label} contains unsupported JSON constant {constant!r}")

    try:
        text = value.decode("utf-8") if isinstance(value, bytes) else value
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite_constant,
        )
    except DynamicExtensionError:
        raise
    except (UnicodeDecodeError, ValueError) as exception:
        raise DynamicExtensionError(code, f"{label} is not valid UTF-8 JSON") from exception


def _parse_manifest(value: object) -> _RepositoryManifest:
    if not isinstance(value, Mapping):
        _fail("REPOSITORY_MANIFEST_INVALID", "repository index must contain an object")
    expected_keys = {"artifacts", "format_version", "repository_id"}
    if set(value) != expected_keys:
        _fail("REPOSITORY_MANIFEST_INVALID", "repository index keys are invalid")
    if type(value["format_version"]) is not int or value["format_version"] != _REPOSITORY_FORMAT_VERSION:
        _fail("REPOSITORY_MANIFEST_INVALID", f"repository index format_version must be {_REPOSITORY_FORMAT_VERSION}")
    repository_id = _validate_repository_id(value["repository_id"], code="REPOSITORY_MANIFEST_INVALID")
    artifacts_value = value["artifacts"]
    if not isinstance(artifacts_value, list):
        _fail("REPOSITORY_MANIFEST_INVALID", "repository index artifacts must be a list")
    artifacts: list[RepositoryArtifact] = []
    identities: set[str] = set()
    paths: set[str] = set()
    for artifact_value in artifacts_value:
        if not isinstance(artifact_value, Mapping):
            _fail("REPOSITORY_MANIFEST_INVALID", "each repository artifact must be an object")
        artifact = RepositoryArtifact.from_dict(artifact_value)
        identity = artifact.descriptor.identity
        if identity in identities:
            _fail("REPOSITORY_MANIFEST_INVALID", f"repository declares {identity} more than once")
        if artifact.artifact_path in paths or artifact.descriptor_path in paths:
            _fail("REPOSITORY_MANIFEST_INVALID", "repository artifact paths must be unique")
        identities.add(identity)
        paths.update((artifact.artifact_path, artifact.descriptor_path))
        artifacts.append(artifact)
    if artifacts != sorted(artifacts, key=lambda artifact: artifact.descriptor.identity):
        _fail("REPOSITORY_MANIFEST_INVALID", "repository artifacts must be sorted by immutable identity")
    return _RepositoryManifest(repository_id=repository_id, artifacts=tuple(artifacts))


def _parse_signature(value: object, *, expected_signer: str) -> bytes:
    if not isinstance(value, Mapping):
        _fail("REPOSITORY_SIGNATURE_INVALID", "repository signature must contain an object")
    expected_keys = {"algorithm", "format_version", "signature", "signer_identity"}
    if set(value) != expected_keys:
        _fail("REPOSITORY_SIGNATURE_INVALID", "repository signature keys are invalid")
    if type(value["format_version"]) is not int or value["format_version"] != _REPOSITORY_FORMAT_VERSION:
        _fail(
            "REPOSITORY_SIGNATURE_INVALID", f"repository signature format_version must be {_REPOSITORY_FORMAT_VERSION}"
        )
    if value["algorithm"] != "ed25519":
        _fail("REPOSITORY_SIGNATURE_INVALID", "repository signature algorithm must be ed25519")
    signer_identity = _validate_signer_identity(value["signer_identity"], code="REPOSITORY_SIGNATURE_INVALID")
    if signer_identity != expected_signer:
        _fail("REPOSITORY_SIGNER_UNTRUSTED", f"repository was signed by {signer_identity!r}")
    signature = _decode_base64(value["signature"], "repository signature", "REPOSITORY_SIGNATURE_INVALID")
    if len(signature) != _ED25519_SIGNATURE_BYTES:
        _fail("REPOSITORY_SIGNATURE_INVALID", "repository signature has an invalid Ed25519 length")
    return signature


def _validate_manifest_closure(manifest: _RepositoryManifest, signer_identity: str) -> None:
    by_identity = manifest.by_identity
    release_identities: dict[tuple[str, str, str, str, str], str] = {}
    for artifact in manifest.artifacts:
        descriptor = artifact.descriptor
        if descriptor.trust_identity != signer_identity:
            _fail(
                "REPOSITORY_TRUST_IDENTITY_MISMATCH",
                f"{descriptor.identity} does not match publisher signer {signer_identity}",
            )
        release_coordinate = (
            descriptor.name,
            descriptor.extension_version,
            descriptor.duckdb_source_id,
            descriptor.vane_version,
            descriptor.platform,
        )
        previous_identity = release_identities.setdefault(release_coordinate, descriptor.identity)
        if previous_identity != descriptor.identity:
            _fail(
                "REPOSITORY_VERSION_CONFLICT",
                f"repository has more than one immutable artifact for {descriptor.name}@{descriptor.extension_version}",
            )
        for dependency in descriptor.dependencies:
            dependency_entry = by_identity.get(dependency.identity)
            if dependency_entry is None:
                _fail("REPOSITORY_DEPENDENCY_NOT_FOUND", f"repository does not retain dependency {dependency.identity}")
            dependency_descriptor = dependency_entry.descriptor
            if (
                dependency_descriptor.duckdb_source_id != descriptor.duckdb_source_id
                or dependency_descriptor.vane_version != descriptor.vane_version
                or dependency_descriptor.platform != descriptor.platform
            ):
                _fail(
                    "REPOSITORY_DEPENDENCY_INCOMPATIBLE",
                    f"dependency {dependency.identity} is incompatible with {descriptor.identity}",
                )

    visiting: set[str] = set()
    resolved: set[str] = set()

    def visit(artifact: RepositoryArtifact) -> None:
        identity = artifact.descriptor.identity
        if identity in resolved:
            return
        if identity in visiting:
            _fail("REPOSITORY_DEPENDENCY_CYCLE", f"repository dependency cycle contains {identity}")
        visiting.add(identity)
        try:
            for dependency in artifact.descriptor.dependencies:
                visit(by_identity[dependency.identity])
            resolved.add(identity)
        finally:
            visiting.discard(identity)

    for artifact in manifest.artifacts:
        visit(artifact)


def _read_existing_manifest(
    directory: Path,
    repository_id: str,
    signer_identity: str,
    public_key: bytes,
) -> _RepositoryManifest:
    index_path = directory / _INDEX_FILE_NAME
    signature_path = directory / _SIGNATURE_FILE_NAME
    if not index_path.exists() and not signature_path.exists():
        return _RepositoryManifest(repository_id=repository_id, artifacts=())
    if not index_path.is_file() or not signature_path.is_file():
        _fail("REPOSITORY_PUBLISH_FAILED", "existing repository index and signature must both be regular files")
    try:
        index_bytes = index_path.read_bytes()
        signature_bytes = signature_path.read_bytes()
    except OSError as exception:
        raise DynamicExtensionError(
            "REPOSITORY_PUBLISH_FAILED", "could not read existing repository index"
        ) from exception
    index = _load_json_document(index_bytes, label="existing repository index", code="REPOSITORY_PUBLISH_FAILED")
    signature = _load_json_document(
        signature_bytes,
        label="existing repository signature",
        code="REPOSITORY_PUBLISH_FAILED",
    )
    signature_bytes = _parse_signature(signature, expected_signer=signer_identity)
    _verify_ed25519(public_key, signature_bytes, _canonical_json_bytes(index))
    manifest = _parse_manifest(index)
    if manifest.repository_id != repository_id:
        _fail(
            "REPOSITORY_PUBLISH_FAILED",
            f"existing repository declares {manifest.repository_id!r}, expected {repository_id!r}",
        )
    _validate_manifest_closure(manifest, signer_identity)
    for artifact in manifest.artifacts:
        _validate_published_files(directory, artifact)
    return manifest


def _read_extension_wheel(value: str | Path) -> _WheelArtifact:
    wheel_path = Path(value).expanduser().resolve()
    if wheel_path.suffix != ".whl":
        _fail("REPOSITORY_PUBLISH_FAILED", f"extension wheel must have a .whl suffix: {wheel_path}")
    try:
        with zipfile.ZipFile(wheel_path) as wheel:
            names = [member.filename for member in wheel.infolist() if not member.is_dir()]
            if len(names) != len(set(names)):
                _fail("REPOSITORY_PUBLISH_FAILED", f"extension wheel has duplicate paths: {wheel_path.name}")
            if any(not _is_safe_repository_path(name) for name in names):
                _fail("REPOSITORY_PUBLISH_FAILED", f"extension wheel has an unsafe path: {wheel_path.name}")
            descriptor_names = [
                name
                for name in names
                if name.startswith("vane_extensions/") and name.endswith(".dynamic-extension.json")
            ]
            if len(descriptor_names) != 1:
                _fail("REPOSITORY_PUBLISH_FAILED", "extension wheel must contain exactly one extension descriptor")
            descriptor_bytes = wheel.read(descriptor_names[0])
            descriptor = DynamicExtensionDescriptor.from_json(descriptor_bytes)
            expected_descriptor_name = f"vane_extensions/{descriptor.name}/{descriptor.name}.dynamic-extension.json"
            if descriptor_names[0] != expected_descriptor_name:
                _fail("REPOSITORY_PUBLISH_FAILED", "extension wheel descriptor path does not match descriptor name")
            artifact_names = [name for name in names if name.endswith(".duckdb_extension")]
            expected_artifact_name = f"vane_extensions/{descriptor.name}/{descriptor.name}.duckdb_extension"
            if artifact_names != [expected_artifact_name]:
                _fail(
                    "REPOSITORY_PUBLISH_FAILED", "extension wheel must contain exactly one matching extension artifact"
                )
            artifact_bytes = wheel.read(expected_artifact_name)
    except DynamicExtensionError:
        raise
    except (OSError, zipfile.BadZipFile) as exception:
        raise DynamicExtensionError(
            "REPOSITORY_PUBLISH_FAILED", f"could not read extension wheel: {wheel_path}"
        ) from exception
    if hashlib.sha256(artifact_bytes).hexdigest() != descriptor.sha256:
        _fail("REPOSITORY_PUBLISH_FAILED", f"extension wheel artifact digest does not match {descriptor.identity}")
    return _WheelArtifact(descriptor=descriptor, descriptor_bytes=descriptor_bytes, artifact_bytes=artifact_bytes)


def _validate_published_files(directory: Path, entry: RepositoryArtifact) -> None:
    try:
        descriptor_bytes = (directory / entry.descriptor_path).read_bytes()
        artifact_bytes = (directory / entry.artifact_path).read_bytes()
    except OSError as exception:
        raise DynamicExtensionError(
            "REPOSITORY_PUBLISH_FAILED", f"existing repository does not retain {entry.descriptor.identity}"
        ) from exception
    try:
        descriptor = DynamicExtensionDescriptor.from_json(descriptor_bytes)
    except DynamicExtensionError as exception:
        raise DynamicExtensionError(
            "REPOSITORY_PUBLISH_FAILED", f"existing descriptor is invalid for {entry.descriptor.identity}"
        ) from exception
    if descriptor != entry.descriptor or hashlib.sha256(artifact_bytes).hexdigest() != entry.descriptor.sha256:
        _fail("REPOSITORY_PUBLISH_FAILED", f"existing repository bytes do not match {entry.descriptor.identity}")


def _write_immutable_file(path: Path, data: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            output.write(data)
    except FileExistsError:
        try:
            if not path.is_file() or path.is_symlink() or path.read_bytes() != data:
                _fail("REPOSITORY_PUBLISH_FAILED", f"refusing to replace immutable repository file: {path}")
        except OSError as exception:
            raise DynamicExtensionError(
                "REPOSITORY_PUBLISH_FAILED", f"could not validate existing repository file: {path}"
            ) from exception
    except OSError as exception:
        raise DynamicExtensionError(
            "REPOSITORY_PUBLISH_FAILED", f"could not write repository file: {path}"
        ) from exception


def _atomic_write(path: Path, data: bytes) -> None:
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}-", suffix=".tmp", delete=False
        ) as output:
            temporary_path = Path(output.name)
            output.write(data)
        os.replace(temporary_path, path)
        temporary_path = None
    except OSError as exception:
        raise DynamicExtensionError(
            "REPOSITORY_PUBLISH_FAILED", f"could not write repository index file: {path}"
        ) from exception
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_content_length(response: Any, maximum_size: int, label: str) -> None:
    content_length = response.headers.get("Content-Length")
    if content_length is None:
        return
    try:
        length = int(content_length)
    except ValueError as exception:
        raise DynamicExtensionError(
            "REPOSITORY_DOWNLOAD_FAILED", f"{label} sent an invalid Content-Length"
        ) from exception
    if length < 0 or length > maximum_size:
        _fail("REPOSITORY_DOWNLOAD_FAILED", f"{label} exceeds the maximum allowed size")


def _validate_public_key(value: object, *, code: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != _ED25519_PUBLIC_KEY_BYTES:
        _fail(code, "trusted_public_key must contain exactly 32 raw Ed25519 public-key bytes")
    return value


def _ed25519_private_key(value: object) -> Any:
    if not isinstance(value, bytes) or len(value) != _ED25519_PRIVATE_KEY_BYTES:
        _fail("REPOSITORY_KEY_INVALID", "private_key must contain exactly 32 raw Ed25519 private-key bytes")
    _, ed25519 = _cryptography_serialization()
    try:
        return ed25519.Ed25519PrivateKey.from_private_bytes(value)
    except ValueError as exception:
        raise DynamicExtensionError(
            "REPOSITORY_KEY_INVALID", "private_key is not a valid Ed25519 private key"
        ) from exception


def _ed25519_public_key_bytes(value: Any) -> bytes:
    serialization, _ = _cryptography_serialization()
    return value.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)


def _verify_ed25519(public_key: bytes, signature: bytes, message: bytes) -> None:
    _, ed25519 = _cryptography_serialization()
    try:
        ed25519.Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
    except ValueError as exception:
        raise DynamicExtensionError(
            "REPOSITORY_KEY_INVALID", "trusted_public_key is not a valid Ed25519 public key"
        ) from exception
    except Exception as exception:
        # cryptography exposes InvalidSignature from a deliberately small
        # exception hierarchy; avoid importing it until the optional extra is
        # requested and turn every verification failure into one deterministic
        # public error.
        raise DynamicExtensionError(
            "REPOSITORY_SIGNATURE_INVALID", "repository index signature did not verify"
        ) from exception


def _cryptography_serialization() -> tuple[Any, Any]:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as exception:
        raise DynamicExtensionError(
            "CRYPTOGRAPHY_UNAVAILABLE",
            "signed extension repositories require the optional vane-ai[extensions] dependency",
        ) from exception
    return serialization, ed25519


def _read_key_file(path: str | Path, kind: str) -> bytes:
    key_path = Path(path).expanduser().resolve()
    try:
        return key_path.read_bytes()
    except OSError as exception:
        raise DynamicExtensionError(
            "REPOSITORY_KEY_INVALID", f"could not read {kind} key file: {key_path}"
        ) from exception


def _decode_base64_key(value: bytes, kind: str) -> bytes:
    decoded = _decode_base64(value, f"{kind} key", "REPOSITORY_KEY_INVALID")
    if len(decoded) != _ED25519_PUBLIC_KEY_BYTES:
        _fail("REPOSITORY_KEY_INVALID", f"{kind} key must decode to exactly 32 Ed25519 key bytes")
    return decoded


def _decode_base64(value: object, label: str, code: str) -> bytes:
    if not isinstance(value, (str, bytes)):
        _fail(code, f"{label} must be base64 text")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exception:
        raise DynamicExtensionError(code, f"{label} is not valid base64") from exception


__all__ = [
    "InstalledExtension",
    "PublishedExtensionRepository",
    "RepositoryArtifact",
    "SignedExtensionRepository",
    "publish_extension_repository",
    "read_ed25519_private_key",
    "read_ed25519_public_key",
]
