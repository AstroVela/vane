# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0
"""Discovery and trusted local resolution for loadable DuckDB extensions.

The remote catalog supplies discovery metadata only. Resolution never obtains
an artifact from that catalog: a caller supplies an explicit artifact or an
installed local provider, and a descriptor pins the exact bytes that may be
loaded into a connection.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import shutil
import stat
import tempfile
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from http.client import HTTPException, HTTPMessage
from importlib.metadata import entry_points
from itertools import islice
from pathlib import Path
from typing import IO, TYPE_CHECKING, NoReturn
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

if TYPE_CHECKING:
    from vane import DuckDBPyConnection, DuckDBPyRelation

_DESCRIPTOR_FORMAT_VERSION = 1
_VALID_ABI_TYPES = frozenset({"CPP", "C_STRUCT", "C_STRUCT_UNSTABLE"})
_EXTENSION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CATALOG_EXTENSION_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_HEX_RE = re.compile(r"^[0-9a-f]{7,64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PLATFORM_RE = re.compile(r"^[a-z0-9_]+$")
_TRUST_IDENTITY_RE = re.compile(r"^[A-Za-z0-9._:/-]+$")
_CAPI_VERSION_RE = re.compile(r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_WINDOWS_PERMISSION_MODEL = os.name == "nt"
_WRITE_PERMISSION_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
_SHARED_DIRECTORY_WRITE_BITS = stat.S_IWGRP | stat.S_IWOTH
_CACHE_DIRECTORY_NORMALIZATION_TIMEOUT_SECONDS = 5.0
_CACHE_DIRECTORY_NORMALIZATION_RETRY_SECONDS = 0.01
_DYNAMIC_EXTENSION_PROVIDER_ENTRY_POINT_GROUP = "vane.dynamic_extension_providers"
_EXTENSION_CATALOG_FORMAT_VERSION = 1
_EXTENSION_CATALOG_MAX_BYTES = 64 * 1024
_EXTENSION_CATALOG_MAX_ENTRIES = 1024
_EXTENSION_CATALOG_TIMEOUT_SECONDS = 10.0
_EXTENSION_CATALOG_READ_CHUNK_BYTES = 16 * 1024
_EXTENSION_CATALOG_MAX_CONCURRENT_FETCHES = 4
_DISTRIBUTION_NAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
DEFAULT_EXTENSION_CATALOG_URL = "https://astrovela.github.io/vane-extensions/v1/index.json"


class _RejectCatalogRedirects(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> None:
        return None


_EXTENSION_CATALOG_OPENER = build_opener(_RejectCatalogRedirects())
_EXTENSION_CATALOG_FETCH_SLOTS = threading.BoundedSemaphore(_EXTENSION_CATALOG_MAX_CONCURRENT_FETCHES)


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


def _make_snapshot_read_only(path: Path, *, description: str) -> None:
    try:
        path.chmod(0o400)
    except OSError as exception:
        raise DynamicExtensionError(
            "ARTIFACT_SNAPSHOT_FAILED",
            f"could not make {description} read-only: {path}",
        ) from exception
    try:
        mode = path.stat().st_mode
    except OSError as exception:
        raise DynamicExtensionError(
            "ARTIFACT_SNAPSHOT_FAILED",
            f"could not inspect {description} permissions: {path}",
        ) from exception
    if not _snapshot_mode_is_read_only(mode):
        _fail("ARTIFACT_SNAPSHOT_FAILED", f"{description} is not read-only: {path}")


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("DESCRIPTOR_INVALID", f"{field_name} must be a non-empty string")
    if any(character.isspace() or ord(character) < 32 or 0xD800 <= ord(character) <= 0xDFFF for character in value):
        _fail(
            "DESCRIPTOR_INVALID",
            f"{field_name} must not contain whitespace, control characters, or lone Unicode surrogates",
        )
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


def _catalog_string(value: object, field_name: str, *, max_length: int) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail("CATALOG_INVALID", f"{field_name} must be a non-empty trimmed string")
    if len(value) > max_length:
        _fail("CATALOG_INVALID", f"{field_name} exceeds its {max_length}-character limit")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        _fail("CATALOG_INVALID", f"{field_name} must not contain control characters")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        _fail("CATALOG_INVALID", f"{field_name} must not contain lone Unicode surrogates")
    return value


def _canonical_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


@dataclass(frozen=True)
class ExtensionCatalogEntry:
    """One discoverable Vane extension provider package."""

    extension_name: str
    distribution_name: str
    description: str
    repository: str
    publisher: str
    license: str


@dataclass(frozen=True)
class ExtensionStatus:
    """Installed-provider and connection state for one extension name."""

    extension_name: str
    cataloged: bool
    installed: bool
    loadable: bool
    loaded: bool
    description: str | None
    distribution_name: str | None
    installed_distribution_name: str | None
    distribution_version: str | None
    repository: str | None
    publisher: str | None
    license: str | None
    provider_distributions: tuple[str, ...]
    provider_count: int
    extension_version: str | None
    trust_identity: str | None
    artifact_sha256: str | None


def _parse_extension_catalog(value: object) -> tuple[ExtensionCatalogEntry, ...]:
    if not isinstance(value, Mapping) or set(value) != {"format_version", "extensions"}:
        _fail("CATALOG_INVALID", "catalog must contain only format_version and extensions")
    if type(value["format_version"]) is not int or value["format_version"] != _EXTENSION_CATALOG_FORMAT_VERSION:
        _fail("CATALOG_INVALID", f"catalog format_version must be {_EXTENSION_CATALOG_FORMAT_VERSION}")
    raw_extensions = value["extensions"]
    if not isinstance(raw_extensions, list):
        _fail("CATALOG_INVALID", "catalog extensions must be a list")
    if len(raw_extensions) > _EXTENSION_CATALOG_MAX_ENTRIES:
        _fail("CATALOG_INVALID", f"catalog exceeds its {_EXTENSION_CATALOG_MAX_ENTRIES}-entry limit")

    expected_entry_keys = {
        "extension_name",
        "distribution_name",
        "description",
        "repository",
        "publisher",
        "license",
    }
    entries: list[ExtensionCatalogEntry] = []
    seen_names: set[str] = set()
    for index, raw_entry in enumerate(raw_extensions):
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != expected_entry_keys:
            _fail("CATALOG_INVALID", f"extensions[{index}] does not contain the exact catalog entry fields")
        extension_name = _catalog_string(
            raw_entry["extension_name"],
            f"extensions[{index}].extension_name",
            max_length=128,
        )
        if not _CATALOG_EXTENSION_NAME_RE.fullmatch(extension_name):
            _fail(
                "CATALOG_INVALID",
                f"extensions[{index}].extension_name must use lowercase ASCII extension-name syntax",
            )
        try:
            canonical_name = _native_canonical_extension_name(extension_name)
        except DynamicExtensionError as exception:
            raise DynamicExtensionError(
                "CATALOG_INVALID",
                f"could not canonicalize extensions[{index}].extension_name",
            ) from exception
        if canonical_name != extension_name:
            _fail(
                "CATALOG_INVALID",
                f"extensions[{index}].extension_name must use canonical DuckDB name {canonical_name!r}",
            )
        if extension_name in seen_names:
            _fail("CATALOG_INVALID", f"catalog repeats extension name {extension_name!r}")

        distribution_name = _catalog_string(
            raw_entry["distribution_name"],
            f"extensions[{index}].distribution_name",
            max_length=256,
        )
        if not _DISTRIBUTION_NAME_RE.fullmatch(distribution_name):
            _fail("CATALOG_INVALID", f"extensions[{index}].distribution_name is not a Python distribution name")
        expected_distribution_name = _canonical_distribution_name(f"vane-extension-{extension_name}")
        if distribution_name != expected_distribution_name:
            _fail(
                "CATALOG_INVALID",
                f"extensions[{index}].distribution_name must be canonical name {expected_distribution_name!r}",
            )

        repository = _catalog_string(
            raw_entry["repository"],
            f"extensions[{index}].repository",
            max_length=2048,
        )
        try:
            repository_url = urlsplit(repository)
            repository_hostname = repository_url.hostname
            repository_port = repository_url.port
        except ValueError as exception:
            raise DynamicExtensionError(
                "CATALOG_INVALID",
                f"extensions[{index}].repository is not a valid URL",
            ) from exception
        if (
            repository_url.scheme != "https"
            or not repository_hostname
            or repository_port is not None
            or repository_url.username is not None
            or repository_url.password is not None
            or not repository_url.path.strip("/")
            or repository_url.query
            or repository_url.fragment
            or not repository.isascii()
            or any(character.isspace() for character in repository)
        ):
            _fail("CATALOG_INVALID", f"extensions[{index}].repository must be a canonical HTTPS repository URL")
        entries.append(
            ExtensionCatalogEntry(
                extension_name=extension_name,
                distribution_name=distribution_name,
                description=_catalog_string(
                    raw_entry["description"],
                    f"extensions[{index}].description",
                    max_length=500,
                ),
                repository=repository,
                publisher=_catalog_string(
                    raw_entry["publisher"],
                    f"extensions[{index}].publisher",
                    max_length=100,
                ),
                license=_catalog_string(
                    raw_entry["license"],
                    f"extensions[{index}].license",
                    max_length=200,
                ),
            )
        )
        seen_names.add(extension_name)

    names = [entry.extension_name for entry in entries]
    if names != sorted(names):
        _fail("CATALOG_INVALID", "catalog extensions must be sorted by extension_name")
    return tuple(entries)


def _validate_extension_catalog_url(value: object) -> str:
    catalog_url = _catalog_string(value, "catalog_url", max_length=2048)
    try:
        parsed = urlsplit(catalog_url)
        port = parsed.port
    except ValueError as exception:
        raise DynamicExtensionError("CATALOG_INVALID", "catalog_url is not a valid URL") from exception
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or not parsed.path.strip("/")
        or parsed.query
        or parsed.fragment
        or not catalog_url.isascii()
        or any(character.isspace() for character in catalog_url)
    ):
        _fail("CATALOG_INVALID", "catalog_url must be a canonical HTTPS URL")
    return catalog_url


def _reject_duplicate_catalog_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("CATALOG_INVALID", f"catalog JSON object repeats key {key!r}")
        result[key] = value
    return result


def _extension_catalog_deadline_remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        _fail("CATALOG_UNAVAILABLE", "extension catalog request exceeded its wall-clock deadline")
    return remaining


def _read_extension_catalog_response(
    response: object,
    *,
    deadline: float,
    cancelled: threading.Event,
) -> bytes:
    read_one = getattr(response, "read1", None)
    if not callable(read_one):
        _fail("CATALOG_UNAVAILABLE", "extension catalog response does not support bounded reads")

    contents = bytearray()
    maximum_read = _EXTENSION_CATALOG_MAX_BYTES + 1
    while len(contents) < maximum_read:
        if cancelled.is_set():
            _fail("CATALOG_UNAVAILABLE", "extension catalog request was cancelled")
        _extension_catalog_deadline_remaining(deadline)
        read_size = min(_EXTENSION_CATALOG_READ_CHUNK_BYTES, maximum_read - len(contents))
        chunk = read_one(read_size)
        _extension_catalog_deadline_remaining(deadline)
        if not isinstance(chunk, bytes) or len(chunk) > read_size:
            _fail("CATALOG_UNAVAILABLE", "extension catalog endpoint returned an invalid response body")
        if not chunk:
            break
        contents.extend(chunk)
    return bytes(contents)


def _extension_catalog_fetch_worker(
    request: Request,
    validated_url: str,
    deadline: float,
    cancelled: threading.Event,
    results: queue.Queue[tuple[bytes | None, Exception | None]],
) -> None:
    try:
        with _EXTENSION_CATALOG_OPENER.open(
            request,
            timeout=_extension_catalog_deadline_remaining(deadline),
        ) as response:
            get_url = getattr(response, "geturl", None)
            if getattr(response, "status", None) != 200 or not callable(get_url) or get_url() != validated_url:
                _fail("CATALOG_UNAVAILABLE", "extension catalog endpoint did not return the requested resource")
            contents = _read_extension_catalog_response(
                response,
                deadline=deadline,
                cancelled=cancelled,
            )
        results.put((contents, None))
    except Exception as exception:
        results.put((None, exception))
    finally:
        _EXTENSION_CATALOG_FETCH_SLOTS.release()


def _fetch_extension_catalog_contents(request: Request, validated_url: str) -> bytes:
    deadline = time.monotonic() + _EXTENSION_CATALOG_TIMEOUT_SECONDS
    cancelled = threading.Event()
    results: queue.Queue[tuple[bytes | None, Exception | None]] = queue.Queue(maxsize=1)
    worker = threading.Thread(
        target=_extension_catalog_fetch_worker,
        args=(request, validated_url, deadline, cancelled, results),
        name="vane-extension-catalog-fetch",
        daemon=True,
    )
    if not _EXTENSION_CATALOG_FETCH_SLOTS.acquire(timeout=_extension_catalog_deadline_remaining(deadline)):
        _fail("CATALOG_UNAVAILABLE", "extension catalog request exceeded its wall-clock deadline")
    try:
        worker.start()
    except RuntimeError as exception:
        _EXTENSION_CATALOG_FETCH_SLOTS.release()
        raise DynamicExtensionError(
            "CATALOG_UNAVAILABLE", "could not start the extension catalog request"
        ) from exception
    except BaseException:
        _EXTENSION_CATALOG_FETCH_SLOTS.release()
        raise

    worker.join(max(0.0, deadline - time.monotonic()))
    if worker.is_alive() or time.monotonic() >= deadline:
        cancelled.set()
        _fail("CATALOG_UNAVAILABLE", "extension catalog request exceeded its wall-clock deadline")
    try:
        contents, exception = results.get_nowait()
    except queue.Empty as queue_exception:
        raise DynamicExtensionError(
            "CATALOG_UNAVAILABLE", "extension catalog request did not return a result"
        ) from queue_exception
    if exception is not None:
        raise exception
    if contents is None:
        _fail("CATALOG_UNAVAILABLE", "extension catalog request did not return a response body")
    return contents


def extension_catalog(
    *,
    catalog_url: str = DEFAULT_EXTENSION_CATALOG_URL,
) -> tuple[ExtensionCatalogEntry, ...]:
    """Fetch and validate one Vane provider discovery catalog.

    The request is bounded, rejects redirects, and has no cache or fallback.
    Catalog entries are discovery metadata only and never initialize providers.
    """
    validated_url = _validate_extension_catalog_url(catalog_url)
    request = Request(
        validated_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "vane-extension-catalog/1",
        },
    )
    try:
        contents = _fetch_extension_catalog_contents(request, validated_url)
    except DynamicExtensionError:
        raise
    except (HTTPError, URLError, HTTPException, OSError, TimeoutError) as exception:
        raise DynamicExtensionError("CATALOG_UNAVAILABLE", "could not fetch the extension catalog") from exception
    if len(contents) > _EXTENSION_CATALOG_MAX_BYTES:
        _fail("CATALOG_INVALID", "extension catalog exceeds its size limit")
    try:
        document = json.loads(contents.decode("utf-8"), object_pairs_hook=_reject_duplicate_catalog_keys)
    except DynamicExtensionError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exception:
        raise DynamicExtensionError("CATALOG_INVALID", "extension catalog is not valid UTF-8 JSON") from exception
    return _parse_extension_catalog(document)


def _validate_extension_catalog_entries(
    entries: Iterable[ExtensionCatalogEntry],
) -> tuple[ExtensionCatalogEntry, ...]:
    try:
        supplied_entries = tuple(islice(iter(entries), _EXTENSION_CATALOG_MAX_ENTRIES + 1))
    except TypeError as exception:
        raise DynamicExtensionError("CATALOG_INVALID", "catalog entries must be iterable") from exception
    if len(supplied_entries) > _EXTENSION_CATALOG_MAX_ENTRIES:
        _fail("CATALOG_INVALID", f"catalog exceeds its {_EXTENSION_CATALOG_MAX_ENTRIES}-entry limit")
    raw_entries: list[dict[str, object]] = []
    for index, entry in enumerate(supplied_entries):
        if not isinstance(entry, ExtensionCatalogEntry):
            _fail("CATALOG_INVALID", f"catalog entry {index} must be ExtensionCatalogEntry")
        raw_entries.append(
            {
                "extension_name": entry.extension_name,
                "distribution_name": entry.distribution_name,
                "description": entry.description,
                "repository": entry.repository,
                "publisher": entry.publisher,
                "license": entry.license,
            }
        )
    return _parse_extension_catalog(
        {
            "format_version": _EXTENSION_CATALOG_FORMAT_VERSION,
            "extensions": raw_entries,
        }
    )


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


def _parse_dynamic_extension_snapshot(snapshot: object) -> tuple[DynamicExtensionDescriptor, ...]:
    """Validate an ordered dynamic-extension manifest without loading it."""
    if not isinstance(snapshot, list):
        _fail("SNAPSHOT_INVALID", "dynamic_extensions must be a list")

    descriptors: list[DynamicExtensionDescriptor] = []
    available_identities: set[str] = set()
    seen_names: set[str] = set()
    for index, raw_descriptor in enumerate(snapshot):
        if not isinstance(raw_descriptor, Mapping):
            _fail("SNAPSHOT_INVALID", f"dynamic_extensions[{index}] must be a descriptor object")
        descriptor = DynamicExtensionDescriptor.from_dict(raw_descriptor)
        if descriptor.identity in available_identities:
            _fail("SNAPSHOT_INVALID", f"dynamic_extensions contains duplicate {descriptor.identity}")
        if descriptor.name in seen_names:
            _fail("SNAPSHOT_INVALID", f"dynamic_extensions contains conflicting name {descriptor.name}")
        missing_dependencies = [
            dependency.identity
            for dependency in descriptor.dependencies
            if dependency.identity not in available_identities
        ]
        if missing_dependencies:
            _fail(
                "SNAPSHOT_DEPENDENCY_ORDER",
                f"dynamic_extensions[{index}] declares dependencies before their descriptors: "
                f"{', '.join(missing_dependencies)}",
            )
        descriptors.append(descriptor)
        available_identities.add(descriptor.identity)
        seen_names.add(descriptor.name)
    return tuple(descriptors)


def _normalize_dynamic_extension_snapshot(snapshot: object) -> list[dict[str, object]]:
    """Return the canonical manifest after strict structural validation."""
    return [descriptor.to_dict() for descriptor in _parse_dynamic_extension_snapshot(snapshot)]


def _dynamic_extension_snapshot_cache_identity(snapshot: object) -> tuple[tuple[str, str], ...]:
    """Return the exact artifact identity used by a worker DatabaseInstance."""
    return tuple((descriptor.name, descriptor.to_json()) for descriptor in _parse_dynamic_extension_snapshot(snapshot))


def _serialized_dynamic_extension_snapshot_entries(connection: DuckDBPyConnection) -> list[str]:
    export_entries = getattr(connection, "_export_dynamic_extension_snapshot_entries", None)
    if not callable(export_entries):
        _fail("SNAPSHOT_UNAVAILABLE", "connection cannot export dynamic extension snapshot entries")
    try:
        serialized_entries = export_entries()
    except Exception as exception:
        raise DynamicExtensionError(
            "SNAPSHOT_UNAVAILABLE", "could not read the connection's dynamic extension snapshot"
        ) from exception
    if not isinstance(serialized_entries, list) or any(not isinstance(entry, str) for entry in serialized_entries):
        _fail("SNAPSHOT_INVALID", "connection dynamic extension entries must be a list of JSON strings")
    return serialized_entries


def _capture_dynamic_extension_snapshot(connection: DuckDBPyConnection) -> list[dict[str, object]]:
    """Capture resolver-owned descriptors without serializing local paths."""
    return _deserialize_dynamic_extension_snapshot_entries(_serialized_dynamic_extension_snapshot_entries(connection))


def _deserialize_dynamic_extension_snapshot_entries(
    serialized_entries: list[str],
) -> list[dict[str, object]]:
    """Parse one native session snapshot captured under a single lock."""
    entries: list[object] = []
    for index, serialized_entry in enumerate(serialized_entries):
        try:
            entries.append(json.loads(serialized_entry))
        except ValueError as exception:
            raise DynamicExtensionError(
                "SNAPSHOT_INVALID",
                f"connection dynamic extension entry {index} is not valid JSON",
            ) from exception
    return _normalize_dynamic_extension_snapshot(entries)


def _record_dynamic_extension_snapshot_entry(
    connection: DuckDBPyConnection,
    candidate: ResolvedDynamicExtension,
) -> None:
    """Record one verified, successfully loaded artifact for snapshot capture."""
    compare_and_record = getattr(connection, "_compare_and_record_dynamic_extension_snapshot_entry", None)
    if not callable(compare_and_record):
        _fail("SNAPSHOT_UNAVAILABLE", "connection cannot record dynamic extension snapshot entries")
    candidate_json = candidate.descriptor.to_json()

    while True:
        serialized_entries = _serialized_dynamic_extension_snapshot_entries(connection)
        existing_descriptors = _parse_dynamic_extension_snapshot(
            _deserialize_dynamic_extension_snapshot_entries(serialized_entries)
        )
        for existing in existing_descriptors:
            if existing.identity == candidate.identity:
                if existing != candidate.descriptor:
                    _fail(
                        "LOADED_IDENTITY_CONFLICT",
                        f"{candidate.identity} has conflicting immutable descriptors on this connection session",
                    )
                return
            if existing.name == candidate.descriptor.name:
                _fail(
                    "LOADED_NAME_CONFLICT",
                    f"{candidate.descriptor.name} is already recorded as {existing.identity} on this connection session",
                )
        available_identities = {descriptor.identity for descriptor in existing_descriptors}
        missing_dependencies = [
            dependency.identity
            for dependency in candidate.descriptor.dependencies
            if dependency.identity not in available_identities
        ]
        if missing_dependencies:
            _fail(
                "SNAPSHOT_DEPENDENCY_ORDER",
                f"cannot record {candidate.identity} before dependencies {', '.join(missing_dependencies)}",
            )
        try:
            if compare_and_record(serialized_entries, candidate_json):
                return
        except Exception as exception:
            raise DynamicExtensionError(
                "SNAPSHOT_UNAVAILABLE", "could not record the connection's dynamic extension snapshot"
            ) from exception


def _load_installed_dynamic_extension_providers(
    descriptors: Iterable[DynamicExtensionDescriptor],
) -> tuple[LocalExtensionProvider, ...]:
    """Load exactly the preinstalled provider entry points named by a manifest."""
    descriptor_by_name = {descriptor.name: descriptor for descriptor in descriptors}
    if not descriptor_by_name:
        return ()
    installed_entry_points = _installed_dynamic_extension_provider_entry_points()

    providers: list[LocalExtensionProvider] = []
    for name, descriptor in sorted(descriptor_by_name.items()):
        provider = _load_installed_dynamic_extension_provider(name, installed_entry_points)
        artifact = provider.find(descriptor.identity)
        if artifact is None or artifact.descriptor != descriptor:
            _fail(
                "PROVIDER_DESCRIPTOR_MISMATCH",
                f"installed local provider descriptor does not exactly match {descriptor.identity}",
            )
        # An entry point authorizes only the artifact named by that entry
        # point. A provider may bundle other artifacts, but those extras must
        # neither become fallback candidates nor make another exact entry
        # point ambiguous.
        providers.append(
            LocalExtensionProvider(
                provider.trust_identity,
                (artifact,),
            )
        )
    return tuple(providers)


def _installed_dynamic_extension_provider_entry_points() -> tuple[object, ...]:
    """Enumerate installed provider metadata without initializing providers."""
    try:
        return tuple(entry_points(group=_DYNAMIC_EXTENSION_PROVIDER_ENTRY_POINT_GROUP))
    except Exception as exception:
        raise DynamicExtensionError(
            "PROVIDER_DISCOVERY_FAILED", "could not enumerate installed dynamic extension providers"
        ) from exception


@dataclass(frozen=True)
class _InstalledExtensionProviderMetadata:
    extension_name: str
    distribution_name: str
    distribution_version: str


def _provider_metadata_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail("PROVIDER_INVALID", f"installed provider {field_name} must be a non-empty trimmed string")
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127 or 0xD800 <= ord(character) <= 0xDFFF
        for character in value
    ):
        _fail(
            "PROVIDER_INVALID",
            f"installed provider {field_name} must not contain whitespace, control characters, or lone Unicode surrogates",
        )
    return value


def _installed_dynamic_extension_provider_metadata(
    installed_entry_points: tuple[object, ...],
) -> tuple[_InstalledExtensionProviderMetadata, ...]:
    """Inspect provider package metadata without importing provider code."""
    providers: list[_InstalledExtensionProviderMetadata] = []
    for index, installed in enumerate(installed_entry_points):
        try:
            raw_name = getattr(installed, "name", None)
            name = _provider_metadata_string(raw_name, f"entry point {index} name")
            if not _EXTENSION_NAME_RE.fullmatch(name) or _native_canonical_extension_name(name) != name:
                _fail("PROVIDER_INVALID", f"installed provider entry point {index} has non-canonical name {name!r}")

            distribution = getattr(installed, "dist", None)
            metadata = getattr(distribution, "metadata", None)
            metadata_get = getattr(metadata, "get", None)
            if not callable(metadata_get):
                _fail("PROVIDER_INVALID", f"installed provider entry point {name!r} has no distribution metadata")
            distribution_name = _provider_metadata_string(
                metadata_get("Name"),
                f"entry point {name!r} distribution name",
            )
            if not _DISTRIBUTION_NAME_RE.fullmatch(distribution_name):
                _fail(
                    "PROVIDER_INVALID",
                    f"installed provider entry point {name!r} has an invalid distribution name",
                )
            distribution_version = _provider_metadata_string(
                getattr(distribution, "version", None),
                f"entry point {name!r} distribution version",
            )
        except DynamicExtensionError:
            raise
        except Exception as exception:
            raise DynamicExtensionError(
                "PROVIDER_INVALID",
                f"could not inspect installed provider entry point {index}",
            ) from exception
        providers.append(
            _InstalledExtensionProviderMetadata(
                extension_name=name,
                distribution_name=_canonical_distribution_name(distribution_name),
                distribution_version=distribution_version,
            )
        )
    return tuple(
        sorted(
            providers,
            key=lambda provider: (
                provider.extension_name,
                provider.distribution_name,
                provider.distribution_version,
            ),
        )
    )


def extension_statuses(
    *,
    connection: DuckDBPyConnection,
    catalog: Iterable[ExtensionCatalogEntry] | None = None,
) -> tuple[ExtensionStatus, ...]:
    """Return catalog, installation, and verified-load state for a connection.

    Enumerating installed providers reads only Python distribution metadata. It
    never imports or initializes an extension provider; provider code runs only
    after an explicit :func:`load_installed_extension` call.
    """
    catalog_entries = extension_catalog() if catalog is None else _validate_extension_catalog_entries(catalog)
    catalog_by_name = {entry.extension_name: entry for entry in catalog_entries}
    provider_metadata = _installed_dynamic_extension_provider_metadata(
        _installed_dynamic_extension_provider_entry_points()
    )
    providers_by_name: dict[str, list[_InstalledExtensionProviderMetadata]] = {}
    for provider in provider_metadata:
        providers_by_name.setdefault(provider.extension_name, []).append(provider)

    loaded_descriptors = _parse_dynamic_extension_snapshot(_capture_dynamic_extension_snapshot(connection))
    loaded_by_name = {descriptor.name: descriptor for descriptor in loaded_descriptors}
    extension_names = sorted(set(catalog_by_name) | set(providers_by_name) | set(loaded_by_name))
    statuses: list[ExtensionStatus] = []
    for extension_name in extension_names:
        catalog_entry = catalog_by_name.get(extension_name)
        installed_providers = providers_by_name.get(extension_name, [])
        installed_provider = installed_providers[0] if len(installed_providers) == 1 else None
        loaded_descriptor = loaded_by_name.get(extension_name)
        statuses.append(
            ExtensionStatus(
                extension_name=extension_name,
                cataloged=catalog_entry is not None,
                installed=bool(installed_providers),
                loadable=len(installed_providers) == 1,
                loaded=loaded_descriptor is not None,
                description=None if catalog_entry is None else catalog_entry.description,
                distribution_name=None if catalog_entry is None else catalog_entry.distribution_name,
                installed_distribution_name=(
                    None if installed_provider is None else installed_provider.distribution_name
                ),
                distribution_version=None if installed_provider is None else installed_provider.distribution_version,
                repository=None if catalog_entry is None else catalog_entry.repository,
                publisher=None if catalog_entry is None else catalog_entry.publisher,
                license=None if catalog_entry is None else catalog_entry.license,
                provider_distributions=tuple(
                    f"{provider.distribution_name}=={provider.distribution_version}" for provider in installed_providers
                ),
                provider_count=len(installed_providers),
                extension_version=None if loaded_descriptor is None else loaded_descriptor.extension_version,
                trust_identity=None if loaded_descriptor is None else loaded_descriptor.trust_identity,
                artifact_sha256=None if loaded_descriptor is None else loaded_descriptor.sha256,
            )
        )
    return tuple(statuses)


def vane_extensions(
    *,
    connection: DuckDBPyConnection,
    catalog: Iterable[ExtensionCatalogEntry] | None = None,
) -> DuckDBPyRelation:
    """Return a DuckDB relation describing Vane extension provider state."""
    import pyarrow as pa

    statuses = extension_statuses(connection=connection, catalog=catalog)
    schema = pa.schema(
        (
            pa.field("extension_name", pa.string(), nullable=False),
            pa.field("loaded", pa.bool_(), nullable=False),
            pa.field("installed", pa.bool_(), nullable=False),
            pa.field("loadable", pa.bool_(), nullable=False),
            pa.field("cataloged", pa.bool_(), nullable=False),
            pa.field("description", pa.string()),
            pa.field("extension_version", pa.string()),
            pa.field("distribution_name", pa.string()),
            pa.field("installed_distribution_name", pa.string()),
            pa.field("distribution_version", pa.string()),
            pa.field("repository", pa.string()),
            pa.field("publisher", pa.string()),
            pa.field("license", pa.string()),
            pa.field("provider_distributions", pa.list_(pa.string()), nullable=False),
            pa.field("provider_count", pa.uint64(), nullable=False),
            pa.field("trust_identity", pa.string()),
            pa.field("artifact_sha256", pa.string()),
        )
    )
    rows = [
        {
            "extension_name": status.extension_name,
            "loaded": status.loaded,
            "installed": status.installed,
            "loadable": status.loadable,
            "cataloged": status.cataloged,
            "description": status.description,
            "extension_version": status.extension_version,
            "distribution_name": status.distribution_name,
            "installed_distribution_name": status.installed_distribution_name,
            "distribution_version": status.distribution_version,
            "repository": status.repository,
            "publisher": status.publisher,
            "license": status.license,
            "provider_distributions": list(status.provider_distributions),
            "provider_count": status.provider_count,
            "trust_identity": status.trust_identity,
            "artifact_sha256": status.artifact_sha256,
        }
        for status in statuses
    ]
    return connection.from_arrow(pa.Table.from_pylist(rows, schema=schema)).set_alias("vane_extensions")


def _load_installed_dynamic_extension_provider(
    name: str,
    installed_entry_points: tuple[object, ...],
) -> LocalExtensionProvider:
    """Initialize the one installed provider authorized for a canonical name."""
    matches = [installed for installed in installed_entry_points if getattr(installed, "name", None) == name]
    if not matches:
        _fail("PROVIDER_NOT_FOUND", f"no installed local provider entry point exists for {name}")
    if len(matches) != 1:
        _fail("PROVIDER_AMBIGUOUS", f"multiple installed local provider entry points exist for {name}")
    try:
        provider_factory = matches[0].load()  # type: ignore[attr-defined]
        provider = provider_factory()
    except DynamicExtensionError:
        raise
    except Exception as exception:
        raise DynamicExtensionError(
            "PROVIDER_INVALID", f"could not initialize installed local provider for {name}"
        ) from exception
    if not isinstance(provider, LocalExtensionProvider):
        _fail("PROVIDER_INVALID", f"installed local provider for {name} did not return LocalExtensionProvider")
    return provider


def load_installed_extension(
    name: str,
    *,
    connection: DuckDBPyConnection,
) -> ResolvedDynamicExtension:
    """Load one named installed provider and its exact dependency closure.

    Provider discovery is limited to the canonical entry-point name and the
    dependency identities declared by its descriptor. The installed provider
    packages form the local trust boundary; this helper performs no repository,
    directory, download, compatibility, or fallback lookup.
    """
    canonical_name = _validate_extension_name(name, "provider name")
    installed_entry_points = _installed_dynamic_extension_provider_entry_points()
    provider_by_name: dict[str, LocalExtensionProvider] = {}

    def installed_provider(provider_name: str) -> LocalExtensionProvider:
        provider = provider_by_name.get(provider_name)
        if provider is None:
            provider = _load_installed_dynamic_extension_provider(provider_name, installed_entry_points)
            provider_by_name[provider_name] = provider
        return provider

    root_provider = installed_provider(canonical_name)
    root_artifacts = tuple(
        artifact
        for artifact in root_provider._artifact_by_identity.values()
        if artifact.descriptor.name == canonical_name
    )
    if not root_artifacts:
        _fail(
            "PROVIDER_DESCRIPTOR_MISMATCH",
            f"installed local provider entry point {canonical_name} does not provide that extension",
        )
    if len(root_artifacts) != 1:
        _fail(
            "PROVIDER_AMBIGUOUS",
            f"installed local provider entry point {canonical_name} provides multiple artifact identities",
        )

    root_descriptor = root_artifacts[0].descriptor
    descriptor_by_identity: dict[str, DynamicExtensionDescriptor] = {}
    descriptor_by_name: dict[str, DynamicExtensionDescriptor] = {}
    scoped_provider_by_identity: dict[str, LocalExtensionProvider] = {}
    pending = [root_descriptor]
    while pending:
        descriptor = pending.pop()
        existing_identity = descriptor_by_identity.get(descriptor.identity)
        if existing_identity is not None:
            if existing_identity != descriptor:
                _fail(
                    "RESOLVED_IDENTITY_CONFLICT",
                    f"{descriptor.identity} resolves to conflicting descriptors",
                )
            continue
        existing_name = descriptor_by_name.get(descriptor.name)
        if existing_name is not None and existing_name.identity != descriptor.identity:
            _fail(
                "RESOLVED_NAME_CONFLICT",
                f"dependency graph resolves {descriptor.name} as both "
                f"{existing_name.identity} and {descriptor.identity}",
            )

        provider = installed_provider(descriptor.name)
        artifact = provider.find(descriptor.identity)
        if artifact is None or artifact.descriptor != descriptor:
            _fail(
                "PROVIDER_DESCRIPTOR_MISMATCH",
                f"installed local provider descriptor does not exactly match {descriptor.identity}",
            )
        descriptor_by_identity[descriptor.identity] = descriptor
        descriptor_by_name[descriptor.name] = descriptor
        scoped_provider_by_identity[descriptor.identity] = LocalExtensionProvider(
            provider.trust_identity,
            (artifact,),
        )

        dependency_descriptors: list[DynamicExtensionDescriptor] = []
        for dependency in descriptor.dependencies:
            dependency_provider = installed_provider(dependency.name)
            dependency_artifact = dependency_provider.find(dependency.identity)
            if dependency_artifact is None:
                _fail(
                    "DEPENDENCY_NOT_FOUND",
                    f"installed local provider does not contain {dependency.identity}",
                )
            dependency_descriptors.append(dependency_artifact.descriptor)
        pending.extend(reversed(dependency_descriptors))

    resolver = DynamicExtensionResolver(
        trusted_identities={descriptor.trust_identity for descriptor in descriptor_by_identity.values()},
        providers=tuple(scoped_provider_by_identity.values()),
    )
    return resolver.load(connection, root_descriptor)


def _prepare_dynamic_extension_snapshot(connection: DuckDBPyConnection, snapshot: object) -> None:
    """Resolve, verify, and load a worker manifest from preinstalled providers."""
    expected_descriptors = _parse_dynamic_extension_snapshot(snapshot)
    existing_descriptors = _parse_dynamic_extension_snapshot(_capture_dynamic_extension_snapshot(connection))
    if existing_descriptors:
        if existing_descriptors != expected_descriptors:
            _fail("WORKER_DISAGREEMENT", "worker dynamic extension manifest differs from the coordinator manifest")
        return
    if not expected_descriptors:
        return

    providers = _load_installed_dynamic_extension_providers(expected_descriptors)
    resolver = DynamicExtensionResolver(
        trusted_identities={descriptor.trust_identity for descriptor in expected_descriptors},
        providers=providers,
    )

    # Verify the complete graph before changing the worker DatabaseInstance.
    for descriptor in expected_descriptors:
        resolver.resolve(connection, descriptor)
    for descriptor in expected_descriptors:
        resolver.load(connection, descriptor)

    prepared_descriptors = _parse_dynamic_extension_snapshot(_capture_dynamic_extension_snapshot(connection))
    if prepared_descriptors != expected_descriptors:
        _fail("WORKER_DISAGREEMENT", "worker dynamic extension identities differ after preparation")


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
        if isinstance(trusted_identities, str):
            _fail("DESCRIPTOR_INVALID", "trusted_identities must be an iterable of identities, not one string")
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
        current_platform = _native_platform()
        current_extension_compatibility_version = _native_extension_compatibility_version()
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
                    current_extension_compatibility_version=current_extension_compatibility_version,
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
            _record_dynamic_extension_snapshot_entry(connection, candidate)
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
        current_extension_compatibility_version: str,
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
            self._prepare_cache_directory(digest_directory)
            self._prepare_cache_directory(artifact_directory)
            source_digest = _sha256_file(artifact_path)
            self._validate_digest(descriptor, source_digest)
            self._validate_cached_snapshot(
                descriptor,
                snapshot_path,
                current_platform=current_platform,
                current_extension_compatibility_version=current_extension_compatibility_version,
            )
            return snapshot_path

        staging_directory = Path(tempfile.mkdtemp(prefix=f".{descriptor.name}-", dir=cache_root))
        staging_path = staging_directory / expected_filename
        try:
            self._prepare_created_private_directory(
                staging_directory,
                description="verified artifact staging",
            )
            actual_digest = _copy_and_hash_artifact(artifact_path, staging_path)
            self._validate_digest(descriptor, actual_digest)
            _make_snapshot_read_only(staging_path, description="verified artifact snapshot")
            metadata = _inspect_native_extension(staging_path)
            self._validate_native_metadata(
                descriptor,
                metadata,
                current_platform=current_platform,
                current_extension_compatibility_version=current_extension_compatibility_version,
            )
            self._prepare_cache_directory(digest_directory)
            try:
                os.rename(staging_directory, artifact_directory)
            except OSError as exception:
                if not (artifact_directory.exists() or artifact_directory.is_symlink()):
                    raise DynamicExtensionError(
                        "ARTIFACT_SNAPSHOT_FAILED",
                        f"could not publish verified artifact snapshot: {snapshot_path}",
                    ) from exception
                self._prepare_cache_directory(artifact_directory)
                self._validate_cached_snapshot(
                    descriptor,
                    snapshot_path,
                    current_platform=current_platform,
                    current_extension_compatibility_version=current_extension_compatibility_version,
                )
            else:
                self._prepare_cache_directory(artifact_directory)
                if _WINDOWS_PERMISSION_MODEL:
                    _secure_native_cache_path(snapshot_path, directory=False)
            return snapshot_path
        finally:
            _cleanup_staging_directory(staging_directory, staging_path)

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
        current_extension_compatibility_version: str,
    ) -> None:
        if snapshot_path.is_symlink() or not snapshot_path.is_file():
            _fail("ARTIFACT_CACHE_CORRUPT", f"verified artifact cache entry is invalid: {snapshot_path}")
        try:
            snapshot_metadata = snapshot_path.stat()
        except OSError as exception:
            raise DynamicExtensionError(
                "ARTIFACT_CACHE_CORRUPT", f"could not inspect verified artifact cache entry: {snapshot_path}"
            ) from exception
        if snapshot_metadata.st_nlink != 1:
            _fail("ARTIFACT_CACHE_CORRUPT", f"verified artifact cache entry has unsafe hard links: {snapshot_path}")
        if _WINDOWS_PERMISSION_MODEL:
            _secure_native_cache_path(snapshot_path, directory=False)
        elif snapshot_metadata.st_uid not in {0, os.geteuid()}:
            _fail("ARTIFACT_CACHE_CORRUPT", f"verified artifact cache entry has an untrusted owner: {snapshot_path}")
        if not _snapshot_mode_is_read_only(snapshot_metadata.st_mode):
            _fail("ARTIFACT_CACHE_CORRUPT", f"verified artifact cache entry is not read-only: {snapshot_path}")
        if _sha256_file(snapshot_path) != descriptor.sha256:
            _fail("ARTIFACT_CACHE_CORRUPT", f"verified artifact cache digest is invalid: {snapshot_path}")
        metadata = _inspect_native_extension(snapshot_path)
        self._validate_native_metadata(
            descriptor,
            metadata,
            current_platform=current_platform,
            current_extension_compatibility_version=current_extension_compatibility_version,
        )

    @staticmethod
    def _validate_native_metadata(
        descriptor: DynamicExtensionDescriptor,
        metadata: _NativeExtensionMetadata,
        *,
        current_platform: str,
        current_extension_compatibility_version: str,
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
        # DuckDB's CPP footer uses its native extension compatibility version,
        # which is distinct from the full engine SourceID pinned by the descriptor.
        elif metadata.duckdb_version != current_extension_compatibility_version:
            _fail(
                "SOURCE_ID_MISMATCH",
                f"{descriptor.identity} footer compatibility version is {metadata.duckdb_version}, "
                f"runtime expects {current_extension_compatibility_version}",
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
        root = self._prepare_cache_directory(root)
        if _WINDOWS_PERMISSION_MODEL:
            self._validate_windows_cache_ancestors(root)
        else:
            self._validate_posix_cache_ancestors(root)
        return root

    @staticmethod
    def _validate_windows_cache_ancestors(path: Path) -> None:
        for ancestor in (path, *path.parents):
            if _native_cache_path_is_replaceable(ancestor):
                _fail(
                    "ARTIFACT_CACHE_CORRUPT",
                    f"verified artifact cache has a replaceable ancestor: {ancestor}",
                )

    @staticmethod
    def _validate_posix_cache_ancestors(path: Path) -> None:
        trusted_owners = {0, os.geteuid()}
        try:
            child_metadata = path.lstat()
            if not stat.S_ISDIR(child_metadata.st_mode) or child_metadata.st_uid not in trusted_owners:
                _fail(
                    "ARTIFACT_CACHE_CORRUPT",
                    f"verified artifact cache directory is invalid: {path}",
                )
            for ancestor in path.parents:
                ancestor_metadata = ancestor.lstat()
                if not stat.S_ISDIR(ancestor_metadata.st_mode):
                    _fail(
                        "ARTIFACT_CACHE_CORRUPT",
                        f"verified artifact cache ancestor is invalid: {ancestor}",
                    )
                if ancestor_metadata.st_uid not in trusted_owners:
                    _fail(
                        "ARTIFACT_CACHE_CORRUPT",
                        f"verified artifact cache has a replaceable ancestor: {ancestor}",
                    )
                ancestor_mode = stat.S_IMODE(ancestor_metadata.st_mode)
                if ancestor_mode & _SHARED_DIRECTORY_WRITE_BITS and (
                    not (ancestor_mode & stat.S_ISVTX) or child_metadata.st_uid not in trusted_owners
                ):
                    _fail(
                        "ARTIFACT_CACHE_CORRUPT",
                        f"verified artifact cache has a replaceable ancestor: {ancestor}",
                    )
                child_metadata = ancestor_metadata
        except DynamicExtensionError:
            raise
        except OSError as exception:
            raise DynamicExtensionError(
                "ARTIFACT_CACHE_CORRUPT",
                f"could not inspect verified artifact cache ancestors: {path}",
            ) from exception

    @staticmethod
    def _wait_for_cache_directory_normalization(path: Path, *, exact_mode: bool) -> None:
        # mkdir applies the process umask before returning. A competing creator
        # may therefore expose a temporarily inaccessible owner-only directory
        # until its immediate chmod(0700) completes.
        deadline = time.monotonic() + _CACHE_DIRECTORY_NORMALIZATION_TIMEOUT_SECONDS
        trusted_owners = {0, os.geteuid()}
        while True:
            directory_metadata = path.lstat()
            directory_mode = stat.S_IMODE(directory_metadata.st_mode)
            if (
                not stat.S_ISDIR(directory_metadata.st_mode)
                or directory_metadata.st_uid not in trusted_owners
                or directory_mode & (stat.S_IRWXG | stat.S_IRWXO)
            ):
                return
            if exact_mode:
                normalized = directory_mode == 0o700
            else:
                normalized = directory_mode & 0o300 == 0o300
            if normalized:
                return
            if time.monotonic() >= deadline:
                _fail(
                    "ARTIFACT_SNAPSHOT_FAILED",
                    f"timed out waiting for verified artifact cache directory permissions: {path}",
                )
            time.sleep(_CACHE_DIRECTORY_NORMALIZATION_RETRY_SECONDS)

    @staticmethod
    def _create_missing_cache_directories(path: Path) -> None:
        missing_directories: list[Path] = []
        candidate = path
        try:
            while True:
                try:
                    candidate.lstat()
                    if not _WINDOWS_PERMISSION_MODEL:
                        DynamicExtensionResolver._wait_for_cache_directory_normalization(
                            candidate,
                            exact_mode=candidate == path,
                        )
                    break
                except FileNotFoundError:
                    missing_directories.append(candidate)
                    parent = candidate.parent
                    if parent == candidate:
                        raise
                    candidate = parent

            for directory in reversed(missing_directories):
                created = False
                try:
                    directory.mkdir(mode=0o700)
                    created = True
                except FileExistsError:
                    if not _WINDOWS_PERMISSION_MODEL:
                        DynamicExtensionResolver._wait_for_cache_directory_normalization(
                            directory,
                            exact_mode=directory == path,
                        )
                if created and not _WINDOWS_PERMISSION_MODEL:
                    directory.chmod(0o700)
                directory_metadata = directory.lstat()
                if not stat.S_ISDIR(directory_metadata.st_mode):
                    _fail(
                        "ARTIFACT_CACHE_CORRUPT",
                        f"verified artifact cache directory is invalid: {directory}",
                    )
                if _WINDOWS_PERMISSION_MODEL and directory != path:
                    _secure_native_cache_path(directory, directory=True)
        except OSError as exception:
            raise DynamicExtensionError(
                "ARTIFACT_SNAPSHOT_FAILED", f"could not create verified artifact cache directory: {path}"
            ) from exception

    @staticmethod
    def _prepare_cache_directory(path: Path) -> Path:
        if path.parent == path:
            _fail("ARTIFACT_CACHE_CORRUPT", f"verified artifact cache must not be a filesystem root: {path}")
        DynamicExtensionResolver._create_missing_cache_directories(path)
        try:
            directory_metadata = path.lstat()
        except OSError as exception:
            raise DynamicExtensionError(
                "ARTIFACT_CACHE_CORRUPT", f"could not inspect verified artifact cache directory: {path}"
            ) from exception
        if not stat.S_ISDIR(directory_metadata.st_mode):
            _fail("ARTIFACT_CACHE_CORRUPT", f"verified artifact cache directory is invalid: {path}")
        if _WINDOWS_PERMISSION_MODEL:
            _secure_native_cache_path(path, directory=True)
            return path
        if directory_metadata.st_uid not in {0, os.geteuid()}:
            _fail("ARTIFACT_CACHE_CORRUPT", f"verified artifact cache directory has an untrusted owner: {path}")
        if stat.S_IMODE(directory_metadata.st_mode) != 0o700:
            _fail("ARTIFACT_CACHE_CORRUPT", f"verified artifact cache directory is not private: {path}")
        return path

    @staticmethod
    def _prepare_created_private_directory(path: Path, *, description: str) -> Path:
        if not _WINDOWS_PERMISSION_MODEL:
            try:
                path.chmod(0o700)
            except OSError as exception:
                raise DynamicExtensionError(
                    "ARTIFACT_SNAPSHOT_FAILED",
                    f"could not normalize {description} directory: {path}",
                ) from exception
        return DynamicExtensionResolver._prepare_cache_directory(path)

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
        snapshot_root = Path(snapshot_directory)
        DynamicExtensionResolver._prepare_created_private_directory(
            snapshot_root,
            description="descriptor snapshot",
        )
        snapshot_path = snapshot_root / expected_filename
        artifact_digest = _copy_and_hash_artifact(artifact_path, snapshot_path)
        if not _WINDOWS_PERMISSION_MODEL:
            _make_snapshot_read_only(snapshot_path, description="descriptor snapshot")
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
    # Preserve the complete runtime SourceID in the descriptor. DuckDB's native
    # footer compatibility identity is validated independently.
    extension_compatibility_version = _native_extension_compatibility_version()
    if metadata.duckdb_version != extension_compatibility_version:
        _fail(
            "SOURCE_ID_MISMATCH",
            f"artifact footer compatibility version is {metadata.duckdb_version}, "
            f"runtime expects {extension_compatibility_version}",
        )
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
    )


def _native_platform() -> str:
    from vane import _native

    try:
        platform = _native._dynamic_extension_platform()
    except Exception as exception:
        raise DynamicExtensionError(
            "RUNTIME_IDENTITY_UNAVAILABLE", "DuckDB could not report its runtime platform"
        ) from exception
    if not isinstance(platform, str):
        _fail("RUNTIME_IDENTITY_UNAVAILABLE", "DuckDB returned a non-string runtime platform")
    return _validate_platform(platform)


def _native_extension_compatibility_version() -> str:
    from vane import _native

    try:
        version = _native._dynamic_extension_compatibility_version()
    except Exception as exception:
        raise DynamicExtensionError(
            "RUNTIME_IDENTITY_UNAVAILABLE",
            "DuckDB could not report its extension compatibility version",
        ) from exception
    if not isinstance(version, str):
        _fail("RUNTIME_IDENTITY_UNAVAILABLE", "DuckDB returned a non-string extension compatibility version")
    return _require_string(version, "runtime DuckDB extension compatibility version")


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


def _secure_native_cache_path(path: Path, *, directory: bool) -> None:
    from vane import _native

    try:
        _native._secure_dynamic_extension_cache_path(str(path), directory=directory)
    except Exception as exception:
        raise DynamicExtensionError(
            "ARTIFACT_CACHE_CORRUPT",
            f"could not validate or apply a private Windows DACL to verified artifact cache path: {path}",
        ) from exception


def _native_cache_path_is_replaceable(path: Path) -> bool:
    from vane import _native

    try:
        replaceable = _native._dynamic_extension_cache_path_is_replaceable(str(path))
    except Exception as exception:
        raise DynamicExtensionError(
            "ARTIFACT_CACHE_CORRUPT",
            f"could not inspect verified artifact cache ancestor: {path}",
        ) from exception
    if not isinstance(replaceable, bool):
        _fail("ARTIFACT_CACHE_CORRUPT", "DuckDB returned invalid Windows cache ancestor state")
    return replaceable


def _cleanup_staging_directory(
    staging_directory: Path,
    staging_path: Path,
) -> None:
    if not (staging_directory.exists() or staging_directory.is_symlink()):
        return
    try:
        if _WINDOWS_PERMISSION_MODEL and staging_path.exists():
            # Windows refuses to unlink a read-only file. This path remains an
            # unpublished private copy because publication renames its parent
            # directory atomically instead of creating a hard link.
            staging_path.chmod(0o600)
        shutil.rmtree(staging_directory)
    except OSError as exception:
        raise DynamicExtensionError(
            "ARTIFACT_SNAPSHOT_FAILED",
            f"could not remove verified artifact staging directory: {staging_directory}",
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
    "DEFAULT_EXTENSION_CATALOG_URL",
    "DynamicExtensionDependency",
    "DynamicExtensionDescriptor",
    "DynamicExtensionError",
    "DynamicExtensionResolver",
    "ExtensionCatalogEntry",
    "ExtensionStatus",
    "LocalExtensionArtifact",
    "LocalExtensionProvider",
    "ResolvedDynamicExtension",
    "create_dynamic_extension_descriptor",
    "extension_catalog",
    "extension_statuses",
    "load_installed_extension",
    "vane_extensions",
]
