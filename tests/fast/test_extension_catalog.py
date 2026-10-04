# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
import time
from http.client import HTTPException
from itertools import repeat
from urllib.error import URLError

import pytest

import vane
import vane.extensions as extension_module
from vane.extensions import DynamicExtensionDescriptor, ExtensionCatalogEntry

_CATALOG_URL = "https://catalog.example/v1/index.json"


class _Distribution:
    def __init__(self, name, version):
        self.metadata = {"Name": name}
        self.version = version


class _EntryPoint:
    def __init__(self, name, distribution_name, distribution_version):
        self.name = name
        self.dist = _Distribution(distribution_name, distribution_version)
        self.load_calls = 0

    def load(self):
        self.load_calls += 1
        pytest.fail("extension status discovery must not initialize providers")


class _SnapshotConnection:
    def __init__(self, descriptors=()):
        self._entries = [descriptor.to_json() for descriptor in descriptors]

    def _export_dynamic_extension_snapshot_entries(self):
        return list(self._entries)


class _CatalogResponse:
    def __init__(self, contents, *, url=_CATALOG_URL, status=200):
        self.contents = contents
        self.url = url
        self.status = status
        self.read_limits = []
        self.read_offset = 0

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        return False

    def geturl(self):
        return self.url

    def read1(self, limit):
        self.read_limits.append(limit)
        chunk = self.contents[self.read_offset : self.read_offset + limit]
        self.read_offset += len(chunk)
        return chunk


class _BlockingCatalogResponse(_CatalogResponse):
    def __init__(self):
        super().__init__(b"")
        self.entered_read = threading.Event()
        self.release_read = threading.Event()
        self.finished = threading.Event()

    def __exit__(self, exception_type, exception, traceback):
        self.finished.set()
        return False

    def read1(self, limit):
        self.entered_read.set()
        self.release_read.wait(timeout=2)
        return b""


class _CatalogOpener:
    def __init__(self, result):
        self.result = result
        self.request = None
        self.timeout = None

    def open(self, request, *, timeout):
        self.request = request
        self.timeout = timeout
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def _catalog_document():
    return {
        "format_version": 1,
        "extensions": [
            {
                "extension_name": "iceberg",
                "distribution_name": "vane-extension-iceberg",
                "description": "Read and write Apache Iceberg tables from Vane.",
                "repository": "https://github.com/AstroVela/duckdb-iceberg",
                "publisher": "AstroVela",
                "license": "MIT",
            },
            {
                "extension_name": "lance",
                "distribution_name": "vane-extension-lance",
                "description": "Read and write Lance tables from Vane.",
                "repository": "https://github.com/AstroVela/lance-duckdb",
                "publisher": "AstroVela",
                "license": "Apache-2.0",
            },
            {
                "extension_name": "paimon",
                "distribution_name": "vane-extension-paimon",
                "description": "Read and write Apache Paimon tables from Vane.",
                "repository": "https://github.com/AstroVela/duckdb-paimon",
                "publisher": "AstroVela",
                "license": "Apache-2.0",
            },
        ],
    }


def _catalog_entries():
    return extension_module._parse_extension_catalog(_catalog_document())


def _loaded_descriptor(name="iceberg"):
    return DynamicExtensionDescriptor(
        name=name,
        extension_version="extension-version",
        abi_type="CPP",
        duckdb_source_id="a" * 40,
        vane_version="0.2.0",
        platform="linux_amd64",
        sha256="b" * 64,
        trust_identity="astrovela/vane",
    )


def test_extension_catalog_fetches_one_bounded_remote_index(monkeypatch):
    response = _CatalogResponse(json.dumps(_catalog_document()).encode())
    opener = _CatalogOpener(response)
    monkeypatch.setattr(extension_module, "_EXTENSION_CATALOG_OPENER", opener)

    entries = vane.extension_catalog(catalog_url=_CATALOG_URL)

    assert [entry.extension_name for entry in entries] == ["iceberg", "lance", "paimon"]
    assert opener.request.full_url == _CATALOG_URL
    assert opener.request.get_header("Accept") == "application/json"
    assert 0 < opener.timeout <= extension_module._EXTENSION_CATALOG_TIMEOUT_SECONDS
    assert response.read_limits
    assert max(response.read_limits) <= extension_module._EXTENSION_CATALOG_READ_CHUNK_BYTES


def test_extension_catalog_enforces_a_total_wall_clock_deadline(monkeypatch):
    response = _BlockingCatalogResponse()
    monkeypatch.setattr(extension_module, "_EXTENSION_CATALOG_OPENER", _CatalogOpener(response))
    monkeypatch.setattr(extension_module, "_EXTENSION_CATALOG_TIMEOUT_SECONDS", 0.5)

    started = time.monotonic()
    try:
        with pytest.raises(extension_module.DynamicExtensionError, match="VANE_DYNAMIC_EXTENSION_CATALOG_UNAVAILABLE"):
            vane.extension_catalog(catalog_url=_CATALOG_URL)
    finally:
        response.release_read.set()

    assert time.monotonic() - started < 2
    assert response.entered_read.is_set()
    assert response.finished.wait(timeout=1)


@pytest.mark.parametrize(
    "catalog_url",
    [
        "http://catalog.example/v1/index.json",
        "https://catalog.example:443/v1/index.json",
        "https://user@catalog.example/v1/index.json",
        "https://catalog.example/v1/index.json?channel=dev",
        "https://catalog.example/v1/index.json#latest",
    ],
)
def test_extension_catalog_rejects_noncanonical_url_before_request(catalog_url, monkeypatch):
    opener = _CatalogOpener(AssertionError("network request was not expected"))
    monkeypatch.setattr(extension_module, "_EXTENSION_CATALOG_OPENER", opener)

    with pytest.raises(extension_module.DynamicExtensionError, match="VANE_DYNAMIC_EXTENSION_CATALOG_INVALID"):
        vane.extension_catalog(catalog_url=catalog_url)

    assert opener.request is None


def test_extension_catalog_does_not_fallback_when_endpoint_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        extension_module,
        "_EXTENSION_CATALOG_OPENER",
        _CatalogOpener(URLError("unavailable")),
    )

    with pytest.raises(extension_module.DynamicExtensionError, match="VANE_DYNAMIC_EXTENSION_CATALOG_UNAVAILABLE"):
        vane.extension_catalog(catalog_url=_CATALOG_URL)


def test_extension_catalog_wraps_an_interrupted_http_response(monkeypatch):
    monkeypatch.setattr(
        extension_module,
        "_EXTENSION_CATALOG_OPENER",
        _CatalogOpener(HTTPException("response interrupted")),
    )

    with pytest.raises(extension_module.DynamicExtensionError, match="VANE_DYNAMIC_EXTENSION_CATALOG_UNAVAILABLE"):
        vane.extension_catalog(catalog_url=_CATALOG_URL)


def test_extension_catalog_rejects_a_redirected_response(monkeypatch):
    response = _CatalogResponse(b'{"format_version": 1, "extensions": []}', url="https://other.example/index.json")
    monkeypatch.setattr(extension_module, "_EXTENSION_CATALOG_OPENER", _CatalogOpener(response))

    with pytest.raises(extension_module.DynamicExtensionError, match="VANE_DYNAMIC_EXTENSION_CATALOG_UNAVAILABLE"):
        vane.extension_catalog(catalog_url=_CATALOG_URL)


def test_extension_catalog_rejects_an_unsuccessful_response(monkeypatch):
    response = _CatalogResponse(b'{"format_version": 1, "extensions": []}', status=204)
    monkeypatch.setattr(extension_module, "_EXTENSION_CATALOG_OPENER", _CatalogOpener(response))

    with pytest.raises(extension_module.DynamicExtensionError, match="VANE_DYNAMIC_EXTENSION_CATALOG_UNAVAILABLE"):
        vane.extension_catalog(catalog_url=_CATALOG_URL)


def test_extension_catalog_rejects_oversized_response(monkeypatch):
    response = _CatalogResponse(b"x" * (extension_module._EXTENSION_CATALOG_MAX_BYTES + 1))
    monkeypatch.setattr(extension_module, "_EXTENSION_CATALOG_OPENER", _CatalogOpener(response))

    with pytest.raises(extension_module.DynamicExtensionError, match="VANE_DYNAMIC_EXTENSION_CATALOG_INVALID"):
        vane.extension_catalog(catalog_url=_CATALOG_URL)


def test_extension_catalog_rejects_duplicate_json_keys(monkeypatch):
    response = _CatalogResponse(b'{"format_version": 1, "format_version": 1, "extensions": []}')
    monkeypatch.setattr(extension_module, "_EXTENSION_CATALOG_OPENER", _CatalogOpener(response))

    with pytest.raises(extension_module.DynamicExtensionError, match="VANE_DYNAMIC_EXTENSION_CATALOG_INVALID"):
        vane.extension_catalog(catalog_url=_CATALOG_URL)


def test_extension_catalog_rejects_a_lone_unicode_surrogate(monkeypatch):
    document = _catalog_document()
    document["extensions"][0]["description"] = "\ud800"
    response = _CatalogResponse(json.dumps(document).encode())
    monkeypatch.setattr(extension_module, "_EXTENSION_CATALOG_OPENER", _CatalogOpener(response))

    with pytest.raises(extension_module.DynamicExtensionError, match="VANE_DYNAMIC_EXTENSION_CATALOG_INVALID"):
        vane.extension_catalog(catalog_url=_CATALOG_URL)


def test_extension_catalog_rejects_names_with_ambiguous_distribution_normalization():
    document = _catalog_document()
    document["extensions"][0]["extension_name"] = "ice__berg"
    document["extensions"][0]["distribution_name"] = "vane-extension-ice-berg"

    with pytest.raises(extension_module.DynamicExtensionError, match="VANE_DYNAMIC_EXTENSION_CATALOG_INVALID"):
        extension_module._parse_extension_catalog(document)


@pytest.mark.parametrize(
    "document",
    [
        {"format_version": 2, "extensions": []},
        {"format_version": 1, "extensions": [], "unexpected": True},
        {
            "format_version": 1,
            "extensions": [
                {
                    "extension_name": "iceberg",
                    "distribution_name": "vane_extension_iceberg",
                    "description": "Iceberg",
                    "repository": "https://example.com/iceberg",
                    "publisher": "Example",
                    "license": "MIT",
                }
            ],
        },
        {
            "format_version": 1,
            "extensions": [
                {
                    "extension_name": "iceberg",
                    "distribution_name": "vane-extension-iceberg",
                    "description": "Iceberg",
                    "repository": "https://example.com:not-a-port/iceberg",
                    "publisher": "Example",
                    "license": "MIT",
                }
            ],
        },
        {
            "format_version": 1,
            "extensions": [
                {
                    "extension_name": "iceberg",
                    "distribution_name": "vane-extension-iceberg",
                    "description": "Iceberg",
                    "repository": "https:///iceberg",
                    "publisher": "Example",
                    "license": "MIT",
                }
            ],
        },
    ],
)
def test_extension_catalog_rejects_noncanonical_documents(document):
    with pytest.raises(extension_module.DynamicExtensionError, match="VANE_DYNAMIC_EXTENSION_CATALOG_INVALID"):
        extension_module._parse_extension_catalog(document)


def test_extension_statuses_combine_catalog_provider_and_loaded_state_without_importing(monkeypatch):
    descriptor = _loaded_descriptor()
    iceberg = _EntryPoint("iceberg", "vane_extension_iceberg", "0.2.0.1")
    community = _EntryPoint("community", "community-provider", "3.4.5")
    entry_points = (community, iceberg)
    monkeypatch.setattr(
        extension_module,
        "entry_points",
        lambda *, group: entry_points if group == "vane.dynamic_extension_providers" else (),
    )

    statuses = {
        status.extension_name: status
        for status in vane.extension_statuses(
            connection=_SnapshotConnection((descriptor,)),
            catalog=_catalog_entries(),
        )
    }

    assert list(statuses) == ["community", "iceberg", "lance", "paimon"]
    assert statuses["iceberg"] == extension_module.ExtensionStatus(
        extension_name="iceberg",
        cataloged=True,
        installed=True,
        loadable=True,
        loaded=True,
        description="Read and write Apache Iceberg tables from Vane.",
        distribution_name="vane-extension-iceberg",
        installed_distribution_name="vane-extension-iceberg",
        distribution_version="0.2.0.1",
        repository="https://github.com/AstroVela/duckdb-iceberg",
        publisher="AstroVela",
        license="MIT",
        provider_distributions=("vane-extension-iceberg==0.2.0.1",),
        provider_count=1,
        extension_version="extension-version",
        trust_identity="astrovela/vane",
        artifact_sha256="b" * 64,
    )
    assert statuses["community"].cataloged is False
    assert statuses["community"].installed_distribution_name == "community-provider"
    assert statuses["community"].loadable is True
    assert statuses["lance"].installed is False
    assert statuses["lance"].loadable is False
    assert [entry_point.load_calls for entry_point in entry_points] == [0, 0]


def test_extension_statuses_report_ambiguous_providers_without_selecting_one(monkeypatch):
    entry_points = (
        _EntryPoint("iceberg", "vane-extension-iceberg", "1"),
        _EntryPoint("iceberg", "another-iceberg-provider", "2"),
    )
    monkeypatch.setattr(extension_module, "entry_points", lambda *, group: entry_points)

    status = next(
        status
        for status in vane.extension_statuses(
            connection=_SnapshotConnection(),
            catalog=_catalog_entries(),
        )
        if status.extension_name == "iceberg"
    )

    assert status.installed is True
    assert status.loadable is False
    assert status.provider_count == 2
    assert status.provider_distributions == (
        "another-iceberg-provider==2",
        "vane-extension-iceberg==1",
    )
    assert status.installed_distribution_name is None
    assert status.distribution_version is None
    assert [entry_point.load_calls for entry_point in entry_points] == [0, 0]


def test_extension_statuses_accept_an_explicit_empty_catalog_without_network(monkeypatch):
    monkeypatch.setattr(
        extension_module,
        "_EXTENSION_CATALOG_OPENER",
        _CatalogOpener(AssertionError("network request was not expected")),
    )
    monkeypatch.setattr(extension_module, "entry_points", lambda *, group: ())

    assert vane.extension_statuses(connection=_SnapshotConnection(), catalog=()) == ()


def test_extension_statuses_validate_explicit_catalog_entries(monkeypatch):
    monkeypatch.setattr(extension_module, "entry_points", lambda *, group: ())
    invalid_entry = ExtensionCatalogEntry(
        extension_name="iceberg",
        distribution_name="another-package",
        description="Iceberg",
        repository="https://example.com/iceberg",
        publisher="Example",
        license="MIT",
    )

    with pytest.raises(extension_module.DynamicExtensionError, match="VANE_DYNAMIC_EXTENSION_CATALOG_INVALID"):
        vane.extension_statuses(connection=_SnapshotConnection(), catalog=(invalid_entry,))


def test_extension_statuses_bound_an_explicit_catalog_iterable():
    with pytest.raises(extension_module.DynamicExtensionError, match="VANE_DYNAMIC_EXTENSION_CATALOG_INVALID"):
        vane.extension_statuses(
            connection=_SnapshotConnection(),
            catalog=repeat(_catalog_entries()[0]),
        )


def test_vane_extensions_returns_a_queryable_relation(duckdb_cursor, monkeypatch):
    monkeypatch.setattr(extension_module, "entry_points", lambda *, group: ())

    relation = vane.vane_extensions(connection=duckdb_cursor, catalog=_catalog_entries())

    assert relation.columns == [
        "extension_name",
        "loaded",
        "installed",
        "loadable",
        "cataloged",
        "description",
        "extension_version",
        "distribution_name",
        "installed_distribution_name",
        "distribution_version",
        "repository",
        "publisher",
        "license",
        "provider_distributions",
        "provider_count",
        "trust_identity",
        "artifact_sha256",
    ]
    assert relation.project("extension_name, installed, loaded").order("extension_name").fetchall() == [
        ("iceberg", False, False),
        ("lance", False, False),
        ("paimon", False, False),
    ]
