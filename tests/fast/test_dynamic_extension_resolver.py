# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

import vane
import vane.extensions as extension_module
from vane.extensions import (
    DynamicExtensionDependency,
    DynamicExtensionDescriptor,
    DynamicExtensionError,
    DynamicExtensionResolver,
    LocalExtensionArtifact,
    LocalExtensionProvider,
    create_dynamic_extension_descriptor,
)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _RecordingDatabase:
    def __init__(self):
        self.lock = threading.Lock()
        self.loaded = {}


class RecordingConnection:
    def __init__(self, platform, *, database=None, before_load=None):
        self.platform = platform
        self.database = database if database is not None else _RecordingDatabase()
        self.before_load = before_load
        self.load_calls = []
        self.loaded_paths = []
        self.loaded_payloads = []

    def execute(self, query):
        if query == "SELECT platform FROM pragma_platform()":
            return _Result([(self.platform,)])
        raise AssertionError(f"unexpected query: {query}")


@pytest.fixture
def fake_native_loader(monkeypatch):
    def load(path, connection):
        path = Path(path)
        canonical_name = extension_module._native_canonical_extension_name(str(path))
        connection.load_calls.append(path)
        with connection.database.lock:
            loaded = connection.database.loaded.get(canonical_name)
            if loaded is not None:
                return loaded
            if connection.before_load is not None:
                connection.before_load(path)
            metadata = extension_module._inspect_native_extension(path)
            loaded = extension_module._NativeLoadedExtension(
                canonical_name=canonical_name,
                full_path=str(path),
                install_mode="NOT_INSTALLED",
                extension_version=metadata.extension_version,
            )
            connection.database.loaded[canonical_name] = loaded
            connection.loaded_paths.append(path)
            connection.loaded_payloads.append(path.read_bytes())
            return loaded

    def loaded(extension, connection):
        canonical_name = extension_module._native_canonical_extension_name(extension)
        with connection.database.lock:
            return connection.database.loaded.get(canonical_name)

    monkeypatch.setattr(extension_module, "_load_native_extension", load)
    monkeypatch.setattr(extension_module, "_loaded_native_extension", loaded)


def _runtime_platform():
    connection = vane.connect()
    try:
        return connection.execute("SELECT platform FROM pragma_platform()").fetchone()[0]
    finally:
        connection.close()


def _write_extension_artifact(path, *, platform, source_id, extension_version="test-version", abi_type="CPP"):
    path.parent.mkdir(parents=True, exist_ok=True)
    footer = bytearray(512)
    fields = ["", "", "", abi_type, extension_version, source_id, platform, "4"]
    for index, value in enumerate(fields):
        start = index * 32
        footer[start : start + len(value)] = value.encode("ascii")
    path.write_bytes(b"Vane extension test payload" + footer)
    return path


def _descriptor(path, *, name, trust_identity="local-tests"):
    return create_dynamic_extension_descriptor(path, name=name, trust_identity=trust_identity)


def _resolver(cache_directory, *artifacts, trust_identity="local-tests"):
    providers = [LocalExtensionProvider(trust_identity, artifacts)] if artifacts else []
    return DynamicExtensionResolver(
        trusted_identities={trust_identity},
        providers=providers,
        cache_directory=cache_directory,
    )


def test_descriptor_round_trip_preserves_ordered_dependency_identity(tmp_path):
    platform = _runtime_platform()
    dependency_path = _write_extension_artifact(
        tmp_path / "dependency.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    root_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
        extension_version="root-version",
    )
    dependency = _descriptor(dependency_path, name="dependency")
    root = replace(
        _descriptor(root_path, name="root"),
        dependencies=(DynamicExtensionDependency(dependency.name, dependency.extension_version, dependency.sha256),),
    )

    restored = DynamicExtensionDescriptor.from_json(root.to_json())

    assert restored == root
    assert restored.dependencies[0].identity == dependency.identity
    assert restored.to_json() == root.to_json()


def test_descriptor_keeps_runtime_source_id_separate_from_footer_compatibility_version(tmp_path, monkeypatch):
    platform = _runtime_platform()
    compatibility_version = extension_module._native_extension_compatibility_version()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=compatibility_version,
    )
    full_source_id = "a" * 40
    monkeypatch.setattr(extension_module, "_runtime_identity", lambda: (full_source_id, vane.__version__))

    descriptor = _descriptor(artifact_path, name="root")
    resolved = _resolver(tmp_path / "verified").resolve(
        RecordingConnection(platform),
        descriptor,
        artifact=artifact_path,
    )

    assert descriptor.duckdb_source_id == full_source_id
    assert resolved[0].descriptor == descriptor


def test_descriptor_rejects_duckdb_aliases(tmp_path):
    artifact_path = _write_extension_artifact(
        tmp_path / "http.duckdb_extension",
        platform=_runtime_platform(),
        source_id=vane.__git_revision__,
    )

    with pytest.raises(DynamicExtensionError, match="canonical DuckDB extension name 'httpfs'"):
        _descriptor(artifact_path, name="http")


def test_resolver_loads_dependencies_in_order_and_reuses_content_cache(tmp_path, fake_native_loader):
    platform = _runtime_platform()
    dependency_path = _write_extension_artifact(
        tmp_path / "source" / "dependency.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    root_path = _write_extension_artifact(
        tmp_path / "source" / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
        extension_version="root-version",
    )
    dependency = _descriptor(dependency_path, name="dependency")
    root = replace(
        _descriptor(root_path, name="root"),
        dependencies=(DynamicExtensionDependency(dependency.name, dependency.extension_version, dependency.sha256),),
    )
    cache_directory = tmp_path / "verified"
    resolver = _resolver(
        cache_directory,
        LocalExtensionArtifact(dependency, dependency_path),
        LocalExtensionArtifact(root, root_path),
    )
    connection = RecordingConnection(platform)

    first = resolver.load(connection, root)
    second = resolver.load(connection, root)

    assert first == second
    assert [path.name for path in connection.loaded_paths] == [dependency_path.name, root_path.name]
    assert all(
        path.parent != source.parent for path, source in zip(connection.loaded_paths, [dependency_path, root_path])
    )
    assert resolver.loaded_identities(connection) == (dependency.identity, root.identity)
    assert len([path for path in cache_directory.rglob("*.duckdb_extension")]) == 2
    assert not [path for path in cache_directory.rglob(".*") if path.name.startswith((".dependency-", ".root-"))]
    assert extension_module._snapshot_mode_is_read_only(first.path.stat().st_mode)


def test_loaded_state_is_database_scoped_not_connection_scoped(tmp_path, fake_native_loader):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    descriptor = _descriptor(artifact_path, name="root")
    resolver = _resolver(tmp_path / "verified", LocalExtensionArtifact(descriptor, artifact_path))
    database = _RecordingDatabase()
    first_connection = RecordingConnection(platform, database=database)
    second_connection = RecordingConnection(platform, database=database)

    first = resolver.load(first_connection, descriptor)
    second = resolver.load(second_connection, descriptor)

    assert first.path == second.path
    assert first_connection.loaded_paths == [first.path]
    assert second_connection.loaded_paths == []
    assert resolver.loaded_identities(second_connection) == (descriptor.identity,)


def test_two_resolvers_accept_the_same_database_winner(tmp_path, fake_native_loader):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    descriptor = _descriptor(artifact_path, name="root")
    cache_directory = tmp_path / "verified"
    first_resolver = _resolver(cache_directory)
    second_resolver = _resolver(cache_directory)
    connection = RecordingConnection(platform)

    first = first_resolver.load(connection, descriptor, artifact=artifact_path)
    second = second_resolver.load(connection, descriptor, artifact=artifact_path)

    assert first.path == second.path
    assert first_resolver.loaded_identities(connection) == (descriptor.identity,)
    assert second_resolver.loaded_identities(connection) == (descriptor.identity,)


def test_resolver_rejects_a_different_database_winner(tmp_path, fake_native_loader):
    platform = _runtime_platform()
    first_path = _write_extension_artifact(
        tmp_path / "first" / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
        extension_version="first-version",
    )
    second_path = _write_extension_artifact(
        tmp_path / "second" / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
        extension_version="second-version",
    )
    first = _descriptor(first_path, name="root")
    second = _descriptor(second_path, name="root")
    connection = RecordingConnection(platform)

    _resolver(tmp_path / "verified-a").load(connection, first, artifact=first_path)
    with pytest.raises(DynamicExtensionError, match="LOADED_ARTIFACT_CONFLICT"):
        _resolver(tmp_path / "verified-b").load(connection, second, artifact=second_path)

    assert connection.loaded_paths[0].read_bytes() == first_path.read_bytes()


def test_resolver_rejects_an_extension_loaded_outside_the_verified_store(tmp_path, fake_native_loader):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    outside_path = tmp_path / "outside" / "root.duckdb_extension"
    outside_path.parent.mkdir()
    outside_path.write_bytes(artifact_path.read_bytes())
    descriptor = _descriptor(artifact_path, name="root")
    database = _RecordingDatabase()
    database.loaded["root"] = extension_module._NativeLoadedExtension(
        canonical_name="root",
        full_path=str(outside_path),
        install_mode="NOT_INSTALLED",
        extension_version=descriptor.extension_version,
    )
    connection = RecordingConnection(platform, database=database)

    with pytest.raises(DynamicExtensionError, match="LOADED_ARTIFACT_CONFLICT"):
        _resolver(tmp_path / "verified").load(connection, descriptor, artifact=artifact_path)

    assert connection.loaded_paths == []


def test_resolver_loads_the_verified_snapshot_if_the_provider_path_is_replaced(tmp_path, fake_native_loader):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    verified_payload = artifact_path.read_bytes()
    descriptor = _descriptor(artifact_path, name="root")

    def replace_provider_path(_snapshot_path):
        artifact_path.write_bytes(b"replacement after verification")

    connection = RecordingConnection(platform, before_load=replace_provider_path)
    loaded = _resolver(tmp_path / "verified").load(connection, descriptor, artifact=artifact_path)

    assert loaded.path != artifact_path
    assert connection.loaded_payloads == [verified_payload]
    assert hashlib.sha256(connection.loaded_payloads[0]).hexdigest() == descriptor.sha256
    assert artifact_path.read_bytes() != verified_payload


def test_content_cache_publication_is_safe_across_resolvers(tmp_path, fake_native_loader, monkeypatch):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    descriptor = _descriptor(artifact_path, name="root")
    cache_directory = tmp_path / "verified"
    connection = RecordingConnection(platform)
    resolvers = [_resolver(cache_directory), _resolver(cache_directory)]
    publication_barrier = threading.Barrier(2)
    copy_and_hash_artifact = extension_module._copy_and_hash_artifact

    def copy_before_publication(source, destination):
        digest = copy_and_hash_artifact(source, destination)
        publication_barrier.wait(timeout=10)
        return digest

    monkeypatch.setattr(extension_module, "_copy_and_hash_artifact", copy_before_publication)

    with ThreadPoolExecutor(max_workers=2) as executor:
        loaded = list(
            executor.map(
                lambda resolver: resolver.load(connection, descriptor, artifact=artifact_path),
                resolvers,
            )
        )

    assert loaded[0].path == loaded[1].path
    assert len(connection.loaded_paths) == 1
    assert len(list(cache_directory.rglob("*.duckdb_extension"))) == 1


def test_concurrent_different_artifacts_report_exactly_one_winner(tmp_path, fake_native_loader):
    platform = _runtime_platform()
    descriptors = []
    for index in range(2):
        path = _write_extension_artifact(
            tmp_path / str(index) / "root.duckdb_extension",
            platform=platform,
            source_id=vane.__git_revision__,
            extension_version=f"version-{index}",
        )
        descriptors.append((_descriptor(path, name="root"), path))
    connection = RecordingConnection(platform)

    def load(candidate):
        descriptor, path = candidate
        try:
            _resolver(tmp_path / descriptor.sha256).load(connection, descriptor, artifact=path)
        except DynamicExtensionError as exception:
            return exception.code
        return "LOADED"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(load, descriptors))

    assert sorted(outcomes) == ["LOADED", "LOADED_ARTIFACT_CONFLICT"]
    assert len(connection.loaded_paths) == 1


def test_default_cache_uses_duckdb_extension_directory(tmp_path, monkeypatch):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    descriptor = _descriptor(artifact_path, name="root")
    extension_directory = tmp_path / "duckdb-extensions"
    monkeypatch.setattr(extension_module, "_native_extension_directory", lambda _connection: extension_directory)

    resolved = DynamicExtensionResolver(trusted_identities={"local-tests"}).resolve(
        RecordingConnection(platform),
        descriptor,
        artifact=artifact_path,
    )

    assert resolved[0].path.parent.parent.parent == extension_directory / ".vane" / "verified-v1"
    if os.name != "nt":
        assert stat.S_IMODE(extension_directory.stat().st_mode) == 0o700
        assert stat.S_IMODE((extension_directory / ".vane").stat().st_mode) == 0o700


def test_content_cache_with_writable_snapshot_fails_closed(tmp_path):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    descriptor = _descriptor(artifact_path, name="root")
    resolver = _resolver(tmp_path / "verified")
    connection = RecordingConnection(platform)
    snapshot = resolver.resolve(connection, descriptor, artifact=artifact_path)[0].path
    snapshot.chmod(0o600)

    with pytest.raises(DynamicExtensionError, match="ARTIFACT_CACHE_CORRUPT"):
        resolver.resolve(connection, descriptor, artifact=artifact_path)


def test_windows_permission_model_secures_cached_paths_and_removes_staging_directories(tmp_path, monkeypatch):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    descriptor = _descriptor(artifact_path, name="root")
    cache_directory = tmp_path / "verified"
    secured_paths = []

    monkeypatch.setattr(extension_module, "_WINDOWS_PERMISSION_MODEL", True)
    monkeypatch.setattr(
        extension_module,
        "_secure_native_cache_path",
        lambda path, *, directory: secured_paths.append((path, directory)),
    )

    resolver = _resolver(cache_directory)
    first = resolver.resolve(RecordingConnection(platform), descriptor, artifact=artifact_path)[0]

    digest_directory = first.path.parent.parent
    artifact_directory = first.path.parent
    staging_directories = [path for path, directory in secured_paths if directory and path.name.startswith(".root-")]
    assert len(staging_directories) == 1
    assert secured_paths == [
        (cache_directory, True),
        (staging_directories[0], True),
        (digest_directory, True),
        (artifact_directory, True),
        (first.path, False),
    ]
    assert not [path for path in cache_directory.iterdir() if path.name.startswith(".root-")]
    assert extension_module._snapshot_mode_is_read_only(first.path.stat().st_mode)

    secured_paths.clear()
    second = resolver.resolve(RecordingConnection(platform), descriptor, artifact=artifact_path)[0]

    assert second == first
    assert secured_paths == [
        (cache_directory, True),
        (digest_directory, True),
        (artifact_directory, True),
        (first.path, False),
    ]
    assert extension_module._snapshot_mode_is_read_only(0o444)
    assert not extension_module._snapshot_mode_is_read_only(0o666)


def test_resolver_rejects_a_bare_trusted_identity_string():
    with pytest.raises(DynamicExtensionError, match="trusted_identities must be an iterable"):
        DynamicExtensionResolver(trusted_identities="local-tests")


def test_resolve_preserves_the_callers_pending_result(tmp_path):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    descriptor = _descriptor(artifact_path, name="root")
    connection = vane.connect()
    try:
        connection.execute("SELECT * FROM (VALUES (41), (42))")

        _resolver(tmp_path / "verified").resolve(connection, descriptor, artifact=artifact_path)

        assert connection.fetchall() == [(41,), (42,)]
    finally:
        connection.close()


def test_corrupt_content_cache_fails_closed(tmp_path):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    descriptor = _descriptor(artifact_path, name="root")
    resolver = _resolver(tmp_path / "verified")
    connection = RecordingConnection(platform)
    snapshot = resolver.resolve(connection, descriptor, artifact=artifact_path)[0].path
    snapshot.chmod(0o600)
    snapshot.write_bytes(b"corrupt")
    snapshot.chmod(0o400)

    with pytest.raises(DynamicExtensionError, match="ARTIFACT_CACHE_CORRUPT"):
        resolver.resolve(connection, descriptor, artifact=artifact_path)


@pytest.mark.skipif(os.name == "nt", reason="Windows directory privacy is represented by its DACL, not st_mode")
def test_shared_writable_content_cache_fails_closed(tmp_path):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    descriptor = _descriptor(artifact_path, name="root")
    cache_directory = tmp_path / "verified"
    cache_directory.mkdir(mode=0o700)
    cache_directory.chmod(0o770)

    with pytest.raises(DynamicExtensionError, match="ARTIFACT_CACHE_CORRUPT"):
        _resolver(cache_directory).resolve(
            RecordingConnection(platform),
            descriptor,
            artifact=artifact_path,
        )


@pytest.mark.skipif(os.name == "nt", reason="Windows cache isolation is represented by DACLs")
def test_cache_root_rejects_a_replaceable_ancestor_but_allows_a_sticky_parent(tmp_path):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    descriptor = _descriptor(artifact_path, name="root")
    shared_parent = tmp_path / "shared"
    shared_parent.mkdir(mode=0o777)
    shared_parent.chmod(0o777)
    cache_directory = shared_parent / "verified"
    resolver = _resolver(cache_directory)

    with pytest.raises(DynamicExtensionError, match="replaceable ancestor"):
        resolver.resolve(RecordingConnection(platform), descriptor, artifact=artifact_path)

    shared_parent.chmod(0o1777)
    resolved = resolver.resolve(RecordingConnection(platform), descriptor, artifact=artifact_path)

    assert resolved[0].path.is_file()


def test_resolver_rejects_a_requested_descriptor_that_differs_from_the_provider(tmp_path):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    provider_descriptor = _descriptor(artifact_path, name="root")
    requested_descriptor = replace(
        provider_descriptor,
        dependencies=(DynamicExtensionDependency("injected", "injected-version", "1" * 64),),
    )
    resolver = _resolver(
        tmp_path / "verified",
        LocalExtensionArtifact(provider_descriptor, artifact_path),
    )

    with pytest.raises(DynamicExtensionError, match="PROVIDER_DESCRIPTOR_MISMATCH"):
        resolver.resolve(RecordingConnection(platform), requested_descriptor)


def test_resolver_rejects_conflicting_dependency_names_before_snapshotting(tmp_path):
    platform = _runtime_platform()
    first_path = _write_extension_artifact(
        tmp_path / "first" / "shared.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
        extension_version="first-version",
    )
    second_path = _write_extension_artifact(
        tmp_path / "second" / "shared.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
        extension_version="second-version",
    )
    root_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    first = _descriptor(first_path, name="shared")
    second = _descriptor(second_path, name="shared")
    root = replace(
        _descriptor(root_path, name="root"),
        dependencies=(
            DynamicExtensionDependency(first.name, first.extension_version, first.sha256),
            DynamicExtensionDependency(second.name, second.extension_version, second.sha256),
        ),
    )
    cache_directory = tmp_path / "verified"
    resolver = _resolver(
        cache_directory,
        LocalExtensionArtifact(first, first_path),
        LocalExtensionArtifact(second, second_path),
        LocalExtensionArtifact(root, root_path),
    )

    with pytest.raises(DynamicExtensionError, match="RESOLVED_NAME_CONFLICT"):
        resolver.resolve(RecordingConnection(platform), root)

    assert not cache_directory.exists()


def test_resolver_rejects_an_unsupported_c_struct_api_before_loading(tmp_path, fake_native_loader):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id="v999.0.0",
        abi_type="C_STRUCT",
    )
    descriptor = _descriptor(artifact_path, name="root")
    connection = RecordingConnection(platform)

    with pytest.raises(DynamicExtensionError, match="CAPI_VERSION_MISMATCH"):
        _resolver(tmp_path / "verified").load(connection, descriptor, artifact=artifact_path)

    assert connection.load_calls == []


def test_descriptor_creation_rejects_a_name_that_does_not_match_the_artifact(tmp_path):
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=_runtime_platform(),
        source_id=vane.__git_revision__,
    )

    with pytest.raises(DynamicExtensionError, match="NAME_MISMATCH"):
        _descriptor(artifact_path, name="different")


def test_descriptor_creation_rejects_an_invalid_duckdb_footer(tmp_path):
    artifact_path = tmp_path / "root.duckdb_extension"
    artifact_path.write_bytes(b"not a DuckDB extension" + bytes(512))

    with pytest.raises(DynamicExtensionError, match="FOOTER_INVALID"):
        _descriptor(artifact_path, name="root")


def test_resolver_rejects_an_invalid_duckdb_footer_before_loading(tmp_path, fake_native_loader):
    artifact_path = tmp_path / "root.duckdb_extension"
    artifact_path.write_bytes(b"not a DuckDB extension" + bytes(512))
    descriptor = DynamicExtensionDescriptor(
        name="root",
        extension_version="test-version",
        abi_type="CPP",
        duckdb_source_id=vane.__git_revision__,
        vane_version=vane.__version__,
        platform=_runtime_platform(),
        sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        trust_identity="local-tests",
    )
    connection = RecordingConnection(descriptor.platform)

    with pytest.raises(DynamicExtensionError, match="FOOTER_INVALID"):
        _resolver(tmp_path / "verified").load(connection, descriptor, artifact=artifact_path)

    assert connection.load_calls == []


def test_descriptor_creation_hashes_the_same_snapshot_that_duckdb_inspects(tmp_path, monkeypatch):
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=_runtime_platform(),
        source_id=vane.__git_revision__,
    )
    expected_payload = artifact_path.read_bytes()
    inspect_native_extension = extension_module._inspect_native_extension

    def inspect_snapshot(snapshot_path):
        assert snapshot_path != artifact_path
        artifact_path.write_bytes(b"replacement during descriptor creation")
        return inspect_native_extension(snapshot_path)

    monkeypatch.setattr(extension_module, "_inspect_native_extension", inspect_snapshot)

    descriptor = _descriptor(artifact_path, name="root")

    assert descriptor.sha256 == hashlib.sha256(expected_payload).hexdigest()
    assert artifact_path.read_bytes() != expected_payload


def test_resolver_rejects_untrusted_descriptor_before_snapshotting(tmp_path):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    descriptor = _descriptor(artifact_path, name="root", trust_identity="untrusted")
    cache_directory = tmp_path / "verified"

    with pytest.raises(DynamicExtensionError, match="TRUST_IDENTITY_UNTRUSTED"):
        _resolver(cache_directory).resolve(RecordingConnection(platform), descriptor, artifact=artifact_path)

    assert not cache_directory.exists()


def test_resolver_rejects_altered_artifact_before_loading(tmp_path, fake_native_loader):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    descriptor = _descriptor(artifact_path, name="root")
    artifact_path.write_bytes(artifact_path.read_bytes() + b"altered")
    connection = RecordingConnection(platform)

    with pytest.raises(DynamicExtensionError, match="DIGEST_MISMATCH"):
        _resolver(tmp_path / "verified").load(connection, descriptor, artifact=artifact_path)

    assert connection.load_calls == []
    assert not list((tmp_path / "verified").iterdir())


def test_resolver_rejects_platform_source_id_and_vane_version_mismatches_before_snapshotting(tmp_path):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    descriptor = _descriptor(artifact_path, name="root")
    resolver = _resolver(tmp_path / "verified")
    connection = RecordingConnection(platform)

    with pytest.raises(DynamicExtensionError, match="PLATFORM_MISMATCH"):
        resolver.resolve(connection, replace(descriptor, platform="linux_arm64"), artifact=artifact_path)
    with pytest.raises(DynamicExtensionError, match="SOURCE_ID_MISMATCH"):
        resolver.resolve(connection, replace(descriptor, duckdb_source_id="a" * 40), artifact=artifact_path)
    with pytest.raises(DynamicExtensionError, match="VANE_VERSION_MISMATCH"):
        resolver.resolve(connection, replace(descriptor, vane_version="different-vane-version"), artifact=artifact_path)

    assert not (tmp_path / "verified").exists()


def test_resolver_rejects_missing_ordered_dependency_before_snapshotting(tmp_path):
    platform = _runtime_platform()
    root_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    root = replace(
        _descriptor(root_path, name="root"),
        dependencies=(DynamicExtensionDependency("missing", "missing-version", "1" * 64),),
    )
    cache_directory = tmp_path / "verified"
    resolver = _resolver(cache_directory, LocalExtensionArtifact(root, root_path))

    with pytest.raises(DynamicExtensionError, match="DEPENDENCY_NOT_FOUND"):
        resolver.resolve(RecordingConnection(platform), root)

    assert not cache_directory.exists()


@pytest.fixture(scope="module")
def staged_tpch_artifact():
    configured_path = os.environ.get("VANE_TEST_LOADABLE_EXTENSION_PATH")
    if configured_path is None:
        pytest.skip("set VANE_TEST_LOADABLE_EXTENSION_PATH to test a staged artifact")
    artifact_path = Path(configured_path).resolve()
    assert artifact_path.name == "tpch.duckdb_extension"
    assert artifact_path.is_file()
    return artifact_path


def test_resolver_loads_staged_tpch_artifact(staged_tpch_artifact, tmp_path):
    descriptor = create_dynamic_extension_descriptor(
        staged_tpch_artifact,
        name="tpch",
        trust_identity="vane-test-artifacts",
    )
    resolver = _resolver(
        tmp_path / "verified",
        LocalExtensionArtifact(descriptor, staged_tpch_artifact),
        trust_identity="vane-test-artifacts",
    )
    connection = vane.connect(config={"allow_unsigned_extensions": "true"})
    cursor = connection.cursor()
    try:
        loaded = resolver.load(connection, descriptor)
        native_loaded = vane._native._loaded_dynamic_extension("tpch", connection=cursor)

        assert loaded.identity == descriptor.identity
        assert native_loaded["full_path"] == str(loaded.path)
        assert resolver.loaded_identities(cursor) == (descriptor.identity,)
        assert connection.execute("SELECT count(*) FROM tpch_queries()").fetchone() == (22,)
    finally:
        cursor.close()
        connection.close()


def test_resolver_rejects_a_staged_artifact_preloaded_from_an_unverified_path(staged_tpch_artifact, tmp_path):
    descriptor = create_dynamic_extension_descriptor(
        staged_tpch_artifact,
        name="tpch",
        trust_identity="vane-test-artifacts",
    )
    resolver = _resolver(tmp_path / "verified", trust_identity="vane-test-artifacts")
    connection = vane.connect(config={"allow_unsigned_extensions": "true"})
    try:
        connection.load_extension(str(staged_tpch_artifact))

        with pytest.raises(DynamicExtensionError, match="LOADED_ARTIFACT_CONFLICT"):
            resolver.load(connection, descriptor, artifact=staged_tpch_artifact)

        native_loaded = vane._native._loaded_dynamic_extension("tpch", connection=connection)
        assert native_loaded["full_path"] == str(staged_tpch_artifact)
        assert resolver.loaded_identities(connection) == ()
    finally:
        connection.close()
