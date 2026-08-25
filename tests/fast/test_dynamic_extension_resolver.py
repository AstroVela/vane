# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import gc
import hashlib
import os
import weakref
from dataclasses import replace
from pathlib import Path

import pytest

import vane
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

    def fetchall(self):
        return list(self._rows)


class RecordingConnection:
    def __init__(self, platform, *, loaded_names=(), before_load=None):
        self.platform = platform
        self.loaded_paths = []
        self.loaded_payloads = []
        self.loaded_names = set(loaded_names)
        self.before_load = before_load

    def execute(self, query):
        if query == "SELECT platform FROM pragma_platform()":
            return _Result([(self.platform,)])
        if query == "SELECT extension_name FROM duckdb_extensions() WHERE loaded ORDER BY extension_name":
            return _Result([(name,) for name in sorted(self.loaded_names)])
        raise AssertionError(f"unexpected query: {query}")

    def load_extension(self, extension):
        path = Path(extension)
        if self.before_load is not None:
            self.before_load(path)
        self.loaded_paths.append(path)
        self.loaded_payloads.append(path.read_bytes())
        self.loaded_names.add(path.name.removesuffix(".duckdb_extension"))


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


def _resolver(*artifacts, trust_identity="local-tests"):
    provider = LocalExtensionProvider(trust_identity, artifacts)
    return DynamicExtensionResolver(trusted_identities={trust_identity}, providers=[provider])


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
        dependencies=(
            DynamicExtensionDependency(
                name=dependency.name,
                extension_version=dependency.extension_version,
                sha256=dependency.sha256,
            ),
        ),
    )

    restored = DynamicExtensionDescriptor.from_json(root.to_json())

    assert restored == root
    assert restored.dependencies[0].identity == dependency.identity
    assert restored.to_json() == root.to_json()


def test_resolver_loads_dependencies_before_root_and_caches_digest(tmp_path):
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
        dependencies=(
            DynamicExtensionDependency(
                name=dependency.name,
                extension_version=dependency.extension_version,
                sha256=dependency.sha256,
            ),
        ),
    )
    resolver = _resolver(
        LocalExtensionArtifact(dependency, dependency_path),
        LocalExtensionArtifact(root, root_path),
    )
    connection = RecordingConnection(platform)

    loaded = resolver.load(connection, root)
    resolver.load(connection, root)

    assert loaded.identity == root.identity
    assert [path.name for path in connection.loaded_paths] == [dependency_path.name, root_path.name]
    assert all(path != source for path, source in zip(connection.loaded_paths, [dependency_path, root_path]))
    assert resolver.loaded_identities(connection) == (dependency.identity, root.identity)


def test_resolver_cache_is_scoped_to_the_connection(tmp_path):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    descriptor = _descriptor(artifact_path, name="root")
    resolver = _resolver(LocalExtensionArtifact(descriptor, artifact_path))
    first_connection = RecordingConnection(platform)
    second_connection = RecordingConnection(platform)

    resolver.load(first_connection, descriptor)
    resolver.load(second_connection, descriptor)

    assert [path.name for path in first_connection.loaded_paths] == [artifact_path.name]
    assert [path.name for path in second_connection.loaded_paths] == [artifact_path.name]


def test_resolver_rejects_a_different_digest_for_an_already_loaded_extension_name(tmp_path):
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
    resolver = DynamicExtensionResolver(trusted_identities={"local-tests"})
    connection = RecordingConnection(platform)

    resolver.load(connection, first, artifact=first_path)
    with pytest.raises(DynamicExtensionError, match="LOADED_NAME_CONFLICT"):
        resolver.load(connection, second, artifact=second_path)

    assert [path.name for path in connection.loaded_paths] == [first_path.name]


def test_resolver_rejects_an_extension_loaded_outside_its_cache(tmp_path):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    descriptor = _descriptor(artifact_path, name="root")
    resolver = DynamicExtensionResolver(trusted_identities={"local-tests"})
    connection = RecordingConnection(platform, loaded_names={"root"})

    with pytest.raises(DynamicExtensionError, match="LOADED_OUTSIDE_RESOLVER"):
        resolver.load(connection, descriptor, artifact=artifact_path)

    assert connection.loaded_paths == []


def test_resolver_loads_the_verified_snapshot_if_the_provider_path_is_replaced(tmp_path):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    verified_payload = artifact_path.read_bytes()
    descriptor = _descriptor(artifact_path, name="root")
    resolver = DynamicExtensionResolver(trusted_identities={"local-tests"})

    def replace_provider_path(_snapshot_path):
        artifact_path.write_bytes(b"replacement after verification")

    connection = RecordingConnection(platform, before_load=replace_provider_path)

    loaded = resolver.load(connection, descriptor, artifact=artifact_path)

    assert loaded.path != artifact_path
    assert connection.loaded_payloads == [verified_payload]
    assert hashlib.sha256(connection.loaded_payloads[0]).hexdigest() == descriptor.sha256
    assert artifact_path.read_bytes() != verified_payload


def test_resolver_cache_does_not_keep_connections_alive(tmp_path):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    descriptor = _descriptor(artifact_path, name="root")
    resolver = DynamicExtensionResolver(trusted_identities={"local-tests"})
    connection = RecordingConnection(platform)
    connection_reference = weakref.ref(connection)

    resolver.load(connection, descriptor, artifact=artifact_path)
    assert len(resolver._loaded_by_connection) == 1

    del connection
    # The pytest timing plugin can retain the most recently returned Python
    # frames while output capture is active. Advance it past load()/resolve()
    # before checking that the resolver cache itself does not retain the key.
    resolver.loaded_identities(RecordingConnection(platform))
    gc.collect()

    assert connection_reference() is None
    assert len(resolver._loaded_by_connection) == 0


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
        dependencies=(
            DynamicExtensionDependency(
                name="injected",
                extension_version="injected-version",
                sha256="1" * 64,
            ),
        ),
    )
    resolver = _resolver(LocalExtensionArtifact(provider_descriptor, artifact_path))
    connection = RecordingConnection(platform)

    with pytest.raises(DynamicExtensionError, match="PROVIDER_DESCRIPTOR_MISMATCH"):
        resolver.load(connection, requested_descriptor)

    assert connection.loaded_paths == []


def test_resolver_rejects_conflicting_dependency_names_before_loading_anything(tmp_path):
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
    resolver = _resolver(
        LocalExtensionArtifact(first, first_path),
        LocalExtensionArtifact(second, second_path),
        LocalExtensionArtifact(root, root_path),
    )
    connection = RecordingConnection(platform)

    with pytest.raises(DynamicExtensionError, match="RESOLVED_NAME_CONFLICT"):
        resolver.load(connection, root)

    assert connection.loaded_paths == []


def test_resolver_rejects_an_unsupported_c_struct_api_before_loading(tmp_path):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id="v999.0.0",
        abi_type="C_STRUCT",
    )
    descriptor = _descriptor(artifact_path, name="root")
    resolver = DynamicExtensionResolver(trusted_identities={"local-tests"})
    connection = RecordingConnection(platform)

    with pytest.raises(DynamicExtensionError, match="CAPI_VERSION_MISMATCH"):
        resolver.load(connection, descriptor, artifact=artifact_path)

    assert connection.loaded_paths == []


def test_descriptor_creation_rejects_a_name_that_does_not_match_the_artifact(tmp_path):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )

    with pytest.raises(DynamicExtensionError, match="NAME_MISMATCH"):
        _descriptor(artifact_path, name="different")


def test_resolver_rejects_untrusted_descriptor_before_loading(tmp_path):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    descriptor = _descriptor(artifact_path, name="root", trust_identity="untrusted")
    resolver = DynamicExtensionResolver(trusted_identities={"local-tests"})
    connection = RecordingConnection(platform)

    with pytest.raises(DynamicExtensionError, match="TRUST_IDENTITY_UNTRUSTED"):
        resolver.load(connection, descriptor, artifact=artifact_path)

    assert connection.loaded_paths == []


def test_resolver_rejects_altered_artifact_before_loading(tmp_path):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    descriptor = _descriptor(artifact_path, name="root")
    artifact_path.write_bytes(artifact_path.read_bytes() + b"altered")
    resolver = DynamicExtensionResolver(trusted_identities={"local-tests"})
    connection = RecordingConnection(platform)

    with pytest.raises(DynamicExtensionError, match="DIGEST_MISMATCH"):
        resolver.load(connection, descriptor, artifact=artifact_path)

    assert connection.loaded_paths == []


def test_resolver_rejects_platform_source_id_and_vane_version_mismatches_before_loading(tmp_path):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    descriptor = _descriptor(artifact_path, name="root")
    resolver = DynamicExtensionResolver(trusted_identities={"local-tests"})
    connection = RecordingConnection(platform)

    with pytest.raises(DynamicExtensionError, match="PLATFORM_MISMATCH"):
        resolver.load(connection, replace(descriptor, platform="linux_arm64"), artifact=artifact_path)
    with pytest.raises(DynamicExtensionError, match="SOURCE_ID_MISMATCH"):
        resolver.load(connection, replace(descriptor, duckdb_source_id="a" * 40), artifact=artifact_path)
    with pytest.raises(DynamicExtensionError, match="VANE_VERSION_MISMATCH"):
        resolver.load(connection, replace(descriptor, vane_version="different-vane-version"), artifact=artifact_path)

    assert connection.loaded_paths == []


def test_resolver_rejects_missing_ordered_dependency_before_loading_root(tmp_path):
    platform = _runtime_platform()
    root_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    root = replace(
        _descriptor(root_path, name="root"),
        dependencies=(
            DynamicExtensionDependency(
                name="missing",
                extension_version="missing-version",
                sha256="1" * 64,
            ),
        ),
    )
    resolver = _resolver(LocalExtensionArtifact(root, root_path))
    connection = RecordingConnection(platform)

    with pytest.raises(DynamicExtensionError, match="DEPENDENCY_NOT_FOUND"):
        resolver.load(connection, root)

    assert connection.loaded_paths == []


@pytest.fixture(scope="module")
def staged_tpch_artifact():
    configured_path = os.environ.get("VANE_TEST_LOADABLE_EXTENSION_PATH")
    if configured_path is None:
        pytest.skip("set VANE_TEST_LOADABLE_EXTENSION_PATH to test a staged artifact")
    artifact_path = Path(configured_path).resolve()
    assert artifact_path.name == "tpch.duckdb_extension"
    assert artifact_path.is_file()
    return artifact_path


def test_resolver_loads_staged_tpch_artifact(staged_tpch_artifact):
    descriptor = create_dynamic_extension_descriptor(
        staged_tpch_artifact,
        name="tpch",
        trust_identity="vane-test-artifacts",
    )
    resolver = _resolver(
        LocalExtensionArtifact(descriptor, staged_tpch_artifact),
        trust_identity="vane-test-artifacts",
    )
    connection = vane.connect(config={"allow_unsigned_extensions": "true"})
    try:
        loaded = resolver.load(connection, descriptor)

        assert loaded.identity == descriptor.identity
        assert resolver.loaded_identities(connection) == (descriptor.identity,)
        assert connection.execute("SELECT count(*) FROM tpch_queries()").fetchone() == (22,)
    finally:
        connection.close()
