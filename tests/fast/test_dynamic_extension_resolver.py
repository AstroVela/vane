# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import os
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
    ResolvedDynamicExtension,
    create_dynamic_extension_descriptor,
)


class _Result:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class RecordingConnection:
    def __init__(self, platform):
        self.platform = platform
        self.loaded_paths = []

    def execute(self, query):
        assert query == "SELECT platform FROM pragma_platform()"
        return _Result((self.platform,))

    def load_extension(self, extension):
        self.loaded_paths.append(Path(extension))


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


class _InstalledProviderEntryPoint:
    def __init__(self, name, provider):
        self.name = name
        self._provider = provider

    def load(self):
        return lambda: self._provider


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
    assert connection.loaded_paths == [dependency_path, root_path]
    assert resolver.loaded_identities(connection) == (dependency.identity, root.identity)


def test_resolver_captures_ordered_immutable_snapshot_without_local_paths(tmp_path):
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

    resolver.load(connection, root)

    assert extension_module._capture_dynamic_extension_snapshot(connection) == [
        {
            "descriptor": dependency.to_dict(),
            "dependency_order": [],
        },
        {
            "descriptor": root.to_dict(),
            "dependency_order": [dependency.identity],
        },
    ]
    snapshot_json = str(extension_module._capture_dynamic_extension_snapshot(connection))
    assert str(dependency_path) not in snapshot_json
    assert str(root_path) not in snapshot_json


def test_vane_connection_cursors_share_dynamic_snapshot_state():
    connection = vane.connect()
    cursor = connection.cursor()
    try:
        platform = connection.execute("SELECT platform FROM pragma_platform()").fetchone()[0]
        descriptor = DynamicExtensionDescriptor(
            name="dynamic_test",
            extension_version="test-version",
            abi_type="CPP",
            duckdb_source_id=vane.__git_revision__,
            vane_version=vane.__version__,
            platform=platform,
            sha256="1" * 64,
            trust_identity="local-tests",
        )
        extension_module._record_dynamic_extension_snapshot_entry(
            cursor,
            ResolvedDynamicExtension(descriptor, Path("/not-a-worker-artifact-path")),
        )

        assert extension_module._capture_dynamic_extension_snapshot(connection) == [
            {
                "descriptor": descriptor.to_dict(),
                "dependency_order": [],
            }
        ]
    finally:
        cursor.close()
        connection.close()


def test_worker_replay_uses_preinstalled_providers_in_snapshot_dependency_order(tmp_path, monkeypatch):
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
    coordinator = RecordingConnection(platform)
    _resolver(
        LocalExtensionArtifact(dependency, dependency_path),
        LocalExtensionArtifact(root, root_path),
    ).load(coordinator, root)
    snapshot = extension_module._capture_dynamic_extension_snapshot(coordinator)
    dependency_provider = LocalExtensionProvider(
        "local-tests",
        (LocalExtensionArtifact(dependency, dependency_path),),
    )
    root_provider = LocalExtensionProvider(
        "local-tests",
        (LocalExtensionArtifact(root, root_path),),
    )
    monkeypatch.setattr(
        extension_module,
        "entry_points",
        lambda *, group: (
            (
                _InstalledProviderEntryPoint("dependency", dependency_provider),
                _InstalledProviderEntryPoint("root", root_provider),
            )
            if group == "vane.dynamic_extension_providers"
            else ()
        ),
    )
    worker = RecordingConnection(platform)

    extension_module._replay_dynamic_extension_snapshot(worker, snapshot)
    extension_module._replay_dynamic_extension_snapshot(worker, snapshot)

    assert worker.loaded_paths == [dependency_path, root_path]
    assert extension_module._capture_dynamic_extension_snapshot(worker) == snapshot


def test_worker_replay_rejects_missing_or_mismatched_installed_provider_before_loading(tmp_path, monkeypatch):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    descriptor = _descriptor(artifact_path, name="root")
    snapshot = [
        {
            "descriptor": descriptor.to_dict(),
            "dependency_order": [],
        }
    ]
    worker = RecordingConnection(platform)
    monkeypatch.setattr(extension_module, "entry_points", lambda *, group: ())

    with pytest.raises(DynamicExtensionError, match="PROVIDER_NOT_FOUND"):
        extension_module._replay_dynamic_extension_snapshot(worker, snapshot)

    mismatched_descriptor = replace(descriptor, trust_identity="different-local-tests")
    mismatched_provider = LocalExtensionProvider(
        "different-local-tests",
        (LocalExtensionArtifact(mismatched_descriptor, artifact_path),),
    )
    monkeypatch.setattr(
        extension_module,
        "entry_points",
        lambda *, group: (_InstalledProviderEntryPoint("root", mismatched_provider),),
    )

    with pytest.raises(DynamicExtensionError, match="PROVIDER_DESCRIPTOR_MISMATCH"):
        extension_module._replay_dynamic_extension_snapshot(worker, snapshot)

    assert worker.loaded_paths == []


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

    assert first_connection.loaded_paths == [artifact_path]
    assert second_connection.loaded_paths == [artifact_path]


def test_resolver_reuses_the_connection_snapshot_across_resolver_instances(tmp_path):
    platform = _runtime_platform()
    artifact_path = _write_extension_artifact(
        tmp_path / "root.duckdb_extension",
        platform=platform,
        source_id=vane.__git_revision__,
    )
    descriptor = _descriptor(artifact_path, name="root")
    artifact = LocalExtensionArtifact(descriptor, artifact_path)
    connection = RecordingConnection(platform)

    _resolver(artifact).load(connection, descriptor)
    _resolver(artifact).load(connection, descriptor)

    assert connection.loaded_paths == [artifact_path]


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

    assert connection.loaded_paths == [first_path]


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
