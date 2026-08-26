# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import pytest

import vane
from vane.extensions import DynamicExtensionResolver, LocalExtensionProvider


def _configured_signed_provider():
    extension_name = os.environ.get("VANE_TEST_SIGNED_DYNAMIC_EXTENSION_NAME")
    if extension_name is None:
        pytest.skip("set VANE_TEST_SIGNED_DYNAMIC_EXTENSION_NAME to test a signed installed provider")
    matches = [
        entry_point
        for entry_point in entry_points(group="vane.dynamic_extension_providers")
        if entry_point.name == extension_name
    ]
    assert len(matches) == 1
    provider = matches[0].load()()
    assert isinstance(provider, LocalExtensionProvider)
    artifacts = tuple(
        artifact for artifact in provider._artifact_by_identity.values() if artifact.descriptor.name == extension_name
    )
    assert len(artifacts) == 1
    return provider, artifacts[0]


def _run_real_ray_dynamic_extension_rejection_matrix(cases: list[dict[str, Any]]) -> dict[str, str]:
    import vane.extensions as extension_module
    from vane.extensions import (
        DynamicExtensionDescriptor,
        DynamicExtensionError,
        LocalExtensionArtifact,
        LocalExtensionProvider,
    )

    class _InstalledProviderEntryPoint:
        def __init__(self, name, provider):
            self.name = name
            self._provider = provider

        def load(self):
            return lambda: self._provider

    original_entry_points = extension_module.entry_points
    results: dict[str, str] = {}
    try:
        for case in cases:
            provider_spec = case.get("provider")
            if provider_spec is None:
                installed_entry_points = ()
            else:
                provider_descriptor = DynamicExtensionDescriptor.from_dict(provider_spec["descriptor"])
                provider = LocalExtensionProvider(
                    provider_descriptor.trust_identity,
                    (
                        LocalExtensionArtifact(
                            provider_descriptor,
                            Path(provider_spec["path"]),
                        ),
                    ),
                )
                installed_entry_points = (_InstalledProviderEntryPoint(provider_descriptor.name, provider),)
            extension_module.entry_points = lambda *, group, installed_entry_points=installed_entry_points: (
                installed_entry_points if group == "vane.dynamic_extension_providers" else ()
            )

            connection = vane.connect()
            try:
                existing_descriptor = case.get("existing_descriptor")
                if existing_descriptor is not None:
                    connection._record_dynamic_extension_snapshot_entry(
                        DynamicExtensionDescriptor.from_dict(existing_descriptor).to_json()
                    )
                extension_module._prepare_dynamic_extension_snapshot(connection, case["manifest"])
            except DynamicExtensionError as exception:
                results[case["name"]] = exception.code
            except Exception as exception:
                results[case["name"]] = f"UNEXPECTED_{type(exception).__name__}: {exception}"
            else:
                results[case["name"]] = "UNEXPECTED_SUCCESS"
            finally:
                connection.close()
    finally:
        extension_module.entry_points = original_entry_points
    return results


def _descriptor_dict(
    *,
    name: str,
    platform: str,
    source_id: str,
    sha256: str,
    trust_identity: str = "local-tests",
    dependencies: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "name": name,
        "extension_version": "test-version",
        "abi_type": "CPP",
        "duckdb_source_id": source_id,
        "vane_version": vane.__version__,
        "platform": platform,
        "sha256": sha256,
        "trust_identity": trust_identity,
        "dependencies": list(dependencies or []),
    }


def test_real_ray_prepares_and_reuses_signed_dynamic_extension(ray_local, monkeypatch):
    import pyarrow as pa

    from vane import runners
    from vane.runners.ray.runner import RayRunner

    provider, artifact = _configured_signed_provider()
    connection = vane.connect()
    resolver = DynamicExtensionResolver(
        trusted_identities={provider.trust_identity},
        providers=(provider,),
    )
    resolver.load(connection, artifact.descriptor)
    runner = RayRunner(address=None, max_task_backlog=None)
    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "ray")
    monkeypatch.setattr(runners, "get_or_create_runner", lambda: runner)
    try:
        for query_id in range(2):
            relation = connection.sql(f"SELECT {query_id} AS replay_attempt")
            parts = list(runner.run_iter_tables(relation))
            result = pa.concat_tables([part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts])
            assert result.column("replay_attempt").to_pylist() == [query_id]
    finally:
        runner.close()
        connection.close()


def test_real_ray_rejects_dynamic_extension_integrity_failures(ray_local, tmp_path):
    import ray

    connection = vane.connect()
    try:
        platform = connection.execute("SELECT platform FROM pragma_platform()").fetchone()[0]
    finally:
        connection.close()

    artifact_path = tmp_path / "root.duckdb_extension"
    artifact_payload = b"real-Ray dynamic extension rejection fixture"
    artifact_path.write_bytes(artifact_payload)
    artifact_sha256 = hashlib.sha256(artifact_payload).hexdigest()
    root = _descriptor_dict(
        name="root",
        platform=platform,
        source_id=vane.__git_revision__,
        sha256=artifact_sha256,
    )
    missing_artifact = _descriptor_dict(
        name="missing",
        platform=platform,
        source_id=vane.__git_revision__,
        sha256=artifact_sha256,
    )
    altered = {**root, "sha256": "0" * 64}
    provider_with_different_trust = {**root, "trust_identity": "worker-local-tests"}
    wrong_platform = {**root, "platform": "test_invalid_platform"}
    wrong_source = {**root, "duckdb_source_id": "0" * 40}
    dependency_failure = {
        **root,
        "dependencies": [
            {
                "name": "dependency",
                "extension_version": "test-version",
                "sha256": artifact_sha256,
            }
        ],
    }
    existing_descriptor = _descriptor_dict(
        name="other",
        platform=platform,
        source_id=vane.__git_revision__,
        sha256=artifact_sha256,
    )
    cases = [
        {
            "name": "missing_provider",
            "manifest": [root],
        },
        {
            "name": "missing_artifact",
            "manifest": [missing_artifact],
            "provider": {
                "descriptor": missing_artifact,
                "path": str(tmp_path / "missing.duckdb_extension"),
            },
        },
        {
            "name": "altered_bytes",
            "manifest": [altered],
            "provider": {"descriptor": altered, "path": str(artifact_path)},
        },
        {
            "name": "trust_identity",
            "manifest": [root],
            "provider": {
                "descriptor": provider_with_different_trust,
                "path": str(artifact_path),
            },
        },
        {
            "name": "platform",
            "manifest": [wrong_platform],
            "provider": {"descriptor": wrong_platform, "path": str(artifact_path)},
        },
        {
            "name": "source_id",
            "manifest": [wrong_source],
            "provider": {"descriptor": wrong_source, "path": str(artifact_path)},
        },
        {
            "name": "dependency",
            "manifest": [dependency_failure],
        },
        {
            "name": "worker_disagreement",
            "manifest": [root],
            "existing_descriptor": existing_descriptor,
        },
    ]

    rejection_task = ray.remote(_run_real_ray_dynamic_extension_rejection_matrix)
    results = ray.get(rejection_task.remote(cases))

    assert results == {
        "missing_provider": "PROVIDER_NOT_FOUND",
        "missing_artifact": "ARTIFACT_NOT_FOUND",
        "altered_bytes": "DIGEST_MISMATCH",
        "trust_identity": "PROVIDER_DESCRIPTOR_MISMATCH",
        "platform": "PLATFORM_MISMATCH",
        "source_id": "SOURCE_ID_MISMATCH",
        "dependency": "SNAPSHOT_DEPENDENCY_ORDER",
        "worker_disagreement": "WORKER_DISAGREEMENT",
    }
