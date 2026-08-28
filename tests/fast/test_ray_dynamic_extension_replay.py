# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import pytest

import vane
from vane.extensions import DynamicExtensionResolver, LocalExtensionProvider


def _configured_signed_provider():
    extension_name = os.environ.get("VANE_TEST_SIGNED_DYNAMIC_EXTENSION_NAME", "").strip()
    if not extension_name:
        if os.environ.get("VANE_REQUIRE_SIGNED_DYNAMIC_EXTENSION_FIXTURE") == "1":
            pytest.fail("CI requires VANE_TEST_SIGNED_DYNAMIC_EXTENSION_NAME")
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


def _physical_plan_with_dynamic_manifest(connection, query_id: str, manifest: list[dict[str, Any]]):
    logical_plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(connection.sql("SELECT 1"), query_id)
    physical_plan = logical_plan.to_physical_plan(connection)
    state = list(physical_plan.__getstate__())
    snapshot = dict(state[6])
    snapshot["dynamic_extensions"] = manifest
    state[6] = snapshot
    replay_plan = vane.ray_cxx.DistributedPhysicalPlan.__new__(vane.ray_cxx.DistributedPhysicalPlan)
    replay_plan.__setstate__(tuple(state))
    return replay_plan


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
                    assert connection._compare_and_record_dynamic_extension_snapshot_entry(
                        [], DynamicExtensionDescriptor.from_dict(existing_descriptor).to_json()
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


def test_real_ray_prepares_and_reuses_explicit_signed_extension_without_driver_provider(ray_local, monkeypatch):
    import pyarrow as pa
    import ray

    import vane.extensions as extension_module
    from vane import runners
    from vane.runners.ray.runner import RayRunner
    from vane.runners.ray.worker import RayWorkerActor

    provider, artifact = _configured_signed_provider()
    connection = vane.connect(
        config={
            "allow_unsigned_extensions": "true",
            "autoinstall_known_extensions": "true",
            "autoload_known_extensions": "true",
        }
    )
    resolver = DynamicExtensionResolver(
        trusted_identities={provider.trust_identity},
        providers=(provider,),
    )
    resolver.load(connection, artifact.descriptor)
    extension_security_settings = """
        SELECT
            CAST(current_setting('allow_unsigned_extensions') AS BOOLEAN),
            CAST(current_setting('autoinstall_known_extensions') AS BOOLEAN),
            CAST(current_setting('autoload_known_extensions') AS BOOLEAN)
    """
    assert connection.execute(extension_security_settings).fetchone() == (True, True, True)
    monkeypatch.setattr(
        extension_module,
        "entry_points",
        lambda *, group: pytest.fail(f"coordinator topology must not discover provider group {group}"),
    )
    runner = RayRunner(address=None, max_task_backlog=None)
    admission_actor = None
    monkeypatch.setattr(runners, "get_or_infer_runner_type", lambda: "ray")
    monkeypatch.setattr(runners, "get_or_create_runner", lambda: runner)
    try:
        for query_id in range(2):
            relation = connection.sql(f"SELECT hello('worker') AS extension_value, {query_id} AS replay_attempt")
            parts = list(runner.run_iter_tables(relation))
            result = pa.concat_tables([part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts])
            assert result.column(0).to_pylist() == [11]
            assert result.column(1).to_pylist() == [query_id]

        assert connection.execute(extension_security_settings).fetchone() == (True, True, True)

        admission_query_id = "real-ray-dynamic-extension-admission-retry"
        admission_plan = _physical_plan_with_dynamic_manifest(
            connection,
            admission_query_id,
            [artifact.descriptor.to_dict()],
        )
        plan_ref = ray.put(admission_plan)
        admission_actor = RayWorkerActor.options(num_cpus=0).remote(1, 0, 1 << 28, 1 << 28)
        first_fragment = {
            "fragment_id": f"{admission_query_id}:node:1",
            "plan": plan_ref,
            "query_id": admission_query_id,
        }
        second_fragment = {
            "fragment_id": f"{admission_query_id}:node:2",
            "plan": plan_ref,
            "query_id": admission_query_id,
        }

        assert ray.get(admission_actor.register_fragments.remote([first_fragment])) == {
            "registered": 1,
            "existing": 0,
            "total": 1,
        }
        first_stats = ray.get(admission_actor.stats_snapshot_databases.remote())
        assert first_stats == {
            "prepare_calls": 1,
            "cache_hits": 0,
            "created_total": 1,
            "active_databases": 1,
        }

        assert ray.get(admission_actor.register_fragments.remote([first_fragment])) == {
            "registered": 0,
            "existing": 1,
            "total": 1,
        }
        assert ray.get(admission_actor.stats_snapshot_databases.remote()) == first_stats

        assert ray.get(admission_actor.register_fragments.remote([second_fragment])) == {
            "registered": 1,
            "existing": 0,
            "total": 2,
        }
        assert ray.get(admission_actor.stats_snapshot_databases.remote()) == {
            "prepare_calls": 2,
            "cache_hits": 1,
            "created_total": 1,
            "active_databases": 1,
        }
        assert ray.get(admission_actor.drop_query_fragments.remote(admission_query_id)) == 2
        ray.get(admission_actor.fte_cleanup_query.remote(admission_query_id))
    finally:
        if admission_actor is not None:
            ray.kill(admission_actor, no_restart=True)
        runner.close()
        connection.close()


def test_real_ray_actor_admission_rejects_altered_signed_extension(ray_local):
    import ray

    from vane.runners.ray.worker import RayWorkerActor

    _provider, artifact = _configured_signed_provider()
    altered_descriptor = replace(artifact.descriptor, sha256="0" * 64)
    connection = vane.connect()
    query_id = "real-ray-altered-dynamic-extension"
    replay_plan = _physical_plan_with_dynamic_manifest(
        connection,
        query_id,
        [altered_descriptor.to_dict()],
    )
    actor = RayWorkerActor.options(
        num_cpus=0,
        runtime_env={
            "env_vars": {
                "VANE_TEST_SIGNED_DYNAMIC_EXTENSION_PATH": str(artifact.path),
                "VANE_TEST_DYNAMIC_EXTENSION_DESCRIPTOR_SHA256": altered_descriptor.sha256,
            }
        },
    ).remote(1, 0, 1 << 28, 1 << 28)
    try:
        with pytest.raises(Exception, match="VANE_DYNAMIC_EXTENSION_DIGEST_MISMATCH"):
            ray.get(
                actor.register_fragments.remote(
                    [
                        {
                            "fragment_id": f"{query_id}:node:1",
                            "plan": ray.put(replay_plan),
                            "query_id": query_id,
                        }
                    ]
                )
            )
        assert ray.get(actor.stats_fragments.remote())["fragments_total"] == 0
    finally:
        ray.kill(actor, no_restart=True)
        connection.close()


def test_real_ray_rejects_dynamic_extension_integrity_failures(ray_local, tmp_path):
    import ray

    from vane.runners.ray.worker import RayWorkerActor

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

    admission_connection = vane.connect()
    admission_query_id = "real-ray-missing-provider-admission"
    admission_plan = _physical_plan_with_dynamic_manifest(
        admission_connection,
        admission_query_id,
        [root],
    )
    admission_actor = RayWorkerActor.options(num_cpus=0).remote(1, 0, 1 << 28, 1 << 28)
    try:
        with pytest.raises(Exception, match="VANE_DYNAMIC_EXTENSION_PROVIDER_NOT_FOUND"):
            ray.get(
                admission_actor.register_fragments.remote(
                    [
                        {
                            "fragment_id": f"{admission_query_id}:node:1",
                            "plan": ray.put(admission_plan),
                            "query_id": admission_query_id,
                        }
                    ]
                )
            )
        assert ray.get(admission_actor.stats_fragments.remote())["fragments_total"] == 0
    finally:
        ray.kill(admission_actor, no_restart=True)
        admission_connection.close()

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
