# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in driver/head plus two Ray containers sharing only real S3 objects."""

from __future__ import annotations

import base64
import json
import os

import pytest
from iceberg_delete_helpers import DeleteFeature, commit_deletes, delete_reference, manifest_entries
from iceberg_multinode_cluster import IcebergMultinodeCluster
from iceberg_s3_helpers import S3_TABLES, append_rows, s3_tables

pytestmark = [
    pytest.mark.external_service,
    pytest.mark.real_ray,
    pytest.mark.ray_cluster_owner,
    pytest.mark.timeout(900),
]


def expected(feature):
    view = feature.current
    label = "renamed_label" if "renamed_label" in view.names else "label"
    label_index = view.names.index(label)
    rows = list(view.rows)
    result = {
        "rows": rows,
        "projection": [(row[0],) for row in rows],
        "filtered": [row for row in rows if row[0] % 4 == 1 or row[label_index] is None],
        "aggregate": [(len(rows), sum(row[0] for row in rows) if rows else None)],
    }
    if feature.history:
        result["history"] = feature.history[0].rows
    return json.loads(json.dumps(result))


def query_specs(namespace, name, feature):
    qualified = f'lake."{namespace}"."{name}"'
    label = "renamed_label" if "renamed_label" in feature.current.names else "label"
    specs = {
        "rows": f"SELECT * FROM {qualified}",
        "projection": f"SELECT id FROM {qualified}",
        "filtered": f"SELECT * FROM {qualified} WHERE id % 4 = 1 OR {label} IS NULL",
        "aggregate": f"SELECT count(*), sum(id)::BIGINT FROM {qualified}",
    }
    if feature.history:
        specs["history"] = f"SELECT * FROM {qualified} AT (VERSION => {feature.history[0].snapshot_id})"
    return specs


@pytest.fixture(scope="module")
def isolated_lake(tmp_path_factory):
    image = os.environ.get("VANE_TEST_ICEBERG_MULTINODE_IMAGE")
    if not image:
        pytest.skip("set VANE_TEST_ICEBERG_MULTINODE_IMAGE to the prepared immutable runtime image")
    directory = tmp_path_factory.mktemp("iceberg-multinode")
    cluster = IcebergMultinodeCluster(directory, image)
    try:
        cluster.start()
        with pytest.MonkeyPatch.context() as scoped:
            scoped.setenv("VANE_TEST_GRAVITINO_ICEBERG_S3_CONFIG", str(cluster.sdk_config))
            with s3_tables(directory, names=(*S3_TABLES, "later")) as (tables, features):
                commit_deletes(tables, "later", [{2: None, 4: 0}, {2: "row-9.点", 4: 1}], equality_ids=(2, 4))
                previous = features["later"]
                features["later"] = DeleteFeature(
                    "later", delete_reference(tables, "later"), previous.history, previous.file_count
                )
                queries, selections = {}, {}
                for name, feature in features.items():
                    specs = query_specs(tables.namespace, name, feature)
                    selections[name] = {mode: name + "_" + mode for mode in specs}
                    queries.update({selections[name][mode]: sql for mode, sql in specs.items()})
                request = {
                    "action": "freeze",
                    "endpoint": "http://rest:9001/iceberg",
                    "s3_endpoint": "objects:9000",
                    "queries": queries,
                }
                assert cluster.request(request)["frozen"] == len(queries)
                # Credential checks read fixture-owned payloads back over the
                # Docker control plane; writer keys never enter the driver.
                serialized = cluster.docker(
                    "exec",
                    cluster.containers["driver"],
                    "python",
                    "-I",
                    "-c",
                    "import base64,json; from pathlib import Path; "
                    "print(json.dumps([base64.b64encode(p.read_bytes()).decode() "
                    "for p in Path('/run/qualification/frozen').glob('*.pickle')]))",
                )
                for encoded in json.loads(serialized):
                    data = base64.b64decode(encoded)
                    assert cluster.config["admin_key"].encode() not in data
                    assert cluster.config["admin_secret"].encode() not in data
                commit_deletes(tables, "later", [{1: key} for key in range(27)], equality_ids=(1,))
                append_rows(tables, "later", [{"id": 800001, "label": "later", "region": None, "rank": None}])
                assert delete_reference(tables, "later").rows == ((800001, "later", None, None),)
                cluster.stop_rest_forwarder()
                cluster.docker(
                    "exec",
                    cluster.containers["driver"],
                    "python",
                    "-I",
                    "-c",
                    "from pathlib import Path; Path('/run/qualification/driver-only-canary').write_text('driver only')",
                )
                cluster.selections = selections
                yield cluster, tables, features
                events = cluster.events()
                assert not any(event["credentials"] or event["query"] for event in events)
                assert any(event["range"] for event in events)
                (directory / "object-audit.json").write_text(json.dumps(events, indent=2))
                cluster.object_audit = events
    finally:
        cluster.close()


def equal_rows(observed, wanted):
    assert sorted(observed, key=repr) == sorted(wanted, key=repr)


def test_distinct_nodes_have_private_filesystems_and_exact_installed_providers(isolated_lake):
    cluster, _tables, _features = isolated_lake
    probes = cluster.request({"action": "probe"})
    assert len({probe["node_id"] for probe in probes}) == len({probe["ip"] for probe in probes}) == 2
    assert len({probe["mount_namespace"] for probe in probes}) == 2
    assert not any(probe["canary_exists"] for probe in probes)
    for probe in probes:
        assert probe["providers"] == cluster.topology["providers"]
        assert probe["vane_version"] == cluster.topology["version"]
        assert not probe["writer_credentials_in_env"]
    for record in cluster.records.values():
        assert not any(mount["Type"] in {"bind", "volume"} for mount in record["mounts"])


@pytest.mark.parametrize("name", S3_TABLES)
def test_public_ray_reads_use_object_storage_across_isolated_nodes(isolated_lake, name):
    cluster, tables, features = isolated_lake
    feature = features[name]
    result = cluster.request(
        {
            "action": "public",
            "endpoint": "http://catalog:9001/iceberg",
            "s3_endpoint": "objects:9000",
            "namespace": tables.namespace,
            "name": name,
            "snapshot_id": feature.history[0].snapshot_id if feature.history else None,
        }
    )
    for mode, wanted in expected(feature).items():
        equal_rows(result[mode], wanted)


@pytest.mark.parametrize("name", (*S3_TABLES, "later"))
def test_frozen_queries_survive_rest_shutdown_metadata_denial_and_later_commits(isolated_lake, name):
    cluster, _tables, features = isolated_lake
    start = len(cluster.events())
    with cluster.blocked_metadata():
        result = cluster.request({"action": "execute", "queries": cluster.selections[name]})
    for mode, wanted in expected(features[name]).items():
        equal_rows(result[mode], wanted)
    assert not any("/metadata/" in event["path"] for event in cluster.events()[start:])


@pytest.mark.parametrize("name", ["identity", "positions", "equality"])
def test_disjoint_assignments_execute_on_both_nodes_with_original_deletes(isolated_lake, name):
    cluster, _tables, features = isolated_lake
    start = len(cluster.events())
    with cluster.blocked_metadata():
        result = cluster.request({"action": "disjoint", "query": name + "_rows"})
    equal_rows(result["rows"], expected(features[name])["rows"])
    assert len(set(result["worker_nodes"])) == len(result["owners"]) == 2
    assert result["cleaned"]
    events = cluster.events()[start:]
    worker_ips = {cluster.records[role]["ip"] for role in ("worker-a", "worker-b")}
    assert worker_ips <= {event["client"] for event in events if event["range"]}
    assert not any("/metadata/" in event["path"] for event in events)


@pytest.mark.parametrize("name", ["positions", "equality"])
def test_actor_failure_retries_on_the_other_node(isolated_lake, name):
    cluster, _tables, features = isolated_lake
    with cluster.blocked_metadata():
        result = cluster.request({"action": "retry", "query": name + "_rows"})
    equal_rows(result["rows"], expected(features[name])["rows"])
    assert len(result["owners"]) == 1 and result["cleaned"]


def test_whole_node_failure_retries_and_the_node_can_rejoin(isolated_lake):
    cluster, _tables, features = isolated_lake
    before = {node["id"] for node in cluster.topology["nodes"]}
    with cluster.blocked_metadata():
        result = cluster.request({"action": "node_failure", "query": "equality_rows"}, stop_node=True, timeout=360)
    equal_rows(result["rows"], expected(features["equality"])["rows"])
    assert len(result["owners"]) == 1 and result["cleaned"]
    assert len(before.intersection(node["id"] for node in cluster.topology["nodes"])) == 2


@pytest.mark.parametrize("content", [0, 2], ids=["data", "equality-delete"])
def test_missing_s3_object_fails_without_incomplete_results(isolated_lake, content):
    cluster, tables, _features = isolated_lake
    table = tables.catalog.load_table((tables.namespace, "equality"))
    path = next(
        entry.data_file.file_path for _manifest, entry in manifest_entries(table) if entry.data_file.content == content
    )
    denied = "/" + path.removeprefix("s3://")
    with cluster.blocked_metadata(denied=[denied]):
        with pytest.raises(AssertionError, match="(?i)(404|not found|file|open)"):
            cluster.request({"action": "execute", "queries": {"rows": "equality_rows"}})
    assert any(event["path"] == denied and event["status"] == 404 for event in cluster.events())


def test_cancellation_releases_remote_query_and_session(isolated_lake):
    cluster, _tables, _features = isolated_lake
    with cluster.blocked_metadata():
        result = cluster.request({"action": "cancel", "query": "equality_rows"})
    assert result["cleaned"] and not result["rows"]


def test_missing_worker_provider_fails_and_exact_offline_reinstall_restores_reads(isolated_lake):
    cluster, _tables, features = isolated_lake
    try:
        cluster.provider_installed("worker-b", False)
        with cluster.blocked_metadata():
            probes = cluster.request({"action": "probe", "query": "equality_rows"})
        broken = next(probe for probe in probes if probe["ip"] == cluster.records["worker-b"]["ip"])
        healthy = next(probe for probe in probes if probe is not broken)
        assert "error" in broken and "rows" not in broken
        assert "iceberg" in broken["error"].lower() or "provider" in broken["error"].lower()
        equal_rows(healthy["rows"], expected(features["equality"])["rows"])
    finally:
        cluster.provider_installed("worker-b", True)
    with cluster.blocked_metadata():
        restored = cluster.request({"action": "probe", "query": "equality_rows"})
    for probe in restored:
        assert probe["providers"] == cluster.topology["providers"]
        equal_rows(probe["rows"], expected(features["equality"])["rows"])
