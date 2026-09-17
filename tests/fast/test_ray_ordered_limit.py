# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Global LIMIT and preview semantics on independently scheduled Ray inputs."""

from __future__ import annotations

import importlib.metadata
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from ray_test_profile import ray_test_object_store_options

import vane

pytestmark = [pytest.mark.real_ray, pytest.mark.ray_cluster_owner]


@pytest.fixture(scope="module")
def ordered_limit_cluster():
    import ray
    from ray.cluster_utils import Cluster

    assert not ray.is_initialized(), "ordered LIMIT tests must own their cluster"
    environment = pytest.MonkeyPatch()
    environment.delenv("VANE_RUNNER", raising=False)
    environment.setenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    cluster = Cluster(shutdown_at_exit=False)
    try:
        for cpus in (0, 1, 1):
            cluster.add_node(
                include_dashboard=False,
                num_cpus=cpus,
                num_gpus=0,
                **ray_test_object_store_options(),
            )
        ray.init(address=cluster.address, log_to_driver=True)
        assert sum(node["Alive"] and node["Resources"].get("CPU", 0) >= 1 for node in ray.nodes()) == 2
        yield
    finally:
        try:
            vane.teardown_runner()
        finally:
            try:
                ray.shutdown()
            finally:
                cluster.shutdown()
                environment.undo()


@pytest.fixture(params=[1, None], ids=["separate-tasks", "default-grouping"])
def limit_connection(request, ordered_limit_cluster, monkeypatch, tmp_path):
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    monkeypatch.setenv("VANE_DISTRIBUTED_NODE_COUNT", "2")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "2")
    monkeypatch.setenv("VANE_RAY_SCAN_SPLIT_MIN_COUNT", "4")
    if request.param is None:
        monkeypatch.delenv("VANE_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION", raising=False)
    else:
        monkeypatch.setenv("VANE_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION", str(request.param))
    monkeypatch.setenv("VANE_SHUFFLE_LOCAL_DIRS", str(tmp_path / "shuffle"))
    monkeypatch.setenv("VANE_PROGRESS", "0")
    with vane.connect() as connection:
        yield connection
        assert vane.runners.get_or_create_runner().name == "ray"


@pytest.fixture(params=["range", "parquet", "lance"])
def limit_source(request, limit_connection, tmp_path):
    connection = limit_connection
    if request.param == "range":
        return connection.sql("SELECT i AS id, 'value-' || i::VARCHAR AS value FROM range(40000) t(i)")

    ids = list(reversed(range(40000)))
    table = pa.table({"id": ids, "value": [f"value-{i}" for i in ids]})
    if request.param == "parquet":
        for fragment in range(4):
            pq.write_table(table.select(["id"]).slice(fragment * 10000, 10000), tmp_path / f"part-{fragment}.parquet")
        # Read the sort key directly, keeping this regression independent of
        # Parquet's late-materialization row-ID join.
        return connection.read_parquet(str(tmp_path / "part-*.parquet")).project("id, 'value-' || id::VARCHAR AS value")

    lance = pytest.importorskip("lance", reason="Lance SDK is required for the provider regression")
    try:
        importlib.metadata.distribution("vane-extension-lance")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("a matching installed Lance provider wheel is required")
    dataset_path = tmp_path / "source.lance"
    dataset = lance.write_dataset(table, str(dataset_path), max_rows_per_file=10000, max_rows_per_group=1000)
    assert len(dataset.get_fragments()) == 4
    vane.load_installed_extension("lance", connection=connection)
    return connection.sql(f"SELECT id, value FROM '{dataset_path.as_posix()}'")


@pytest.mark.parametrize(("descending", "offset"), [(False, 0), (True, 137)])
def test_ordered_limited_preview_matches_fetch(limit_source, descending, offset, capsys):
    relation = limit_source.filter(vane.col("id") >= 100).order("id DESC" if descending else "id").limit(5, offset)
    ids = list(range(100, 40000))
    if descending:
        ids.reverse()
    expected = [(i, f"value-{i}") for i in ids[offset : offset + 5]]

    assert relation.fetchall() == expected
    assert relation.limit(10000).fetchall() == expected
    relation.show()
    displayed = re.findall(r"│\s*(\d+)\s*│\s*value-(\d+)\s*│", capsys.readouterr().out)
    assert [(int(identifier), f"value-{value}") for identifier, value in displayed] == expected


@pytest.mark.parametrize("descending", [False, True])
def test_nested_ordered_limits_preserve_offsets_across_chunks(limit_source, descending):
    relation = limit_source.order("id DESC" if descending else "id").limit(9000, 137).limit(6000, 211)
    ids = list(range(40000))
    if descending:
        ids.reverse()
    expected = [(i, f"value-{i}") for i in ids[348 : 348 + 6000]]
    assert relation.fetchall() == expected
    assert relation.limit(10000).fetchall() == expected


def test_batch_limit_over_distributed_scan_uses_global_offset(limit_connection):
    relation = limit_connection.sql("SELECT i FROM range(40000) t(i) WHERE i % 2 = 0").limit(17, 9)
    assert "LIMIT" in relation.explain()
    assert relation.fetchall() == [(i,) for i in range(18, 52, 2)]


@pytest.mark.parametrize("limit", [0, 5, 10000, 10001])
def test_preview_limits_handle_empty_and_threshold_boundaries(limit_connection, limit, capsys):
    relation = limit_connection.sql("SELECT i FROM range(40000) t(i)").order("i").limit(5).limit(limit)
    expected = [(i,) for i in range(min(5, limit))]
    assert relation.fetchall() == expected
    relation.show()
    displayed = re.findall(r"│\s*(\d+)\s*│", capsys.readouterr().out)
    assert [(int(value),) for value in displayed] == expected


def test_order_by_without_topn_preserves_order_through_limit_exchange(limit_connection):
    limit_connection.execute("SET disabled_optimizers='top_n'")
    relation = limit_connection.sql("SELECT i FROM range(40000) t(i)").order("i DESC").limit(6000, 137)
    assert "ORDER_BY" in relation.explain()
    assert relation.fetchall() == [(i,) for i in range(39862, 33862, -1)]


def test_percent_limit_uses_complete_ordered_input(limit_connection):
    relation = limit_connection.sql("SELECT i FROM range(40000) t(i) ORDER BY i DESC LIMIT 15% OFFSET 137")
    expected = [(i,) for i in range(39862, 33862, -1)]
    assert relation.fetchall() == expected
    assert relation.limit(10000).fetchall() == expected
