# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real Ray qualification of frozen per-file identity partition constants."""

from __future__ import annotations

import pickle

import pytest
from gravitino_iceberg_helpers import iceberg_tables
from iceberg_partition_helpers import append_partition_rows, identity_partition_tables, populate_identity_partitions
from iceberg_ray_helpers import await_result, drain_query, frozen_physical, native_workers, wait_for_open_scan
from iceberg_scan_helpers import RestForwarder, attach, connection, freeze, unavailable_metadata
from ray_native_scan_helpers import NativeScanWorkerTask, register_scan_query

import vane
from vane.catalog import IcebergCatalog
from vane.runners.ray import worker_handle as handles
from vane.runners.ray.runner import RayRunner

ray = pytest.importorskip("ray")
pytestmark = [pytest.mark.external_service, pytest.mark.real_ray]


@pytest.fixture(scope="module")
def partition_tables():
    with identity_partition_tables() as setup:
        yield setup


@pytest.fixture(autouse=True)
def ray_policy(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setenv("VANE_DISTRIBUTED_NODE_COUNT", "1")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "2")
    monkeypatch.setenv("VANE_FTE_RETRY_INITIAL_DELAY_S", "0")
    monkeypatch.setenv("VANE_FTE_STATUS_WAIT_TIMEOUT_S", "5")
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_INITIAL_BACKOFF_S", "0")
    monkeypatch.setenv("VANE_FTE_SPLIT_QUEUE_SPACE_WAIT_TIMEOUT_S", "0.1")


@pytest.mark.parametrize("name", ["partition_present", "partition_omitted", "partition_specs"])
def test_public_ray_identity_partition_relations(partition_tables, ray_local, name):
    endpoint, tables, features = partition_tables
    expected = list(features[name].rows)
    with connection(tables) as con:
        vane.load_installed_extension("iceberg", connection=con)
        catalog = IcebergCatalog.attach(con, 'Lake."partitions', endpoint=endpoint)
        relation = catalog.table(tables.namespace, name)
        assert relation._get_runner_type() == "ray"
        assert relation.columns == ["id", "label", "region"]
        assert relation.order("id").fetchall() == expected
        assert relation.filter("region IS NULL").project("region, id").order("id").fetchall() == [
            (region, key) for key, _, region in expected if region is None
        ]
        assert relation.filter("region = 'east.点' AND id >= 5").project("region").fetchall() == [
            (region,) for key, _, region in expected if region == "east.点" and key >= 5
        ]
        assert relation.aggregate("count(*), sum(id)::BIGINT").fetchone() == (
            len(expected),
            sum(row[0] for row in expected),
        )
        old = catalog.table(tables.namespace, name, snapshot_id=features[name].initial_snapshot_id)
        assert old.order("id").fetchall() == expected[:9]
        catalog.detach()
        with pytest.raises(ValueError, match="detached or replaced"):
            catalog.table(tables.namespace, name)


def test_public_ray_typed_and_empty_identity_partitions(partition_tables, ray_local):
    endpoint, tables, features = partition_tables
    with connection(tables) as con:
        vane.load_installed_extension("iceberg", connection=con)
        catalog = IcebergCatalog.attach(con, "lake", endpoint=endpoint)
        typed = catalog.table(tables.namespace, "partition_typed")
        assert typed.order("id").fetchall() == list(features["partition_typed"].rows)
        assert typed.filter("amount < 0 AND day < DATE '1970-01-01'").order("id").fetchall() == [
            row for row in features["partition_typed"].rows if row[2] is not None and row[2] < 0
        ]
        empty = catalog.table(tables.namespace, "partition_empty")
        assert empty.columns == ["id", "label", "region"]
        assert empty.types == [vane.sqltypes.BIGINT, vane.sqltypes.VARCHAR, vane.sqltypes.VARCHAR]
        assert empty.fetchall() == []
        assert empty.aggregate("count(*)").fetchone() == (0,)


def test_ray_partition_snapshot_survives_append_closed_rest_and_connection(partition_tables, ray_local):
    endpoint, shared, _features = partition_tables
    with iceberg_tables(endpoint, shared.directory.parent) as tables:
        features = populate_identity_partitions(tables)
        proxy = RestForwarder(endpoint)
        runner = None
        try:
            with connection(tables) as con:
                attach(con, proxy.endpoint)
                payload = freeze(con, f'SELECT id, label, region FROM lake."{tables.namespace}".partition_omitted')
            updated = append_partition_rows(
                tables,
                "partition_omitted",
                [
                    {"id": 800001, "label": "new", "region": "new-partition"},
                    {"id": 800002, "label": "new-null", "region": None},
                ],
                omit_partition_columns=True,
            )
            assert updated.scan().to_arrow().num_rows == 29
            proxy.stop()
            runner = RayRunner(address=None, max_task_backlog=None)
            with unavailable_metadata(tables):
                parts = list(runner.run_iter_tables(pickle.loads(payload)))
            observed = [
                row for table in parts for row in zip(*(column.to_pylist() for column in table.columns), strict=True)
            ]
            assert sorted(observed) == list(features["partition_omitted"].rows)
        finally:
            if runner is not None:
                runner.close()
            proxy.stop()


def test_two_workers_receive_disjoint_files_with_distinct_partition_constants(partition_tables, ray_local):
    endpoint, tables, features = partition_tables
    with (
        frozen_physical(
            endpoint, tables, table_name="partition_omitted", columns="id, label, region", file_count=9
        ) as (logical, plan, node, batches),
        native_workers(2) as (query_id, actors, workers),
    ):
        tasks = [
            NativeScanWorkerTask(
                query_id=query_id,
                node_id=node,
                fragment_node_id=str(index),
                plan=plan,
                split_batch=vane.ray_cxx.merge_scan_split_batches(batches[index::2]),
            )
            for index in range(2)
        ]
        register_scan_query(tasks)
        with unavailable_metadata(tables):
            attempts = [
                attempt for worker, task in zip(workers, tasks, strict=True) for attempt in worker.submit_tasks([task])
            ]
            workers[0].task_input_stream_exhausted_for_query(query_id, [node])
            observed, owners, _selected = drain_query(query_id, workers, attempts)
            assert owners == {worker.worker_id for worker in workers}
            assert sorted(observed) == list(features["partition_omitted"].rows)
        for actor, worker in zip(actors, workers, strict=True):
            assert ray.get(actor.stats_fragments.remote())["lookup_hits"] > 0
            assert ray.get(actor.stats_snapshot_databases.remote())["created_total"] == 1
            worker.fte_drop_query(query_id)
            assert ray.get(actor.stats_fragments.remote())["fragments_total"] == 0
            ray.get(actor.close_session.remote(str(logical.session_id())))
            assert ray.get(actor.stats_snapshot_databases.remote())["active_databases"] == 0
        assert query_id not in handles.fte_registry_stats()["queries"]


def test_partition_worker_loss_replays_original_constants_once(partition_tables, ray_local):
    endpoint, tables, features = partition_tables
    with (
        frozen_physical(
            endpoint, tables, table_name="partition_omitted", columns="id, label, region", file_count=9
        ) as (_logical, plan, node, batches),
        native_workers(2) as (query_id, actors, workers),
    ):
        task = NativeScanWorkerTask(
            query_id=query_id, node_id=node, plan=plan, split_batch=vane.ray_cxx.merge_scan_split_batches(batches)
        )
        register_scan_query([task])
        with unavailable_metadata(tables):
            submitted = workers[0].submit_tasks([task])
            assert submitted
            attempt = submitted[0]
            workers[0].pop_fte_result_handles(query_id)
            wait_for_open_scan(actors[0], attempt)
            ray.kill(actors[0], no_restart=True)
            with pytest.raises(Exception):
                await_result(attempt)
            workers[0].wait_fte_worker_failure_reconciliation(timeout_s=20)
            retries = workers[1].pop_fte_result_handles(query_id)
            workers[1].task_input_stream_exhausted_for_query(query_id, [node])
            observed, owners, selected = drain_query(query_id, workers, retries)
            assert owners == {workers[1].worker_id}
            assert sorted(observed) == list(features["partition_omitted"].rows)
            assert str(attempt.task_id) not in selected
            assert str(attempt.task_id).rsplit(".", 1)[0] + ".1" in selected
        assert workers[0].worker_id not in handles._FTE_WORKER_HANDLES


def test_cancel_partition_scan_releases_file_constants_and_session(partition_tables, ray_local):
    endpoint, tables, _features = partition_tables
    with (
        frozen_physical(endpoint, tables, table_name="partition_specs", columns="id, label, region", file_count=7) as (
            logical,
            plan,
            node,
            batches,
        ),
        native_workers(1) as (query_id, actors, workers),
    ):
        task = NativeScanWorkerTask(
            query_id=query_id, node_id=node, plan=plan, split_batch=vane.ray_cxx.merge_scan_split_batches(batches)
        )
        register_scan_query([task])
        attempt = workers[0].submit_tasks([task])[0]
        wait_for_open_scan(actors[0], attempt)
        attempt.cancel()
        assert ray.get(actors[0].fte_get_task_status.remote(attempt.task_id.to_dict()))["state"] == "CANCELED"
        workers[0].fte_drop_query(query_id)
        assert ray.get(actors[0].stats_fragments.remote())["queries_tracked"] == 0
        ray.get(actors[0].close_session.remote(str(logical.session_id())))
        assert ray.get(actors[0].stats_snapshot_databases.remote())["active_databases"] == 0
        assert query_id not in handles.fte_registry_stats()["queries"]
        assert not any(owner[0] == query_id for owner in handles._FTE_PARTITION_OWNERS)
