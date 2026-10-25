# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real Ray reads, retry and cleanup with frozen position/equality deletes."""

from __future__ import annotations

import pickle

import pytest
from gravitino_iceberg_helpers import iceberg_tables
from iceberg_delete_helpers import DELETE_TABLES, commit_deletes, delete_tables, populate_delete_tables
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
def frozen_deletes():
    with delete_tables() as setup:
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


@pytest.mark.parametrize("name", DELETE_TABLES)
def test_public_ray_delete_relations_and_old_snapshots(frozen_deletes, ray_local, name):
    endpoint, tables, features = frozen_deletes
    feature = features[name]
    view = feature.current
    id_index, label_index = view.names.index("id"), view.names.index("label")
    with connection(tables) as con:
        vane.load_installed_extension("iceberg", connection=con)
        catalog = IcebergCatalog.attach(con, "lake", endpoint=endpoint)
        relation = catalog.table(tables.namespace, name)
        assert relation._get_runner_type() == "ray"
        assert relation.columns == list(view.names)
        assert relation.order("id").fetchall() == list(view.rows)
        assert relation.project("id").order("id").fetchall() == [(row[id_index],) for row in view.rows]
        assert relation.filter("id % 4 = 1 OR label IS NULL").order("id").fetchall() == [
            row for row in view.rows if row[id_index] % 4 == 1 or row[label_index] is None
        ]
        assert relation.aggregate("count(*), sum(id)::BIGINT").fetchone() == (
            len(view.rows),
            sum(row[id_index] for row in view.rows) if view.rows else None,
        )
        for previous in feature.history:
            old = catalog.table(tables.namespace, name, snapshot_id=previous.snapshot_id)
            assert old.columns == list(previous.names)
            assert old.order("id").fetchall() == list(previous.rows)
        catalog.detach()


def test_two_workers_keep_disjoint_files_and_partition_specific_deletes(frozen_deletes, ray_local):
    endpoint, tables, features = frozen_deletes
    feature = features["delete_specs"]
    with (
        frozen_physical(endpoint, tables, table_name=feature.name, columns="*", file_count=feature.file_count) as (
            logical,
            plan,
            node,
            batches,
        ),
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
            rows, owners, _selected = drain_query(query_id, workers, attempts)
            assert owners == {worker.worker_id for worker in workers}
            assert sorted(rows, key=repr) == sorted(feature.current.rows, key=repr)
        for actor, worker in zip(actors, workers, strict=True):
            worker.fte_drop_query(query_id)
            assert ray.get(actor.stats_fragments.remote())["fragments_total"] == 0
            ray.get(actor.close_session.remote(str(logical.session_id())))
            assert ray.get(actor.stats_snapshot_databases.remote())["active_databases"] == 0
        assert query_id not in handles.fte_registry_stats()["queries"]


@pytest.mark.parametrize("name", ["delete_positions", "delete_equality_multi"])
def test_worker_replacement_replays_the_original_delete_set(frozen_deletes, ray_local, name):
    endpoint, tables, features = frozen_deletes
    feature = features[name]
    with (
        frozen_physical(endpoint, tables, table_name=name, columns="*", file_count=feature.file_count) as (
            _logical,
            plan,
            node,
            batches,
        ),
        native_workers(2) as (query_id, actors, workers),
    ):
        task = NativeScanWorkerTask(
            query_id=query_id, node_id=node, plan=plan, split_batch=vane.ray_cxx.merge_scan_split_batches(batches)
        )
        register_scan_query([task])
        with unavailable_metadata(tables):
            attempt = workers[0].submit_tasks([task])[0]
            workers[0].pop_fte_result_handles(query_id)
            wait_for_open_scan(actors[0], attempt)
            ray.kill(actors[0], no_restart=True)
            with pytest.raises(Exception):
                await_result(attempt)
            workers[0].wait_fte_worker_failure_reconciliation(timeout_s=20)
            retries = workers[1].pop_fte_result_handles(query_id)
            workers[1].task_input_stream_exhausted_for_query(query_id, [node])
            rows, owners, selected = drain_query(query_id, workers, retries)
            assert owners == {workers[1].worker_id}
            assert sorted(rows, key=repr) == sorted(feature.current.rows, key=repr)
            assert str(attempt.task_id) not in selected
            assert str(attempt.task_id).rsplit(".", 1)[0] + ".1" in selected
        assert workers[0].worker_id not in handles._FTE_WORKER_HANDLES


def test_later_delete_commit_does_not_change_detached_ray_replay(frozen_deletes, ray_local):
    endpoint, shared, _features = frozen_deletes
    with iceberg_tables(endpoint, shared.directory.parent) as tables:
        feature = populate_delete_tables(tables, names=("delete_equality_multi",))["delete_equality_multi"]
        proxy = RestForwarder(endpoint)
        runner = None
        try:
            with connection(tables) as con:
                attach(con, proxy.endpoint)
                payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".delete_equality_multi')
            commit_deletes(tables, feature.name, [{1: key} for key in range(27)], equality_ids=(1,))
            proxy.stop()
            runner = RayRunner(address=None, max_task_backlog=None)
            with unavailable_metadata(tables):
                parts = list(runner.run_iter_tables(pickle.loads(payload)))
            rows = [
                row for table in parts for row in zip(*(column.to_pylist() for column in table.columns), strict=True)
            ]
            assert sorted(rows, key=repr) == sorted(feature.current.rows, key=repr)
        finally:
            if runner is not None:
                runner.close()
            proxy.stop()


def test_cancel_delete_scan_releases_query_and_session(frozen_deletes, ray_local):
    endpoint, tables, features = frozen_deletes
    feature = features["delete_equality_multi"]
    with (
        frozen_physical(endpoint, tables, table_name=feature.name, columns="*", file_count=feature.file_count) as (
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
