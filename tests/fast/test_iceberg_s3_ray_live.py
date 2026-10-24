# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real single-node Ray S3 reads, detached retries and resource ownership."""

from __future__ import annotations

import pickle

import pytest
from iceberg_delete_helpers import commit_deletes, manifest_entries
from iceberg_ray_helpers import await_result, drain_query, native_workers, wait_for_open_scan
from iceberg_s3_helpers import S3_TABLES, ObjectForwarder, attach, connection, frozen_physical, s3_tables
from iceberg_scan_helpers import RestForwarder, freeze
from ray_native_scan_helpers import NativeScanWorkerTask, register_scan_query

import vane
from vane.runners.ray import worker_handle as handles
from vane.runners.ray.runner import RayRunner

ray = pytest.importorskip("ray")
pytestmark = [pytest.mark.external_service, pytest.mark.real_ray]


@pytest.fixture(scope="module")
def remote_tables(tmp_path_factory):
    with s3_tables(tmp_path_factory.mktemp("iceberg-s3-ray")) as setup:
        yield setup
        tables, _features = setup
        events = tables.objects.events()
        assert events and any(event["range"] for event in events)
        assert not any(event["credentials"] or event["query"] for event in events)


@pytest.fixture(autouse=True)
def ray_policy(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setenv("VANE_DISTRIBUTED_NODE_COUNT", "1")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "2")
    monkeypatch.setenv("VANE_FTE_RETRY_INITIAL_DELAY_S", "0")
    monkeypatch.setenv("VANE_FTE_STATUS_WAIT_TIMEOUT_S", "5")
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_INITIAL_BACKOFF_S", "0")
    monkeypatch.setenv("VANE_FTE_SPLIT_QUEUE_SPACE_WAIT_TIMEOUT_S", "0.1")


@pytest.mark.parametrize("name", S3_TABLES)
def test_public_ray_s3_rows_projection_filters_counts_and_history(remote_tables, ray_local, name):
    tables, features = remote_tables
    feature = features[name]
    view = feature.current
    label = "renamed_label" if "renamed_label" in view.names else "label"
    label_index = view.names.index(label)
    with connection(tables) as con:
        catalog = attach(con, tables.endpoint)
        relation = catalog.table(tables.namespace, name)
        assert relation._get_runner_type() == "ray"
        assert relation.columns == list(view.names)
        assert relation.order("id").fetchall() == list(view.rows)
        assert relation.project("id").order("id").fetchall() == [(row[0],) for row in view.rows]
        assert relation.filter(f"id % 4 = 1 OR {label} IS NULL").order("id").fetchall() == [
            row for row in view.rows if row[0] % 4 == 1 or row[label_index] is None
        ]
        assert relation.aggregate("count(*), sum(id)::BIGINT").fetchone() == (
            len(view.rows),
            sum(row[0] for row in view.rows) if view.rows else None,
        )
        for previous in feature.history:
            old = catalog.table(tables.namespace, name, snapshot_id=previous.snapshot_id)
            assert old.columns == list(previous.names)
            assert old.order("id").fetchall() == list(previous.rows)


@pytest.mark.parametrize("name", ["identity", "positions", "equality"])
def test_two_ray_workers_read_disjoint_s3_assignments_without_metadata(remote_tables, ray_local, name):
    tables, features = remote_tables
    with (
        frozen_physical(tables, name) as (logical, plan, node, batches, _payload),
        native_workers(2) as (query_id, actors, workers),
    ):
        assert len(batches) == features[name].file_count == 9
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
        tables.objects.policy(deny_metadata=True)
        start = len(tables.objects.events())
        try:
            attempts = [
                attempt for worker, task in zip(workers, tasks, strict=True) for attempt in worker.submit_tasks([task])
            ]
            workers[0].task_input_stream_exhausted_for_query(query_id, [node])
            rows, owners, _selected = drain_query(query_id, workers, attempts)
            assert owners == {worker.worker_id for worker in workers}
            assert sorted(rows, key=repr) == sorted(features[name].current.rows, key=repr)
        finally:
            tables.objects.policy()
        assert not any("/metadata/" in event["path"] for event in tables.objects.events()[start:])
        for actor, worker in zip(actors, workers, strict=True):
            worker.fte_drop_query(query_id)
            assert ray.get(actor.stats_fragments.remote())["fragments_total"] == 0
            ray.get(actor.close_session.remote(str(logical.session_id())))
            assert ray.get(actor.stats_snapshot_databases.remote())["active_databases"] == 0
        assert query_id not in handles.fte_registry_stats()["queries"]


@pytest.mark.parametrize("name", ["positions", "equality"])
def test_replacement_worker_replays_s3_data_and_original_deletes(remote_tables, ray_local, name):
    tables, features = remote_tables
    with (
        frozen_physical(tables, name) as (_logical, plan, node, batches, _payload),
        native_workers(2) as (query_id, actors, workers),
    ):
        task = NativeScanWorkerTask(
            query_id=query_id, node_id=node, plan=plan, split_batch=vane.ray_cxx.merge_scan_split_batches(batches)
        )
        register_scan_query([task])
        tables.objects.policy(deny_metadata=True)
        try:
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
            assert sorted(rows, key=repr) == sorted(features[name].current.rows, key=repr)
            assert str(attempt.task_id) not in selected
            assert str(attempt.task_id).rsplit(".", 1)[0] + ".1" in selected
        finally:
            tables.objects.policy()
        assert workers[0].worker_id not in handles._FTE_WORKER_HANDLES


def test_ray_replays_old_s3_snapshot_after_later_delete_and_rest_shutdown(tmp_path, ray_local):
    with s3_tables(tmp_path, names=("equality",)) as (tables, features):
        proxy = RestForwarder(tables.endpoint)
        runner = None
        try:
            with connection(tables) as con:
                attach(con, proxy.endpoint)
                payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".equality')
            commit_deletes(tables, "equality", [{1: key} for key in range(27)], equality_ids=(1,))
            proxy.stop()
            runner = RayRunner(address=None, max_task_backlog=None)
            tables.objects.policy(deny_metadata=True)
            parts = list(runner.run_iter_tables(pickle.loads(payload)))
            rows = [
                row for table in parts for row in zip(*(column.to_pylist() for column in table.columns), strict=True)
            ]
            assert sorted(rows, key=repr) == sorted(features["equality"].current.rows, key=repr)
        finally:
            tables.objects.policy()
            if runner is not None:
                runner.close()
            proxy.stop()


@pytest.mark.parametrize("content", [0, 2], ids=["data", "equality-delete"])
def test_ray_missing_s3_object_fails_without_returning_incomplete_rows(remote_tables, ray_local, content):
    tables, _features = remote_tables
    with connection(tables) as con:
        attach(con, tables.endpoint)
        payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".equality')
    table = tables.catalog.load_table((tables.namespace, "equality"))
    path = next(
        entry.data_file.file_path for _manifest, entry in manifest_entries(table) if entry.data_file.content == content
    )
    denied = "/" + path.removeprefix("s3://")
    runner = RayRunner(address=None, max_task_backlog=None)
    tables.objects.policy(deny_metadata=True, denied=[denied])
    try:
        with pytest.raises(Exception, match="(?i)(404|file|failed|open)"):
            list(runner.run_iter_tables(pickle.loads(payload)))
    finally:
        runner.close()
        tables.objects.policy()
    assert any(event["path"] == denied and event["status"] == 404 for event in tables.objects.events())


def test_cancel_remote_scan_releases_query_and_session(remote_tables, ray_local):
    tables, _features = remote_tables
    with (
        frozen_physical(tables, "equality") as (logical, plan, node, batches, _payload),
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


def test_same_session_replays_each_querys_s3_endpoint(remote_tables, ray_local, tmp_path):
    from urllib.parse import urlsplit

    tables, features = remote_tables
    second = ObjectForwarder(tables.config["s3_endpoint"], tmp_path)
    proxy = RestForwarder(tables.endpoint)
    runner = None
    try:
        with connection(tables) as con:
            attach(con, proxy.endpoint)
            sql = f'SELECT * FROM lake."{tables.namespace}".equality'
            first_payload = freeze(con, sql)
            with con.cursor() as cursor:
                cursor.execute("SET s3_endpoint=$endpoint", {"endpoint": urlsplit(second.endpoint).netloc})
                second_payload = freeze(cursor, sql)
        assert pickle.loads(first_payload).session_id() == pickle.loads(second_payload).session_id()
        proxy.stop()
        tables.objects.policy(deny_metadata=True)
        second.policy(deny_metadata=True)
        runner = RayRunner(address=None, max_task_backlog=None)
        for payload, selected, other in (
            (first_payload, tables.objects, second),
            (second_payload, second, tables.objects),
            (first_payload, tables.objects, second),
        ):
            before_selected, before_other = len(selected.events()), len(other.events())
            parts = list(runner.run_iter_tables(pickle.loads(payload)))
            rows = [
                row for table in parts for row in zip(*(column.to_pylist() for column in table.columns), strict=True)
            ]
            assert sorted(rows, key=repr) == sorted(features["equality"].current.rows, key=repr)
            assert len(selected.events()) > before_selected
            assert len(other.events()) == before_other
        assert not any(event["credentials"] or event["query"] for event in second.events())
    finally:
        if runner is not None:
            runner.close()
        tables.objects.policy()
        proxy.stop()
        second.stop()
