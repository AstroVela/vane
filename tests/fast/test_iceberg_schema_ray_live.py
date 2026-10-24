# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real Ray reads and lifecycle checks for frozen flat schema history."""

from __future__ import annotations

import pickle

import pytest
from gravitino_iceberg_helpers import iceberg_tables
from iceberg_ray_helpers import await_result, drain_query, frozen_physical, native_workers, wait_for_open_scan
from iceberg_scan_helpers import RestForwarder, attach, connection, freeze, unavailable_metadata
from iceberg_schema_helpers import (
    SCHEMA_TABLES,
    evolve_after_freeze,
    populate_schema_evolution,
    schema_evolution_tables,
    schema_filter,
    schema_reference,
)
from ray_native_scan_helpers import NativeScanWorkerTask, register_scan_query

import vane
from vane.catalog import IcebergCatalog
from vane.runners.ray import worker_handle as handles
from vane.runners.ray.runner import RayRunner

ray = pytest.importorskip("ray")
pytestmark = [pytest.mark.external_service, pytest.mark.real_ray]


@pytest.fixture(scope="module")
def schema_tables():
    with schema_evolution_tables() as setup:
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


@pytest.mark.parametrize("name", SCHEMA_TABLES)
def test_public_ray_schema_history_relations(schema_tables, ray_local, name):
    endpoint, tables, features = schema_tables
    feature = features[name]
    view = feature.current
    where, predicate = schema_filter(name)
    filtered = schema_reference(tables, name, row_filter=predicate)
    with connection(tables) as con:
        vane.load_installed_extension("iceberg", connection=con)
        catalog = IcebergCatalog.attach(con, 'Lake."schemas', endpoint=endpoint)
        relation = catalog.table(tables.namespace, name)
        assert relation._get_runner_type() == "ray"
        assert relation.columns == list(view.names)
        assert [str(value) for value in relation.types] == list(view.types)
        assert relation.order("id").fetchall() == list(view.rows)
        assert relation.filter(where).order("id").fetchall() == list(filtered.rows)
        if "value" in view.names:
            import pyarrow.compute as pc

            precise = schema_reference(
                tables, name, row_filter=(pc.field("value") > 16777219.8) & (pc.field("value") < 16777221.0)
            )
            assert relation.filter("value > 16777219.8 AND value < 16777221.0").project("id, value").order(
                "id"
            ).fetchall() == [
                (row[precise.names.index("id")], row[precise.names.index("value")]) for row in precise.rows
            ]
        assert relation.aggregate("count(*), sum(id)::BIGINT").fetchone() == (
            len(view.rows),
            sum(row[view.names.index("id")] for row in view.rows),
        )
        for previous in feature.history:
            old = catalog.table(tables.namespace, name, snapshot_id=previous.snapshot_id)
            assert old.columns == list(previous.names)
            assert [str(value) for value in old.types] == list(previous.types)
            assert old.order("id").fetchall() == list(previous.rows)
        catalog.detach()
        with pytest.raises(ValueError, match="detached or replaced"):
            catalog.table(tables.namespace, name)


def test_public_ray_evolved_empty_table(schema_tables, ray_local):
    endpoint, tables, features = schema_tables
    view = features["schema_empty"].current
    with connection(tables) as con:
        vane.load_installed_extension("iceberg", connection=con)
        catalog = IcebergCatalog.attach(con, "lake", endpoint=endpoint)
        relation = catalog.table(tables.namespace, "schema_empty")
        assert relation.columns == list(view.names)
        assert [str(value) for value in relation.types] == list(view.types)
        assert relation.fetchall() == []
        assert relation.aggregate("count(*)").fetchone() == (0,)


def test_ray_keeps_original_schema_after_independent_schema_change_and_append(schema_tables, ray_local):
    endpoint, shared, _features = schema_tables
    with iceberg_tables(endpoint, shared.directory.parent) as tables:
        feature = populate_schema_evolution(tables, names=("schema_columns",))["schema_columns"]
        proxy = RestForwarder(endpoint)
        runner = None
        try:
            with connection(tables) as con:
                attach(con, proxy.endpoint)
                payload = freeze(con, f'SELECT * FROM lake."{tables.namespace}".schema_columns')
            evolve_after_freeze(tables)
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


def test_two_workers_read_disjoint_files_with_different_physical_schemas(schema_tables, ray_local):
    endpoint, tables, features = schema_tables
    feature = features["schema_identity"]
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
            assert ray.get(actor.stats_fragments.remote())["lookup_hits"] > 0
            worker.fte_drop_query(query_id)
            assert ray.get(actor.stats_fragments.remote())["fragments_total"] == 0
            ray.get(actor.close_session.remote(str(logical.session_id())))
            assert ray.get(actor.stats_snapshot_databases.remote())["active_databases"] == 0
        assert query_id not in handles.fte_registry_stats()["queries"]


def test_worker_replacement_replays_original_field_ids_types_and_defaults(schema_tables, ray_local):
    endpoint, tables, features = schema_tables
    feature = features["schema_identity"]
    with (
        frozen_physical(endpoint, tables, table_name=feature.name, columns="*", file_count=feature.file_count) as (
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


def test_cancel_evolved_schema_scan_releases_query_and_session(schema_tables, ray_local):
    endpoint, tables, features = schema_tables
    feature = features["schema_identity"]
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
