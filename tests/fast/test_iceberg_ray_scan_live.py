# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real Ray scheduling, provider replay and recovery against owned Gravitino tables."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import pickle
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest
from gravitino_iceberg_helpers import iceberg_tables
from iceberg_scan_helpers import RestForwarder, attach, connection, freeze, unavailable_metadata
from ray_native_scan_helpers import NativeScanWorkerTask, register_scan_query

import vane
from vane.catalog import IcebergCatalog
from vane.runners.ray import worker_handle as handles
from vane.runners.ray.fragment_worker_results import fte_query_status
from vane.runners.ray.query_resource_runtime import release_query_resource_manager
from vane.runners.ray.runner import RayRunner
from vane.runners.ray.worker import RayWorkerActor

ray = pytest.importorskip("ray")
pytestmark = [pytest.mark.external_service, pytest.mark.real_ray]


@pytest.fixture(scope="module")
def ray_tables():
    endpoint = os.environ.get("VANE_TEST_GRAVITINO_ICEBERG_URL")
    shared = os.environ.get("VANE_TEST_GRAVITINO_ICEBERG_SHARED_DIR")
    if not endpoint and not shared:
        pytest.skip("set VANE_TEST_GRAVITINO_ICEBERG_URL and VANE_TEST_GRAVITINO_ICEBERG_SHARED_DIR")
    if not endpoint or not shared:
        pytest.fail("Ray Iceberg qualification requires both REST URL and shared directory")
    with iceberg_tables(endpoint.rstrip("/"), Path(shared), file_count=9) as tables:
        yield endpoint.rstrip("/"), tables


@pytest.fixture(autouse=True)
def ray_policy(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setenv("VANE_DISTRIBUTED_NODE_COUNT", "1")
    monkeypatch.setenv("VANE_DISTRIBUTED_WORKER_SLOTS", "2")
    monkeypatch.setenv("VANE_FTE_RETRY_INITIAL_DELAY_S", "0")
    monkeypatch.setenv("VANE_FTE_STATUS_WAIT_TIMEOUT_S", "5")
    monkeypatch.setenv("VANE_FTE_CONTROL_RPC_INITIAL_BACKOFF_S", "0")
    monkeypatch.setenv("VANE_FTE_SPLIT_QUEUE_SPACE_WAIT_TIMEOUT_S", "0.1")


def test_public_ray_catalog_relations(ray_tables, ray_local):
    endpoint, tables = ray_tables
    with connection(tables) as con:
        vane.load_installed_extension("iceberg", connection=con)
        catalog = IcebergCatalog.attach(con, 'Lake."quoted', endpoint=endpoint)
        relation = catalog.table(tables.namespace, "items")
        assert relation._get_runner_type() == "ray"
        assert relation.columns == ["id", "label"]
        assert relation.order("id").fetchall() == list(tables.rows)
        assert relation.filter("id BETWEEN 2050 AND 2053").project("id, label").order("id").fetchall() == list(
            tables.rows[2050:2054]
        )
        assert relation.aggregate("count(*), sum(id)::BIGINT").fetchone() == (
            len(tables.rows),
            sum(key for key, _ in tables.rows),
        )
        assert catalog.table(tables.namespace, "items", snapshot_id=tables.snapshot_id).aggregate(
            "count(*)"
        ).fetchone() == (len(tables.rows),)
        empty = catalog.table(tables.namespace, "empty")
        assert empty.columns == ["id", "label"]
        assert empty.types == [vane.sqltypes.BIGINT, vane.sqltypes.VARCHAR]
        assert empty.fetchall() == []
        catalog.detach()
        with pytest.raises(ValueError, match="detached or replaced"):
            catalog.table(tables.namespace, "items")


def test_ray_restored_snapshot_survives_append_closed_connection_and_rest(ray_tables, ray_local):
    import pyarrow as pa
    from pyiceberg.io.pyarrow import schema_to_pyarrow

    endpoint, shared_tables = ray_tables
    with iceberg_tables(endpoint, shared_tables.directory.parent, file_count=9) as tables:
        proxy = RestForwarder(endpoint)
        runner = None
        try:
            with connection(tables) as con:
                attach(con, proxy.endpoint)
                payload = freeze(con, f'SELECT id, label FROM lake."{tables.namespace}".items')
            table = tables.catalog.load_table((tables.namespace, "items"))
            table.append(
                pa.Table.from_pylist(
                    [{"id": 800001, "label": "committed-after-freeze"}],
                    schema=schema_to_pyarrow(table.schema(), include_field_ids=False),
                )
            )
            table.refresh()
            assert table.scan().to_arrow().num_rows == len(tables.rows) + 1
            proxy.stop()
            # Start the runner after closing the original connection. Worker
            # admission must replay the signed provider, never REST attachments.
            runner = RayRunner(address=None, max_task_backlog=None)
            with unavailable_metadata(tables):
                parts = list(runner.run_iter_tables(pickle.loads(payload)))
            observed = [
                row for part in parts for row in zip(*(part.column(i).to_pylist() for i in range(2)), strict=True)
            ]
            assert sorted(observed) == list(tables.rows)
        finally:
            if runner is not None:
                runner.close()
            proxy.stop()


@contextmanager
def frozen_physical(endpoint, tables):
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f'SELECT id, label FROM lake."{tables.namespace}".items')
    with connection(tables) as worker:
        logical = pickle.loads(payload)
        physical = logical.to_physical_plan(worker)
        batch_map = physical.scan_split_batch_map()
        assert len(batch_map) == 1
        node, batches = next(iter(batch_map.items()))
        assert len(batches) == 9
        try:
            yield logical, physical, str(node), [bytes(batch) for batch in batches]
        finally:
            physical = logical = None


@contextmanager
def native_workers(count):
    actors, workers = [], []
    query_id = "iceberg-ray-" + uuid.uuid4().hex
    try:
        for index in range(count):
            actor = RayWorkerActor.options(num_cpus=0).remote(1, 0, 1 << 30, 1 << 60)
            actors.append(actor)
            workers.append(
                handles.RayWorkerActorHandle(
                    actor,
                    memory_capacity_bytes=1 << 60,
                    worker_id=f"{query_id}-{index}",
                    node_id=ray.get_runtime_context().get_node_id(),
                )
            )
        yield query_id, actors, workers
    finally:
        cleanup_errors = []
        for worker in workers:
            try:
                worker.fte_drop_query(query_id)
            except Exception as error:
                cleanup_errors.append(error)
            finally:
                handles._FTE_WORKER_HANDLES.pop(worker.worker_id, None)
        try:
            release_query_resource_manager(query_id, reason="iceberg_qualification_complete")
        finally:
            for actor in actors:
                ray.kill(actor, no_restart=True)
        if cleanup_errors:
            raise RuntimeError(
                "Iceberg worker cleanup failed: " + "; ".join(map(str, cleanup_errors))
            ) from cleanup_errors[0]


def await_result(task_handle):
    return asyncio.run(asyncio.wait_for(task_handle.get_result(), timeout=30))


def task_rows(actor, task_handle):
    info = ray.get(actor.fte_get_task_info.remote(task_handle.task_id.to_dict()), timeout=10)
    raw = info["result"]
    if isinstance(raw, dict):
        raw = raw["result"]
    refs, _metadata, *_ = raw
    return [
        row for table in ray.get(refs) for row in zip(*(column.to_pylist() for column in table.columns), strict=True)
    ]


def drain_query(query_id, workers, initial_attempts=()):
    """Consume every scheduler partition and only its selected winning attempt."""
    attempts = {str(attempt.task_id): attempt for attempt in initial_attempts}
    rows_by_attempt = {}
    failed_attempts = set()
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        for attempt in workers[0].pop_fte_result_handles(query_id):
            attempts.setdefault(str(attempt.task_id), attempt)
        for key, attempt in list(attempts.items()):
            if key in rows_by_attempt or key in failed_attempts or not attempt.done():
                continue
            try:
                result = await_result(attempt)
            except Exception:
                # The scheduler owns recovery. A failed attempt is not a query
                # failure; only the committed selection may supply output.
                failed_attempts.add(key)
                continue
            assert result.ok
            rows_by_attempt[key] = task_rows(attempt.worker_handle.actor_handle, attempt)
            # Rows are now ordinary Python values. Release both remote result
            # storage and adopted output leases so pending partitions can run.
            attempt.release_result_payload()
            attempt.ack()
        status = fte_query_status(query_id)
        if status["failed"]:
            raise AssertionError("FTE query failed: " + json.dumps(status, indent=2, default=str))
        selected = set(status["selected_attempt_task_ids"])
        assert not selected.intersection(failed_attempts), status
        if status["finished"] and not status["registration_pending"] and selected <= rows_by_attempt.keys():
            return (
                [row for key in sorted(selected) for row in rows_by_attempt[key]],
                {attempts[key].worker_id for key in selected},
                selected,
            )
        time.sleep(0.01)
    raise AssertionError(f"FTE file assignments did not finish: {fte_query_status(query_id)}")


def wait_for_open_scan(actor, task_handle):
    task_handle.done()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        info = ray.get(actor.fte_get_task_info.remote(task_handle.task_id.to_dict()), timeout=5)
        status = info["status"]
        if status.get("state") == "RUNNING" and int(status.get("queued_split_count", 0)) == 0:
            return
        if status.get("state") in {"FAILED", "CANCELED", "FINISHED"}:
            raise AssertionError(f"scan ended before the controlled fault: {status}")
        time.sleep(0.05)
    raise AssertionError("scan did not consume its assignments while waiting for no-more-splits")


def test_two_real_workers_read_disjoint_file_assignments_and_release_state(ray_tables, ray_local):
    endpoint, tables = ray_tables
    with (
        frozen_physical(endpoint, tables) as (logical, plan, node, batches),
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
            assert sorted(observed) == list(tables.rows)
            for actor in actors:
                assert ray.get(actor.stats_fragments.remote())["lookup_hits"] > 0
                assert ray.get(actor.stats_snapshot_databases.remote())["created_total"] == 1
        for actor, worker in zip(actors, workers, strict=True):
            worker.fte_drop_query(query_id)
            assert ray.get(actor.stats_fragments.remote())["fragments_total"] == 0
            ray.get(actor.close_session.remote(str(logical.session_id())))
            assert ray.get(actor.stats_snapshot_databases.remote())["active_databases"] == 0
        assert query_id not in handles.fte_registry_stats()["queries"]


def test_worker_loss_replays_frozen_files_once_on_replacement(ray_tables, ray_local):
    endpoint, tables = ray_tables
    with (
        frozen_physical(endpoint, tables) as (_logical, plan, node, batches),
        native_workers(2) as (query_id, actors, workers),
    ):
        task = NativeScanWorkerTask(
            query_id=query_id,
            node_id=node,
            plan=plan,
            split_batch=vane.ray_cxx.merge_scan_split_batches(batches),
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
            assert sorted(observed) == list(tables.rows)
            assert owners == {workers[1].worker_id}
            assert str(attempt.task_id) not in selected
            assert str(attempt.task_id).rsplit(".", 1)[0] + ".1" in selected
        assert workers[0].worker_id not in handles._FTE_WORKER_HANDLES


def test_cancel_open_iceberg_scan_releases_tasks_fragments_and_session(ray_tables, ray_local):
    endpoint, tables = ray_tables
    with (
        frozen_physical(endpoint, tables) as (logical, plan, node, batches),
        native_workers(1) as (query_id, actors, workers),
    ):
        task = NativeScanWorkerTask(
            query_id=query_id,
            node_id=node,
            plan=plan,
            split_batch=vane.ray_cxx.merge_scan_split_batches(batches),
        )
        register_scan_query([task])
        attempt = workers[0].submit_tasks([task])[0]
        wait_for_open_scan(actors[0], attempt)
        attempt.cancel()
        status = ray.get(actors[0].fte_get_task_status.remote(attempt.task_id.to_dict()))
        assert status["state"] == "CANCELED"
        workers[0].fte_drop_query(query_id)
        assert ray.get(actors[0].stats_fragments.remote())["queries_tracked"] == 0
        ray.get(actors[0].close_session.remote(str(logical.session_id())))
        assert ray.get(actors[0].stats_snapshot_databases.remote())["active_databases"] == 0
        stats = handles.fte_registry_stats()
        assert query_id not in stats["queries"]
        assert not any(owner[0] == query_id for owner in handles._FTE_PARTITION_OWNERS)


def test_worker_admission_rejects_a_different_iceberg_provider_digest(ray_tables, ray_local):
    endpoint, tables = ray_tables
    with (
        frozen_physical(endpoint, tables) as (_logical, plan, _node, _batches),
        native_workers(1) as (query_id, actors, _workers),
    ):
        state = list(plan.__getstate__())
        snapshot = copy.deepcopy(state[6])
        specs = snapshot["dynamic_extensions"]
        iceberg = next(spec for spec in specs if spec["name"] == "iceberg")
        iceberg["sha256"] = "0" * 64
        state[6] = snapshot
        wrong = vane.ray_cxx.DistributedPhysicalPlan.__new__(vane.ray_cxx.DistributedPhysicalPlan)
        wrong.__setstate__(tuple(state))
        with pytest.raises(Exception, match="(?i)(provider|artifact|descriptor|digest)"):
            ray.get(
                actors[0].register_fragments.remote(
                    [
                        {
                            "query_id": query_id,
                            "fragment_id": query_id + ":node:1",
                            "plan": ray.put(wrong),
                        }
                    ]
                ),
                timeout=30,
            )
        assert ray.get(actors[0].stats_fragments.remote())["fragments_total"] == 0
        assert ray.get(actors[0].stats_snapshot_databases.remote())["active_databases"] == 0
