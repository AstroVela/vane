# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared real-Ray file assignment, replay and cleanup qualification helpers."""

from __future__ import annotations

import asyncio
import json
import pickle
import time
import uuid
from contextlib import contextmanager

import ray
from iceberg_scan_helpers import attach, connection, freeze

from vane.runners.ray import worker_handle as handles
from vane.runners.ray.fragment_worker_results import fte_query_status
from vane.runners.ray.query_resource_runtime import release_query_resource_manager
from vane.runners.ray.worker import RayWorkerActor


@contextmanager
def frozen_physical(endpoint, tables, *, table_name="items", columns="id, label", file_count=9):
    with connection(tables) as con:
        attach(con, endpoint)
        payload = freeze(con, f'SELECT {columns} FROM lake."{tables.namespace}"."{table_name}"')
    with connection(tables) as worker:
        logical = pickle.loads(payload)
        physical = logical.to_physical_plan(worker)
        batch_map = physical.scan_split_batch_map()
        assert len(batch_map) == 1
        node, batches = next(iter(batch_map.items()))
        assert len(batches) == file_count
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
