# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Database reuse must preserve per-query state and cleanup ownership."""

from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import vane
from vane._native import execution_runtime as native
from vane.execution.request_admission import RequestAdmissionLimits
from vane.execution.worker_database import WorkerDatabasePool


@pytest.fixture
def pool():
    value = WorkerDatabasePool(vane.RayResources(max_active_queries=1, cpus_per_worker=1))
    yield value
    assert not value.leases
    for anchor in value.idle:
        anchor.close()


def test_reuse_creates_an_independent_session_and_clears_query_state(pool):
    first = pool.acquire("first", reusable=True)
    with first.open() as connection:
        anchor = first.anchor
        timezone = connection.execute("SELECT current_setting('TimeZone')").fetchone()
        connection.execute("SET integer_division=true")
        connection.execute("SET TimeZone='Asia/Shanghai'")
        connection.execute("CREATE TEMP TABLE private_state AS SELECT 42 AS value")
        runtime = connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 0))
        connection.interrupt()
    first.release()
    second = pool.acquire("second", reusable=True)
    try:
        assert second.anchor is anchor
        with second.open() as connection:
            other_runtime = connection.configure_local_runtime(request_limit=RequestAdmissionLimits(1, 0))
            assert other_runtime._runtime._session_id != runtime._runtime._session_id
            assert connection.execute("SELECT 3/2").fetchone() == (1.5,)
            assert connection.execute("SELECT current_setting('TimeZone')").fetchone() == timezone
            with pytest.raises(vane.CatalogException, match="private_state"):
                connection.execute("SELECT * FROM private_state")
            assert connection.execute("SELECT 7").fetchone() == (7,)
    finally:
        second.release()
    assert pool.snapshot()["created_instances"] == 1
    assert pool.snapshot()["reused_instances"] == 1


@pytest.mark.parametrize("borrower", ["connection", "service"])
def test_cleanup_waits_for_native_borrowers_and_release_is_idempotent(pool, borrower):
    lease = pool.acquire("first", reusable=True)
    with lease.open() as connection:
        extra = lease.anchor.cursor() if borrower == "connection" else native.TaskService(connection)
    with pytest.raises(vane.InvalidInputException, match="cleanup is pending"):
        lease.release()
    assert pool.snapshot()["owners"] == ["first"]
    assert pool.snapshot()["idle_instances"] == 0
    with pytest.raises(RuntimeError, match="capacity"):
        pool.acquire("second", reusable=False)
    if borrower == "connection":
        extra.close()
    else:
        extra.release()
        assert extra.diagnostics() == []  # Diagnostics may outlive the database owner.
    lease.release()
    assert pool.snapshot()["idle_instances"] == 0  # A failed reset retires this instance on retry.
    successor = pool.acquire("second", reusable=True)
    try:
        lease.release()
        assert pool.snapshot()["owners"] == ["second"]
        with pytest.raises(RuntimeError, match="no longer accepts"):
            lease.open()
        with successor.open() as connection:
            if borrower == "service":
                extra.cancel("late cancellation from the retired query")
            assert connection.execute("SELECT 13").fetchone() == (13,)
    finally:
        successor.release()


def test_reuse_evicts_file_data_before_reading_an_overwritten_source(pool, tmp_path):
    source = tmp_path / "values.parquet"
    pq.write_table(pa.table({"value": [11, 12, 13]}), source)
    sql = f"SELECT value FROM read_parquet('{source}') ORDER BY value"
    first = pool.acquire("first", reusable=True)
    with first.open() as connection:
        anchor = first.anchor
        assert connection.execute(sql).fetchall() == [(11,), (12,), (13,)]
    first.release()
    pq.write_table(pa.table({"value": [21, 22, 23]}), source)
    second = pool.acquire("second", reusable=True)
    try:
        assert second.anchor is anchor
        with second.open() as connection:
            assert connection.execute("SELECT count(*) FROM duckdb_external_file_cache()").fetchone() == (0,)
            assert connection.execute(sql).fetchall() == [(21,), (22,), (23,)]
    finally:
        second.release()


def test_instance_with_retained_buffers_is_retired(pool):
    lease = pool.acquire("first", reusable=True)
    with lease.open() as connection:
        anchor = lease.anchor
        connection.execute("CREATE TABLE retained AS SELECT range FROM range(10000)")
    lease.release()
    assert pool.snapshot()["idle_instances"] == 0
    with pytest.raises(vane.ConnectionException):
        anchor.execute("SELECT 1")
    next_lease = pool.acquire("second", reusable=True)
    try:
        with next_lease.open() as connection:
            with pytest.raises(vane.CatalogException, match="retained"):
                connection.execute("SELECT * FROM retained")
    finally:
        next_lease.release()


def test_retired_database_does_not_invalidate_exported_batch(pool):
    from tests.fast.test_direct_exchange import next_batch, submission
    from vane.execution.direct_exchange import DirectExchangeLimits, InProcessTaskService

    lease = pool.acquire("exported", reusable=False)
    with lease.open() as connection:
        anchor = lease.anchor
        spec = submission(connection, "SELECT repeat('payload-', 3) || range::varchar AS value FROM range(1000)")
        with InProcessTaskService(connection, spec, DirectExchangeLimits(1024, 256, 3, 1)) as service:
            service.start()
            status, batch = next_batch(service)
            assert status == "data"
            expected = batch.to_rows()
            arrow = batch.to_arrow(["value"])
    lease.release()
    assert pool.snapshot()["idle_instances"] == 0
    with pytest.raises(vane.ConnectionException):
        anchor.execute("SELECT 1")
    assert batch.to_rows() == expected
    batch.close()
    assert arrow.column(0).to_pylist() == [row[0] for row in expected]


def test_pipelined_and_fte_share_slots_and_fte_replaces_an_idle_instance():
    pool = WorkerDatabasePool(vane.RayResources(max_active_queries=2, cpus_per_worker=1))
    warm = [pool.acquire(str(i), reusable=True) for i in range(2)]
    for lease in warm:
        lease.open().close()
    for lease in warm:
        lease.release()
    pipeline = pool.acquire("pipeline", reusable=True)
    attempt = pool.acquire("fte", reusable=False)
    previous = attempt.anchor
    try:
        with pipeline.open() as left, attempt.open() as right:
            with pytest.raises(vane.ConnectionException):
                previous.execute("SELECT 1")
            assert left.execute("SELECT 3").fetchone() == (3,)
            assert right.execute("SELECT 5").fetchone() == (5,)
            with pytest.raises(RuntimeError, match="capacity"):
                pool.acquire("extra", reusable=True)
            assert pool.snapshot()["idle_instances"] == 0
    finally:
        pipeline.release()
        attempt.release()
    state = pool.snapshot()
    assert state["owners"] == []
    assert state["idle_instances"] == 1
    assert state["created_instances"] == 3
    for anchor in pool.idle:
        anchor.close()


def test_slot_bound_matches_integer_operator_admission():
    resources = replace(vane.RayResources(), max_active_queries=4, operator_memory_bytes=5)
    pool = WorkerDatabasePool(resources)
    leases = [pool.acquire(str(i), reusable=True) for i in range(5)]
    assert pool.snapshot()["capacity"] == 5
    assert pool.snapshot()["operator_bytes_per_slot"] == 1
    with pytest.raises(RuntimeError, match="capacity"):
        pool.acquire("overflow", reusable=False)
    for lease in leases:
        lease.release()
