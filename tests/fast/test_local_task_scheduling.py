# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import threading
import time

import pytest

from vane.execution.resources import ResourceVector
from vane.execution.udf_admission import LocalExecutionCapacity, LocalExecutionSlotPool
from vane.execution.udf_lifecycle import ExecutionCancellationScope, ExecutionCancelledError
from vane.execution.udf_local_resources import local_task_capacity


def _pool(capacity, name, *, cpu=1, heap=0, slots=4):
    return LocalExecutionSlotPool(
        max_slots=slots,
        execution_slot_prefix=name,
        execution_capacity=capacity,
        resources=ResourceVector(cpu=cpu, heap_bytes=heap),
    )


def _take(authority):
    authority.request(0)
    assert authority.state()["state"] == "ready"
    return authority.take(0)


def test_fractional_cpu_heap_and_resident_actors_share_one_node_budget():
    capacity = LocalExecutionCapacity(max_slots=None, resource_limit=ResourceVector(cpu=2, heap_bytes=100))
    release_actor = capacity.reserve_resident(ResourceVector(cpu=0.5, heap_bytes=20))
    pool = _pool(capacity, "decode", cpu=0.5, heap=30)
    authorities = [pool.create_authority() for _ in range(3)]
    leases = [_take(authority) for authority in authorities[:2]]
    try:
        authorities[2].request(0)
        assert authorities[2].state()["state"] == "requested"
        assert capacity.resource_snapshot()["usage"] == ResourceVector(cpu=1.5, heap_bytes=80).to_dict()
        leases[0].complete_execution()
        leases.append(authorities[2].take(0))
        assert capacity.resource_snapshot()["usage"] == ResourceVector(cpu=1.5, heap_bytes=80).to_dict()
    finally:
        for lease in leases:
            lease.release()
        pool.close()
        release_actor()
        release_actor()
    assert capacity.resource_snapshot()["usage"] == ResourceVector().to_dict()
    assert capacity.reserved_slots == 0


def test_closing_an_unsubmitted_grant_returns_its_process_resources():
    capacity = LocalExecutionCapacity(max_slots=None, resource_limit=ResourceVector(cpu=1, heap_bytes=100))
    pool = _pool(capacity, "decode", heap=100)
    first, second = pool.create_authority(), pool.create_authority()
    first.request(0)
    second.request(0)
    assert second.state()["state"] == "requested"
    first.close()
    lease = second.take(0)
    lease.release()
    pool.close()
    assert capacity.resource_snapshot()["usage"] == ResourceVector().to_dict()


def test_transport_wait_yields_cpu_retains_heap_and_reacquires_before_user_execution():
    capacity = LocalExecutionCapacity(max_slots=None, resource_limit=ResourceVector(cpu=1, heap_bytes=100))
    source = _pool(capacity, "source", heap=20)
    downstream = _pool(capacity, "downstream", heap=70)
    source_lease = _take(source.create_authority())
    scope = ExecutionCancellationScope("source", 1)
    waiting, transport_ready, resumed = threading.Event(), threading.Event(), threading.Event()
    errors = []

    def run():
        try:
            with source_lease.suspend_for_wait(scope):
                waiting.set()
                assert transport_ready.wait(5)
            resumed.set()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    downstream_lease = None
    try:
        assert waiting.wait(2)
        assert capacity.resource_snapshot()["usage"] == ResourceVector(heap_bytes=20).to_dict()
        downstream_lease = _take(downstream.create_authority())
        assert capacity.resource_snapshot()["usage"] == ResourceVector(cpu=1, heap_bytes=90).to_dict()
        transport_ready.set()
        assert not resumed.wait(0.1)
        downstream_lease.release()
        assert resumed.wait(2)
        assert capacity.resource_snapshot()["usage"] == ResourceVector(cpu=1, heap_bytes=20).to_dict()
    finally:
        transport_ready.set()
        if downstream_lease is not None:
            downstream_lease.release()
        thread.join(5)
        source_lease.release()
        source.close()
        downstream.close()
    assert not thread.is_alive() and errors == []
    assert capacity.resource_snapshot()["usage"] == ResourceVector().to_dict()


def test_failed_transport_releases_its_owner_without_reacquiring_cpu():
    capacity = LocalExecutionCapacity(max_slots=None, resource_limit=ResourceVector(cpu=1, heap_bytes=100))
    pool = _pool(capacity, "source", heap=20)
    first = _take(pool.create_authority())
    second = None
    try:
        with pytest.raises(ValueError, match="transport failed"):
            with first.suspend_for_wait(ExecutionCancellationScope("failed transport", 1)):
                second = _take(pool.create_authority())
                raise ValueError("transport failed")
        first.release()
        assert capacity.resource_snapshot()["usage"] == ResourceVector(cpu=1, heap_bytes=20).to_dict()
    finally:
        first.release()
        if second is not None:
            second.release()
        pool.close()
    assert capacity.resource_snapshot()["usage"] == ResourceVector().to_dict()


def test_task_pool_capacity_uses_declared_resources():
    limit = ResourceVector(cpu=8, heap_bytes=1000)
    assert local_task_capacity(ResourceVector(cpu=0.5), limit) == 16
    assert local_task_capacity(ResourceVector(cpu=2), limit) == 4
    assert local_task_capacity(ResourceVector(cpu=1, heap_bytes=400), limit) == 2
    with pytest.raises(ValueError, match="exceeds"):
        local_task_capacity(ResourceVector(cpu=9), limit)


def test_cancelled_cpu_resumption_does_not_take_another_tasks_resources():
    capacity = LocalExecutionCapacity(max_slots=None, resource_limit=ResourceVector(cpu=1, heap_bytes=100))
    pool = _pool(capacity, "source", heap=20)
    first = _take(pool.create_authority())
    scope = ExecutionCancellationScope("resume", 1)
    suspended, resume = threading.Event(), threading.Event()
    errors = []

    def run():
        try:
            with first.suspend_for_wait(scope):
                suspended.set()
                assert resume.wait(5)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    second = None
    try:
        assert suspended.wait(2)
        second = _take(pool.create_authority())
        resume.set()
        deadline = time.monotonic() + 2
        while not capacity.resource_snapshot()["resuming_tasks"] and time.monotonic() < deadline:
            time.sleep(0.005)
        assert capacity.resource_snapshot()["resuming_tasks"] == 1
        scope.cancel("test cancellation")
        thread.join(2)
        assert not thread.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], ExecutionCancelledError)
        first.release()
        assert capacity.resource_snapshot()["usage"] == ResourceVector(cpu=1, heap_bytes=20).to_dict()
        assert capacity.resource_snapshot()["resuming_tasks"] == 0
    finally:
        resume.set()
        scope.cancel()
        thread.join(5)
        first.release()
        if second is not None:
            second.release()
        pool.close()


def test_resident_startup_cannot_claim_cpu_yielded_by_a_live_task():
    capacity = LocalExecutionCapacity(max_slots=None, resource_limit=ResourceVector(cpu=1, heap_bytes=100))
    pool = _pool(capacity, "source", heap=20)
    lease = _take(pool.create_authority())
    started, acquired = threading.Event(), threading.Event()
    scope = ExecutionCancellationScope("actor startup", 1)
    releases, errors = [], []

    def reserve():
        started.set()
        try:
            releases.append(capacity.reserve_resident(ResourceVector(cpu=1, heap_bytes=80), scope))
            acquired.set()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=reserve, daemon=True)
    try:
        with lease.suspend_for_wait(ExecutionCancellationScope("source", 1)):
            thread.start()
            assert started.wait(2)
            assert not acquired.wait(0.1)
        assert not acquired.is_set()
        lease.release()
        assert acquired.wait(2)
        thread.join(2)
        assert errors == []
        assert capacity.resource_snapshot()["resident"] == ResourceVector(cpu=1, heap_bytes=80).to_dict()
        with pytest.raises(ValueError, match="exceed"):
            capacity.reserve_resident(ResourceVector(cpu=0.5))
    finally:
        scope.cancel()
        lease.release()
        if thread.ident is not None:
            thread.join(5)
        for release in releases:
            release()
        pool.close()
    assert capacity.resource_snapshot()["usage"] == ResourceVector().to_dict()


@pytest.mark.timeout(60)
@pytest.mark.parametrize("downstream", [None, "subprocess_task", "subprocess_actor"])
def test_one_row_group_runs_multiple_batches_with_one_native_thread(tmp_path, monkeypatch, downstream):
    import pyarrow as pa
    import pyarrow.parquet as pq

    import vane
    import vane.execution.udf_subprocess as subprocess_exec

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    subprocess_exec._shutdown_global_task_runtime()
    runtime = subprocess_exec._GlobalSubprocessTaskRuntime(resource_limit=ResourceVector(cpu=3, heap_bytes=1024**3))
    monkeypatch.setattr(subprocess_exec, "_GLOBAL_TASK_RUNTIME", runtime)
    source_path = tmp_path / "one-row-group.parquet"
    pq.write_table(pa.table({"x": list(range(8192))}), source_path, row_group_size=8192)
    assert pq.ParquetFile(source_path).num_row_groups == 1
    rendezvous = str(tmp_path / "workers")

    def decode(table):
        import os
        import time
        from pathlib import Path

        directory = Path(rendezvous)
        directory.mkdir(exist_ok=True)
        (directory / str(os.getpid())).touch()
        deadline = time.monotonic() + 10
        while len(list(directory.iterdir())) < 2:
            if time.monotonic() > deadline:
                raise RuntimeError("single-row-group UDF did not admit a second batch worker")
            time.sleep(0.01)
        return pa.table({"x": table.column("x"), "pid": [os.getpid()] * table.num_rows})

    def identity(table):
        return table

    class Actor:
        def __call__(self, table):
            return table

    connection = vane.connect(config={"threads": 1})
    schema = {"x": vane.sqltypes.BIGINT, "pid": vane.sqltypes.BIGINT}
    try:
        relation = connection.read_parquet(str(source_path)).map_batches(
            decode,
            schema=schema,
            execution_backend="subprocess_task",
            batch_size=256,
            memory_bytes=64 * 1024**2,
        )
        if downstream is not None:
            options = {"actor_number": 1} if downstream == "subprocess_actor" else {}
            relation = relation.map_batches(
                Actor if downstream == "subprocess_actor" else identity,
                schema=schema,
                execution_backend=downstream,
                batch_size=256,
                **options,
            )
        rows = relation.fetchall()
        assert sorted(row[0] for row in rows) == list(range(8192))
        assert len({row[1] for row in rows}) >= 2
    finally:
        connection.close()
        runtime.close(kill=True)
    assert runtime.execution_capacity.resource_snapshot()["usage"] == ResourceVector().to_dict()
