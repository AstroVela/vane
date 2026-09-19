# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest

from vane.execution.data_lifecycle import OutputBlockLeaseOwner
from vane.execution.udf_data_lease import DataAllocation, RuntimeDataLedger, current_data_task


@pytest.fixture
def data_scope():
    ledger = RuntimeDataLedger()
    query = ledger.open_query()
    task = query.open_task()
    yield ledger, query, task
    task.finish()
    query.shutdown()
    ledger.close()


@pytest.mark.parametrize(
    "provider,identity,size",
    [("", "a", 1), (1, "a", 1), ("shm", "", 1), ("shm", 1, 1), ("shm", "a", -1), ("shm", "a", True), ("shm", "a", 1.0)],
)
def test_data_allocation_requires_an_identity_and_exact_nonnegative_size(provider, identity, size):
    with pytest.raises(ValueError):
        DataAllocation(provider, identity, size)


def test_shared_inputs_and_outputs_count_one_allocation_across_queries(data_scope):
    ledger, first_query, first = data_scope
    second_query = ledger.open_query()
    second = second_query.open_task()
    allocation = DataAllocation("local_shm", "shared", 4096)
    first.hold_inputs([allocation, allocation])
    second.hold_inputs([allocation])
    output = first.own_output(allocation)
    view = output.fork()
    snapshot = ledger.snapshot()
    assert snapshot["retained_bytes"] == snapshot["input_bytes"] == snapshot["output_bytes"] == 4096
    assert snapshot["allocations"] == 1
    assert snapshot["leases"] == 4
    assert snapshot["output_state_bytes"]["external_consumer"] == 4096
    first.finish()
    first_query.shutdown()
    output.release()
    assert ledger.snapshot()["retained_bytes"] == 4096
    assert ledger.snapshot()["queries"] == 1
    second.finish()
    second_query.shutdown()
    ledger.close()
    assert ledger.snapshot()["input_bytes"] == 0
    assert ledger.snapshot()["retained_bytes"] == 4096
    view.release()
    assert ledger.snapshot()["leases"] == ledger.snapshot()["retained_bytes"] == 0


def test_input_batch_validation_does_not_publish_partial_accounting(data_scope):
    ledger, _, task = data_scope
    known = DataAllocation("local_shm", "known", 32)
    task.hold_inputs([known])
    before = ledger.snapshot()
    with pytest.raises(ValueError, match="different size"):
        task.hold_inputs([DataAllocation("local_shm", "new", 64), DataAllocation("local_shm", "known", 31)])
    assert ledger.snapshot() == before
    with pytest.raises(ValueError, match="different size"):
        task.hold_inputs([DataAllocation("local_shm", "new", 64), DataAllocation("local_shm", "new", 63)])
    assert ledger.snapshot() == before


def test_allocation_identity_includes_provider_and_can_be_reused_after_release(data_scope):
    ledger, _, task = data_scope
    first = task.own_output(DataAllocation("provider-a", "same", 32))
    second = task.own_output(DataAllocation("provider-b", "same", 64))
    assert ledger.snapshot()["retained_bytes"] == 96
    with pytest.raises(ValueError, match="different size"):
        task.own_output(DataAllocation("provider-a", "same", 33))
    first.release()
    replacement = task.own_output(DataAllocation("provider-a", "same", 33))
    assert ledger.snapshot()["retained_bytes"] == 97
    second.release()
    replacement.release()


def test_drain_fences_queries_and_close_waits_for_submitted_tasks_but_not_outputs(data_scope):
    ledger, query, task = data_scope
    output = task.own_output(DataAllocation("local_shm", "output", 8))
    ledger.drain()
    with pytest.raises(RuntimeError, match="draining"):
        ledger.open_query()
    later = query.open_task()  # Already prepared queries can finish their work.
    query.shutdown(kill=True)
    with pytest.raises(RuntimeError, match="closed"):
        query.open_task()
    with pytest.raises(TimeoutError, match="active queries or tasks"):
        ledger.close()
    task.finish()
    assert query.cleanup_pending()
    later.finish()
    assert not query.cleanup_pending()
    ledger.close()
    view = output.fork()
    output.release()
    assert ledger.snapshot()["retained_bytes"] == 8
    assert ledger.snapshot()["closed"]
    view.release()
    view.release()
    assert ledger.snapshot()["retained_bytes"] == 0
    with pytest.raises(RuntimeError, match="released"):
        output.fork()
    with pytest.raises(RuntimeError, match="finished"):
        task.hold_inputs([DataAllocation("local_shm", "late", 1)])
    with pytest.raises(RuntimeError, match="finished"):
        task.own_output(DataAllocation("local_shm", "late", 1))


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan")])
def test_close_rejects_invalid_deadlines(data_scope, timeout):
    ledger, _, _ = data_scope
    with pytest.raises(ValueError, match="finite and non-negative"):
        ledger.close(timeout=timeout)
    assert not ledger.snapshot()["draining"]


def test_invocation_context_restores_nested_scopes_after_failure(data_scope):
    _, query, outer = data_scope
    inner = query.open_task()
    assert current_data_task() is None
    with outer.activate():
        assert current_data_task() is outer
        with pytest.raises(RuntimeError, match="user failure"), inner.activate():
            assert current_data_task() is inner
            raise RuntimeError("user failure")
        assert current_data_task() is outer
    assert current_data_task() is None
    inner.finish()


def test_output_owners_do_not_retain_query_task_or_request_objects():
    ledger = RuntimeDataLedger()

    class Request:
        pass

    def execute():
        request = Request()
        query = ledger.open_query()
        task = query.open_task()
        output = task.own_output(DataAllocation("local_shm", "out", 64))
        task.finish()
        query.shutdown()
        return output, weakref.ref(request), weakref.ref(query), weakref.ref(task)

    output, request_ref, query_ref, task_ref = execute()
    gc.collect()
    assert request_ref() is query_ref() is task_ref() is None
    ledger.close()
    del output
    gc.collect()
    assert ledger.snapshot()["retained_bytes"] == 0


def test_concurrent_view_forks_preserve_allocation_until_last_release(data_scope):
    ledger, query, task = data_scope
    output = task.own_output(DataAllocation("local_shm", "out", 1024))
    task.finish()
    query.shutdown()
    ledger.close()
    barrier = threading.Barrier(8)

    def hold_view(_):
        view = output.fork()
        try:
            barrier.wait(timeout=5)
            assert ledger.snapshot()["retained_bytes"] == 1024
        finally:
            view.release()
            view.release()

    with ThreadPoolExecutor(max_workers=8) as threads:
        list(threads.map(hold_view, range(8)))
    assert ledger.snapshot()["leases"] == 1
    output.release()
    assert ledger.snapshot()["retained_bytes"] == 0


@pytest.fixture(params=["local", "ray"])
def output_owner(request):
    if request.param == "local":
        ledger = RuntimeDataLedger()
        query = ledger.open_query()
        task = query.open_task()
        owner = task.own_output(DataAllocation("local_shm", "output", 80))

        def finish():
            task.finish()
            query.shutdown()
            ledger.close()

        def retained():
            return ledger.snapshot()["retained_bytes"]
    else:
        from vane.execution.resources import ResourceVector
        from vane.runners.ray.query_resource_graph import QueryAllocation, QueryResourceGraph, ResourceUnitSpec
        from vane.runners.ray.query_resource_manager import (
            OutputBlockLeaseOwner as RayOutputBlockLeaseOwner,
        )
        from vane.runners.ray.query_resource_manager import (
            OutputBlockRequest,
            RayQueryResourceManager,
            TaskRequest,
        )

        assert RayOutputBlockLeaseOwner is OutputBlockLeaseOwner
        unit = ResourceUnitSpec(
            query_id="q",
            resource_unit_id="resource:q:node",
            physical_node_id="node",
            unit_kind="ray_task_udf",
            backend="ray_task",
            input_unit_ids=(),
            per_task=ResourceVector(cpu=1),
            target_output_block_bytes=80,
            generator_buffer_blocks=1,
            max_concurrency=None,
        )
        graph = QueryResourceGraph("q", "sha256:ownership-test", (unit,), (unit.resource_unit_id,))
        manager = RayQueryResourceManager(
            graph, QueryAllocation(resources=ResourceVector(cpu=1, object_store_bytes=1000), generation=1)
        )
        manager.update_unit_state(unit.resource_unit_id, runnable=True)
        task = manager.try_acquire_task(TaskRequest("q", unit.resource_unit_id, "task:0", "0", None))
        assert task.granted
        output = manager.try_acquire_output_block(
            OutputBlockRequest("q", unit.resource_unit_id, task.lease.lease_id, "0", "out", 80)
        )
        assert output.granted
        owner = RayOutputBlockLeaseOwner(manager, output.lease)

        def finish():
            manager.release_task_lease(task.lease.lease_id, attempt_id="0")

        def retained():
            return manager.snapshot()["usage"]["object_store_bytes"]

    try:
        yield owner, finish, retained
    finally:
        finish()
        owner.release()


def test_shared_owner_keeps_output_through_task_completion_and_forward_transitions(output_owner):
    owner, finish, retained = output_owner
    finish()
    assert retained() == 80
    assert owner.transition_to("downstream_input")
    assert owner.transition_to("downstream_input")
    assert owner.transition_to("external_consumer")
    with pytest.raises(ValueError, match="backward"):
        owner.transition_to("unit_queue")
    with pytest.raises(ValueError, match="invalid"):
        owner.transition_to("released")
    assert retained() == 80
    assert owner.release()
    assert not owner.release()
    assert owner.state == "released"
    assert not owner.transition_to("external_consumer")
    assert retained() == 0


def test_shared_owner_concurrent_release_returns_capacity_once(output_owner):
    owner, finish, retained = output_owner
    finish()
    with ThreadPoolExecutor(max_workers=8) as threads:
        released = list(threads.map(lambda _: owner.release(), range(32)))
    assert sum(released) == 1
    assert retained() == 0
