# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from vane.execution.ref_bundle import LocalShmBudgetManager
from vane.execution.udf_input_cleanup import QueryInputCleanup, current_input_cleanup


class InputOwner:
    def __init__(self):
        self.fail = True
        self.released = False

    def release(self):
        if self.fail:
            raise RuntimeError("planned input release failure")
        self.released = True


def hold(task, manager, owner):
    lease_id = manager.create_input_lease([owner], 296)
    task.hold_input_transport(manager, lease_id)
    return lease_id


def test_query_shutdown_waits_for_running_task_and_retries_only_finished_inputs():
    manager = LocalShmBudgetManager(limit_factory=lambda: 4096)
    owner = InputOwner()
    query = QueryInputCleanup()
    task = query.open_task()
    lease_id = hold(task, manager, owner)
    query.shutdown()
    assert query.cleanup_pending()
    assert manager.input_lease_pending(lease_id)
    with pytest.raises(RuntimeError, match="closed"):
        query.open_task()
    with pytest.raises(RuntimeError, match="release failure"):
        task.finish()
    with pytest.raises(RuntimeError, match="release failure"):
        query.shutdown()
    assert query.cleanup_pending()
    owner.fail = False
    query.shutdown()
    assert not query.cleanup_pending()
    assert not manager.input_lease_pending(lease_id)
    assert owner.released
    task.finish()
    query.shutdown()


@pytest.mark.parametrize("fails", [False, True])
def test_query_retry_during_input_ack_retains_owner(monkeypatch, fails):
    manager = LocalShmBudgetManager(limit_factory=lambda: 4096)
    query = QueryInputCleanup()
    task = query.open_task()
    owner = InputOwner()
    owner.fail = False
    lease_id = hold(task, manager, owner)
    entered, proceed = threading.Event(), threading.Event()
    release = manager._release_input_ack_ref

    def blocked_release(*args, **kwargs):
        entered.set()
        assert proceed.wait(5)
        if fails:
            raise RuntimeError("planned blocked ACK failure")
        return release(*args, **kwargs)

    with ThreadPoolExecutor(max_workers=1) as threads:
        with monkeypatch.context() as fault:
            fault.setattr(manager, "_release_input_ack_ref", blocked_release)
            ack = threads.submit(manager.consume_input_lease, lease_id)
            try:
                assert entered.wait(3)
                with pytest.raises(RuntimeError, match="still in progress"):
                    task.finish()
                with pytest.raises(RuntimeError, match="still in progress"):
                    query.shutdown()
                assert query.cleanup_pending()
                assert manager.input_lease_pending(lease_id)
                assert manager.snapshot()["output_credit_bytes"] == 0
            finally:
                proceed.set()
            if fails:
                with pytest.raises(RuntimeError, match="ACK failure"):
                    ack.result(timeout=5)
            else:
                ack.result(timeout=5)
    query.shutdown()
    assert not query.cleanup_pending()
    assert not manager.input_lease_pending(lease_id)
    assert manager.snapshot()["output_credit_bytes"] == 0


def test_failed_cleanup_retry_preserves_another_querys_shared_input():
    manager = LocalShmBudgetManager(limit_factory=lambda: 4096)
    first, second = QueryInputCleanup(), QueryInputCleanup()
    first_task, second_task = first.open_task(), second.open_task()
    owner = InputOwner()
    first_id = hold(first_task, manager, owner)
    with pytest.raises(RuntimeError, match="release failure"):
        first_task.finish()
    second_id = hold(second_task, manager, owner)
    owner.fail = False
    first.shutdown()
    assert not first.cleanup_pending()
    assert not manager.input_lease_pending(first_id)
    assert manager.input_lease_pending(second_id)
    assert not owner.released
    second.shutdown()
    assert second.cleanup_pending()
    second_task.finish()
    assert not second.cleanup_pending()
    assert owner.released


def test_failed_cleanup_does_not_skip_other_tasks_or_transport_managers():
    first_manager = LocalShmBudgetManager(limit_factory=lambda: 4096)
    second_manager = LocalShmBudgetManager(limit_factory=lambda: 4096)
    query = QueryInputCleanup()
    one, two = query.open_task(), query.open_task()
    failed, other = InputOwner(), InputOwner()
    failed_id = hold(one, first_manager, failed)
    other_id = hold(one, second_manager, other)
    third_id = hold(two, first_manager, other)
    assert failed_id == other_id  # Lease IDs are scoped to their manager.
    for task in (one, two):
        with pytest.raises(RuntimeError, match="release failure"):
            task.finish()
    other.fail = False
    with pytest.raises(RuntimeError, match="release failure"):
        query.shutdown()
    assert first_manager.input_lease_pending(failed_id)
    assert not first_manager.input_lease_pending(third_id)
    assert not second_manager.input_lease_pending(other_id)
    failed.fail = False
    query.shutdown()
    assert not query.cleanup_pending()


def test_task_context_is_scoped_and_successful_tasks_retire_during_query():
    query = QueryInputCleanup()
    manager = LocalShmBudgetManager(limit_factory=lambda: 4096)
    owner = InputOwner()
    owner.fail = False
    first = query.open_task()
    assert current_input_cleanup() is None
    with first.activate():
        for _ in range(100):
            task = query.open_task()
            with task.activate():
                assert current_input_cleanup() is task
                hold(task, manager, owner)
            assert current_input_cleanup() is first
            task.finish()
        assert len(query._tasks) == 1
    assert current_input_cleanup() is None
    first.finish()
    assert not query.cleanup_pending()
    assert manager.snapshot()["active_input_leases"] == 0


@pytest.mark.parametrize("failure", ["ack", "completion"])
def test_materialized_input_cleanup_failure_is_a_consumable_task_result(monkeypatch, failure):
    import pyarrow as pa

    from vane import pickle as vane_pickle
    from vane.execution import ref_bundle
    from vane.execution.udf import build_executor

    manager = LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    query = QueryInputCleanup()
    payload = dict(
        function_pickle=vane_pickle.dumps(lambda table: table),
        call_mode="map_batches",
        execution_backend="subprocess_task",
        udf_worker_slots=1,
        produce_ref_bundle_output=True,
        streaming_output_mode="local_shm_ref_bundle",
    )
    executor = build_executor(payload, {"local_input_cleanup": query})
    cancel = manager.cancel_input_lease

    def fail(*args, **kwargs):
        raise RuntimeError("planned materialized input cleanup failure")

    def fail_completion(*args, **kwargs):
        if kwargs.get("name") == "task-input-cleanup":
            fail()
        return cancel(*args, **kwargs)

    try:
        with monkeypatch.context() as fault:
            if failure == "ack":
                fault.setattr(manager, "_release_input_ack_ref", fail)
            else:
                fault.setattr(manager, "cancel_input_lease", fail_completion)
            assert executor.request_task_admission(8)
            executor.submit_with_id(1, pa.table({"x": [7]}))
            assert executor._wait_for_pending_futures(15)
            result = executor.take_ready_result()
            assert isinstance(result[2], BaseException)
            assert "materialized input cleanup failure" in str(result[2])
            executor.stats()  # Cleanup must not poison wakeups or strand the result's slot.
            assert query.cleanup_pending()
            with pytest.raises(RuntimeError, match="materialized input cleanup failure"):
                query.shutdown()
        query.shutdown()
        assert not query.cleanup_pending()
        assert manager.snapshot()["active_input_leases"] == 0
        # A successful worker output rejected by completion cleanup was released.
        assert manager.snapshot()["usage_bytes"] == 0
    finally:
        executor.close(kill=True)
        query.shutdown()


@pytest.mark.parametrize("failure", ["schedule", "spawn"])
def test_ref_input_cleanup_survives_failure_before_worker_submission(monkeypatch, failure):
    import pyarrow as pa

    from vane import pickle as vane_pickle
    from vane.execution import ref_bundle
    from vane.execution.udf import build_executor

    manager = LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    query = QueryInputCleanup()
    payload = dict(
        function_pickle=vane_pickle.dumps(lambda table: table),
        call_mode="map_batches",
        execution_backend="subprocess_task",
        udf_worker_slots=1,
        produce_ref_bundle_output=True,
        streaming_output_mode="local_shm_ref_bundle",
    )
    executor = build_executor(payload, {"local_input_cleanup": query})
    original = ref_bundle.make_local_shm_ref_bundle_result(pa.table({"x": [7]}))

    class MetadataOwner:
        name, size = original[1][0].name, original[1][0].size

        def release(self):
            original[1][0].release()

        def __reduce__(self):
            raise AssertionError("custom input owners must stay in the parent process")

    def fail(*args, **kwargs):
        raise RuntimeError("planned input submission or cleanup failure")

    try:
        with monkeypatch.context() as fault:
            fault.setattr(manager, "_release_input_ack_ref", fail)
            if failure == "schedule":
                fault.setattr(executor._task_runtime.executor, "submit", fail)
            else:
                fault.setattr(executor._task_pool, "_spawn_worker", fail)
            assert executor.request_task_admission(8)
            if failure == "schedule":
                with pytest.raises(RuntimeError, match="submission or cleanup failure"):
                    executor.submit_ref_bundle_with_id(1, [MetadataOwner()], None, original[2], original[3])
            else:
                executor.submit_ref_bundle_with_id(1, [MetadataOwner()], None, original[2], original[3])
                assert executor._wait_for_pending_futures(15)
                assert isinstance(executor.take_ready_result()[2], BaseException)
            assert manager.snapshot()["active_input_leases"] == 1
            with pytest.raises(RuntimeError, match="submission or cleanup failure"):
                query.shutdown()
            assert query.cleanup_pending()
        query.shutdown()
        assert not query.cleanup_pending()
        assert manager.snapshot()["active_input_leases"] == 0
        assert manager.snapshot()["usage_bytes"] == 0
    finally:
        executor.close(kill=True)
        query.shutdown()
        original[1][0].release()
