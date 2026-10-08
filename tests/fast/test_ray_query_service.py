# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Application service ownership across sessions and distributed query modes."""

import gc
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import vane
from tests.fast.test_ray_recovery_runtime import assert_idle, close_result, resources
from vane.execution.request_admission import RequestCancelled, RequestQueueTimeout
from vane.execution.result_delivery import ResultDeliveryFull

pytestmark = [pytest.mark.real_ray, pytest.mark.usefixtures("ray_local")]


def test_sessions_share_workers_and_persistent_result_service(tmp_path):
    with vane.Runtime(resources(tmp_path)) as application:
        assert application.resource_snapshot() == {"started": False, "service": None}
        actors, epochs = [], []
        for mode, batch_rows, cast in (("pipelined", 2, "bigint"), ("fte", 5, "varchar"), ("pipelined", 1, "double")):
            with application.connect(execution=mode) as connection:
                with connection.query(
                    f"select range::{cast} as value from range(17)", rows_per_batch=batch_rows
                ) as result:
                    scheduler = result.context._reader
                    actors.append(scheduler.relay._actor_id)
                    epochs.append(tuple(connection.query_runtime.pool.epochs))
                    rows = []
                    for batch in result:
                        assert 0 < batch.num_rows <= batch_rows
                        rows.extend(batch.column(0).to_pylist())
                    del batch
                    convert = {"bigint": int, "varchar": str, "double": float}[cast]
                    assert sorted(rows) == sorted(map(convert, range(17)))
                assert_idle(connection)
            service = application.resource_snapshot()["service"]
            assert service["sessions"] == {}
            assert service["result_service"] == {"active_contexts": 0, "started": True, "capacity": 4}
        assert len(set(actors)) == 1
        assert len(set(epochs)) == 1


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_late_controls_cannot_touch_other_result_contexts(tmp_path, mode):
    import ray

    with vane.Runtime(resources(tmp_path)) as application, application.connect() as connection:
        first = connection.query("select 42", execution=mode)
        old = first.context._reader
        old_sequence = connection.query_runtime.pool.results.sequences[old.result_id]
        first.collect()
        with connection.query("select range from range(100)", execution=mode) as result:
            current = result.context._reader
            assert current.relay._actor_id == old.relay._actor_id
            assert current.result_id != old.result_id
            ray.get(old.relay.cancel.remote(old.result_id, "late cancellation"), timeout=5)
            ray.get(old.relay.release.remote(old.result_id, old_sequence), timeout=5)
            stale = [
                old.relay.status.remote(old.result_id),
                old.relay.prepare.remote(old.result_id, b"", ""),
                old.relay.connect.remote(old.result_id, "", "", 1),
                old.relay.connect_materialized.remote(old.result_id, {}, {}),
            ]
            for reference in stale:
                with pytest.raises(Exception, match="unknown or closed result context"):
                    ray.get(reference, timeout=5)
            assert sorted(result.collect().column(0).to_pylist()) == list(range(100))
        assert_idle(connection)


def test_closing_one_session_preserves_another_and_global_result_capacity(tmp_path):
    with (
        vane.Runtime(resources(tmp_path, max_results=2)) as application,
        application.connect() as first,
        application.connect() as second,
        application.connect() as overflow,
    ):
        a = first.query("select range from range(1000000)")
        b = second.query("select range from range(100)", execution="fte")
        try:
            assert a.context._reader.relay._actor_id == b.context._reader.relay._actor_id
            assert first.query_runtime.pool is second.query_runtime.pool
            assert application.resource_snapshot()["service"]["result_delivery"]["active_results"] == 2
            with pytest.raises(ResultDeliveryFull, match="slots are full"):
                overflow.query("select 7")
            first.close()
            assert sorted(b.collect().column(0).to_pylist()) == list(range(100))
            assert overflow.query("select 7").collect().column(0).to_pylist() == [7]
        finally:
            close_result(a)
            close_result(b)
        assert_idle(second)
        assert application.resource_snapshot()["service"]["request_admission"]["active_requests"] == 0


def test_global_admission_cannot_be_bypassed_by_new_sessions(tmp_path):
    with (
        vane.Runtime(resources(tmp_path, max_active_queries=1)) as application,
        application.connect() as first,
        application.connect() as second,
    ):
        a = first.query("select range from range(1000000)")
        options = vane.QueryExecutionOptions(vane.RayExecution(), 0.1, 30, 30)
        try:
            with pytest.raises(RequestQueueTimeout):
                second.query("select 2", options=options)
        finally:
            close_result(a)
        assert second.query("select 7").collect().column(0).to_pylist() == [7]
        assert application.resource_snapshot()["service"]["request_admission"]["active_requests"] == 0


def test_session_admission_does_not_occupy_other_sessions_slots(tmp_path):
    with (
        vane.Runtime(resources(tmp_path, max_active_queries=2)) as application,
        application.connect(resources=vane.QueryResources(max_active_queries=1)) as first,
        first.cursor() as sibling,
        application.connect() as second,
    ):
        a = first.query("select range from range(1000000)")
        try:
            options = vane.QueryExecutionOptions(vane.RayExecution(), 0.1, 30, 30)
            with pytest.raises(RequestQueueTimeout):
                sibling.query("select 2", options=options)
            assert second.query("select 7").collect().column(0).to_pylist() == [7]
        finally:
            close_result(a)


def test_global_buffer_budget_survives_session_close_and_retained_views(tmp_path):
    with (
        vane.Runtime(resources(tmp_path, result_buffer_bytes=1024)) as application,
        application.connect() as first,
        application.connect() as second,
    ):
        a = first.query("select range from range(1000000)", rows_per_batch=64)
        held = a.read_batch()
        assert held.num_rows == 64
        held_bytes = application.resource_snapshot()["service"]["result_delivery"]["usage_bytes"]
        # IPC includes schema and message framing as well as the 512 data bytes.
        # Each batch fits; two simultaneously retained batches do not.
        assert 512 < held_bytes <= 1024
        first.close()
        state = application.resource_snapshot()["service"]["result_delivery"]
        assert state["usage_bytes"] == held_bytes
        b = second.query("select range from range(1000000)", rows_per_batch=64)
        with ThreadPoolExecutor(1) as threads:
            pending = threads.submit(b.read_batch)
            deadline = time.monotonic() + 10
            while not b._waiting_bytes:
                assert not pending.done()
                assert time.monotonic() < deadline
                time.sleep(0.005)
            assert held.column(0).to_pylist()
            del held
            gc.collect()
            batch = pending.result(timeout=5)
            assert batch.num_rows == 64
            del batch, pending  # The completed Future also owns the Arrow view.
        close_result(b)
        close_result(a)
        gc.collect()
        assert application.resource_snapshot()["service"]["result_delivery"]["usage_bytes"] == 0


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_cancel_during_context_creation_keeps_service_usable(tmp_path, monkeypatch, mode):
    from vane.execution import pipelined_runtime

    with vane.Runtime(resources(tmp_path)) as application, application.connect() as connection:
        connection.query("select 1", execution=mode).collect()
        service = connection.query_runtime.pool.results
        actor = service.actor
        entered, proceed = threading.Event(), threading.Event()
        original = pipelined_runtime._get

        def delay(reference, context=None, timeout=30):
            if context is not None:
                entered.set()
                assert proceed.wait(10)
            return original(reference, context, timeout)

        with monkeypatch.context() as patch, ThreadPoolExecutor(1) as threads:
            patch.setattr(pipelined_runtime, "_get", delay)
            pending = threads.submit(connection.query, "select 2", execution=mode)
            try:
                assert entered.wait(10)
                connection.interrupt()
            finally:
                proceed.set()
            with pytest.raises(RequestCancelled):
                pending.result(timeout=10)
        assert service.snapshot()["active_contexts"] == 0
        assert service.actor is actor
        assert connection.query("select 7", execution=mode).collect().column(0).to_pylist() == [7]


def test_result_process_loss_fails_all_resident_results_without_replacement(tmp_path):
    import ray

    with (
        vane.Runtime(resources(tmp_path)) as application,
        application.connect() as first,
        application.connect() as second,
    ):
        a = first.query("select range from range(1000000)")
        b = second.query("select range from range(100)", execution="fte")
        actor = a.context._reader.relay
        assert actor._actor_id == b.context._reader.relay._actor_id
        ray.kill(actor, no_restart=True)
        for result in (a, b):
            try:
                with pytest.raises(Exception):
                    result.collect()
                assert result.execution_state == "FAILED"
            finally:
                close_result(result)
        with pytest.raises(Exception):
            first.query("select 7")
        assert first.query_runtime.pool.results.actor is actor


@pytest.mark.timeout(90)
@pytest.mark.parametrize("mode", ["pipelined", "fte"])
@pytest.mark.parametrize("failed_rpc", ["release", "release_receipt"])
def test_result_service_outage_keeps_public_cleanup_retryable(tmp_path, monkeypatch, mode, failed_rpc):
    import ray

    with (
        vane.Runtime(resources(tmp_path, worker_count=1, partitions=1, max_results=1)) as application,
        application.connect(execution=mode) as connection,
    ):
        result = connection.query("select range from range(8)")
        service = connection.query_runtime.pool.results
        actor = service.actor
        original_release = actor.release.remote

        def fail_release(query_id, sequence):
            if failed_rpc == "release_receipt":
                # The cleanup may have run even though its reply was lost.
                ray.get(original_release(query_id, sequence), timeout=5)
                return ray.put(ray.exceptions.ActorUnavailableError("lost release reply", None))
            raise ray.exceptions.ActorUnavailableError("temporary release outage", None)

        try:
            with monkeypatch.context() as patch:
                patch.setattr(actor, "release", SimpleNamespace(remote=fail_release))
                for _ in range(2):
                    with pytest.raises(RuntimeError, match=r"retry result\.close"):
                        result.close()
                    assert service.snapshot()["active_contexts"] == 1
                    assert application.resource_snapshot()["service"]["result_delivery"]["active_results"] == 1
                    with pytest.raises(RuntimeError, match="capacity is full"):
                        service.create("next", service.resources)
            close_result(result)
            assert_idle(connection)
            assert connection.query("select 42").collect().column(0).to_pylist() == [42]
            assert service.actor is actor
        finally:
            close_result(result)


@pytest.mark.timeout(90)
@pytest.mark.parametrize("mode", ["pipelined", "fte"])
@pytest.mark.parametrize("failed_rpc", ["release", "release_receipt"])
def test_worker_outage_keeps_session_and_capacity_until_cleanup_succeeds(tmp_path, monkeypatch, mode, failed_rpc):
    import ray

    from vane.execution.recovery_runtime import RecoveryScheduler

    dispatched = threading.Event()
    dispatch = RecoveryScheduler._dispatch

    def hold_attempt(owner, *args):
        accepted = dispatch(owner, *args)
        if accepted:
            # Keep a real, prepared FTE context resident until session close
            # cancels it, regardless of how quickly its task finishes.
            attempt = next(iter(owner.active.values()))
            ray.get(attempt.prepare, timeout=10)
            dispatched.set()
            assert owner.stop.wait(20), "test did not cancel the prepared attempt"
        return accepted

    limits = resources(tmp_path, worker_count=1, partitions=1, task_contexts_per_worker=1, max_active_queries=1)
    with vane.Runtime(limits) as application, application.connect() as first, application.connect() as second:
        with monkeypatch.context() as patch:
            if mode == "fte":
                patch.setattr(RecoveryScheduler, "_dispatch", hold_attempt)
            result = first.query(
                "select range from range(1000000)" if mode == "pipelined" else "select 42", execution=mode
            )
            session = first.query_runtime
            scheduler = result.context._reader
            pool = session.pool
            worker = pool.workers[0]
            method = "release" if mode == "pipelined" else "release_materialized"
            original_release = getattr(worker, method).remote

            def unavailable(*args):
                if failed_rpc == "release_receipt":
                    ray.get(original_release(*args), timeout=5)
                    return ray.put(ray.exceptions.ActorUnavailableError("lost worker release reply", None))
                raise ray.exceptions.ActorUnavailableError("temporary worker release outage", None)

            try:
                if mode == "fte":
                    assert dispatched.wait(10)
                native = ray.get(worker.resources_snapshot.remote(), timeout=5)["reservations"]
                coordinator = pool.admission.snapshot()["reservations"]
                assert native and coordinator
                with monkeypatch.context() as outage:
                    outage.setattr(worker, method, SimpleNamespace(remote=unavailable))
                    for _ in range(2):
                        with pytest.raises(RuntimeError, match="retry"):
                            first.close()
                        assert session.session_id in application.resource_snapshot()["service"]["sessions"]
                        assert pool.admission.snapshot()["reservations"] == coordinator
                        assert ray.get(worker.resources_snapshot.remote(), timeout=5)["reservations"] == (
                            native if failed_rpc == "release" else {}
                        )
                        assert not scheduler.closed
                    with pytest.raises(RequestQueueTimeout):
                        second.query("select 7", options=vane.QueryExecutionOptions(vane.RayExecution(), 0.1, 30, 30))
                first.close()
                assert session.session_id not in application.resource_snapshot()["service"]["sessions"]
                assert pool.admission.snapshot()["reservations"] == {}
                assert ray.get(worker.resources_snapshot.remote(), timeout=5)["reservations"] == {}
                assert second.query("select 7").collect().column(0).to_pylist() == [7]
            finally:
                close_result(result)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("mode", ["pipelined", "fte"])
@pytest.mark.parametrize("receipt_kind", ["create", "prepare"])
@pytest.mark.parametrize("actor_state", ["alive", "dead"])
def test_failed_creation_receipt_does_not_block_fresh_cleanup(tmp_path, monkeypatch, mode, receipt_kind, actor_state):
    import ray

    from vane.execution.recovery_runtime import RecoveryScheduler

    dispatched = threading.Event()
    dispatch = RecoveryScheduler._dispatch

    def hold_attempt(owner, *args):
        accepted = dispatch(owner, *args)
        if accepted:
            attempt = next(iter(owner.active.values()))
            ray.get(attempt.prepare, timeout=10)
            dispatched.set()
            assert owner.stop.wait(20), "test did not cancel the prepared attempt"
        return accepted

    if mode == "fte":
        monkeypatch.setattr(RecoveryScheduler, "_dispatch", hold_attempt)
    application = vane.Runtime(resources(tmp_path, worker_count=1, partitions=1, max_active_queries=1, max_results=1))
    connection = application.connect(execution=mode)
    result = connection.query("select range from range(1000000)" if mode == "pipelined" else "select 42")
    scheduler = result.context._reader
    session = connection.query_runtime
    service = session.pool.results
    try:
        if mode == "fte":
            assert dispatched.wait(10)
        # This is an immutable failed ObjectRef, not a mocked get() which can
        # become successful later. Recovery must use a new remote invocation.
        failed = ray.put(ray.exceptions.ActorUnavailableError("lost creation reply", None))
        for _ in range(2):
            with pytest.raises(ray.exceptions.ActorUnavailableError):
                ray.get(failed)
        if receipt_kind == "create":
            service.contexts[scheduler.result_id] = failed
            actor = service.actor
        else:
            if mode == "pipelined":
                scheduler.prepare_calls[:] = [failed]
            else:
                next(iter(scheduler.active.values())).prepare = failed
            actor = session.pool.workers[0]
        if actor_state == "dead":
            ray.kill(actor, no_restart=True)
            deadline = time.monotonic() + 10
            while True:
                try:
                    ray.get(actor.__ray_ready__.remote(), timeout=1)
                except ray.exceptions.ActorDiedError:
                    break
                except ray.exceptions.ActorUnavailableError:
                    pass
                assert time.monotonic() < deadline
                time.sleep(0.01)
        connection.close()
        assert scheduler.closed
        snapshot = application.resource_snapshot()["service"]
        assert snapshot["sessions"] == {}
        assert snapshot["request_admission"]["active_requests"] == 0
        assert snapshot["workers"]["reservations"] == {}
        assert snapshot["result_service"]["active_contexts"] == 0
        assert snapshot["result_delivery"]["active_results"] == 0
        if actor_state == "alive":
            assert ray.get(session.pool.workers[0].resources_snapshot.remote(), timeout=5)["reservations"] == {}
            with application.connect() as second:
                assert second.query("select 7").collect().column(0).to_pylist() == [7]
        application.close()
        assert application.resource_snapshot()["service"]["closed"]
    finally:
        close_result(result)
        connection.close()
        application.close()


@pytest.mark.timeout(60)
@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_runtime_close_deadline_covers_remote_cleanup_and_preserves_retry(tmp_path, monkeypatch, mode):
    from vane.execution import pipelined_runtime

    application = vane.Runtime(resources(tmp_path, worker_count=1, partitions=1))
    connection = application.connect(execution=mode)
    result = connection.query("select 42")
    service = connection.query_runtime.pool.results
    receipts = []
    original_release = service.actor.release.remote
    original_get = pipelined_runtime._get
    entered, proceed = threading.Event(), threading.Event()
    timeouts = []

    def release(*args):
        reference = original_release(*args)
        receipts.append(reference)
        return reference

    def delayed(reference, context=None, timeout=30):
        if reference in receipts:
            timeouts.append(timeout)
            entered.set()
            assert proceed.wait(10), "test did not release cleanup RPC"
        return original_get(reference, context, timeout)

    try:
        with monkeypatch.context() as patch, ThreadPoolExecutor(1) as threads:
            patch.setattr(service.actor, "release", SimpleNamespace(remote=release))
            patch.setattr(pipelined_runtime, "_get", delayed)
            started = time.monotonic()
            pending = threads.submit(application.close, timeout=0.2)
            try:
                assert entered.wait(5)
                with pytest.raises(TimeoutError, match="retry Runtime.close"):
                    pending.result(timeout=1)
                assert time.monotonic() - started < 1
                assert len(timeouts) == 1 and 0 < timeouts[0] <= 0.2
                assert service.snapshot()["active_contexts"] == 1
                snapshot = application.resource_snapshot()["service"]
                assert snapshot["closing"] and not snapshot["closed"]
                assert snapshot["sessions"]
            finally:
                proceed.set()
            assert application._service.close_attempt.done.wait(5)
        application.close()
        snapshot = application.resource_snapshot()["service"]
        assert snapshot["closed"] and snapshot["sessions"] == {}
        assert snapshot["workers"]["reservations"] == {}
        assert service.snapshot()["active_contexts"] == 0
    finally:
        proceed.set()
        application.close()
        close_result(result)
        connection.close()


def test_runtime_close_cancels_queries_and_stops_shared_processes(tmp_path):
    import ray

    application = vane.Runtime(resources(tmp_path))
    first, second = application.connect(), application.connect()
    a = first.query("select range from range(1000000)")
    b = second.query("select range from range(100)", execution="fte")
    actor = a.context._reader.relay
    workers = tuple(first.query_runtime.pool.workers)
    try:
        application.close()
        with pytest.raises(RuntimeError, match="closed"):
            application.connect()
        with pytest.raises(vane.ConnectionException, match="clos(ed|ing)"):
            first.query("select 7")
        # ray.kill sends a termination request; its return does not acknowledge
        # process exit. Observe death with a bound instead of assuming that the
        # very next RPC cannot race the kill. A live-but-closed service can reject
        # status(), so use Ray's health method to distinguish that from death.
        deadline = time.monotonic() + 10
        for process in (*workers, actor):
            while True:
                remaining = deadline - time.monotonic()
                assert remaining > 0, "Runtime actor did not terminate"
                try:
                    ray.get(process.__ray_ready__.remote(), timeout=remaining)
                except ray.exceptions.ActorDiedError:
                    break
                except ray.exceptions.ActorUnavailableError:
                    pass
                time.sleep(min(0.01, remaining))
    finally:
        close_result(a)
        close_result(b)
        first.close()
        second.close()
        application.close()


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_runtime_close_retires_query_on_orphaned_nested_cursor(tmp_path, mode):
    application = vane.Runtime(resources(tmp_path, worker_count=1, partitions=1))
    connection = application.connect(execution=mode)
    nested = connection.cursor().cursor()
    result = nested.query("select range from range(1000000)" if mode == "pipelined" else "select 42")
    try:
        application.close()
        snapshot = application.resource_snapshot()["service"]
        assert snapshot["closed"] and snapshot["sessions"] == {}
        assert snapshot["request_admission"]["active_requests"] == 0
        assert snapshot["result_delivery"]["active_results"] == 0
        assert snapshot["workers"]["reservations"] == {}
        assert not result.context.cleanup_pending()
        with pytest.raises(vane.ConnectionException, match="closed"):
            nested.cursor()
    finally:
        close_result(result)
        nested.close()
        connection.close()
        application.close()


@pytest.mark.timeout(60)
@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_runtime_close_retries_failed_result_actor_termination(tmp_path, monkeypatch, mode):
    import ray

    application = vane.Runtime(resources(tmp_path, worker_count=1, partitions=1))
    connection = application.connect(execution=mode)
    original_kill = ray.kill
    actor = None
    attempts = 0
    try:
        assert connection.query("select 7").collect().column(0).to_pylist() == [7]
        actor = connection.query_runtime.pool.results.actor

        def fail_once(candidate, *, no_restart):
            nonlocal attempts
            if candidate._actor_id == actor._actor_id:
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("temporary actor termination failure")
            return original_kill(candidate, no_restart=no_restart)

        with monkeypatch.context() as patch:
            patch.setattr(ray, "kill", fail_once)
            with pytest.raises(RuntimeError, match="temporary actor termination failure"):
                application.close()
            snapshot = application.resource_snapshot()["service"]
            assert snapshot["closing"] and not snapshot["closed"]
            assert snapshot["result_service"]["active_contexts"] == 0
            assert attempts == 1
            assert ray.get(actor.__ray_ready__.remote(), timeout=5)
            with pytest.raises(RuntimeError, match="closed"):
                application.connect()

            application.close()
            assert application.resource_snapshot()["service"]["closed"]
            assert attempts == 2
            application.close()
            assert attempts == 2

        deadline = time.monotonic() + 10
        while True:
            remaining = deadline - time.monotonic()
            assert remaining > 0, "result actor did not terminate after retry"
            try:
                ray.get(actor.__ray_ready__.remote(), timeout=remaining)
            except ray.exceptions.ActorDiedError:
                break
            except ray.exceptions.ActorUnavailableError:
                pass
            time.sleep(min(0.01, remaining))
    finally:
        if actor is not None:
            original_kill(actor, no_restart=True)
        connection.close()
        application.close()
