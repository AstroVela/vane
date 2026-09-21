# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

import vane
from vane.execution import ref_bundle, udf_local_request, udf_subprocess
from vane.execution.request_admission import RequestAdmissionLimits, RequestCancelled, RequestExecutionTimeout
from vane.execution.request_deadline import RequestExecutionDeadline
from vane.execution.udf_local_model import LocalModelRuntime


def _plan(conn, function=None, value=7, *, actor=False):
    relation = conn.sql(value if isinstance(value, str) else f"SELECT {value}::BIGINT AS x")
    if function is not None:
        relation = relation.map_batches(
            function,
            schema={"x": vane.sqltypes.BIGINT},
            execution_backend="subprocess_actor" if actor else "subprocess_task",
            actor_number=1 if actor else None,
        )
    return vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, uuid.uuid4().hex).to_physical_plan(conn)


def _wait(predicate):
    deadline = time.monotonic() + 10
    while not predicate():
        assert time.monotonic() < deadline, "native request did not reach the expected state"
        time.sleep(0.01)


def _values(result):
    return [value for table in result.partition_payloads for value in table.column(0).to_pylist()]


@pytest.mark.parametrize("reason", ["cancelled", "execution_timeout"])
def test_recorded_cancellation_prevents_native_start_before_signal_dispatch(monkeypatch, tmp_path, reason):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    expired = threading.Event()
    # Drive the watcher callback from a second thread only after preparation
    # succeeds, so this handoff test does not depend on a wall-clock timeout.
    monkeypatch.setattr(RequestExecutionDeadline, "start", lambda self: None)
    monkeypatch.setattr(RequestExecutionDeadline, "expired", lambda self: expired.is_set())
    marker = str(tmp_path / "cancelled_udf_ran")

    def process(table):
        from pathlib import Path

        Path(marker).write_text("native work started after cancellation was recorded")
        return table

    with vane.connect() as conn:
        plan = _plan(conn, process)
        with (
            LocalModelRuntime(
                session_id=plan.session_id(),
                session_config=plan.session_config(),
                request_limit=RequestAdmissionLimits(1, 1),
            ) as runtime,
            ThreadPoolExecutor(max_workers=2) as threads,
        ):
            request, queued = runtime.request(), runtime.request()
            prepared, recorded, dispatch, finishing = (threading.Event() for _ in range(4))
            original_prepare = runtime._prepare
            original_cancel = request._cancellation.cancel
            original_finish = request._finish_execution
            original_native = udf_local_request._execute_native
            native_calls = []

            def prepare(*args, **kwargs):
                resources = original_prepare(*args, **kwargs)
                prepared.set()
                assert recorded.wait(10)
                return resources

            def delayed_cancel(message):
                recorded.set()
                assert dispatch.wait(20)
                return original_cancel(message)

            def finish(**kwargs):
                finishing.set()
                return original_finish(**kwargs)

            def native(*args, **kwargs):
                native_calls.append(request.cancellation_reason)
                return original_native(*args, **kwargs)

            monkeypatch.setattr(runtime, "_prepare", prepare)
            monkeypatch.setattr(request._cancellation, "cancel", delayed_cancel)
            monkeypatch.setattr(request, "_finish_execution", finish)
            monkeypatch.setattr(udf_local_request, "_execute_native", native)
            future = threads.submit(
                request.execute, plan, {}, conn=conn, execution_timeout=60 if reason == "execution_timeout" else None
            )
            try:
                assert prepared.wait(10)
                if reason == "execution_timeout":
                    expired.set()
                    cancellation = threads.submit(request._expire_deadline)
                else:
                    cancellation = threads.submit(request.cancel)
                assert recorded.wait(5)
                assert finishing.wait(10)
                assert request.cancellation_reason == reason
                assert not request._cancellation.is_set()
                assert request.state == "cancelling" and queued.state == "queued"
                assert not future.done()
                assert (native_calls, (tmp_path / "cancelled_udf_ran").exists()) == ([], False)
            finally:
                recorded.set()
                dispatch.set()
            cancellation.result(timeout=5)
            with pytest.raises(RequestExecutionTimeout if reason == "execution_timeout" else RequestCancelled):
                future.result(timeout=10)
            assert request.state == ("execution_timed_out" if reason == "execution_timeout" else "cancelled")
            assert queued.state == "ready"
            assert conn.sql("SELECT 42").fetchall() == [(42,)]


@pytest.mark.parametrize("drain", [False, True])
def test_deadline_interrupts_native_sql_and_fences_cursor_reuse(monkeypatch, drain):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    started = threading.Event()
    original = udf_local_request._NativeRequestCancellation.started

    def native_started(self, conn):
        original(self, conn)
        started.set()

    monkeypatch.setattr(udf_local_request._NativeRequestCancellation, "started", native_started)
    with vane.connect() as conn:
        plan = _plan(conn, value="SELECT sum(i) AS x FROM range(1000000000000) t(i)")
        with (
            LocalModelRuntime(
                session_id=plan.session_id(),
                session_config=plan.session_config(),
                request_limit=RequestAdmissionLimits(1, 1),
            ) as runtime,
            ThreadPoolExecutor(max_workers=1) as threads,
        ):
            request = runtime.request()
            future = threads.submit(request.execute, plan, {}, conn=conn, execution_timeout=1)
            assert started.wait(5)
            if drain:
                runtime.drain()
            with pytest.raises(RequestExecutionTimeout):
                future.result(timeout=10)
            assert request.state == "execution_timed_out"
            assert not request.cancel()
            # Simulate a copied deadline callback arriving after native detach.
            request._expire_deadline()
            assert conn.sql("SELECT 42").fetchall() == [(42,)]
            state = runtime.resource_snapshot()["request_admission"]
            assert state["execution_timed_out_requests"] == 1
            assert state["active_requests"] == state["timed_out_requests"] == state["cancelled_requests"] == 0


@pytest.mark.parametrize("actor", [False, True])
@pytest.mark.parametrize("retry_runtime", [False, True])
def test_deadline_keeps_request_charged_after_callback_cleanup_timeout(monkeypatch, actor, retry_runtime):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    monkeypatch.setenv("VANE_UDF_SUBPROCESS_SHUTDOWN_GRACE_S", "0.05")
    entered, release, completed = (threading.Event() for _ in range(3))
    executors = []
    original = udf_subprocess.UDFExecutor._complete_task_submit

    def delayed(self, *args, **kwargs):
        executors.append(self)
        entered.set()
        try:
            assert release.wait(15)
            return original(self, *args, **kwargs)
        finally:
            completed.set()

    monkeypatch.setattr(udf_subprocess.UDFExecutor, "_complete_task_submit", delayed)
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100_000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)

    def process(table):
        time.sleep(0.1)
        return table

    class Model:
        def __call__(self, table):
            return process(table)

    with vane.connect() as conn:
        plan = _plan(conn, Model if actor else process, actor=actor)
        runtime = LocalModelRuntime(
            session_id=plan.session_id(),
            session_config=plan.session_config(),
            request_limit=RequestAdmissionLimits(1, 1),
        )
        bindings = {}
        if actor:
            node = plan.collect_udf_nodes(conn=conn)[0]
            runtime.register("model", version="v1", payload=node["payload"])
            bindings[str(node["node_id"])] = "model"
        request, queued = runtime.request(), runtime.request()
        try:
            with ThreadPoolExecutor(max_workers=1) as threads:
                future = threads.submit(request.execute, plan, bindings, conn=conn, execution_timeout=2)
                try:
                    assert entered.wait(5)
                    with pytest.raises(RequestExecutionTimeout):
                        future.result(timeout=10)
                    assert request.state == "cancelling" and queued.state == "queued"
                    assert request.cancellation_reason == "execution_timeout"
                    assert executors[0].cleanup_pending()
                    assert manager.snapshot()["allocated_bytes"] > 0
                    state = runtime.resource_snapshot()["request_admission"]
                    assert state["active_requests"] == state["cleanup_pending_requests"] == 1
                    assert state["execution_timed_out_requests"] == 0
                    with pytest.raises(RuntimeError, match="request cleanup failed"):
                        if retry_runtime:
                            runtime.close(kill=True)
                        else:
                            request.shutdown(kill=True)
                finally:
                    release.set()
                assert completed.wait(5)
            if retry_runtime:
                runtime.close(timeout=5, kill=True)
                assert queued.state == "drained"
            else:
                request.shutdown(kill=True)
                assert queued.state == "ready"
                queued.cancel()
            assert request.state == "execution_timed_out"
            assert not executors[0].cleanup_pending()
            assert manager.snapshot()["usage_bytes"] == 0
            state = runtime.resource_snapshot()["request_admission"]
            assert state["execution_timed_out_requests"] == 1
            assert state["active_requests"] == state["cleanup_pending_requests"] == 0
        finally:
            release.set()
            runtime.close(timeout=5, kill=True)


@pytest.mark.parametrize("timeout_initializer", [False, True])
def test_deadline_during_shared_model_initialization_keeps_model_owned(monkeypatch, tmp_path, timeout_initializer):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    marker = str(tmp_path)

    class Model:
        def __init__(self):
            from pathlib import Path

            with Path(marker, "initializations").open("a") as output:
                output.write(f"{os.getpid()}\n")
            deadline = time.monotonic() + 20
            while not Path(marker, "release").exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("test did not release model construction")
                time.sleep(0.01)

        def __call__(self, table):
            return table

    with vane.connect() as first_conn, first_conn.cursor() as second_conn:
        plans = [_plan(conn, Model, index, actor=True) for index, conn in enumerate((first_conn, second_conn), 1)]
        runtime = LocalModelRuntime(
            session_id=plans[0].session_id(),
            session_config=plans[0].session_config(),
            request_limit=RequestAdmissionLimits(2, 1),
        )
        nodes = [plan.collect_udf_nodes(conn=conn)[0] for plan, conn in zip(plans, (first_conn, second_conn))]
        model = runtime.register("model", version="v1", payload=nodes[0]["payload"])
        first, second = runtime.request(), runtime.request()
        try:
            with ThreadPoolExecutor(max_workers=2) as threads:
                initializing = threads.submit(
                    first.execute,
                    plans[0],
                    {str(nodes[0]["node_id"]): "model"},
                    conn=first_conn,
                    execution_timeout=2 if timeout_initializer else None,
                )
                try:
                    _wait(lambda: (tmp_path / "initializations").exists())
                    waiting = threads.submit(
                        second.execute,
                        plans[1],
                        {str(nodes[1]["node_id"]): "model"},
                        conn=second_conn,
                        execution_timeout=None if timeout_initializer else 0.1,
                    )
                    timed = first if timeout_initializer else second
                    _wait(lambda: timed.cancellation_reason == "execution_timeout")
                    if timeout_initializer:
                        assert first.state == "cancelling"
                        assert not initializing.done() and not waiting.done()
                    else:
                        with pytest.raises(RequestExecutionTimeout):
                            waiting.result(timeout=5)
                        assert not initializing.done()
                finally:
                    (tmp_path / "release").touch()
                with pytest.raises(RequestExecutionTimeout):
                    (initializing if timeout_initializer else waiting).result(timeout=10)
                successful = waiting if timeout_initializer else initializing
                assert _values(successful.result(timeout=10)) == [2 if timeout_initializer else 1]
            pids = [int(value) for value in (tmp_path / "initializations").read_text().splitlines()]
            with model.acquire() as borrow:
                assert borrow.pool.worker_pids() == pids
            assert len(pids) == 1
            assert runtime.resource_snapshot()["request_admission"]["active_requests"] == 0
        finally:
            (tmp_path / "release").touch()
            runtime.close(timeout=15, kill=True)
