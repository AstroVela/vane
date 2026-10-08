# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Remote failure boundaries with real clients, native streams and Ray owners."""

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pyarrow.flight as flight
import pytest
import ray

from tests.fast.test_flight_server import TOKEN
from tests.fast.test_ray_recovery_runtime import resources
from tests.fast.test_ray_server_queries import query_options
from tests.fast.test_server_sessions import wait_until
from vane.client import Client
from vane.execution.server_session import SessionLimits
from vane.server import Server

pytestmark = [pytest.mark.real_ray, pytest.mark.usefixtures("ray_local"), pytest.mark.timeout(90)]


@pytest.fixture
def server(tmp_path):
    value = Server(
        token=TOKEN,
        port=0,
        resources=resources(tmp_path, max_active_queries=2),
        sessions=SessionLimits(lease_seconds=3, cleanup_timeout=0.5, maintenance_interval=0.02),
    )
    try:
        yield value
    finally:
        value.close(timeout=30)


def assert_idle(server, *, dead_workers=False):
    """Both coordinator and actual worker ledgers, not just session counts."""
    core = server.service.runtime._service
    state = core.snapshot()
    assert server.service.snapshot()["queries"] == 0
    assert server._gateway.active_links == 0
    assert state["request_admission"]["active_requests"] == 0
    assert state["request_admission"]["queued_requests"] == 0
    for key in ("active_results", "usage_bytes", "buffers", "cleanup_pending_results"):
        assert state["result_delivery"][key] == 0
    assert state["result_service"]["active_contexts"] == 0
    assert state["workers"]["reservations"] == {}
    assert state["workers"]["waiting"] == []
    for session in state["sessions"].values():
        assert session["queries"] == {}
    for worker in core.pool.workers:
        try:
            assert ray.get(worker.resources_snapshot.remote(), timeout=5)["reservations"] == {}
        except ray.exceptions.ActorDiedError:
            if not dead_workers:
                raise
    for store in core.stores.values():
        assert store.snapshot() == {"queries": 0, "reserved_bytes": 0}


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
@pytest.mark.parametrize("phase", ["queued", "streaming"])
def test_killed_client_retires_leases_and_preserves_other_session(server, mode, phase, tmp_path):
    code = """
import json, sys, threading, ray
from vane.client import Client
assert not ray.is_initialized()
client = Client(sys.argv[1], token=sys.argv[2], execution=sys.argv[3])
query = client.submit('select range from range(10000)', rows_per_batch=64)
if sys.argv[4] == 'streaming':
    result = query.result()
    batch = result.read_batch()
    assert batch.num_rows > 0
print(json.dumps(client.identity), flush=True)
threading.Event().wait()
"""
    blocker = None
    with Client(server.location, token=TOKEN) as survivor:
        survivor.query("select 1").collect()
        epochs = tuple(server.service.runtime._service.pool.epochs)
        if phase == "queued":
            # Occupy every service admission slot without buffering the input.
            capacity = server.service.runtime.resources.max_active_queries
            blocker = [
                survivor.query(
                    "select range from range(1000000)", options=query_options("pipelined", execution=60, delivery=60)
                )
                for _ in range(capacity)
            ]
        with (tmp_path / "client.stderr").open("w+") as errors:
            process = subprocess.Popen(
                [sys.executable, "-I", "-c", code, server.location.uri.decode(), TOKEN, mode, phase],
                stdout=subprocess.PIPE,
                stderr=errors,
                text=True,
            )
            try:
                with ThreadPoolExecutor(1) as threads:
                    ready = threads.submit(process.stdout.readline)
                    try:
                        handle = json.loads(ready.result(timeout=30))
                    except BaseException:
                        process.kill()
                        errors.seek(0)
                        pytest.fail(errors.read() or "client did not report readiness")
                if phase == "queued":
                    wait_until(lambda: server.service.runtime._service.admission.snapshot()["queued_requests"] == 1)
                process.kill()
                process.wait(timeout=5)
                # The actual lease timer must retire the abandoned owner.
                wait_until(lambda: handle["session_id"] not in server.service._sessions, timeout=15)
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=5)
                if blocker is not None:
                    for result in blocker:
                        result.close()
        assert_idle(server)
        assert survivor.query("select 7").collect().column(0).to_pylist() == [7]
        assert tuple(server.service.runtime._service.pool.epochs) == epochs
        assert_idle(server)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_lost_execute_ack_and_stopped_heartbeat_reclaim_owned_submission(server, mode, monkeypatch):
    with Client(server.location, token=TOKEN) as survivor, Client(server.location, token=TOKEN, execution=mode) as lost:
        survivor.query("select 1").collect()
        original = lost._call

        def call(operation, **fields):
            reply = original(operation, **fields)
            if operation == "query.execute":
                raise flight.FlightUnavailableError("injected lost Execute reply")
            return reply

        monkeypatch.setattr(lost, "_call", call)
        with pytest.raises(flight.FlightUnavailableError):
            lost.submit("select range from range(10000)")
        session = server.service._sessions[lost.identity["session_id"]]
        assert len(session.queries) == 1
        lost._stop.set()
        lost._heartbeat.join(5)
        wait_until(lambda: lost.identity["session_id"] not in server.service._sessions, timeout=15)
        assert next(iter(session.queries.values())).done.is_set()
        assert_idle(server)
        assert survivor.query("select 7").collect().column(0).to_pylist() == [7]


@pytest.mark.ray_fault
def test_result_actor_loss_fails_resident_remote_queries_without_replacement(server):
    with Client(server.location, token=TOKEN) as a, Client(server.location, token=TOKEN, execution="fte") as b:
        queries = [client.submit("select range from range(10000)") for client in (a, b)]
        results = [query.result() for query in queries]
        records = [
            server.service._sessions[query.client.identity["session_id"]].queries[query.query_id] for query in queries
        ]
        core = server.service.runtime._service
        actor = core.pool.results.actor
        ray.kill(actor, no_restart=True)
        for record, result in zip(records, results):
            # The client's failure monitor can already have retired the handle.
            wait_until(lambda: record.snapshot()["state"] == "FAILED", timeout=15)
            with pytest.raises(RuntimeError):
                result.collect()
            result.close()
        assert_idle(server)
        with pytest.raises(RuntimeError):
            a.query("select 7")
        assert core.pool.results.actor is actor
        assert_idle(server)


@pytest.mark.ray_fault
def test_worker_loss_after_partial_remote_delivery_is_an_error(server):
    with Client(server.location, token=TOKEN) as client:
        query = client.submit("select range from range(1000000)")
        result = query.result()
        batch = result.read_batch()
        assert batch.num_rows > 0
        del batch
        record = server.service._sessions[client.identity["session_id"]].queries[query.query_id]
        ray.kill(server.service.runtime._service.pool.workers[0], no_restart=True)
        wait_until(lambda: record.snapshot()["state"] == "FAILED", timeout=15)
        with pytest.raises(RuntimeError):
            result.collect()
        result.close()
        assert_idle(server, dead_workers=True)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_failed_result_release_receipt_keeps_remote_cleanup_retryable(server, mode, monkeypatch):
    with Client(server.location, token=TOKEN, execution=mode) as client:
        query = client.submit("select range from range(10000)")
        result = query.result()
        record = server.service._sessions[client.identity["session_id"]].queries[query.query_id]
        core = server.service.runtime._service
        actor = core.pool.results.actor
        original = actor.release.remote
        attempts = []

        def release(*args):
            ray.get(original(*args), timeout=5)
            attempts.append(args)
            return ray.put(ray.exceptions.ActorUnavailableError("lost result release reply", None))

        with monkeypatch.context() as outage:
            outage.setattr(actor, "release", SimpleNamespace(remote=release))
            query.cancel()
            wait_until(lambda: len(attempts) >= 2, timeout=10)
            assert not record.done.is_set()
            assert core.pool.results.snapshot()["active_contexts"] == 1
            assert core.admission.snapshot()["active_requests"] == 1
        wait_until(record.done.is_set, timeout=15)
        result.close()
        assert_idle(server)
        assert client.query("select 7").collect().column(0).to_pylist() == [7]
        assert core.pool.results.actor is actor


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
@pytest.mark.parametrize("receipt", [False, True])
def test_worker_outage_retains_remote_session_until_fresh_cleanup(server, mode, receipt, monkeypatch):
    from vane.execution.recovery_runtime import RecoveryScheduler

    dispatched = threading.Event()
    original_dispatch = RecoveryScheduler._dispatch

    def hold_attempt(owner, *args):
        accepted = original_dispatch(owner, *args)
        if accepted:
            ray.get(next(iter(owner.active.values())).prepare, timeout=10)
            dispatched.set()
            assert owner.stop.wait(20), "test did not cancel the prepared FTE attempt"
        return accepted

    with Client(server.location, token=TOKEN) as survivor, Client(server.location, token=TOKEN, execution=mode) as lost:
        survivor.query("select 1").collect()
        if mode == "fte":
            monkeypatch.setattr(RecoveryScheduler, "_dispatch", hold_attempt)
        query = lost.submit("select range from range(1000000)" if mode == "pipelined" else "select 42")
        result = query.result()
        if mode == "fte":
            assert dispatched.wait(10)
        session = server.service._sessions[lost.identity["session_id"]]
        record = session.queries[query.query_id]
        scheduler = record.context._reader
        worker = session.owner.pool.workers[0] if mode == "pipelined" else next(iter(scheduler.active.values())).worker
        method = "release" if mode == "pipelined" else "release_materialized"
        original_release = getattr(worker, method).remote
        attempts = []

        def unavailable(*args):
            attempts.append(args)
            if receipt:
                ray.get(original_release(*args), timeout=5)
                return ray.put(ray.exceptions.ActorUnavailableError("lost worker release reply", None))
            raise ray.exceptions.ActorUnavailableError("temporary worker outage", None)

        with monkeypatch.context() as outage:
            outage.setattr(worker, method, SimpleNamespace(remote=unavailable))
            lost._call("session.close", **lost.identity)
            wait_until(lambda: len(attempts) >= 2, timeout=10)
            assert not record.done.is_set()
            assert session.closing and session.session_id in server.service._sessions
            assert session.owner.pool.admission.snapshot()["reservations"]
            assert server.service.runtime._service.admission.snapshot()["active_requests"] == 1
            if not receipt:
                assert ray.get(worker.resources_snapshot.remote(), timeout=5)["reservations"]
            # Unrelated sessions continue to renew despite cleanup failures.
            survivor._call("session.renew", **survivor.identity)
        monkeypatch.setattr(RecoveryScheduler, "_dispatch", original_dispatch)
        wait_until(lambda: session.session_id not in server.service._sessions, timeout=15)
        result.close()
        assert_idle(server)
        assert survivor.query("select 7").collect().column(0).to_pylist() == [7]


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_slow_remote_client_has_bounded_native_windows_and_does_not_block_other_session(server, mode):
    with Client(server.location, token=TOKEN, execution=mode) as slow, Client(server.location, token=TOKEN) as fast:
        query = slow.submit("select range from range(10000)", options=query_options(mode), rows_per_batch=64)
        result = query.result()
        batch = result.read_batch()
        record = server.service._sessions[slow.identity["session_id"]].queries[query.query_id]
        limits = server.service.runtime.resources.exchange
        for _ in range(3):
            time.sleep(0.05)
            for channel in (record.consumer.channel, query.reader.channel):
                state = channel.snapshot()
                assert state["bytes"] <= limits.window_bytes
                assert state["peak_bytes"] <= limits.window_bytes
                assert state["frames"] <= limits.frame_slots
            assert server._gateway.active_links <= server.service.runtime.resources.max_results
        assert fast.query("select 7").collect().column(0).to_pylist() == [7]
        query.cancel()
        result.close()
        assert batch.column(0).to_pylist()  # Caller-owned views survive remote cleanup.
        assert slow.resource_snapshot()["exported_bytes"] > 0
        del batch
        wait_until(lambda: slow.resource_snapshot()["usage_bytes"] == 0)
        assert_idle(server)
