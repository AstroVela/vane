# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Leased session ownership, independently of the network transport."""

import gc
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import vane
from vane.execution.server_session import SessionError, SessionLimits, SessionService


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.005)


@pytest.fixture
def service():
    value = SessionService(limits=SessionLimits(max_sessions=2, maintenance_interval=0.01))
    try:
        yield value
    finally:
        value.close()


def identity(handle):
    return handle["server_id"], handle["session_id"]


def test_native_sessions_share_runtime_and_retire_without_starting_ray(service):
    first, second = service.open_session(), service.open_session()
    state = service.runtime.resource_snapshot()["service"]
    assert len(state["sessions"]) == 2
    assert state["result_service"]["started"] is False
    assert service.renew_session(*identity(first)) == first
    assert service.close_session(*identity(first)) == {"state": "CLOSING"}
    wait_until(lambda: service.snapshot()["sessions"] == 1)
    assert service.close_session(*identity(first)) == {"state": "CLOSED"}
    assert service.renew_session(*identity(second)) == second
    assert len(service.runtime.resource_snapshot()["service"]["sessions"]) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_sessions", True),
        ("max_sessions", 0),
        ("max_sessions", 1025),
        ("lease_seconds", float("nan")),
        ("lease_seconds", 0),
        ("cleanup_timeout", -1),
        ("maintenance_interval", float("inf")),
    ],
)
def test_invalid_limits(field, value):
    with pytest.raises(ValueError):
        SessionLimits(**{field: value})


def test_capacity_includes_cleanup_pending_and_retry_does_not_overlap(service, monkeypatch):
    first, second = service.open_session(), service.open_session()
    owner = service._sessions[first["session_id"]].owner
    original = owner.close_session
    entered, proceed = threading.Event(), threading.Event()
    attempts = []

    def close(*, timeout):
        attempts.append(timeout)
        entered.set()
        assert proceed.wait(5)
        original(timeout=timeout)

    monkeypatch.setattr(owner, "close_session", close)
    try:
        service.close_session(*identity(first))
        assert entered.wait(5)
        for _ in range(5):
            assert service.close_session(*identity(first))["state"] == "CLOSING"
            assert service.renew_session(*identity(second)) == second
        with pytest.raises(SessionError) as caught:
            service.open_session()
        assert caught.value.code == "SESSION_CAPACITY"
        assert len(attempts) == 1
        assert len(service.runtime.resource_snapshot()["service"]["sessions"]) == 2
    finally:
        proceed.set()
    wait_until(lambda: service.snapshot()["sessions"] == 1)
    service.open_session()


def test_failed_cleanup_is_retried_with_the_same_native_owner(service, monkeypatch):
    handle = service.open_session()
    owner = service._sessions[handle["session_id"]].owner
    original = owner.close_session
    fail = threading.Event()
    fail.set()

    def close(*, timeout):
        if fail.is_set():
            raise RuntimeError("temporary cleanup failure")
        original(timeout=timeout)

    monkeypatch.setattr(owner, "close_session", close)
    try:
        service.close_session(*identity(handle))
        wait_until(lambda: service.snapshot()["cleanup_pending"] == 1)
        assert service._sessions[handle["session_id"]].owner is owner
        assert service.close_session(*identity(handle))["state"] == "CLOSING"
    finally:
        fail.clear()
    wait_until(lambda: service.snapshot()["sessions"] == 0)
    assert service.runtime.resource_snapshot()["service"]["sessions"] == {}


def test_cleanup_retry_can_confirm_a_close_whose_acknowledgement_failed(service, monkeypatch):
    handle = service.open_session()
    owner = service._sessions[handle["session_id"]].owner
    original = owner.close_session
    calls = []

    def close(*, timeout):
        original(timeout=timeout)
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("close acknowledgement failed")

    monkeypatch.setattr(owner, "close_session", close)
    service.close_session(*identity(handle))
    wait_until(lambda: service.snapshot()["sessions"] == 0)
    assert len(calls) == 2


def test_expiry_cannot_be_revived_even_before_maintenance_observes_it(service):
    handle = service.open_session()
    with service._condition:
        service._sessions[handle["session_id"]].expires_at = time.monotonic() - 1
        with pytest.raises(SessionError) as caught:
            service.renew_session(*identity(handle))
        assert caught.value.code == "SESSION_EXPIRED"
    wait_until(lambda: service.snapshot()["sessions"] == 0)
    with pytest.raises(SessionError, match="unknown or closing"):
        service.renew_session(*identity(handle))


def test_abandoned_session_expires_without_client_activity():
    service = SessionService(limits=SessionLimits(lease_seconds=1, maintenance_interval=0.01))
    try:
        service.open_session()
        wait_until(lambda: service.snapshot()["sessions"] == 0)
        assert service.runtime.resource_snapshot()["service"]["sessions"] == {}
    finally:
        service.close()


@pytest.mark.parametrize("operation", ["renew_session", "close_session"])
def test_old_server_instance_is_rejected(service, operation):
    handle = service.open_session()
    with pytest.raises(SessionError) as caught:
        getattr(service, operation)("old-server", handle["session_id"])
    assert caught.value.code == "SERVER_CHANGED"
    assert service.renew_session(*identity(handle)) == handle


def test_close_deadline_covers_blocked_native_cleanup_and_retains_owner(service, monkeypatch):
    handle = service.open_session()
    owner = service._sessions[handle["session_id"]].owner
    original = owner.close_session
    entered, proceed = threading.Event(), threading.Event()

    def close(*, timeout):
        entered.set()
        assert proceed.wait(5)
        original(timeout=timeout)

    monkeypatch.setattr(owner, "close_session", close)
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="session cleanup is pending"):
            service.close(timeout=0.1)
        assert time.monotonic() - started < 1
        assert entered.wait(1)
        assert service.snapshot()["state"] == "DRAINING"
        assert service.snapshot()["sessions"] == 1
        with pytest.raises(SessionError, match="server is closing"):
            service.open_session()
    finally:
        proceed.set()
    service.close()
    assert service.snapshot()["state"] == "CLOSED"


def test_server_close_racing_connection_creation_keeps_cleanup_owner(service, monkeypatch):
    original = service.runtime.connect
    entered, proceed = threading.Event(), threading.Event()

    def connect(*args, **kwargs):
        entered.set()
        assert proceed.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(service.runtime, "connect", connect)
    with ThreadPoolExecutor(1) as threads:
        pending = threads.submit(service.open_session)
        try:
            assert entered.wait(5)
            with pytest.raises(TimeoutError):
                service.close(timeout=0.01)
            assert service.snapshot()["opening"] == 1
        finally:
            proceed.set()
        with pytest.raises(SessionError, match="expired while opening"):
            pending.result(timeout=5)
    service.close()
    assert service.runtime.resource_snapshot()["service"]["sessions"] == {}


def test_close_deadline_includes_registry_lock_wait(service):
    with ThreadPoolExecutor(1) as threads:
        with service._condition:
            pending = threads.submit(service.close, timeout=0.01)
            with pytest.raises(TimeoutError, match="session cleanup is pending"):
                pending.result(timeout=1)
        service.close()


def test_failed_native_open_releases_session_capacity():
    service = SessionService(config={"not_a_setting": "value"}, limits=SessionLimits(max_sessions=1))
    try:
        for _ in range(3):
            with pytest.raises(Exception, match="not_a_setting"):
                service.open_session()
            wait_until(lambda: service.snapshot()["sessions"] == 0)
            assert service.runtime.resource_snapshot()["service"]["sessions"] == {}
    finally:
        service.close()


def test_native_open_rollback_failure_keeps_owner_and_capacity(monkeypatch):
    from vane.execution.pipelined_runtime import RayQueryRuntime

    service = SessionService(config={"not_a_setting": "value"}, limits=SessionLimits(max_sessions=1))
    original = RayQueryRuntime.close
    fail = threading.Event()
    fail.set()

    def close(self, *, timeout=5):
        if fail.is_set():
            raise RuntimeError("native open rollback failed")
        original(self, timeout=timeout)

    monkeypatch.setattr(RayQueryRuntime, "close", close)
    try:
        with pytest.raises(RuntimeError, match="native open rollback failed"):
            service.open_session()
        wait_until(lambda: service.snapshot()["cleanup_pending"] == 1)
        assert len(service.runtime.resource_snapshot()["service"]["sessions"]) == 1
        with pytest.raises(SessionError, match="capacity is full"):
            service.open_session()
        fail.clear()
        wait_until(lambda: service.snapshot()["sessions"] == 0)
        assert service.runtime.resource_snapshot()["service"]["sessions"] == {}
    finally:
        fail.clear()
        service.close()


def test_session_close_releases_all_native_cursors_and_database_lock(tmp_path):
    database = str(tmp_path / "server.db")
    service = SessionService(database=database)
    try:
        handle = service.open_session()
        owner = service._sessions[handle["session_id"]].owner
        nested = owner._connection.cursor().cursor()
        gc.collect()
        service.close()
        assert service.snapshot()["state"] == "CLOSED"
        opened = subprocess.run(
            [sys.executable, "-I", "-c", "import vane,sys; vane.connect(sys.argv[1]).close()", database],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert opened.returncode == 0, opened.stderr
        with pytest.raises(vane.ConnectionException, match="clos(ed|ing)"):
            nested.query("select 42")
    finally:
        service.close()
