# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded remote query identities, cancellation and cleanup ownership."""

import threading

import pytest

from tests.fast.test_flight_server import TOKEN, call
from tests.fast.test_server_sessions import identity, wait_until
from vane.client import Client
from vane.execution.server_query import ServerQuery
from vane.execution.server_session import SessionError, SessionLimits
from vane.server import Server


@pytest.fixture
def server():
    value = Server(token=TOKEN, port=0, sessions=SessionLimits(max_query_handles=2, maintenance_interval=0.01))
    try:
        yield value
    finally:
        value.close()


def test_lost_execute_reply_deduplicates_and_retired_sequence_cannot_replay(server, monkeypatch):
    runs = []
    finish = threading.Event()

    def run(query, *args):
        runs.append(args)
        assert finish.wait(5)
        query.done.set()

    monkeypatch.setattr(ServerQuery, "run", run)
    handle = server.service.open_session()
    args = identity(handle)
    try:
        first = server.service.execute(*args, sequence=1, sql="select 42")
        assert server.service.execute(*args, sequence=1, sql="select 42")["query_id"] == first["query_id"]
        wait_until(lambda: len(runs) == 1)
        with pytest.raises(SessionError) as error:
            server.service.execute(*args, sequence=1, sql="select 7")
        assert error.value.code == "SUBMISSION_CONFLICT"
        with pytest.raises(SessionError) as error:
            server.service.execute(*args, sequence=3, sql="select 7")
        assert error.value.code == "SUBMISSION_ORDER"
        server.service.execute(*args, sequence=2, sql="select 7")
        with pytest.raises(SessionError) as error:
            server.service.execute(*args, sequence=3, sql="select 9")
        assert error.value.code == "QUERY_CAPACITY"
    finally:
        finish.set()
    wait_until(lambda: server.service.query_action(*args, 1, "close")["state"] == "CLOSED")
    with pytest.raises(SessionError) as error:
        server.service.execute(*args, sequence=1, sql="select 42")
    assert error.value.code == "QUERY_RETIRED"
    assert len(runs) == 2


def test_cancel_is_metadata_only_and_session_retains_blocked_query(server, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def run(query, *args):
        entered.set()
        assert release.wait(5)
        query.done.set()

    monkeypatch.setattr(ServerQuery, "run", run)
    handle = server.service.open_session()
    args = identity(handle)
    server.service.execute(*args, sequence=1, sql="select 42")
    assert entered.wait(2)
    try:
        server.service.query_action(*args, 1, "cancel")
        assert server.service.renew_session(*args)["state"] == "OPEN"
        with pytest.raises(TimeoutError):
            server.close(timeout=0.01)
        assert server.service.snapshot()["sessions"] == 1
        assert server.service.snapshot()["queries"] == 1
    finally:
        release.set()


def test_client_rejected_submission_does_not_poison_sequence(server):
    with Client(server.location, token=TOKEN) as client:
        with pytest.raises(Exception, match="SQL must"):
            client.submit("")
        assert client._sequence == 0
        query = client.submit("select 1")
        assert query.query_id == 1
        wait_until(lambda: query.status()["state"] == "FAILED")  # Ray deliberately absent.
        assert "ray.init" in query.status()["error"]["message"]
        query.close()


@pytest.mark.parametrize(
    "extra",
    [
        {"sequence": True},
        {"sequence": 0},
        {"sequence": 2**53},
        {"sql": ""},
        {"sql": "x" * 32769},
        {"rows_per_batch": 0},
        {"rows_per_batch": True},
    ],
)
def test_invalid_execute_has_no_owner(server, extra):
    import pyarrow as pa
    import pyarrow.flight as flight

    with flight.FlightClient(server.location) as client:
        opened = call(client, "vane.session.open")["result"]
        request = {name: opened[name] for name in ("server_id", "session_id")}
        with pytest.raises(pa.ArrowInvalid):
            call(client, "vane.query.execute", {**request, "sequence": 1, "sql": "select 1", **extra})
        assert server.service.snapshot()["queries"] == 0
