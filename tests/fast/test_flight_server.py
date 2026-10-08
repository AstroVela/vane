# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real Flight RPCs for authentication, session control and shutdown."""

import json
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.flight as flight
import pytest

from tests.fast.test_server_sessions import identity, wait_until
from vane.execution.server_session import SessionLimits
from vane.server import Server

TOKEN = "session-test-credential-0123456789abcdef"


def options(token=TOKEN, timeout=3):
    return flight.FlightCallOptions(timeout=timeout, headers=[(b"authorization", ("Bearer " + token).encode())])


def call(client, action, payload=None):
    body = json.dumps({"protocol": 1, **(payload or {})}).encode()
    response = list(client.do_action(flight.Action(action, body), options()))
    assert len(response) == 1
    return json.loads(response[0].body.to_pybytes())


@pytest.fixture
def server():
    server = Server(token=TOKEN, port=0, sessions=SessionLimits(max_sessions=2, maintenance_interval=0.01))
    try:
        yield server
    finally:
        server.close()


def test_wire_session_lifecycle_and_server_capabilities(server):
    with flight.FlightClient(server.location) as client:
        info = call(client, "vane.info")["result"]
        assert info["state"] == "READY"
        assert info["capabilities"] == ["sessions", "queries", "native-results"]
        opened = call(client, "vane.session.open")
        assert opened["ok"]
        handle = {key: opened["result"][key] for key in ("server_id", "session_id")}
        assert call(client, "vane.session.renew", handle) == opened
        assert call(client, "vane.session.close", handle)["result"] == {"state": "CLOSING"}
        wait_until(lambda: call(client, "vane.session.close", handle)["result"]["state"] == "CLOSED")
        assert call(client, "vane.session.renew", handle)["error"]["code"] == "SESSION_EXPIRED"
        assert call(client, "vane.info")["result"]["sessions"] == 0


@pytest.mark.parametrize("action", ["vane.info", "vane.session.open", "vane.session.close"])
@pytest.mark.parametrize("authenticated", [False, True])
def test_invalid_credentials_are_rejected_before_dispatch(server, action, authenticated):
    with flight.FlightClient(server.location) as client:
        with pytest.raises(flight.FlightUnauthenticatedError):
            list(
                client.do_action(
                    flight.Action(action, b'{"protocol":1}'), options("incorrect") if authenticated else None
                )
            )
    assert server.service.snapshot()["sessions"] == 0


def test_authentication_applies_to_other_flight_methods(server):
    with flight.FlightClient(server.location) as client:
        with pytest.raises(flight.FlightUnauthenticatedError):
            list(client.list_actions())
        with pytest.raises(flight.FlightUnauthenticatedError):
            client.do_get(flight.Ticket(b"unknown"))
        assert {action.type for action in client.list_actions(options())} == {
            "vane.query.execute",
            "vane.query.status",
            "vane.query.cancel",
            "vane.query.finish",
            "vane.query.close",
            "vane.info",
            "vane.session.open",
            "vane.session.renew",
            "vane.session.close",
        }


@pytest.mark.parametrize(
    "body",
    [
        b"{}",
        b"[]",
        b'{"protocol":true}',
        b'{"protocol":2}',
        b'{"protocol":1,"protocol":1}',
        b'{"protocol":1,"extra":true}',
        b'{"protocol":1,"resources":NaN}',
        b'{"protocol":1,"resources":{"worker_count":5}}',
        b'{"protocol":1,"execution":"local"}',
        b'{"protocol":1,"resources":{"max_active_queries":true}}',
        b" " * 65537,
    ],
)
def test_invalid_open_request_has_no_session_side_effect(server, body):
    with flight.FlightClient(server.location) as client:
        with pytest.raises(pa.ArrowInvalid):
            list(client.do_action(flight.Action("vane.session.open", body), options()))
    assert server.service.snapshot()["sessions"] == 0


def test_capacity_and_server_epoch_errors_are_structured(server):
    with flight.FlightClient(server.location) as client:
        handles = [call(client, "vane.session.open")["result"] for _ in range(2)]
        assert call(client, "vane.session.open")["error"]["code"] == "SESSION_CAPACITY"
        assert (
            call(client, "vane.session.close", {"server_id": "old", "session_id": handles[0]["session_id"]})["error"][
                "code"
            ]
            == "SERVER_CHANGED"
        )
        assert server.service.snapshot()["sessions"] == 2


def test_lost_open_response_is_cleaned_after_rpc_deadline(server, monkeypatch):
    original = server.service.runtime.connect
    entered, proceed = threading.Event(), threading.Event()

    def connect(*args, **kwargs):
        entered.set()
        assert proceed.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(server.service.runtime, "connect", connect)
    with flight.FlightClient(server.location) as client, ThreadPoolExecutor(1) as threads:
        pending = threads.submit(
            lambda: list(client.do_action(flight.Action("vane.session.open", b'{"protocol":1}'), options(timeout=0.1)))
        )
        try:
            assert entered.wait(5)
            with pytest.raises(flight.FlightTimedOutError):
                pending.result(timeout=3)
        finally:
            proceed.set()
    wait_until(lambda: server.service.snapshot()["sessions"] == 0)
    assert server.service.runtime.resource_snapshot()["service"]["sessions"] == {}


def test_shutdown_wait_is_bounded_and_retries_the_same_attempt(server, monkeypatch):
    original = server._flight.shutdown
    entered, proceed = threading.Event(), threading.Event()
    calls = []

    def shutdown():
        calls.append(1)
        entered.set()
        assert proceed.wait(5)
        original()

    monkeypatch.setattr(server._flight, "shutdown", shutdown)
    try:
        with pytest.raises(TimeoutError, match="Flight shutdown is pending"):
            server.close(timeout=0.1)
        assert entered.wait(1)
        with pytest.raises(TimeoutError, match="Flight shutdown is pending"):
            server.close(timeout=0.01)
        assert len(calls) == 1
    finally:
        proceed.set()
    server.close()
    server.close()
    assert len(calls) == 1


def test_slow_session_close_does_not_block_other_control_rpcs(server, monkeypatch):
    first = server.service.open_session()
    second = server.service.open_session()
    owner = server.service._sessions[first["session_id"]].owner
    original = owner.close_session
    entered, proceed = threading.Event(), threading.Event()

    def close(*, timeout):
        entered.set()
        assert proceed.wait(5)
        original(timeout=timeout)

    monkeypatch.setattr(owner, "close_session", close)
    try:
        server.service.close_session(*identity(first))
        assert entered.wait(5)
        with flight.FlightClient(server.location) as client:
            handle = {key: second[key] for key in ("server_id", "session_id")}
            assert call(client, "vane.session.renew", handle)["ok"]
            assert call(client, "vane.info")["result"]["sessions"] == 2
    finally:
        proceed.set()


def test_failed_flight_shutdown_can_be_retried(server, monkeypatch):
    original = server._flight.shutdown
    calls = []

    def shutdown():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("injected shutdown failure")
        original()

    monkeypatch.setattr(server._flight, "shutdown", shutdown)
    with pytest.raises(RuntimeError, match="Flight shutdown failed"):
        server.close()
    server.close()
    assert len(calls) == 2


@pytest.mark.parametrize("kwargs", [{"token": "short"}, {"port": True}, {"port": -1}, {"host": "0.0.0.0"}])
def test_invalid_listener_configuration(kwargs):
    with pytest.raises(ValueError):
        Server(**{"token": TOKEN, "port": 0, **kwargs})


def test_tls_listener_validates_server_certificate(tmp_path):
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("OpenSSL command is required to generate a temporary test certificate")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=IP:127.0.0.1,DNS:localhost",
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    with Server(token=TOKEN, port=0, tls_certificates=[(cert.read_bytes(), key.read_bytes())]) as server:
        assert server.location.uri.startswith(b"grpc+tls:")
        with flight.FlightClient(server.location, tls_root_certs=cert.read_bytes()) as client:
            assert call(client, "vane.session.open")["ok"]
