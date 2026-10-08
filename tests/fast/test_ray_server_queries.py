# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Remote SQL and native streams through an independently reachable gateway."""

import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import vane
from tests.fast.test_flight_server import TOKEN
from tests.fast.test_ray_recovery_runtime import resources
from tests.fast.test_server_sessions import wait_until
from vane.client import Client
from vane.execution.request_admission import RequestCancelled, RequestExecutionTimeout
from vane.execution.result_delivery import ResultDeliveryLimits, ResultDeliveryTimeout
from vane.execution.server_session import SessionLimits
from vane.server import Server

pytestmark = [pytest.mark.real_ray, pytest.mark.usefixtures("ray_local")]


@pytest.fixture
def server(tmp_path):
    value = Server(token=TOKEN, port=0, resources=resources(tmp_path), sessions=SessionLimits(lease_seconds=120))
    try:
        yield value
    finally:
        value.close(timeout=30)


def query_options(mode, *, execution=30, delivery=30):
    target = vane.RayExecution(mode, vane.FteOptions("shared", 2, 0) if mode == "fte" else None)
    return vane.QueryExecutionOptions(target, 30, execution, delivery)


def idle(server):
    snapshot = server.service.runtime.resource_snapshot()["service"]
    return snapshot["request_admission"]["active_requests"] == 0 and snapshot["result_service"]["active_contexts"] == 0


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
@pytest.mark.parametrize(
    "sql,expected",
    [
        ("select 42 as value", [{"value": 42}]),
        ("select range as value from range(0)", []),
        (
            "select range%3 as k, sum(range)::bigint as v from range(12) group by k order by k",
            [{"k": 0, "v": 18}, {"k": 1, "v": 22}, {"k": 2, "v": 26}],
        ),
        ("select [[1,2]::bigint[2],[3,4]::bigint[2]] as value", [{"value": [[1, 2], [3, 4]]}]),
    ],
)
def test_remote_sql_results_schema_and_retirement(server, mode, sql, expected, monkeypatch):
    from vane.execution.pipelined_runtime import PipelinedScheduler
    from vane.execution.recovery_runtime import RecoveryScheduler

    def no_python_forwarding(*args):
        raise AssertionError("server must not read or forward Arrow batches")

    monkeypatch.setattr(PipelinedScheduler, "read_next_batch", no_python_forwarding)
    monkeypatch.setattr(RecoveryScheduler, "read_next_batch", no_python_forwarding)
    with Client(server.location, token=TOKEN, execution=mode) as client:
        result = client.query(sql)
        table = result.collect()
        assert table.to_pylist() == expected
        assert len(table.schema) > 0
        assert result.state == "delivered"
        assert client.resource_snapshot()["active_results"] == 0
        wait_until(lambda: idle(server), timeout=10)
        assert client.query("select 7 as value").collect().to_pylist() == [{"value": 7}]
    assert server.service.snapshot()["sessions"] == 0


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_client_in_separate_process_never_connects_to_ray(server, mode):
    code = """
import json, sys, ray
from vane.client import Client
assert not ray.is_initialized()
with Client(sys.argv[1], token=sys.argv[2], execution=sys.argv[3]) as client:
    data = client.query('select sum(range)::bigint as value from range(1000)').collect()
    print(json.dumps(data.to_pylist()))
assert not ray.is_initialized()
"""
    response = subprocess.run(
        [sys.executable, "-I", "-c", code, server.location.uri.decode(), TOKEN, mode],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert response.returncode == 0, response.stderr
    assert json.loads(response.stdout) == [{"value": 499500}]
    assert idle(server)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_retained_client_views_are_charged_and_cancel_wakes_buffer_wait(server, mode):
    with Client(server.location, token=TOKEN, execution=mode, result_limits=ResultDeliveryLimits(2, 4096)) as client:
        query = client.submit("select range as value from range(10000)", rows_per_batch=64)
        result = query.result()
        batches = []
        with ThreadPoolExecutor(1) as threads:
            pending = None
            try:
                for _ in range(12):
                    pending = threads.submit(result.take)
                    wait_until(lambda: pending.done() or client.resource_snapshot()["waiting_byte_results"] > 0)
                    if not pending.done():
                        break
                    batches.append(pending.result())
                assert client.resource_snapshot()["waiting_byte_results"] == 1
                query.cancel()
                with pytest.raises(RequestCancelled):
                    pending.result(timeout=10)
                assert batches[0].column(0).to_pylist() == list(range(64))
                assert client.resource_snapshot()["exported_bytes"] > 0
            finally:
                batches.clear()
        result.close()
        wait_until(lambda: client.resource_snapshot()["usage_bytes"] == 0)
        wait_until(lambda: idle(server), timeout=10)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_delivery_timeout_preserves_exception_type(server, mode):
    with Client(server.location, token=TOKEN, execution=mode) as client:
        client.query("select 1").collect()  # Warm the shared actors.
        result = client.query("select range from range(10000)", options=query_options(mode, delivery=1))
        time.sleep(1.2)
        with pytest.raises(ResultDeliveryTimeout):
            result.take()
        result.close()
        wait_until(lambda: idle(server), timeout=10)


@pytest.mark.parametrize("mode", ["pipelined", "fte"])
def test_execution_timeout_preserves_exception_type(server, mode):
    with Client(server.location, token=TOKEN, execution=mode) as client:
        client.query("select 1").collect()
        with pytest.raises(RequestExecutionTimeout):
            client.query(
                "select sum(range % 97) from range(1000000000)", options=query_options(mode, execution=0.2)
            ).collect()
        wait_until(lambda: idle(server), timeout=10)


def test_lost_execute_reply_keeps_single_owner(server, monkeypatch):
    import pyarrow.flight as flight

    with Client(server.location, token=TOKEN) as client:
        original = client._call
        lost = []

        def call(operation, **fields):
            reply = original(operation, **fields)
            if operation == "query.execute" and not lost:
                lost.append(fields["sequence"])
                raise flight.FlightUnavailableError("injected lost acknowledgement")
            return reply

        monkeypatch.setattr(client, "_call", call)
        with pytest.raises(flight.FlightUnavailableError):
            client.submit("select 42")
        first = client._queries[1]
        second = client.submit("select 7")  # Reconfirms sequence 1 before 2.
        assert first.result().collect().column(0).to_pylist() == [42]
        assert second.result().collect().column(0).to_pylist() == [7]
        assert server.service.snapshot()["queries"] == 0
        assert idle(server)


def test_result_gateway_uses_tls_and_configured_public_address(tmp_path):
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("OpenSSL CLI required for temporary test certificates")
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
            "subjectAltName=DNS:localhost",
        ],
        capture_output=True,
        check=True,
        timeout=20,
    )
    with Server(
        token=TOKEN,
        host="0.0.0.0",
        advertise_host="localhost",
        port=0,
        resources=resources(tmp_path),
        tls_certificates=[(cert.read_bytes(), key.read_bytes())],
    ) as server:
        assert server._gateway.location.startswith("grpc+tls://localhost:")
        with Client(server.location, token=TOKEN, tls_root_certs=cert.read_bytes()) as client:
            query = client.submit("select 42")
            wait_until(lambda: query.status()["state"] != "PREPARING", timeout=30)
            assert query.status()["result"]["location"] == server._gateway.location
            assert query.result().collect().column(0).to_pylist() == [42]


def test_successful_delivery_keeps_ownership_until_cleanup_retry(server, monkeypatch):
    import threading

    from vane.execution.result_consumer import NativeResultConsumer

    entered, proceed = threading.Event(), threading.Event()
    original = NativeResultConsumer.close

    def close(consumer):
        if consumer.gateway is not None and not proceed.is_set():
            entered.set()
            raise RuntimeError("injected gateway cleanup failure")
        return original(consumer)

    monkeypatch.setattr(NativeResultConsumer, "close", close)
    with Client(server.location, token=TOKEN) as client, ThreadPoolExecutor(1) as threads:
        query = client.submit("select 42")
        result = query.result()
        collecting = threads.submit(result.collect)
        try:
            assert entered.wait(10)
            status = query.status()
            assert status["state"] == "SUCCEEDED" and not status["cleaned"]
            assert not idle(server)
        finally:
            proceed.set()
        assert collecting.result(timeout=10).column(0).to_pylist() == [42]
        assert idle(server)
