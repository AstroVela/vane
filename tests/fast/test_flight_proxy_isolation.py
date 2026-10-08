# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Flight connects to explicit endpoints without inheriting HTTP proxy routing.

Run probes in fresh interpreters: gRPC can cache environment/channel state. The
controlled proxy only forwards to explicitly registered loopback test listeners.
"""

import json
import os
import select
import socket
import socketserver
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest


class Proxy(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), Tunnel)
        self.ports = set()
        self.connections = []
        self.paused = threading.Event()
        self.stopped = threading.Event()


class Tunnel(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(2)
        try:
            header = b""
            while b"\r\n\r\n" not in header:
                data = self.request.recv(4096)
                if not data or len(header) + len(data) > 16384:
                    return
                header += data
            headers, remainder = header.split(b"\r\n\r\n", 1)
            method, authority, _ = headers.split(b"\r\n", 1)[0].decode("ascii").split()
            host, port = authority.rsplit(":", 1)
            if method != "CONNECT" or host not in {"127.0.0.1", "localhost"} or int(port) not in self.server.ports:
                self.request.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
                return
            with socket.create_connection((host, int(port)), timeout=2) as upstream:
                self.server.connections.append(authority)
                self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                if remainder:
                    upstream.sendall(remainder)
                self.request.settimeout(0.2)
                upstream.settimeout(0.2)
                peers = {self.request: upstream, upstream: self.request}
                while not self.server.stopped.is_set():
                    if self.server.paused.is_set():
                        self.server.stopped.wait(0.01)
                        continue
                    readable, _, _ = select.select(list(peers), [], [], 0.05)
                    if self.server.paused.is_set():
                        continue
                    for source in readable:
                        data = source.recv(65536)
                        if not data:
                            return
                        peers[source].sendall(data)
        except (OSError, ValueError):
            # Closing a query or timing out a stalled RPC closes its sockets.
            return


@contextmanager
def proxy_server():
    with Proxy() as proxy:
        thread = threading.Thread(target=proxy.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            yield proxy
        finally:
            proxy.stopped.set()
            proxy.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


def wait_for(operation, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while True:
        value = operation()
        if predicate(value):
            return value
        assert time.monotonic() < deadline, value
        time.sleep(0.005)


def native_probe(proxy):
    import vane
    from vane._native import execution_runtime as native
    from vane.execution.compiler import compile_fragment_graph

    with vane.connect() as connection:
        graph = compile_fragment_graph(connection, "select 1::bigint", query_id="proxy-probe")
        root = next(f for f in graph.fragments if f.fragment_id == graph.result.fragment_id)
        schema = root.outputs[0].schema

    def channel():
        value = native.DirectChannel(schema, native.DirectLimits(128, 128, 3, 1), 1, ["consumer"])
        value.add_producer("producer")
        value.seal_producers()
        return value

    def service():
        return native.DirectFlight("127.0.0.1", "127.0.0.1", 1, native.DirectFlight.staging_per_link(128), 128)

    source, target = channel(), channel()
    sender, receiver = service(), service()
    proxy.ports.add(int(sender.location.rsplit(":", 1)[1]))
    try:
        sender.publish("test-capability", source, "consumer")
        receiver.subscribe(sender.location, "test-capability", target, "producer", 20)
        wait_for(lambda: receiver.ready, bool)
        proxy.paused.set()
        started = time.monotonic()
        # No data is produced. Query/data timeout is 20 s; control timeout 2 s.
        # Pausing an inherited proxy used to abort this otherwise healthy link.
        while time.monotonic() - started < 2.5 and not receiver.error:
            time.sleep(0.01)
        error = receiver.error
        elapsed = time.monotonic() - started
        proxy.paused.clear()
        if not error:
            assert source._write_rows("producer", 1, [(42,)]) == "accepted"
            source.finish("producer", 1)
            _, batch = wait_for(lambda: target.poll("consumer"), lambda value: value[0] == "data")
            assert batch.to_rows() == [(42,)]
            batch.close()
            wait_for(lambda: target.poll("consumer"), lambda value: value[0] == "end")
            assert sender.delivered("test-capability")
        return {"error": error, "pause_seconds": elapsed, "initial_handshake": True}
    finally:
        proxy.paused.clear()
        receiver.close()
        sender.close()


def control_probe(proxy, raw):
    import pyarrow.flight as flight

    from vane.client import Client
    from vane.server import Server

    token = "flight-proxy-test-token-0123456789abcdef"
    with Server(token=token, port=0) as server:
        proxy.ports.add(server.port)
        if raw:
            client = flight.FlightClient(server.location)

            def call(timeout):
                values = list(
                    client.do_action(
                        flight.Action("vane.info", b'{"protocol":1}'),
                        flight.FlightCallOptions(
                            timeout=timeout, headers=[(b"authorization", ("Bearer " + token).encode())]
                        ),
                    )
                )
                assert json.loads(values[0].body.to_pybytes())["ok"]
        else:
            client = Client(server.location, token=token)

            def call(timeout):
                client.rpc_timeout = timeout
                assert client._call("session.renew", **client.identity)["state"] == "OPEN"

        try:
            call(5)  # A healthy CONNECT tunnel must work before fault injection.
            proxy.paused.set()
            started = time.monotonic()
            error = ""
            try:
                call(0.5)
            except flight.FlightTimedOutError as caught:
                error = str(caught)
            return {"error": error, "pause_seconds": time.monotonic() - started, "initial_handshake": True}
        finally:
            proxy.paused.clear()
            if not raw:
                client.rpc_timeout = 5
            client.close()


def probe(kind, variable):
    # Called before importing Arrow/Vane in this process; other tests' clients,
    # worker actors and HTTP integrations never inherit these temporary settings.
    with proxy_server() as proxy:
        for name in (
            "grpc_proxy",
            "https_proxy",
            "http_proxy",
            "all_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "no_grpc_proxy",
            "no_proxy",
            "NO_PROXY",
            "GRPC_ADDRESS_HTTP_PROXY",
            "GRPC_ADDRESS_HTTP_PROXY_ENABLED_ADDRESSES",
        ):
            os.environ.pop(name, None)
        os.environ[variable] = f"http://127.0.0.1:{proxy.server_address[1]}"
        # Empty lists ensure loopback is eligible for gRPC's default proxy mapper.
        os.environ["no_grpc_proxy"] = os.environ["no_proxy"] = ""
        environment = dict(os.environ)
        value = native_probe(proxy) if kind == "native" else control_probe(proxy, kind == "raw_control")
        value["proxy_connections"] = len(proxy.connections)
        assert os.environ == environment
        print(json.dumps(value), flush=True)


@pytest.mark.timeout(30)
@pytest.mark.parametrize("variable", ["grpc_proxy", "https_proxy", "http_proxy"])
@pytest.mark.parametrize("kind", ["native", "control", "raw_control"])
def test_only_explicit_flight_endpoints_are_used_despite_proxy_environment(kind, variable):
    completed = subprocess.run(
        [sys.executable, "-I", str(Path(__file__).resolve()), kind, variable],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert result["initial_handshake"]
    if kind == "raw_control":
        # The same fault must break unconfigured Arrow, proving that the proxy
        # settings and the paused tunnel are active even after Vane is fixed.
        assert result["proxy_connections"] > 0 and "Deadline" in result["error"], result
    else:
        assert result["error"] == "", result
        assert result["proxy_connections"] == 0, result


if __name__ == "__main__":
    probe(*sys.argv[1:])
