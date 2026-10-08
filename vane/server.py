# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Standalone Flight session server. Remote SQL execution is a subsequent increment."""

from __future__ import annotations

import argparse
import ipaddress
import json
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pyarrow.flight as flight

from vane.execution.flight_control import FlightControlServer
from vane.execution.pipelined_plan import RayResources
from vane.execution.request_admission import _timeout
from vane.execution.server_session import SessionLimits, SessionService


class _Shutdown:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.error: BaseException | None = None


class Server:
    """Host leased sessions on Flight while keeping their Runtime server-owned.

    The listener starts during construction. close() has a caller deadline;
    a timed-out call retains all cleanup owners and can be retried. This first
    increment exposes session actions only, as advertised by vane.info.
    """

    def __init__(
        self,
        *,
        token: str,
        host: str = "127.0.0.1",
        port: int = 8815,
        tls_certificates: list[tuple[bytes, bytes]] | None = None,
        database: str = ":memory:",
        read_only: bool = False,
        config: dict[str, Any] | None = None,
        resources: RayResources | None = None,
        sessions: SessionLimits | None = None,
    ) -> None:
        if not isinstance(token, str) or not token.isascii() or not 32 <= len(token) <= 4096 or not token.isprintable():
            raise ValueError("token must contain 32 to 4096 printable ASCII characters")
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if not tls_certificates:
            if host != "localhost" and not ipaddress.ip_address(host).is_loopback:
                raise ValueError("TLS certificates are required for non-loopback listeners")
            location = flight.Location.for_grpc_tcp(host, port)
        else:
            location = flight.Location.for_grpc_tls(host, port)
        self.service = SessionService(
            database=database, read_only=read_only, config=config, resources=resources, limits=sessions
        )
        try:
            self._flight = FlightControlServer(self.service, location, token, tls_certificates)
        except BaseException:
            self.service.close()
            raise
        self._lock = threading.Lock()
        self._shutdown: _Shutdown | None = None
        self.port = self._flight.port
        self.location = (
            flight.Location.for_grpc_tls(host, self.port)
            if tls_certificates
            else flight.Location.for_grpc_tcp(host, self.port)
        )

    def close(self, *, timeout: float = 10) -> None:
        deadline = time.monotonic() + _timeout(timeout, "server close timeout")
        self.service.close(timeout=max(0, deadline - time.monotonic()))
        remaining = max(0, deadline - time.monotonic())
        if not self._lock.acquire(timeout=min(remaining, threading.TIMEOUT_MAX)):
            raise TimeoutError("Flight shutdown is pending; retry Server.close()")
        try:
            attempt = self._shutdown
            if attempt is None or (attempt.done.is_set() and attempt.error is not None):
                attempt = _Shutdown()
                self._shutdown = attempt
                thread = threading.Thread(
                    target=self._stop_flight, args=(attempt,), name="vane-flight-stop", daemon=True
                )
                try:
                    thread.start()
                except BaseException:
                    self._shutdown = None
                    raise
        finally:
            self._lock.release()
        # Flight shutdown waits for RPC handlers. It must run outside a handler
        # and must not be reissued while a previous call is still in flight.
        if not attempt.done.wait(min(max(0, deadline - time.monotonic()), threading.TIMEOUT_MAX)):
            raise TimeoutError("Flight shutdown is pending; retry Server.close()")
        if attempt.error is not None:
            raise RuntimeError("Flight shutdown failed; retry Server.close()") from attempt.error

    def _stop_flight(self, attempt: _Shutdown) -> None:
        try:
            self._flight.shutdown()
        except BaseException as error:
            attempt.error = error.with_traceback(None)
        finally:
            attempt.done.set()

    def __enter__(self) -> Server:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8815)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--tls-cert", type=Path)
    parser.add_argument("--tls-key", type=Path)
    parser.add_argument("--database", default=":memory:")
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--max-sessions", type=int, default=64)
    parser.add_argument("--lease-seconds", type=float, default=60)
    args = parser.parse_args()
    if bool(args.tls_cert) != bool(args.tls_key):
        parser.error("--tls-cert and --tls-key must be supplied together")
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    server = Server(
        token=args.token_file.read_text().strip(),
        host=args.host,
        port=args.port,
        tls_certificates=[(args.tls_cert.read_bytes(), args.tls_key.read_bytes())] if args.tls_cert else None,
        database=args.database,
        read_only=args.read_only,
        sessions=SessionLimits(max_sessions=args.max_sessions, lease_seconds=args.lease_seconds),
    )
    try:
        print(json.dumps({"location": server.location.uri.decode(), **server.service.snapshot()}), flush=True)
        stop.wait()
    finally:
        while True:
            try:
                server.close()
                break
            except (TimeoutError, RuntimeError) as error:
                print(f"Server cleanup pending: {error}", file=sys.stderr, flush=True)
                time.sleep(0.1)


if __name__ == "__main__":
    main()
