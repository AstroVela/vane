# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import http.client
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class _ProxyState:
    def __init__(self, upstream_endpoint: str) -> None:
        parsed = urlparse(upstream_endpoint)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("The Iceberg REST fault proxy requires an HTTP upstream endpoint")
        self.upstream_host = parsed.hostname
        self.upstream_port = parsed.port or 80
        self.upstream_path = parsed.path.rstrip("/")
        self.lock = threading.Lock()
        self.commit_fault: str | None = None
        self.commit_attempts = 0
        self.get_cache: dict[str, tuple[int, str, list[tuple[str, str]], bytes]] = {}

    @staticmethod
    def is_table_commit(method: str, path: str) -> bool:
        request_path = path.split("?", 1)[0]
        if method != "POST":
            return False
        return request_path.endswith("/transactions/commit") or (
            re.search(r"/namespaces/[^/]+/tables/[^/]+$", request_path) is not None
        )

    @staticmethod
    def respond(
        handler: BaseHTTPRequestHandler,
        status: int,
        reason: str,
        headers: list[tuple[str, str]],
        body: bytes,
    ) -> None:
        handler.send_response(status, reason)
        for name, value in headers:
            if name.lower() not in _HOP_BY_HOP_HEADERS and name.lower() != "content-length":
                handler.send_header(name, value)
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        if handler.command != "HEAD":
            try:
                handler.wfile.write(body)
            except BrokenPipeError:
                pass
        handler.close_connection = True

    def status(self) -> dict[str, int]:
        with self.lock:
            cached_table_gets = sum(
                re.search(r"/namespaces/[^/]+/tables/[^/?]+(?:\?|$)", path) is not None for path in self.get_cache
            )
            return {
                "cached_table_get_count": cached_table_gets,
                "commit_attempts": self.commit_attempts,
            }

    def handle_forward(self, handler: BaseHTTPRequestHandler) -> None:
        content_length = int(handler.headers.get("Content-Length", "0"))
        body = handler.rfile.read(content_length) if content_length else b""
        with self.lock:
            active_fault = self.commit_fault
            commit_fault = active_fault if self.is_table_commit(handler.command, handler.path) else None
            if commit_fault is not None:
                self.commit_attempts += 1
            cached_get = (
                self.get_cache.get(handler.path) if active_fault == "reject" and handler.command == "GET" else None
            )
        if commit_fault == "reject":
            response = json.dumps(
                {
                    "error": {
                        "message": "planned catalog commit failure",
                        "type": "CommitFailedException",
                        "code": 409,
                    }
                }
            ).encode()
            self.respond(handler, 409, "Conflict", [("Content-Type", "application/json")], response)
            return
        if cached_get is not None:
            self.respond(handler, *cached_get)
            return

        upstream_path = self.upstream_path + handler.path
        headers = {
            name: value
            for name, value in handler.headers.items()
            if name.lower() not in _HOP_BY_HOP_HEADERS and name.lower() not in {"host", "content-length"}
        }
        headers["Host"] = f"{self.upstream_host}:{self.upstream_port}"
        headers["Connection"] = "close"
        if body:
            headers["Content-Length"] = str(len(body))
        connection = http.client.HTTPConnection(self.upstream_host, self.upstream_port, timeout=30)
        try:
            connection.request(handler.command, upstream_path, body=body or None, headers=headers)
            upstream_response = connection.getresponse()
            response = upstream_response.read()
            response_headers = upstream_response.getheaders()
            if commit_fault == "lose-response" and upstream_response.status < 400:
                response = json.dumps(
                    {
                        "error": {
                            "message": "planned catalog commit response loss after upstream success",
                            "type": "CommitStateUnknownException",
                            "code": 502,
                        }
                    }
                ).encode()
                self.respond(handler, 502, "Bad Gateway", [("Content-Type", "application/json")], response)
                return
            if handler.command == "GET" and upstream_response.status < 400:
                with self.lock:
                    self.get_cache[handler.path] = (
                        upstream_response.status,
                        upstream_response.reason,
                        response_headers,
                        response,
                    )
            self.respond(
                handler,
                upstream_response.status,
                upstream_response.reason,
                response_headers,
                response,
            )
        except Exception as exc:
            response = json.dumps(
                {
                    "error": {
                        "message": f"Iceberg REST fault proxy upstream failure: {exc}",
                        "type": type(exc).__name__,
                        "code": 502,
                    }
                }
            ).encode()
            self.respond(handler, 502, "Bad Gateway", [("Content-Type", "application/json")], response)
        finally:
            connection.close()


class _ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128


def _serve(upstream_endpoint: str) -> None:
    state = _ProxyState(upstream_endpoint)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_DELETE(self) -> None:  # noqa: N802
            state.handle_forward(self)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/__vane_fault/status":
                response = json.dumps(state.status()).encode()
                state.respond(self, 200, "OK", [("Content-Type", "application/json")], response)
                return
            state.handle_forward(self)

        def do_HEAD(self) -> None:  # noqa: N802
            state.handle_forward(self)

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/__vane_fault/reject":
                with state.lock:
                    state.commit_fault = "reject"
                state.respond(self, 200, "OK", [], b"")
                return
            if self.path == "/__vane_fault/lose-commit-response":
                with state.lock:
                    state.commit_fault = "lose-response"
                state.respond(self, 200, "OK", [], b"")
                return
            if self.path == "/__vane_fault/shutdown":
                state.respond(self, 200, "OK", [], b"")
                threading.Thread(target=server.shutdown, name="iceberg-rest-fault-shutdown", daemon=True).start()
                return
            state.handle_forward(self)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = _ProxyServer(("127.0.0.1", 0), Handler)
    host, port = server.server_address
    print(f"http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    args = parser.parse_args()
    _serve(args.upstream)


if __name__ == "__main__":
    main()
