# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Observe anonymous object reads against real MinIO; never serve fixture data."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen


def main(endpoint, audit, control):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def forward(self):
            parsed = urlsplit(self.path)
            path = unquote(parsed.path)
            policy = json.loads(Path(control).read_text())
            credentials = bool(self.headers.get("Authorization") or self.headers.get("X-Amz-Security-Token"))
            status = 500
            try:
                if credentials or parsed.query:
                    status = 403
                    self.send_error(status, "Only anonymous object reads are permitted")
                    return
                if (policy.get("deny_metadata") and "/metadata/" in path) or path in policy.get("denied", []):
                    status = 404
                    self.send_error(status, "Object unavailable for detached replay")
                    return
                headers = {
                    name: self.headers[name]
                    for name in ("Range", "If-Match", "If-None-Match")
                    if self.headers.get(name)
                }
                request = Request(endpoint + self.path, method=self.command, headers=headers)
                try:
                    response = urlopen(request, timeout=20)
                except HTTPError as error:
                    response = error
                with response:
                    status = response.status
                    body = response.read() if self.command == "GET" else b""
                    self.send_response(status)
                    for name in ("Content-Type", "Content-Length", "Content-Range", "ETag", "Accept-Ranges"):
                        if name in response.headers:
                            self.send_header(name, response.headers[name])
                    self.end_headers()
                    if body:
                        self.wfile.write(body)
            finally:
                event = {
                    "method": self.command,
                    "path": path,
                    "status": status,
                    "range": bool(self.headers.get("Range")),
                    "credentials": credentials,
                    "query": bool(parsed.query),
                }
                with lock, Path(audit).open("a") as stream:
                    stream.write(json.dumps(event) + "\n")

        do_GET = forward
        do_HEAD = forward

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        print(json.dumps({"endpoint": f"http://127.0.0.1:{server.server_port}"}), flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main(*sys.argv[1:])
