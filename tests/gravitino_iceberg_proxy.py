# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Test-owned loopback forwarding endpoint for a real anonymous Iceberg REST service."""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def main(endpoint):
    parsed = urlsplit(endpoint)
    upstream = f"{parsed.scheme}://{parsed.netloc}"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = Request(upstream + self.path)
            try:
                response = urlopen(request, timeout=20)
            except HTTPError as error:
                response = error
            with response:
                body = response.read()
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        print(json.dumps({"endpoint": f"http://127.0.0.1:{server.server_port}{parsed.path}"}), flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1])
