# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real SDK and actor transport against a local embedding server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import vane
from vane.ai import embed


@pytest.mark.parametrize("entrypoint", ["expression", "relation", "sql"])
def test_fixed_dimension_endpoint_through_actor(entrypoint, monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "local-embedding-test")
    # A loopback fixture must not depend on the developer's proxy packages
    # or route requests through an externally configured proxy.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(request)
            if "dimensions" in request or len(request["input"]) > 2:
                self.send_error(400)
                return
            data = [
                {"object": "embedding", "index": i, "embedding": [float(text), 1.0]}
                for i, text in enumerate(request["input"])
            ]
            response = json.dumps(
                {
                    "object": "list",
                    "model": "fixed",
                    "data": list(reversed(data)),
                    "usage": {"prompt_tokens": 4, "total_tokens": 4},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = vane.connect()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1"
        options = {
            "base_url": endpoint,
            "supports_overriding_dimensions": False,
            "request_batch_size": 2,
            "max_concurrency_per_actor": 2,
            "batch_size": 8,
            "max_retries": 0,
        }
        source = "SELECT * FROM (VALUES (0, '0'), (1, NULL), (2, '2'), (3, '3'), (4, '4')) AS t(id, text)"
        relation = connection.sql(source)
        if entrypoint == "expression":
            result = relation.select(
                vane.col("id"), embed(vane.col("text"), model="fixed", dimensions=2, **options).alias("embedding")
            )
        elif entrypoint == "relation":
            result = relation.embed(vane.col("text"), model="fixed", dimensions=2, **options).select("id", "embedding")
        else:
            result = connection.sql(
                f"""SELECT id, ai_embed(text, model := 'fixed', dimensions := 2,
                    options := {{'base_url': '{endpoint}', 'supports_overriding_dimensions': false,
                                 'request_batch_size': 2, 'max_concurrency_per_actor': 2,
                                 'batch_size': 8, 'max_retries': 0}}) AS embedding
                    FROM ({source})"""
            )
        assert result.order("id").fetchall() == [
            (0, (0.0, 1.0)),
            (1, None),
            (2, (2.0, 1.0)),
            (3, (3.0, 1.0)),
            (4, (4.0, 1.0)),
        ]
        assert sorted(text for call in calls for text in call["input"]) == ["0", "2", "3", "4"]
        assert all("dimensions" not in call and len(call["input"]) <= 2 for call in calls)
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
