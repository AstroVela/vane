# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real SDK and actor transport against a local embedding server."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import vane
from vane.ai import embed


@pytest.mark.parametrize("reason", ["API_KEY_INVALID", "BILLING_DISABLED"])
def test_google_sdk_account_errors_do_not_bisect(reason):
    errors = pytest.importorskip("google.genai.errors")
    from vane.ai.providers.google import GoogleTextEmbedder

    error = errors.ClientError(
        400,
        {"error": {"code": 400, "status": "INVALID_ARGUMENT", "details": [{"reason": reason}]}},
    )
    request = AsyncMock(side_effect=error)
    embedder = GoogleTextEmbedder.__new__(GoogleTextEmbedder)
    embedder._client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(embed_content=request)))
    embedder._model = "gemini-embedding-001"
    embedder._dimensions = 2
    embedder._options = {}
    embedder.configure_execution(max_retries=0, on_error="ignore", validate=lambda value: value)

    assert asyncio.run(embedder.embed_text(["input"] * 64)) == [None] * 64
    assert request.await_count == 1
    assert embedder.metrics.requests == 1


@pytest.mark.parametrize("code,expected_requests", [("insufficient_quota", 1), ("rate_limit_exceeded", 4)])
def test_openai_sdk_terminal_quota_is_not_retried(code, expected_requests, monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "local-embedding-test")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"error": {"code": code, "type": code, "message": "private quota diagnostic"}}).encode()
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = vane.connect()
    try:
        result = connection.sql("SELECT * FROM (VALUES ('first'), ('second')) AS t(text)").select(
            embed(
                vane.col("text"),
                model="fixed",
                dimensions=2,
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                supports_overriding_dimensions=False,
                request_batch_size=64,
                batch_size=8,
                max_retries=3,
                on_error="ignore",
            ).alias("embedding")
        )
        assert result.fetchall() == [(None,), (None,)]
        assert len(calls) == expected_requests
        assert all(call["input"] == ["first", "second"] for call in calls)
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    "vertexai,on_error,failure",
    [
        (True, "raise", "none"),
        (True, "ignore", "vector"),
        (False, "ignore", "vector"),
        (False, "raise", "vector"),
        (False, "ignore", "count"),
        (False, "raise", "count"),
    ],
)
def test_google_sdk_validation_and_vertex_limits_through_actor(monkeypatch, vertexai, on_error, failure):
    pytest.importorskip("google.genai")
    monkeypatch.setenv("GOOGLE_API_KEY", "local-embedding-test")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", str(vertexai).lower())
    for name in (
        "GEMINI_API_KEY",
        "GOOGLE_GENAI_USE_ENTERPRISE",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    calls = []

    def vector(text):
        return [float(text)] + [1.0] * 127

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            contents = [request["content"]] if vertexai else [item["content"] for item in request["requests"]]
            texts = [content["parts"][0]["text"] for content in contents]
            calls.append(texts)
            embeddings = [{"values": ["private malformed vector"] if text == "bad" else vector(text)} for text in texts]
            if failure == "count" and "bad" in texts:
                embeddings = []
            response = {"embedding": embeddings[0]} if vertexai else {"embeddings": embeddings}
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    endpoint = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("GOOGLE_VERTEX_BASE_URL", endpoint)
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", endpoint)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = vane.connect()
    try:
        middle = "2" if failure == "none" else "bad"
        source = f"SELECT * FROM (VALUES (0, '0'), (1, NULL), (2, '{middle}'), (3, '3')) AS t(id, text)"
        result = connection.sql(source).select(
            vane.col("id"),
            embed(
                vane.col("text"),
                provider="google",
                model="gemini-embedding-2",
                dimensions=128,
                request_batch_size=64,
                batch_size=8,
                max_retries=3,
                on_error=on_error,
            ).alias("embedding"),
        )
        if on_error == "raise" and failure != "none":
            with pytest.raises(Exception) as caught:
                result.fetchall()
            assert "private malformed vector" not in str(caught.value)
            assert calls == [["0", "bad", "3"]]
        else:
            assert result.order("id").fetchall() == [
                (0, tuple(vector("0"))),
                (1, None),
                (2, tuple(vector("2")) if failure == "none" else None),
                (3, tuple(vector("3"))),
            ]
            if vertexai:
                assert calls == [["0"], [middle], ["3"]]
            else:
                assert calls == [["0", "bad", "3"], ["0"], ["bad", "3"], ["bad"], ["3"]]
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("entrypoint", ["expression", "relation", "sql"])
@pytest.mark.parametrize("failure", ["none", "payload_limit", "short", "duplicate_index"])
def test_fixed_dimension_endpoint_through_actor(entrypoint, monkeypatch, failure):
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
            if failure == "payload_limit" and len(request["input"]) > 1:
                self.send_error(413)
                return
            data = [
                {"object": "embedding", "index": i, "embedding": [float(text), 1.0]}
                for i, text in enumerate(request["input"])
            ]
            if len(data) > 1:
                if failure == "short":
                    data = data[:1]
                elif failure == "duplicate_index":
                    for item in data:
                        item["index"] = 0
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
            "on_error": "ignore",
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
                f"""SELECT id, ai_embed(text, model := 'fixed', dimensions := 2, on_error := 'ignore',
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
        successful = [call for call in calls if failure == "none" or len(call["input"]) == 1]
        assert sorted(text for call in successful for text in call["input"]) == ["0", "2", "3", "4"]
        assert len(calls) == (2 if failure == "none" else 6)
        assert all("dimensions" not in call and len(call["input"]) <= 2 for call in calls)
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("on_error", ["raise", "ignore"])
@pytest.mark.parametrize("envelope", ["null", "missing", "object", "string", "number", "bool"])
def test_openai_sdk_malformed_envelope_recovers_valid_neighbors(monkeypatch, on_error, envelope):
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "local-embedding-test")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            texts = request["input"]
            calls.append(texts)
            response = {
                "object": "list",
                "model": "fixed",
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            }
            if "bad" in texts:
                if envelope != "missing":
                    response["data"] = {
                        "null": None,
                        "object": {"private diagnostic": "bad"},
                        "string": "private diagnostic",
                        "number": 1,
                        "bool": True,
                    }[envelope]
            else:
                response["data"] = [
                    {"object": "embedding", "index": i, "embedding": [float(text), 1.0]} for i, text in enumerate(texts)
                ]
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = vane.connect()
    try:
        result = connection.sql("SELECT * FROM (VALUES ('0'), (NULL), ('bad'), ('3'), ('4')) AS t(text)").select(
            embed(
                vane.col("text"),
                model="fixed",
                dimensions=2,
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                supports_overriding_dimensions=False,
                request_batch_size=3,
                batch_size=8,
                max_retries=3,
                on_error=on_error,
            ).alias("embedding")
        )
        if on_error == "raise":
            with pytest.raises(Exception, match="invalid data array") as caught:
                result.fetchall()
            assert "private" not in str(caught.value)
            assert calls == [["0", "bad", "3"]]
        else:
            assert result.fetchall() == [((0.0, 1.0),), (None,), (None,), ((3.0, 1.0),), ((4.0, 1.0),)]
            assert calls == [["0", "bad", "3"], ["0"], ["bad", "3"], ["bad"], ["3"], ["4"]]
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
