# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real HTTP and installed executor acceptance for the inference providers."""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pyarrow as pa
import pytest

import vane
from vane.ai import AudioInput, MultimodalInput, TextInput, VideoInput
from vane.ai.providers.sglang import SGLangProvider
from vane.ai.providers.vllm import VLLMProvider


@pytest.fixture
def embedding_server():
    pytest.importorskip("httpx")
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            payload = json.loads(body)
            calls.append((self.path, dict(self.headers), payload, len(body)))
            rows = payload.get("input", payload.get("messages"))
            data = []
            for i, row in enumerate(rows):
                if isinstance(row, str):
                    text = row
                elif isinstance(row, dict):
                    text = row.get("text", "0")
                else:
                    text = next((part["text"] for part in row[0]["content"] if part["type"] == "text"), "0")
                data.append({"index": i, "embedding": [float(text), 1.0]})
            response = json.dumps({"data": list(reversed(data))}).encode()
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
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
def test_runtime_http_rows_optional_constant_null_and_encoded_media(embedding_server, family, backend):
    endpoint, calls = embedding_server
    factory = SGLangProvider if family == "sglang" else VLLMProvider
    video = b"\x00\x01\xff original encoded media"
    with vane.connect() as conn:
        relation = conn.from_arrow(pa.table({"id": [1, 2, 3], "text": ["1", "2", "3"], "video": [video, None, video]}))
        rows = (
            relation.embed(
                MultimodalInput(
                    [
                        VideoInput(vane.col("video"), mime_type="video/mp4"),
                        TextInput(vane.col("text")),
                        VideoInput(vane.ConstantExpression(None), mime_type="video/mp4"),
                    ]
                ),
                provider=factory(api_key="fixture-key"),
                model="served-model",
                dimensions=2,
                runtime="http",
                base_url=endpoint,
                http_input_format="multimodal",
                media_mime_types=["video/mp4"],
                batch_size=3,
                request_batch_size=2,
                max_retries=0,
                execution_backend=backend,
            )
            .select("id", "embedding")
            .order("id")
            .fetchall()
        )
    assert rows == [(i, (float(i), 1.0)) for i in range(1, 4)]
    inputs = []
    for path, headers, payload, _ in calls:
        assert path == "/v1/embeddings"
        assert headers["Authorization"] == "Bearer fixture-key"
        assert "dimensions" not in payload
        inputs.extend(payload.get("input", payload.get("messages")))
    assert len(inputs) == 3
    video_urls = []
    for row in inputs:
        if family == "sglang":
            if "video" in row:
                video_urls.append(row["video"])
        else:
            video_urls.extend(part["video_url"]["url"] for part in row[0]["content"] if part["type"] == "video_url")
    assert len(video_urls) == 2
    assert all(base64.b64decode(url.split(",", 1)[1]) == video for url in video_urls)


@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
@pytest.mark.parametrize("invalid", ["empty", "serialized"])
def test_input_errors_cannot_be_ignored_by_real_executors(embedding_server, family, backend, invalid):
    endpoint, calls = embedding_server
    with vane.connect() as conn:
        relation = conn.from_arrow(
            pa.table(
                {
                    "id": [1, 2],
                    "text": ["1", "" if invalid == "empty" else "2"],
                    "video": [None, None if invalid == "empty" else b"x" * 768],
                }
            )
        )
        with pytest.raises(Exception, match="no nonempty parts|serialized request byte budget"):
            relation.embed(
                MultimodalInput([TextInput(vane.col("text")), VideoInput(vane.col("video"), mime_type="video/mp4")]),
                provider=family,
                model="served-model",
                dimensions=2,
                runtime="http",
                base_url=endpoint,
                http_input_format="multimodal",
                media_mime_types=["video/mp4"],
                max_input_bytes=1024,
                max_request_bytes=1024,
                batch_size=2,
                max_retries=0,
                execution_backend=backend,
                on_error="ignore",
            ).fetchall()
    assert calls == []


@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
def test_text_search_and_media_share_the_multimodal_http_contract(embedding_server, family, backend):
    endpoint, calls = embedding_server
    with vane.connect() as conn:
        rows = (
            conn.from_arrow(pa.table({"text": ["1", None, "2"]}))
            .embed(
                vane.col("text"),
                provider=family,
                model="served-model",
                dimensions=2,
                runtime="http",
                base_url=endpoint,
                http_input_format="multimodal",
                media_mime_types=["video/mp4"],
                batch_size=3,
                execution_backend=backend,
            )
            .select("embedding")
            .fetchall()
        )
    assert sorted([row[0] for row in rows if row[0] is not None]) == [(1.0, 1.0), (2.0, 1.0)]
    assert sum(row[0] is None for row in rows) == 1
    payload = calls[0][2]
    if family == "sglang":
        assert all(isinstance(row, dict) and "text" in row for row in payload["input"])
    else:
        assert all(isinstance(row, list) and row[0]["role"] == "user" for row in payload["messages"])


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
def test_vllm_encoded_audio_blob(embedding_server, backend):
    endpoint, calls = embedding_server
    audio = b"unchanged encoded audio"
    with vane.connect() as conn:
        rows = (
            conn.from_arrow(pa.table({"audio": [audio]}))
            .embed(
                AudioInput(vane.col("audio"), mime_type="audio/mpeg"),
                provider="vllm",
                model="served-model",
                dimensions=2,
                runtime="http",
                base_url=endpoint,
                http_input_format="multimodal",
                media_mime_types=["audio/mpeg"],
                execution_backend=backend,
            )
            .select("embedding")
            .fetchall()
        )
    assert rows == [((0.0, 1.0),)]
    part = calls[0][2]["messages"][0][0]["content"][0]["input_audio"]
    assert part["format"] == "mp3"
    assert base64.b64decode(part["data"]) == audio
