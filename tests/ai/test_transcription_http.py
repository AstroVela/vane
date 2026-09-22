# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Actual OpenAI multipart transport and Ray execution against a local fixture."""

from __future__ import annotations

import asyncio
import io
import json
import subprocess
import sys
import threading
import wave
from contextlib import contextmanager
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pyarrow as pa
import pytest

import vane
from vane.ai import transcribe
from vane.ai._transcription import _TranscribeBatch
from vane.ai.provider import load_provider
from vane.ai.typing import UDFOptions
from vane.execution._async_runtime import AsyncRuntime

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")


def response():
    return {
        "task": "transcribe",
        "language": "english",
        "duration": 3.0,
        "text": " A bicycle. A car.",
        "segments": [
            {
                "id": 0,
                "seek": 0,
                "start": 0.25,
                "end": 1.5,
                "text": " A bicycle.",
                "tokens": [1],
                "temperature": 0.0,
                "avg_logprob": -0.1,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.01,
            },
            {
                "id": 1,
                "seek": 0,
                "start": 1.5,
                "end": 2.8,
                "text": " A car.",
                "tokens": [2],
                "temperature": 0.0,
                "avg_logprob": -0.1,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.01,
            },
        ],
    }


def audio_bytes():
    data = io.BytesIO()
    with wave.open(data, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(b"\x01\x00" * 24_000)
    return data.getvalue()


def multipart(content_type, body):
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    )
    return {
        part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
        for part in message.iter_parts()
    }


def install_transport(monkeypatch, handler):
    clients = []

    class Client(openai.AsyncOpenAI):
        def __init__(self, **kwargs):
            self.created_loop = asyncio.get_running_loop()
            self.closed_loop = None
            super().__init__(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kwargs)
            clients.append(self)

        async def close(self):
            self.closed_loop = asyncio.get_running_loop()
            await super().close()

    monkeypatch.setattr(openai, "AsyncOpenAI", Client)
    return clients


@contextmanager
def wrapper(**options):
    runtime = AsyncRuntime()
    descriptor = load_provider("openai", api_key="application-key", base_url="http://asr.test/v1").get_transcriber(
        options={"language": "en", "prompt": "Bicycle and car"}
    )
    batch = _TranscribeBatch(descriptor, UDFOptions(max_concurrency_per_actor=2, **options))
    batch.bind_async_runtime(runtime.run)
    try:
        yield batch, runtime
    finally:
        batch.close()
        runtime.close()


def table(count=1):
    return pa.table(
        {"audio": [{"message_0": {"data": audio_bytes(), "content_type": "audio/wav", "error": None}}] * count}
    )


def test_async_sdk_reuses_loop_and_limits_requests(monkeypatch):
    requests = []
    active, peak = 0, 0

    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(active, peak)
        try:
            fields = multipart(request.headers["content-type"], request.content)
            requests.append(fields)
            assert request.headers["authorization"] == "Bearer application-key"
            await asyncio.sleep(0.01)
            return httpx.Response(200, json=response())
        finally:
            active -= 1

    clients = install_transport(monkeypatch, handler)
    with wrapper(max_retries=0) as (batch, runtime):
        assert len(batch(table(5))) == 5 and len(batch(table())) == 1
        assert peak == 2 and len(clients) == 1
        assert clients[0].created_loop is runtime.loop
    assert clients[0].closed_loop is clients[0].created_loop
    for fields in requests:
        assert fields["file"] == audio_bytes()
        assert fields["model"] == b"whisper-1" and fields["response_format"] == b"verbose_json"
        assert fields["timestamp_granularities[]"] == b"segment" and fields["language"] == b"en"
        assert fields["prompt"] == b"Bicycle and car"


@pytest.mark.parametrize("case", ["missing_segments", "wrong_end", "untimed_text", "http_failure"])
@pytest.mark.parametrize("on_error", ["raise", "ignore"])
def test_invalid_endpoint_results_have_explicit_row_error_policy(monkeypatch, case, on_error, caplog):
    def handler(request):
        body = response()
        if case == "missing_segments":
            body.pop("segments")
        elif case == "wrong_end":
            body["segments"][1]["end"] = 4.0
        elif case == "untimed_text":
            body["text"] += " private extra words"
        else:
            return httpx.Response(
                400, json={"error": {"message": "private-input-and-key", "type": "invalid_request_error"}}
            )
        return httpx.Response(200, json=body)

    install_transport(monkeypatch, handler)
    with wrapper(max_retries=0, on_error=on_error) as (batch, _):
        if on_error == "raise":
            with pytest.raises(RuntimeError, match="Transcribe execution") as error:
                batch(table())
            assert "private" not in str(error.value) and error.value.__context__ is None
        else:
            assert batch(table())["transcription"].to_pylist() == [None]
            assert "substituted NULL" in caplog.text and "private" not in caplog.text


def test_http_retry_preserves_the_same_audio(monkeypatch):
    requests = []

    def handler(request):
        requests.append(multipart(request.headers["content-type"], request.content))
        if len(requests) == 1:
            return httpx.Response(503, json={"error": {"message": "retry", "type": "server_error"}})
        return httpx.Response(200, json=response())

    install_transport(monkeypatch, handler)
    with wrapper(max_retries=1) as (batch, _):
        assert json.loads(batch(table())["transcription"][0].as_py())["duration"] == 3.0
    assert len(requests) == 2 and requests[0] == requests[1]


@contextmanager
def server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["content-length"]))
            requests.append((self.path, dict(self.headers), multipart(self.headers["content-type"], body)))
            output = json.dumps(response()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(output)))
            self.end_headers()
            self.wfile.write(output)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/v1", requests
    finally:
        httpd.shutdown()
        thread.join(timeout=10)
        httpd.server_close()


def run_api_forms(path, provider, backend):
    with vane.connect() as conn:
        source = conn.from_arrow(pa.table({"id": [0, 1], "audio": [str(path), None]}))
        for form in ("expression", "relation", "method"):
            options = {"provider": provider, "execution_backend": backend, "max_retries": 0}
            if form == "expression":
                result = source.select("id", transcribe(vane.col("audio"), **options).alias("speech"))
            elif form == "relation":
                result = transcribe(source, vane.col("audio"), output_column="speech", **options)
            else:
                result = source.transcribe(vane.col("audio"), output_column="speech", **options)
            rows = result.order("id").select("speech").to_arrow_table().to_pylist()
            assert rows[0]["speech"]["segments"][1] == {"start": 1.5, "end": 2.8, "text": " A car."}
            assert rows[1]["speech"] is None


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
def test_installed_native_file_and_sdk_transport(tmp_path, monkeypatch, backend):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    path = tmp_path / "recording.wav"
    path.write_bytes(audio_bytes())
    with server() as (url, requests):
        run_api_forms(path, load_provider("openai", api_key="application-key", base_url=url), backend)
    assert len(requests) == 3
    assert all(fields["file"] == path.read_bytes() for _, _, fields in requests)


@pytest.mark.real_ray
@pytest.mark.ray_cluster_owner
def test_default_ray_existing_workers_use_captured_transcription_clients(tmp_path):
    pytest.importorskip("ray")
    path = tmp_path / "recording.wav"
    path.write_bytes(audio_bytes())
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,sys; sys.path.append(sys.argv[1]); runpy.run_path(sys.argv[2], run_name='__main__')",
            str(Path(__file__).resolve().parents[1]),
            __file__,
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_invalid_file_never_reaches_transcription_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    missing = tmp_path / "missing.wav"
    with server() as (url, requests), vane.connect() as conn:
        source = conn.from_arrow(pa.table({"audio": [str(missing), None]}))
        provider = load_provider("openai", api_key="application-key", base_url=url)
        result = transcribe(
            source, vane.col("audio"), provider=provider, on_error="ignore", execution_backend="subprocess_task"
        )
        assert result.select("transcription").fetchall() == [(None,), (None,)]
        with pytest.raises(Exception, match="FILE"):
            transcribe(source, vane.col("audio"), provider=provider, execution_backend="subprocess_task").fetchall()
        assert not requests


def run_ray_case(path):
    import ray
    from ray_test_profile import ray_test_object_store_options

    with pytest.MonkeyPatch.context() as env:
        for name in ("VANE_RUNNER", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"):
            env.delenv(name, raising=False)
        env.setenv("OPENAI_API_KEY", "wrong-worker-key")
        env.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1/v1")
        env.setenv("OPENAI_ORG_ID", "wrong-worker-org")
        env.setenv("OPENAI_PROJECT_ID", "wrong-worker-project")
        env.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer wrong-worker-custom-header")
        ray.init(address="local", num_cpus=2, include_dashboard=False, **ray_test_object_store_options())
        try:
            with server() as (url, requests):
                env.setenv("OPENAI_API_KEY", "application-key")
                env.setenv("OPENAI_BASE_URL", url)
                env.delenv("OPENAI_ORG_ID")
                env.delenv("OPENAI_PROJECT_ID")
                provider = load_provider("openai")
                # Both the normal default and explicitly distributed UDF actors
                # must use the application snapshot rather than Ray's old env.
                for backend in (None, "ray_actor", "ray_task"):
                    run_api_forms(path, provider, backend)
                assert vane.get_or_infer_runner_type() == "ray"
            assert len(requests) == 9
            for route, headers, fields in requests:
                headers = {key.lower(): value for key, value in headers.items()}
                assert route == "/v1/audio/transcriptions" and fields["file"] == path.read_bytes()
                assert headers["authorization"] == "Bearer application-key"
                assert "openai-project" not in headers and "openai-organization" not in headers
        finally:
            vane.teardown_runner()
            ray.shutdown()


if __name__ == "__main__":
    run_ray_case(Path(sys.argv[3]))
