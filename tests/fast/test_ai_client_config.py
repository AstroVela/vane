# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Provider identity survives serialization into an existing Ray runtime."""

from __future__ import annotations

import asyncio
import json
import os
import pickle
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from vane.ai._client_config import ProviderClientConfigurationError
from vane.ai.provider import load_provider
from vane.ai.providers._google_client_config import capture_google_client
from vane.ai.providers._openai_client_config import capture_openai_client, create_openai_client

CLIENT_ENV = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT_ID",
    "OPENAI_CUSTOM_HEADERS",
    "OPENAI_ADMIN_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_GENAI_USE_ENTERPRISE",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_GEMINI_BASE_URL",
    "GOOGLE_VERTEX_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
)
CASES = [("openai", "embed"), ("openai", "prompt"), ("google", "embed"), ("google", "prompt"), ("anthropic", "prompt")]
WORKER_ENV = {
    "OPENAI_API_KEY": "worker-openai-key",
    "OPENAI_ORG_ID": "org-worker",
    "OPENAI_PROJECT_ID": "proj-worker",
    "OPENAI_BASE_URL": "http://127.0.0.1:9",
    "OPENAI_ADMIN_KEY": "worker-admin-key",
    "OPENAI_CUSTOM_HEADERS": "Authorization: Bearer worker-header-key\nOpenAI-Project: proj-worker-header",
    "GOOGLE_API_KEY": "worker-google-key",
    "GOOGLE_GENAI_USE_VERTEXAI": "true",
    "GOOGLE_GENAI_USE_ENTERPRISE": "true",
    "GOOGLE_CLOUD_PROJECT": "worker-project",
    "GOOGLE_CLOUD_LOCATION": "us-central1",
    "GOOGLE_GEMINI_BASE_URL": "http://127.0.0.1:9",
    "GOOGLE_VERTEX_BASE_URL": "http://127.0.0.1:9",
    "ANTHROPIC_API_KEY": "worker-anthropic-key",
    "ANTHROPIC_AUTH_TOKEN": "worker-anthropic-token",
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:9",
    "ANTHROPIC_CUSTOM_HEADERS": "X-Api-Key: worker-header-key\nAuthorization: Bearer worker-header-token",
}


@pytest.fixture(autouse=True)
def clean_client_environment(monkeypatch):
    for key in CLIENT_ENV:
        monkeypatch.delenv(key, raising=False)
    # The SDK constructors inspect proxy settings even without a request.
    # These tests use loopback servers and must not depend on developer proxies.
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)


def _descriptor(family, operation, **settings):
    provider = load_provider(family, **settings)
    if operation == "embed":
        return provider.get_text_embedder(dimensions=2 if family == "openai" else 128)
    return provider.get_prompter(
        model={"openai": "gpt-4.1", "google": "gemini-2.5-flash", "anthropic": "claude-test"}[family],
        options={"max_tokens": 32} if family == "anthropic" else {},
    )


def _sdk_state(payload):
    """Construct the real SDK without sending a model request."""

    async def inspect_client():
        descriptor = pickle.loads(payload)
        runtime = descriptor.instantiate()
        client = runtime._client
        try:
            if descriptor.get_provider() == "google":
                return {
                    "api_key": client._api_client.api_key,
                    "vertexai": client.vertexai,
                    "base_url": client._api_client._http_options.base_url,
                    "project": client._api_client.project,
                    "location": client._api_client.location,
                }
            return {
                "api_key": client.api_key,
                "base_url": str(client.base_url),
                "organization": getattr(client, "organization", None),
                "project": getattr(client, "project", None),
                "auth_token": getattr(client, "auth_token", None),
                "headers": dict(client.default_headers),
            }
        finally:
            await runtime.aclose()

    return asyncio.run(inspect_client())


@pytest.mark.parametrize("family,operation", CASES)
def test_descriptor_pins_application_identity_under_conflicting_sdk_environment(monkeypatch, family, operation):
    pytest.importorskip("google.genai" if family == "google" else family)
    monkeypatch.setenv(f"{family.upper()}_API_KEY", "application-key")
    descriptor = _descriptor(family, operation)
    payload = pickle.dumps(descriptor)
    for key, value in WORKER_ENV.items():
        monkeypatch.setenv(key, value)
    before = dict(os.environ)
    state = _sdk_state(payload)
    assert dict(os.environ) == before
    assert state["api_key"] == "application-key"
    if family == "google":
        assert state["vertexai"] is False
        assert state["base_url"] == "https://generativelanguage.googleapis.com/"
    else:
        assert state["project"] is None
        assert state["organization"] is None
        assert state["auth_token"] is None
        assert "worker" not in repr(state["headers"])
        assert state["base_url"].startswith("https://api." + family + ".com")


@pytest.mark.parametrize("operation", ["embed", "prompt"])
@pytest.mark.parametrize(
    "google_key,settings,expected_key",
    [
        pytest.param(None, {}, "application-gemini-key", id="google-key-unset"),
        pytest.param("", {}, "application-gemini-key", id="google-key-empty"),
        pytest.param("application-google-key", {}, "application-google-key", id="google-key-precedence"),
        pytest.param(
            "application-google-key", {"api_key": "explicit-key"}, "explicit-key", id="explicit-key-precedence"
        ),
    ],
)
def test_google_key_selection_survives_conflicting_worker_environment(
    monkeypatch, operation, google_key, settings, expected_key
):
    pytest.importorskip("google.genai")
    if google_key is not None:
        monkeypatch.setenv("GOOGLE_API_KEY", google_key)
    monkeypatch.setenv("GEMINI_API_KEY", "application-gemini-key")
    payload = pickle.dumps(_descriptor("google", operation, **settings))

    for key, value in WORKER_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("GEMINI_API_KEY", "worker-gemini-key")
    state = _sdk_state(payload)
    assert state["api_key"] == expected_key
    assert state["vertexai"] is False


@pytest.mark.parametrize("operation", ["embed", "prompt"])
def test_google_empty_environment_keys_never_use_worker_credentials(monkeypatch, operation):
    pytest.importorskip("google.genai")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    payload = pickle.dumps(_descriptor("google", operation))

    for key, value in WORKER_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("GEMINI_API_KEY", "worker-gemini-key")
    with pytest.raises(ProviderClientConfigurationError, match="application"):
        _sdk_state(payload)


def test_google_explicit_empty_key_rejected_with_valid_environment(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "application-google-key")
    monkeypatch.setenv("GEMINI_API_KEY", "application-gemini-key")
    with pytest.raises(ValueError, match="Google api_key must be a non-empty string"):
        load_provider("google", api_key="")


@pytest.mark.parametrize("family,operation", CASES)
def test_missing_application_credentials_never_use_worker_credentials(monkeypatch, family, operation):
    pytest.importorskip("google.genai" if family == "google" else family)
    payload = pickle.dumps(_descriptor(family, operation))
    for key, value in WORKER_ENV.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ProviderClientConfigurationError, match="application"):
        _sdk_state(payload)


@pytest.mark.parametrize("family,operation", CASES)
def test_explicit_credentials_are_redacted_but_serializable(family, operation):
    descriptor = _descriptor(family, operation, api_key="explicit-secret-sentinel")
    payload = pickle.dumps(descriptor)
    # This deliberately documents the first-stage transport contract (#243).
    assert b"explicit-secret-sentinel" in payload
    assert "explicit-secret-sentinel" not in repr(descriptor)
    assert "explicit-secret-sentinel" not in repr(pickle.loads(payload))
    assert "api_key" not in descriptor.get_options()


def test_provider_snapshots_once_and_descriptors_do_not_share_mutable_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "first-key")
    provider = load_provider("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "second-key")
    first = provider.get_text_embedder()
    second = provider.get_prompter()
    first.client_options["base_url"] = "https://changed.example/v1"
    assert second.client_options["base_url"] == "https://api.openai.com/v1"
    assert b"first-key" in pickle.dumps(second)
    assert b"second-key" not in pickle.dumps(second)


def test_openai_endpoint_metadata_matches_captured_client_configuration(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://compatible.example/v1")
    with pytest.raises(ValueError, match="pass dimensions"):
        load_provider("openai").get_text_embedder()
    descriptor = load_provider("openai").get_text_embedder(dimensions=7)
    assert descriptor.get_dimensions() == 7
    assert descriptor.options["base_url"] == "https://compatible.example/v1"
    assert load_provider("openai", base_url="https://api.openai.com/v1").get_text_embedder().get_dimensions() == 1536


@pytest.mark.parametrize("mode_variable", ["GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE"])
def test_google_mode_and_cloud_coordinates_are_captured(monkeypatch, mode_variable):
    monkeypatch.setenv(mode_variable, "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "application-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west4")
    options = capture_google_client(api_key="key")
    assert options["vertexai"] is True
    assert options["project"].reveal() == "application-project"
    assert options["location"] == "europe-west4"
    assert options["base_url"] == "https://europe-west4-aiplatform.googleapis.com/"
    developer = capture_google_client(api_key="key", vertexai=False)
    assert developer["vertexai"] is False
    assert developer["project"] is None
    assert developer["location"] is None


@pytest.mark.parametrize("operation", ["embed", "prompt"])
def test_google_explicit_vertex_credentials_survive_worker_key_and_mode(monkeypatch, operation):
    pytest.importorskip("google.genai")
    from google.oauth2.credentials import Credentials

    descriptor = _descriptor(
        "google",
        operation,
        vertexai=True,
        project="application-project",
        location="europe-west4",
        credentials=Credentials(token="application-access-token"),
    )
    payload = pickle.dumps(descriptor)
    monkeypatch.setenv("GOOGLE_API_KEY", "worker-key")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "worker-project")
    state = _sdk_state(payload)
    assert state["vertexai"] is True
    assert state["api_key"] is None
    assert state["project"] == "application-project"
    assert state["location"] == "europe-west4"


@pytest.mark.parametrize("operation", ["embed", "prompt"])
def test_google_vertex_api_key_does_not_inherit_worker_project(monkeypatch, operation):
    pytest.importorskip("google.genai")
    payload = pickle.dumps(_descriptor("google", operation, api_key="application-key", vertexai=True))
    for key, value in WORKER_ENV.items():
        monkeypatch.setenv(key, value)
    state = _sdk_state(payload)
    assert state["vertexai"] is True
    assert state["api_key"] == "application-key"
    assert state["project"] is None


def test_anthropic_auth_token_isolated_from_worker_api_key(monkeypatch):
    pytest.importorskip("anthropic")
    payload = pickle.dumps(_descriptor("anthropic", "prompt", auth_token="application-token"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "worker-key")
    state = _sdk_state(payload)
    assert state["api_key"] is None
    assert state["auth_token"] == "application-token"
    assert state["headers"]["Authorization"] == "Bearer application-token"


def test_concurrent_sdk_construction_keeps_credentials_and_organization_isolated(monkeypatch):
    pytest.importorskip("openai")
    payloads = [
        pickle.dumps(_descriptor("openai", "prompt", api_key=f"key-{i}", organization=f"org-{i}", project=f"proj-{i}"))
        for i in range(4)
    ]
    for key, value in WORKER_ENV.items():
        monkeypatch.setenv(key, value)
    before = dict(os.environ)
    with ThreadPoolExecutor(max_workers=4) as pool:
        states = list(pool.map(_sdk_state, payloads))
    assert dict(os.environ) == before
    for i, state in enumerate(states):
        assert (state["api_key"], state["organization"], state["project"]) == (f"key-{i}", f"org-{i}", f"proj-{i}")


def test_failed_client_construction_does_not_mutate_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "worker-key")
    before = dict(os.environ)

    def fail(**kwargs):
        assert kwargs["api_key"] == "application-key"
        raise RuntimeError("constructor failed")

    with pytest.raises(RuntimeError, match="constructor failed"):
        create_openai_client(fail, capture_openai_client(api_key="application-key"), {})
    assert dict(os.environ) == before


@pytest.mark.parametrize(
    "family,settings",
    [
        ("openai", {"api_key": "secret\nvalue"}),
        ("openai", {"base_url": "https://user:secret@example.com"}),
        ("google", {"vertexai": "true"}),
        ("google", {"vertexai": False, "project": "project"}),
        ("google", {"vertexai": True, "api_key": "secret", "credentials": object()}),
        ("anthropic", {"api_key": "secret", "auth_token": "secret"}),
    ],
)
def test_invalid_client_settings_fail_without_echoing_secrets(family, settings):
    with pytest.raises(ValueError) as caught:
        load_provider(family, **settings)
    assert "secret" not in str(caught.value)


@contextmanager
def _model_server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, dict(self.headers), body))
            if "embeddings" in self.path:
                result = {
                    "data": [{"index": 0, "embedding": [1.0, 2.0]}],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            elif "batchEmbedContents" in self.path:
                result = {"embeddings": [{"values": [1.0] * 128} for _ in body["requests"]]}
            elif "embedContent" in self.path:
                result = {"embedding": {"values": [1.0] * 128}}
            elif "generateContent" in self.path:
                result = {
                    "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}]
                }
            elif "messages" in self.path:
                result = {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            else:
                result = {
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 0,
                    "status": "completed",
                    "model": "gpt-4.1",
                    "output": [
                        {
                            "id": "msg_test",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                        }
                    ],
                }
            data = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _worker_environment():
    return {key: os.environ.get(key) for key in CLIENT_ENV}


@pytest.mark.parametrize("operation", ["embed", "prompt"])
@pytest.mark.filterwarnings(
    "ignore:Inheritance class AiohttpClientSession from ClientSession is discouraged:DeprecationWarning"
)
def test_vertex_requests_use_captured_oauth_credentials_and_project(monkeypatch, operation):
    pytest.importorskip("google.genai")
    from google.oauth2.credentials import Credentials

    async def request(descriptor):
        runtime = pickle.loads(pickle.dumps(descriptor)).instantiate()
        try:
            if operation == "embed":
                result = await runtime.embed_text(["hello"])
                assert len(result[0]) == 128
            else:
                assert await runtime.prompt(("hello",)) == "ok"
        finally:
            await runtime.aclose()

    with _model_server() as (endpoint, requests):
        descriptor = _descriptor(
            "google",
            operation,
            vertexai=True,
            project="application-project",
            location="us-central1",
            credentials=Credentials(token="application-oauth-token"),
            base_url=endpoint,
        )
        for key, value in WORKER_ENV.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "false")
        asyncio.run(request(descriptor))
    assert len(requests) == 1
    path, headers, _ = requests[0]
    headers = {key.lower(): value for key, value in headers.items()}
    assert headers["authorization"] == "Bearer application-oauth-token"
    assert "x-goog-api-key" not in headers
    assert "projects/application-project/locations/us-central1/" in path


@pytest.mark.real_ray
@pytest.mark.ray_cluster_owner
@pytest.mark.parametrize("conflicting_workers", [False, True])
def test_default_ray_existing_cluster_uses_application_clients(conflicting_workers):
    for sdk in ("ray", "openai", "google.genai", "anthropic"):
        pytest.importorskip(sdk)
    # Vane relation destructors can outlive ray.shutdown(). A separate driver
    # process per scenario keeps cluster startup and worker environments exact.
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; sys.path.append(sys.argv[1]); runpy.run_path(sys.argv[2], run_name='__main__')",
            str(Path(__file__).resolve().parents[1]),
            __file__,
            str(conflicting_workers),
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _run_existing_cluster_case(monkeypatch, conflicting_workers):
    ray = pytest.importorskip("ray")
    for sdk in ("openai", "google.genai", "anthropic"):
        pytest.importorskip(sdk)
    from ray_test_profile import ray_test_object_store_options

    import vane

    monkeypatch.delenv("VANE_RUNNER", raising=False)
    try:
        # Start Ray before the application has any provider credentials.
        with monkeypatch.context() as worker_environment:
            if conflicting_workers:
                for key, value in WORKER_ENV.items():
                    worker_environment.setenv(key, value)
            ray.init(address="local", num_cpus=2, include_dashboard=False, **ray_test_object_store_options())
            initial = ray.get(ray.remote(_worker_environment).remote())
            assert initial["GOOGLE_API_KEY"] == ("worker-google-key" if conflicting_workers else None)
        with _model_server() as (endpoint, requests):
            for family in ("openai", "google", "anthropic"):
                monkeypatch.setenv(f"{family.upper()}_API_KEY", f"application-{family}-key")
            monkeypatch.setenv("OPENAI_BASE_URL", endpoint + "/v1")
            monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", endpoint)
            monkeypatch.setenv("ANTHROPIC_BASE_URL", endpoint)
            with vane.connect() as connection:
                for backend in ("subprocess_task", "subprocess_actor", "ray_task", "ray_actor"):
                    for family, operation in CASES:
                        source = connection.sql("SELECT 'hello' AS text")
                        if operation == "embed":
                            result = vane.ai.embed(
                                source,
                                vane.col("text"),
                                output_column="result",
                                provider=family,
                                dimensions=2 if family == "openai" else 128,
                                execution_backend=backend,
                                max_retries=0,
                            )
                        else:
                            result = vane.ai.prompt(
                                source,
                                vane.col("text"),
                                output_column="result",
                                provider=family,
                                model={"openai": "gpt-4.1", "google": "gemini-2.5-flash", "anthropic": "claude-test"}[
                                    family
                                ],
                                execution_backend=backend,
                                max_retries=0,
                                **({"max_tokens": 32} if family == "anthropic" else {}),
                            )
                        value = result.select("result").fetchone()[0]
                        if operation == "prompt":
                            assert value == "ok"
                        else:
                            assert len(value) == (2 if family == "openai" else 128)
                # SQL binds credentials on the application just like expressions.
                assert connection.sql("""
                    SELECT ai_prompt('hello', provider := 'google', model := 'gemini-2.5-flash',
                        options := {'max_retries': 0})
                """).fetchone() == ("ok",)
                assert (
                    len(
                        connection.sql("""
                    SELECT ai_embed('hello', provider := 'openai', dimensions := 2,
                        options := {'max_retries': 0})
                """).fetchone()[0]
                    )
                    == 2
                )
                # Explicit providers retain separate identities from each other and the environment.
                for account in ("a", "b"):
                    provider = load_provider(
                        "openai",
                        api_key=f"account-{account}-key",
                        base_url=endpoint + "/v1",
                        organization=f"org-{account}",
                        project=f"proj-{account}",
                    )
                    result = connection.sql("SELECT 'hello' AS text").select(
                        vane.ai.prompt(
                            vane.col("text"),
                            provider=provider,
                            model="gpt-4.1",
                            max_retries=0,
                        )
                    )
                    assert result.fetchone() == ("ok",)
            assert len(requests) == 24
            for path, headers, body in requests:
                headers = {key.lower(): value for key, value in headers.items()}
                assert "worker" not in repr(headers)
                if "models/" in path:
                    assert headers["x-goog-api-key"] == "application-google-key"
                    assert "projects/" not in path
                elif "messages" in path:
                    assert headers["x-api-key"] == "application-anthropic-key"
                    assert "authorization" not in headers
                else:
                    if headers["authorization"].startswith("Bearer account-"):
                        account = headers["authorization"].split("-")[1]
                        assert headers["openai-project"] == f"proj-{account}"
                        assert headers["openai-organization"] == f"org-{account}"
                    else:
                        assert headers["authorization"] == "Bearer application-openai-key"
                        assert "openai-project" not in headers
                        assert "openai-organization" not in headers
    finally:
        vane.teardown_runner()
        ray.shutdown()


if __name__ == "__main__":
    with pytest.MonkeyPatch.context() as patch:
        _run_existing_cluster_case(patch, sys.argv[-1] == "True")
