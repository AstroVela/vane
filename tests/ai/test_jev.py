# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real async TypeSafe SDK without calling the hosted service."""

from __future__ import annotations

import asyncio
import json
import pickle
import sys
import threading
import traceback
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pyarrow as pa
import pytest

import vane
from vane.ai import jev
from vane.ai._jev import _JevBatch, _prepare_options, _prepare_questions, _serialize_response
from vane.ai.provider import ProviderImportError
from vane.ai.typing import UDFOptions
from vane.execution._async_runtime import AsyncRuntime

sdk = pytest.importorskip("typesafe_sdk")
httpx = pytest.importorskip("httpx2")

QUESTIONS = {
    "billing": {"type": "noul", "instructions": "Is this ticket about billing?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {"billing": None, "technical": None},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgently does this ticket need attention?",
        "criteria": ["Can wait", "Needs attention today"],
    },
}


def _response():
    return {
        "model": "jev-test",
        "usage": {"input_tokens": 40, "output_tokens": 8},
        "answers": {
            "billing": {"type": "noul", "noul": 0.8},
            "team": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.8, "technical": 0.2},
                "confidence": 0.6,
            },
            "urgency": {
                "type": "score",
                "score": 0.8,
                "legend": {"0": "Can wait", "1": "Needs attention today"},
                "probabilities": {"0": 0.2, "1": 0.8},
                "confidence": 0.6,
            },
        },
    }


@contextmanager
def _batch(**options):
    runtime = AsyncRuntime()
    wrapper = _JevBatch(
        QUESTIONS,
        "jev-test",
        {"timeout": 5.0, "api_key": "local-test"},
        UDFOptions(max_concurrency_per_actor=2, **options),
    )
    wrapper.bind_async_runtime(runtime.run)
    try:
        yield wrapper, runtime
    finally:
        try:
            wrapper.close()
        finally:
            runtime.close()


def _drive(wrapper, states):
    table = pa.table({"state": pa.array([json.dumps(s) if s is not None else None for s in states], type=pa.string())})
    return [json.loads(value) if value is not None else None for value in wrapper(table)["response"].to_pylist()]


def _install_transport(monkeypatch, handler):
    clients = []

    class Client(sdk.AsyncTypeSafeClient):
        def __init__(self, **kwargs):
            self.created_on = asyncio.get_running_loop()
            self.closed_on = None
            self.options = kwargs
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)
            clients.append(self)

        async def aclose(self):
            self.closed_on = asyncio.get_running_loop()
            await super().aclose()

    monkeypatch.setattr(sdk, "AsyncTypeSafeClient", Client)
    return clients


def test_sdk_batches_reuse_client_and_loop_with_bounded_concurrency(monkeypatch):
    active, peak = 0, 0
    calls = []

    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        body = json.loads(request.content)
        calls.append((body, asyncio.get_running_loop()))
        try:
            await asyncio.sleep(0.01 if body["state"] == "slow" else 0)
            payload = _response()
            payload["usage"]["input_tokens"] = len(body["state"])
            return httpx.Response(200, json=payload)
        finally:
            active -= 1

    clients = _install_transport(monkeypatch, handler)
    with _batch() as (wrapper, runtime):
        first = _drive(wrapper, ["slow", None, "a", "bb", "ccc"])
        second = _drive(wrapper, ["next batch"])
        assert [item["usage"]["input_tokens"] if item else None for item in first] == [4, None, 1, 2, 3]
        assert second[0]["usage"]["input_tokens"] == 10
        assert peak == 2
        assert len(clients) == 1
        assert all(loop is runtime.loop for _, loop in calls)
        assert clients[0].created_on is runtime.loop
        assert clients[0].options["retry"].max_retries == 3
        assert clients[0].options["timeout"] == 5.0
        assert all(body["questions"] == QUESTIONS and body["model"] == "jev-test" for body, _ in calls)
        wrapper.close()
        wrapper.close()
        assert clients[0].closed_on is runtime.loop
        # After teardown, the same wrapper can be bound to a fresh executor.
        runtime.close()
        _drive(wrapper, ["again"])
        assert len(clients) == 2
        assert clients[1].created_on is runtime.loop
        assert clients[1].created_on is not clients[0].created_on


@pytest.mark.parametrize("status,expected", [(429, 2), (529, 2), (401, 1), (422, 1)])
def test_sdk_owns_retries_and_failed_rows_become_null(monkeypatch, status, expected, caplog):
    calls = []
    secret = "private-request-and-api-key"

    async def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if body["state"] == "fail":
            return httpx.Response(status, json={"message": secret}, headers={"Retry-After": "0"})
        return httpx.Response(200, json=_response())

    _install_transport(monkeypatch, handler)
    with _batch(max_retries=1, on_error="ignore") as (wrapper, _):
        results = _drive(wrapper, ["ok", "fail", None])
        assert results[0] == _response()
        assert results[1:] == [None, None]
        assert sum(body["state"] == "fail" for body in calls) == expected
    assert secret not in caplog.text


def test_failure_drains_other_requests_and_redacts_error(monkeypatch):
    slow_started = asyncio.Event()
    cancelled = []

    async def handler(request):
        if json.loads(request.content)["state"] == "fail":
            await slow_started.wait()
            return httpx.Response(401, json={"message": "secret-body-never-expose"})
        slow_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    _install_transport(monkeypatch, handler)
    with _batch(max_retries=0) as (wrapper, runtime):
        with pytest.raises(RuntimeError, match="Jev execution.*status=401") as caught:
            _drive(wrapper, ["fail", "slow"])
        assert "secret-body-never-expose" not in "".join(traceback.format_exception(caught.value))
        assert caught.value.__context__ is None
        assert cancelled == [True]
        assert not asyncio.all_tasks(runtime.loop)


def test_pickle_drops_client_and_runtime(monkeypatch):
    async def handler(request):
        return httpx.Response(200, json=_response())

    clients = _install_transport(monkeypatch, handler)
    with _batch() as (wrapper, _):
        _drive(wrapper, ["first"])
        restored = pickle.loads(pickle.dumps(wrapper))
        assert restored._client is None
        assert restored._run_async is None
        fresh_runtime = AsyncRuntime()
        try:
            restored.bind_async_runtime(fresh_runtime.run)
            assert _drive(restored, ["second"]) == [_response()]
            assert len(clients) == 2
        finally:
            restored.close()
            fresh_runtime.close()


def test_null_and_empty_batches_do_not_require_client_or_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    wrapper = _JevBatch(QUESTIONS, "jev-test", {}, UDFOptions(max_concurrency_per_actor=1))
    assert _drive(wrapper, [None, None]) == [None, None]
    assert _drive(wrapper, []) == []
    assert wrapper(pa.table({"state": ["null"]}))["response"].to_pylist() == [None]
    wrapper.close()
    with pytest.raises(RuntimeError, match="bind_async_runtime"):
        _drive(wrapper, ["text"])


@pytest.mark.parametrize("state", [42, True, {"value": float("nan")}, [float("inf")]])
@pytest.mark.parametrize("on_error", ["raise", "ignore"])
def test_invalid_state_is_rejected_without_network(state, on_error):
    wrapper = _JevBatch(QUESTIONS, "jev-test", {}, UDFOptions(on_error=on_error, max_concurrency_per_actor=1))
    if on_error == "raise":
        with pytest.raises(ValueError, match="Jev state must be"):
            _drive(wrapper, [state])
    else:
        assert _drive(wrapper, [state]) == [None]


def test_client_snapshot_survives_conflicting_worker_environment(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "application-secret")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://application.invalid")
    udf_options, options = _prepare_options({}, "raise")
    assert "application-secret" not in repr(options)
    wrapper = pickle.loads(pickle.dumps(_JevBatch(QUESTIONS, "jev-test", options, udf_options)))
    monkeypatch.setenv("TYPESAFE_API_KEY", "worker-secret")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://worker.invalid")
    seen = []

    async def handler(request):
        seen.append((request.url.host, request.headers["Authorization"]))
        return httpx.Response(200, json=_response())

    _install_transport(monkeypatch, handler)
    runtime = AsyncRuntime()
    wrapper.bind_async_runtime(runtime.run)
    try:
        assert _drive(wrapper, ["text"]) == [_response()]
        assert seen == [("application.invalid", "Bearer application-secret")]
    finally:
        wrapper.close()
        runtime.close()


@pytest.mark.parametrize("on_error", ["raise", "ignore"])
def test_missing_application_key_never_uses_worker_credentials(monkeypatch, on_error):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    udf_options, options = _prepare_options({}, on_error)
    wrapper = _JevBatch(QUESTIONS, "jev-test", options, udf_options)
    monkeypatch.setenv("TYPESAFE_API_KEY", "worker-secret")
    runtime = AsyncRuntime()
    wrapper.bind_async_runtime(runtime.run)
    try:
        with pytest.raises(RuntimeError, match="API key was not configured on the application"):
            _drive(wrapper, ["text"])
    finally:
        wrapper.close()
        runtime.close()


def test_question_objects_are_copied_and_no_client_is_created(monkeypatch):
    def unexpected_client(**kwargs):
        pytest.fail("Expression construction must not instantiate an SDK client")

    monkeypatch.setattr(sdk, "AsyncTypeSafeClient", unexpected_client)
    question = sdk.Choice(instructions="Choose", criteria={"one": {"detail": "first"}, "two": None})
    original = {"q": question}
    prepared = _prepare_questions(original)
    question.criteria["one"]["detail"] = "changed"
    original.clear()
    assert prepared["q"]["criteria"]["one"] == {"detail": "first"}
    assert isinstance(jev(vane.col("text"), questions=prepared), vane.Expression)


@pytest.mark.parametrize(
    "questions",
    [
        {},
        [],
        {1: QUESTIONS["billing"]},
        {"": QUESTIONS["billing"]},
        {"q": {"type": "unknown"}},
        {"q": {"type": "choice"}},
    ],
)
def test_invalid_questions_fail_before_execution(questions):
    with pytest.raises((TypeError, ValueError)):
        jev(vane.col("text"), questions=questions)


@pytest.mark.parametrize(
    "options",
    [
        {"batch_size": 0},
        {"actor_number": True},
        {"max_concurrency_per_actor": -1},
        {"max_retries": -1},
        {"timeout": 0},
        {"timeout": float("nan")},
        {"base_url": "https://secret@example.com"},
        {"base_url": "https://example.com?api_key=secret"},
        {"api_key": "secret"},
        {"execution_backend": "invalid"},
        {"execution_backend": "subprocess_task", "actor_number": 2},
        {"model": ""},
        {"on_error": "invalid"},
    ],
)
def test_invalid_options_fail_before_execution(options):
    with pytest.raises((TypeError, ValueError)):
        jev(vane.col("text"), questions=QUESTIONS, **options)


def test_missing_sdk_has_optional_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    with pytest.raises(ProviderImportError, match=r"vane-ai\[typesafe\]"):
        jev(vane.col("text"), questions=QUESTIONS)


@pytest.mark.parametrize("bad", ["missing", "wrong_type", "wrong_choice", "wrong_levels"])
def test_answer_contract_is_checked(bad):
    payload = _response()
    if bad == "missing":
        del payload["answers"]["billing"]
    elif bad == "wrong_type":
        payload["answers"]["billing"] = payload["answers"]["team"]
    elif bad == "wrong_choice":
        payload["answers"]["team"]["choice"] = "invented"
    else:
        payload["answers"]["urgency"]["probabilities"] = {"0": 0.2, "2": 0.8}
    response = sdk.SystemOneResponse.model_validate_json(json.dumps(payload))
    with pytest.raises(ValueError, match="Jev"):
        _serialize_response(response, QUESTIONS)


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "local-jev-test")
    monkeypatch.setenv("TYPESAFE_LOG_LEVEL", "off")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, body, self.headers.get("Authorization")))
            response = json.dumps(_response()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    http_server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http_server.server_port}", calls
    finally:
        http_server.shutdown()
        http_server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("api", ["expression", "relation", "method"])
@pytest.mark.parametrize("backend", ["subprocess_actor", "subprocess_task"])
def test_real_sdk_through_vane_worker(server, api, backend):
    url, calls = server
    with vane.connect() as connection:
        source = connection.sql("SELECT * FROM (VALUES (1, 'first'), (2, NULL), (3, 'second')) AS t(id, text)")
        options = {"questions": QUESTIONS, "base_url": url, "execution_backend": backend, "max_retries": 0}
        if api == "expression":
            result = source.select(vane.col("id"), jev(state=vane.col("text"), **options).alias("result"))
        elif api == "relation":
            result = jev(rel=source, state=vane.col("text"), output_column="result", **options)
        else:
            result = source.jev(vane.col("text"), output_column="result", **options)
        assert str(result.types[-1]) == "VARCHAR"
        rows = (
            result.order("id")
            .select(vane.col("id"), vane.sql_expr("result ->> '$.answers.team.choice'").alias("team"))
            .fetchall()
        )
        assert rows == [(1, "billing"), (2, None), (3, "billing")]
        assert sorted(body["state"] for _, body, _ in calls) == ["first", "second"]
        assert all(path == "/v1/systemone" for path, _, _ in calls)
        assert all(auth == "Bearer local-jev-test" for _, _, auth in calls)
        assert all(body["questions"] == QUESTIONS for _, body, _ in calls)


@pytest.mark.real_ray
@pytest.mark.parametrize("backend", [None, "ray_task"])
def test_ray_transports_json_text_and_application_credentials(ray_local, server, monkeypatch, backend):
    url, calls = server
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    with vane.connect() as connection:
        source = connection.sql("SELECT * FROM (VALUES (1, 'first'), (2, NULL), (3, 'second')) AS t(id, text)")
        result = source.jev(
            vane.col("text"), questions=QUESTIONS, base_url=url, batch_size=2, max_retries=0, execution_backend=backend
        )
        assert ("ray_actor" if backend is None else backend) in result.explain()
        # Node workers were started by ray_local before this server fixture
        # configured credentials. The expression must carry its own snapshot.
        monkeypatch.setenv("TYPESAFE_API_KEY", "changed-after-binding")
        rows = result.order("id").fetchall()
        assert [json.loads(row[-1]) if row[-1] else None for row in rows] == [_response(), None, _response()]
        assert all(auth == "Bearer local-jev-test" for _, _, auth in calls)
        assert len(calls) == 2


@pytest.mark.parametrize(
    "expression,state",
    [
        ("struct_pack(text := 'charged twice', count := 2)", {"text": "charged twice", "count": 2}),
        ("['one', 'two']", ["one", "two"]),
        ('\'{"text":"hello"}\'::JSON', {"text": "hello"}),
        ('\'{"text":"hello"}\'::VARCHAR', '{"text":"hello"}'),
    ],
)
def test_structured_state_and_output_column_replacement(server, expression, state):
    url, calls = server
    # Retaining source JSON columns across any Python UDF requires the engine's
    # canonical Arrow extension metadata.
    with vane.connect(config={"arrow_lossless_conversion": True}) as connection:
        source = connection.sql(f"SELECT {expression} AS state, 'old' AS RESPONSE")
        result = jev(source, vane.col("state"), questions=QUESTIONS, base_url=url)
        assert result.columns == ["state", "response"]
        assert json.loads(result.fetchall()[0][-1]) == _response()
        assert calls[0][1]["state"] == state
