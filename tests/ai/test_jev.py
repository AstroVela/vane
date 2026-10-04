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


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "wrong_type",
        "wrong_choice",
        "wrong_levels",
        "wrong_legend",
        "choice_total",
        "score_total",
        "choice_argmax",
        "score_mean",
    ],
)
def test_answer_contract_is_checked(bad):
    payload = _response()
    if bad == "missing":
        del payload["answers"]["billing"]
    elif bad == "wrong_type":
        payload["answers"]["billing"] = payload["answers"]["team"]
    elif bad == "wrong_choice":
        payload["answers"]["team"]["choice"] = "invented"
    elif bad == "wrong_levels":
        payload["answers"]["urgency"]["probabilities"] = {"0": 0.2, "2": 0.8}
    elif bad == "wrong_legend":
        payload["answers"]["urgency"]["legend"] = {"0": "Needs attention today", "1": "Can wait"}
    elif bad == "choice_total":
        payload["answers"]["team"]["probabilities"] = {"billing": 0.8, "technical": 0.8}
    elif bad == "score_total":
        payload["answers"]["urgency"]["probabilities"] = {"0": 0.8, "1": 0.8}
    elif bad == "choice_argmax":
        payload["answers"]["team"]["choice"] = "technical"
    else:
        payload["answers"]["urgency"]["score"] = 0.3
    response = sdk.SystemOneResponse.model_validate_json(json.dumps(payload))
    with pytest.raises(ValueError, match="Jev"):
        _serialize_response(response, QUESTIONS)


@pytest.mark.parametrize("question,level", [("team", "technical"), ("urgency", "1")])
def test_probability_validation_accepts_rounding(question, level):
    payload = _response()
    payload["answers"][question]["probabilities"][level] += 1e-7
    response = sdk.SystemOneResponse.model_validate_json(json.dumps(payload))
    assert json.loads(_serialize_response(response, QUESTIONS)) == payload


@pytest.mark.parametrize("other_probability", [0.33, 0.34])
def test_probability_validation_bounds_hundredth_rounding(other_probability):
    questions = {
        **QUESTIONS,
        "team": {**QUESTIONS["team"], "criteria": {"billing": None, "technical": None, "other": None}},
    }
    payload = _response()
    payload["answers"]["team"]["probabilities"] = {
        "billing": 0.34,
        "technical": 0.34,
        "other": other_probability,
    }
    response = sdk.SystemOneResponse.model_validate_json(json.dumps(payload))
    if other_probability == 0.33:
        assert json.loads(_serialize_response(response, questions)) == payload
    else:
        with pytest.raises(ValueError, match="sum to 1"):
            _serialize_response(response, questions)


@pytest.mark.parametrize(
    "score,valid",
    [(0.87, True), (0.855, True), (0.925, True), (0.854, False), (0.926, False), (0.84, False), (0.83, False)],
)
def test_score_validation_bounds_independently_rounded_live_response(score, valid):
    # A real Jev 1.13 response reports 0.87 while its rounded probabilities
    # yield 0.89. Preserve the service's score rather than recomputing it.
    criteria = ["Neutral", "Concerned", "Dissatisfied", "Angry", "Extremely angry"]
    questions = {**QUESTIONS, "urgency": {**QUESTIONS["urgency"], "criteria": criteria}}
    payload = _response()
    payload["answers"]["urgency"].update(
        score=score,
        probabilities={"0": 0.43, "1": 0.34, "2": 0.15, "3": 0.07, "4": 0.01},
        legend={str(index): text for index, text in enumerate(criteria)},
    )
    response = sdk.SystemOneResponse.model_validate_json(json.dumps(payload))
    if valid:
        assert json.loads(_serialize_response(response, questions)) == payload
    else:
        with pytest.raises(ValueError, match="probability-weighted"):
            _serialize_response(response, questions)


@pytest.mark.parametrize(
    "probabilities,score,valid",
    [
        ({"2": 0.33, "1": 0.34, "0": 0.34}, 1.00, True),
        ({"2": 0.33, "1": 0.34, "0": 0.34}, 1.01, False),
        ({"0": 1.0, "1": 0.0, "2": 0.0}, 0.01, True),
        ({"0": 1.0, "1": 0.0, "2": 0.0}, 0.02, False),
        # Captured service responses outside the normalized rounding interval.
        ({"0": 0.52, "1": 0.43, "2": 0.03, "3": 0.0, "4": 0.02}, 0.61, False),
        ({"0": 0.45, "1": 0.38, "2": 0.14, "3": 0.03, "4": 0.0}, 0.79, False),
    ],
)
def test_score_rounding_respects_normalization_and_probability_bounds(probabilities, score, valid):
    criteria = [f"Level {index}" for index in range(len(probabilities))]
    questions = {**QUESTIONS, "urgency": {**QUESTIONS["urgency"], "criteria": criteria}}
    payload = _response()
    payload["answers"]["urgency"].update(
        score=score,
        probabilities=probabilities,
        legend={str(index): text for index, text in enumerate(criteria)},
    )
    response = sdk.SystemOneResponse.model_validate_json(json.dumps(payload))
    if valid:
        assert json.loads(_serialize_response(response, questions)) == payload
    else:
        with pytest.raises(ValueError, match="probability-weighted"):
            _serialize_response(response, questions)


def test_probability_rounding_cannot_subtract_mass_from_zero_entries():
    probabilities = {"billing": 0.51, "technical": 0.51, "sales": 0.0, "fraud": 0.0, "other": 0.0}
    questions = {**QUESTIONS, "team": {**QUESTIONS["team"], "criteria": dict.fromkeys(probabilities)}}
    payload = _response()
    payload["answers"]["team"]["probabilities"] = probabilities
    response = sdk.SystemOneResponse.model_validate_json(json.dumps(payload))
    with pytest.raises(ValueError, match="sum to 1"):
        _serialize_response(response, questions)


def test_choice_validation_accepts_tied_winners():
    payload = _response()
    payload["answers"]["team"].update(
        choice="technical", probabilities={"billing": 0.5, "technical": 0.5}, confidence=0
    )
    response = sdk.SystemOneResponse.model_validate_json(json.dumps(payload))
    assert json.loads(_serialize_response(response, QUESTIONS)) == payload


def test_confidence_is_preserved_without_assuming_a_top_two_margin():
    # The official confidence explorer uses this three-option distribution
    # with an illustrative confidence of 0.1, not its top-two margin of 0.07.
    # The docs call that formula an approximation, so the service owns it.
    # https://docs.typesafe.ai/confidence
    questions = {
        **QUESTIONS,
        "team": {**QUESTIONS["team"], "criteria": {"billing": None, "technical": None, "other": None}},
    }
    payload = _response()
    payload["answers"]["team"].update(probabilities={"billing": 0.4, "technical": 0.33, "other": 0.27}, confidence=0.1)
    response = sdk.SystemOneResponse.model_validate_json(json.dumps(payload))
    assert json.loads(_serialize_response(response, questions)) == payload


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


_SQL_QUESTIONS = "'" + json.dumps(QUESTIONS).replace("'", "''") + "'"


@pytest.mark.parametrize(
    "expression,state",
    [
        ("'charged twice'", "charged twice"),
        (
            "struct_pack(text := 'charged twice', count := 2, ratio := 0.25)",
            {"text": "charged twice", "count": 2, "ratio": 0.25},
        ),
        ("['one', 'two']", ["one", "two"]),
        ("[1, 2]::INTEGER[2]", [1, 2]),
        ('\'{"text":"hello"}\'::JSON', {"text": "hello"}),
        ('\'{"text":"hello"}\'::VARCHAR', '{"text":"hello"}'),
    ],
)
def test_sql_preserves_state_types_and_null_rows(server, expression, state):
    url, calls = server
    with vane.connect() as connection:
        result = connection.sql(
            f"""SELECT id, ai_jev(state, questions := {_SQL_QUESTIONS}, model := 'jev-test',
                       options := struct_pack(base_url := '{url}', batch_size := 1,
                                              actor_number := 1, max_concurrency_per_actor := 2,
                                              max_retries := 0, timeout := 5.0)) AS judgment
                FROM (VALUES (1, {expression}), (2, NULL)) AS t(id, state)"""
        )
        assert str(result.types[-1]) == "VARCHAR"
        rows = result.order("id").fetchall()
        assert json.loads(rows[0][1]) == _response()
        assert rows[1] == (2, None)
    assert len(calls) == 1
    path, body, auth = calls[0]
    assert path == "/v1/systemone"
    assert body == {"state": state, "questions": QUESTIONS, "model": "jev-test"}
    assert auth == "Bearer local-jev-test"


def test_sql_struct_questions_preserve_nested_json_numbers(server):
    url, calls = server
    with vane.connect() as connection:
        result = connection.sql(
            f"""SELECT ai_jev('hello', questions := struct_pack(
                    billing := struct_pack(type := 'noul',
                        instructions := struct_pack(text := 'Is this about billing?', threshold := 0.25)),
                    team := struct_pack(type := 'choice', instructions := 'Which team?',
                        criteria := struct_pack(billing := NULL, technical := NULL)),
                    urgency := struct_pack(type := 'score', instructions := 'How urgent?',
                        criteria := ['Can wait', 'Needs attention today'])),
                    options := struct_pack(base_url := '{url}', max_retries := 0))"""
        )
        assert json.loads(result.fetchall()[0][0]) == _response()
    assert calls[0][1]["questions"]["billing"]["instructions"] == {"text": "Is this about billing?", "threshold": 0.25}
    assert calls[0][1]["model"] == "jev-latest"


@pytest.mark.parametrize("questions_cast", ["", "::JSON"])
def test_sql_prepared_parameters_and_json_projection(server, questions_cast):
    url, calls = server
    with vane.connect() as connection:
        result = connection.sql(
            f"""SELECT judgment ->> '$.answers.team.choice' AS team,
                       (judgment ->> '$.answers.billing.noul')::DOUBLE AS probability
                FROM (SELECT ai_jev(state := ?, questions := ?{questions_cast}, model := ?,
                           on_error := ?, options := struct_pack(base_url := ?, max_retries := 0)) AS judgment)""",
            params=["a customer's invoice", json.dumps(QUESTIONS), "jev-test", "raise", url],
        )
        assert result.fetchall() == [("billing", 0.8)]
    assert len(calls) == 1
    assert calls[0][1]["state"] == "a customer's invoice"


@pytest.mark.parametrize("argument", ["questions", "model", "on_error", "options"])
def test_sql_prepare_defers_unresolved_configuration(server, argument):
    url, calls = server
    arguments = {
        "questions": _SQL_QUESTIONS,
        "model": "'jev-test'",
        "on_error": "'raise'",
        "options": f"struct_pack(base_url := '{url}', max_retries := 0)",
    }
    value = arguments[argument]
    arguments[argument] = "$1"
    named = ", ".join(f"{key} := {item}" for key, item in arguments.items())
    with vane.connect() as connection:
        connection.execute(f"PREPARE jev_query AS SELECT ai_jev('hello', {named})")
        assert calls == []
        for _ in range(2):
            row = connection.execute(f"EXECUTE jev_query({value})").fetchone()
            assert json.loads(row[0]) == _response()
    assert len(calls) == 2


def test_sql_cached_prepare_recreates_query_scoped_actors(server):
    url, calls = server
    with vane.connect() as connection:
        connection.execute(
            f"""PREPARE jev_query AS SELECT ai_jev('hello', {_SQL_QUESTIONS},
                    options := struct_pack(base_url := '{url}', max_retries := 0))"""
        )
        assert calls == []
        for _ in range(2):
            assert json.loads(connection.execute("EXECUTE jev_query").fetchone()[0]) == _response()
    assert len(calls) == 2


def test_sql_executemany_recreates_query_scoped_actors(server):
    url, calls = server
    with vane.connect() as connection:
        connection.executemany(
            f"""SELECT ai_jev(?, {_SQL_QUESTIONS},
                    options := struct_pack(base_url := '{url}', max_retries := 0))""",
            [["first"], ["second"]],
        )
        assert json.loads(connection.fetchone()[0]) == _response()
    assert [body["state"] for _, body, _ in calls] == ["first", "second"]


def test_sql_empty_input_needs_no_key_or_request(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    with vane.connect() as connection:
        result = connection.sql(f"SELECT ai_jev('hello', {_SQL_QUESTIONS}) FROM range(0)")
        assert str(result.types[0]) == "VARCHAR"
        assert result.fetchall() == []


@pytest.mark.parametrize("state", ["NULL", "'null'::JSON"])
def test_sql_null_state_needs_no_key_or_request(monkeypatch, state):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    with vane.connect() as connection:
        result = connection.sql(f"SELECT ai_jev({state}, {_SQL_QUESTIONS}) AS judgment")
        assert str(result.types[0]) == "VARCHAR"
        assert result.fetchall() == [(None,)]
        if state == "NULL":
            assert "UDF" not in result.explain()


@pytest.mark.parametrize("on_error", ["raise", "ignore"])
def test_sql_invalid_state_obeys_row_error_policy(server, on_error):
    url, calls = server
    with vane.connect() as connection:
        result = connection.sql(
            f"""SELECT id, ai_jev(state, {_SQL_QUESTIONS}, on_error := '{on_error}',
                        options := struct_pack(base_url := '{url}', max_retries := 0)) AS judgment
                FROM (VALUES (1, '42'::JSON), (2, '"hello"'::JSON)) AS t(id, state)"""
        )
        if on_error == "raise":
            with pytest.raises(Exception, match="Jev state must be"):
                result.fetchall()
        else:
            rows = result.order("id").fetchall()
            assert rows[0] == (1, None)
            assert json.loads(rows[1][1]) == _response()
            assert len(calls) == 1


@pytest.mark.parametrize(
    "questions", ["NULL", "'{'", "'{}'", "'[]'", "42", "struct_pack(q := struct_pack(type := 'unknown'))"]
)
def test_sql_rejects_invalid_questions_during_planning(questions):
    with vane.connect() as connection:
        with pytest.raises((TypeError, ValueError), match="questions|question"):
            connection.sql(f"SELECT ai_jev(NULL, questions := {questions}, on_error := 'ignore')")


@pytest.mark.parametrize("argument", ["questions", "model", "on_error", "options"])
def test_sql_requires_constant_call_configuration(argument):
    arguments = {
        "questions": _SQL_QUESTIONS,
        "model": "'jev-test'",
        "on_error": "'raise'",
        "options": "struct_pack(batch_size := 2)",
    }
    row_value = arguments[argument]
    arguments[argument] = "configuration"
    named = ", ".join(f"{key} := {value}" for key, value in arguments.items())
    with vane.connect() as connection:
        with pytest.raises(vane.BinderException, match=f"'{argument}' must be constant"):
            connection.sql(f"SELECT ai_jev('hello', {named}) FROM (VALUES ({row_value})) t(configuration)")


@pytest.mark.parametrize(
    "options,error",
    [
        ("struct_pack(batch_size := 0)", "batch_size"),
        ("struct_pack(batch_size := 1.5)", "batch_size"),
        ("struct_pack(timeout := 0)", "timeout"),
        ("struct_pack(max_concurrency_per_actor := NULL)", "max_concurrency_per_actor"),
        ("struct_pack(unknown := NULL)", "Unsupported Jev"),
        ("struct_pack(execution_backend := 'ray_task')", "execution_backend"),
        ("struct_pack(api_key := 'test-only-secret')", "inline credential"),
        ("struct_pack(nested := struct_pack(api_key := 'test-only-secret'))", "inline credential"),
        ("struct_pack(base_url := 'https://secret@example.com')", "base_url"),
    ],
)
def test_sql_rejects_invalid_options_during_planning(options, error):
    with vane.connect() as connection:
        with pytest.raises((TypeError, ValueError), match=error):
            connection.sql(f"SELECT ai_jev(NULL, {_SQL_QUESTIONS}, options := {options}, on_error := 'ignore')")


@pytest.mark.parametrize("options", ["42", "'{}'", "[]"])
def test_sql_options_require_struct(options):
    with vane.connect() as connection:
        with pytest.raises(vane.BinderException, match="foldable STRUCT"):
            connection.sql(f"SELECT ai_jev(NULL, {_SQL_QUESTIONS}, options := {options})")


@pytest.mark.parametrize("argument", ["model := ''", "model := NULL", "on_error := 'invalid'", "on_error := NULL"])
def test_sql_rejects_invalid_model_and_error_policy(argument):
    with vane.connect() as connection:
        with pytest.raises((TypeError, ValueError), match="model|on_error"):
            connection.sql(f"SELECT ai_jev(NULL, {_SQL_QUESTIONS}, {argument})")


def test_sql_plan_round_trip_preserves_client_configuration(server, monkeypatch):
    from tests.ai.test_expression_ai_sql import _execute_ai_physical_plan, _round_trip_ai_plan

    url, calls = server
    with vane.connect() as connection:
        result = connection.sql(
            f"""SELECT id, ai_jev(struct_pack(text := text, id := id), {_SQL_QUESTIONS}, model := 'jev-test',
                       options := struct_pack(base_url := '{url}', batch_size := 2, max_retries := 0)) AS judgment
                FROM (VALUES (1, 'first'), (2, 'second')) t(id, text)"""
        )
        target, physical, serialized = _round_trip_ai_plan(result)
        try:
            monkeypatch.setenv("TYPESAFE_API_KEY", "changed-after-binding")
            monkeypatch.setenv("TYPESAFE_BASE_URL", "https://unreachable.invalid")
            node = physical.collect_udf_nodes()[0]
            payload = node["payload"]
            assert payload["input_names"] == ["state"]
            assert payload["ai_provider"] == "typesafe"
            assert payload["ai_model"] == "jev-test"
            assert payload["ai_return_type"] == "VARCHAR"
            assert payload["batch_size"] == 2
            table = _execute_ai_physical_plan(target, physical)
            assert [json.loads(value) for value in table.column(1).to_pylist()] == [_response(), _response()]
            assert 0 < len(serialized) < 1_000_000
        finally:
            target.close()
    assert sorted(body["state"]["id"] for _, body, _ in calls) == [1, 2]
    assert all(auth == "Bearer local-jev-test" for _, _, auth in calls)


@pytest.mark.real_ray
def test_sql_ray_actor_preserves_rows_and_application_credentials(ray_local, server, monkeypatch):
    url, calls = server
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    with vane.connect() as connection:
        sql = f"""SELECT id, ai_jev(state, {_SQL_QUESTIONS}, options := struct_pack(
                    base_url := '{url}', batch_size := 2, actor_number := 2, max_retries := 0)) AS judgment
                FROM (VALUES (1, 'first'), (2, NULL), (3, 'second')) t(id, state) ORDER BY id"""
        assert "ray_actor" in connection.sql(sql).explain()
        # execute binds once before returning a result; a lazy Relation may be
        # rebound when composing it or asking for another physical plan.
        connection.execute(sql)
        monkeypatch.setenv("TYPESAFE_API_KEY", "changed-after-binding")
        rows = connection.fetchall()
        assert [row[0] for row in rows] == [1, 2, 3]
        assert [json.loads(row[1]) if row[1] else None for row in rows] == [_response(), None, _response()]
    assert len(calls) == 2
    assert all(auth == "Bearer local-jev-test" for _, _, auth in calls)
