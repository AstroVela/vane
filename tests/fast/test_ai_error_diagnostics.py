# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import errno
import json
import pickle

import pytest

from vane.ai import summarize_error
from vane.ai.provider import _safe_provider_execution_error


@pytest.mark.parametrize(
    "error,expected",
    [
        (TypeError("ServerArgs.__init__() got an unexpected keyword argument 'max_model_len'"), "max_model_len"),
        (
            ValueError(
                "Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.3. "
                "Raise --mem-fraction-static above 0.346"
            ),
            "mem_fraction_static=0.3, required above 0.346",
        ),
        (RuntimeError("CUDA out of memory. private request content"), "CUDA memory allocation failed"),
        (MemoryError("private allocation description"), "memory allocation failed"),
    ],
)
@pytest.mark.parametrize("envelope", ["plain", "native_json", "traceback"])
def test_diagnostic_survives_provider_pickle_and_native_query_transport(error, expected, envelope):
    public = _safe_provider_execution_error("sglang", "private-model", "embedding initialization", error)
    restored = pickle.loads(pickle.dumps(public))
    if envelope == "native_json":
        message = "FTE query failed: " + json.dumps(
            {"exception_type": "Invalid Input", "exception_message": "private SQL\n" + str(restored)}
        )
    elif envelope == "traceback":
        message = str(restored) + "\nTraceback: private source content"
    else:
        message = "SQL query contains private source content\n" + str(restored)
    flattened = RuntimeError(message)
    for result in (summarize_error(error), summarize_error(public), summarize_error(flattened)):
        assert expected in result
        assert "private" not in result


@pytest.mark.parametrize(
    "message",
    [
        "opaque-secret-without-a-label",
        "ServerArgs.__init__() got an unexpected keyword argument 'opaque_secret'",
        "ServerArgs.__init__() got an unexpected keyword argument 'max_model_len' extra-private-text",
        "Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=opaque-secret. "
        "Raise --mem-fraction-static above 0.346",
        "x" * 100_000,
    ],
)
def test_unknown_provider_messages_are_not_copied(message):
    error = TypeError(message)
    result = summarize_error(_safe_provider_execution_error("test", "model", "embed", error))
    assert "opaque" not in result and "private" not in result
    assert len(result) <= 512


def test_status_errno_chain_and_cleanup_survive_without_paths_or_payloads():
    primary = TypeError("ServerArgs.__init__() got an unexpected keyword argument 'max_model_len'")
    outer = RuntimeError("private SQL and credentials")
    outer.primary_error = primary
    outer.cleanup_errors = (OSError(errno.ENOSPC, "private path and record"),)
    result = summarize_error(outer)
    assert "max_model_len" in result
    assert "cleanup: OSError (errno=28)" in result
    assert result.index("max_model_len") < result.index("cleanup:")
    assert "private" not in result


def test_hostile_stringification_cycles_and_oversized_cleanup_are_bounded():
    class HostileError(Exception):
        def __str__(self):
            raise AssertionError("must not stringify arbitrary exceptions")

    error = HostileError("opaque-private-key")
    error.status_code = 503
    error.__cause__ = error
    error.cleanup_errors = [error] * 1000
    result = summarize_error(error, max_chars=64)
    assert len(result) <= 64 and "503" in result and "private" not in result


def test_cleanup_exception_does_not_replace_initialization_cause():
    try:
        try:
            raise ValueError(
                "Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.3. "
                "Raise --mem-fraction-static above 0.346"
            )
        finally:
            raise OSError(errno.EIO, "private cleanup path")
    except OSError as error:
        result = summarize_error(error)
    assert "required above 0.346" in result and "errno=5" in result
    assert "private" not in result


def test_actor_creation_error_preserves_pickled_initializer_cause():
    from ray.exceptions import ActorDiedError, RayTaskError

    original = _safe_provider_execution_error(
        "sglang",
        "fixture",
        "initialization",
        TypeError("ServerArgs.__init__() got an unexpected keyword argument 'max_model_len'"),
    )
    actor_error = ActorDiedError(RayTaskError("initialize", "private traceback", original))
    restored = pickle.loads(pickle.dumps(actor_error))
    for error in (actor_error, restored):
        summary = summarize_error(error)
        assert "max_model_len" in summary and "ActorDiedError" in summary
        assert "private" not in summary


def test_embedding_batch_preserves_initialization_cause_before_detaching_errors():
    import asyncio

    import pyarrow as pa

    from vane.ai.functions import _EmbedTextBatch

    class Descriptor:
        def get_provider(self):
            return "sglang"

        def get_model(self):
            return "private-model"

        def instantiate(self):
            try:
                raise TypeError("ServerArgs.__init__() got an unexpected keyword argument 'max_model_len'")
            finally:
                raise OSError(errno.EIO, "private cleanup path")

    batch = _EmbedTextBatch(Descriptor(), "text", "embedding", 3, max_retries=0)
    loop = asyncio.new_event_loop()
    batch.bind_async_runtime(loop.run_until_complete)
    try:
        with pytest.raises(RuntimeError) as caught:
            batch(pa.table({"text": ["private source"]}))
    finally:
        loop.close()
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    restored = pickle.loads(pickle.dumps(caught.value))
    flattened = RuntimeError(json.dumps({"exception_message": str(restored)}))
    for error in (caught.value, restored, flattened):
        summary = summarize_error(error)
        assert "max_model_len" in summary and " <- OSError (errno=5)" in summary
        assert "private" not in summary
    assert b"private cleanup path" not in pickle.dumps(caught.value)


@pytest.mark.parametrize("explicit_cleanup", [False, True])
def test_flattened_errors_retain_each_causes_own_status(explicit_cleanup):
    primary = ConnectionError("private request")
    cleanup = OSError(errno.EIO, "private path")
    if explicit_cleanup:
        error = RuntimeError("private wrapper")
        error.primary_error = primary
        error.cleanup_errors = (cleanup,)
    else:
        error = cleanup
        error.__context__ = primary
    expected = summarize_error(error)
    for message in (
        "transcription failed; upstream error: " + expected,
        json.dumps({"exception_message": "upstream error: " + expected}),
    ):
        assert summarize_error(RuntimeError(message)) == expected
    assert "OSError (errno=5)" in expected and "ConnectionError (errno=" not in expected


def test_transport_does_not_attribute_status_from_arbitrary_suffix():
    message = "upstream error: ConnectionError private-url?status_code=401 errno=5"
    assert summarize_error(RuntimeError(message)) == "ConnectionError"


@pytest.mark.parametrize("status", [401, 429, 503])
def test_httpx_response_status_survives_provider_and_native_transport(status):
    httpx = pytest.importorskip("httpx")
    response = httpx.Response(
        status,
        request=httpx.Request("POST", "https://example.test/private-path?token=private-key"),
        json={"private": "body"},
    )
    with pytest.raises(httpx.HTTPStatusError) as caught:
        response.raise_for_status()
    public = _safe_provider_execution_error("httpx", "private-model", "request", caught.value)
    flattened = RuntimeError(json.dumps({"exception_message": str(public)}))
    expected = f"HTTPStatusError (status_code={status})"
    for error in (caught.value, public, pickle.loads(pickle.dumps(public)), flattened):
        assert summarize_error(error) == expected


def test_numeric_class_status_does_not_invoke_sdk_properties():
    class SDKError(Exception):
        status_code = 429

        @property
        def response(self):
            raise AssertionError("must not evaluate SDK properties")

        @property
        def code(self):
            raise AssertionError("must not evaluate SDK properties")

    assert summarize_error(SDKError("private payload")) == "SDKError (status_code=429)"
