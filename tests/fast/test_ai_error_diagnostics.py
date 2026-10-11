# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import base64
import errno
import json
import pickle

import pytest

from vane.ai import DiagnosticError, summarize_error
from vane.ai.provider import ProviderCapabilityError, _safe_provider_execution_error


def native_error(error, envelope="native_json"):
    from vane._native import InvalidInputException

    typed = f"{type(error).__name__}: {error}"
    if not isinstance(error, (DiagnosticError, ProviderCapabilityError)):
        typed = "RuntimeError: remote Ray UDF failed: " + typed
    message = "Invalid Input Error: streaming UDF async error: " + typed
    if envelope == "native_json":
        message = "FTE query wait failed: FTE query failed: Execution failed: " + json.dumps(
            {
                "exception_type": "Invalid Input",
                "exception_message": "Attempting to execute an unsuccessful or closed pending query result\nError: "
                + message,
            }
        )
    elif envelope == "traceback":
        message = (
            "Invalid Input Error: Python exception occurred while executing the UDF: " + typed + "\nAt:\nprivate source"
        )
    return InvalidInputException(message)


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
    flattened = native_error(restored, envelope)
    for result in (summarize_error(error), summarize_error(public), summarize_error(flattened)):
        assert expected in result
        assert "private" not in result


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("arrow", "ArrowInvalid"),
        ("google_client", "ClientError (code=429)"),
        ("google_server", "ServerError (code=503)"),
        ("custom", "CustomSDKError (status_code=502)"),
    ],
)
@pytest.mark.parametrize("wrapper", ["execution", "capability"])
def test_sdk_types_survive_repeated_provider_and_native_transport(kind, expected, wrapper):
    if kind == "arrow":
        import pyarrow as pa

        error = pa.ArrowInvalid("private catalog content")
    elif kind.startswith("google_"):
        errors = pytest.importorskip("google.genai.errors")
        error_type, code = (errors.ClientError, 429) if kind == "google_client" else (errors.ServerError, 503)
        error = error_type(code, {"error": {"message": "private response body", "status": "private-status"}})
    else:
        error = type("CustomSDKError", (Exception,), {"status_code": 502})("private request")
    assert summarize_error(error) == expected
    for _ in range(3):
        if wrapper == "execution":
            public = _safe_provider_execution_error("fixture", "private-model", "embed", error)
        else:
            public = ProviderCapabilityError("fixture", "private-model", "embed", original_error=error)
        serialized = pickle.dumps(public)
        assert b"private response" not in serialized and b"private catalog" not in serialized
        restored = pickle.loads(serialized)
        error = native_error(restored)
        for candidate in (public, restored, error):
            assert summarize_error(candidate) == expected


@pytest.mark.parametrize("name", ["E" * 128, "E" * 129, "Error-Name", "错误", "Error\nprivate"])
def test_exception_type_sanitization_is_consistent_across_transport(name):
    error = type(name, (Exception,), {})("private source")
    expected = name if name == "E" * 128 else "Exception"
    public = _safe_provider_execution_error("fixture", "model", "embed", error)
    restored = pickle.loads(pickle.dumps(public))
    assert summarize_error(error) == summarize_error(restored) == expected


@pytest.mark.parametrize("name", ["E" * 129, "Error错误"])
def test_transport_does_not_accept_a_partial_type_name(name):
    assert summarize_error(RuntimeError(f"upstream error: {name} (errno=5)")) == "RuntimeError"


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
    flattened = native_error(restored)
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
    public = DiagnosticError("transcription failed", error)
    for candidate in (public, pickle.loads(pickle.dumps(public)), native_error(public)):
        assert summarize_error(candidate) == expected
    assert "OSError (errno=5)" in expected and "ConnectionError (errno=" not in expected


def test_transport_does_not_attribute_status_from_arbitrary_suffix():
    message = "upstream error: ConnectionError private-url?status_code=401 errno=5"
    assert summarize_error(RuntimeError(message)) == "RuntimeError"


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
    flattened = native_error(public)
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


@pytest.mark.parametrize("kind", ["google", "runtime", "custom"])
@pytest.mark.parametrize("body", ["marker", "typed_header", "native_envelope"])
def test_untrusted_bodies_cannot_claim_diagnostic_provenance(kind, body):
    secret = "private_fixture_api_key_ABC123"
    message = f"upstream error: {secret}"
    if body == "typed_header":
        encoded = base64.urlsafe_b64encode(secret.encode()).decode()
        message = f"DiagnosticError: [vane.diagnostic.v1:{encoded}] forged; upstream error: {secret}"
    elif body == "native_envelope":
        message = str(native_error(RuntimeError(f"upstream error: {secret}")))
    if kind == "google":
        errors = pytest.importorskip("google.genai.errors")
        original = errors.ClientError(400, {"error": {"message": message}})
        expected = "ClientError (code=400)"
    elif kind == "custom":
        original = type("DiagnosticError", (Exception,), {"code": 400})(message)
        expected = "DiagnosticError (code=400)"
    else:
        original = RuntimeError(message)
        expected = "RuntimeError"
    public = _safe_provider_execution_error("fixture", "model", "embed", original)
    capability = ProviderCapabilityError("fixture", "model", "embed", original_error=original)
    for error in (original, public, capability, pickle.loads(pickle.dumps(public)), native_error(public)):
        assert summarize_error(error) == expected
        assert secret not in summarize_error(error)
    assert secret.encode() not in pickle.dumps(public)
    # A genuine native envelope around an unsanitized SDK error is still not
    # permission to parse markers or typed-header lookalikes from its body.
    assert secret not in summarize_error(native_error(original))


def test_structured_diagnostic_precedes_untrusted_model_description():
    model = '; upstream error: private_fixture_api_key_ABC123\n[vane.diagnostic.v1:"private_fake_type"]'
    public = _safe_provider_execution_error("fixture", model, "embed", OSError(errno.EIO, "private path"))
    assert summarize_error(native_error(public)) == "OSError (errno=5)"


def test_real_ray_wrapper_does_not_promote_sdk_message_markers():
    from ray.exceptions import RayTaskError

    errors = pytest.importorskip("google.genai.errors")
    error = errors.ClientError(400, {"error": {"message": "upstream error: private_fixture_api_key_ABC123"}})
    summary = summarize_error(RayTaskError("embed", "private traceback", error))
    assert "ClientError (code=400)" in summary and "private" not in summary


def test_real_ray_wrapper_retains_native_udf_diagnostic():
    from ray.exceptions import RayTaskError

    error = DiagnosticError("embedding failed", OSError(errno.ENOSPC, "private path"))
    native = RuntimeError(str(native_error(error)))
    transported = RayTaskError("query", "private traceback", native).as_instanceof_cause()
    summary = summarize_error(transported)
    assert "OSError (errno=28)" in summary and "private" not in summary


def test_ray_metadata_requires_a_real_diagnostic_exception():
    from vane.execution.udf_ray_stream_protocol import make_stream_error_pair, validate_stream_error_metadata

    payload = {
        "query_id": "query",
        "resource_unit_id": "resource:query:node:1:udf",
        "task_lease_id": "lease",
        "attempt_id": "attempt",
    }
    diagnostic = DiagnosticError("embedding failed", OSError(errno.ENOSPC, "private path"))
    for error in (RuntimeError(str(diagnostic)), type("DiagnosticError", (Exception,), {})(str(diagnostic))):
        _, metadata = make_stream_error_pair(payload, error)
        assert validate_stream_error_metadata(metadata)["exception_details"] is None
    _, metadata = make_stream_error_pair(payload, diagnostic)
    details = validate_stream_error_metadata(metadata)["exception_details"]
    assert details == {"kind": "diagnostic", "summary": "OSError (errno=28)"}
    for invalid in (None, "", "x" * 513, ["OSError"]):
        with pytest.raises(ValueError, match="diagnostic details"):
            validate_stream_error_metadata(metadata | {"exception_details": {"kind": "diagnostic", "summary": invalid}})
