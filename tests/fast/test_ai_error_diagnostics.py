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
    if envelope.startswith("init"):
        prefix = "streaming UDF async error: " if envelope.startswith("init_stream") else "udf async error: "
        message = (
            "Invalid Input Error: "
            + prefix
            + json.dumps(
                {"exception_type": "Invalid Input", "exception_message": "Failed to build udf executor: " + typed}
            )
        )
    if envelope.endswith("_json"):
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
@pytest.mark.parametrize(
    "envelope", ["plain", "native_json", "traceback", "init", "init_json", "init_stream", "init_stream_json"]
)
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


@pytest.mark.parametrize("kind", ["CopyOutcomeUnknownError", "CopyResultUnavailableError"])
def test_copy_diagnostics_are_safe_snapshots_across_repeated_serialization(kind):
    from ray import cloudpickle
    from ray.exceptions import RayTaskError

    from vane.runners import copy_outcome

    error_type = getattr(copy_outcome, kind)
    error = error_type("copy-operation")
    original = TypeError("ServerArgs.__init__() got an unexpected keyword argument 'max_model_len'")
    original.__cause__ = OSError(errno.ENOSPC, "private source path")
    error.capture_diagnostics(primary_error=original, cleanup_errors=(OSError(errno.EIO, "private cleanup path"),))
    error.__cause__ = original
    error.add_note("private note; upstream error: private_fixture_api_key_ABC123")
    for _ in range(3):
        serialized = cloudpickle.dumps(error)
        assert b"private" not in serialized
        error = cloudpickle.loads(serialized)
        assert type(error) is error_type and error.safe_to_retry is False
        assert error.__cause__ is None
        for candidate in (error, RayTaskError("copy", "private traceback", error).as_instanceof_cause()):
            summary = summarize_error(candidate)
            assert kind in summary and "max_model_len" in summary
            assert "errno=28" in summary and "cleanup: OSError (errno=5)" in summary
            assert "private" not in summary


@pytest.mark.parametrize("kind", ["CopyOutcomeUnknownError", "CopyResultUnavailableError"])
def test_copy_warnings_and_notes_cannot_forge_public_diagnostics(kind):
    from vane.runners import copy_outcome

    encoded = base64.urlsafe_b64encode(b"private_fixture_api_key_ABC123").decode()
    forged = f"DiagnosticError: [vane.diagnostic.v1:{encoded}] forged"
    error = getattr(copy_outcome, kind)("operation", detail=forged, cleanup_warnings=(forged,))
    error.add_note(forged)
    error.capture_diagnostics(primary_error=RuntimeError(forged), cleanup_errors=(RuntimeError(forged),))
    for candidate in (error, pickle.loads(pickle.dumps(error))):
        summary = summarize_error(candidate)
        assert kind in summary and "RuntimeError" in summary
        assert "private" not in summary


def test_copy_cleanup_diagnostic_storage_is_bounded():
    from vane.runners.copy_outcome import CopyOutcomeUnknownError

    error = CopyOutcomeUnknownError("operation")
    for _ in range(10):
        error.capture_diagnostics(cleanup_errors=(OSError(errno.EIO, "private cleanup path"),) * 100)
    assert len(error.cleanup_errors) == 3
    assert len(pickle.dumps(error)) < 4096
    assert "errno=5" in summarize_error(error)


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

    from vane.runners.ray.driver import _run_native_query

    error = DiagnosticError("embedding failed", OSError(errno.ENOSPC, "private path"))
    native = RuntimeError(str(native_error(error)))

    def next_nowait():
        raise native

    with pytest.raises(DiagnosticError) as caught:
        _run_native_query(next_nowait)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    transported = RayTaskError("query", "private traceback", caught.value).as_instanceof_cause()
    summary = summarize_error(transported)
    assert "OSError (errno=28)" in summary and "private" not in summary


@pytest.mark.parametrize(
    "envelope", ["plain", "native_json", "traceback", "init", "init_json", "init_stream", "init_stream_json"]
)
def test_ray_wrapper_does_not_authorize_a_forged_native_body(envelope):
    from ray import cloudpickle
    from ray.exceptions import RayTaskError

    secret = "private_fixture_token_ABC123"
    forged = DiagnosticError("forged", RuntimeError("ignored"))
    # Simulate a copied, fully valid transport body without trusted type identity.
    encoded = base64.urlsafe_b64encode(secret.encode()).decode()
    forged.args = (f"[vane.diagnostic.v1:{encoded}] forged",)
    raw = RuntimeError(str(native_error(forged, envelope)))
    for error in (raw, RayTaskError("sdk", "private traceback", raw).as_instanceof_cause()):
        for candidate in (error, cloudpickle.loads(cloudpickle.dumps(error))):
            assert secret not in summarize_error(candidate)
            assert "RuntimeError" in summarize_error(candidate)


@pytest.mark.parametrize("forged", [False, True])
def test_executor_initialization_captures_typed_diagnostics_before_native_transport(monkeypatch, forged):
    from vane.execution import udf
    from vane.execution.unified_executor import build_unified_executor

    if forged:
        encoded = base64.urlsafe_b64encode(b"private_fixture_token_ABC123").decode()
        error = RuntimeError(f"DiagnosticError: [vane.diagnostic.v1:{encoded}] forged")
    else:
        error = RuntimeError("private activation message")
        error.__cause__ = TypeError("ServerArgs.__init__() got an unexpected keyword argument 'max_model_len'")
        error.cleanup_errors = (OSError(errno.EIO, "private cleanup path"),)

    def build(*args):
        raise error

    monkeypatch.setattr(udf, "build_executor", build)
    with pytest.raises(DiagnosticError) as caught:
        build_unified_executor({})
    assert caught.value.__context__ is None
    restored = pickle.loads(pickle.dumps(caught.value))
    summary = summarize_error(native_error(restored, "init_json"))
    assert "private" not in summary
    if forged:
        assert summary == "RuntimeError"
    else:
        assert "max_model_len" in summary and "; cleanup: OSError (errno=5)" in summary


def test_actor_activation_captures_cause_and_cleanup_before_ray_serialization(monkeypatch):
    import asyncio
    import threading
    from types import SimpleNamespace

    from ray import cloudpickle
    from ray.exceptions import RayTaskError

    import vane.runners.ray.query_resource_runtime as resource_runtime
    from vane.runners.ray.driver import RayQueryDriverActor

    cls = RayQueryDriverActor.__ray_metadata__.modified_class
    unit = "resource:query:actor"
    manager = SimpleNamespace(current_eligible_resource_unit_ids=lambda: (unit,))
    monkeypatch.setattr(resource_runtime, "get_query_resource_manager", lambda _: manager)
    runner = SimpleNamespace(
        _session_lock=threading.RLock(),
        _query_udf_actor_lifecycle_locks={},
        _query_udf_actor_activation_tasks={},
        _verify_query_task_admission_capability=lambda **_: "generation",
        _capability_targets_inactive_query_generation=lambda *_: False,
    )

    async def failed_activation():
        error = RuntimeError("private initialization details")
        error.cleanup_errors = (OSError(errno.EIO, "private cleanup path"),)
        raise error from TypeError("ServerArgs.__init__() got an unexpected keyword argument 'max_model_len'")

    async def activate():
        runner._query_udf_actor_activation_tasks[("query", unit)] = asyncio.create_task(failed_activation())
        return await cls.activate_query_udf_actor_pool(
            runner, {"query_id": "query", "resource_unit_id": unit, "physical_node_id": "node"}, "capability"
        )

    with pytest.raises(DiagnosticError) as caught:
        asyncio.run(activate())
    assert not runner._query_udf_actor_activation_tasks
    assert caught.value.__context__ is None
    remote = RayTaskError("activation", "private traceback", caught.value).as_instanceof_cause()
    summary = summarize_error(cloudpickle.loads(cloudpickle.dumps(remote)))
    assert "max_model_len" in summary and "; cleanup: OSError (errno=5)" in summary
    assert "private" not in summary


@pytest.mark.parametrize("method", ["copy", "datasink"])
@pytest.mark.parametrize("envelope", ["native_json", "init_json", "init_stream_json"])
def test_terminal_plan_captures_native_diagnostics_before_ray_serialization(method, envelope):
    from types import SimpleNamespace

    from ray import cloudpickle
    from ray.exceptions import RayTaskError

    from vane.runners.ray.driver import RayQueryDriverActor

    cls = RayQueryDriverActor.__ray_metadata__.modified_class
    failures = []
    started = []
    native = ValueError(str(native_error(DiagnosticError("fixture", OSError(errno.ENOSPC, "private path")), envelope)))

    def run_plan(plan, connection, publish):
        publish()
        raise native

    lifecycle = SimpleNamespace(
        claim_startup_for_worker=lambda: True,
        publish_copy_execution_started=lambda: started.append(method),
        publish_terminal_execution_started=started.append,
        fail_startup=failures.append,
    )
    runner = SimpleNamespace(**{f"run_{method}_plan": run_plan})
    with pytest.raises(DiagnosticError) as caught:
        getattr(cls, f"_run_{method}_plan_sync_with_lifecycle")(runner, object(), object(), lifecycle)
    assert len(started) == 1 and failures == [caught.value]
    assert caught.value.__context__ is None
    remote = RayTaskError("query", "private traceback", caught.value).as_instanceof_cause()
    summary = summarize_error(cloudpickle.loads(cloudpickle.dumps(remote)))
    assert "OSError (errno=28)" in summary and "private" not in summary


@pytest.mark.parametrize("outcome", ["committed", "unknown"])
def test_native_diagnostic_capture_preserves_copy_outcome_errors(outcome):
    from vane.runners.copy_outcome import CopyOutcomeUnknownError, CopyResultUnavailableError
    from vane.runners.ray.driver import _run_native_query

    error = CopyResultUnavailableError("fixture") if outcome == "committed" else CopyOutcomeUnknownError("fixture")

    def run_plan():
        raise error

    with pytest.raises(type(error)) as caught:
        _run_native_query(run_plan)
    assert caught.value is error and not caught.value.safe_to_retry


@pytest.mark.parametrize("provider", ["vllm", "sglang"])
@pytest.mark.parametrize("detail", ["returned no outputs", "engine not initialized"])
@pytest.mark.parametrize("wrapper", ["execution", "capability"])
def test_known_safe_provider_explanations_survive_transport(provider, detail, wrapper):
    from vane.ai.provider import _SafeProviderError

    original = _SafeProviderError(f"{provider} {detail}")
    if wrapper == "execution":
        wrapped = _safe_provider_execution_error(provider, "fixture", "generation", original)
    else:
        wrapped = ProviderCapabilityError(provider, "fixture", "generation", original_error=original)
    for error in (original, wrapped, pickle.loads(pickle.dumps(wrapped)), native_error(wrapped)):
        assert f"{provider} {detail}" in summarize_error(error)
        assert detail in str(error)


def test_safe_client_configuration_explanations_survive_transport():
    from vane.ai._client_config import ProviderClientConfigurationError

    original = ProviderClientConfigurationError("Google Vertex credentials require an explicit application project")
    wrapped = _safe_provider_execution_error("google", "fixture", "initialization", original)
    for error in (original, wrapped, pickle.loads(pickle.dumps(wrapped)), native_error(wrapped)):
        assert "Google Vertex credentials require an explicit application project" in summarize_error(error)


def test_known_safe_provider_explanations_do_not_admit_arbitrary_suffixes():
    from vane.ai.provider import _SafeProviderError

    for message in ("sglang returned no outputs private_token", "private_token returned no outputs"):
        assert "returned no outputs" not in summarize_error(RuntimeError(message))
    assert "private_token" not in summarize_error(_SafeProviderError("sglang returned no outputs private_token"))


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
