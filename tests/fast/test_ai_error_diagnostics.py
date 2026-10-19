# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import base64
import errno
import json
import pickle
import sys

import pytest

from vane.ai import DiagnosticError, summarize_error
from vane.ai.provider import ProviderCapabilityError, _safe_provider_execution_error


def onnx_allocation_error(kind="Fail", *, initialize=False, wrapped=False):
    status = "1 : FAIL" if kind == "Fail" else "6 : RUNTIME_EXCEPTION"
    operation = (
        "Exception during initialization: "
        if initialize
        else "Non-zero status code returned while running Conv node. Name:'private-node' Status Message: "
    )
    error = type(kind, (Exception,), {})(
        f"[ONNXRuntimeError] : {status} : {operation}"
        "/private/build/bfc_arena.cc:360 void* onnxruntime::BFCArena::AllocateRawInternal"
        "(size_t, bool, onnxruntime::Stream*) Failed to allocate memory for requested buffer of size 1073741824"
    )
    if wrapped:
        outer = type("ONNXRuntimeError", (Exception,), {})("private traceback with credentials")
        outer.__context__ = error
        return outer
    return error


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
    if envelope == "task_json":
        message = json.dumps(
            {
                "exception_type": "INTERNAL",
                "exception_message": "[PlanExecutionContext::spawn] task error: DuckDBError::InternalError: " + message,
                "stack_trace_pointers": "private native stack",
            }
        )
    return InvalidInputException(message)


@pytest.mark.parametrize("invalid", ["type", "prefix", "task_type", "message_type", "json", "forged"])
def test_spawn_diagnostic_requires_complete_typed_native_envelopes(invalid):
    from vane.ai._errors import _native_query_diagnostic

    diagnostic = DiagnosticError("fixture", OSError(errno.ENOSPC, "private path"))
    error = RuntimeError(str(diagnostic)) if invalid == "forged" else diagnostic
    payload = json.loads(str(native_error(error, "task_json")))
    if invalid == "type":
        payload["exception_type"] = "Invalid Input"
    elif invalid == "prefix":
        payload["exception_message"] = "private text " + payload["exception_message"]
    elif invalid == "task_type":
        payload["exception_message"] = payload["exception_message"].replace(
            "DuckDBError::InternalError:", "RuntimeError:", 1
        )
    elif invalid == "message_type":
        payload["exception_message"] = {"private": "body"}
    message = "DuckDBError::ExternalError: " + json.dumps(payload) + ("trailing text" if invalid == "json" else "")
    assert _native_query_diagnostic(ValueError(message)) is None
    assert "private" not in summarize_error(ValueError(message))


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
        (RuntimeError("CUDA failed with error out of memory"), "CUDA memory allocation failed"),
        (onnx_allocation_error(), "ONNX Runtime memory allocation failed"),
        (onnx_allocation_error("RuntimeException", initialize=True), "ONNX Runtime memory allocation failed"),
        (onnx_allocation_error(wrapped=True), "ONNX Runtime memory allocation failed"),
        (
            RuntimeError(
                "[enforce fail at /private/build/alloc_cpu.cpp:127] err == 0. "
                "DefaultCPUAllocator: can't allocate memory: you tried to allocate 1073741824 bytes. "
                "Error code 12 (Cannot allocate memory)\nprivate C++ stack"
            ),
            "CPU memory allocation failed",
        ),
        (MemoryError("private allocation description"), "memory allocation failed"),
    ],
)
@pytest.mark.parametrize(
    "envelope",
    ["plain", "native_json", "task_json", "traceback", "init", "init_json", "init_stream", "init_stream_json"],
)
def test_diagnostic_survives_provider_pickle_and_native_query_transport(error, expected, envelope):
    public = _safe_provider_execution_error("sglang", "private-model", "embedding initialization", error)
    restored = pickle.loads(pickle.dumps(public))
    flattened = native_error(restored, envelope)
    for result in (summarize_error(error), summarize_error(public), summarize_error(flattened)):
        assert expected in result
        assert "private" not in result


@pytest.mark.parametrize(
    "message",
    [
        "private input mentions DefaultCPUAllocator: can't allocate memory",
        "DefaultCPUAllocator: invalid argument in private input",
        "DefaultCPUAllocator: can't allocate memory: private body",
    ],
)
def test_cpu_allocator_summary_rejects_unrecognized_messages(message):
    assert summarize_error(RuntimeError(message)) == "RuntimeError"


@pytest.mark.parametrize(
    "message",
    [
        "CUDA failed with error invalid argument",
        "private input mentions CUDA failed with error out of memory",
        "CUDA failed with error out of memory private input",
    ],
)
def test_ctranslate_summary_accepts_only_the_allocation_error(message):
    assert summarize_error(RuntimeError(message)) == "RuntimeError"


@pytest.mark.parametrize("kind", ["Fail", "RuntimeException"])
@pytest.mark.parametrize("change", ["prefix", "suffix", "node_text", "allocator", "size"])
def test_onnx_summary_requires_an_allocator_failure(kind, change):
    error = onnx_allocation_error(kind)
    message = str(error)
    message = {
        "prefix": "private text " + message,
        "suffix": message + " private text",
        "node_text": "[ONNXRuntimeError] : 1 : FAIL : private input mentions Failed to allocate memory",
        "allocator": message.replace("Failed to allocate memory", "Invalid argument"),
        "size": message.replace("1073741824", "private input"),
    }[change]
    assert summarize_error(type(error)(message)) == kind


@pytest.mark.parametrize("kind", ["RuntimeError", "ONNXRuntimeError"])
def test_onnx_traceback_text_without_a_typed_cause_is_not_decoded(kind):
    error = type(kind, (Exception,), {})(str(onnx_allocation_error()))
    assert summarize_error(error) == kind


@pytest.mark.skipif(sys.platform != "linux", reason="Linux address-space limit")
def test_real_torch_cpu_allocator_survives_diagnostic_wrapping():
    import subprocess
    import sys

    pytest.importorskip("torch")
    code = """
import json, pickle, resource
from pathlib import Path
import torch
from vane.ai import DiagnosticError, summarize_error
virtual = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmSize:'))) * 1024
old = resource.getrlimit(resource.RLIMIT_AS)
try:
    resource.setrlimit(resource.RLIMIT_AS, (virtual + 64 * 1024**2, old[1]))
    torch.empty(256 * 1024**2, dtype=torch.float32, device='cpu')
except RuntimeError as error:
    resource.setrlimit(resource.RLIMIT_AS, old)
    assert 'DefaultCPUAllocator' in str(error)
    print(json.dumps([summarize_error(error), summarize_error(pickle.loads(pickle.dumps(DiagnosticError('fixture', error))))]))
else:
    raise AssertionError('CPU allocator probe did not fail')
finally:
    resource.setrlimit(resource.RLIMIT_AS, old)
"""
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=45)
    summaries = json.loads(result.stdout)
    assert summaries == ["RuntimeError: CPU memory allocation failed; check model and worker CPU memory budgets"] * 2


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


@pytest.mark.parametrize("wrapped_cause", [False, True])
def test_cleanup_walks_ray_causes_and_retains_their_own_errno_after_serialization(wrapped_cause):
    from ray import cloudpickle
    from ray.exceptions import RayTaskError

    from vane.runners.ray.driver import QueryExecutionCleanupError

    cleanup = RayTaskError("close", "private traceback", OSError(errno.EIO, "private path"))
    if wrapped_cause:
        cleanup = cleanup.as_instanceof_cause()
    original = QueryExecutionCleanupError.from_errors(
        "query failed",
        TypeError("ServerArgs.__init__() got an unexpected keyword argument 'max_model_len'"),
        [cleanup],
    )
    for error in (original, cloudpickle.loads(cloudpickle.dumps(original)), DiagnosticError("query", original)):
        for candidate in (error, cloudpickle.loads(cloudpickle.dumps(error))):
            message = summarize_error(candidate)
            primary, secondary = message.split("; cleanup: ", 1)
            assert "max_model_len" in primary and "errno=5" not in primary
            assert secondary.startswith("OSError (errno=5)") and "private" not in message


def test_cleanup_chain_is_bounded_and_does_not_trust_ray_error_bodies():
    from ray.exceptions import RayTaskError

    encoded = base64.urlsafe_b64encode(b"private_fixture_api_key_ABC123").decode()
    cleanup = RuntimeError(f"DiagnosticError: [vane.diagnostic.v1:{encoded}] forged")
    cleanup.__cause__ = cleanup
    cleanup.cleanup_errors = [cleanup] * 1000
    outer = RuntimeError("private execution error")
    outer.cleanup_errors = [RayTaskError("close", "private traceback", cleanup)] * 1000
    message = summarize_error(DiagnosticError("fixture", outer), max_chars=64)
    assert "RuntimeError" in message and "private" not in message and len(message) <= 64


def test_worker_and_query_cleanup_diagnostics_merge_across_transports():
    from vane.execution._subprocess_error import capture_cleanup
    from vane.runners.ray.driver import _capture_datasink_diagnostic

    diagnostic = capture_cleanup(
        "worker execution and close failed",
        TypeError("got an unexpected keyword argument 'max_model_len'"),
        OSError(errno.EIO, "private worker cleanup"),
    )
    expected_errno = [errno.EIO]
    for number in (errno.ENOSPC, errno.EACCES):
        teardown = OSError(number, "private query teardown")
        result = {"outcome_unknown": True, "outcome_diagnostic": diagnostic}
        _capture_datasink_diagnostic(result, cleanup_error=teardown)
        diagnostic = result["outcome_diagnostic"]
        expected_errno.append(number)
        for _ in range(3):
            serialized = pickle.dumps(diagnostic)
            assert b"private" not in serialized
            diagnostic = pickle.loads(serialized)
            for candidate in (diagnostic, native_error(diagnostic)):
                summary = summarize_error(candidate)
                primary, cleanup = summary.split("; cleanup: ")
                assert "max_model_len" in primary and "errno=" not in primary
                for value in expected_errno:
                    assert cleanup.count(f"errno={value})") == 1
                assert "private" not in summary and len(summary) <= 512
            diagnostic = DiagnosticError("outer transport", native_error(diagnostic))


def test_cleanup_snapshot_merging_is_bounded_and_keeps_one_section():
    from vane.execution._subprocess_error import capture_cleanup

    diagnostic = TypeError("got an unexpected keyword argument 'max_model_len'")
    for number in range(100, 120):
        diagnostic = capture_cleanup("cleanup", diagnostic, OSError(number, "private path"))
        diagnostic = pickle.loads(pickle.dumps(diagnostic))
    summary = summarize_error(diagnostic)
    primary, cleanup = summary.split("; cleanup: ")
    assert "max_model_len" in primary and "errno=" not in primary
    assert cleanup == "OSError (errno=100); OSError (errno=101); OSError (errno=102)"
    assert len(summary) <= 512 and "private" not in summary


def test_cleanup_snapshot_does_not_trust_nested_raw_error_bodies():
    from vane.execution._subprocess_error import capture_cleanup

    raw = RuntimeError("RuntimeError; cleanup: private_fixture_token (errno=28)")
    diagnostic = capture_cleanup("worker cleanup", ConnectionError("private source"), raw)
    diagnostic = capture_cleanup("query cleanup", diagnostic, OSError(errno.EIO, "private query path"))
    for candidate in (diagnostic, native_error(pickle.loads(pickle.dumps(diagnostic)))):
        summary = summarize_error(candidate)
        assert "ConnectionError" in summary and "errno=5" in summary
        assert "private" not in summary and "errno=28" not in summary


def _long_diagnostic_cause(kind):
    from vane.ai._client_config import ProviderClientConfigurationError

    errors = [TypeError("got an unexpected keyword argument 'max_model_len'")]
    if kind == "technical":
        errors.extend(
            [
                ValueError(
                    "Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.123456. "
                    "Raise --mem-fraction-static above 0.999999."
                ),
                ProviderClientConfigurationError(
                    "Anthropic api_key or auth_token was not configured on the application before creating the provider/descriptor"
                ),
                RuntimeError(
                    "vllm actor finished without returning all submitted results: "
                    "actor_idx=1234567890123456 submitted=1234567890123456 received=1234567890123455"
                ),
            ]
        )
    else:
        for index in range(4):
            error = type("SDKError" + str(index) + "E" * 119, (Exception,), {})("private payload")
            for field in ("status_code", "status", "code", "errno"):
                setattr(error, field, 999999)
            errors.append(error)
    for previous, current in zip(errors, errors[1:]):
        current.__cause__ = previous
    return errors[-1]


@pytest.mark.parametrize("kind", ["technical", "numeric"])
@pytest.mark.parametrize("long_section", ["primary", "cleanup"])
def test_long_diagnostics_keep_cleanup_across_repeated_transports(kind, long_section):
    from vane.execution._subprocess_error import capture_cleanup

    primary = _long_diagnostic_cause(kind)
    cleanup = OSError(errno.EIO, "private cleanup path")
    if long_section == "cleanup":
        last = primary
        while last.__cause__ is not None:
            last = last.__cause__
        last.__cause__ = cleanup
        cleanup = primary
        primary = TypeError("got an unexpected keyword argument 'max_model_len'")
    diagnostic = capture_cleanup("worker failure", primary, cleanup)
    expected = diagnostic.summary
    assert len(expected) <= 512
    cause, secondary = expected.split("; cleanup: ")
    assert "max_model_len" in cause and "errno=5" not in cause
    assert "OSError (errno=5)" in secondary and "private" not in expected
    for _ in range(3):
        serialized = pickle.dumps(diagnostic)
        assert b"private" not in serialized
        restored = pickle.loads(serialized)
        for candidate in (diagnostic, restored, native_error(restored)):
            assert summarize_error(candidate) == expected
        diagnostic = DiagnosticError("query wrapper", native_error(restored))


@pytest.mark.parametrize("max_chars", [64, 80, 128, 256, 512, 4096])
@pytest.mark.parametrize("kind", ["technical", "numeric", "long_type"])
@pytest.mark.parametrize("with_cleanup", [False, True])
def test_diagnostic_budgets_produce_complete_parseable_items(max_chars, kind, with_cleanup):
    from vane.ai.provider import _SafeProviderError

    error = (
        type("E" * 128, (Exception,), {})("private payload") if kind == "long_type" else _long_diagnostic_cause(kind)
    )
    if with_cleanup:
        error.cleanup_errors = (
            OSError(errno.EIO, "private path"),
            _long_diagnostic_cause("technical"),
            OSError(errno.ENOSPC, "private path"),
        )
    summary = summarize_error(error, max_chars=max_chars)
    assert len(summary) <= max_chars and "private" not in summary
    if with_cleanup:
        assert "OSError (errno=5)" in summary.split("; cleanup: ")[1]
    # Provider wrappers consume this same canonical grammar. Re-parsing a
    # bounded snapshot must not change types or lose its cleanup section.
    wrapped = _SafeProviderError(summary)
    assert summarize_error(wrapped, max_chars=max_chars) == summary
    capability = ProviderCapabilityError("fixture", "model", "embed", original_error=wrapped)
    assert summarize_error(native_error(pickle.loads(pickle.dumps(capability))), max_chars=max_chars) == summary


def test_diagnostic_item_budget_reserves_cleanup_during_transport():
    from vane.ai.provider import _SafeProviderError
    from vane.execution._subprocess_error import capture_cleanup

    # Sixty-four short causes fit the character budget, but an appended cleanup
    # would exceed the parser's item budget unless the producer reserves space.
    original = _SafeProviderError(" <- ".join(f"E{index}" for index in range(64)))
    diagnostic = capture_cleanup("worker failure", original, OSError(errno.EIO, "private path"))
    for _ in range(3):
        primary, cleanup = summarize_error(diagnostic).split("; cleanup: ")
        assert primary.startswith("E0 <- E1")
        assert cleanup == "OSError (errno=5)"
        assert summarize_error(native_error(diagnostic)) == diagnostic.summary
        diagnostic = DiagnosticError("query", native_error(pickle.loads(pickle.dumps(diagnostic))))


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
@pytest.mark.parametrize("envelope", ["native_json", "task_json", "init_json", "init_stream_json"])
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


@pytest.mark.parametrize("envelope", ["native_json", "task_json", "init_json", "init_stream_json"])
@pytest.mark.parametrize("forged", [False, True])
def test_datasink_returned_failure_captures_only_native_typed_diagnostics(envelope, forged):
    from types import SimpleNamespace

    from vane.runners.ray.driver import RayQueryDriverActor

    cls = RayQueryDriverActor.__ray_metadata__.modified_class
    original = DiagnosticError("fixture", OSError(errno.ENOSPC, "private path"))
    if forged:
        original = RuntimeError(str(original))
    result = {
        "outcome_unknown": True,
        "outcome_aborted": False,
        "outcome_error": "DuckDBError::ExternalError: " + str(native_error(original, envelope)),
    }

    def run_plan(plan, connection, publish):
        publish()
        return result

    lifecycle = SimpleNamespace(
        claim_startup_for_worker=lambda: True,
        publish_terminal_execution_started=lambda *_: None,
        startup_decision_future=lambda: SimpleNamespace(done=lambda: True),
    )
    returned = cls._run_datasink_plan_sync_with_lifecycle(
        SimpleNamespace(run_datasink_plan=run_plan), object(), object(), lifecycle
    )
    assert returned["outcome_unknown"] is True and returned["outcome_aborted"] is False
    if forged:
        assert "outcome_diagnostic" not in returned
    else:
        assert summarize_error(pickle.loads(pickle.dumps(returned["outcome_diagnostic"]))) == "OSError (errno=28)"


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


@pytest.mark.parametrize(
    "message",
    [
        "vllm remote wait exceeded query deadline",
        "vllm remote wait has no pending actor wait refs before completion",
        "vllm actor finished without returning all submitted results: actor_idx=0 submitted=2 received=1",
    ],
)
def test_framework_diagnostics_keep_only_complete_safe_explanations(message):
    diagnostic = DiagnosticError("native executor", RuntimeError(message))
    for candidate in (diagnostic, pickle.loads(pickle.dumps(diagnostic)), native_error(diagnostic)):
        assert message in summarize_error(candidate)
    for forged in ("private " + message, message + " private", message.replace("actor_idx=0", "actor_idx=private")):
        if forged != message:
            assert summarize_error(RuntimeError(forged)) == "RuntimeError"


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


@pytest.mark.parametrize("method", ["submit", "take_ready_result", "wait_for_result"])
@pytest.mark.parametrize("fte", [False, True])
def test_native_prompt_envelopes_require_a_typed_exception(method, fte):
    from vane._native import InvalidInputException
    from vane.ai._errors import _native_query_diagnostic

    root = TypeError("ServerArgs.__init__() got an unexpected keyword argument 'max_model_len'")
    diagnostic = DiagnosticError("fixture", root)
    for typed in (True, False):
        message = f"vllm {method} failed: "
        if not typed:
            message += "RuntimeError: private provider body: "
        message += f"DiagnosticError: {diagnostic}\n\nAt:\nprivate traceback"
        if fte:
            message = "FTE query wait failed: FTE query failed: Execution failed: " + json.dumps(
                {"exception_type": "Invalid Input", "exception_message": message}
            )
        error = InvalidInputException(message)
        if typed:
            assert "max_model_len" in summarize_error(error)
            assert "max_model_len" in summarize_error(_native_query_diagnostic(error))
        else:
            assert summarize_error(error) == "InvalidInputException"
            assert _native_query_diagnostic(error) is None
        assert summarize_error(RuntimeError(message)) == "RuntimeError"
        assert "private" not in summarize_error(error)


def test_subprocess_diagnostic_captures_objects_instead_of_formatted_text():
    from vane.execution._subprocess_error import decode_error, error_payload, native_result_error
    from vane.execution.udf_subprocess_worker import _format_exception

    error = OSError(errno.ENOSPC, "private path")
    for original in (
        error,
        DiagnosticError("fixture", error),
        RuntimeError("DiagnosticError: " + str(DiagnosticError("fixture", error))),
    ):
        message, transported = decode_error(error_payload(original, _format_exception(original)))
        assert message == _format_exception(original)
        assert summarize_error(transported) == summarize_error(original)
        assert summarize_error(pickle.loads(pickle.dumps(transported))) == summarize_error(original)
        from vane._native import InvalidInputException

        # Native async results use str(exception) without a type prefix.
        queued = native_result_error(transported)
        native = InvalidInputException("streaming UDF async error: " + str(queued))
        assert summarize_error(native) == summarize_error(original)


@pytest.mark.parametrize("detail", ["query_id", "pre-registered resource_unit_id"])
def test_safe_runtime_contract_errors_remain_actionable(detail):
    message = "distributed Ray UDF payload requires " + detail
    error = DiagnosticError("UDF initialization", RuntimeError(message))
    assert message in summarize_error(native_error(error, "init"))
    assert message not in summarize_error(RuntimeError(message + " private suffix"))


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": "x", "summary": ""},
        {"message": "中" * 4096, "summary": "RuntimeError"},
        {"message": "x", "summary": "x" * 513},
    ],
)
def test_subprocess_error_payload_rejects_invalid_fields(payload):
    from vane.execution._subprocess_error import decode_error

    with pytest.raises(RuntimeError, match="Invalid UDF subprocess error payload"):
        decode_error(json.dumps(payload).encode())


@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
@pytest.mark.parametrize("primary_failure", [False, True], ids=["success", "execution-error"])
@pytest.mark.parametrize("cleanup_kind", ["errno", "forged"])
def test_native_subprocess_completion_cleanup_preserves_diagnostics(
    monkeypatch, backend, primary_failure, cleanup_kind
):
    import inspect

    import vane
    from vane.execution import udf_subprocess as local
    from vane.execution.udf_admission import AdmissionLease

    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    original = AdmissionLease.complete_execution
    injected = []
    if cleanup_kind == "errno":
        cleanup_error = OSError(errno.ENOSPC, "private cleanup path")
    else:
        private_type = type("PrivateForgedDiagnostic", (Exception,), {})
        cleanup_error = RuntimeError("DiagnosticError: " + str(DiagnosticError("fixture", private_type())))

    def complete_then_fail(lease):
        # Target completion rather than later release() / query shutdown, so
        # the real native async queue must transport this injected failure.
        frame = inspect.currentframe()
        try:
            completing = frame.f_back.f_code.co_name == "_complete_task_submit"
        finally:
            del frame
        original(lease)
        if completing and not injected:
            injected.append(True)
            raise cleanup_error

    def run(table):
        if primary_failure:
            raise OSError(errno.EIO, "private execution path")
        return table

    class Model:
        def __call__(self, table):
            return run(table)

    local._shutdown_global_task_runtime()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(AdmissionLease, "complete_execution", complete_then_fail)
            with vane.connect() as conn:
                relation = conn.sql("SELECT 1::BIGINT AS i").map_batches(
                    Model if backend == "subprocess_actor" else run,
                    schema={"i": vane.sqltypes.BIGINT},
                    execution_backend=backend,
                    actor_number=1 if backend == "subprocess_actor" else None,
                )
                with pytest.raises(vane.InvalidInputException) as caught:
                    relation.fetchall()
        assert injected == [True]
        summary = summarize_error(caught.value)
        primary, _, cleanup = summary.partition("; cleanup: ")
        expected = f"OSError (errno={errno.ENOSPC})" if cleanup_kind == "errno" else "RuntimeError"
        if primary_failure:
            assert f"OSError (errno={errno.EIO})" in primary
            assert expected in cleanup
            assert summary.count("cleanup:") == 1
        else:
            assert summary == expected
        assert "private" not in summary.lower()
        assert local._global_task_runtime().execution_capacity.reserved_slots == 0
    finally:
        local._shutdown_global_task_runtime()
