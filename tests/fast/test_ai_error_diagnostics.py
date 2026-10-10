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
        (RuntimeError("CUDNN_STATUS_NOT_INITIALIZED private payload"), "cuDNN initialization failed"),
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
