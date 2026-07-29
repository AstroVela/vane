# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from duckdb import OutOfMemoryException
from duckdb.runners.fte.fte_failures import (
    _failure_allows_retry,
    _failure_payload,
    _is_memory_failure,
    _normalize_failure_payload,
)
from duckdb.runners.fte.fte_state import FteTaskState
from duckdb.runners.fte.fte_worker_runtime import FteTaskExecution, _failure_payload_from_exception
from duckdb.runners.fte.memory_config import DuckDBMemoryLimitError


@pytest.mark.parametrize(
    "error_code",
    [
        "OUT_OF_MEMORY",
        "out-of-memory",
        "EXCEEDED_LOCAL_MEMORY_LIMIT",
        "exceeded local memory limit",
    ],
)
def test_memory_failure_matches_normalized_error_code(error_code):
    assert _is_memory_failure({"error_code": error_code}) is True


@pytest.mark.parametrize(
    "message",
    [
        "OOM",
        "out of memory",
        "bloom filter construction failed",
        "no room left in the queue",
        "zoom request failed",
    ],
)
def test_memory_failure_ignores_message(message):
    failure = _failure_payload("GENERIC_INTERNAL_ERROR", message)

    assert _is_memory_failure(failure) is False


def test_memory_failure_uses_code_when_message_is_unrelated():
    failure = _failure_payload("OUT_OF_MEMORY", "bloom filter construction failed")

    assert _is_memory_failure(failure) is True


@pytest.mark.parametrize(
    "error_code",
    [
        "BLOOM_FILTER_ERROR",
        "MEMORY_LIMIT_CONFIGURATION_FAILED",
    ],
)
def test_memory_failure_requires_known_error_code(error_code):
    assert _is_memory_failure({"error_code": error_code}) is False


@pytest.mark.parametrize(
    "exception",
    [
        OutOfMemoryException("duckdb exhausted memory"),
        MemoryError("python exhausted memory"),
    ],
)
def test_exception_failure_payload_maps_memory_exceptions(exception):
    failure = _failure_payload_from_exception(exception)

    assert failure["error_code"] == "OUT_OF_MEMORY"


@pytest.mark.parametrize(
    ("exception", "expected_code", "expected_retryable"),
    [
        (RuntimeError("task failed"), "GENERIC_INTERNAL_ERROR", None),
        (
            DuckDBMemoryLimitError("failed to configure memory limit"),
            "MEMORY_LIMIT_CONFIGURATION_FAILED",
            False,
        ),
    ],
)
def test_exception_failure_payload_assigns_non_memory_error_code(
    exception,
    expected_code,
    expected_retryable,
):
    failure = _failure_payload_from_exception(exception)

    assert failure["error_code"] == expected_code
    assert failure.get("retryable") is expected_retryable
    if expected_retryable is False:
        assert _failure_allows_retry(failure) is False


def test_exception_failure_payload_reads_structured_memory_exception_chain():
    exception = RuntimeError("task execution failed")
    exception.__cause__ = OutOfMemoryException("duckdb exhausted memory")

    failure = _failure_payload_from_exception(exception)

    assert failure["error_code"] == "OUT_OF_MEMORY"


def test_exception_failure_payload_reads_implicit_memory_exception_context():
    try:
        raise OutOfMemoryException("duckdb exhausted memory")
    except OutOfMemoryException:
        try:
            raise RuntimeError("task execution failed")
        except RuntimeError as exception:
            failure = _failure_payload_from_exception(exception)

    assert failure["error_code"] == "OUT_OF_MEMORY"


def test_exception_failure_payload_ignores_suppressed_memory_exception_context():
    try:
        raise OutOfMemoryException("duckdb exhausted memory")
    except OutOfMemoryException:
        try:
            raise RuntimeError("replacement task failure") from None
        except RuntimeError as exception:
            failure = _failure_payload_from_exception(exception)

    assert failure["error_code"] == "GENERIC_INTERNAL_ERROR"


@pytest.mark.parametrize(
    "unreadable_attribute",
    [
        "as_instanceof_cause",
        "cause",
        "__cause__",
        "__suppress_context__",
        "__context__",
    ],
)
def test_task_execution_reaches_failed_state_when_exception_chain_attributes_raise(
    unreadable_attribute,
):
    class _UnreadableExceptionChain(RuntimeError):
        def __getattribute__(self, name):
            if name == unreadable_attribute:
                raise RuntimeError(f"{name} is unreadable")
            return super().__getattribute__(name)

    async def run_task():
        async def execute_fn(_request):
            raise _UnreadableExceptionChain("task failed")

        execution = FteTaskExecution(
            {"task_id": "q-unreadable-chain.0.0.0"},
            execute_fn,
            default_task_memory_bytes=1,
        )
        execution.start()
        assert execution._future is not None
        await execution._future
        return execution.status

    status = asyncio.run(run_task())

    assert status.state == FteTaskState.FAILED
    assert status.failure is not None
    assert status.failure["error_code"] == "GENERIC_INTERNAL_ERROR"


def test_task_execution_reaches_failed_state_when_exception_message_is_unprintable():
    class _UnprintableException(RuntimeError):
        def __str__(self):
            raise RuntimeError("exception message is unreadable")

    async def run_task():
        async def execute_fn(_request):
            raise _UnprintableException()

        execution = FteTaskExecution(
            {"task_id": "q-unprintable-exception.0.0.0"},
            execute_fn,
            default_task_memory_bytes=1,
        )
        execution.start()
        assert execution._future is not None
        await execution._future
        return execution.status

    status = asyncio.run(run_task())

    assert status.state == FteTaskState.FAILED
    assert status.failure is not None
    assert status.failure["error_code"] == "GENERIC_INTERNAL_ERROR"
    assert status.failure["message"] == "<unprintable _UnprintableException>"


def test_task_execution_publishes_out_of_memory_error_code():
    async def run_task():
        async def execute_fn(_request):
            raise OutOfMemoryException("duckdb exhausted memory")

        execution = FteTaskExecution(
            {"task_id": "q-out-of-memory.0.0.0"},
            execute_fn,
            default_task_memory_bytes=1,
        )
        execution.start()
        assert execution._future is not None
        await execution._future
        return execution.status

    status = asyncio.run(run_task())

    assert status.state == FteTaskState.FAILED
    assert status.failure is not None
    assert status.failure["error_code"] == "OUT_OF_MEMORY"


@pytest.mark.parametrize(
    ("payload", "error_type", "message"),
    [
        ("OOM", TypeError, "must be a mapping"),
        ({"error_code": 1}, TypeError, "error_code must be a string"),
        ({"message": "OOM"}, ValueError, "requires error_code"),
        ({"error_code": "  "}, ValueError, "requires error_code"),
    ],
)
def test_failure_payload_rejects_missing_error_code(payload, error_type, message):
    with pytest.raises(error_type, match=message):
        _normalize_failure_payload(payload)


def test_task_execution_enforces_failure_code_for_terminal_status():
    async def execute_fn(_request):
        return None

    execution = FteTaskExecution(
        {"task_id": "q-failure-code.0.0.0"},
        execute_fn,
        default_task_memory_bytes=1,
    )

    with pytest.raises(ValueError, match="requires error_code"):
        execution._transition(
            FteTaskState.FAILED,
            failure={"message": "out of memory"},
        )

    assert execution.status.state == FteTaskState.PLANNED
    canceled = execution.cancel()
    assert canceled.failure is not None
    assert canceled.failure["error_code"] == "TASK_CANCELED"
