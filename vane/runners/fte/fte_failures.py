# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from vane.runners.fte.fte_config import _FALSE_VALUES, _TRUE_VALUES

if TYPE_CHECKING:
    from vane.runners.fte.fte_types import FteTaskAttemptId


def _failure_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, Mapping):
        parts: list[str] = []
        for key in ("error_code", "errorCode", "code", "type", "message", "error", "exception"):
            value = payload.get(key)
            if value is not None:
                parts.append(str(value))
        failure = payload.get("failure")
        if failure is not None and failure is not payload:
            parts.append(_failure_text(failure))
        return " ".join(part for part in parts if part)
    return str(payload)


def _normalized_error_token(value: Any) -> str:
    return str(value or "").strip().upper().replace("-", "_").replace(" ", "_")


def _failure_field(payload: Any, *keys: str) -> Any:
    if not isinstance(payload, Mapping):
        return None
    for key in keys:
        if key in payload:
            return payload.get(key)
    for nested_key in ("failure", "status", "error", "error_code", "errorCode"):
        nested = payload.get(nested_key)
        if nested is not payload:
            value = _failure_field(nested, *keys)
            if value is not None:
                return value
    return None


def _failure_explicit_retryable(payload: Any) -> bool | None:
    value = _failure_field(payload, "retryable", "is_retryable", "isRetryable")
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return None


def _failure_error_type(payload: Any) -> str:
    value = _failure_field(
        payload,
        "error_type",
        "errorType",
        "error_code_type",
        "errorCodeType",
        "type",
    )
    token = _normalized_error_token(value)
    if token in {"USER_ERROR", "INTERNAL_ERROR", "EXTERNAL"}:
        return token
    return ""


def _is_user_error_failure(payload: Any) -> bool:
    if _failure_error_type(payload) == "USER_ERROR":
        return True
    code = _normalized_error_token(_failure_field(payload, "error_code", "errorCode", "code", "name"))
    return code.startswith("USER_ERROR") or code.endswith("_USER_ERROR")


def _is_fatal_failure(payload: Any) -> bool:
    value = _failure_field(payload, "fatal", "is_fatal", "isFatal")
    if isinstance(value, bool):
        return value
    if value is not None:
        text = str(value).strip().lower()
        if text in _TRUE_VALUES:
            return True
        if text in _FALSE_VALUES:
            return False
    code = _normalized_error_token(_failure_field(payload, "error_code", "errorCode", "code", "name"))
    return code.startswith("FATAL") or code.endswith("_FATAL")


def _failure_allows_retry(payload: Any, *, default: bool = True) -> bool:
    explicit = _failure_explicit_retryable(payload)
    if explicit is not None:
        return explicit
    if _is_user_error_failure(payload) or _is_fatal_failure(payload):
        return False
    return default


def _missing_output_stats_failure(attempt_id: FteTaskAttemptId) -> dict[str, Any]:
    return {
        "error_type": "INTERNAL_ERROR",
        "error_code": "MISSING_SPOOLING_OUTPUT_STATS",
        "message": (f"Treating FINISHED task {attempt_id} as FAILED because spooling output stats are missing"),
    }


def _is_memory_failure(payload: Any) -> bool:
    text = _failure_text(payload).lower()
    return any(
        token in text for token in ("out_of_memory", "out of memory", "oom", "memory limit", "exceeded_local_memory")
    )
