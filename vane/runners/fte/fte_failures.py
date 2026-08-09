# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from vane.runners.fte.fte_config import _FALSE_VALUES, _TRUE_VALUES

if TYPE_CHECKING:
    from vane.runners.fte.fte_types import FteTaskAttemptId


_MEMORY_FAILURE_CODES = frozenset(
    {
        "EXCEEDED_LOCAL_MEMORY_LIMIT",
        "OUT_OF_MEMORY",
    }
)


class FteTaskTerminalControlError(RuntimeError):
    """A mutable control reached a known task after it became terminal."""

    def __init__(
        self,
        *,
        operation: str,
        task_id: FteTaskAttemptId,
        status: Mapping[str, Any],
    ) -> None:
        self.operation = str(operation)
        self.task_id = task_id
        self.status = dict(status)
        operation_text = self.operation.replace("_", " ")
        super().__init__(f"cannot {operation_text} to terminal task {task_id}")


def _normalized_error_token(value: Any) -> str:
    return str(value or "").strip().upper().replace("-", "_").replace(" ", "_")


def _safe_failure_message(message: Any) -> str:
    if message is None:
        return ""
    try:
        return str(message)
    except BaseException:
        return f"<unprintable {type(message).__name__}>"


def _failure_payload(
    error_code: str,
    message: Any,
    *,
    error_type: str = "INTERNAL_ERROR",
) -> dict[str, Any]:
    if not isinstance(error_code, str):
        raise TypeError("FTE failure error_code must be a string")
    code = _normalized_error_token(error_code)
    if not code:
        raise ValueError("FTE failure error_code must be non-empty")
    return {
        "error_type": _normalized_error_token(error_type),
        "error_code": code,
        "message": _safe_failure_message(message),
    }


def _normalize_failure_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("FTE failure payload must be a mapping")
    if "error_code" not in payload:
        raise ValueError("FTE failure payload requires error_code")
    raw_code = payload["error_code"]
    if not isinstance(raw_code, str):
        raise TypeError("FTE failure payload error_code must be a string")
    code = _normalized_error_token(raw_code)
    if not code:
        raise ValueError("FTE failure payload requires error_code")
    normalized = dict(payload)
    normalized["error_code"] = code
    if "error_type" in normalized:
        normalized["error_type"] = _normalized_error_token(normalized["error_type"])
    return normalized


_UNREADABLE_EXCEPTION_ATTRIBUTE = object()


def _read_exception_attribute(error: BaseException, attribute: str) -> Any:
    try:
        return getattr(error, attribute, None)
    except BaseException:
        return _UNREADABLE_EXCEPTION_ATTRIBUTE


def _failure_exception_matches(
    error: Any,
    exception_types: tuple[type[BaseException], ...],
) -> bool:
    if not issubclass(type(error), BaseException):
        return False
    candidates = [error]
    candidate_ids = {id(error)}
    candidate_index = 0
    while candidate_index < len(candidates) and candidate_index < 16:
        candidate = candidates[candidate_index]
        candidate_index += 1
        if issubclass(type(candidate), exception_types):
            return True
        converted = _read_exception_attribute(candidate, "as_instanceof_cause")
        if callable(converted):
            try:
                converted_cause = converted()
            except BaseException:
                converted_cause = None
            if issubclass(type(converted_cause), BaseException) and id(converted_cause) not in candidate_ids:
                candidate_ids.add(id(converted_cause))
                candidates.append(converted_cause)
        chained_causes = [
            cause
            for cause in (
                _read_exception_attribute(candidate, "cause"),
                _read_exception_attribute(candidate, "__cause__"),
            )
            if issubclass(type(cause), BaseException)
        ]
        suppress_context = _read_exception_attribute(candidate, "__suppress_context__")
        if not chained_causes and suppress_context is False:
            context = _read_exception_attribute(candidate, "__context__")
            if issubclass(type(context), BaseException):
                chained_causes.append(context)
        for cause in chained_causes:
            if issubclass(type(cause), BaseException) and id(cause) not in candidate_ids:
                candidate_ids.add(id(cause))
                candidates.append(cause)
    return False


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
    return _failure_payload(
        "MISSING_SPOOLING_OUTPUT_STATS",
        f"Treating FINISHED task {attempt_id} as FAILED because spooling output stats are missing",
    )


def _is_memory_failure(payload: Any) -> bool:
    return _normalize_failure_payload(payload)["error_code"] in _MEMORY_FAILURE_CODES
