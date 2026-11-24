# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Typed diagnostics on the owned UDF subprocess control channel."""

from __future__ import annotations

import json

from vane.ai._errors import DiagnosticError, _diagnostic_message, _restore_diagnostic_error, summarize_error
from vane.execution._diagnostics import exception_message_from_args


def error_payload(error: BaseException, message: str) -> bytes:
    # Capture from the exception object, independently of its formatted body.
    # A user RuntimeError containing a native envelope stays a RuntimeError.
    return json.dumps({"message": message, "summary": summarize_error(error)}).encode("utf-8")


def decode_error(payload: bytes, *, prefix: str = "") -> tuple[str, DiagnosticError]:
    value = json.loads(payload)
    if (
        type(value) is not dict
        or set(value) != {"message", "summary"}
        or type(message := value["message"]) is not str
        or len(message.encode("utf-8")) > 4096
        or type(summary := value["summary"]) is not str
        or not 0 < len(summary) <= 512
    ):
        raise RuntimeError("Invalid UDF subprocess error payload")
    # Only the worker's structured field is diagnostic authority. Keep the
    # existing bounded traceback for local debugging, outside the safe header.
    diagnostic = _restore_diagnostic_error(_diagnostic_message(summary, prefix + message), summary)
    return message, diagnostic


def capture_cleanup(message: str, primary: BaseException, *cleanup: BaseException) -> DiagnosticError:
    failure = RuntimeError(message)
    failure.primary_error = primary  # type: ignore[attr-defined]
    failure.cleanup_errors = cleanup  # type: ignore[attr-defined]
    summary = summarize_error(failure)
    return _restore_diagnostic_error(_diagnostic_message(summary, message), summary)


def native_result_error(error: BaseException) -> DiagnosticError:
    # The native async queue reads str(result), unlike py::error_already_set
    # which includes the exception type. Stamp every failed result here, before
    # that conversion, so an arbitrary RuntimeError body cannot forge a header.
    summary = summarize_error(error)
    detail = exception_message_from_args(error) or "Subprocess UDF execution failed"
    message = "DiagnosticError: " + _diagnostic_message(summary, detail[:4096])
    return _restore_diagnostic_error(message, summary)
