# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import traceback
from typing import Any

_MAX_REMOTE_EXCEPTION_CHAIN_DEPTH = 16
_MAX_REMOTE_TRACEBACK_CHARS = 64 * 1024


class RemoteRayException(RuntimeError):
    """Pickle-safe carrier for an exception chain crossing a Ray boundary."""

    def __init__(self, message: str, payload: dict[str, Any]) -> None:
        self.message = str(message)
        self.payload = dict(payload)
        super().__init__(self.message, self.payload)
        cause = _restore_remote_exception(self.payload.get("cause"))
        if cause is not None:
            self.__cause__ = cause
            self.__suppress_context__ = True

    @classmethod
    def from_exception(cls, exc: BaseException) -> RemoteRayException:
        payload = _serialize_remote_exception(exc)
        return cls(str(payload["message"]), payload)

    def restore(self) -> BaseException:
        return _restore_remote_exception(self.payload) or RuntimeError(self.message)

    def __str__(self) -> str:
        return self.message


class RemoteRayCause(RuntimeError):
    """Fallback when the original remote exception type cannot be rebuilt."""

    def __init__(self, remote_type: str, message: str) -> None:
        self.remote_type = str(remote_type)
        self.remote_message = str(message)
        super().__init__(f"{self.remote_type}: {self.remote_message}")


def remote_ray_exception(message: str, cause: BaseException) -> RemoteRayException:
    """Build a transport exception while retaining the in-process cause."""
    outer = RuntimeError(str(message))
    outer.__cause__ = cause
    outer.__suppress_context__ = True
    transported = RemoteRayException.from_exception(outer)
    transported.__cause__ = cause
    transported.__suppress_context__ = True
    return transported


def restore_remote_ray_exception(exc: BaseException) -> BaseException | None:
    """Restore a transported exception from a RayTaskError or direct carrier."""
    cause = getattr(exc, "cause", None)
    if isinstance(cause, RemoteRayException):
        return cause.restore()
    if isinstance(exc, RemoteRayException):
        return exc.restore()
    return None


def _safe_exception_message(exc: BaseException) -> str:
    try:
        return str(exc)
    except BaseException:
        return f"<{type(exc).__name__} failed to render>"


def _safe_exception_traceback(exc: BaseException) -> str:
    try:
        rendered = "".join(
            traceback.format_exception(
                type(exc),
                exc,
                exc.__traceback__,
                chain=False,
            )
        )
    except BaseException:
        return ""
    if len(rendered) <= _MAX_REMOTE_TRACEBACK_CHARS:
        return rendered
    return rendered[-_MAX_REMOTE_TRACEBACK_CHARS:]


def _serialize_remote_exception(
    exc: BaseException,
    *,
    depth: int = 0,
    seen: set[int] | None = None,
) -> dict[str, Any]:
    if seen is None:
        seen = set()
    exc_type = type(exc)
    payload: dict[str, Any] = {
        "module": str(getattr(exc_type, "__module__", "builtins")),
        "qualname": str(getattr(exc_type, "__qualname__", exc_type.__name__)),
        "message": _safe_exception_message(exc),
        "traceback": _safe_exception_traceback(exc),
        "cause": None,
    }
    if depth >= _MAX_REMOTE_EXCEPTION_CHAIN_DEPTH or id(exc) in seen:
        return payload
    seen.add(id(exc))
    cause = exc.__cause__
    if cause is not None:
        payload["cause"] = _serialize_remote_exception(
            cause,
            depth=depth + 1,
            seen=seen,
        )
    return payload


def _resolve_remote_exception_type(module_name: str, qualname: str) -> type[Exception] | None:
    if not module_name or not qualname or "<locals>" in qualname:
        return None
    try:
        value: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            value = getattr(value, part)
    except (AttributeError, ImportError, ValueError):
        return None
    if not isinstance(value, type) or not issubclass(value, Exception):
        return None
    return value


def _restore_remote_exception(payload: Any) -> BaseException | None:
    if not isinstance(payload, dict):
        return None
    module_name = str(payload.get("module") or "builtins")
    qualname = str(payload.get("qualname") or "RuntimeError")
    message = str(payload.get("message") or "")
    remote_type = f"{module_name}.{qualname}"
    exception_type = _resolve_remote_exception_type(module_name, qualname)
    if exception_type is None:
        restored: BaseException = RemoteRayCause(remote_type, message)
    else:
        try:
            restored = exception_type(message)
        except BaseException:
            restored = RemoteRayCause(remote_type, message)

    try:
        restored.remote_exception_type = remote_type  # type: ignore[attr-defined]
        restored.remote_traceback = str(payload.get("traceback") or "")  # type: ignore[attr-defined]
    except BaseException:
        pass

    cause = _restore_remote_exception(payload.get("cause"))
    if cause is not None:
        restored.__cause__ = cause
        restored.__suppress_context__ = True
    return restored
