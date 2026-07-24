# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import traceback
from typing import Any

_MAX_REMOTE_EXCEPTION_CHAIN_DEPTH = 16
_MAX_REMOTE_EXCEPTION_MESSAGE_CHARS = 64 * 1024
_MAX_REMOTE_TRACEBACK_CHARS = 64 * 1024
_MAX_REMOTE_EXCEPTION_VALUE_DEPTH = 16
_MAX_REMOTE_EXCEPTION_VALUE_ITEMS = 1024
_MAX_REMOTE_EXCEPTION_VALUE_BYTES = 64 * 1024


class RemoteRayException(RuntimeError):
    """Pickle-safe carrier for an exception chain crossing a Ray boundary."""

    def __init__(self, message: str, payload: dict[str, Any]) -> None:
        self.message = str(message)
        self.payload = dict(payload)
        super().__init__(self.message, self.payload)
        chain_seen = {id(payload), id(self.payload)}
        cause = _restore_optional_remote_exception(
            self.payload["cause"],
            depth=1,
            seen=chain_seen,
        )
        context = _restore_optional_remote_exception(
            self.payload["context"],
            depth=1,
            seen=chain_seen,
        )
        if cause is not None:
            self.__cause__ = cause
        if context is not None:
            self.__context__ = context
        self.__suppress_context__ = self.payload["suppress_context"]

    @classmethod
    def from_exception(cls, exc: BaseException) -> RemoteRayException:
        payload = _serialize_remote_exception(exc)
        return cls(str(payload["message"]), payload)

    def restore(self) -> BaseException:
        return _restore_remote_exception(self.payload)

    def __str__(self) -> str:
        return self.message


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
        rendered = str(exc)
    except BaseException:
        return f"<{type(exc).__name__} failed to render>"
    if len(rendered) <= _MAX_REMOTE_EXCEPTION_MESSAGE_CHARS:
        return rendered
    suffix = "... <remote exception message truncated>"
    return rendered[: _MAX_REMOTE_EXCEPTION_MESSAGE_CHARS - len(suffix)] + suffix


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


def _copy_transport_value(
    value: Any,
    *,
    depth: int = 0,
    seen: set[int] | None = None,
    budget: list[int] | None = None,
) -> Any:
    if budget is None:
        budget = [
            _MAX_REMOTE_EXCEPTION_VALUE_ITEMS,
            _MAX_REMOTE_EXCEPTION_VALUE_BYTES,
        ]
    if budget[0] <= 0:
        raise ValueError("remote exception constructor exceeds the item limit")
    budget[0] -= 1

    if value is None or type(value) in (bool, int, float, str, bytes):
        if type(value) is int:
            value_size = max(1, (value.bit_length() + 7) // 8)
        elif type(value) is float:
            value_size = 8
        elif type(value) is str:
            if len(value) > budget[1]:
                raise ValueError("remote exception constructor exceeds the byte limit")
            value_size = len(value.encode("utf-8", errors="surrogatepass"))
        elif type(value) is bytes:
            value_size = len(value)
        else:
            value_size = 1
        if value_size > budget[1]:
            raise ValueError("remote exception constructor exceeds the byte limit")
        budget[1] -= value_size
        return value

    if depth >= _MAX_REMOTE_EXCEPTION_VALUE_DEPTH:
        raise ValueError("remote exception constructor exceeds the depth limit")
    if type(value) not in (dict, list, tuple):
        raise TypeError("remote exception constructor contains an unsupported value")

    if seen is None:
        seen = set()
    if id(value) in seen:
        raise ValueError("remote exception constructor contains a cycle")
    seen.add(id(value))
    try:
        if type(value) is dict:
            copied_dict: dict[Any, Any] = {}
            for key, item in value.items():
                copied_key = _copy_transport_value(
                    key,
                    depth=depth + 1,
                    seen=seen,
                    budget=budget,
                )
                copied_item = _copy_transport_value(
                    item,
                    depth=depth + 1,
                    seen=seen,
                    budget=budget,
                )
                copied_dict[copied_key] = copied_item
            return copied_dict

        copied_items = []
        for item in value:
            copied_item = _copy_transport_value(
                item,
                depth=depth + 1,
                seen=seen,
                budget=budget,
            )
            copied_items.append(copied_item)
        return tuple(copied_items) if type(value) is tuple else copied_items
    finally:
        seen.remove(id(value))


def _exception_constructor(exc: BaseException) -> dict[str, Any]:
    reduced = exc.__reduce__()
    if type(reduced) is not tuple or len(reduced) not in (2, 3):
        raise TypeError("remote exception has an unsupported reducer")
    if reduced[0] is not type(exc) or type(reduced[1]) is not tuple:
        raise TypeError("remote exception has an unsupported constructor")

    budget = [
        _MAX_REMOTE_EXCEPTION_VALUE_ITEMS,
        _MAX_REMOTE_EXCEPTION_VALUE_BYTES,
    ]
    args = _copy_transport_value(reduced[1], budget=budget)
    state = _copy_transport_value(
        reduced[2] if len(reduced) == 3 else None,
        budget=budget,
    )
    if state is not None and (type(state) is not dict or any(type(name) is not str for name in state)):
        raise TypeError("remote exception has an unsupported constructor state")
    return {"args": args, "state": state}


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
        "constructor": _exception_constructor(exc),
        "cause": None,
        "context": None,
        "suppress_context": bool(exc.__suppress_context__),
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
    elif exc.__context__ is not None and not exc.__suppress_context__:
        payload["context"] = _serialize_remote_exception(
            exc.__context__,
            depth=depth + 1,
            seen=seen,
        )
    return payload


def _resolve_remote_exception_type(module_name: str, qualname: str) -> type[BaseException]:
    if not module_name or not qualname or "<locals>" in qualname:
        raise TypeError("remote exception type is not importable")
    value: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        value = getattr(value, part)
    if not isinstance(value, type) or not issubclass(value, BaseException):
        raise TypeError("remote exception type is not an exception")
    return value


def _restore_optional_remote_exception(
    payload: Any,
    *,
    depth: int,
    seen: set[int],
) -> BaseException | None:
    if payload is None:
        return None
    return _restore_remote_exception(payload, depth=depth, seen=seen)


def _restore_remote_exception(
    payload: Any,
    *,
    depth: int = 0,
    seen: set[int] | None = None,
) -> BaseException:
    if depth > _MAX_REMOTE_EXCEPTION_CHAIN_DEPTH:
        raise ValueError("remote exception chain exceeds the depth limit")
    if type(payload) is not dict:
        raise TypeError("remote exception payload must be a dict")
    if seen is None:
        seen = set()
    if id(payload) in seen:
        raise ValueError("remote exception chain contains a cycle")
    seen.add(id(payload))
    try:
        return _restore_remote_exception_payload(payload, depth=depth, seen=seen)
    finally:
        seen.remove(id(payload))


def _restore_remote_exception_payload(
    payload: dict[str, Any],
    *,
    depth: int,
    seen: set[int],
) -> BaseException:
    module_name = payload["module"]
    qualname = payload["qualname"]
    message = payload["message"]
    remote_traceback = payload["traceback"]
    suppress_context = payload["suppress_context"]
    if not all(type(value) is str for value in (module_name, qualname, message, remote_traceback)):
        raise TypeError("remote exception payload strings must be strings")
    if type(suppress_context) is not bool:
        raise TypeError("remote exception suppress_context must be a bool")

    remote_type = f"{module_name}.{qualname}"
    exception_type = _resolve_remote_exception_type(module_name, qualname)
    constructor = payload["constructor"]
    if type(constructor) is not dict:
        raise TypeError("remote exception constructor must be a dict")
    budget = [
        _MAX_REMOTE_EXCEPTION_VALUE_ITEMS,
        _MAX_REMOTE_EXCEPTION_VALUE_BYTES,
    ]
    args = _copy_transport_value(constructor["args"], budget=budget)
    state = _copy_transport_value(
        constructor["state"],
        budget=budget,
    )
    if type(args) is not tuple:
        raise TypeError("remote exception constructor arguments must be a tuple")
    if state is not None and (type(state) is not dict or any(type(name) is not str for name in state)):
        raise TypeError("remote exception constructor state must be a string-keyed dict")
    restored = exception_type(*args)
    if not isinstance(restored, BaseException):
        raise TypeError("remote exception constructor returned a non-exception")
    if state is not None:
        restored.__setstate__(state)

    try:
        restored.remote_exception_type = remote_type  # type: ignore[attr-defined]
        restored.remote_traceback = remote_traceback  # type: ignore[attr-defined]
    except BaseException:
        pass

    cause = _restore_optional_remote_exception(
        payload["cause"],
        depth=depth + 1,
        seen=seen,
    )
    context = _restore_optional_remote_exception(
        payload["context"],
        depth=depth + 1,
        seen=seen,
    )
    if cause is not None:
        restored.__cause__ = cause
    if context is not None:
        restored.__context__ = context
    restored.__suppress_context__ = suppress_context
    return restored
