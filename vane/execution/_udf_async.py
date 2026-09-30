# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded coroutine execution on the UDF executor's owned event loop."""

from __future__ import annotations

import asyncio
import functools
import inspect
import math
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Any, Generic, TypeVar

from vane.execution._diagnostics import attach_cleanup_error
from vane.execution._udf_validation import ensure_synchronous_udf_result, validate_udf_callable

T = TypeVar("T")
R = TypeVar("R")
_OPTIONS_ATTRIBUTE = "_vane_udf_call_options"


def _validated_timeout(timeout: Any) -> float | None:
    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, Real) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_s must be a positive finite number; bool is not accepted")
    return float(timeout)


@dataclass(frozen=True)
class UDFCallOptions:
    execution_kind: str
    invocation_granularity: str
    max_concurrency: int
    timeout_s: float | None

    def as_payload(self) -> dict[str, Any]:
        return {
            "execution_kind": self.execution_kind,
            "invocation_granularity": self.invocation_granularity,
            "max_concurrency": self.max_concurrency,
            "timeout_s": self.timeout_s,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> UDFCallOptions:
        if "payload_version" in payload and payload["payload_version"] != 2:
            raise ValueError("unsupported UDF payload_version; expected 2")
        kind = payload.get("execution_kind", "sync")
        if kind not in {"sync", "async"}:
            raise ValueError("UDF execution_kind must be sync or async")
        if kind == "sync":
            return cls("sync", "row" if payload.get("call_mode") == "map" else "batch", 1, None)
        granularity = payload.get("invocation_granularity")
        if granularity not in {"row", "batch"}:
            raise ValueError("async UDF requires row or batch invocation_granularity")
        concurrency = payload.get("max_concurrency")
        if type(concurrency) is not int or concurrency <= 0:
            raise ValueError("async UDF max_concurrency must be a positive integer")
        timeout = _validated_timeout(payload.get("timeout_s"))
        if payload.get("call_mode") == "flat_map":
            raise TypeError("flat_map does not support async UDFs")
        if payload.get("call_mode") == "map" and granularity != "row":
            raise ValueError("async map requires row invocation_granularity")
        return cls(kind, granularity, concurrency, timeout)


def resolve_call_options(
    fn: Any,
    call_mode: str,
    max_concurrency: int | None = None,
    timeout_s: float | None = None,
) -> UDFCallOptions:
    kind = validate_udf_callable(fn)
    if call_mode not in {"map", "map_batches", "map_batches_rows", "flat_map"}:
        raise ValueError(f"unsupported UDF call mode: {call_mode}")
    if kind == "async" and call_mode == "flat_map":
        raise TypeError("flat_map does not support async UDFs")
    granularity = "row" if call_mode in {"map", "flat_map"} else "batch"
    declared = getattr(fn, _OPTIONS_ATTRIBUTE, None)
    if declared is not None:
        if not isinstance(declared, UDFCallOptions) or declared.execution_kind != kind:
            raise TypeError("invalid UDF callable execution metadata")
        if max_concurrency is not None or timeout_s is not None:
            raise ValueError("async options are already configured on the UDF and cannot be overridden")
        if declared.invocation_granularity != granularity and not (
            declared.invocation_granularity == "row"
            and call_mode in {"map_batches", "map_batches_rows"}
            and getattr(fn, "_vane_row_actor_adapter", False)
        ):
            raise ValueError("UDF callable invocation granularity does not match the entrypoint")
        return UDFCallOptions.from_payload({"call_mode": call_mode, **declared.as_payload()})
    if max_concurrency is not None and (type(max_concurrency) is not int or max_concurrency <= 0):
        raise ValueError("max_concurrency must be a positive integer; bool is not accepted")
    timeout_s = _validated_timeout(timeout_s)
    if kind == "sync" and (max_concurrency is not None or timeout_s is not None):
        raise ValueError("max_concurrency and timeout_s require an async UDF")
    default_concurrency = 32 if kind == "async" and granularity == "row" and not inspect.isclass(fn) else 1
    return UDFCallOptions(kind, granularity, max_concurrency or default_concurrency, timeout_s)


def callable_payload_options(fn: Any, call_mode: str) -> dict[str, Any]:
    """Called by native payload builders before serialization."""
    return resolve_call_options(fn, call_mode).as_payload()


def configure_udf_callable(
    fn: Any,
    call_mode: str,
    max_concurrency: int | None = None,
    timeout_s: float | None = None,
) -> Any:
    """Capture configuration on a new callable, leaving the user's definition untouched."""
    if not (inspect.isfunction(fn) or inspect.ismethod(fn) or inspect.isclass(fn)):
        raise TypeError("UDFs require a Python function, bound method, or callable class")
    options = resolve_call_options(fn, call_mode, max_concurrency, timeout_s)
    if options.execution_kind == "sync":
        return fn
    if inspect.isclass(fn):

        class Configured(fn):
            pass

        configured = Configured
        configured.__name__ = fn.__name__
        configured.__qualname__ = fn.__qualname__
        configured.__module__ = fn.__module__
    else:

        @functools.wraps(fn)
        async def configured(*args: Any, **kwargs: Any) -> Any:
            return await fn(*args, **kwargs)

    setattr(configured, _OPTIONS_ATTRIBUTE, options)
    return configured


def set_call_options(fn: Any, options: UDFCallOptions) -> None:
    setattr(fn, _OPTIONS_ATTRIBUTE, options)


async def await_udf_call(result: Awaitable[R], timeout_s: float | None) -> R:
    return await result if timeout_s is None else await asyncio.wait_for(result, timeout_s)


async def run_rows(indices: Iterable[T], invoke: Callable[[T], Awaitable[None]], max_concurrency: int) -> None:
    """Keep O(concurrency) tasks, replenishing each worker without group barriers."""
    iterator = iter(indices)
    failed = asyncio.Event()

    async def worker(first: T) -> None:
        if failed.is_set():
            return
        try:
            await invoke(first)
            for index in iterator:
                if failed.is_set():
                    return
                await invoke(index)
        except BaseException:
            failed.set()
            raise

    tasks: list[asyncio.Task[None]] = []
    try:
        for _ in range(max_concurrency):
            try:
                first = next(iterator)
            except StopIteration:
                break
            tasks.append(asyncio.create_task(worker(first)))
        if tasks:
            await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class AsyncBatchWindow(Generic[T, R]):
    """One ordered window; completed later batches still occupy a slot."""

    def __init__(self, batches: Iterable[T], invoke: Callable[[T], Coroutine[Any, Any, R]], concurrency: int) -> None:
        self._batches = iter(batches)
        self._invoke = invoke
        self._concurrency = concurrency
        self._tasks: list[asyncio.Task[R]] = []
        self._closed = False

    async def next(self) -> tuple[bool, R | None]:
        if self._closed:
            return False, None
        try:
            # Observe failures before replenishing a window paused by its consumer.
            for task in self._tasks:
                if task.done() and (task.cancelled() or task.exception() is not None):
                    task.result()
            while len(self._tasks) < self._concurrency:
                try:
                    batch = next(self._batches)
                except StopIteration:
                    break
                self._tasks.append(asyncio.create_task(self._invoke(batch)))
            if not self._tasks:
                self._closed = True
                return False, None
            first = self._tasks[0]
            # Fail fast even when a later batch fails behind a slow first batch.
            while not first.done():
                pending = [task for task in self._tasks if not task.done()]
                await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in self._tasks:
                    if task.done() and (task.cancelled() or task.exception() is not None):
                        task.result()
            result = first.result()
            self._tasks.pop(0)
            return True, result
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        self._closed = True
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class AsyncClassInstance:
    """Construct, open, invoke and close an instance on one running loop."""

    def __init__(self, user_class: type, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None) -> None:
        self._class = user_class
        self._args = args
        self._kwargs = dict(kwargs or {})
        self.instance: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._opened = False
        self._closed = False

    async def open(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError("async UDF instance cannot be used from another event loop")
        if self._closed:
            raise RuntimeError("async UDF instance is closed")
        if self._opened:
            return
        self._loop = loop
        try:
            self.instance = ensure_synchronous_udf_result(self._class(*self._args, **self._kwargs))
            hook = getattr(self.instance, "aopen", None)
            if hook is not None:
                result = await hook()
                if ensure_synchronous_udf_result(result) is not None:
                    raise TypeError("async UDF aopen must return None")
            self._opened = True
        except BaseException as primary:
            try:
                await self.close()
            except BaseException as cleanup:
                attach_cleanup_error(primary, cleanup)
            raise

    def require_instance(self) -> Any:
        if not self._opened or self._closed:
            raise RuntimeError("async UDF eager calls require 'async with' on the class instance")
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("async UDF instance cannot be used from another event loop")
        return self.instance

    async def close(self) -> None:
        if self._closed:
            return
        if self._loop is not None and asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("async UDF instance cannot be closed from another event loop")
        hook = getattr(self.instance, "aclose", None)
        if hook is not None:
            result = await hook()
            if ensure_synchronous_udf_result(result) is not None:
                raise TypeError("async UDF aclose must return None")
        self.instance = None
        self._closed = True
