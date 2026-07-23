# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
from typing import Any

import pytest

import duckdb
from duckdb._ray_errors import RemoteRayException
from duckdb.runners.ray import driver
from duckdb.runners.ray.safe_get import resolve_object_refs_blocking


def _chained_error(label: str) -> RuntimeError:
    try:
        raise duckdb.NotImplementedException(f"{label} original")
    except duckdb.NotImplementedException as cause:
        try:
            raise RuntimeError(f"{label} outer") from cause
        except RuntimeError as outer:
            return outer


def _assert_restored_chain(exc: RuntimeError, label: str) -> None:
    assert str(exc) == f"{label} outer"
    assert isinstance(exc.__cause__, duckdb.NotImplementedException)
    assert str(exc.__cause__) == f"{label} original"
    assert exc.__cause__.remote_exception_type == "_duckdb.NotImplementedException"
    assert f"NotImplementedException: {label} original" in exc.__cause__.remote_traceback


def test_remote_ray_exception_pickle_round_trip_restores_cause_chain():
    transported = RemoteRayException.from_exception(_chained_error("pickle"))

    restored = pickle.loads(pickle.dumps(transported)).restore()

    assert isinstance(restored, RuntimeError)
    _assert_restored_chain(restored, "pickle")


def test_safe_get_restores_serialized_ray_exception_chain():
    class FakeRayTaskError(RuntimeError):
        def __init__(self, cause: BaseException) -> None:
            self.cause = cause
            super().__init__(cause)

    class FailedFuture:
        def result(self, timeout=None):
            raise FakeRayTaskError(RemoteRayException.from_exception(_chained_error("safe-get")))

    class FailedRef:
        def future(self):
            return FailedFuture()

    with pytest.raises(RuntimeError) as exc_info:
        resolve_object_refs_blocking(FailedRef())

    _assert_restored_chain(exc_info.value, "safe-get")


def test_ray_driver_client_restores_preflight_stream_and_copy_causes(ray_local, monkeypatch):
    import ray

    @ray.remote
    def fail_with_chain(label: str) -> None:
        import duckdb as remote_duckdb
        from duckdb._ray_errors import remote_ray_exception as build_remote_ray_exception

        try:
            raise remote_duckdb.NotImplementedException(f"{label} original")
        except remote_duckdb.NotImplementedException as cause:
            raise build_remote_ray_exception(f"{label} outer", cause) from cause

    @ray.remote
    def succeed(value: Any = None) -> Any:
        return value

    class SuccessMethod:
        def __init__(self, value: Any = None) -> None:
            self.value = value

        def remote(self, *_args, **_kwargs):
            return succeed.remote(self.value)

    class FailureMethod:
        def __init__(self, label: str) -> None:
            self.label = label

        def remote(self, *_args, **_kwargs):
            return fail_with_chain.remote(self.label)

    class Plan:
        def idx(self) -> str:
            return "remote-error-plan"

    class Runner:
        install_env_overrides = SuccessMethod()
        close_plan = SuccessMethod()
        progress_snapshot = SuccessMethod()

        def __init__(self, failure_path: str) -> None:
            self.run_plan = FailureMethod("preflight") if failure_path == "preflight" else SuccessMethod()
            self.get_next_partition = FailureMethod("stream") if failure_path == "stream" else SuccessMethod(None)
            self.run_copy_plan = FailureMethod("copy") if failure_path == "copy" else SuccessMethod()

    monkeypatch.setattr(driver, "_collect_vane_env_overrides", dict)
    monkeypatch.setattr(driver, "progress_enabled", lambda: False)

    for failure_path in ("preflight", "stream", "copy"):
        client = object.__new__(driver.RayQueryDriverClient)
        client.runner = Runner(failure_path)
        with pytest.raises(RuntimeError) as exc_info:
            if failure_path == "copy":
                client.run_copy_plan(Plan())
            else:
                list(client.stream_plan(Plan()))
        _assert_restored_chain(exc_info.value, failure_path)
