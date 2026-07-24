# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
from typing import Any

import pytest

import duckdb
import duckdb._ray_errors as ray_errors
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


@pytest.mark.parametrize(
    ("original", "attributes"),
    [
        (KeyError("missing-key"), {}),
        (
            FileNotFoundError(2, "No such file", "/tmp/input.parquet"),
            {"errno": 2, "filename": "/tmp/input.parquet"},
        ),
        (
            ImportError("provider failed", name="provider", path="/tmp/provider.py"),
            {"name": "provider", "path": "/tmp/provider.py"},
        ),
    ],
)
def test_remote_ray_exception_preserves_builtin_constructor_state(original, attributes):
    transported = RemoteRayException.from_exception(original)

    restored = pickle.loads(pickle.dumps(transported)).restore()

    assert type(restored) is type(original)
    assert restored.args == original.args
    assert str(restored) == str(original)
    for name, value in attributes.items():
        assert getattr(restored, name) == value


@pytest.mark.parametrize(
    "argument",
    [
        object(),
        list(range(2048)),
        "x" * (65 * 1024),
    ],
    ids=["unsupported-type", "too-many-items", "too-many-bytes"],
)
def test_remote_ray_exception_rejects_unsafe_constructor_arguments(argument):
    original = RuntimeError(argument)

    with pytest.raises((TypeError, ValueError), match="remote exception constructor"):
        RemoteRayException.from_exception(original)


def test_remote_ray_exception_pickle_round_trip_restores_implicit_context():
    try:
        try:
            raise KeyError("context")
        except KeyError:
            raise RuntimeError("outer")
    except RuntimeError as original:
        transported = RemoteRayException.from_exception(original)

    restored = pickle.loads(pickle.dumps(transported)).restore()

    assert isinstance(restored, RuntimeError)
    assert restored.__cause__ is None
    assert isinstance(restored.__context__, KeyError)
    assert str(restored.__context__) == "'context'"
    assert restored.__suppress_context__ is False


def test_remote_ray_exception_type_resolution_failure_is_propagated(monkeypatch):
    def fail_import(_module_name):
        raise RuntimeError("provider import exploded")

    monkeypatch.setattr(ray_errors.importlib, "import_module", fail_import)
    transported = RemoteRayException(
        "remote failure",
        {
            "module": "provider.module",
            "qualname": "ProviderError",
            "message": "remote failure",
            "traceback": "",
            "constructor": {"args": ("remote failure",), "state": None},
            "cause": None,
            "context": None,
            "suppress_context": False,
        },
    )

    with pytest.raises(RuntimeError, match="provider import exploded"):
        transported.restore()


def test_remote_ray_exception_rejects_old_payload_schema():
    with pytest.raises(KeyError):
        RemoteRayException(
            "remote failure",
            {
                "module": "builtins",
                "qualname": "RuntimeError",
                "message": "remote failure",
                "traceback": "",
                "cause": None,
            },
        )


def test_remote_ray_exception_rejects_cyclic_exception_chain():
    payload = {
        "module": "builtins",
        "qualname": "RuntimeError",
        "message": "cycle",
        "traceback": "",
        "constructor": {"args": ("cycle",), "state": None},
        "cause": None,
        "context": None,
        "suppress_context": True,
    }
    payload["cause"] = payload

    with pytest.raises(ValueError, match="remote exception chain contains a cycle"):
        RemoteRayException("cycle", payload)


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


def test_safe_get_preserves_restored_implicit_context():
    try:
        try:
            raise KeyError("context")
        except KeyError:
            raise RuntimeError("outer")
    except RuntimeError as original:
        transported = RemoteRayException.from_exception(original)

    class FakeRayTaskError(RuntimeError):
        def __init__(self, cause: BaseException) -> None:
            self.cause = cause
            super().__init__(cause)

    class FailedFuture:
        def result(self, timeout=None):
            raise FakeRayTaskError(transported)

    class FailedRef:
        def future(self):
            return FailedFuture()

    with pytest.raises(RuntimeError) as exc_info:
        resolve_object_refs_blocking(FailedRef())

    assert exc_info.value.__cause__ is None
    assert isinstance(exc_info.value.__context__, KeyError)
    assert str(exc_info.value.__context__) == "'context'"


def test_real_ray_preserves_implicit_context_and_constructor_state(ray_local):
    import ray

    @ray.remote
    def fail_with_implicit_context():
        from duckdb._ray_errors import RemoteRayException as RemoteCarrier

        try:
            try:
                raise KeyError("context")
            except KeyError:
                raise RuntimeError("outer")
        except RuntimeError as original:
            transported = RemoteCarrier.from_exception(original)
        raise transported

    @ray.remote
    def fail_with_structured_exception():
        from duckdb._ray_errors import RemoteRayException as RemoteCarrier

        original = FileNotFoundError(2, "No such file", "/tmp/input.parquet")
        raise RemoteCarrier.from_exception(original)

    with pytest.raises(RuntimeError) as context_info:
        resolve_object_refs_blocking(fail_with_implicit_context.remote())
    assert context_info.value.__cause__ is None
    assert isinstance(context_info.value.__context__, KeyError)
    assert str(context_info.value.__context__) == "'context'"

    with pytest.raises(FileNotFoundError) as structured_info:
        resolve_object_refs_blocking(fail_with_structured_exception.remote())
    assert structured_info.value.errno == 2
    assert structured_info.value.filename == "/tmp/input.parquet"


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
