# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the unified UDF executor lifecycle."""

from __future__ import annotations

import asyncio
import gc
import struct
import sys
import threading
import time
import types
from collections import deque
from concurrent.futures import Future

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa


def test_subprocess_actor_warmup_attribute_error_fails_startup():
    import duckdb.execution.udf_subprocess as subprocess_exec

    class BrokenWarmup:
        def warm_up(self):
            raise AttributeError("warmup exploded")

        def __call__(self, table):
            return table

    payload = _subprocess_map_payload(
        BrokenWarmup,
        execution_backend="subprocess_actor",
        actor_number=1,
    )
    with pytest.raises(RuntimeError, match="warmup exploded"):
        subprocess_exec._SingleSubprocessExecutor(payload)


def test_udf_resource_and_layout_validation_has_no_silent_fallbacks():
    from duckdb.execution._udf_runtime import UDFExecutor as RuntimeUDFExecutor
    from duckdb.execution.udf_ray_config import payload_num_cpus, payload_num_gpus, stream_output_enabled
    from duckdb.execution.udf_ray_env import normalize_actor_node_ids
    from duckdb.execution.udf_threading import payload_cpu_thread_count

    with pytest.raises(ValueError, match="cpus"):
        payload_num_cpus({"cpus": -1})
    with pytest.raises(ValueError, match="gpus"):
        payload_num_gpus({"gpus": -1})
    with pytest.raises(ValueError, match="cpus"):
        payload_num_cpus({"cpus": float("nan")})
    with pytest.raises(ValueError, match="gpus"):
        payload_num_gpus({"gpus": True})
    with pytest.raises(ValueError, match="stream_output"):
        stream_output_enabled({"stream_output": "true"})
    with pytest.raises(ValueError, match="cpus"):
        payload_cpu_thread_count({"cpus": "invalid"})
    with pytest.raises(ValueError, match="actor node IDs"):
        normalize_actor_node_ids(["node-a"], expected_count=2)

    def identity(table):
        return table

    runtime = RuntimeUDFExecutor(
        {
            "function_pickle": _pickle_function(identity),
            "call_mode": "map_batches",
            "execution_backend": "subprocess_task",
            "input_names": ["a", "b"],
        }
    )
    with pytest.raises(ValueError, match="input_names"):
        runtime.submit(pa.table({"a": [1]}))


def test_subprocess_backend_rejects_unreserved_gpu_request():
    from duckdb.execution.udf import build_executor

    def identity(table):
        return table

    with pytest.raises(ValueError, match="GPU resources require a Ray UDF backend"):
        build_executor(
            _subprocess_map_payload(
                identity,
                gpus=1.0,
            )
        )


def _pickle_function(fn):
    cloudpickle = pytest.importorskip("cloudpickle")
    return cloudpickle.dumps(fn)


def _unsupported_local_map_payload(fn, **extra):
    payload = {
        "function_pickle": _pickle_function(fn),
        "call_mode": "map_batches",
        "execution_backend": "local",
    }
    payload.update(extra)
    return payload


def _subprocess_map_payload(fn, **extra):
    payload = {
        "function_pickle": _pickle_function(fn),
        "call_mode": "map_batches",
        "execution_backend": "subprocess_task",
        "udf_worker_slots": 1,
    }
    payload.update(extra)
    return payload


def _wait_for_results(executor, count: int, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    results = []
    wakeup_event = threading.Event()
    previous_wakeup = getattr(executor, "_wakeup", None)

    def notify_test_waiter():
        if previous_wakeup is not None:
            previous_wakeup()
        wakeup_event.set()

    if hasattr(executor, "register_wakeup"):
        executor.register_wakeup(notify_test_waiter)
    try:
        while len(results) < count:
            item = executor.take_ready_result()
            if item is not None:
                results.append(item)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wakeup_event.wait(timeout=remaining)
            wakeup_event.clear()
    finally:
        if hasattr(executor, "register_wakeup"):
            executor.register_wakeup(previous_wakeup)
    return results


def _submit_with_admission(executor, table, *, submit_id=None, retained_input_bytes=None):
    retained = int(table.nbytes if retained_input_bytes is None else retained_input_bytes)
    assert executor.request_task_admission(retained) is True
    state = executor.task_admission_state()
    assert state["state"] == "ready"
    assert state["retained_input_bytes"] == retained
    if submit_id is None:
        return executor.submit(table)
    return executor.submit_with_id(submit_id, table)


def _submit_ref_bundle_with_admission(
    executor,
    submit_id,
    refs,
    slices,
    metadata,
    names,
    *,
    retained_input_bytes=0,
):
    retained = int(retained_input_bytes)
    assert executor.request_task_admission(retained) is True
    state = executor.task_admission_state()
    assert state["state"] == "ready"
    assert state["retained_input_bytes"] == retained
    return executor.submit_ref_bundle_with_id(submit_id, refs, slices, metadata, names)


def _wait_for_runtime_stats(runtime, predicate, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    best_stats = runtime.stats()
    while True:
        stats = runtime.stats()
        best_stats = stats
        if predicate(stats):
            return stats
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return best_stats
        with runtime.cond:
            runtime.cond.wait(timeout=remaining)


def _wait_for_executor_stats(executor, predicate, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    best_stats = executor.stats()
    while True:
        stats = executor.stats()
        best_stats = stats
        if predicate(stats):
            return stats
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return best_stats
        cv = getattr(executor, "_task_futures_cv", None)
        if cv is None:
            threading.Event().wait(timeout=remaining)
            continue
        with cv:
            cv.wait(timeout=remaining)


def _wait_for_wakeup(callback_owner, trigger, timeout_s: float = 5.0):
    wakeup_event = threading.Event()
    previous_wakeup = getattr(callback_owner, "_wakeup", None)

    def notify_test_waiter():
        if previous_wakeup is not None:
            previous_wakeup()
        wakeup_event.set()

    callback_owner.register_wakeup(notify_test_waiter)
    try:
        trigger()
        return wakeup_event.wait(timeout=timeout_s)
    finally:
        callback_owner.register_wakeup(previous_wakeup)


def _subprocess_scalar_payload(fn, **extra):
    payload = {
        "function_pickle": _pickle_function(fn),
        "call_mode": "map",
        "execution_backend": "subprocess_task",
        "udf_worker_slots": 1,
    }
    payload.update(extra)
    return payload


def _make_subprocess_actor_executor(
    subprocess_exec,
    payload,
    *,
    pool_size=None,
    name="test-local-subprocess-actor",
):
    size = int(pool_size or payload.get("actor_number") or payload.get("udf_worker_slots") or 1)
    pool = subprocess_exec.LocalSubprocessActorPool(payload, size, name=name)
    try:
        executor = subprocess_exec.UDFExecutor(payload, options={"local_actor_pool": pool})
    except Exception:
        pool.shutdown(kill=True)
        raise
    return executor, pool


def test_ray_get_uses_query_deadline_timeout(monkeypatch):
    from duckdb.runners.ray import safe_get

    class FakeFuture:
        def __init__(self):
            self.calls = []

        def result(self, timeout=None):
            self.calls.append(timeout)
            return "resolved"

    class FakeRef:
        def __init__(self, future):
            self._future = future

        def future(self):
            return self._future

    future = FakeFuture()
    monkeypatch.setattr(safe_get.time, "time", lambda: 100.0)
    monkeypatch.setenv("VANE_QUERY_DEADLINE_EPOCH_S", "101.0")

    assert safe_get.resolve_object_refs_blocking(FakeRef(future), timeout=300.0) == "resolved"
    assert future.calls == [pytest.approx(1.0)]


@pytest.mark.parametrize(
    "raw",
    ["nan", "inf", "-inf", "-1", "invalid"],
)
@pytest.mark.parametrize(
    "env_name",
    [
        "VANE_QUERY_DEADLINE_EPOCH_S",
        "VANE_RAY_OBJECT_GET_TIMEOUT_S",
        "VANE_RAY_ACTOR_INIT_TIMEOUT_S",
        "VANE_UDF_SUBPROCESS_CONTROL_TIMEOUT_S",
        "VANE_UDF_SUBPROCESS_SHUTDOWN_GRACE_S",
        "VANE_UDF_STREAM_SHUTDOWN_TIMEOUT_S",
    ],
)
def test_timeout_env_parsers_reject_non_finite_negative_and_invalid(monkeypatch, env_name, raw):
    from duckdb.execution import udf_ray_actor_pool, udf_stream_result_collector, udf_subprocess
    from duckdb.runners.ray import safe_get

    for name in (
        "VANE_QUERY_DEADLINE_EPOCH_S",
        "VANE_RAY_OBJECT_GET_TIMEOUT_S",
        "VANE_RAY_ACTOR_INIT_TIMEOUT_S",
        "VANE_UDF_SUBPROCESS_CONTROL_TIMEOUT_S",
        "VANE_UDF_SUBPROCESS_SHUTDOWN_GRACE_S",
        "VANE_UDF_STREAM_SHUTDOWN_TIMEOUT_S",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(env_name, raw)

    with pytest.raises(ValueError):
        if env_name in {"VANE_QUERY_DEADLINE_EPOCH_S", "VANE_RAY_OBJECT_GET_TIMEOUT_S"}:
            safe_get.configured_ray_get_timeout_s()
        elif env_name == "VANE_RAY_ACTOR_INIT_TIMEOUT_S":
            udf_ray_actor_pool._actor_init_timeout_s()
        elif env_name == "VANE_UDF_SUBPROCESS_CONTROL_TIMEOUT_S":
            udf_subprocess._subprocess_control_timeout_s()
        elif env_name == "VANE_UDF_SUBPROCESS_SHUTDOWN_GRACE_S":
            udf_subprocess._subprocess_shutdown_grace_s()
        elif env_name == "VANE_UDF_STREAM_SHUTDOWN_TIMEOUT_S":
            udf_stream_result_collector.AsyncResultCollector(ray_module=object())


@pytest.mark.parametrize(
    "env_name",
    [
        "VANE_RAY_ACTOR_INIT_TIMEOUT_S",
        "VANE_UDF_SUBPROCESS_CONTROL_TIMEOUT_S",
        "VANE_UDF_SUBPROCESS_SHUTDOWN_GRACE_S",
        "VANE_UDF_STREAM_SHUTDOWN_TIMEOUT_S",
    ],
)
def test_positive_timeout_env_parsers_reject_zero(monkeypatch, env_name):
    from duckdb.execution import udf_ray_actor_pool, udf_stream_result_collector, udf_subprocess

    for name in (
        "VANE_QUERY_DEADLINE_EPOCH_S",
        "VANE_RAY_OBJECT_GET_TIMEOUT_S",
        "VANE_RAY_ACTOR_INIT_TIMEOUT_S",
        "VANE_UDF_SUBPROCESS_CONTROL_TIMEOUT_S",
        "VANE_UDF_SUBPROCESS_SHUTDOWN_GRACE_S",
        "VANE_UDF_STREAM_SHUTDOWN_TIMEOUT_S",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(env_name, "0")

    with pytest.raises(ValueError):
        if env_name == "VANE_RAY_ACTOR_INIT_TIMEOUT_S":
            udf_ray_actor_pool._actor_init_timeout_s()
        elif env_name == "VANE_UDF_SUBPROCESS_CONTROL_TIMEOUT_S":
            udf_subprocess._subprocess_control_timeout_s()
        elif env_name == "VANE_UDF_SUBPROCESS_SHUTDOWN_GRACE_S":
            udf_subprocess._subprocess_shutdown_grace_s()
        elif env_name == "VANE_UDF_STREAM_SHUTDOWN_TIMEOUT_S":
            udf_stream_result_collector.AsyncResultCollector(ray_module=object())


@pytest.mark.parametrize(
    "env_name",
    [
        "VANE_QUERY_DEADLINE_EPOCH_S",
        "VANE_RAY_OBJECT_GET_TIMEOUT_S",
    ],
)
def test_ray_safe_get_timeout_envs_preserve_zero(monkeypatch, env_name):
    from duckdb.runners.ray import safe_get

    monkeypatch.delenv("VANE_QUERY_DEADLINE_EPOCH_S", raising=False)
    monkeypatch.delenv("VANE_RAY_OBJECT_GET_TIMEOUT_S", raising=False)
    monkeypatch.setenv(env_name, "0")

    if env_name == "VANE_QUERY_DEADLINE_EPOCH_S":
        with pytest.raises(safe_get.QueryDeadlineExceeded):
            safe_get.configured_ray_get_timeout_s()
    else:
        assert safe_get.configured_ray_get_timeout_s() == 0.0


def test_ray_get_in_async_actor_background_thread_uses_object_ref_future(monkeypatch):
    from duckdb.runners.ray import safe_get

    class FakeFuture:
        def __init__(self):
            self.calls = []

        def result(self, timeout=None):
            self.calls.append(timeout)
            return "resolved"

    class FakeRef:
        def __init__(self, future):
            self._future = future

        def future(self):
            return self._future

    future = FakeFuture()

    assert safe_get.resolve_object_refs_blocking(FakeRef(future)) == "resolved"
    assert future.calls == [None]


def test_ray_get_heartbeat_runs_between_bounded_waits(monkeypatch):
    from duckdb.runners.ray import safe_get

    class FakeFuture:
        def __init__(self):
            self.calls = []
            self.remaining_timeouts = 2

        def result(self, timeout=None):
            self.calls.append(timeout)
            if self.remaining_timeouts:
                self.remaining_timeouts -= 1
                raise TimeoutError
            return "resolved"

        def done(self):
            return False

    class FakeRef:
        def __init__(self, future):
            self._future = future

        def future(self):
            return self._future

    monkeypatch.delenv("VANE_QUERY_DEADLINE_EPOCH_S", raising=False)
    monkeypatch.delenv("VANE_RAY_OBJECT_GET_TIMEOUT_S", raising=False)
    heartbeats = []
    future = FakeFuture()

    result = safe_get.resolve_object_refs_blocking(
        FakeRef(future),
        on_wait=lambda: heartbeats.append("tick"),
        wait_interval_s=0.25,
    )

    assert result == "resolved"
    assert future.calls == [0.25, 0.25, 0.25]
    assert heartbeats == ["tick", "tick"]


def test_ray_get_heartbeat_preserves_one_total_timeout(monkeypatch):
    from duckdb.runners.ray import safe_get

    clock = [10.0]

    class FakeFuture:
        def __init__(self):
            self.calls = []

        def result(self, timeout=None):
            self.calls.append(timeout)
            clock[0] += float(timeout)
            raise TimeoutError

        def done(self):
            return False

    class FakeRef:
        def __init__(self, future):
            self._future = future

        def future(self):
            return self._future

    monkeypatch.delenv("VANE_QUERY_DEADLINE_EPOCH_S", raising=False)
    monkeypatch.delenv("VANE_RAY_OBJECT_GET_TIMEOUT_S", raising=False)
    monkeypatch.setattr(safe_get.time, "monotonic", lambda: clock[0])
    heartbeats = []
    future = FakeFuture()

    with pytest.raises(TimeoutError):
        safe_get.resolve_object_refs_blocking(
            FakeRef(future),
            timeout=1.0,
            on_wait=lambda: heartbeats.append("tick"),
            wait_interval_s=0.4,
        )

    assert future.calls == [pytest.approx(0.4), pytest.approx(0.4), pytest.approx(0.2)]
    assert heartbeats == ["tick", "tick"]


def test_ray_get_in_async_actor_event_loop_rejects_sync_wait(monkeypatch):
    from duckdb.runners.ray import safe_get

    class AwaitableRef:
        def __await__(self):
            async def _resolve():
                return "resolved"

            return _resolve().__await__()

    async def _invoke():
        with pytest.raises(RuntimeError, match="cannot run on an event loop"):
            safe_get.resolve_object_refs_blocking(AwaitableRef())

    asyncio.run(_invoke())


def test_udf_actor_pool_init_timeout_kills_owned_actors(monkeypatch):
    import duckdb.execution.udf_ray_actor_pool as actor_pool_mod

    class FakeRay:
        def __init__(self):
            self.future_calls = []
            self.killed = []

        def kill(self, actor, **kwargs):
            self.killed.append((actor, kwargs))

    fake_ray = FakeRay()

    class FakeRef:
        def __init__(self, name):
            self.name = name

        def future(self):
            ref = self

            class _Future:
                def result(self, timeout=None):
                    fake_ray.future_calls.append((ref, timeout))
                    raise TimeoutError("init did not finish")

            return _Future()

    init_refs = [FakeRef("init-0"), FakeRef("init-1")]
    actors_obj = types.SimpleNamespace(
        actors=["actor-0", "actor-1"],
        _init_refs=init_refs,
        _confirmed_ready=set(),
        _owns_actors=True,
    )
    monkeypatch.setenv("VANE_RAY_ACTOR_INIT_TIMEOUT_S", "0.25")

    with pytest.raises(RuntimeError, match="UDF actor pool initialization timed out"):
        actor_pool_mod._resolve_actor_pool_init_refs(fake_ray, actors_obj)

    assert len(fake_ray.future_calls) == 1
    assert fake_ray.future_calls[0][0] is init_refs[0]
    assert 0.0 < fake_ray.future_calls[0][1] <= 0.25
    assert fake_ray.killed == [
        ("actor-0", {"no_restart": True}),
        ("actor-1", {"no_restart": True}),
    ]
    assert actors_obj.actors == []


def test_udf_actor_pool_init_failure_preserves_root_cause():
    import duckdb.execution.udf_ray_actor_pool as actor_pool_mod

    class FakeRay:
        def kill(self, _actor, **_kwargs):
            return None

    class FakeRef:
        def future(self):
            class _Future:
                def result(self, timeout=None):
                    raise TypeError(f"invalid init ObjectRef, timeout={timeout}")

            return _Future()

    actors_obj = types.SimpleNamespace(
        actors=["actor-0"],
        _init_refs=[FakeRef()],
        _confirmed_ready=set(),
        _owns_actors=True,
    )

    with pytest.raises(
        RuntimeError,
        match=r"UDF actor pool initialization failed: TypeError: invalid init ObjectRef",
    ):
        actor_pool_mod._resolve_actor_pool_init_refs(FakeRay(), actors_obj)


def test_local_vllm_submit_fails_fast_when_engine_init_deadline_expires():
    import duckdb.execution.vllm as vllm

    class FakeReady:
        def __init__(self):
            self.wait_calls = []

        def is_set(self):
            return False

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            return False

    ready = FakeReady()
    executor = vllm.LocalVLLMExecutor.__new__(vllm.LocalVLLMExecutor)
    executor.use_threading = True
    executor.engine_ready = ready
    executor.engine_error_message = None
    executor.engine_init_timeout_s = 0.125
    executor.on_error = "raise"

    with pytest.raises(RuntimeError, match="vllm engine init did not finish"):
        executor.submit(None, ["prompt"], pa.table({"x": [1]}))

    assert ready.wait_calls == [pytest.approx(0.125)]


def test_vllm_remote_submit_failure_rolls_back_inflight(monkeypatch):
    import duckdb.execution.vllm as vllm

    class FakeRemoteMethod:
        def __init__(self, result=None, exc: Exception | None = None):
            self.result = result
            self.exc = exc

        def remote(self, *_args, **_kwargs):
            if self.exc is not None:
                raise self.exc
            return self.result

    class FakeRouter:
        report_start = FakeRemoteMethod("router-start")
        report_completion = FakeRemoteMethod("router-complete")

    class FakeActor:
        wait_for_result = FakeRemoteMethod("actor-wait")
        take_ready_result = FakeRemoteMethod("actor-ready")
        submit_async = FakeRemoteMethod(exc=RuntimeError("submit failed"))
        finished_executor = FakeRemoteMethod("actor-finish-executor")
        finished_submitting = FakeRemoteMethod("actor-finish")

    class FakeLLMActors:
        router_actor = FakeRouter()
        llm_actors = [FakeActor()]

    monkeypatch.setattr(vllm, "resolve_object_refs_blocking", lambda ref: ref)

    pool_name = "submit-failure-rollback"
    vllm._shared_inflight.pop(pool_name, None)
    executor = vllm.RemoteVLLMExecutor(FakeLLMActors(), pool_name=pool_name)

    rows = pa.table({"x": [1, 2, 3]})
    with pytest.raises(RuntimeError, match="submit failed"):
        executor.submit(None, ["a", "b", "c"], rows)

    assert executor._inflight_per_actor == [0]
    assert executor._submit_per_actor == [0]


def test_vllm_remote_submit_async_ref_failure_becomes_executor_error(monkeypatch):
    import duckdb.execution.vllm as vllm

    class FakeRef:
        def __init__(self, value=None, exc: Exception | None = None):
            self.value = value
            self.exc = exc

        def future(self):
            return self

        def add_done_callback(self, callback):
            callback(self)

    class FakeRemoteMethod:
        def __init__(self, value=None, exc: Exception | None = None):
            self.value = value
            self.exc = exc

        def remote(self, *_args, **_kwargs):
            return FakeRef(self.value, self.exc)

    class FakeRouter:
        report_start = FakeRemoteMethod("router-start")
        report_completion = FakeRemoteMethod("router-complete")

    class FakeActor:
        wait_for_result = FakeRemoteMethod(False)
        take_ready_result = FakeRemoteMethod(None)
        submit_async = FakeRemoteMethod(exc=RuntimeError("actor submit failed after dispatch"))
        finished_executor = FakeRemoteMethod("actor-finish-executor")
        finished_submitting = FakeRemoteMethod("actor-finish")

    class FakeLLMActors:
        router_actor = FakeRouter()
        llm_actors = [FakeActor()]

    def fake_get(ref):
        if ref.exc is not None:
            raise ref.exc
        return ref.value

    monkeypatch.setattr(vllm, "resolve_object_refs_blocking", fake_get)

    pool_name = "submit-async-ref-failure"
    vllm._shared_inflight.pop(pool_name, None)
    executor = vllm.RemoteVLLMExecutor(FakeLLMActors(), pool_name=pool_name)

    executor.submit(None, ["a", "b"], pa.table({"x": [1, 2]}))

    with pytest.raises(RuntimeError, match="actor submit failed after dispatch"):
        executor.take_ready_result()
    assert executor._inflight_per_actor == [0]
    assert executor._submit_per_actor == [0]


def test_vllm_remote_observes_router_lifecycle_refs(monkeypatch):
    import duckdb.execution.vllm as vllm

    observed: list[str] = []

    class FakeRef:
        def __init__(self, name: str):
            self.name = name

    class FakeRemoteMethod:
        def __init__(self, name: str):
            self.name = name

        def remote(self, *_args, **_kwargs):
            return FakeRef(self.name)

    class FakeRouter:
        report_start = FakeRemoteMethod("router-start")
        report_completion = FakeRemoteMethod("router-complete")

    class FakeActor:
        wait_for_result = FakeRemoteMethod("actor-wait")
        take_ready_result = FakeRemoteMethod("actor-ready")
        submit_async = FakeRemoteMethod("actor-submit")
        finished_executor = FakeRemoteMethod("actor-finish-executor")
        finished_submitting = FakeRemoteMethod("actor-finish")

    class FakeLLMActors:
        router_actor = FakeRouter()
        llm_actors = [FakeActor()]

    def fake_get(ref):
        observed.append(ref.name)
        return None

    monkeypatch.setattr(vllm, "resolve_object_refs_blocking", fake_get)

    executor = vllm.RemoteVLLMExecutor(FakeLLMActors())
    assert "router-start" in observed

    executor.finished_submitting()
    assert "router-complete" in observed


def test_vllm_remote_shutdown_reports_completion_once_and_clears_wait_refs(monkeypatch):
    import duckdb.execution.vllm as vllm

    observed: list[str] = []

    class FakeRef:
        def __init__(self, name: str):
            self.name = name

    class FakeRemoteMethod:
        def __init__(self, name: str):
            self.name = name

        def remote(self, *_args, **_kwargs):
            return FakeRef(self.name)

    class FakeRouter:
        report_start = FakeRemoteMethod("router-start")
        report_completion = FakeRemoteMethod("router-complete")

    class FakeActor:
        wait_for_result = FakeRemoteMethod("actor-wait")
        take_ready_result = FakeRemoteMethod("actor-ready")
        submit_async = FakeRemoteMethod("actor-submit")
        finished_executor = FakeRemoteMethod("actor-finish-executor")
        finished_submitting = FakeRemoteMethod("actor-finish")

    class FakeLLMActors:
        router_actor = FakeRouter()
        llm_actors = [FakeActor(), FakeActor()]

    def fake_get(ref):
        observed.append(ref.name)
        return None

    monkeypatch.setattr(vllm, "resolve_object_refs_blocking", fake_get)

    executor = vllm.RemoteVLLMExecutor(FakeLLMActors())
    executor._wait_refs_by_actor = [FakeRef("pending-0"), FakeRef("pending-1")]
    executor.shutdown()
    executor.shutdown()

    assert observed == ["router-start", "actor-finish-executor", "actor-finish-executor", "router-complete"]
    assert executor._wait_refs_by_actor == [None, None]


def test_vllm_actor_wait_for_result_finishes_per_executor_before_global_finish():
    import duckdb.execution.vllm as vllm

    executor = vllm.RayLocalVLLMExecutor.__new__(vllm.RayLocalVLLMExecutor)
    executor.completed_tasks = deque()
    executor.error_message = None
    executor._finished_submitting = False
    executor.running_task_count = 1
    executor._result_cv = threading.Condition(threading.Lock())
    executor._per_executor_deques = {"exec-a": deque(), "exec-b": deque()}
    executor._per_executor_running_task_count = {"exec-a": 0, "exec-b": 1}
    executor._per_executor_finished = set()

    executor.finished_executor("exec-a")

    import asyncio

    assert asyncio.run(executor.wait_for_result("exec-a")) is False


def test_vllm_remote_wait_for_result_drains_ready_actor(monkeypatch):
    import duckdb.execution.vllm as vllm

    rows = pa.table({"x": [1, 2]})
    wait_ref = None
    wait_callbacks = []
    submit_callbacks = []

    class FakeRef:
        def __init__(self, value, kind="submit"):
            self.value = value
            self.kind = kind

        def future(self):
            return self

        def add_done_callback(self, callback):
            if self.kind == "wait":
                wait_callbacks.append(self)
            else:
                submit_callbacks.append(self)
            callback(self)

    class FakeRemoteMethod:
        def __init__(self, value=None):
            self.value = value

        def remote(self, *_args, **_kwargs):
            return FakeRef(self.value)

    class FakeWaitMethod:
        def remote(self, _executor_id):
            nonlocal wait_ref
            wait_ref = FakeRef(True, kind="wait")
            return wait_ref

    class FakeReadyMethod:
        def remote(self, _executor_id):
            return FakeRef((["out-a", "out-b"], rows))

    class FakeRouter:
        report_start = FakeRemoteMethod()
        report_completion = FakeRemoteMethod()

    class FakeActor:
        wait_for_result = FakeWaitMethod()
        take_ready_result = FakeReadyMethod()
        submit_async = FakeRemoteMethod()
        finished_executor = FakeRemoteMethod()
        finished_submitting = FakeRemoteMethod()

    class FakeLLMActors:
        router_actor = FakeRouter()
        llm_actors = [FakeActor()]

    monkeypatch.setattr(vllm, "resolve_object_refs_blocking", lambda ref: ref.value)

    pool_name = "wait-ref-drain"
    vllm._shared_inflight.pop(pool_name, None)
    executor = vllm.RemoteVLLMExecutor(FakeLLMActors(), pool_name=pool_name)

    executor.submit(None, ["a", "b"], rows)
    executor.wait_for_result()

    assert len(submit_callbacks) == 1
    assert wait_callbacks == [wait_ref]
    assert executor._inflight_per_actor == [0]
    assert executor._results_per_actor == [2]
    result = executor.take_ready_result()
    assert result is not None
    outputs, output_rows = result
    assert outputs == ["out-a", "out-b"]
    assert output_rows is rows


def test_vllm_remote_wait_ref_stays_armed_until_callback_records_result():
    import duckdb.execution.vllm as vllm

    class FakeRef:
        pass

    ref = FakeRef()
    executor = vllm.RemoteVLLMExecutor.__new__(vllm.RemoteVLLMExecutor)
    executor._result_cv = threading.Condition(threading.Lock())
    executor._shutdown_called = False
    executor._finished = False
    executor._wait_refs_by_actor = [ref]

    assert executor._take_ready_wait_ref(ref) == 0
    assert executor._wait_refs_by_actor == [ref]


def test_vllm_remote_wait_deadline_cancels_outstanding_refs(monkeypatch):
    import duckdb.execution.vllm as vllm

    cancelled = []

    class FakeRay(types.ModuleType):
        def __init__(self):
            super().__init__("ray")

        def cancel(self, ref):
            cancelled.append(ref)

    class FakeCondition:
        def __init__(self):
            self.timeout = None
            self.notified = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def wait_for(self, _predicate, timeout=None):
            self.timeout = timeout
            return False

        def notify_all(self):
            self.notified = True

    submit_ref = object()
    wait_ref = object()
    condition = FakeCondition()
    executor = vllm.RemoteVLLMExecutor.__new__(vllm.RemoteVLLMExecutor)
    executor.llm_actors = [object()]
    executor._result_cv = condition
    executor._finished = False
    executor._finished_submitting_flag = False
    executor._submit_per_actor = [1]
    executor._results_per_actor = [0]
    executor._wait_refs_by_actor = [wait_ref]
    executor._submit_refs = {submit_ref: (0, 1)}
    executor._error_message = None
    executor._result_buffer = deque()
    executor._inflight_lock = threading.Lock()
    executor._inflight_per_actor = [1]
    executor._shutdown_called = False
    executor._released_outstanding_inflight = False

    monkeypatch.setitem(sys.modules, "ray", FakeRay())
    monkeypatch.setattr(vllm, "configured_ray_get_timeout_s", lambda: 0.125)

    with pytest.raises(RuntimeError, match="vllm remote task failed: RuntimeError: vllm remote wait exceeded"):
        executor.wait_for_result()

    assert condition.timeout == pytest.approx(0.125)
    assert cancelled == [submit_ref, wait_ref]
    assert executor._inflight_per_actor == [0]
    assert executor._submit_refs == {}
    assert executor._wait_refs_by_actor == [None]


def test_vllm_remote_wait_without_pending_ref_before_completion_is_error():
    import duckdb.execution.vllm as vllm

    executor = vllm.RemoteVLLMExecutor.__new__(vllm.RemoteVLLMExecutor)
    executor.llm_actors = []
    executor._result_cv = threading.Condition(threading.Lock())
    executor._finished = False
    executor._finished_submitting_flag = True
    executor._submit_per_actor = [1]
    executor._results_per_actor = [0]
    executor._wait_refs_by_actor = [None]
    executor._submit_refs = {}
    executor._error_message = None
    executor._result_buffer = deque()
    executor._inflight_lock = threading.Lock()
    executor._inflight_per_actor = [1]
    executor._shutdown_called = False
    executor._released_outstanding_inflight = False
    executor._ensure_remote_wait_refs = lambda: None

    with pytest.raises(RuntimeError, match="no pending actor wait refs"):
        executor.wait_for_result()
    assert executor._finished is True
    assert executor._inflight_per_actor == [0]


def test_vllm_remote_actor_without_results_after_executor_finish_is_error(monkeypatch):
    import duckdb.execution.vllm as vllm

    class FakeRef:
        def __init__(self, value):
            self.value = value

        def future(self):
            return self

        def add_done_callback(self, callback):
            callback(self)

    class FakeWaitMethod:
        def remote(self, _executor_id):
            return FakeRef(False)

    class FakeReadyMethod:
        def remote(self, _executor_id):
            raise AssertionError("take_ready_result should not be called when wait_for_result returns false")

    class FakeActor:
        wait_for_result = FakeWaitMethod()
        take_ready_result = FakeReadyMethod()

    def fake_get(ref):
        return ref.value

    monkeypatch.setattr(vllm, "resolve_object_refs_blocking", fake_get)

    executor = vllm.RemoteVLLMExecutor.__new__(vllm.RemoteVLLMExecutor)
    executor.llm_actors = [FakeActor()]
    executor._executor_id = "exec-a"
    executor._result_cv = threading.Condition(threading.Lock())
    executor._finished = False
    executor._finished_submitting_flag = True
    executor._submit_per_actor = [1]
    executor._results_per_actor = [0]
    executor._wait_refs_by_actor = [None]
    executor._submit_refs = {}
    executor._error_message = None
    executor._result_buffer = deque()
    executor._inflight_lock = threading.Lock()
    executor._inflight_per_actor = [0]
    executor._shutdown_called = False

    with pytest.raises(RuntimeError, match="finished without returning all submitted results"):
        executor.wait_for_result()

    assert executor._error_message is not None
    assert executor._finished is True


def test_vllm_remote_actor_missing_results_rolls_back_shared_inflight(monkeypatch):
    import duckdb.execution.vllm as vllm

    class FakeRef:
        def __init__(self, value):
            self.value = value

        def future(self):
            return self

        def add_done_callback(self, callback):
            callback(self)

    class FakeWaitMethod:
        def remote(self, _executor_id):
            return FakeRef(False)

    class FakeReadyMethod:
        def remote(self, _executor_id):
            raise AssertionError("take_ready_result should not be called when wait_for_result returns false")

    class FakeActor:
        wait_for_result = FakeWaitMethod()
        take_ready_result = FakeReadyMethod()

    monkeypatch.setattr(vllm, "resolve_object_refs_blocking", lambda ref: ref.value)

    executor = vllm.RemoteVLLMExecutor.__new__(vllm.RemoteVLLMExecutor)
    executor.llm_actors = [FakeActor()]
    executor._executor_id = "exec-a"
    executor._result_cv = threading.Condition(threading.Lock())
    executor._finished = False
    executor._finished_submitting_flag = True
    executor._submit_per_actor = [3]
    executor._results_per_actor = [0]
    executor._wait_refs_by_actor = [None]
    executor._submit_refs = {}
    executor._error_message = None
    executor._result_buffer = deque()
    executor._inflight_lock = threading.Lock()
    executor._inflight_per_actor = [3]
    executor._shutdown_called = False

    with pytest.raises(RuntimeError, match="finished without returning all submitted results"):
        executor.wait_for_result()

    assert executor._inflight_per_actor == [0]


def test_vllm_router_waits_for_actor_finished_refs(monkeypatch):
    import duckdb.execution.vllm as vllm

    observed: list[str] = []

    class FakeRef:
        def __init__(self, name: str):
            self.name = name

    class FakeFinishedMethod:
        def __init__(self, name: str):
            self.name = name

        def remote(self):
            return FakeRef(self.name)

    class FakeActor:
        def __init__(self, name: str):
            self.finished_submitting = FakeFinishedMethod(name)

    def fake_get(ref):
        observed.append(ref.name)
        return None

    monkeypatch.setattr(vllm, "resolve_object_refs_blocking", fake_get)

    router = vllm.PrefixRouter([FakeActor("finish-0"), FakeActor("finish-1")], 0)
    router.report_start()
    router.report_completion()

    assert observed == ["finish-0", "finish-1"]


def test_vllm_named_actor_pool_partial_lookup_fails_without_creation_fallback(monkeypatch):
    import duckdb.execution.vllm as vllm

    class FakeRay:
        def __init__(self):
            self.lookups: list[str] = []

        def get_actor(self, name):
            self.lookups.append(name)
            if name in {"pool-router", "pool-llm-0"}:
                return f"actor:{name}"
            raise ValueError(name)

    def fail_create(*_args, **_kwargs):
        raise AssertionError("partial named vLLM pool must not fall back to actor creation")

    monkeypatch.setitem(sys.modules, "ray", FakeRay())
    monkeypatch.setattr(vllm.LLMActors, "__init__", fail_create)

    with pytest.raises(RuntimeError, match="partially available|incomplete"):
        vllm.LLMActors.get_or_create_named(
            model="model",
            engine_args={},
            generate_args={},
            on_error="raise",
            gpus_per_actor=1,
            concurrency=2,
            load_balance_threshold=32,
            name_prefix="pool",
        )


def test_vllm_ray_execution_requires_runner_owned_runtime(monkeypatch):
    import ray

    import duckdb.execution.vllm as vllm

    monkeypatch.setattr(ray, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="initialized RayRunner runtime"):
        vllm.build_executor("model", {"use_ray": True})


def test_unified_executor_passes_local_subprocess_actor_pool_option():
    from duckdb.execution.unified_executor import build_unified_executor

    class Identity:
        def __call__(self, table):
            return table

    class FakeLocalActorPool:
        pool_size = 2
        name = "fake-local-actor-pool"

        def __init__(self):
            from duckdb.execution.udf_admission import LocalExecutionSlotPool

            self.admission_slots = LocalExecutionSlotPool(
                max_slots=self.pool_size,
                execution_slot_prefix="fake-local-actor",
            )

        def create_admission_authority(self):
            return self.admission_slots.create_authority()

        def submit(self, *_args, **_kwargs):
            raise AssertionError("submit should not be called")

        def stats(self):
            return {"active_workers": 0, "idle_workers": self.pool_size}

        def first_proc(self):
            return None

        def worker_pids(self):
            return []

        def cancel_output_grants(self):
            return None

    pool = FakeLocalActorPool()
    executor = build_unified_executor(
        _subprocess_map_payload(
            Identity,
            execution_backend="subprocess_actor",
            actor_number=2,
            gpus=0.0,
        ),
        {"local_actor_pool": pool},
    )
    try:
        assert executor._actor_pool is pool
        assert executor.stats()["udf_max_running_tasks"] == 2
    finally:
        executor.close()


def test_subprocess_actor_requires_precreated_local_actor_pool(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec

    class Identity:
        def __call__(self, table):
            return table

    def fail_worker_creation(*_args, **_kwargs):
        raise AssertionError("subprocess_actor must not create workers in executor")

    monkeypatch.setattr(subprocess_exec, "_SingleSubprocessExecutor", fail_worker_creation)

    with pytest.raises(RuntimeError, match="requires a pre-created local_actor_pool"):
        subprocess_exec.UDFExecutor(
            _subprocess_map_payload(
                Identity,
                execution_backend="subprocess_actor",
                actor_number=1,
            )
        )


def test_subprocess_actor_rejects_local_actor_pool_name():
    import duckdb.execution.udf_subprocess as subprocess_exec

    class Identity:
        def __call__(self, table):
            return table

    payload = _subprocess_map_payload(
        Identity,
        execution_backend="subprocess_actor",
        actor_number=1,
        local_actor_pool_name="old-name",
    )

    with pytest.raises(ValueError, match="local_actor_pool_name is unsupported"):
        subprocess_exec.UDFExecutor(payload, options={"local_actor_pool": object()})


def test_subprocess_actor_invalid_local_actor_pool_size_preserves_validation_error():
    import duckdb.execution.udf_subprocess as subprocess_exec

    class Identity:
        def __call__(self, table):
            return table

    class BadPool:
        pool_size = "bad-int"

        def submit(self, *_args, **_kwargs):
            raise AssertionError("submit should not be called")

    payload = _subprocess_map_payload(
        Identity,
        execution_backend="subprocess_actor",
        actor_number=1,
    )

    with pytest.raises(ValueError, match="local_actor_pool.pool_size must be a positive integer"):
        subprocess_exec.UDFExecutor(payload, options={"local_actor_pool": BadPool()})


def test_subprocess_actor_local_actor_pool_requires_full_runtime_contract():
    import duckdb.execution.udf_subprocess as subprocess_exec

    class Identity:
        def __call__(self, table):
            return table

    class MinimalPool:
        pool_size = 1

        def submit(self, *_args, **_kwargs):
            raise AssertionError("submit should not be called")

    payload = _subprocess_map_payload(
        Identity,
        execution_backend="subprocess_actor",
        actor_number=1,
    )

    with pytest.raises(ValueError, match="local_actor_pool must expose"):
        subprocess_exec.UDFExecutor(payload, options={"local_actor_pool": MinimalPool()})


def test_ensure_local_subprocess_actor_pools_for_plan_injects_by_udf_node(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec

    created_args = []

    class FakeLocalActorPool:
        def __init__(self, payload, pool_size, *, name=None):
            self.payload = payload
            self.pool_size = pool_size
            self.name = name
            created_args.append((payload, pool_size, name))

        def shutdown(self, *, kill=False):
            pass

    class FakePlan:
        def __init__(self):
            self.set_calls = []

        def collect_udf_nodes(self, conn=None):
            return [
                {
                    "node_id": 7,
                    "payload": {
                        "execution_backend": "subprocess_actor",
                        "actor_number": 3,
                        "function_pickle": b"unused",
                        "call_mode": "map_batches",
                    },
                },
                {
                    "node_id": 8,
                    "payload": {
                        "execution_backend": "subprocess_task",
                        "udf_worker_slots": 3,
                        "function_pickle": b"unused",
                        "call_mode": "map_batches",
                    },
                },
            ]

        def set_udf_actor_handles(self, handles_map, conn=None):
            self.set_calls.append((handles_map, conn))

    monkeypatch.setattr(subprocess_exec, "LocalSubprocessActorPool", FakeLocalActorPool)
    plan = FakePlan()

    created, handles_map = subprocess_exec.ensure_local_subprocess_actor_pools_for_plan(plan, conn="conn")

    assert len(created) == 1
    assert created_args[0][1] == 3
    assert set(handles_map) == {"7"}
    assert handles_map["7"] == {"local_actor_pool": created[0]}
    assert plan.set_calls == [(handles_map, "conn")]


def test_ensure_local_subprocess_actor_pools_for_nodes_injects_with_callback(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec

    created_args = []
    injected = []

    class FakeLocalActorPool:
        def __init__(self, payload, pool_size, *, name=None):
            self.payload = payload
            self.pool_size = pool_size
            self.name = name
            created_args.append((payload, pool_size, name))

        def shutdown(self, *, kill=False):
            pass

    monkeypatch.setattr(subprocess_exec, "LocalSubprocessActorPool", FakeLocalActorPool)
    nodes = [
        {
            "node_id": 4,
            "payload": {
                "execution_backend": "subprocess_actor",
                "actor_number": 2,
                "function_pickle": b"unused",
                "call_mode": "map_batches",
            },
        },
        {
            "node_id": 5,
            "payload": {
                "execution_backend": "subprocess_task",
                "function_pickle": b"unused",
                "call_mode": "map_batches",
            },
        },
    ]

    def inject(handles_map):
        injected.append(handles_map)

    created, handles_map = subprocess_exec.ensure_local_subprocess_actor_pools_for_nodes(
        nodes,
        plan_identity="direct-plan",
        set_handles=inject,
    )

    assert len(created) == 1
    assert created_args[0][1] == 2
    assert created_args[0][2] == "local-subprocess-actor-direct-plan-4"
    assert handles_map == {"4": {"local_actor_pool": created[0]}}
    assert injected == [handles_map]


def test_ensure_local_subprocess_actor_pools_for_plan_propagates_collection_errors():
    import duckdb.execution.udf_subprocess as subprocess_exec

    class BrokenPlan:
        def collect_udf_nodes(self, conn=None):
            raise RuntimeError("collect failed")

    with pytest.raises(RuntimeError, match="collect failed"):
        subprocess_exec.ensure_local_subprocess_actor_pools_for_plan(BrokenPlan(), conn="conn")


def test_ensure_local_subprocess_actor_pools_for_plan_rolls_back_created_pools_on_failure(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec

    created = []

    class FakeLocalActorPool:
        def __init__(self, payload, pool_size, *, name=None):
            self.payload = payload
            self.pool_size = pool_size
            self.name = name
            self.shutdown_calls = []
            created.append(self)

        def shutdown(self, *, kill=False):
            self.shutdown_calls.append({"kill": kill})

    class BrokenPlan:
        def collect_udf_nodes(self, conn=None):
            return [
                {
                    "node_id": 7,
                    "payload": {
                        "execution_backend": "subprocess_actor",
                        "actor_number": 1,
                        "function_pickle": b"unused",
                        "call_mode": "map_batches",
                    },
                },
            ]

        def set_udf_actor_handles(self, handles_map, conn=None):
            raise RuntimeError("inject failed")

    monkeypatch.setattr(subprocess_exec, "LocalSubprocessActorPool", FakeLocalActorPool)

    with pytest.raises(RuntimeError, match="inject failed"):
        subprocess_exec.ensure_local_subprocess_actor_pools_for_plan(BrokenPlan(), conn="conn")

    assert len(created) == 1
    assert created[0].shutdown_calls == [{"kill": True}]


def test_subprocess_actor_fail_fast_unregisters_local_shm_budget_wakeup():
    import duckdb.execution.udf_subprocess as subprocess_exec
    from duckdb.execution import ref_bundle

    class Identity:
        def __call__(self, table):
            return table

    callbacks = ref_bundle._local_shm_budget_wakeup_callbacks
    before = set(callbacks)
    try:
        with pytest.raises(RuntimeError, match="requires a pre-created local_actor_pool"):
            subprocess_exec.UDFExecutor(
                _subprocess_map_payload(
                    Identity,
                    execution_backend="subprocess_actor",
                    actor_number=1,
                    produce_ref_bundle_output=True,
                    streaming_output_mode="local_shm_ref_bundle",
                )
            )

        assert set(callbacks) == before
    finally:
        callbacks.clear()
        callbacks.update(before)


def test_unified_executor_routes_subprocess_scalar_native():
    from duckdb.execution.unified_executor import build_unified_executor

    def add_two(value):
        return value + 2

    executor = build_unified_executor(_subprocess_scalar_payload(add_two))
    try:
        _submit_with_admission(executor, pa.table({"x": [1, None, 3]}))
        output = _wait_for_results(executor, 1, timeout_s=10.0)[0]

        assert output is not None
        assert output.to_pydict() == {"value": [3, None, 5]}
        assert executor.take_ready_result() is None
    finally:
        executor.close()


def test_udf_runtime_map_batches_stream_output_buffers_compute_subbatches_until_submit_flush():
    from duckdb.execution._udf_runtime import UDFExecutor

    def add_one(table):
        values = table.column("x").to_pylist()
        return pa.table(
            {
                "result": [value + 1 for value in values],
                "batch_rows": [table.num_rows for _ in values],
            }
        )

    payload = _subprocess_map_payload(
        add_one,
        batch_size=2,
        stream_output=True,
    )
    executor = UDFExecutor(payload)
    executor.submit(pa.table({"x": [1, 2, 3, 4]}))

    first = executor.take_ready_result()

    assert first is not None
    assert first.column("result").to_pylist() == [2, 3, 4, 5]
    assert first.column("batch_rows").to_pylist() == [2, 2, 2, 2]
    assert executor.take_ready_result() is None


def test_udf_runtime_iter_submit_stream_output_buffers_until_output_batch_size():
    from duckdb.execution._udf_runtime import UDFExecutor

    def add_one(table):
        values = table.column("x").to_pylist()
        return pa.table(
            {
                "result": [value + 1 for value in values],
                "batch_rows": [table.num_rows for _ in values],
            }
        )

    payload = _subprocess_map_payload(
        add_one,
        batch_size=2,
        output_batch_size=3,
        stream_output=True,
    )
    executor = UDFExecutor(payload)

    outputs = list(executor.iter_submit(pa.table({"x": [1, 2, 3, 4, 5]})))

    assert [output.num_rows for output in outputs] == [3, 2]
    assert [value for output in outputs for value in output.column("result").to_pylist()] == [2, 3, 4, 5, 6]
    assert [value for output in outputs for value in output.column("batch_rows").to_pylist()] == [2, 2, 2, 2, 1]


def test_udf_runtime_can_flush_stream_output_at_each_compute_batch_end():
    from duckdb.execution._udf_runtime import UDFExecutor

    def add_one(table):
        values = table.column("x").to_pylist()
        return pa.table(
            {
                "result": [value + 1 for value in values],
                "batch_rows": [table.num_rows for _ in values],
            }
        )

    payload = _subprocess_map_payload(
        add_one,
        batch_size=2,
        output_batch_size=3,
        stream_output=True,
        preserve_compute_batch_boundaries=True,
    )
    executor = UDFExecutor(payload)

    outputs = list(executor.iter_submit(pa.table({"x": [1, 2, 3, 4, 5]})))

    assert [output.num_rows for output in outputs] == [2, 2, 1]
    assert [value for output in outputs for value in output.column("result").to_pylist()] == [2, 3, 4, 5, 6]


def test_udf_runtime_output_buffer_flushes_by_target_bytes():
    from duckdb.execution._udf_runtime import RuntimeOutputBuffer

    table = pa.table({"payload": [b"x" * 64, b"y" * 64, b"z" * 64]})
    buffer = RuntimeOutputBuffer(target_rows=2048, target_bytes=1)

    outputs = list(buffer.append(table))
    outputs.extend(buffer.flush())

    assert [output.num_rows for output in outputs] == [1, 1, 1]
    assert [value for output in outputs for value in output.column("payload").to_pylist()] == [
        b"x" * 64,
        b"y" * 64,
        b"z" * 64,
    ]


def test_udf_runtime_output_buffer_accepts_zero_byte_null_tables():
    from duckdb.execution._udf_runtime import RuntimeOutputBuffer

    table = pa.table({"payload": [None, None, None]})
    buffer = RuntimeOutputBuffer(target_rows=2048, target_bytes=1)

    outputs = list(buffer.append(table))
    outputs.extend(buffer.flush())

    assert [output.num_rows for output in outputs] == [1, 1, 1]
    assert [value for output in outputs for value in output.column("payload").to_pylist()] == [None, None, None]


def test_udf_runtime_iter_submit_stream_output_flushes_by_target_bytes():
    from duckdb.execution._udf_runtime import UDFExecutor

    def make_payload(table):
        values = table.column("x").to_pylist()
        return pa.table({"payload": [bytes([value]) * 64 for value in values]})

    payload = _subprocess_map_payload(
        make_payload,
        batch_size=4,
        output_batch_size=2048,
        stream_output=True,
        udf_output_target_max_bytes=1,
    )
    executor = UDFExecutor(payload)

    outputs = list(executor.iter_submit(pa.table({"x": [1, 2, 3, 4]})))

    assert [output.num_rows for output in outputs] == [1, 1, 1, 1]
    assert [value[:1] for output in outputs for value in output.column("payload").to_pylist()] == [
        b"\x01",
        b"\x02",
        b"\x03",
        b"\x04",
    ]


def test_udf_runtime_iter_submit_stream_output_accepts_all_null_output():
    from duckdb.execution._udf_runtime import UDFExecutor

    def make_nulls(table):
        return pa.table({"payload": [None for _ in range(table.num_rows)]})

    payload = _subprocess_map_payload(
        make_nulls,
        batch_size=4,
        stream_output=True,
    )
    executor = UDFExecutor(payload)

    outputs = list(executor.iter_submit(pa.table({"x": [1, 2, 3]})))

    assert [output.num_rows for output in outputs] == [3]
    assert outputs[0].column("payload").to_pylist() == [None, None, None]


def test_udf_runtime_map_batches_buffers_input_until_compute_batch_size():
    from duckdb.execution._udf_runtime import UDFExecutor

    def report_compute_batch(table):
        values = table.column("x").to_pylist()
        return pa.table(
            {
                "x": values,
                "batch_rows": [table.num_rows for _ in values],
            }
        )

    payload = _subprocess_map_payload(
        report_compute_batch,
        batch_size=3000,
        output_batch_size=3000,
        stream_output=True,
    )
    executor = UDFExecutor(payload)

    executor.submit(pa.table({"x": list(range(2048))}))
    assert executor.take_ready_result() is None

    executor.submit(pa.table({"x": list(range(2048, 3000))}))
    output = executor.take_ready_result()

    assert output is not None
    assert output.num_rows == 3000
    assert output.column("x").to_pylist() == list(range(3000))
    assert set(output.column("batch_rows").to_pylist()) == {3000}
    assert executor.take_ready_result() is None


def test_udf_runtime_map_batches_finished_submitting_flushes_compute_tail():
    from duckdb.execution._udf_runtime import UDFExecutor

    def report_compute_batch(table):
        values = table.column("x").to_pylist()
        return pa.table(
            {
                "x": values,
                "batch_rows": [table.num_rows for _ in values],
            }
        )

    payload = _subprocess_map_payload(
        report_compute_batch,
        batch_size=3000,
        stream_output=True,
    )
    executor = UDFExecutor(payload)

    executor.submit(pa.table({"x": list(range(2048))}))
    assert executor.take_ready_result() is None

    executor.finished_submitting()
    output = executor.take_ready_result()

    assert output is not None
    assert output.num_rows == 2048
    assert output.column("x").to_pylist() == list(range(2048))
    assert set(output.column("batch_rows").to_pylist()) == {2048}
    assert executor.take_ready_result() is None


def test_udf_runtime_actor_backend_does_not_buffer_input_across_submits():
    from duckdb.execution._udf_runtime import UDFExecutor

    class ReportComputeBatch:
        def __call__(self, table):
            values = table.column("x").to_pylist()
            return pa.table(
                {
                    "x": values,
                    "batch_rows": [table.num_rows for _ in values],
                }
            )

    payload = _subprocess_map_payload(
        ReportComputeBatch,
        execution_backend="subprocess_actor",
        actor_number=1,
        batch_size=3000,
        output_batch_size=3000,
        stream_output=True,
    )
    executor = UDFExecutor(payload)

    executor.submit(pa.table({"x": list(range(2048))}))
    output = executor.take_ready_result()

    assert output is not None
    assert output.num_rows == 2048
    assert set(output.column("batch_rows").to_pylist()) == {2048}
    assert executor.take_ready_result() is None


def test_udf_runtime_stream_output_empty_result_supports_nested_schema():
    from duckdb.execution._udf_runtime import UDFExecutor

    feature_type = pa.struct(
        [
            ("label", pa.int64()),
            ("confidence", pa.float32()),
            ("bbox", pa.list_(pa.float32())),
        ]
    )

    def no_output(_table):
        return None

    payload = _subprocess_map_payload(
        no_output,
        stream_output=True,
        output_schema=[
            {"name": "frame_index", "kind": "duckdb_type", "type": "INTEGER"},
            {
                "name": "features",
                "kind": "duckdb_type",
                "type": 'STRUCT("label" BIGINT, confidence FLOAT, bbox FLOAT[])',
            },
            {"name": "object", "kind": "duckdb_type", "type": "BLOB"},
        ],
    )
    executor = UDFExecutor(payload)

    outputs = list(executor.iter_submit(pa.table({"x": [1]})))

    assert len(outputs) == 1
    assert outputs[0].num_rows == 0
    assert outputs[0].schema.field("frame_index").type == pa.int32()
    assert outputs[0].schema.field("features").type == feature_type
    assert outputs[0].schema.field("object").type == pa.binary()


def test_udf_runtime_map_batches_without_batch_size_passes_entire_block():
    from duckdb.execution._udf_runtime import UDFExecutor

    def report_batch_size(table):
        return pa.table({"rows": [table.num_rows]})

    payload = _subprocess_map_payload(
        report_batch_size,
    )
    executor = UDFExecutor(payload)
    executor.submit(pa.table({"x": list(range(4097))}))

    output = executor.take_ready_result()

    assert output is not None
    assert output.column("rows").to_pylist() == [4097]
    assert executor.take_ready_result() is None


def test_udf_runtime_callable_class_actor_backend_reuses_instance_state():
    from duckdb.execution._udf_runtime import UDFExecutor

    class StatefulBatchUDF:
        def __init__(self):
            self.calls = 0

        def __call__(self, table):
            self.calls += 1
            values = table.column("x").to_pylist()
            return pa.table(
                {
                    "y": [value + 10 for value in values],
                    "calls": [self.calls for _ in values],
                }
            )

    executor = UDFExecutor(
        _subprocess_map_payload(StatefulBatchUDF, execution_backend="subprocess_actor", actor_number=1)
    )
    executor.submit(pa.table({"x": [1, 2]}))
    first = executor.take_ready_result()
    executor.submit(pa.table({"x": [3]}))
    second = executor.take_ready_result()

    assert first is not None
    assert second is not None
    assert first.to_pydict() == {"y": [11, 12], "calls": [1, 1]}
    assert second.to_pydict() == {"y": [13], "calls": [2]}
    assert executor.take_ready_result() is None


def test_udf_runtime_callable_class_rejects_task_backend():
    from duckdb.execution._udf_runtime import UDFExecutor

    class StatefulBatchUDF:
        def __call__(self, table):
            return table

    with pytest.raises(ValueError, match="task UDF backends require a function, not a callable class"):
        UDFExecutor(_subprocess_map_payload(StatefulBatchUDF, execution_backend="subprocess_task"))


def test_udf_runtime_function_rejects_actor_backend():
    from duckdb.execution._udf_runtime import UDFExecutor

    def identity(table):
        return table

    with pytest.raises(ValueError, match="actor UDF backends require a callable class"):
        UDFExecutor(_subprocess_map_payload(identity, execution_backend="subprocess_actor", actor_number=1))


def test_udf_runtime_callable_class_constructor_must_be_zero_argument():
    from duckdb.execution._udf_runtime import UDFExecutor

    class NeedsModelName:
        def __init__(self, _model_name):
            pass

        def __call__(self, table):
            return table

    with pytest.raises(TypeError, match="constructors must be zero-argument"):
        UDFExecutor(_subprocess_map_payload(NeedsModelName, execution_backend="subprocess_actor", actor_number=1))


def test_udf_runtime_scalar_callable_class_actor_backend():
    from duckdb.execution._udf_runtime import UDFExecutor

    class AddOffset:
        def __init__(self):
            self.offset = 5

        def __call__(self, value):
            return value + self.offset

    executor = UDFExecutor(_subprocess_scalar_payload(AddOffset, execution_backend="subprocess_actor", actor_number=1))
    executor.submit(pa.table({"x": [1, None, 3]}))
    output = executor.take_ready_result()

    assert output is not None
    assert output.to_pydict() == {"value": [6, None, 8]}
    assert executor.take_ready_result() is None


def test_udf_runtime_stream_output_default_uses_runtime_batch_size():
    from duckdb.execution._udf_runtime import UDFExecutor

    def identity(table):
        values = table.column("x").to_pylist()
        return pa.table({"result": values})

    payload = _subprocess_map_payload(
        identity,
        batch_size=10,
        prebatched_input=True,
        stream_output=True,
    )
    executor = UDFExecutor(payload)
    executor.submit(pa.table({"x": list(range(25))}))

    outputs = [executor.take_ready_result()]

    assert [output.num_rows for output in outputs if output is not None] == [25]
    assert [value for output in outputs if output is not None for value in output.column("result").to_pylist()] == list(
        range(25)
    )
    assert executor.take_ready_result() is None


def test_udf_runtime_stream_output_default_ignores_submit_and_method_batch_size():
    from duckdb.execution._udf_runtime import UDFExecutor

    def identity(table):
        values = table.column("x").to_pylist()
        return pa.table({"result": values})

    payload = _subprocess_map_payload(
        identity,
        batch_size=10,
        submit_batch_size=4,
        method_batch_size=5,
        prebatched_input=True,
        stream_output=True,
    )
    executor = UDFExecutor(payload)
    executor.submit(pa.table({"x": list(range(25))}))

    outputs = [executor.take_ready_result()]

    assert [output.num_rows for output in outputs if output is not None] == [25]
    assert executor.take_ready_result() is None


def test_udf_runtime_stream_output_default_has_no_implicit_system_batch_size():
    from duckdb.execution._udf_runtime import UDFExecutor

    def identity(table):
        values = table.column("x").to_pylist()
        return pa.table({"result": values})

    payload = _subprocess_map_payload(
        identity,
        batch_size=3000,
        prebatched_input=True,
        stream_output=True,
    )
    executor = UDFExecutor(payload)
    executor.submit(pa.table({"x": list(range(5000))}))

    outputs = [executor.take_ready_result()]

    assert [output.num_rows for output in outputs if output is not None] == [5000]
    assert outputs[0] is not None
    assert outputs[0].column("result").to_pylist() == list(range(5000))
    assert executor.take_ready_result() is None


def test_udf_runtime_stream_output_batch_size_is_independent():
    from duckdb.execution._udf_runtime import UDFExecutor

    def identity(table):
        values = table.column("x").to_pylist()
        return pa.table({"result": values})

    payload = _subprocess_map_payload(
        identity,
        batch_size=10,
        output_batch_size=8,
        prebatched_input=True,
        stream_output=True,
    )
    executor = UDFExecutor(payload)
    executor.submit(pa.table({"x": list(range(25))}))

    outputs = [
        executor.take_ready_result(),
        executor.take_ready_result(),
        executor.take_ready_result(),
        executor.take_ready_result(),
    ]

    assert [output.num_rows for output in outputs if output is not None] == [8, 8, 8, 1]
    assert executor.take_ready_result() is None


def test_udf_runtime_flat_map_stream_output_yields_per_output_batch(tmp_path):
    from duckdb.execution._udf_runtime import UDFExecutor

    marker = tmp_path / "flat-map-seen.txt"

    def duplicate(row):
        with marker.open("a", encoding="utf-8") as handle:
            handle.write(f"{row['x']}\n")
        yield {"y": row["x"]}
        yield {"y": row["x"] + 10}

    payload = _subprocess_map_payload(
        duplicate,
        call_mode="flat_map",
        batch_size=3,
        output_batch_size=2,
        prebatched_input=True,
        stream_output=True,
        output_schema=[{"name": "y", "kind": "duckdb_type", "type": "INTEGER"}],
    )
    executor = UDFExecutor(payload)

    iterator = executor.iter_submit(pa.table({"x": [1, 2, 3]}))
    first = next(iterator)
    assert first.column("y").to_pylist() == [1, 11]
    assert marker.read_text().splitlines() == ["1"]

    outputs = [first, *list(iterator)]
    assert [output.num_rows for output in outputs] == [2, 2, 2]
    assert [value for output in outputs for value in output.column("y").to_pylist()] == [1, 11, 2, 12, 3, 13]
    assert marker.read_text().splitlines() == ["1", "2", "3"]


def test_udf_runtime_flat_map_stream_output_returns_empty_schema_when_all_rows_skip():
    from duckdb.execution._udf_runtime import UDFExecutor

    def skip(_row):
        return None

    payload = _subprocess_map_payload(
        skip,
        call_mode="flat_map",
        output_batch_size=2,
        prebatched_input=True,
        stream_output=True,
        output_schema=[{"name": "y", "kind": "duckdb_type", "type": "INTEGER"}],
    )
    executor = UDFExecutor(payload)

    outputs = list(executor.iter_submit(pa.table({"x": [1, 2, 3]})))

    assert len(outputs) == 1
    assert outputs[0].schema.names == ["y"]
    assert outputs[0].num_rows == 0


def test_ray_task_streaming_payload_enables_flat_map_stream_output():
    from duckdb.execution.udf_ray import _streaming_task_payload

    payload = {
        "call_mode": "flat_map",
        "prebatched_input": True,
    }

    stream_payload = _streaming_task_payload(payload)

    assert stream_payload["stream_output"] is True
    assert stream_payload["prebatched_input"] is False
    assert payload == {"call_mode": "flat_map", "prebatched_input": True}


def test_ray_task_ref_bundle_stream_flushes_compute_tail_after_finished_submitting(monkeypatch):
    import duckdb.execution.udf_ray as udf_ray

    monkeypatch.setattr(
        udf_ray,
        "validate_task_runtime_node",
        lambda payload: str(payload["node_id"]),
    )

    def report_compute_batch(table):
        values = table.column("x").to_pylist()
        return pa.table(
            {
                "x": values,
                "batch_rows": [table.num_rows for _ in values],
            }
        )

    payload = _subprocess_map_payload(
        report_compute_batch,
        execution_backend="ray_task",
        batch_size=3000,
        output_batch_size=3000,
    )
    payload.update(
        query_id="query-tail",
        stage_id="stage:query-tail:node:1:udf",
        task_lease_id="lease-tail",
        attempt_id="attempt-tail",
        node_id="node-a",
        udf_output_target_max_bytes=128 * 1024**2,
        output_window_bytes=256 * 1024**2,
    )
    table = pa.table({"x": list(range(2048))})

    outputs = list(
        udf_ray._iter_ref_bundle_task_outputs(
            payload,
            [table],
            None,
            [{"num_rows": table.num_rows}],
            ["x"],
        )
    )

    assert len(outputs) == 2
    assert outputs[0].num_rows == 2048
    assert outputs[0].column("x").to_pylist() == list(range(2048))
    assert set(outputs[0].column("batch_rows").to_pylist()) == {2048}
    assert outputs[1]["task_lease_id"] == "lease-tail"
    assert outputs[1]["attempt_id"] == "attempt-tail"


def test_ray_task_ref_bundle_map_batches_without_batch_size_passes_entire_block(monkeypatch):
    import duckdb.execution.udf_ray as udf_ray

    monkeypatch.setattr(
        udf_ray,
        "validate_task_runtime_node",
        lambda payload: str(payload["node_id"]),
    )

    def report_batch_size(table):
        return pa.table({"rows": [table.num_rows]})

    payload = _subprocess_map_payload(
        report_batch_size,
        execution_backend="ray_task",
    )
    payload.update(
        query_id="query-whole-block",
        stage_id="stage:query-whole-block:node:1:udf",
        task_lease_id="lease-whole-block",
        attempt_id="attempt-whole-block",
        node_id="node-a",
        udf_output_target_max_bytes=128 * 1024**2,
        output_window_bytes=256 * 1024**2,
    )
    table = pa.table({"x": list(range(4097))})

    outputs = list(
        udf_ray._iter_ref_bundle_task_outputs(
            payload,
            [table],
            None,
            [{"num_rows": table.num_rows}],
            ["x"],
        )
    )

    assert len(outputs) == 2
    assert outputs[0].column("rows").to_pylist() == [4097]
    assert outputs[1]["task_lease_id"] == "lease-whole-block"
    assert outputs[1]["attempt_id"] == "attempt-whole-block"


def test_subprocess_map_batches_concatenates_stream_output():
    from duckdb.execution.udf_subprocess import UDFExecutor

    def add_one(table):
        values = table.column("x").to_pylist()
        return pa.table({"result": [value + 1 for value in values]})

    payload = _subprocess_map_payload(
        add_one,
        batch_size=2,
        stream_output=True,
    )
    executor = UDFExecutor(payload)
    try:
        _submit_with_admission(executor, pa.table({"x": [1, 2, 3, 4]}))
        output = _wait_for_results(executor, 1, timeout_s=10.0)[0]

        assert output is not None
        assert output.to_pydict() == {"result": [2, 3, 4, 5]}
        assert executor.take_ready_result() is None
    finally:
        executor.close()


def test_subprocess_error_propagates_and_closes_worker():
    import duckdb.execution.udf_subprocess as subprocess_exec

    class Fail:
        def __call__(self, _table):
            raise ValueError("bad edge udf")

    payload = _subprocess_map_payload(Fail, execution_backend="subprocess_actor", actor_number=1)
    executor, pool = _make_subprocess_actor_executor(subprocess_exec, payload)
    try:
        _submit_with_admission(executor, pa.table({"x": [1]}))
        result = _wait_for_results(executor, 1, timeout_s=10.0)[0]
        assert isinstance(result, RuntimeError)
        assert "bad edge udf" in str(result)
        proc = executor._proc
        assert proc is None or proc.poll() is not None
    finally:
        executor.close(kill=True)
        pool.shutdown(kill=True)


def test_subprocess_close_is_idempotent():
    from duckdb.execution.udf_subprocess import UDFExecutor

    def identity(table):
        return table

    executor = UDFExecutor(_subprocess_map_payload(identity))
    proc = executor._proc
    executor.close()
    executor.close()
    assert proc is None or proc.poll() is not None


def test_subprocess_resizes_shared_memory_for_large_output():
    from duckdb.execution.udf_subprocess import UDFExecutor

    large_value = "x" * (2 * 1024 * 1024)

    def expand(table):
        return pa.table({"out": [large_value for _ in range(table.num_rows)]})

    executor = UDFExecutor(_subprocess_map_payload(expand))
    try:
        _submit_with_admission(executor, pa.table({"x": [1, 2]}))
        output = _wait_for_results(executor, 1, timeout_s=10.0)[0]

        assert output is not None
        assert output.column("out").to_pylist() == [large_value, large_value]
    finally:
        executor.close()


def test_subprocess_map_batches_none_output_returns_empty_table():
    from duckdb.execution.udf_subprocess import UDFExecutor

    def no_output(_table):
        return None

    executor = UDFExecutor(
        _subprocess_map_payload(
            no_output,
            output_schema=[{"name": "y", "kind": "duckdb_type", "type": "INTEGER"}],
        )
    )
    try:
        _submit_with_admission(executor, pa.table({"x": [1, 2]}))
        output = _wait_for_results(executor, 1, timeout_s=10.0)[0]

        assert output.schema.names == ["y"]
        assert output.num_rows == 0
    finally:
        executor.close(kill=True)


def test_subprocess_flat_map_all_skipped_rows_returns_empty_table():
    from duckdb.execution.udf_subprocess import UDFExecutor

    def skip_all(_row):
        return None

    executor = UDFExecutor(
        _subprocess_map_payload(
            skip_all,
            call_mode="flat_map",
            output_schema=[{"name": "y", "kind": "duckdb_type", "type": "INTEGER"}],
        )
    )
    try:
        _submit_with_admission(executor, pa.table({"x": [1, 2]}))
        output = _wait_for_results(executor, 1, timeout_s=10.0)[0]

        assert output.schema.names == ["y"]
        assert output.num_rows == 0
    finally:
        executor.close(kill=True)


def test_local_shm_ref_bundle_roundtrip():
    from duckdb.execution.ref_bundle import (
        REF_BUNDLE_RESULT_MARKER,
        make_local_shm_ref_bundle_result,
        materialize_ref_bundle,
    )

    table = pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    marker, refs, metadata, names = make_local_shm_ref_bundle_result(table)
    try:
        assert marker == REF_BUNDLE_RESULT_MARKER
        assert metadata[0]["provider"] == "local_shm"

        out = materialize_ref_bundle(refs, [(1, 3)], metadata, names)

        assert out.to_pydict() == {"x": [2, 3], "y": ["b", "c"]}
    finally:
        refs[0].release()


def test_local_shm_ref_bundle_byte_metadata_is_positive_for_all_null_nonempty_tables():
    import duckdb.execution.ref_bundle as ref_bundle
    from duckdb.execution._common import estimate_table_bytes

    table = pa.table({"payload": [None, None, None]})
    empty = table.slice(0, 0)

    assert table.nbytes == 0
    assert estimate_table_bytes(empty) == 0
    assert estimate_table_bytes(table) == table.num_rows

    _marker, refs, metadata, _names = ref_bundle.make_local_shm_ref_bundle_result(table)
    try:
        assert metadata[0]["num_rows"] == table.num_rows
        assert metadata[0]["size_bytes"] == table.num_rows
    finally:
        refs[0].release()

    descriptor = ref_bundle.make_local_shm_ref_bundle_descriptor(table)
    descriptor_refs = []
    try:
        _marker, descriptor_refs, descriptor_metadata, _names = (
            ref_bundle.make_local_shm_ref_bundle_result_from_descriptor(
                descriptor,
                block_on_budget=False,
            )
        )
        assert descriptor["metadata"][0]["num_rows"] == table.num_rows
        assert descriptor["metadata"][0]["size_bytes"] == table.num_rows
        assert descriptor_metadata[0]["size_bytes"] == table.num_rows
    finally:
        for ref in descriptor_refs:
            ref.release()
        if not descriptor_refs:
            ref_bundle._unlink_shared_memory_name(descriptor["block_refs"][0]["shm_name"])


def test_local_shm_multi_block_descriptor_splits_single_output_grant_budget(monkeypatch):
    import duckdb.execution.ref_bundle as ref_bundle

    monkeypatch.setenv("VANE_LOCAL_SHM_REF_BUDGET_BYTES", "1g")
    before = ref_bundle.local_shm_ref_budget_snapshot()["allocated_bytes"]
    first = ref_bundle.make_local_shm_ref_bundle_descriptor(pa.table({"payload": [b"a" * 64]}))
    second = ref_bundle.make_local_shm_ref_bundle_descriptor(pa.table({"payload": [b"b" * 64]}))
    descriptor = {
        "block_refs": first["block_refs"] + second["block_refs"],
        "metadata": first["metadata"] + second["metadata"],
        "names": first["names"],
    }
    total_size = sum(meta["ipc_size_bytes"] for meta in descriptor["metadata"])
    grant_id = ref_bundle.request_local_shm_output_grant(total_size, name="test-multi-block-grant")
    descriptor["grant_id"] = grant_id
    refs = []
    try:
        _marker, refs, metadata, names = ref_bundle.make_local_shm_ref_bundle_result_from_descriptor(
            descriptor,
            block_on_budget=False,
        )
        assert [meta["num_rows"] for meta in metadata] == [1, 1]
        assert names == ["payload"]
        after = ref_bundle.local_shm_ref_budget_snapshot()["allocated_bytes"]
        assert after - before == total_size
    finally:
        for ref in refs:
            ref.release()
        if not refs:
            for ref_desc in descriptor["block_refs"]:
                ref_bundle._unlink_shared_memory_name(ref_desc["shm_name"])


def test_materialize_ref_bundle_accepts_ray_object_ref(monkeypatch):
    import duckdb.execution.ref_bundle as ref_bundle

    class FakeObjectRef:
        pass

    table = pa.table({"x": [1, 2, 3]})
    fake_ref = FakeObjectRef()
    monkeypatch.setattr(ref_bundle, "_is_ray_object_ref", lambda ref: isinstance(ref, FakeObjectRef))
    monkeypatch.setattr(ref_bundle, "_resolve_ray_object_ref_blocks", lambda refs: [table])

    out = ref_bundle.materialize_ref_bundle([fake_ref], [(1, 3)], [{"num_rows": 3}], ["x"])

    assert out.to_pydict() == {"x": [2, 3]}


def test_materialize_ref_bundle_rejects_unresolved_refs_without_ray_fallback(monkeypatch):
    import duckdb.execution.ref_bundle as ref_bundle

    class UnresolvedRef:
        pass

    def fail_ray_materialize(*_args, **_kwargs):
        raise AssertionError("materialize_ref_bundle must not fall back to Ray materialization")

    import duckdb.execution.udf_ray_actor_runtime as actor_runtime

    monkeypatch.setattr(actor_runtime, "_materialize_ref_bundle", fail_ray_materialize)

    with pytest.raises(ValueError, match="unsupported ref bundle block"):
        ref_bundle.materialize_ref_bundle([UnresolvedRef()], None, [{}], ["x"])


def test_local_shm_descriptor_requires_current_schema():
    import duckdb.execution.ref_bundle as ref_bundle

    with pytest.raises(ValueError, match="missing shm_name"):
        ref_bundle._local_shm_descriptor_from_mapping(
            {"provider": "local_shm", "name": "psm-old", "ipc_size_bytes": 12},
            strict=True,
        )

    with pytest.raises(ValueError, match="missing ipc_size_bytes"):
        ref_bundle._local_shm_descriptor_from_mapping(
            {"provider": "local_shm", "shm_name": "psm-current", "size": 12},
            strict=True,
        )


def test_local_shm_ref_bundle_release_is_idempotent_and_observable():
    import duckdb.execution.ref_bundle as ref_bundle

    before = ref_bundle.local_shm_ref_lifecycle_snapshot()
    _marker, refs, _metadata, _names = ref_bundle.make_local_shm_ref_bundle_result(pa.table({"x": [1, 2, 3]}))
    created = ref_bundle.local_shm_ref_lifecycle_snapshot()

    refs[0].release()
    refs[0].release()

    after = ref_bundle.local_shm_ref_lifecycle_snapshot()
    assert created["created"] == before["created"] + 1
    assert after["released"] == before["released"] + 1
    assert after["live"] == before["live"]
    assert after["reserved_bytes"] == before["reserved_bytes"]


def test_local_shm_ref_bundle_materialize_avoids_bytes_copy(monkeypatch):
    import duckdb.execution.ref_bundle as ref_bundle

    def fail_bytes_copy(*_args, **_kwargs):
        raise AssertionError("local_shm materialization should not copy IPC payload into bytes")

    monkeypatch.setattr(ref_bundle, "_read_ipc_from_shm", fail_bytes_copy)

    _marker, refs, metadata, names = ref_bundle.make_local_shm_ref_bundle_result(pa.table({"x": [1, 2, 3]}))
    try:
        out = ref_bundle.materialize_ref_bundle(refs, None, metadata, names)

        assert out.to_pydict() == {"x": [1, 2, 3]}
    finally:
        refs[0].release()


def test_local_shm_ref_bundle_slice_retains_consumer_mapping_after_ref_release():
    import duckdb.execution.ref_bundle as ref_bundle

    _marker, refs, metadata, names = ref_bundle.make_local_shm_ref_bundle_result(
        pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    )
    try:
        out = ref_bundle.materialize_ref_bundle(refs, [(1, 3)], metadata, names)

        refs[0].release()

        assert out.to_pydict() == {"x": [2, 3], "y": ["b", "c"]}
    finally:
        refs[0].release()


def test_local_shm_ref_bundle_arrow_column_outlives_materialized_table():
    import duckdb.execution.ref_bundle as ref_bundle

    tensor = pa.FixedShapeTensorArray.from_numpy_ndarray(
        __import__("numpy").arange(24, dtype="float32").reshape(2, 3, 4)
    )
    _marker, refs, metadata, names = ref_bundle.make_local_shm_ref_bundle_result(pa.table({"x": tensor}))
    try:
        out = ref_bundle.materialize_ref_bundle(refs, None, metadata, names)
        column = out.column("x")

        del out
        gc.collect()
        refs[0].release()

        assert column.to_pylist()[0] == [float(value) for value in range(12)]
    finally:
        refs[0].release()


def test_local_shm_ref_bundle_auto_budget_uses_ray_like_capacity(monkeypatch):
    import duckdb.execution.ref_bundle as ref_bundle

    monkeypatch.delenv("VANE_LOCAL_SHM_REF_BUDGET_BYTES", raising=False)
    monkeypatch.setattr(ref_bundle, "_available_system_memory_bytes", lambda: 100 * ref_bundle._GIB)
    monkeypatch.setattr(ref_bundle, "_available_local_shm_bytes", lambda: 80 * ref_bundle._GIB)

    expected_capacity = min(
        int(100 * ref_bundle._GIB * ref_bundle._RAY_LIKE_OBJECT_STORE_MEMORY_FRACTION),
        int(80 * ref_bundle._GIB * ref_bundle._RAY_LIKE_SHM_MEMORY_FRACTION),
        ref_bundle._RAY_LIKE_OBJECT_STORE_MAX_BYTES,
    )
    assert ref_bundle._auto_local_shm_ref_budget_bytes() == int(expected_capacity * 0.5)


def test_local_shm_ref_bundle_auto_budget_never_exceeds_small_shm_capacity(monkeypatch):
    from duckdb.execution import ref_bundle

    shm_capacity = 256 * ref_bundle._MIB
    monkeypatch.setattr(ref_bundle, "_available_system_memory_bytes", lambda: 80 * ref_bundle._GIB)
    monkeypatch.setattr(ref_bundle, "_available_local_shm_bytes", lambda: shm_capacity)

    assert ref_bundle._auto_local_shm_ref_budget_bytes() <= shm_capacity


def test_local_shm_ref_bundle_acquires_budget_before_creating_shm(monkeypatch):
    from duckdb.execution import ref_bundle

    events = []
    acquire = ref_bundle._acquire_local_shm_ref_budget
    create = ref_bundle._create_shm

    def record_acquire(size, *, name=""):
        events.append(("acquire", size))
        return acquire(size, name=name)

    def record_create(size, *, track):
        events.append(("create", size))
        return create(size, track=track)

    monkeypatch.setenv("VANE_LOCAL_SHM_REF_BUDGET_BYTES", "1g")
    monkeypatch.setattr(ref_bundle, "_acquire_local_shm_ref_budget", record_acquire)
    monkeypatch.setattr(ref_bundle, "_create_shm", record_create)
    _marker, refs, _metadata, _names = ref_bundle.make_local_shm_ref_bundle_result(pa.table({"x": [1, 2, 3]}))
    try:
        assert [event[0] for event in events[:2]] == ["acquire", "create"]
    finally:
        for ref in refs:
            ref.release()


def test_local_shm_budget_manager_auto_limit_does_not_shrink_after_first_resolution(monkeypatch):
    import duckdb.execution.ref_bundle as ref_bundle

    monkeypatch.delenv("VANE_LOCAL_SHM_REF_BUDGET_BYTES", raising=False)
    available_system = [100 * ref_bundle._GIB]
    available_shm = [80 * ref_bundle._GIB]
    monkeypatch.setattr(ref_bundle, "_available_system_memory_bytes", lambda: available_system[0])
    monkeypatch.setattr(ref_bundle, "_available_local_shm_bytes", lambda: available_shm[0])

    manager = ref_bundle.LocalShmBudgetManager()
    first_limit = manager.snapshot()["limit_bytes"]

    available_system[0] = 16 * ref_bundle._GIB
    available_shm[0] = 8 * ref_bundle._GIB

    assert manager.snapshot()["limit_bytes"] == first_limit


def test_local_shm_ref_bundle_can_claim_output_budget_matches_claim_semantics(monkeypatch):
    import duckdb.execution.ref_bundle as ref_bundle

    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)

    manager.acquire_allocation(80, name="existing")
    assert ref_bundle.can_claim_local_shm_ref_output_budget(20)
    assert not ref_bundle.can_claim_local_shm_ref_output_budget(21)

    manager.release_allocation(80, name="existing")
    assert ref_bundle.can_claim_local_shm_ref_output_budget(200)


def test_local_shm_ref_bundle_submit_admission_allows_small_consumer_when_hard_budget_full(monkeypatch):
    import duckdb.execution.ref_bundle as ref_bundle

    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 1000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_REF_OUTPUT_PRODUCER_SOFT_MIN_BYTES", 100)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_REF_OUTPUT_PRODUCER_SOFT_LIMIT_FRACTION", 0.75)

    manager.acquire_allocation(995, name="full")

    assert not ref_bundle.can_claim_local_shm_ref_output_budget(10)
    assert ref_bundle.can_admit_local_shm_ref_output_submit(10)


def test_local_shm_ref_bundle_submit_admission_throttles_large_producer_at_soft_watermark(monkeypatch):
    import duckdb.execution.ref_bundle as ref_bundle

    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 1000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_REF_OUTPUT_PRODUCER_SOFT_MIN_BYTES", 100)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_REF_OUTPUT_PRODUCER_SOFT_LIMIT_FRACTION", 0.75)

    manager.acquire_allocation(700, name="backlog")

    assert ref_bundle.can_claim_local_shm_ref_output_budget(100)
    assert not ref_bundle.can_admit_local_shm_ref_output_submit(100)
    assert ref_bundle.can_admit_local_shm_ref_output_submit(10)


def test_local_shm_ref_bundle_submit_admission_counts_projected_inflight_output(monkeypatch):
    import duckdb.execution.ref_bundle as ref_bundle

    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 1000)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_REF_OUTPUT_PRODUCER_SOFT_MIN_BYTES", 100)
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_REF_OUTPUT_PRODUCER_SOFT_LIMIT_FRACTION", 0.75)

    manager.acquire_allocation(600, name="backlog")

    assert ref_bundle.can_admit_local_shm_ref_output_submit(100)
    assert not ref_bundle.can_admit_local_shm_ref_output_submit(100, projected_output_bytes=100)


def test_local_shm_ref_output_budget_cancel_wait_is_event_driven(monkeypatch):
    import duckdb.execution.ref_bundle as ref_bundle

    cancel_event = threading.Event()

    class RecordingCondition:
        def __init__(self):
            self.wait_timeouts = []

        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return False

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            cancel_event.set()

        def wait_for(self, predicate):
            if predicate():
                return True
            self.wait()
            return predicate()

        def notify_all(self):
            return None

    condition = RecordingCondition()
    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 100)
    manager.acquire_allocation(90, name="existing")
    manager._cond = condition
    monkeypatch.setattr(ref_bundle, "_LOCAL_SHM_BUDGET_MANAGER", manager)

    with pytest.raises(RuntimeError, match="local_shm output budget admission cancelled"):
        ref_bundle.claim_local_shm_ref_output_budget(20, name="cancel-test", cancel_event=cancel_event)

    assert condition.wait_timeouts == [None]


def test_local_shm_ref_bundle_budget_blocks_until_release(monkeypatch):
    import duckdb.execution.ref_bundle as ref_bundle

    table = pa.table({"x": ["x" * 4096]})
    required = ref_bundle._IPC_HEADER_SIZE + len(ref_bundle._arrow_table_to_ipc_bytes(table))
    monkeypatch.setenv("VANE_LOCAL_SHM_REF_BUDGET_BYTES", str(required + 1))

    wait_started = threading.Event()
    original_log = ref_bundle._shm_debug_log

    def record_budget_wait(event, **fields):
        if event == "budget_wait":
            wait_started.set()
        original_log(event, **fields)

    monkeypatch.setattr(ref_bundle, "_shm_debug_log", record_budget_wait)

    _marker, refs, _metadata, _names = ref_bundle.make_local_shm_ref_bundle_result(table)
    done = threading.Event()
    holder = {}

    def make_second_ref():
        holder["result"] = ref_bundle.make_local_shm_ref_bundle_result(table)
        done.set()

    thread = threading.Thread(target=make_second_ref, daemon=True)
    try:
        thread.start()
        assert wait_started.wait(5.0)
        assert not done.is_set()

        refs[0].release()

        assert done.wait(5.0)
        _marker2, refs2, _metadata2, _names2 = holder["result"]
    finally:
        refs[0].release()
        if "result" in holder:
            holder["result"][1][0].release()
        thread.join(timeout=5.0)

    with ref_bundle._local_shm_budget_cond:
        assert ref_bundle._local_shm_budget_reserved_bytes == 0
        assert ref_bundle._local_shm_budget_pending_output_bytes == 0


def test_local_shm_ref_bundle_descriptor_wrap_can_overcommit_without_block(monkeypatch):
    import duckdb.execution.ref_bundle as ref_bundle

    table = pa.table({"x": ["x" * 4096]})
    required = ref_bundle._IPC_HEADER_SIZE + len(ref_bundle._arrow_table_to_ipc_bytes(table))
    monkeypatch.setenv("VANE_LOCAL_SHM_REF_BUDGET_BYTES", str(required + 1))

    descriptor = ref_bundle.make_local_shm_ref_bundle_descriptor(table)
    _marker, refs, _metadata, _names = ref_bundle.make_local_shm_ref_bundle_result(table)

    wait_events = []
    original_log = ref_bundle._shm_debug_log

    def record_budget_wait(event, **fields):
        if event == "budget_wait":
            wait_events.append(fields)
        original_log(event, **fields)

    monkeypatch.setattr(ref_bundle, "_shm_debug_log", record_budget_wait)

    done = threading.Event()
    holder = {}

    def wrap_descriptor():
        holder["result"] = ref_bundle.make_local_shm_ref_bundle_result_from_descriptor(
            descriptor,
            block_on_budget=False,
        )
        done.set()

    thread = threading.Thread(target=wrap_descriptor, daemon=True)
    try:
        thread.start()
        assert done.wait(5.0)
        assert not wait_events
    finally:
        refs[0].release()
        if "result" in holder:
            holder["result"][1][0].release()
        thread.join(timeout=5.0)

    with ref_bundle._local_shm_budget_cond:
        assert ref_bundle._local_shm_budget_reserved_bytes == 0
        assert ref_bundle._local_shm_budget_pending_output_bytes == 0


def test_subprocess_ref_bundle_output_materializes():
    from duckdb.execution.ref_bundle import REF_BUNDLE_RESULT_MARKER, SUBMIT_RESULT_MARKER, materialize_ref_bundle
    from duckdb.execution.udf_subprocess import UDFExecutor

    def add_one(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    executor = UDFExecutor(
        _subprocess_map_payload(
            add_one,
            produce_ref_bundle_output=True,
            streaming_output_mode="local_shm_ref_bundle",
        )
    )
    try:
        _submit_with_admission(executor, pa.table({"x": [1, 2, 3]}), submit_id=11)
        wrapped = _wait_for_results(executor, 1, timeout_s=10.0)[0]

        assert wrapped is not None
        assert wrapped[0] == SUBMIT_RESULT_MARKER
        assert wrapped[1] == 11
        marker, refs, metadata, names = wrapped[2]
        try:
            assert marker == REF_BUNDLE_RESULT_MARKER
            output = materialize_ref_bundle(refs, None, metadata, names)
            assert output.to_pydict() == {"y": [2, 3, 4]}
        finally:
            refs[0].release()
    finally:
        executor.close()


def test_subprocess_ref_bundle_output_avoids_parent_table_roundtrip(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec
    from duckdb.execution.ref_bundle import REF_BUNDLE_RESULT_MARKER, materialize_ref_bundle

    def fail_parent_deserialize(*_args, **_kwargs):
        raise AssertionError("subprocess parent should not deserialize worker output IPC")

    monkeypatch.setattr(subprocess_exec, "_arrow_table_from_ipc_bytes", fail_parent_deserialize)

    def add_one(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    executor = subprocess_exec.UDFExecutor(
        _subprocess_map_payload(
            add_one,
            produce_ref_bundle_output=True,
            streaming_output_mode="local_shm_ref_bundle",
        )
    )
    try:
        _submit_with_admission(executor, pa.table({"x": [1, 2, 3]}))
        wrapped = _wait_for_results(executor, 1, timeout_s=10.0)[0]

        assert wrapped is not None
        marker, refs, metadata, names = wrapped
        try:
            assert marker == REF_BUNDLE_RESULT_MARKER
            assert metadata[0]["provider"] == "local_shm"
            output = materialize_ref_bundle(refs, None, metadata, names)
            assert output.to_pydict() == {"y": [2, 3, 4]}
        finally:
            refs[0].release()
    finally:
        executor.close()


def test_subprocess_ref_bundle_mode_rejects_direct_ipc(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec

    def add_one(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    executor = subprocess_exec._SingleSubprocessExecutor(
        _subprocess_map_payload(
            add_one,
            produce_ref_bundle_output=True,
            streaming_output_mode="local_shm_ref_bundle",
        )
    )
    try:
        monkeypatch.setattr(
            executor,
            "_recv_expected",
            lambda _expected: (subprocess_exec._MSG_OK, struct.pack("<Q", 128)),
        )
        with pytest.raises(RuntimeError, match="must be a local_shm ref-bundle result"):
            executor._recv_submit_result()
    finally:
        executor.close(kill=True)


def test_subprocess_ref_bundle_contract_requires_output_mode():
    import duckdb.execution.udf_subprocess as subprocess_exec

    def add_one(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    with pytest.raises(RuntimeError, match="streaming_output_mode='local_shm_ref_bundle'"):
        subprocess_exec.UDFExecutor(_subprocess_map_payload(add_one, produce_ref_bundle_output=True))


def test_subprocess_ref_bundle_contract_requires_produce_flag():
    import duckdb.execution.udf_subprocess as subprocess_exec

    def add_one(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    with pytest.raises(RuntimeError, match="produce_ref_bundle_output=True"):
        subprocess_exec.UDFExecutor(_subprocess_map_payload(add_one, streaming_output_mode="local_shm_ref_bundle"))


def test_subprocess_materialized_input_releases_budget_before_output_grant(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec
    from duckdb.execution import ref_bundle

    def identity(table):
        return pa.table({"y": table.column("x")})

    monkeypatch.setenv("VANE_LOCAL_SHM_REF_BUDGET_BYTES", "2m")
    table = pa.table({"x": [b"x" * 1_200_000]})
    executor = subprocess_exec.UDFExecutor(
        _subprocess_map_payload(
            identity,
            produce_ref_bundle_output=True,
            streaming_output_mode="local_shm_ref_bundle",
        )
    )
    output = None
    try:
        _submit_with_admission(executor, table)
        results = _wait_for_results(executor, 1, timeout_s=5.0)
        assert len(results) == 1
        output = results[0]
        assert output[0] == ref_bundle.REF_BUNDLE_RESULT_MARKER
        assert ref_bundle.materialize_ref_bundle(output[1], None, output[2], output[3]).to_pydict() == {
            "y": [b"x" * 1_200_000]
        }
    finally:
        executor.close(kill=True)
        if output is not None:
            for ref in output[1]:
                ref.release()


def test_subprocess_consumes_local_shm_ref_bundle_with_id_without_parent_materialize(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec
    from duckdb.execution.ref_bundle import SUBMIT_RESULT_MARKER, make_local_shm_ref_bundle_result

    def add_one(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    _marker, refs, metadata, names = make_local_shm_ref_bundle_result(pa.table({"x": [1, 2, 3]}))
    executor = subprocess_exec.UDFExecutor(_subprocess_map_payload(add_one))
    try:
        _submit_ref_bundle_with_admission(
            executor,
            12,
            refs,
            [(1, 3)],
            metadata,
            names,
        )
        wrapped = _wait_for_results(executor, 1, timeout_s=10.0)[0]

        assert wrapped is not None
        assert wrapped[0] == SUBMIT_RESULT_MARKER
        assert wrapped[1] == 12
        assert wrapped[2].to_pydict() == {"y": [3, 4]}
    finally:
        refs[0].release()
        executor.close()


def test_subprocess_actor_pool_submit_with_id_uses_multiple_workers():
    import duckdb.execution.udf_subprocess as subprocess_exec
    from duckdb.execution.ref_bundle import SUBMIT_RESULT_MARKER

    class WorkerPID:
        def __call__(self, table):
            import os
            import time

            time.sleep(0.2)
            return pa.table({"pid": [os.getpid() for _ in range(table.num_rows)]})

    payload = _subprocess_map_payload(
        WorkerPID,
        execution_backend="subprocess_actor",
        actor_number=2,
    )
    executor, pool = _make_subprocess_actor_executor(subprocess_exec, payload)
    try:
        assert executor._workers == []
        assert executor._executor is None
        assert executor._actor_pool is pool
        assert pool.pool_size == 2

        _submit_with_admission(executor, pa.table({"x": [1]}), submit_id=101)
        _submit_with_admission(executor, pa.table({"x": [2]}), submit_id=102)
        results = _wait_for_results(executor, 2)

        assert len(results) == 2
        assert {result[0] for result in results} == {SUBMIT_RESULT_MARKER}
        assert {result[1] for result in results} == {101, 102}
        pids = {result[2].column("pid").to_pylist()[0] for result in results}
        assert len(pids) == 2
    finally:
        executor.close()
        pool.shutdown(kill=True)


def test_subprocess_task_rejects_callable_instance():
    from duckdb.execution._udf_runtime import UDFExecutor

    class StatefulBatchUDF:
        def __init__(self):
            self.calls = 0

        def __call__(self, table):
            self.calls += 1
            return pa.table({"calls": [self.calls for _ in range(table.num_rows)]})

    with pytest.raises(ValueError, match="task UDF backends require a function"):
        UDFExecutor(
            _subprocess_map_payload(
                StatefulBatchUDF(),
                execution_backend="subprocess_task",
                udf_worker_slots=1,
            )
        )


def test_subprocess_actor_reuses_callable_class_state():
    import duckdb.execution.udf_subprocess as subprocess_exec
    from duckdb.execution.ref_bundle import SUBMIT_RESULT_MARKER

    class StatefulBatchUDF:
        def __init__(self):
            self.calls = 0

        def __call__(self, table):
            import os

            import pyarrow as pa

            self.calls += 1
            return pa.table(
                {
                    "pid": [os.getpid() for _ in range(table.num_rows)],
                    "calls": [self.calls for _ in range(table.num_rows)],
                }
            )

    payload = _subprocess_map_payload(
        StatefulBatchUDF,
        execution_backend="subprocess_actor",
        actor_number=1,
    )
    executor, pool = _make_subprocess_actor_executor(subprocess_exec, payload)
    try:
        _submit_with_admission(executor, pa.table({"x": [1]}), submit_id=141)
        first = _wait_for_results(executor, 1, timeout_s=10.0)[0]
        _submit_with_admission(executor, pa.table({"x": [2]}), submit_id=142)
        second = _wait_for_results(executor, 1, timeout_s=10.0)[0]

        assert first[0] == SUBMIT_RESULT_MARKER
        assert second[0] == SUBMIT_RESULT_MARKER
        assert {first[1], second[1]} == {141, 142}
        assert first[2].column("pid").to_pylist() == second[2].column("pid").to_pylist()
        assert first[2].column("calls").to_pylist() == [1]
        assert second[2].column("calls").to_pylist() == [2]
    finally:
        executor.close()
        pool.shutdown(kill=True)


def test_subprocess_task_worker_slots_control_pool_size(monkeypatch):
    from duckdb.execution.ref_bundle import SUBMIT_RESULT_MARKER
    from duckdb.execution.udf_subprocess import UDFExecutor

    def worker_pid(table):
        import os
        import time

        time.sleep(0.15)
        return pa.table({"pid": [os.getpid() for _ in range(table.num_rows)]})

    executor = UDFExecutor(
        _subprocess_map_payload(
            worker_pid,
            execution_backend="subprocess_task",
            udf_worker_slots=1,
        )
    )
    try:
        assert len(executor._workers) == 0
        assert executor._executor is None
        assert executor._task_pool is not None

        results = []
        for submit_id, value in ((151, 1), (152, 2), (153, 3)):
            _submit_with_admission(executor, pa.table({"x": [value]}), submit_id=submit_id)
            results.extend(_wait_for_results(executor, 1, timeout_s=10.0))

        assert len(results) == 3
        assert {result[0] for result in results} == {SUBMIT_RESULT_MARKER}
        assert {result[1] for result in results} == {151, 152, 153}
        pids = {result[2].column("pid").to_pylist()[0] for result in results}
        assert len(pids) == 1
    finally:
        executor.close()


def test_subprocess_task_stats_report_worker_slot_admission():
    import duckdb.execution.udf_subprocess as subprocess_exec

    subprocess_exec._shutdown_global_task_runtime()

    def sleeper(table):
        import time

        time.sleep(0.2)
        return table

    executor = subprocess_exec.UDFExecutor(
        _subprocess_map_payload(
            sleeper,
            execution_backend="subprocess_task",
            udf_worker_slots=3,
        )
    )
    try:
        assert executor.stats() == {
            "udf_running_task_count": 0,
            "udf_queued_task_count": 0,
            "udf_max_running_tasks": 3,
        }

        _submit_with_admission(executor, pa.table({"x": [1]}), submit_id=171)
        _submit_with_admission(executor, pa.table({"x": [2]}), submit_id=172)
        _submit_with_admission(executor, pa.table({"x": [3]}), submit_id=173)

        observed_stats = _wait_for_executor_stats(
            executor,
            lambda stats: stats["udf_running_task_count"] >= 2,
        )

        assert observed_stats["udf_running_task_count"] >= 2
        assert len(_wait_for_results(executor, 3, timeout_s=10.0)) == 3
        assert executor.stats()["udf_running_task_count"] == 0
        assert executor.stats()["udf_queued_task_count"] == 0
    finally:
        executor.close(kill=True)
        subprocess_exec._shutdown_global_task_runtime()


def test_subprocess_task_ref_bundle_output_claims_schema_budget(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec

    subprocess_exec._shutdown_global_task_runtime()
    claims = []
    releases = []

    def make_output(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    payload = _subprocess_map_payload(
        make_output,
        execution_backend="subprocess_task",
        udf_worker_slots=1,
        produce_ref_bundle_output=True,
        streaming_output_mode="local_shm_ref_bundle",
        output_schema=[
            {
                "name": "y",
                "kind": "tensor",
                "type": "",
                "dtype": "FLOAT",
                "shape": [3, 4],
            }
        ],
    )
    executor = subprocess_exec.UDFExecutor(payload)
    try:
        _submit_with_admission(executor, pa.table({"x": [1, 2]}), submit_id=175)
        wrapped = _wait_for_results(executor, 1, timeout_s=10.0)[0]

        assert wrapped[1] == 175
        marker, refs, _metadata, _names = wrapped[2]
        try:
            assert marker == "__vane_ref_bundle_result__"
            assert claims == []
            assert releases == []
        finally:
            refs[0].release()
    finally:
        executor.close(kill=True)
        subprocess_exec._shutdown_global_task_runtime()


def test_subprocess_ref_bundle_output_stats_report_budget_availability(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec

    subprocess_exec._shutdown_global_task_runtime()
    monkeypatch.setattr(
        subprocess_exec,
        "can_admit_local_shm_ref_output_submit",
        lambda _size, *, projected_output_bytes=0: False,
    )
    monkeypatch.setattr(
        subprocess_exec,
        "local_shm_ref_budget_snapshot",
        lambda: {
            "limit_bytes": 100,
            "usage_bytes": 99,
            "reserved_bytes": 80,
            "pending_output_bytes": 19,
        },
    )

    def make_output(table):
        return pa.table({"y": table.column("x")})

    executor = subprocess_exec.UDFExecutor(
        _subprocess_map_payload(
            make_output,
            execution_backend="subprocess_task",
            udf_worker_slots=1,
            produce_ref_bundle_output=True,
            streaming_output_mode="local_shm_ref_bundle",
            output_schema=[
                {
                    "name": "y",
                    "kind": "tensor",
                    "type": "",
                    "dtype": "FLOAT",
                    "shape": [3, 4],
                }
            ],
        )
    )
    try:
        assert executor._output_budget_estimate(2) >= 2 * 3 * 4 * 4
        stats = executor.stats()
        assert stats["udf_output_budget_available"] == 0
        assert stats["udf_output_budget_estimated_bytes"] >= 2 * 3 * 4 * 4
        assert stats["udf_output_budget_limit_bytes"] == 100
        assert stats["udf_output_budget_usage_bytes"] == 99
    finally:
        executor.close(kill=True)
        subprocess_exec._shutdown_global_task_runtime()


def test_subprocess_ref_bundle_blob_output_schema_has_initial_budget_estimate():
    from duckdb.execution.udf_subprocess import UDFExecutor

    def make_output(table):
        return pa.table({"blob": table.column("x")})

    executor = UDFExecutor(
        _subprocess_map_payload(
            make_output,
            execution_backend="subprocess_task",
            udf_worker_slots=1,
            produce_ref_bundle_output=True,
            streaming_output_mode="local_shm_ref_bundle",
            output_schema=[
                {
                    "name": "blob",
                    "kind": "duckdb_type",
                    "type": "BLOB",
                    "dtype": "",
                    "shape": [],
                }
            ],
        )
    )
    try:
        assert executor._output_budget_estimate(64) >= 64 * (1 << 20)
    finally:
        executor.close(kill=True)


def test_subprocess_output_grant_request_uses_executor_cancel_event(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec

    parent_sock, child_sock = subprocess_exec.socket.socketpair()
    executor = subprocess_exec._SingleSubprocessExecutor.__new__(subprocess_exec._SingleSubprocessExecutor)
    executor._closed = False
    executor._broken_error = None
    executor._sock = parent_sock
    executor._wakeup = None
    executor._active_output_grants = set()
    executor._active_output_grants_lock = threading.Lock()
    executor._output_grant_cancel_event = threading.Event()

    captured: dict[str, object] = {}

    def fake_request(size, *, name="", priority="producer", input_lease_id=None, cancel_event=None):
        captured["size"] = size
        captured["name"] = name
        captured["priority"] = priority
        captured["input_lease_id"] = input_lease_id
        captured["cancel_event"] = cancel_event
        return 77

    monkeypatch.setattr(subprocess_exec, "request_local_shm_output_grant", fake_request)

    try:
        payload = subprocess_exec.duckdb_pickle.dumps(
            {
                "request_id": 9,
                "size_bytes": 128,
                "priority": "consumer",
                "input_lease_id": 123,
            }
        )
        assert executor._handle_submit_control_message(subprocess_exec._MSG_OUTPUT_GRANT_REQUEST, payload)

        msg_type, response_payload = subprocess_exec._recv_message(child_sock)
        response = subprocess_exec.duckdb_pickle.loads(response_payload)
        assert msg_type == subprocess_exec._MSG_OUTPUT_GRANT_GRANTED
        assert response["grant_id"] == 77
        assert captured == {
            "size": 128,
            "name": "udf-output-9",
            "priority": "consumer",
            "input_lease_id": 123,
            "cancel_event": executor._output_grant_cancel_event,
        }
    finally:
        parent_sock.close()
        child_sock.close()


def test_single_subprocess_close_without_kill_cancels_output_grant_wait(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec

    events: list[object] = []
    executor = subprocess_exec._SingleSubprocessExecutor.__new__(subprocess_exec._SingleSubprocessExecutor)
    executor._closed = False
    executor._proc = None
    executor._sock = None
    executor._payload_shm = None
    executor._data_shm = None
    executor._finalizer = None
    executor._active_input_leases = {42}
    executor._active_input_leases_lock = threading.Lock()
    executor._active_output_grants = set()
    executor._active_output_grants_lock = threading.Lock()
    executor._output_grant_cancel_event = threading.Event()

    monkeypatch.setattr(
        subprocess_exec,
        "cancel_local_shm_input_lease",
        lambda lease_id, *, name="": events.append(("cancel-lease", lease_id, name)),
    )
    monkeypatch.setattr(
        subprocess_exec,
        "wake_local_shm_ref_budget_waiters",
        lambda: events.append("wake-budget"),
    )

    executor.close(kill=False)

    assert executor._output_grant_cancel_event.is_set()
    assert events == [
        "wake-budget",
        ("cancel-lease", 42, "udf-input-close"),
    ]


def test_local_subprocess_actor_pool_shutdown_cancels_worker_grants_before_executor_wait():
    import duckdb.execution.udf_subprocess as subprocess_exec

    events: list[str] = []

    class FakeWorker:
        def __init__(self, name: str):
            self.name = name

        def cancel_output_grants(self) -> None:
            events.append(f"cancel:{self.name}")

        def close(self, *, kill: bool = False) -> None:
            events.append(f"close:{self.name}:{kill}")

    class FakeExecutor:
        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            assert events[:3] == ["admission-close", "cancel:w0", "cancel:w1"]
            events.append(f"shutdown:{wait}:{cancel_futures}")

    class FakeAdmissionSlots:
        def close(self) -> None:
            events.append("admission-close")

    pool = subprocess_exec.LocalSubprocessActorPool.__new__(subprocess_exec.LocalSubprocessActorPool)
    pool.name = "pool"
    pool._closed = False
    pool._lock = threading.Lock()
    pool._workers = [FakeWorker("w0"), FakeWorker("w1")]
    pool._executor = FakeExecutor()
    pool.admission_slots = FakeAdmissionSlots()

    pool.shutdown(kill=False)

    assert events == [
        "admission-close",
        "cancel:w0",
        "cancel:w1",
        "shutdown:False:True",
        "close:w0:False",
        "close:w1:False",
    ]
    assert pool._workers == []


def test_local_subprocess_actor_pool_rolls_back_created_workers_on_worker_init_failure(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec

    class Identity:
        def __call__(self, table):
            return table

    events: list[str] = []

    class FakeWorker:
        def __init__(self, name: str):
            self.name = name

        def close(self, *, kill: bool = False) -> None:
            events.append(f"close:{self.name}:{kill}")

    created: list[FakeWorker] = []

    def fake_worker(payload, *, worker_env=None):
        if len(created) == 1:
            raise RuntimeError("worker init failed")
        worker = FakeWorker(f"w{len(created)}")
        created.append(worker)
        return worker

    monkeypatch.setattr(subprocess_exec, "_SingleSubprocessExecutor", fake_worker)

    with pytest.raises(RuntimeError, match="worker init failed"):
        subprocess_exec.LocalSubprocessActorPool(
            _subprocess_map_payload(
                Identity,
                execution_backend="subprocess_actor",
                actor_number=2,
            ),
            2,
        )

    assert events == ["close:w0:True"]


def test_local_subprocess_actor_pool_rolls_back_created_workers_on_thread_pool_init_failure(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec

    class Identity:
        def __call__(self, table):
            return table

    events: list[str] = []
    created: list[str] = []

    class FakeWorker:
        _proc = None

        def __init__(self, payload, *, worker_env=None):
            self.name = f"w{len(created)}"
            created.append(self.name)

        def close(self, *, kill: bool = False) -> None:
            events.append(f"close:{self.name}:{kill}")

    def fail_thread_pool(*_args, **_kwargs):
        raise RuntimeError("thread pool init failed")

    monkeypatch.setattr(subprocess_exec, "_SingleSubprocessExecutor", FakeWorker)
    monkeypatch.setattr(subprocess_exec, "ThreadPoolExecutor", fail_thread_pool)

    with pytest.raises(RuntimeError, match="thread pool init failed"):
        subprocess_exec.LocalSubprocessActorPool(
            _subprocess_map_payload(
                Identity,
                execution_backend="subprocess_actor",
                actor_number=2,
            ),
            2,
        )

    assert events == ["close:w1:True", "close:w0:True"]


def test_global_subprocess_task_runtime_close_without_kill_does_not_wait_for_executor():
    import duckdb.execution.udf_subprocess as subprocess_exec

    events: list[str] = []

    class FakeWorker:
        def __init__(self, name: str):
            self.name = name

        def cancel_output_grants(self) -> None:
            events.append(f"cancel:{self.name}")

        def close(self, *, kill: bool = False) -> None:
            events.append(f"close:{self.name}:{kill}")

    class FakeExecutor:
        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            events.append(f"shutdown:{wait}:{cancel_futures}")

    idle_wrapper = subprocess_exec._PooledTaskWorker(FakeWorker("idle"))
    active_wrapper = subprocess_exec._PooledTaskWorker(FakeWorker("active"))
    runtime = subprocess_exec._GlobalSubprocessTaskRuntime.__new__(subprocess_exec._GlobalSubprocessTaskRuntime)
    runtime.executor = FakeExecutor()
    runtime.cond = threading.Condition()
    runtime.closed = False
    runtime.total_workers = 2

    pool = subprocess_exec._TaskWorkerPool.__new__(subprocess_exec._TaskWorkerPool)
    pool.runtime = runtime
    pool.key = "pool"
    pool.closing = False
    pool.kill_on_release = False
    pool.idle = [idle_wrapper]
    pool._active_wrappers = {active_wrapper}
    pool.active = 1
    pool.total = 2
    runtime.pools = {pool.key: pool}

    runtime.close(kill=False)

    assert events == [
        "cancel:active",
        "shutdown:False:True",
        "close:idle:False",
        "close:active:True",
    ]


def test_udf_executor_close_without_kill_cancels_local_shm_waits_before_waiting(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec

    events: list[str] = []

    class FakeExecutor:
        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            events.append(f"shutdown:{wait}:{cancel_futures}")

    executor = subprocess_exec.UDFExecutor.__new__(subprocess_exec.UDFExecutor)
    executor._closed = False
    executor._budget_wakeup_unregister = None
    executor._actor_pool = None
    executor._task_pool = None
    executor._executor = FakeExecutor()
    executor._workers = []
    executor._task_futures = set()
    executor._task_future_meta = {}
    executor._task_futures_cv = threading.Condition()
    executor._task_futures_lock = executor._task_futures_cv

    def fake_cancel_active_input_leases():
        events.append("cancel-input-leases")

    def fake_wait_for_pending_futures(_timeout_s=None):
        assert events == ["cancel-input-leases", "wake-budget"]
        events.append("wait-futures")
        return True

    executor._cancel_active_input_leases = fake_cancel_active_input_leases
    executor._wait_for_pending_futures = fake_wait_for_pending_futures
    monkeypatch.setattr(
        subprocess_exec,
        "wake_local_shm_ref_budget_waiters",
        lambda: events.append("wake-budget"),
    )

    executor.close(kill=False)

    assert events == [
        "cancel-input-leases",
        "wake-budget",
        "wait-futures",
        "shutdown:True:True",
    ]


def test_subprocess_wakeup_callback_errors_are_reported_on_ready_result_take():
    import duckdb.execution.udf_subprocess as subprocess_exec

    executor = subprocess_exec.UDFExecutor.__new__(subprocess_exec.UDFExecutor)
    executor._wakeup = lambda: (_ for _ in ()).throw(RuntimeError("wakeup failed"))
    executor._queue = deque()
    executor._queue_lock = threading.Lock()
    executor._wakeup_error = None

    executor._notify_wakeup()

    with pytest.raises(RuntimeError, match="wakeup failed"):
        executor.take_ready_result()


def test_subprocess_submit_without_worker_owner_fails_fast():
    import duckdb.execution.udf_subprocess as subprocess_exec

    executor = subprocess_exec.UDFExecutor.__new__(subprocess_exec.UDFExecutor)
    executor._closed = False
    executor._debug_submit_count = 0
    executor._actor_pool = None
    executor._task_pool = None
    executor._task_runtime = None
    executor._pending_lock = threading.Lock()
    executor._pending_batches = 0
    executor._task_futures = set()
    executor._task_futures_cv = threading.Condition()
    executor._task_futures_lock = executor._task_futures_cv
    executor._task_future_meta = {}

    with pytest.raises(RuntimeError, match="not initialized with an actor or task worker owner"):
        executor._submit_async(411, lambda _worker: None)

    assert executor._pending_batches == 0
    assert executor._task_futures == set()


def test_subprocess_ref_bundle_output_stats_include_pending_projected_bytes(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec

    calls = []

    def fake_admit(size, *, projected_output_bytes=0):
        calls.append((int(size), int(projected_output_bytes)))
        return projected_output_bytes == 0

    monkeypatch.setattr(subprocess_exec, "can_admit_local_shm_ref_output_submit", fake_admit)
    monkeypatch.setattr(
        subprocess_exec,
        "local_shm_ref_budget_snapshot",
        lambda: {
            "limit_bytes": 1000,
            "usage_bytes": 600,
            "reserved_bytes": 600,
            "pending_output_bytes": 0,
        },
    )

    def make_output(table):
        return pa.table({"blob": table.column("x")})

    executor = subprocess_exec.UDFExecutor(
        _subprocess_map_payload(
            make_output,
            execution_backend="subprocess_task",
            udf_worker_slots=1,
            produce_ref_bundle_output=True,
            streaming_output_mode="local_shm_ref_bundle",
            output_schema=[
                {
                    "name": "blob",
                    "kind": "duckdb_type",
                    "type": "BLOB",
                    "dtype": "",
                    "shape": [],
                }
            ],
        )
    )
    try:
        estimated = executor._output_budget_estimate(64)
        with executor._pending_lock:
            executor._pending_batches = 3
        stats = executor.stats()
        assert stats["udf_output_budget_available"] == 0
        assert calls[-1] == (estimated, 3 * estimated)
    finally:
        executor.close(kill=True)


def test_subprocess_task_runtime_keeps_cpu_count_worker_cap(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec

    subprocess_exec._shutdown_global_task_runtime()
    monkeypatch.setattr(subprocess_exec.os, "cpu_count", lambda: 1)

    def sleeper(_table):
        import os
        import time

        time.sleep(0.25)
        return pa.table({"pid": [os.getpid()]})

    payload = _subprocess_map_payload(
        sleeper,
        execution_backend="subprocess_task",
        udf_worker_slots=2,
    )
    executor_a = subprocess_exec.UDFExecutor(payload)
    executor_b = subprocess_exec.UDFExecutor(payload)
    try:
        assert len(executor_a._workers) == 0
        assert len(executor_b._workers) == 0
        assert executor_a._task_pool is not None
        assert executor_a._task_pool.pool_size == 2
        assert subprocess_exec._global_task_runtime().stats()["max_workers"] == 1

        _submit_with_admission(executor_a, pa.table({"x": [1]}), submit_id=171)
        _submit_with_admission(executor_b, pa.table({"x": [2]}), submit_id=172)

        observed_stats = _wait_for_runtime_stats(
            subprocess_exec._global_task_runtime(),
            lambda stats: stats["total_workers"] >= 1 and stats["active_workers"] >= 1,
            timeout_s=10.0,
        )

        results = _wait_for_results(executor_a, 1, timeout_s=10.0) + _wait_for_results(
            executor_b,
            1,
            timeout_s=10.0,
        )

        assert {result[1] for result in results} == {171, 172}
        assert observed_stats["total_workers"] == 1
        assert observed_stats["active_workers"] == 1
        assert subprocess_exec._global_task_runtime().stats()["max_workers"] == 1
    finally:
        executor_a.close(kill=True)
        executor_b.close(kill=True)
        subprocess_exec._shutdown_global_task_runtime()


def test_subprocess_task_pool_kill_closes_active_worker(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec

    runtime = subprocess_exec._GlobalSubprocessTaskRuntime()

    class FakeWorker:
        def __init__(self):
            self.close_calls = []
            self._closed = False
            self._broken_error = None

        def close(self, kill=False):
            self.close_calls.append(bool(kill))
            self._closed = True

    fake_worker = FakeWorker()
    pool = subprocess_exec._TaskWorkerPool(runtime, "test-pool", {"execution_backend": "subprocess_task"}, 1)
    monkeypatch.setattr(
        pool,
        "_spawn_worker",
        lambda _worker_idx: subprocess_exec._PooledTaskWorker(fake_worker),
    )
    try:
        pool.acquire_ref()
        wrapper = pool.acquire_worker()

        pool.release_ref(kill=True)

        assert fake_worker.close_calls == [True]
        pool.release_worker(wrapper, reusable=False)
    finally:
        runtime.close(kill=True)


def test_subprocess_task_pool_kill_closes_worker_spawned_during_close(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec

    runtime = subprocess_exec._GlobalSubprocessTaskRuntime()
    spawn_started = threading.Event()
    release_done = threading.Event()

    class FakeWorker:
        def __init__(self):
            self.close_calls = []
            self._closed = False
            self._broken_error = None

        def close(self, kill=False):
            self.close_calls.append(bool(kill))
            self._closed = True

    fake_worker = FakeWorker()
    pool = subprocess_exec._TaskWorkerPool(runtime, "spawn-close-pool", {"execution_backend": "subprocess_task"}, 1)

    def spawn_worker(_worker_idx):
        spawn_started.set()
        assert release_done.wait(timeout=5.0)
        return subprocess_exec._PooledTaskWorker(fake_worker)

    monkeypatch.setattr(pool, "_spawn_worker", spawn_worker)

    acquired: list[object] = []
    errors: list[BaseException] = []

    def acquire_worker():
        try:
            acquired.append(pool.acquire_worker())
        except BaseException as exc:
            errors.append(exc)

    try:
        pool.acquire_ref()
        thread = threading.Thread(target=acquire_worker)
        thread.start()
        assert spawn_started.wait(timeout=5.0)

        pool.release_ref(kill=True)
        release_done.set()
        thread.join(timeout=5.0)

        assert not thread.is_alive()
        assert not acquired
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert fake_worker.close_calls == [True]
        assert runtime.stats()["total_workers"] == 0
    finally:
        release_done.set()
        runtime.close(kill=True)


def test_subprocess_task_shared_payload_pool_keeps_worker_slots_global(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec

    subprocess_exec._shutdown_global_task_runtime()

    def sleeper(table):
        import os
        import time

        time.sleep(0.25)
        return pa.table({"pid": [os.getpid() for _ in range(table.num_rows)]})

    payload = _subprocess_map_payload(
        sleeper,
        execution_backend="subprocess_task",
        udf_worker_slots=1,
    )
    executor_a = subprocess_exec.UDFExecutor(payload)
    executor_b = subprocess_exec.UDFExecutor(payload)
    try:
        assert executor_a._task_pool is executor_b._task_pool
        assert executor_a._task_pool is not None
        assert executor_a._task_pool.pool_size == 1
        assert executor_a._task_pool.ref_count == 2
        assert executor_a._task_pool.pool_size == 1

        first_table = pa.table({"x": [1]})
        second_table = pa.table({"x": [2]})
        _submit_with_admission(executor_a, first_table, submit_id=181)
        second_ready = threading.Event()
        executor_b.register_wakeup(second_ready.set)
        assert executor_b.request_task_admission(second_table.nbytes) is True
        assert executor_b.task_admission_state()["state"] == "requested"

        observed_stats = _wait_for_runtime_stats(
            subprocess_exec._global_task_runtime(),
            lambda stats: stats["active_workers"] >= 1 and stats["total_workers"] >= 1,
            timeout_s=10.0,
        )

        results = _wait_for_results(executor_a, 1, timeout_s=10.0)
        assert second_ready.wait(timeout=5.0)
        assert executor_b.task_admission_state()["state"] == "ready"
        executor_b.submit_with_id(182, second_table)
        results += _wait_for_results(executor_b, 1, timeout_s=10.0)
        pids = {result[2].column("pid").to_pylist()[0] for result in results}

        assert len(results) == 2
        assert observed_stats["active_workers"] == 1
        assert observed_stats["total_workers"] == 1
        assert len(pids) == 1
    finally:
        executor_a.close(kill=True)
        executor_b.close(kill=True)
        subprocess_exec._shutdown_global_task_runtime()


def test_subprocess_actor_number_controls_pool_size(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec

    class Identity:
        def __call__(self, table):
            return table

    payload = _subprocess_map_payload(
        Identity,
        execution_backend="subprocess_actor",
        actor_number=2,
    )
    executor, pool = _make_subprocess_actor_executor(subprocess_exec, payload)
    try:
        assert executor._workers == []
        assert executor._executor is None
        assert executor._actor_pool is pool
        assert pool.pool_size == 2
    finally:
        executor.close()
        pool.shutdown(kill=True)


def test_subprocess_actor_number_controls_parallel_workers(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec
    from duckdb.execution.ref_bundle import SUBMIT_RESULT_MARKER

    class WorkerPID:
        def __call__(self, table):
            import os
            import time

            time.sleep(0.15)
            return pa.table({"pid": [os.getpid() for _ in range(table.num_rows)]})

    payload = _subprocess_map_payload(
        WorkerPID,
        execution_backend="subprocess_actor",
        actor_number=2,
    )
    executor, pool = _make_subprocess_actor_executor(subprocess_exec, payload)
    try:
        assert executor._workers == []
        assert executor._executor is None
        assert executor._actor_pool is pool
        assert pool.pool_size == 2

        _submit_with_admission(executor, pa.table({"x": [1]}), submit_id=201)
        _submit_with_admission(executor, pa.table({"x": [2]}), submit_id=202)
        results = _wait_for_results(executor, 2, timeout_s=10.0)
        _submit_with_admission(executor, pa.table({"x": [3]}), submit_id=203)
        _submit_with_admission(executor, pa.table({"x": [4]}), submit_id=204)
        results += _wait_for_results(executor, 2, timeout_s=10.0)

        assert len(results) == 4
        assert {result[0] for result in results} == {SUBMIT_RESULT_MARKER}
        assert {result[1] for result in results} == {201, 202, 203, 204}
        pids = {result[2].column("pid").to_pylist()[0] for result in results}
        assert len(pids) == 2
    finally:
        executor.close()
        pool.shutdown(kill=True)


def test_subprocess_actor_requires_actor_number():
    from duckdb.execution.udf_subprocess import UDFExecutor

    class Identity:
        def __call__(self, table):
            return table

    with pytest.raises(ValueError, match="payload.actor_number must be a positive integer"):
        UDFExecutor(_subprocess_map_payload(Identity, execution_backend="subprocess_actor"))


def test_subprocess_pool_consumes_local_shm_ref_bundle_without_parent_materialize(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec
    from duckdb.execution.ref_bundle import SUBMIT_RESULT_MARKER, make_local_shm_ref_bundle_result

    def add_one(table):
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    _marker, refs, metadata, names = make_local_shm_ref_bundle_result(pa.table({"x": [1, 2, 3]}))
    executor = subprocess_exec.UDFExecutor(
        _subprocess_map_payload(
            add_one,
            execution_backend="subprocess_task",
            udf_worker_slots=2,
        )
    )
    try:
        _submit_ref_bundle_with_admission(
            executor,
            201,
            refs,
            [(0, 2)],
            metadata,
            names,
        )
        results = _wait_for_results(executor, 1)

        assert len(results) == 1
        wrapped = results[0]
        assert wrapped[0] == SUBMIT_RESULT_MARKER
        assert wrapped[1] == 201
        assert wrapped[2].to_pydict() == {"y": [2, 3]}
    finally:
        refs[0].release()
        executor.close()


def test_subprocess_pool_ref_bundle_retains_local_shm_until_background_submit():
    from multiprocessing import shared_memory

    from duckdb.execution.ref_bundle import SUBMIT_RESULT_MARKER, make_local_shm_ref_bundle_result
    from duckdb.execution.udf_subprocess import UDFExecutor

    def maybe_sleep_add_one(table):
        import time

        import pyarrow as pa

        sleep_ms = table.column("sleep_ms").to_pylist()[0]
        if sleep_ms:
            time.sleep(float(sleep_ms) / 1000.0)
        values = table.column("x").to_pylist()
        return pa.table({"y": [value + 1 for value in values]})

    _marker, refs, metadata, names = make_local_shm_ref_bundle_result(pa.table({"x": [1], "sleep_ms": [0]}))
    shm_name = refs[0].name
    executor = UDFExecutor(
        _subprocess_map_payload(
            maybe_sleep_add_one,
            execution_backend="subprocess_task",
            udf_worker_slots=2,
        )
    )
    try:
        _submit_with_admission(
            executor,
            pa.table({"x": [0], "sleep_ms": [300]}),
            submit_id=301,
        )
        _submit_ref_bundle_with_admission(executor, 302, refs, None, metadata, names)
        del refs
        gc.collect()

        results = _wait_for_results(executor, 2, timeout_s=10.0)

        assert len(results) == 2
        assert {result[0] for result in results} == {SUBMIT_RESULT_MARKER}
        assert {result[1] for result in results} == {301, 302}
        by_id = {result[1]: result[2].to_pydict() for result in results}
        assert by_id[301] == {"y": [1]}
        assert by_id[302] == {"y": [2]}
    finally:
        executor.close(kill=True)
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
        except FileNotFoundError:
            pass
        else:
            try:
                shm.unlink()
            finally:
                shm.close()


def test_subprocess_pool_zero_row_submit_wakeup_sees_no_inflight():
    from duckdb.execution.ref_bundle import SUBMIT_RESULT_MARKER
    from duckdb.execution.udf_subprocess import UDFExecutor

    def identity(table):
        return table

    executor = UDFExecutor(
        _subprocess_map_payload(
            identity,
            execution_backend="subprocess_task",
            udf_worker_slots=2,
        )
    )

    def pending_batches():
        with executor._pending_lock:
            return executor._pending_batches

    wakeup_pending_counts = []
    wakeup_event = threading.Event()

    def record_wakeup():
        wakeup_pending_counts.append(pending_batches())
        wakeup_event.set()

    executor.register_wakeup(record_wakeup)
    try:
        _submit_with_admission(
            executor,
            pa.table({"x": pa.array([], type=pa.int64())}),
            submit_id=301,
        )

        assert wakeup_event.wait(timeout=5.0)

        assert wakeup_pending_counts
        assert wakeup_pending_counts[-1] == 0
        assert pending_batches() == 0
        assert executor.take_ready_result() == (SUBMIT_RESULT_MARKER, 301, None)
        assert executor.take_ready_result() is None
    finally:
        executor.close()


def test_subprocess_admission_holds_worker_slot_until_completed_result_is_consumed():
    from duckdb.execution.ref_bundle import SUBMIT_RESULT_MARKER
    from duckdb.execution.udf_subprocess import UDFExecutor

    def identity(table):
        return table

    executor = UDFExecutor(
        _subprocess_map_payload(
            identity,
            execution_backend="subprocess_task",
            udf_worker_slots=1,
        )
    )
    try:
        assert executor.request_task_admission(8)
        assert executor.task_admission_state()["state"] == "ready"
        executor.submit_with_id(401, pa.table({"x": [1]}))

        assert executor.request_task_admission(8)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with executor._queue_lock:
                if executor._queue:
                    break
            time.sleep(0.005)

        assert executor.task_admission_state()["state"] == "requested"
        first = executor.take_ready_result()
        assert first == (SUBMIT_RESULT_MARKER, 401, pa.table({"x": [1]}))
        assert executor.task_admission_state()["state"] == "ready"

        executor.submit_with_id(402, pa.table({"x": [2]}))
        second = _wait_for_results(executor, 1)[0]
        assert second[0:2] == (SUBMIT_RESULT_MARKER, 402)
        assert second[2].to_pydict() == {"x": [2]}
    finally:
        executor.close()


def test_subprocess_worker_env_does_not_assign_cuda_devices(monkeypatch):
    from duckdb.execution.udf_subprocess import _worker_env_for_pool_index

    payload = {"execution_backend": "subprocess_actor", "gpus": 1.0}
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    first = _worker_env_for_pool_index(payload, 0, 3)
    second = _worker_env_for_pool_index(payload, 1, 3)
    third = _worker_env_for_pool_index(payload, 2, 3)

    assert "CUDA_VISIBLE_DEVICES" not in first
    assert "CUDA_VISIBLE_DEVICES" not in second
    assert "CUDA_VISIBLE_DEVICES" not in third
    assert first["VANE_SUBPROCESS_WORKER_INDEX"] == "0"
    assert first["VANE_SUBPROCESS_POOL_SIZE"] == "3"


def test_subprocess_task_worker_env_defaults_omp_num_threads_from_assigned_cpus(monkeypatch):
    from duckdb.execution.udf_subprocess import _worker_env_for_pool_index

    thread_env_names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "RAYON_NUM_THREADS",
        "VANE_TORCH_NUM_THREADS",
        "VANE_TORCH_INTEROP_THREADS",
    )
    for name in thread_env_names:
        monkeypatch.delenv(name, raising=False)

    env = _worker_env_for_pool_index(
        {"execution_backend": "subprocess_task", "cpus": 2.75},
        0,
        4,
    )

    assert env["OMP_NUM_THREADS"] == "2"
    assert env["OPENBLAS_NUM_THREADS"] == "2"
    assert env["MKL_NUM_THREADS"] == "2"
    assert env["NUMEXPR_NUM_THREADS"] == "2"
    assert env["VECLIB_MAXIMUM_THREADS"] == "2"
    assert env["RAYON_NUM_THREADS"] == "2"
    assert env["VANE_TORCH_NUM_THREADS"] == "2"
    assert env["VANE_TORCH_INTEROP_THREADS"] == "1"


def test_subprocess_task_worker_env_defaults_omp_num_threads_to_one_without_cpus(monkeypatch):
    from duckdb.execution.udf_subprocess import _worker_env_for_pool_index

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)

    env = _worker_env_for_pool_index(
        {"execution_backend": "subprocess_task"},
        0,
        4,
    )

    assert env["OMP_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["NUMEXPR_NUM_THREADS"] == "1"


def test_subprocess_task_worker_env_defaults_fractional_cpus_to_one(monkeypatch):
    from duckdb.execution.udf_subprocess import _worker_env_for_pool_index

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)

    env = _worker_env_for_pool_index(
        {"execution_backend": "subprocess_task", "cpus": 0.25},
        0,
        4,
    )

    assert env["OMP_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["NUMEXPR_NUM_THREADS"] == "1"


def test_subprocess_task_worker_env_preserves_explicit_omp_num_threads(monkeypatch):
    from duckdb.execution.udf_subprocess import _worker_env_for_pool_index

    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VANE_TORCH_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)

    env = _worker_env_for_pool_index(
        {"execution_backend": "subprocess_task"},
        0,
        4,
    )

    assert "OMP_NUM_THREADS" not in env
    assert env["OPENBLAS_NUM_THREADS"] == "8"
    assert env["MKL_NUM_THREADS"] == "8"
    assert env["NUMEXPR_NUM_THREADS"] == "8"
    assert env["VANE_TORCH_NUM_THREADS"] == "8"


def test_subprocess_actor_worker_env_defaults_omp_num_threads(monkeypatch):
    from duckdb.execution.udf_subprocess import _worker_env_for_pool_index

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)

    env = _worker_env_for_pool_index(
        {"execution_backend": "subprocess_actor", "cpus": 3.0},
        0,
        4,
    )

    assert env["OMP_NUM_THREADS"] == "3"
    assert env["OPENBLAS_NUM_THREADS"] == "3"
    assert env["MKL_NUM_THREADS"] == "3"
    assert env["NUMEXPR_NUM_THREADS"] == "3"


def test_udf_threading_configures_loaded_torch_once(monkeypatch):
    import sys

    from duckdb.execution import udf_threading

    class FakeTorch:
        def __init__(self) -> None:
            self.num_threads = 1
            self.interop_threads = 2

        def get_num_threads(self) -> int:
            return self.num_threads

        def get_num_interop_threads(self) -> int:
            return self.interop_threads

        def set_num_threads(self, value: int) -> None:
            self.num_threads = int(value)

        def set_num_interop_threads(self, value: int) -> None:
            self.interop_threads = int(value)

    fake_torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("VANE_TORCH_NUM_THREADS", "3")
    monkeypatch.setenv("VANE_TORCH_INTEROP_THREADS", "1")
    monkeypatch.setattr(udf_threading, "_TORCH_THREADS_CONFIGURED", False)

    assert udf_threading.configure_loaded_torch_threads() is True
    assert fake_torch.num_threads == 3
    assert fake_torch.interop_threads == 1
    assert udf_threading.configure_loaded_torch_threads() is False


def test_udf_threading_accepts_matching_preconfigured_torch_threads(monkeypatch):
    import sys

    from duckdb.execution import udf_threading

    class FakeTorch:
        @staticmethod
        def get_num_threads() -> int:
            return 4

        @staticmethod
        def get_num_interop_threads() -> int:
            return 1

        @staticmethod
        def set_num_threads(_value: int) -> None:
            raise AssertionError("matching intra-op threads must not be reset")

        @staticmethod
        def set_num_interop_threads(_value: int) -> None:
            raise AssertionError("matching inter-op threads must not be reset")

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setenv("VANE_TORCH_NUM_THREADS", "4")
    monkeypatch.setenv("VANE_TORCH_INTEROP_THREADS", "1")
    monkeypatch.setattr(udf_threading, "_TORCH_THREADS_CONFIGURED", False)

    assert udf_threading.configure_loaded_torch_threads() is True


def test_ray_native_actor_thread_policy_leaves_torch_defaults_untouched(monkeypatch):
    import sys

    from duckdb.execution import udf_threading

    class FakeTorch:
        num_threads = 1
        interop_threads = 18

        @classmethod
        def get_num_threads(cls):
            return cls.num_threads

        @classmethod
        def get_num_interop_threads(cls):
            return cls.interop_threads

        @staticmethod
        def set_num_threads(_value):
            raise AssertionError("ray-native policy must not set Torch intra-op threads")

        @staticmethod
        def set_num_interop_threads(_value):
            raise AssertionError("ray-native policy must not set Torch inter-op threads")

    payload = {"ray_actor_thread_policy": "ray_native", "cpus": 0.0}
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(udf_threading, "_TORCH_THREADS_CONFIGURED", False)

    assert udf_threading.ray_actor_thread_env(payload) == {}
    assert udf_threading.configure_ray_actor_loaded_torch_threads(payload) is False
    assert FakeTorch.num_threads == 1
    assert FakeTorch.interop_threads == 18


def test_ray_actor_thread_policy_defaults_to_ray_native(monkeypatch):
    from duckdb.execution import udf_threading

    monkeypatch.delenv(udf_threading.RAY_ACTOR_THREAD_POLICY_ENV, raising=False)

    assert udf_threading.ray_actor_thread_policy() == "ray_native"
    assert udf_threading.ray_actor_thread_env({"cpus": 1.0}) == {}


def test_ray_actor_thread_policy_reads_actor_runtime_marker(monkeypatch):
    from duckdb.execution import udf_threading

    monkeypatch.setenv(udf_threading.RAY_ACTOR_THREAD_POLICY_ENV, "ray_native")

    assert udf_threading.ray_actor_uses_native_threads() is True


def test_ray_actor_thread_policy_rejects_unknown_value():
    from duckdb.execution import udf_threading

    with pytest.raises(ValueError, match="Ray actor thread policy"):
        udf_threading.ray_actor_thread_policy({"ray_actor_thread_policy": "different"})


def test_subprocess_worker_receives_worker_env_without_cuda_assignment(monkeypatch):
    from duckdb.execution.udf_subprocess import UDFExecutor

    def worker_env(_table):
        import os

        return pa.table(
            {
                "worker_index": [os.environ.get("VANE_SUBPROCESS_WORKER_INDEX")],
                "pool_size": [os.environ.get("VANE_SUBPROCESS_POOL_SIZE")],
                "cuda_visible_devices": [os.environ.get("CUDA_VISIBLE_DEVICES")],
            }
        )

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    executor = UDFExecutor(_subprocess_map_payload(worker_env, udf_worker_slots=1, gpus=1.0))
    try:
        _submit_with_admission(executor, pa.table({"x": [1]}))
        output = _wait_for_results(executor, 1, timeout_s=10.0)[0]

        assert output is not None
        assert output.to_pydict() == {
            "worker_index": ["0"],
            "pool_size": ["1"],
            "cuda_visible_devices": [None],
        }
    finally:
        executor.close()


def test_unified_subprocess_does_not_import_ray(monkeypatch):
    import builtins

    from duckdb.execution.unified_executor import build_unified_executor

    original_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ray" or name.startswith("ray."):
            raise AssertionError("subprocess backend imported ray")
        return original_import(name, globals, locals, fromlist, level)

    def identity(table):
        return table

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    executor = build_unified_executor(_subprocess_map_payload(identity))
    try:
        _submit_with_admission(executor, pa.table({"x": [1, 2]}))
        output = _wait_for_results(executor, 1, timeout_s=10.0)[0]
        assert output is not None
        assert output.to_pydict() == {"x": [1, 2]}
    finally:
        executor.close()


def test_subprocess_direct_ref_bundle_submit_without_admission_is_rejected():
    from duckdb.execution.unified_executor import build_unified_executor

    def identity(table):
        return table

    executor = build_unified_executor(_subprocess_map_payload(identity))
    try:
        with pytest.raises(RuntimeError, match="pregranted admission lease"):
            executor.submit_ref_bundle([], [], [], [])
    finally:
        executor.close()


class _FakeStreamRemote:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def _ray_task_executor(*, stream_result="stream-ref", ref_stream_result="ref-stream-ref"):
    from duckdb.execution.udf_ray import RayTaskUDFExecutor

    run_stream = _FakeStreamRemote(stream_result)
    run_ref_stream = _FakeStreamRemote(ref_stream_result)
    executor = RayTaskUDFExecutor(
        {
            "call_mode": "map_batches",
            "execution_backend": "ray_task",
            "query_id": "query-submit",
            "stage_id": "stage:query-submit:node:1:udf",
            "produce_ray_block_stream": True,
            "udf_output_target_max_bytes": 128 * 1024**2,
            "udf_task_input_max_bytes": 128 * 1024**2,
        },
        run_stream,
        run_ref_stream,
    )
    return executor, run_stream, run_ref_stream


def test_ray_task_submit_with_id_uses_generator_remote_with_pregranted_lease(monkeypatch):
    import duckdb.execution.udf_ray as udf_ray

    captured = {}
    submitted_nodes = []

    class _Pregranted:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.generator = kwargs["submitter"](dict(kwargs["admission"].lease))

    monkeypatch.setattr(udf_ray, "TaskLeaseObjectRefGenerator", _Pregranted)
    monkeypatch.setattr(
        udf_ray,
        "_submit_ray_remote",
        lambda remote_fn, node_id, *args, **kwargs: (
            submitted_nodes.append(str(node_id)),
            remote_fn.remote(*args, **kwargs),
        )[1],
    )
    executor, run_stream, _ = _ray_task_executor(stream_result="generator")
    table = pa.table({"x": [1, 2]})
    lease = {
        "query_id": "query-submit",
        "stage_id": "stage:query-submit:node:1:udf",
        "lease_id": "lease-42",
        "attempt_id": "attempt-42",
        "node_id": "node-a",
        "execution_slot_id": "ray_task:stage:query-submit:node:1:udf:lease-42",
        "output_window_bytes": 256 * 1024**2,
    }
    admission = types.SimpleNamespace(driver=object(), request_id="request-42", lease=lease)
    monkeypatch.setattr(executor, "_take_task_admission", lambda: admission)

    result = executor.submit_with_id(42, table)

    assert isinstance(result, _Pregranted)
    assert result.generator == "generator"
    assert captured["admission"] is admission
    assert submitted_nodes == ["node-a"]
    assert len(run_stream.calls) == 1

    args, kwargs = run_stream.calls[0]
    assert kwargs == {}
    assert args[0]["execution_backend"] == "ray_task"
    assert args[0]["task_lease_id"] == "lease-42"
    assert args[0]["attempt_id"] == "attempt-42"
    assert len(args[1]) == 1
    assert args[1][0].to_pydict() == {"x": [1, 2]}


def test_ray_task_submit_ref_bundle_with_id_uses_pregranted_lease(monkeypatch):
    import duckdb.execution.udf_ray as udf_ray

    captured = {}
    submitted_nodes = []

    class _Pregranted:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.generator = kwargs["submitter"](dict(kwargs["admission"].lease))

    monkeypatch.setattr(udf_ray, "TaskLeaseObjectRefGenerator", _Pregranted)
    monkeypatch.setattr(
        udf_ray,
        "_submit_ray_remote",
        lambda remote_fn, node_id, *args, **kwargs: (
            submitted_nodes.append(str(node_id)),
            remote_fn.remote(*args, **kwargs),
        )[1],
    )
    executor, _, run_ref_stream = _ray_task_executor(ref_stream_result="ref-generator")
    lease = {
        "query_id": "query-submit",
        "stage_id": "stage:query-submit:node:1:udf",
        "lease_id": "lease-7",
        "attempt_id": "attempt-7",
        "node_id": "node-a",
        "execution_slot_id": "ray_task:stage:query-submit:node:1:udf:lease-7",
        "output_window_bytes": 256 * 1024**2,
    }
    admission = types.SimpleNamespace(driver=object(), request_id="request-7", lease=lease)
    monkeypatch.setattr(executor, "_take_task_admission", lambda: admission)

    result = executor.submit_ref_bundle_with_id(
        7,
        ["block-ref"],
        [(0, 1)],
        [{"num_rows": 1, "size_bytes": 8}],
        ["x"],
    )

    assert isinstance(result, _Pregranted)
    assert result.generator == "ref-generator"
    assert captured["admission"] is admission
    assert submitted_nodes == ["node-a"]
    assert len(run_ref_stream.calls) == 1

    args, kwargs = run_ref_stream.calls[0]
    assert args == ("block-ref",)
    assert kwargs["payload"]["execution_backend"] == "ray_task"
    assert kwargs["payload"]["task_lease_id"] == "lease-7"
    assert kwargs["payload"]["attempt_id"] == "attempt-7"
    assert kwargs["slices"] == [(0, 1)]
    assert kwargs["metadata"] == [{"num_rows": 1, "size_bytes": 8}]
    assert kwargs["names"] == ["x"]


def test_ref_bundle_slices_apply_projection_and_names():
    import duckdb.execution.udf_ray as ray_exec

    block = pa.table({"id": [1, 2, 3], "path": ["a", "b", "c"], "ok": [True, False, True]})

    out = ray_exec._apply_ref_bundle_slices(
        [block],
        [(1, 3)],
        metadata=[{"column_ids": [2, 0]}],
        names=["is_ok", "image_id"],
    )

    assert out.to_pydict() == {"is_ok": [False, True], "image_id": [2, 3]}


def test_callable_cache_reuses_deserialized_callable(monkeypatch):
    import duckdb.execution._common as common
    from duckdb import pickle as duckdb_pickle

    common.clear_udf_callable_cache()
    calls = []

    def _fake_loads(data):
        calls.append(data)
        return object()

    monkeypatch.setattr(duckdb_pickle, "loads", _fake_loads)

    payload = {
        "function_pickle": b"function-a",
        "call_mode": "map",
        "scalar_udf_type": "native",
        "execution_backend": "ray_task",
    }
    first = common.load_udf_from_payload_cached(payload, max_entries=8)
    second = common.load_udf_from_payload_cached(payload, max_entries=8)

    assert first is second
    assert calls == [b"function-a"]
    assert common.udf_callable_cache_stats() == {
        "python_udf_callable_cache_hit": 1,
        "python_udf_callable_cache_miss": 1,
        "python_udf_callable_cache_bypass": 0,
    }
    common.clear_udf_callable_cache()


def test_local_shm_budget_manager_input_lease_is_diagnostic_only(monkeypatch):
    from duckdb.execution import ref_bundle

    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 1024)

    class FakeRef:
        size = 400

        def __init__(self):
            self.released = False

        def release(self):
            if not self.released:
                self.released = True
                manager.release_allocation(self.size, name="input-a")

    fake_ref = FakeRef()
    manager.acquire_allocation(400, name="input-a")
    lease_id = manager.create_input_lease([fake_ref], 400, name="lease-a", submit_id=7)

    snapshot = manager.snapshot()
    assert snapshot["limit_bytes"] == 1024
    assert snapshot["allocated_bytes"] == 400
    assert snapshot["input_lease_bytes"] == 400
    assert snapshot["usage_bytes"] == 400
    assert snapshot["available_bytes"] == 624

    released = manager.consume_input_lease(lease_id, name="lease-a")
    assert released == 400
    assert fake_ref.released is True
    snapshot = manager.snapshot()
    assert snapshot["allocated_bytes"] == 0
    assert snapshot["input_lease_bytes"] == 0
    assert snapshot["output_credit_bytes"] == 400
    assert snapshot["usage_bytes"] == 400
    assert snapshot["available_bytes"] == 624

    manager.cancel_input_lease(lease_id, name="lease-a")
    snapshot = manager.snapshot()
    assert snapshot["output_credit_bytes"] == 0
    assert snapshot["usage_bytes"] == 0
    assert snapshot["available_bytes"] == 1024


def test_local_shm_budget_manager_defers_shared_input_ref_release_until_last_lease():
    from duckdb.execution import ref_bundle

    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 1024)

    class FakeRef:
        name = "psm-shared"
        size = 400

        def __init__(self):
            self.release_count = 0

        def release(self):
            self.release_count += 1
            manager.release_allocation(self.size, name=self.name)

    shared_ref = FakeRef()
    manager.acquire_allocation(400, name=shared_ref.name)
    lease_a = manager.create_input_lease([shared_ref], 400, name="lease-a")
    lease_b = manager.create_input_lease([shared_ref], 400, name="lease-b")

    manager.consume_input_lease(lease_a, name="lease-a")
    snapshot = manager.snapshot()
    assert shared_ref.release_count == 0
    assert snapshot["allocated_bytes"] == 400
    assert snapshot["input_lease_bytes"] == 400

    manager.consume_input_lease(lease_b, name="lease-b")
    snapshot = manager.snapshot()
    assert shared_ref.release_count == 1
    assert snapshot["allocated_bytes"] == 0


def test_local_shm_input_ack_does_not_fallback_to_destructive_release():
    from duckdb.execution import ref_bundle

    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 1024)

    class BrokenBudgetRef:
        name = "psm-broken"
        size = 400

        def __init__(self):
            self.release_count = 0

        def release_budget(self):
            raise RuntimeError("budget release failed")

        def release(self):
            self.release_count += 1

    ref = BrokenBudgetRef()
    manager.acquire_allocation(ref.size, name=ref.name)
    lease_id = manager.create_input_lease([ref], ref.size, name="lease")

    with pytest.raises(RuntimeError, match="budget release failed"):
        manager.consume_input_lease(lease_id, name="lease")

    assert ref.release_count == 0


def test_local_shm_input_ack_releases_budget_without_invalidating_descriptor(monkeypatch):
    from duckdb.execution import ref_bundle

    monkeypatch.setenv("VANE_LOCAL_SHM_REF_BUDGET_BYTES", "1g")
    marker, refs, metadata, names = ref_bundle.make_local_shm_ref_bundle_result(pa.table({"x": [1, 2, 3]}))
    assert marker == ref_bundle.REF_BUNDLE_RESULT_MARKER
    ref = refs[0]
    try:
        assert getattr(ref, "_closed", False) is False
        assert getattr(ref, "_budget_bytes", 0) > 0

        lease_id = ref_bundle.create_local_shm_input_lease(
            refs,
            name="input",
            reserve_output_credit=False,
        )
        ref_bundle.consume_local_shm_input_lease(lease_id, name="input")

        assert getattr(ref, "_closed", False) is False
        assert getattr(ref, "_budget_bytes", 0) == 0
        assert ref_bundle.materialize_ref_bundle(refs, None, metadata, names).to_pydict() == {"x": [1, 2, 3]}
    finally:
        for ref in refs:
            ref.release()


def test_local_shm_budget_manager_reserves_consumed_input_for_matching_output():
    import threading

    from duckdb.execution import ref_bundle

    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 1000)

    class BudgetedRef:
        size = 300

        def release(self):
            manager.release_allocation(self.size, name="input")

    manager.acquire_allocation(650, name="backlog")
    manager.acquire_allocation(300, name="input")
    lease_id = manager.create_input_lease((BudgetedRef(),), 300, name="input")

    manager.consume_input_lease(lease_id, name="input")
    snapshot = manager.snapshot()
    assert snapshot["allocated_bytes"] == 650
    assert snapshot["output_credit_bytes"] == 300
    assert snapshot["usage_bytes"] == 950

    producer_grants = []

    def request_producer_grant():
        producer_grants.append(manager.request_output_grant(100, name="producer", priority="producer"))

    producer_thread = threading.Thread(target=request_producer_grant)
    producer_thread.start()
    threading.Event().wait(0.05)
    assert producer_grants == []

    consumer_grant = manager.request_output_grant(
        200,
        name="consumer",
        priority="consumer",
        input_lease_id=lease_id,
    )
    assert isinstance(consumer_grant, int)
    snapshot = manager.snapshot()
    assert snapshot["allocated_bytes"] == 650
    assert snapshot["output_grant_bytes"] == 200
    assert snapshot["output_credit_bytes"] == 0
    assert snapshot["usage_bytes"] == 850

    producer_thread.join(timeout=2)
    assert len(producer_grants) == 1


def test_local_shm_budget_manager_matching_output_grant_waits_for_other_input_credits():
    import threading

    from duckdb.execution import ref_bundle

    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 1000)

    class BudgetedRef:
        size = 400

        def release(self):
            manager.release_allocation(self.size, name="input")

    manager.acquire_allocation(400, name="input-a")
    manager.acquire_allocation(400, name="input-b")
    lease_a = manager.create_input_lease((BudgetedRef(),), 400, name="input-a")
    lease_b = manager.create_input_lease((BudgetedRef(),), 400, name="input-b")

    manager.consume_input_lease(lease_a, name="input-a")
    manager.consume_input_lease(lease_b, name="input-b")
    snapshot = manager.snapshot()
    assert snapshot["allocated_bytes"] == 0
    assert snapshot["output_credit_bytes"] == 800

    cancel_event = threading.Event()
    grants = []
    errors = []

    def request_matching_output():
        try:
            grants.append(
                manager.request_output_grant(
                    700,
                    name="consumer-a",
                    priority="consumer",
                    input_lease_id=lease_a,
                    cancel_event=cancel_event,
                )
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=request_matching_output)
    thread.start()
    thread.join(timeout=0.2)
    assert grants == []
    assert thread.is_alive()

    manager.cancel_input_lease(lease_b, name="input-b")
    thread.join(timeout=2)

    assert len(grants) == 1
    assert errors == []
    snapshot = manager.snapshot()
    assert snapshot["usage_bytes"] == 700
    assert snapshot["usage_bytes"] <= snapshot["limit_bytes"]
    manager.release_output_grant(grants[0], name="consumer-a")
    manager.cancel_input_lease(lease_a, name="input-a")


def test_local_shm_budget_manager_cancel_releases_consumed_input_output_credit():
    from duckdb.execution import ref_bundle

    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 1000)

    class BudgetedRef:
        size = 300

        def release(self):
            manager.release_allocation(self.size, name="input")

    manager.acquire_allocation(500, name="backlog")
    manager.acquire_allocation(300, name="input")
    lease_id = manager.create_input_lease((BudgetedRef(),), 300, name="input")

    manager.consume_input_lease(lease_id, name="input")
    assert manager.snapshot()["output_credit_bytes"] == 300

    manager.cancel_input_lease(lease_id, name="input-error")
    snapshot = manager.snapshot()
    assert snapshot["allocated_bytes"] == 500
    assert snapshot["output_credit_bytes"] == 0
    assert snapshot["usage_bytes"] == 500


def test_local_shm_budget_manager_can_consume_input_without_output_credit():
    from duckdb.execution import ref_bundle

    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 1000)

    class BudgetedRef:
        size = 300

        def release(self):
            manager.release_allocation(self.size, name="input")

    manager.acquire_allocation(300, name="input")
    lease_id = manager.create_input_lease(
        (BudgetedRef(),),
        300,
        name="legacy-output-input",
        reserve_output_credit=False,
    )

    manager.consume_input_lease(lease_id, name="legacy-output-input")
    snapshot = manager.snapshot()
    assert snapshot["allocated_bytes"] == 0
    assert snapshot["output_credit_bytes"] == 0
    assert snapshot["usage_bytes"] == 0


def test_local_shm_budget_manager_output_grant_converts_to_allocation():
    from duckdb.execution import ref_bundle

    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 1024)
    grant_id = manager.request_output_grant(256, name="grant-a", priority="consumer")

    snapshot = manager.snapshot()
    assert snapshot["allocated_bytes"] == 0
    assert snapshot["output_grant_bytes"] == 256
    assert snapshot["usage_bytes"] == 256

    converted = manager.convert_output_grant_to_allocation(grant_id, name="grant-a")
    assert converted == 256
    snapshot = manager.snapshot()
    assert snapshot["allocated_bytes"] == 256
    assert snapshot["output_grant_bytes"] == 0
    assert snapshot["usage_bytes"] == 256

    manager.release_allocation(256, name="grant-a")
    assert manager.snapshot()["usage_bytes"] == 0


def test_local_shm_budget_manager_output_grant_waits_until_allocation_released():
    import threading

    from duckdb.execution import ref_bundle

    manager = ref_bundle.LocalShmBudgetManager(limit_factory=lambda: 512)
    manager.acquire_allocation(512, name="full")
    acquired = []

    def claim():
        acquired.append(manager.request_output_grant(128, name="grant-wait", priority="producer"))

    thread = threading.Thread(target=claim)
    thread.start()
    threading.Event().wait(0.05)
    assert acquired == []

    manager.release_allocation(512, name="full")
    thread.join(timeout=2)
    assert len(acquired) == 1
    assert isinstance(acquired[0], int)


@pytest.mark.parametrize("raw", ["not-a-byte-size", "-1", "-1g"])
def test_local_shm_budget_invalid_env_raises_without_fallback(monkeypatch, raw):
    from duckdb.execution import ref_bundle

    monkeypatch.setenv("VANE_LOCAL_SHM_REF_BUDGET_BYTES", raw)

    with pytest.raises(ValueError, match="VANE_LOCAL_SHM_REF_BUDGET_BYTES"):
        ref_bundle._make_local_shm_ref_budget_limit_factory()()

    def fail_wakeup():
        raise RuntimeError("budget wake failed")

    unregister = ref_bundle.register_local_shm_ref_budget_wakeup(fail_wakeup)
    try:
        with pytest.raises(RuntimeError, match="budget wakeup callback failed"):
            ref_bundle.wake_local_shm_ref_budget_waiters()
    finally:
        unregister()


def test_local_shm_auto_budget_rejects_missing_capacity_instead_of_ignoring_it(monkeypatch):
    from duckdb.execution import ref_bundle

    monkeypatch.setattr(ref_bundle, "_available_system_memory_bytes", lambda: 16 * ref_bundle._GIB)
    monkeypatch.setattr(ref_bundle, "_available_local_shm_bytes", lambda: 0)

    with pytest.raises(RuntimeError, match="/dev/shm has no available capacity"):
        ref_bundle._auto_local_shm_ref_budget_bytes()


def test_local_ref_bundle_worker_payload_carries_input_lease_id():
    from duckdb.execution.ref_bundle import make_local_ref_bundle_worker_payload, make_local_shm_ref_bundle_result

    marker, refs, metadata, names = make_local_shm_ref_bundle_result(pa.table({"x": [1, 2]}))
    try:
        payload = make_local_ref_bundle_worker_payload(refs, None, metadata, names, input_lease_id=42)
        assert payload is not None
        assert payload["input_lease_id"] == 42
    finally:
        for ref in refs:
            ref.release()


def test_local_shm_descriptor_wrap_balances_resource_tracking_without_global_monkeypatch(monkeypatch):
    from duckdb.execution import ref_bundle

    shared_registers = []
    shared_unregisters = []
    original_register = ref_bundle._resource_tracker.register
    original_unregister = ref_bundle._resource_tracker.unregister

    def record_register(name, rtype):
        if rtype == "shared_memory":
            shared_registers.append(name)
        return original_register(name, rtype)

    def record_unregister(name, rtype):
        if rtype == "shared_memory":
            shared_unregisters.append(name)
        return original_unregister(name, rtype)

    monkeypatch.setattr(ref_bundle._resource_tracker, "register", record_register)
    monkeypatch.setattr(ref_bundle._resource_tracker, "unregister", record_unregister)

    descriptor = ref_bundle.make_local_shm_ref_bundle_descriptor(pa.table({"x": [1, 2, 3]}))
    result = ref_bundle.make_local_shm_ref_bundle_result_from_descriptor(descriptor, block_on_budget=False)
    refs = result[1]
    try:
        assert ref_bundle.materialize_ref_bundle(refs, None, result[2], result[3]).to_pydict() == {"x": [1, 2, 3]}
    finally:
        for ref in refs:
            ref.release()

    assert shared_registers
    assert sorted(shared_registers) == sorted(shared_unregisters)


def test_subprocess_ref_bundle_input_ack_releases_upstream_before_result(monkeypatch):
    from duckdb.execution import ref_bundle
    from duckdb.execution.udf_subprocess import UDFExecutor

    def identity(table):
        return pa.table({"y": table.column("x").to_pylist()})

    monkeypatch.setenv("VANE_LOCAL_SHM_REF_BUDGET_BYTES", "1g")
    marker, refs, metadata, names = ref_bundle.make_local_shm_ref_bundle_result(pa.table({"x": [1, 2, 3]}))

    executor = UDFExecutor(
        _subprocess_map_payload(
            identity,
            produce_ref_bundle_output=True,
            streaming_output_mode="local_shm_ref_bundle",
            udf_worker_slots=1,
        )
    )
    try:
        _submit_ref_bundle_with_admission(executor, 123, refs, None, metadata, names)
        item = _wait_for_results(executor, 1, timeout_s=10.0)[0]
        assert item[0] == ref_bundle.SUBMIT_RESULT_MARKER
        assert item[1] == 123
        result = item[2]
        assert result[0] == ref_bundle.REF_BUNDLE_RESULT_MARKER
        assert all(not getattr(ref, "_closed", False) for ref in refs)
        assert all(getattr(ref, "_budget_bytes", 0) == 0 for ref in refs)
        assert ref_bundle.materialize_ref_bundle(refs, None, metadata, names).to_pydict() == {"x": [1, 2, 3]}
    finally:
        executor.close(kill=True)
        for ref in refs:
            ref.release()


def test_subprocess_ref_bundle_consumer_can_start_when_output_budget_full(monkeypatch):
    from duckdb.execution import ref_bundle
    from duckdb.execution.udf_subprocess import UDFExecutor

    def identity(table):
        return pa.table({"y": table.column("x").to_pylist()})

    monkeypatch.setenv("VANE_LOCAL_SHM_REF_BUDGET_BYTES", "2m")
    payload_bytes = b"x" * (1024 * 1024)
    marker, refs, metadata, names = ref_bundle.make_local_shm_ref_bundle_result(
        pa.table({"x": [payload_bytes, payload_bytes, payload_bytes, payload_bytes]})
    )
    before = ref_bundle.local_shm_ref_budget_snapshot()
    assert before["reserved_bytes"] > before["limit_bytes"]

    executor = UDFExecutor(
        _subprocess_map_payload(
            identity,
            produce_ref_bundle_output=True,
            streaming_output_mode="local_shm_ref_bundle",
            udf_worker_slots=1,
            output_schema=[{"kind": "tensor", "dtype": "UINT8", "shape": [1024 * 1024]}],
        )
    )
    try:
        _submit_ref_bundle_with_admission(executor, 77, refs, None, metadata, names)
        item = _wait_for_results(executor, 1, timeout_s=2.0)[0]
        assert item[0] == ref_bundle.SUBMIT_RESULT_MARKER
        assert item[1] == 77
        result = item[2]
        assert result[0] == ref_bundle.REF_BUNDLE_RESULT_MARKER
    finally:
        executor.close(kill=True)
        for ref in refs:
            ref.release()


class _FakeControlSocket:
    def __init__(self, recv_payload: bytes = b"") -> None:
        self._recv_payload = bytearray(recv_payload)
        self.sent = bytearray()
        self.closed = False

    def sendall(self, data: bytes) -> None:
        self.sent.extend(data)

    def recv(self, size: int) -> bytes:
        if not self._recv_payload:
            return b""
        chunk = self._recv_payload[:size]
        del self._recv_payload[:size]
        return bytes(chunk)

    def close(self) -> None:
        self.closed = True


def _decode_control_messages(data: bytes, header) -> list[tuple[int, bytes]]:
    messages = []
    offset = 0
    while offset < len(data):
        msg_type, payload_len = header.unpack(data[offset : offset + header.size])
        offset += header.size
        payload = data[offset : offset + payload_len]
        offset += payload_len
        messages.append((msg_type, payload))
    return messages


def test_single_subprocess_close_releases_active_output_grants(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec
    from duckdb import pickle as duckdb_pickle
    from duckdb.execution import ref_bundle

    monkeypatch.setenv("VANE_LOCAL_SHM_REF_BUDGET_BYTES", "1g")
    before = ref_bundle.local_shm_ref_budget_snapshot()["output_grant_bytes"]

    executor = object.__new__(subprocess_exec._SingleSubprocessExecutor)
    executor._queue = deque()
    executor._finished_submitting = False
    executor._closed = False
    executor._broken_error = None
    executor._pending_batches = 0
    executor._wakeup = None
    executor._wakeup_error = None
    executor._ref_bundle_output = True
    executor._worker_env = {}
    executor._active_input_leases = set()
    executor._active_input_leases_lock = threading.Lock()
    executor._active_output_grants = set()
    executor._active_output_grants_lock = threading.Lock()
    executor._output_grant_cancel_event = threading.Event()
    executor._payload_shm = None
    executor._data_shm = None
    executor._sock = _FakeControlSocket()
    executor._proc = None

    payload = duckdb_pickle.dumps({"request_id": 1, "size_bytes": 4096, "priority": "consumer"})
    assert executor._handle_submit_control_message(subprocess_exec._MSG_OUTPUT_GRANT_REQUEST, payload)
    assert ref_bundle.local_shm_ref_budget_snapshot()["output_grant_bytes"] >= before + 4096

    executor.close(kill=True)

    assert ref_bundle.local_shm_ref_budget_snapshot()["output_grant_bytes"] == before


def test_subprocess_worker_releases_output_grant_when_descriptor_creation_fails(monkeypatch):
    import duckdb.execution.udf_subprocess_worker as worker
    from duckdb import pickle as duckdb_pickle

    class FakeExecutor:
        def submit(self, _table):
            return None

        def drain_outputs(self):
            return [pa.table({"y": [1]})]

    def fail_descriptor(_table, *, grant_id=None):
        assert grant_id is None
        raise RuntimeError("descriptor failed")

    grant_payload = duckdb_pickle.dumps({"request_id": 7, "grant_id": 99})
    recv_payload = worker._HEADER.pack(worker._MSG_OUTPUT_GRANT_GRANTED, len(grant_payload)) + grant_payload
    sock = _FakeControlSocket(recv_payload)
    monkeypatch.setattr(worker, "make_local_shm_ref_bundle_descriptor", fail_descriptor)

    with pytest.raises(RuntimeError, match="descriptor failed"):
        worker._execute_submit(
            FakeExecutor(),
            pa.table({"x": [1]}),
            data_shm=None,
            produce_ref_bundle_output=True,
            sock=sock,
            submit_count=7,
        )

    messages = _decode_control_messages(bytes(sock.sent), worker._HEADER)
    release_type = getattr(worker, "_MSG_OUTPUT_GRANT_RELEASE", 0x0F)
    assert [msg_type for msg_type, _ in messages] == [worker._MSG_OUTPUT_GRANT_REQUEST, release_type]
    release_payload = duckdb_pickle.loads(messages[1][1])
    assert release_payload == {"grant_id": 99}


def test_subprocess_worker_ref_bundle_output_preserves_runtime_output_blocks(monkeypatch):
    import duckdb.execution.udf_subprocess_worker as worker
    from duckdb import pickle as duckdb_pickle

    class FakeExecutor:
        def submit(self, _table):
            return None

        def drain_outputs(self):
            return [
                pa.table({"payload": [b"a" * 64]}),
                pa.table({"payload": [b"b" * 64]}),
            ]

    grant_payload = duckdb_pickle.dumps({"request_id": 11, "grant_id": 101})
    recv_payload = worker._HEADER.pack(worker._MSG_OUTPUT_GRANT_GRANTED, len(grant_payload)) + grant_payload
    sock = _FakeControlSocket(recv_payload)

    _data_shm, msg_type, payload = worker._execute_submit(
        FakeExecutor(),
        pa.table({"x": [1, 2]}),
        data_shm=None,
        produce_ref_bundle_output=True,
        sock=sock,
        submit_count=11,
    )

    assert msg_type == worker._MSG_REF_BUNDLE_RESULT
    descriptor = duckdb_pickle.loads(payload)
    assert len(descriptor["block_refs"]) == 2
    assert [meta["num_rows"] for meta in descriptor["metadata"]] == [1, 1]
    assert [meta["size_bytes"] for meta in descriptor["metadata"]] == [
        pa.table({"payload": [b"a" * 64]}).nbytes,
        pa.table({"payload": [b"b" * 64]}).nbytes,
    ]
    assert descriptor["grant_id"] == 101

    messages = _decode_control_messages(bytes(sock.sent), worker._HEADER)
    assert [msg_type for msg_type, _ in messages] == [worker._MSG_OUTPUT_GRANT_REQUEST]
    request_payload = duckdb_pickle.loads(messages[0][1])
    assert request_payload["request_id"] == 11
    assert request_payload["size_bytes"] >= sum(meta["ipc_size_bytes"] for meta in descriptor["metadata"])


def test_subprocess_task_submit_flushes_compute_tail_before_drain(monkeypatch):
    import duckdb.execution.udf_subprocess_worker as worker
    from duckdb import pickle as duckdb_pickle

    created = []

    class FakeRuntimeExecutor:
        def __init__(self, _payload, *, cache_callable=False):
            self.finished = False
            self.input_rows = 0
            created.append(self)

        def submit(self, table):
            self.input_rows = table.num_rows

        def finished_submitting(self):
            self.finished = True

        def drain_outputs(self):
            if not self.finished:
                raise RuntimeError("compute tail was not flushed")
            return [pa.table({"rows": [self.input_rows]})]

    def make_descriptor(table, *, grant_id=None):
        assert grant_id is None
        return {
            "block_refs": [{"provider": "local_shm", "shm_name": "fake-shm", "ipc_size_bytes": 1}],
            "metadata": [{"rows": table.column("rows").to_pylist(), "num_rows": table.num_rows, "ipc_size_bytes": 1}],
            "names": list(table.schema.names),
        }

    grant_payload = duckdb_pickle.dumps({"request_id": 3, "grant_id": 88})
    recv_payload = worker._HEADER.pack(worker._MSG_OUTPUT_GRANT_GRANTED, len(grant_payload)) + grant_payload
    sock = _FakeControlSocket(recv_payload)
    monkeypatch.setattr(worker, "RuntimeUDFExecutor", FakeRuntimeExecutor)
    monkeypatch.setattr(worker, "make_local_shm_ref_bundle_descriptor", make_descriptor)
    monkeypatch.setattr(worker, "configure_loaded_torch_threads", lambda: None)

    _data_shm, msg_type, payload = worker._execute_task_submit(
        {"call_mode": "map_batches"},
        pa.table({"x": [1, 2, 3]}),
        data_shm=None,
        produce_ref_bundle_output=True,
        submit_count=3,
        log_submit=False,
        sock=sock,
    )

    assert msg_type == worker._MSG_REF_BUNDLE_RESULT
    descriptor = duckdb_pickle.loads(payload)
    assert descriptor["grant_id"] == 88
    assert descriptor["metadata"] == [{"rows": [3], "num_rows": 1, "ipc_size_bytes": 1}]
    assert descriptor["names"] == ["rows"]
    assert created and created[0].finished


@pytest.mark.parametrize(
    "payload",
    [
        {"call_mode": "map_batches_rows"},
        {"call_mode": "map_batches_rows", "scalar_arg_count": 0},
        {"call_mode": "map_batches_rows", "scalar_arg_count": "not-an-int"},
    ],
)
def test_subprocess_worker_row_preserving_ref_bundle_requires_scalar_arg_count(payload):
    import duckdb.execution.udf_subprocess_worker as worker

    with pytest.raises(RuntimeError, match="map_batches_rows requires scalar_arg_count > 0"):
        worker.split_row_preserving_input(payload, pa.table({"arg": [1], "passthrough": [2]}))


def test_subprocess_executor_close_escalates_when_pending_futures_do_not_finish(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec

    class FakeWorker:
        def __init__(self) -> None:
            self.cancelled = False
            self.closed: list[bool] = []

        def cancel_output_grants(self) -> None:
            self.cancelled = True

        def close(self, *, kill: bool = False) -> None:
            self.closed.append(kill)

    class FakeThreadPool:
        def __init__(self) -> None:
            self.shutdown_calls = []

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            self.shutdown_calls.append({"wait": wait, "cancel_futures": cancel_futures})

    monkeypatch.setenv("VANE_UDF_SUBPROCESS_SHUTDOWN_GRACE_S", "0.01")
    executor = object.__new__(subprocess_exec.UDFExecutor)
    executor._closed = False
    executor._budget_wakeup_unregister = None
    executor._task_futures_cv = threading.Condition()
    executor._task_futures = {Future()}
    executor._task_future_meta = {}
    executor._active_input_leases = set()
    executor._active_input_leases_lock = threading.Lock()
    executor._actor_pool = None
    executor._task_pool = None
    thread_pool = FakeThreadPool()
    executor._executor = thread_pool
    executor._workers = [FakeWorker()]

    done = threading.Event()
    thread = threading.Thread(target=lambda: (executor.close(kill=False), done.set()), daemon=True)
    thread.start()

    assert done.wait(timeout=1.0)
    assert executor._workers[0].cancelled
    assert executor._workers[0].closed == [True]
    assert thread_pool.shutdown_calls == [{"wait": False, "cancel_futures": True}]


def test_ray_actor_init_has_default_timeout(monkeypatch):
    import duckdb.execution.udf_ray_actor_pool as actor_pool_mod

    monkeypatch.delenv("VANE_QUERY_DEADLINE_EPOCH_S", raising=False)
    monkeypatch.delenv("VANE_RAY_OBJECT_GET_TIMEOUT_S", raising=False)
    monkeypatch.delenv("VANE_RAY_ACTOR_INIT_TIMEOUT_S", raising=False)

    assert actor_pool_mod._actor_init_timeout_s() > 0.0


def test_subprocess_close_kill_releases_active_local_shm_leases(monkeypatch):
    import duckdb.execution.udf_subprocess as subprocess_exec
    from duckdb.execution import ref_bundle

    class SlowIdentity:
        def __call__(self, table):
            import time

            time.sleep(5)
            return pa.table({"y": table.column("x").to_pylist()})

    monkeypatch.setenv("VANE_LOCAL_SHM_REF_BUDGET_BYTES", "1g")
    marker, refs, metadata, names = ref_bundle.make_local_shm_ref_bundle_result(pa.table({"x": [1, 2]}))
    before = ref_bundle.local_shm_ref_budget_snapshot()["reserved_bytes"]
    payload = _subprocess_map_payload(
        SlowIdentity,
        execution_backend="subprocess_actor",
        actor_number=2,
        produce_ref_bundle_output=True,
        streaming_output_mode="local_shm_ref_bundle",
    )
    executor, pool = _make_subprocess_actor_executor(subprocess_exec, payload)
    try:
        _submit_ref_bundle_with_admission(executor, 88, refs, None, metadata, names)
        executor.close(kill=True)
    finally:
        pool.shutdown(kill=True)

    after = ref_bundle.local_shm_ref_budget_snapshot()["reserved_bytes"]
    assert after < before
    for ref in refs:
        ref.release()


def test_subprocess_stats_expose_local_shm_budget_keys():
    from duckdb.execution.udf_subprocess import UDFExecutor

    def identity(table):
        return pa.table({"y": table.column("x").to_pylist()})

    executor = UDFExecutor(
        _subprocess_map_payload(
            identity,
            produce_ref_bundle_output=True,
            streaming_output_mode="local_shm_ref_bundle",
            udf_worker_slots=1,
        )
    )
    try:
        stats = executor.stats()
        for key in (
            "udf_local_shm_budget_limit_bytes",
            "udf_local_shm_allocated_bytes",
            "udf_local_shm_output_grant_bytes",
            "udf_local_shm_output_credit_bytes",
            "udf_local_shm_input_lease_bytes",
            "udf_local_shm_available_bytes",
            "udf_local_shm_active_input_leases",
            "udf_local_shm_active_output_credits",
            "udf_local_shm_waiting_output_grants",
            "udf_local_shm_input_consumed_count",
            "udf_local_shm_refs_released_by_input_ack",
            "udf_local_shm_oversized_output_grants",
        ):
            assert key in stats
            assert isinstance(stats[key], int)
    finally:
        executor.close(kill=True)
