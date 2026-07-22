# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import concurrent.futures
import sys
import threading
import time
import types
import warnings
from collections import deque
from decimal import Decimal

import pyarrow as pa
import pytest


def test_vllm_control_rpc_timeout_is_configurable(monkeypatch):
    import duckdb.execution.vllm as vllm

    monkeypatch.delenv("VANE_VLLM_CONTROL_RPC_TIMEOUT_S", raising=False)
    assert vllm._vllm_control_rpc_timeout_s() == 30.0

    monkeypatch.setenv("VANE_VLLM_CONTROL_RPC_TIMEOUT_S", "7.5")
    observed = {}

    def resolve(ref, *, timeout, honor_query_deadline):
        observed.update(timeout=timeout, honor_query_deadline=honor_query_deadline)
        return ref

    monkeypatch.setattr(vllm, "resolve_object_refs_blocking", resolve)
    assert vllm._resolve_vllm_control_ref("control-ref") == "control-ref"
    assert observed == {"timeout": 7.5, "honor_query_deadline": False}


@pytest.mark.parametrize("configured", ["not-a-number", "0", "-1", "nan", "inf"])
def test_vllm_control_rpc_timeout_falls_back_for_invalid_values(monkeypatch, configured):
    import duckdb.execution.vllm as vllm

    monkeypatch.setenv("VANE_VLLM_CONTROL_RPC_TIMEOUT_S", configured)
    assert vllm._vllm_control_rpc_timeout_s() == 30.0


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": True}, "batch_size"),
        ({"batch_size": 1.5}, "batch_size"),
        ({"prefix_match_threshold": float("nan")}, "prefix_match_threshold"),
        ({"prefix_match_threshold": float("inf")}, "prefix_match_threshold"),
        ({"prefix_match_threshold": True}, "prefix_match_threshold"),
        ({"prefix_match_threshold": Decimal("1.01")}, "prefix_match_threshold"),
        ({"gpus_per_actor": 0}, "gpus_per_actor"),
        ({"gpus_per_actor": 1.5}, "gpus_per_actor"),
        ({"gpus_per_actor": True}, "gpus_per_actor"),
        ({"gpus_per_actor": Decimal("NaN")}, "gpus_per_actor"),
        ({"gpus_per_actor": Decimal("1.5")}, "gpus_per_actor"),
        ({"concurrency": True}, "concurrency"),
        ({"do_prefix_routing": "false"}, "do_prefix_routing"),
        ({"engine_init_timeout_s": Decimal("-0.1")}, "engine_init_timeout_s"),
        ({"engine_init_timeout_s": True}, "engine_init_timeout_s"),
    ],
)
def test_vllm_numeric_options_are_strict(options, message):
    from duckdb.execution.vllm import normalize_options

    with pytest.raises(ValueError, match=message):
        normalize_options(options)


def test_vllm_fractional_gpu_option_is_preserved():
    from duckdb.execution.vllm import normalize_options

    assert normalize_options({"gpus_per_actor": 0.25})["gpus_per_actor"] == pytest.approx(0.25)


def test_vllm_decimal_options_are_normalized_to_floats():
    from duckdb.execution.vllm import normalize_options

    normalized = normalize_options(
        {
            "gpus_per_actor": Decimal("0.25"),
            "prefix_match_threshold": Decimal("0.33"),
            "engine_init_timeout_s": Decimal("1.5"),
        }
    )

    assert normalized["gpus_per_actor"] == pytest.approx(0.25)
    assert normalized["prefix_match_threshold"] == pytest.approx(0.33)
    assert normalized["engine_init_timeout_s"] == pytest.approx(1.5)
    assert type(normalized["gpus_per_actor"]) is float
    assert type(normalized["prefix_match_threshold"]) is float
    assert type(normalized["engine_init_timeout_s"]) is float


@pytest.mark.parametrize(
    "name",
    [
        "use_ray",
        "use_threading",
        "require_ray_worker",
        "ray_worker_only",
        "_force_background_thread",
    ],
)
def test_vllm_execution_boolean_options_are_strict(name):
    from duckdb.execution.vllm import normalize_options

    with pytest.raises(ValueError, match=rf"vllm {name} must be a boolean"):
        normalize_options({name: "false"})

    assert normalize_options({name: False})[name] is False


def test_native_descriptor_forces_background_loop_inside_ray_actor(monkeypatch):
    import duckdb.execution.vllm as vllm_executor
    from vane.ai.providers.vllm import VLLMPrompterDescriptor

    fake_vllm = types.ModuleType("vllm")

    class SamplingParams:
        pass

    fake_vllm.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(vllm_executor.LocalVLLMExecutor, "_detect_ray_actor", staticmethod(lambda: True))

    def fake_run_event_loop(executor):
        executor.loop = object()
        executor.loop_ready.set()

    monkeypatch.setattr(vllm_executor.LocalVLLMExecutor, "_run_event_loop", fake_run_event_loop)

    options = VLLMPrompterDescriptor(vllm_options={"use_threading": False}).build_physical_vllm_options()
    executor = vllm_executor.build_executor("test-model", options)

    assert executor._ray_actor_mode is False
    assert executor.use_threading is True
    assert executor.loop_ready.is_set()


def test_ray_actor_releases_only_terminal_per_executor_state():
    from duckdb.execution.vllm import RayLocalVLLMExecutor

    executor = RayLocalVLLMExecutor.__new__(RayLocalVLLMExecutor)
    executor.llm = None
    executor.on_error = "raise"
    executor.completed_tasks = deque()
    executor.error_message = None
    executor._shutdown_called = False
    executor._finished_submitting = False
    executor.running_task_count = 0
    executor.task_count_lock = threading.Lock()
    executor._result_cv = threading.Condition(threading.RLock())
    rows = pa.table({"x": [1]})
    executor._per_executor_deques = {"executor": deque([(None, rows, "reservation")])}
    executor._per_executor_running_task_count = {"executor": 0}
    executor._per_executor_finished = {"executor"}
    executor._per_executor_request_ids = {"executor": set()}
    executor._per_executor_tasks = {"executor": set()}
    executor._per_executor_errors = {}
    executor._per_executor_aborted = set()
    executor._per_executor_waiters = {}
    executor._per_executor_abort_wait_required = set()
    executor._per_executor_terminal_wait_observed = set()

    assert executor.release_executor("executor") is False
    assert executor.take_ready_result("executor") == ([None], rows, "reservation")
    assert executor.release_executor("executor") is True

    executor._per_executor_deques["aborted"] = deque()
    executor._per_executor_running_task_count["aborted"] = 0
    executor._per_executor_request_ids["aborted"] = set()
    executor._per_executor_tasks["aborted"] = set()
    asyncio.run(executor.abort_executor("aborted"))
    assert "aborted" in executor._per_executor_aborted
    assert executor.release_executor("aborted") is True


def test_ray_actor_abort_waiter_does_not_depend_on_default_thread_pool_capacity():
    from duckdb.execution.vllm import RayLocalVLLMExecutor

    executor = RayLocalVLLMExecutor.__new__(RayLocalVLLMExecutor)
    executor.llm = None
    executor.completed_tasks = deque()
    executor.error_message = None
    executor._shutdown_called = False
    executor._finished_submitting = False
    executor.running_task_count = 0
    executor.task_count_lock = threading.Lock()
    executor._result_cv = threading.Condition(threading.RLock())
    executor._per_executor_deques = {"executor": deque()}
    executor._per_executor_running_task_count = {"executor": 0}
    executor._per_executor_finished = set()
    executor._per_executor_request_ids = {"executor": set()}
    executor._per_executor_tasks = {"executor": set()}
    executor._per_executor_errors = {}
    executor._per_executor_aborted = set()
    executor._per_executor_waiters = {}
    executor._per_executor_abort_wait_required = set()
    executor._per_executor_terminal_wait_observed = set()

    async def run_scenario():
        loop = asyncio.get_running_loop()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(pool)
        pool_occupied = threading.Event()
        release_pool = threading.Event()
        waiter = None

        def occupy_only_worker():
            pool_occupied.set()
            release_pool.wait()

        blocker = asyncio.create_task(asyncio.to_thread(occupy_only_worker))
        try:
            while not pool_occupied.is_set():
                await asyncio.sleep(0)

            waiter = asyncio.create_task(executor.wait_for_result("executor"))
            while executor._per_executor_waiters.get("executor", 0) == 0:
                await asyncio.sleep(0)

            await asyncio.wait_for(executor.abort_executor("executor", wait_expected=True), timeout=2.0)
            assert await asyncio.wait_for(waiter, timeout=2.0) is False
        finally:
            release_pool.set()
            if waiter is not None and not waiter.done():
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)
            await blocker

    asyncio.run(run_scenario())


def test_ray_actor_abort_wait_uses_control_rpc_timeout(monkeypatch):
    import duckdb.execution.vllm as vllm

    executor = vllm.RayLocalVLLMExecutor.__new__(vllm.RayLocalVLLMExecutor)
    executor.llm = None
    executor.completed_tasks = deque()
    executor.error_message = None
    executor._shutdown_called = False
    executor._finished_submitting = False
    executor.running_task_count = 0
    executor.task_count_lock = threading.Lock()
    executor._result_cv = threading.Condition(threading.RLock())
    executor._per_executor_deques = {"executor": deque()}
    executor._per_executor_running_task_count = {"executor": 0}
    executor._per_executor_finished = set()
    executor._per_executor_request_ids = {"executor": set()}
    executor._per_executor_tasks = {"executor": set()}
    executor._per_executor_errors = {}
    executor._per_executor_aborted = set()
    executor._per_executor_waiters = {"executor": 1}
    executor._per_executor_abort_wait_required = set()
    executor._per_executor_terminal_wait_observed = set()

    clock = iter((100.0, 106.0, 111.0))
    sleep_calls = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setenv("VANE_VLLM_CONTROL_RPC_TIMEOUT_S", "10")
    monkeypatch.setattr(vllm, "time", types.SimpleNamespace(monotonic=lambda: next(clock)))
    monkeypatch.setattr(vllm.asyncio, "sleep", fake_sleep)

    with pytest.raises(RuntimeError, match="abort waiter did not acknowledge termination"):
        asyncio.run(executor.abort_executor("executor", wait_expected=True))

    assert sleep_calls == [0.01]


def test_ray_actor_abort_installs_tombstone_before_awaiting_engine_abort():
    from duckdb.execution.vllm import RayLocalVLLMExecutor

    abort_started = asyncio.Event()
    allow_abort = asyncio.Event()

    class Engine:
        async def abort(self, _request_id):
            abort_started.set()
            await allow_abort.wait()

        async def generate(self, *_args, **_kwargs):
            yield types.SimpleNamespace(outputs=[types.SimpleNamespace(text="late")])

    executor = RayLocalVLLMExecutor.__new__(RayLocalVLLMExecutor)
    executor.llm = Engine()
    executor.on_error = "raise"
    executor.sampling_params = object()
    executor.generate_args = {}
    executor.counter = 0
    executor.counter_lock = threading.Lock()
    executor.completed_tasks = deque()
    executor.error_message = None
    executor._shutdown_called = False
    executor._finished_submitting = False
    executor.running_task_count = 0
    executor.task_count_lock = threading.Lock()
    executor._result_cv = threading.Condition(threading.RLock())
    executor._ray_actor_mode = True
    executor.engine_error_message = None
    executor._per_executor_deques = {"executor": deque()}
    executor._per_executor_running_task_count = {"executor": 0}
    executor._per_executor_finished = set()
    executor._per_executor_request_ids = {"executor": {"old-request"}}
    executor._per_executor_tasks = {"executor": set()}
    executor._per_executor_errors = {}
    executor._per_executor_aborted = set()
    executor._per_executor_waiters = {}
    executor._per_executor_abort_wait_required = set()
    executor._per_executor_terminal_wait_observed = set()

    async def run_scenario():
        abort_task = asyncio.create_task(executor.abort_executor("executor"))
        await abort_started.wait()
        tombstone_installed = "executor" in executor._per_executor_aborted
        late_error = None
        try:
            await executor.submit_async(
                ["late"],
                pa.table({"id": [1]}),
                "executor",
                "late-reservation",
            )
        except RuntimeError as exc:
            late_error = exc
        finally:
            allow_abort.set()
            await abort_task
            await asyncio.sleep(0)
        return tombstone_installed, late_error

    tombstone_installed, late_error = asyncio.run(run_scenario())

    assert tombstone_installed is True
    assert late_error is not None
    assert "already finished" in str(late_error)
    assert executor.running_task_count == 0
    assert "executor" not in executor._per_executor_deques
    assert "executor" not in executor._per_executor_running_task_count
    assert "executor" not in executor._per_executor_request_ids
    assert "executor" not in executor._per_executor_tasks


def test_prefix_router_serializes_global_reservations_and_releases_exactly_once():
    from duckdb.execution.vllm import PrefixRouter

    router = PrefixRouter([object(), object()], load_balance_threshold=0)
    executor_ids = [f"executor-{index}" for index in range(32)]
    for executor_id in executor_ids:
        router.report_start(executor_id)

    def reserve(executor_id):
        return router.route_and_reserve(None, 1, executor_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        reservations = list(pool.map(reserve, executor_ids))

    assert sum(router.inflight) == len(reservations)
    assert abs(router.inflight[0] - router.inflight[1]) <= 1
    for index, reservation in enumerate(reservations):
        reservation_id = reservation["reservation_id"]
        operation_id = f"release-{index}"
        if index % 2:
            first = router.complete_once(reservation_id, 1, operation_id)
            replay = router.complete_once(reservation_id, 1, operation_id)
        else:
            first = router.rollback_once(reservation_id, 1, operation_id)
            replay = router.rollback_once(reservation_id, 1, operation_id)
        assert first["released"] == 1
        assert replay["released"] == 1
        assert replay["replayed"] is True
    assert router.inflight == [0, 0]


def test_prefix_router_keeps_affinity_until_load_threshold_is_exceeded():
    from duckdb.execution.vllm import PrefixRouter

    router = PrefixRouter([object(), object()], load_balance_threshold=1)
    router.report_start("executor")
    first = router.route_and_reserve("shared-prefix", 4, "executor")
    other = router.route_and_reserve("other-prefix", 1, "executor")
    migrated = router.route_and_reserve("shared-prefix", 1, "executor")

    assert first["actor_idx"] == 0
    assert other["actor_idx"] == 1
    assert migrated["actor_idx"] == 1
    assert migrated["route_reason"] == "load_balance"


class _Ref:
    def __init__(self, value=None, *, ready=True):
        self._future = concurrent.futures.Future()
        if ready:
            self._future.set_result(value)

    def future(self):
        return self._future

    def resolve(self):
        return self._future.result(timeout=1)

    def set_result(self, value):
        self._future.set_result(value)


class _RemoteMethod:
    def __init__(self, function, *, raw_ref=False):
        self._function = function
        self._raw_ref = raw_ref

    def remote(self, *args, **kwargs):
        result = self._function(*args, **kwargs)
        return result if self._raw_ref else _Ref(result)


class _RemoteProxy:
    def __init__(self, target):
        self._target = target

    def __getattr__(self, name):
        return _RemoteMethod(getattr(self._target, name))


class _FakeVLLMActor:
    def __init__(self):
        self.submissions = []
        self.results = deque()
        self.wait_refs = deque()
        self.released = []
        self.aborted = []
        self.finished = []
        self.submit_async = _RemoteMethod(self._submit)
        self.wait_for_result = _RemoteMethod(self._wait, raw_ref=True)
        self.take_ready_result = _RemoteMethod(self._take)
        self.finished_executor = _RemoteMethod(self._finish)
        self.release_executor = _RemoteMethod(self._release)
        self.abort_executor = _RemoteMethod(self._abort)

    def _submit(self, prompts, rows, executor_id, reservation_id):
        self.submissions.append((list(prompts), rows, executor_id, reservation_id))

    def _wait(self, _executor_id):
        ref = _Ref(ready=False)
        self.wait_refs.append(ref)
        return ref

    def _take(self, _executor_id):
        return self.results.popleft()

    def _finish(self, executor_id):
        self.finished.append(executor_id)

    def _release(self, executor_id):
        self.released.append(executor_id)
        return True

    def _abort(self, executor_id, _wait_expected):
        self.aborted.append(executor_id)

    def publish(self, outputs, rows, reservation_id):
        self.results.append((outputs, rows, reservation_id))
        self.wait_refs.popleft().set_result(True)


def test_remote_reservation_rpc_does_not_block_submit_and_serializes_shutdown(monkeypatch):
    import duckdb.execution.vllm as vllm

    complete_entered = threading.Event()
    allow_complete = threading.Event()
    shutdown_reported = threading.Event()
    release_executor_entered = threading.Event()

    class SlowCompleteRouter(vllm.PrefixRouter):
        def complete_once(self, reservation_id, count, operation_id):
            complete_entered.set()
            if not allow_complete.wait(2):
                raise RuntimeError("test timed out waiting to release complete RPC")
            return super().complete_once(reservation_id, count, operation_id)

        def report_completion(self, executor_id):
            result = super().report_completion(executor_id)
            shutdown_reported.set()
            return result

        def release_executor_once(self, executor_id, operation_id):
            release_executor_entered.set()
            return super().release_executor_once(executor_id, operation_id)

    actor = _FakeVLLMActor()
    router = SlowCompleteRouter([actor], load_balance_threshold=0)

    class Owner:
        router_actor = _RemoteProxy(router)
        llm_actors = [actor]

        @staticmethod
        def shutdown():
            return None

    monkeypatch.setattr(vllm, "resolve_object_refs_blocking", lambda ref, **_kwargs: ref.resolve())
    executor = vllm.RemoteVLLMExecutor(Owner(), pool_name="pool")
    executor.submit(None, ["first"], pa.table({"id": [1]}))
    first_reservation = actor.submissions[0][3]
    submit_finished = threading.Event()

    def submit_second():
        executor.submit(None, ["second"], pa.table({"id": [2]}))
        submit_finished.set()

    shutdown_future = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        complete_future = pool.submit(executor._complete_reservation, first_reservation, 1)
        assert complete_entered.wait(1)
        submit_future = pool.submit(submit_second)
        submit_was_not_blocked = submit_finished.wait(1)
        if submit_was_not_blocked:
            shutdown_future = pool.submit(executor.shutdown)
            shutdown_was_reported = shutdown_reported.wait(1)
            release_started_during_complete = release_executor_entered.wait(0.1) if shutdown_was_reported else False
        else:
            shutdown_was_reported = False
            release_started_during_complete = False
        allow_complete.set()
        complete_future.result(timeout=2)
        submit_future.result(timeout=2)
        if shutdown_future is not None:
            shutdown_future.result(timeout=2)

    if shutdown_future is None:
        executor.shutdown()

    assert submit_was_not_blocked, "submit waited for a concurrent reservation-release RPC"
    assert shutdown_was_reported, "shutdown did not reach terminal reservation release"
    assert not release_started_during_complete, "terminal release raced an in-progress completion"
    assert release_executor_entered.is_set()
    assert router.inflight == [0]
    assert executor._reservations == {}
    assert executor._inflight_per_actor == [0]
    assert executor._released_outstanding_inflight is True


def test_remote_executor_uses_router_reservation_and_one_shot_wakeup(monkeypatch):
    import duckdb.execution.vllm as vllm

    actors = [_FakeVLLMActor(), _FakeVLLMActor()]
    router = vllm.PrefixRouter(actors, load_balance_threshold=32)

    class Owner:
        router_actor = _RemoteProxy(router)
        llm_actors = actors
        shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1

    monkeypatch.setattr(vllm, "resolve_object_refs_blocking", lambda ref, **_kwargs: ref.resolve())
    executor = vllm.RemoteVLLMExecutor(Owner(), pool_name="pool")
    rows = pa.table({"id": [1, 2]})
    first_wakeup = threading.Event()
    assert executor.register_wakeup_callback(first_wakeup.set) is True

    executor.submit("prefix", ["a", "b"], rows)
    assert first_wakeup.wait(1)
    assert actors[0].submissions
    reservation_id = actors[0].submissions[0][3]
    assert router.inflight == [2, 0]

    # Drain the already-ready submit acknowledgement, then arm for actor data.
    assert executor.take_ready_result() is None
    result_wakeup = threading.Event()
    assert executor.register_wakeup_callback(result_wakeup.set) is True
    actors[0].publish(["out-a", "out-b"], rows, reservation_id)
    assert result_wakeup.wait(1)
    assert executor.register_wakeup_callback(lambda: None) is False

    assert executor.take_ready_result() == (["out-a", "out-b"], rows)
    executor.finished_submitting()
    assert executor.all_tasks_finished() is True
    assert router.inflight == [0, 0]
    assert actors[0].released and actors[1].released
    executor.shutdown()
    executor.shutdown()
    assert executor._actors_owner.shutdown_calls == 1


def test_remote_executor_rejects_actor_result_without_reservation_id(monkeypatch):
    import duckdb.execution.vllm as vllm

    actor = _FakeVLLMActor()
    router = vllm.PrefixRouter([actor], load_balance_threshold=0)

    class Owner:
        router_actor = _RemoteProxy(router)
        llm_actors = [actor]

        @staticmethod
        def shutdown():
            return None

    monkeypatch.setattr(vllm, "resolve_object_refs_blocking", lambda ref, **_kwargs: ref.resolve())
    executor = vllm.RemoteVLLMExecutor(Owner(), pool_name="pool")
    rows = pa.table({"id": [1]})
    executor.submit(None, ["prompt"], rows)
    actor.results.append((["output"], rows))

    try:
        with pytest.raises(RuntimeError, match="3-item tuple"):
            executor._drain_ready_actor(0, True, actor.wait_refs[0])
    finally:
        executor.shutdown()


def test_remote_success_is_not_rejected_when_terminal_actor_cleanup_keeps_failing(monkeypatch):
    import duckdb.execution.vllm as vllm

    class Actor(_FakeVLLMActor):
        def __init__(self):
            super().__init__()
            self.fail_release = True
            self.release_attempts = 0

        def _release(self, executor_id):
            self.release_attempts += 1
            if self.fail_release:
                raise RuntimeError("persistent release failure")
            return super()._release(executor_id)

    actor = Actor()
    router = vllm.PrefixRouter([actor], load_balance_threshold=0)

    class Owner:
        router_actor = _RemoteProxy(router)
        llm_actors = [actor]
        shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1

    monkeypatch.setattr(vllm, "resolve_object_refs_blocking", lambda ref, **_kwargs: ref.resolve())
    owner = Owner()
    executor = vllm.RemoteVLLMExecutor(owner, pool_name="pool")
    rows = pa.table({"id": [1]})
    executor.submit(None, ["prompt"], rows)
    reservation_id = actor.submissions[0][3]
    actor.publish(["output"], rows, reservation_id)

    assert executor.take_ready_result() == (["output"], rows)
    executor.finished_submitting()
    with pytest.warns(RuntimeWarning, match="persistent release failure"):
        assert executor.all_tasks_finished() is True
    assert executor._error_message is None
    assert executor._finished is True
    assert any("persistent release failure" in error for error in executor._terminal_cleanup_errors)

    with warnings.catch_warnings(record=True) as repeated_cleanup_warnings:
        warnings.simplefilter("always")
        executor.shutdown()
    assert repeated_cleanup_warnings == []
    assert executor._shutdown_complete is False
    assert executor._error_message is None
    assert actor.release_attempts == 2
    assert owner.shutdown_calls == 1

    actor.fail_release = False
    executor.shutdown()
    assert executor._shutdown_complete is True
    assert actor.released == [executor._executor_id]
    assert owner.shutdown_calls == 2


def test_remote_executor_rearms_wait_for_already_buffered_actor_result(monkeypatch):
    import duckdb.execution.vllm as vllm

    class ReadyAwareActor(_FakeVLLMActor):
        def _wait(self, executor_id):
            if self.results:
                return _Ref(True)
            return super()._wait(executor_id)

    actor = ReadyAwareActor()
    router = vllm.PrefixRouter([actor], load_balance_threshold=0)

    class Owner:
        router_actor = _RemoteProxy(router)
        llm_actors = [actor]

        @staticmethod
        def shutdown():
            return None

    monkeypatch.setattr(vllm, "resolve_object_refs_blocking", lambda ref, **_kwargs: ref.resolve())
    executor = vllm.RemoteVLLMExecutor(Owner(), pool_name="pool")
    first_rows = pa.table({"id": [1]})
    second_rows = pa.table({"id": [2]})
    executor.submit(None, ["first"], first_rows)
    executor.submit(None, ["second"], second_rows)
    first_reservation = actor.submissions[0][3]
    second_reservation = actor.submissions[1][3]

    actor.publish(["out-first"], first_rows, first_reservation)
    actor.results.append((["out-second"], second_rows, second_reservation))

    assert executor.take_ready_result() == (["out-first"], first_rows)
    assert executor.take_ready_result() == (["out-second"], second_rows)
    executor.finished_submitting()
    assert executor.all_tasks_finished() is True
    assert router.inflight == [0]


def test_remote_executor_acknowledges_submissions_before_finishing_actor(monkeypatch):
    import duckdb.execution.vllm as vllm

    events = []

    class DeferredSubmitRef(_Ref):
        def __init__(self):
            super().__init__(ready=False)

        def resolve(self):
            events.append("submit-accepted")
            self.set_result(None)
            return None

    class DeferredSubmitMethod:
        @staticmethod
        def remote(*_args, **_kwargs):
            return DeferredSubmitRef()

    class Actor(_FakeVLLMActor):
        def __init__(self):
            super().__init__()
            self.submit_async = DeferredSubmitMethod()

        def _finish(self, executor_id):
            assert events == ["submit-accepted"]
            events.append("executor-finished")
            super()._finish(executor_id)

    actor = Actor()
    router = vllm.PrefixRouter([actor], load_balance_threshold=0)

    class Owner:
        router_actor = _RemoteProxy(router)
        llm_actors = [actor]

        @staticmethod
        def shutdown():
            return None

    monkeypatch.setattr(vllm, "resolve_object_refs_blocking", lambda ref, **_kwargs: ref.resolve())
    executor = vllm.RemoteVLLMExecutor(Owner(), pool_name="pool")
    executor.submit(None, ["prompt"], pa.table({"id": [1]}))

    executor.finished_submitting()

    assert events == ["submit-accepted", "executor-finished"]


def test_remote_executor_consumes_all_submit_acks_before_aborting(monkeypatch):
    import duckdb.execution.vllm as vllm

    events = []

    class FailingSubmitRef(_Ref):
        def __init__(self, name, *, ready):
            super().__init__(ready=ready)
            self.name = name

        def resolve(self):
            events.append(self.name)
            raise RuntimeError(f"{self.name} failed")

    class SubmitMethod:
        def __init__(self):
            self.refs = deque(
                [
                    FailingSubmitRef("first-submit-ack", ready=True),
                    FailingSubmitRef("second-submit-ack", ready=False),
                ]
            )

        def remote(self, *_args, **_kwargs):
            return self.refs.popleft()

    class Actor(_FakeVLLMActor):
        def __init__(self):
            super().__init__()
            self.submit_async = SubmitMethod()

        def _abort(self, executor_id, wait_expected):
            events.append("abort")
            return super()._abort(executor_id, wait_expected)

    actor = Actor()
    router = vllm.PrefixRouter([actor], load_balance_threshold=0)

    class Owner:
        router_actor = _RemoteProxy(router)
        llm_actors = [actor]

        @staticmethod
        def shutdown():
            return None

    monkeypatch.setattr(vllm, "resolve_object_refs_blocking", lambda ref, **_kwargs: ref.resolve())
    executor = vllm.RemoteVLLMExecutor(Owner(), pool_name="pool")
    executor.submit(None, ["first"], pa.table({"id": [1]}))
    executor.submit(None, ["second"], pa.table({"id": [2]}))

    with pytest.raises(RuntimeError, match="first-submit-ack failed"):
        executor.take_ready_result()

    assert events == ["first-submit-ack", "second-submit-ack", "abort"]
    assert actor.aborted == [executor._executor_id]
    assert "second-submit-ack failed" in executor._error_message
    assert executor._submit_refs == {}
    assert executor._reservations == {}
    assert router.inflight == [0]


def test_remote_executor_terminalizes_after_both_route_ack_attempts_fail(monkeypatch):
    import duckdb.execution.vllm as vllm

    class LostRouteAckRef(_Ref):
        def resolve(self):
            raise RuntimeError("route acknowledgement lost")

    class LostRouteAckMethod:
        def __init__(self, router):
            self.router = router

        def remote(self, *args):
            decision = self.router.route_and_reserve_once(*args)
            return LostRouteAckRef(decision)

    actor = _FakeVLLMActor()
    router = vllm.PrefixRouter([actor], load_balance_threshold=0)
    router_proxy = _RemoteProxy(router)
    router_proxy.route_and_reserve_once = LostRouteAckMethod(router)

    class Owner:
        router_actor = router_proxy
        llm_actors = [actor]

        @staticmethod
        def shutdown():
            return None

    monkeypatch.setattr(vllm, "resolve_object_refs_blocking", lambda ref, **_kwargs: ref.resolve())
    executor = vllm.RemoteVLLMExecutor(Owner(), pool_name="pool")

    with pytest.raises(RuntimeError, match="route acknowledgement lost"):
        executor.submit(None, ["prompt"], pa.table({"id": [1]}))

    assert executor._finished is True
    assert "route acknowledgement lost" in executor._error_message
    assert actor.aborted == [executor._executor_id]
    assert executor._executor_id not in router._active_executors
    assert router.inflight == [0]
    assert router._reservations == {}
    assert executor._released_outstanding_inflight is True
    with pytest.raises(RuntimeError, match="no longer accepts submissions"):
        executor.submit(None, ["another"], pa.table({"id": [2]}))


def test_remote_executor_reports_router_completion_when_actor_finish_ack_fails_once(monkeypatch):
    import duckdb.execution.vllm as vllm

    class Actor(_FakeVLLMActor):
        def __init__(self):
            super().__init__()
            self.fail_finish = True

        def _finish(self, executor_id):
            if self.fail_finish:
                self.fail_finish = False
                raise RuntimeError("finish acknowledgement failed")
            super()._finish(executor_id)

    actor = Actor()
    router = vllm.PrefixRouter([actor], load_balance_threshold=0)

    class Owner:
        router_actor = _RemoteProxy(router)
        llm_actors = [actor]

        @staticmethod
        def shutdown():
            return None

    monkeypatch.setattr(vllm, "resolve_object_refs_blocking", lambda ref, **_kwargs: ref.resolve())
    executor = vllm.RemoteVLLMExecutor(Owner(), pool_name="pool")

    with pytest.raises(RuntimeError, match="finish acknowledgement failed"):
        executor.finished_submitting()

    assert executor._router_completion_reported is True
    assert executor._executor_id not in router._active_executors
    executor.shutdown()


def test_remote_shutdown_ignores_expired_query_deadline_for_control_rpcs(monkeypatch):
    import duckdb.execution.vllm as vllm

    monkeypatch.delenv("VANE_QUERY_DEADLINE_EPOCH_S", raising=False)
    monkeypatch.delenv("VANE_RAY_OBJECT_GET_TIMEOUT_S", raising=False)
    actor = _FakeVLLMActor()
    router = vllm.PrefixRouter([actor], load_balance_threshold=0)

    class Owner:
        router_actor = _RemoteProxy(router)
        llm_actors = [actor]
        shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1

    owner = Owner()
    executor = vllm.RemoteVLLMExecutor(owner, pool_name="pool")
    executor_id = executor._executor_id
    assert executor_id in router._active_executors
    executor.submit(None, ["prompt"], pa.table({"id": [1]}))
    assert router.inflight == [1]
    monkeypatch.setenv("VANE_QUERY_DEADLINE_EPOCH_S", str(time.time() - 1.0))

    executor.shutdown()

    assert executor._shutdown_complete is True
    assert executor_id not in router._active_executors
    assert router.inflight == [0]
    assert actor.finished == [executor_id]
    assert actor.aborted == [executor_id]
    assert actor.released == [executor_id]
    assert owner.shutdown_calls == 1


def test_remote_wait_marks_finished_after_releasing_result_condition():
    import duckdb.execution.vllm as vllm

    executor = vllm.RemoteVLLMExecutor.__new__(vllm.RemoteVLLMExecutor)
    executor._error_message = None
    executor._result_buffer = deque()
    executor._finished = False
    executor._finished_submitting_flag = True
    executor._submit_per_actor = []
    executor._results_per_actor = []
    executor._wait_refs_by_actor = []
    executor._result_cv = threading.Condition(threading.Lock())
    executor._drain_queued_submit_refs = lambda: None
    executor._drain_queued_wait_refs = lambda: None
    executor._ensure_remote_wait_refs = lambda: None
    observed = []

    def mark_finished():
        assert executor._result_cv.acquire(blocking=False), "_mark_finished called while _result_cv is held"
        executor._result_cv.release()
        observed.append("finished")
        executor._finished = True

    def record_error(exc):
        executor._error_message = f"{type(exc).__name__}: {exc}"

    executor._mark_finished = mark_finished
    executor._record_error = record_error

    executor.wait_for_result()

    assert observed == ["finished"]


def test_remote_ref_cleanup_does_not_initialize_ray(monkeypatch):
    import duckdb.execution.vllm as vllm

    calls = []
    fake_ray = types.ModuleType("ray")

    def is_initialized():
        calls.append("checked")
        return False

    def cancel(_ref):
        pytest.fail("cleanup must not call ray.cancel before Ray is initialized")

    fake_ray.is_initialized = is_initialized
    fake_ray.cancel = cancel
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    executor = vllm.RemoteVLLMExecutor.__new__(vllm.RemoteVLLMExecutor)
    executor._cancel_refs([object()])

    assert calls == ["checked"]
