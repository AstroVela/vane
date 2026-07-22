# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Real-Ray protocol tests for vLLM routing and executor finalization."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
import time
import uuid
import warnings
from pathlib import Path

import pytest

ray = pytest.importorskip("ray")
pa = pytest.importorskip("pyarrow")

from duckdb.execution.vllm import PrefixRouter, RayLocalVLLMExecutor, RemoteVLLMExecutor


@pytest.fixture(scope="module")
def ray_runtime():
    owned = not ray.is_initialized()
    if owned:
        os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
        pythonpath = os.pathsep.join(
            entry for entry in (str(Path(__file__).parent), os.environ.get("PYTHONPATH", "")) if entry
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Tip: In future versions of Ray, Ray will no longer override accelerator",
            )
            ray.init(
                address="local",
                namespace=f"vane-vllm-protocol-{uuid.uuid4().hex}",
                num_cpus=2,
                num_gpus=0,
                include_dashboard=False,
                log_to_driver=False,
                runtime_env={
                    "env_vars": {
                        "PYTHONPATH": pythonpath,
                        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
                    }
                },
            )
    try:
        yield ray
    finally:
        if owned:
            ray.shutdown()


class _DelayedAsyncVLLMActor:
    """Minimal async actor that makes a terminal-before-submit race deterministic."""

    def __init__(self):
        self._submitted: dict[str, int] = {}
        self._finished: set[str] = set()
        self._events: list[tuple[str, str]] = []

    async def submit_async(self, prompts, _rows, executor_id, _reservation_id):
        await asyncio.sleep(0.15)
        self._submitted[executor_id] = self._submitted.get(executor_id, 0) + len(prompts)
        self._events.append(("submit", executor_id))
        return True

    async def finished_executor(self, executor_id):
        if self._submitted.get(executor_id, 0) == 0:
            raise RuntimeError("finished_executor overtook submit_async")
        self._finished.add(executor_id)
        self._events.append(("finish", executor_id))
        return True

    async def wait_for_result(self, executor_id):
        while executor_id not in self._finished:
            await asyncio.sleep(0.01)
        return False

    def abort_executor(self, _executor_id, _wait_expected):
        return True

    def release_executor(self, _executor_id):
        return True

    def events(self):
        return list(self._events)


class _FailingSubmitBarrierActor:
    """Expose whether abort overtakes another in-flight submit acknowledgement."""

    def __init__(self):
        self._aborted: set[str] = set()
        self._events: list[tuple[str, str]] = []

    async def submit_async(self, prompts, _rows, executor_id, _reservation_id):
        if prompts == ["fail"]:
            await asyncio.sleep(0.05)
            self._events.append(("submit-failed", executor_id))
            raise RuntimeError("submit acknowledgement failed")
        await asyncio.sleep(0.2)
        self._events.append(("submit-settled", executor_id))
        return True

    async def finished_executor(self, executor_id):
        self._events.append(("finish", executor_id))
        return True

    async def wait_for_result(self, executor_id):
        while executor_id not in self._aborted:
            await asyncio.sleep(0.01)
        return False

    async def abort_executor(self, executor_id, _wait_expected):
        self._aborted.add(executor_id)
        self._events.append(("abort", executor_id))
        return True

    def release_executor(self, _executor_id):
        return True

    def events(self):
        return list(self._events)


class _ThreadPoolSaturatedWaitActor:
    """Exercise RayLocalVLLMExecutor waits on a real Ray actor event loop."""

    def __init__(self):
        executor = RayLocalVLLMExecutor.__new__(RayLocalVLLMExecutor)
        executor.llm = None
        executor.completed_tasks = []
        executor.error_message = None
        executor._shutdown_called = False
        executor._finished_submitting = False
        executor.running_task_count = 0
        executor.task_count_lock = threading.Lock()
        executor._result_cv = threading.Condition(threading.RLock())
        executor._per_executor_deques = {"executor": []}
        executor._per_executor_running_task_count = {"executor": 0}
        executor._per_executor_finished = set()
        executor._per_executor_request_ids = {"executor": set()}
        executor._per_executor_tasks = {"executor": set()}
        executor._per_executor_errors = {}
        executor._per_executor_aborted = set()
        executor._per_executor_waiters = {}
        executor._per_executor_abort_wait_required = set()
        executor._per_executor_terminal_wait_observed = set()
        self.executor = executor
        self.pool_started = threading.Event()
        self.release_pool = threading.Event()

    async def saturate_default_pool(self):
        loop = asyncio.get_running_loop()
        loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1))

        def occupy_only_worker():
            self.pool_started.set()
            self.release_pool.wait()

        await asyncio.to_thread(occupy_only_worker)
        return True

    async def pool_is_saturated(self):
        return self.pool_started.is_set()

    async def wait_for_result(self):
        return await self.executor.wait_for_result("executor")

    async def waiter_count(self):
        return self.executor._per_executor_waiters.get("executor", 0)

    async def abort_executor(self):
        await self.executor.abort_executor("executor", wait_expected=True)
        return True

    async def release_default_pool(self):
        self.release_pool.set()


class _RayActorOwner:
    def __init__(self, ray_module, llm_actor, router_actor):
        self._ray = ray_module
        self.llm_actors = [llm_actor]
        self.router_actor = router_actor
        self._closed = False

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        for actor in (*self.llm_actors, self.router_actor):
            try:
                self._ray.kill(actor, no_restart=True)
            except TypeError:
                self._ray.kill(actor)


def test_remote_executor_waits_for_real_ray_submit_ack_before_finish(ray_runtime):
    actor = ray_runtime.remote(_DelayedAsyncVLLMActor).remote()
    router = ray_runtime.remote(PrefixRouter).remote([actor], 0)
    owner = _RayActorOwner(ray_runtime, actor, router)
    executor = RemoteVLLMExecutor(owner)

    try:
        executor.submit(None, ["prompt"], pa.table({"id": [1]}))
        executor.finished_submitting()

        events = ray_runtime.get(actor.events.remote())
        assert [event[0] for event in events] == ["submit", "finish"]
        assert events[0][1] == events[1][1]
    finally:
        executor.shutdown()


def test_remote_executor_waits_for_all_real_ray_submit_acks_before_abort(ray_runtime):
    actor = ray_runtime.remote(max_concurrency=32)(_FailingSubmitBarrierActor).remote()
    router = ray_runtime.remote(PrefixRouter).remote([actor], 0)
    owner = _RayActorOwner(ray_runtime, actor, router)
    executor = RemoteVLLMExecutor(owner)

    try:
        executor.submit(None, ["fail"], pa.table({"id": [1]}))
        executor.submit(None, ["settle"], pa.table({"id": [2]}))

        with pytest.raises(RuntimeError, match="submit acknowledgement failed"):
            executor.finished_submitting()

        events = ray_runtime.get(actor.events.remote())
        assert [event[0] for event in events] == ["submit-failed", "submit-settled", "abort"]
        assert len({event[1] for event in events}) == 1
        assert ray_runtime.get(router.release_executor.remote(executor._executor_id)) == 0
    finally:
        executor.shutdown()


def test_prefix_router_real_ray_rpc_burst_is_atomic_and_idempotent(ray_runtime):
    router = ray_runtime.remote(PrefixRouter).remote(["actor-0", "actor-1"], 0)
    executor_id = f"executor-{uuid.uuid4().hex}"
    try:
        assert ray_runtime.get(router.report_start.remote(executor_id)) is True

        first = ray_runtime.get(router.route_and_reserve_once.remote("shared", 3, executor_id, "route-1"))
        second = ray_runtime.get(router.route_and_reserve_once.remote("shared", 1, executor_id, "route-2"))
        replay = ray_runtime.get(router.route_and_reserve_once.remote("shared", 1, executor_id, "route-2"))

        assert first["actor_idx"] == 0
        assert first["route_reason"] == "initial"
        assert second["actor_idx"] == 1
        assert second["route_reason"] == "load_balance"
        assert replay["reservation_id"] == second["reservation_id"]
        assert replay["replayed"] is True

        burst = ray_runtime.get(
            [router.route_and_reserve_once.remote(None, 1, executor_id, f"burst-{index}") for index in range(16)]
        )
        assert len({decision["reservation_id"] for decision in burst}) == 16
        assert {decision["actor_idx"] for decision in burst} == {0, 1}

        complete = ray_runtime.get(router.complete_once.remote(first["reservation_id"], 2, "complete-first"))
        complete_replay = ray_runtime.get(router.complete_once.remote(first["reservation_id"], 2, "complete-first"))
        assert complete == {"operation_id": "complete-first", "released": 2, "replayed": False}
        assert complete_replay == {"operation_id": "complete-first", "released": 2, "replayed": True}

        release = ray_runtime.get(router.release_executor_once.remote(executor_id, "release-all"))
        release_replay = ray_runtime.get(router.release_executor_once.remote(executor_id, "release-all"))
        assert release["released"] == 18
        assert release_replay == {"operation_id": "release-all", "released": 18, "replayed": True}
        assert ray_runtime.get(router.release_executor.remote(executor_id)) == 0
        assert ray_runtime.get(router.report_completion.remote(executor_id)) is True
        assert ray_runtime.get(router.report_completion.remote(executor_id)) is False
    finally:
        try:
            ray_runtime.kill(router, no_restart=True)
        except TypeError:
            ray_runtime.kill(router)


def test_ray_actor_abort_waiter_survives_saturated_default_thread_pool(ray_runtime):
    actor = ray_runtime.remote(max_concurrency=32)(_ThreadPoolSaturatedWaitActor).remote()
    saturation_ref = actor.saturate_default_pool.remote()
    wait_ref = None
    try:
        deadline = time.monotonic() + 2.0
        while not ray_runtime.get(actor.pool_is_saturated.remote()):
            assert time.monotonic() < deadline, "default thread pool did not saturate"
            time.sleep(0.01)

        wait_ref = actor.wait_for_result.remote()
        deadline = time.monotonic() + 2.0
        while ray_runtime.get(actor.waiter_count.remote()) == 0:
            assert time.monotonic() < deadline, "executor waiter was not registered"
            time.sleep(0.01)

        assert ray_runtime.get(actor.abort_executor.remote(), timeout=2.0) is True
        assert ray_runtime.get(wait_ref, timeout=2.0) is False
    finally:
        try:
            ray_runtime.get(actor.release_default_pool.remote(), timeout=2.0)
            ray_runtime.get(saturation_ref, timeout=2.0)
        finally:
            try:
                ray_runtime.kill(actor, no_restart=True)
            except TypeError:
                ray_runtime.kill(actor)
