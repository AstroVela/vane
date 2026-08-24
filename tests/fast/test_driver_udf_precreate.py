# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap
import threading
import types
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

_QUERY_GENERATION_CAPABILITY = "test-query-generation-capability"
_TEST_QUERY_ID = "query:test"


def _assert_actor_location_runtime_env(env_vars, resource_unit_id):
    from vane.runners.ray import ray_env
    from vane.runners.ray.query_runtime_protocol import (
        RAY_ACTOR_GENERATION_CAPABILITY_ENV,
        RAY_ACTOR_POOL_NONCE_ENV,
        RAY_ACTOR_QUERY_ID_ENV,
        RAY_ACTOR_RESOURCE_UNIT_ID_ENV,
    )

    values = dict(env_vars)
    assert values.pop(RAY_ACTOR_QUERY_ID_ENV) == _TEST_QUERY_ID
    assert values.pop(RAY_ACTOR_RESOURCE_UNIT_ID_ENV) == resource_unit_id
    assert values.pop(RAY_ACTOR_GENERATION_CAPABILITY_ENV) == _QUERY_GENERATION_CAPABILITY
    assert values.pop(RAY_ACTOR_POOL_NONCE_ENV)
    assert values == ray_env.build_session_runtime_env_vars({})


class _FakePlan:
    def __init__(self, nodes):
        self._nodes = nodes
        self.set_calls = []

    def idx(self):
        return "test-plan"

    def collect_udf_nodes(self, conn=None):
        return self._nodes

    def set_udf_actor_handles(self, handles_map, conn=None):
        self.set_calls.append(
            {
                "handles_map": handles_map,
            }
        )


def test_drop_query_fragments_releases_registered_query_resources():
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    dropped_fragments = []
    released_queries = []

    runner._get_plan_runner = lambda: SimpleNamespace(
        drop_query_fragments=lambda query_id: dropped_fragments.append(query_id)
    )
    runner._release_query_resources = lambda query_id, reason, **_kwargs: released_queries.append((query_id, reason))

    runner._drop_query_fragments_sync("q-drop")

    assert dropped_fragments == ["q-drop"]
    assert released_queries == [("q-drop", "query_fragments_dropped")]


def test_drop_resource_query_closes_owned_internal_fte_queries(monkeypatch):
    import vane.runners.ray.fte_fragment_scheduler as fte_scheduler
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    dropped_fragments = []
    runner._get_plan_runner = lambda: SimpleNamespace(drop_query_fragments=dropped_fragments.append)
    runner._release_query_resources = lambda *_args, **_kwargs: None
    monkeypatch.setattr(
        fte_scheduler,
        "fte_execution_query_ids_for_resource",
        lambda query_id: (
            f"{query_id}_orderby_range",
            f"{query_id}_orderby_sample",
        ),
    )

    runner._drop_query_fragments_sync("q-drop-owned")

    assert dropped_fragments == [
        "q-drop-owned_orderby_range",
        "q-drop-owned_orderby_sample",
        "q-drop-owned",
    ]


def test_drop_resource_query_remembers_failed_internal_teardown_after_registry_loss(monkeypatch):
    import vane.runners.ray.fte_fragment_scheduler as fte_scheduler
    from vane.runners.ray.driver import QueryTeardownOwnershipError, RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    runner._query_pending_execution_teardowns = {}
    runner._release_query_resources = lambda *_args, **_kwargs: released_queries.append("released")

    resource_query_id = "q-drop-retry"
    execution_query_id = f"{resource_query_id}_orderby"
    discovered = [(execution_query_id,), ()]
    monkeypatch.setattr(
        fte_scheduler,
        "fte_execution_query_ids_for_resource",
        lambda _query_id: discovered.pop(0),
    )
    monkeypatch.setattr(fte_scheduler, "fte_query_remote_teardown_blockers", lambda _query_id: ())
    monkeypatch.setattr(fte_scheduler, "_drop_fte_registry_for_query", lambda _query_id: None)

    calls: list[str] = []
    failed_once = False

    def drop_query_fragments(query_id):
        nonlocal failed_once
        calls.append(query_id)
        if query_id == execution_query_id and not failed_once:
            failed_once = True
            raise RuntimeError("planned nested cleanup failure")

    released_queries: list[str] = []
    runner._get_plan_runner = lambda: SimpleNamespace(drop_query_fragments=drop_query_fragments)

    with pytest.raises(QueryTeardownOwnershipError, match="pending_execution_queries"):
        runner._drop_query_fragments_sync(resource_query_id)

    assert runner._query_pending_execution_teardowns == {
        resource_query_id: {
            execution_query_id,
            resource_query_id,
        }
    }
    assert released_queries == []

    runner._drop_query_fragments_sync(resource_query_id)

    assert calls == [
        execution_query_id,
        execution_query_id,
        resource_query_id,
    ]
    assert runner._query_pending_execution_teardowns == {}
    assert released_queries == ["released"]


def test_drop_resource_query_keeps_outer_owner_until_internal_blockers_clear(monkeypatch):
    import vane.runners.ray.fte_fragment_scheduler as fte_scheduler
    from vane.runners.ray.driver import QueryTeardownOwnershipError, RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    runner._query_pending_execution_teardowns = {}
    runner._release_query_resources = lambda *_args, **_kwargs: released_queries.append("released")

    resource_query_id = "q-drop-blocked"
    execution_query_id = f"{resource_query_id}_stage"
    monkeypatch.setattr(
        fte_scheduler,
        "fte_execution_query_ids_for_resource",
        lambda _query_id: (execution_query_id,),
    )
    monkeypatch.setattr(fte_scheduler, "_drop_fte_registry_for_query", lambda _query_id: None)

    blocked = True

    def teardown_blockers(query_id):
        if blocked and query_id == execution_query_id:
            return ("active_execution=1",)
        return ()

    monkeypatch.setattr(fte_scheduler, "fte_query_remote_teardown_blockers", teardown_blockers)

    calls: list[str] = []
    released_queries: list[str] = []
    runner._get_plan_runner = lambda: SimpleNamespace(drop_query_fragments=calls.append)

    with pytest.raises(QueryTeardownOwnershipError, match="active_execution=1"):
        runner._drop_query_fragments_sync(resource_query_id)

    assert calls == [execution_query_id]
    assert runner._query_pending_execution_teardowns == {
        resource_query_id: {
            execution_query_id,
            resource_query_id,
        }
    }
    assert released_queries == []

    blocked = False
    runner._drop_query_fragments_sync(resource_query_id)

    assert calls == [
        execution_query_id,
        execution_query_id,
        resource_query_id,
    ]
    assert runner._query_pending_execution_teardowns == {}
    assert released_queries == ["released"]


def test_driver_actor_runtime_shutdown_reaches_plan_runner_once():
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    events: list[str] = []
    runner._driver_shutdown_lock = asyncio.Lock()
    runner._session_lock = threading.RLock()
    runner._plan_runner_lifecycle_lock = threading.RLock()
    runner._driver_shutdown_started = False
    runner._driver_shutdown_complete = False
    runner._datasink_cleanup_tasks = {}
    runner.plan_runner = SimpleNamespace(shutdown=lambda: events.append("workers-shutdown"))

    async def stop_maintenance():
        events.append("maintenance-stopped")

    runner.stop_query_resource_maintenance = stop_maintenance

    async def shutdown_twice():
        await runner_cls._shutdown_runtime(runner)
        await runner_cls._shutdown_runtime(runner)

    asyncio.run(shutdown_twice())

    assert events == ["maintenance-stopped", "workers-shutdown"]
    assert runner._driver_shutdown_started is True
    assert runner._driver_shutdown_complete is True
    assert runner._driver_executors_shutdown is True
    with pytest.raises(RuntimeError, match="executor is shut down"):
        runner_cls._get_driver_lifecycle_executor(runner)


def test_driver_actor_runtime_shutdown_cancels_unbounded_datasink_cleanup_retry(monkeypatch):
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    runner._driver_shutdown_lock = asyncio.Lock()
    runner._session_lock = threading.RLock()
    runner._plan_runner_lifecycle_lock = threading.RLock()
    runner._driver_shutdown_started = False
    runner._driver_shutdown_complete = False
    runner._datasink_cleanup_tasks = {}
    runner.plan_runner = None
    attempts = 0

    async def stop_maintenance():
        return None

    def fail_cleanup(*_args):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("persistent cleanup failure")

    runner.stop_query_resource_maintenance = stop_maintenance
    monkeypatch.setattr(runner_cls, "_teardown_plan_resources", fail_cleanup)

    async def start_retry_and_shutdown():
        lifecycle = SimpleNamespace()
        cleanup_key = ("plan", id(lifecycle))
        cleanup_task = asyncio.create_task(runner_cls._retry_datasink_cleanup_until_complete(runner, "plan", lifecycle))
        runner._datasink_cleanup_tasks[cleanup_key] = cleanup_task
        while attempts == 0:
            await asyncio.sleep(0)
        await asyncio.wait_for(runner_cls._shutdown_runtime(runner), timeout=1.0)
        return cleanup_task

    cleanup_task = asyncio.run(start_retry_and_shutdown())

    assert attempts == 1
    assert cleanup_task.done()
    assert runner._driver_shutdown_complete is True


def test_driver_cleanup_warning_is_bounded_before_actor_transport():
    from vane.runners.ray.driver import _cleanup_warning

    warning = _cleanup_warning(
        "DataSink teardown",
        RuntimeError("diagnostic-head:" + "x" * 10_000 + ":diagnostic-tail"),
    )

    assert len(warning.encode("utf-8")) <= 4 * 1024
    assert warning.startswith("DataSink teardown failed: RuntimeError: diagnostic-head:")
    assert warning.endswith(":diagnostic-tail")


def test_driver_actor_rejects_detach_from_non_owner():
    from vane.runners.ray.driver import BoundedReplayMap, RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = object.__new__(runner_cls)
    runner._client_ids = {"owner-a"}
    runner._detaching_client_ids = set()
    runner._detached_client_results = BoundedReplayMap(capacity=65_536)
    runner._client_detach_lock = asyncio.Lock()
    runner._sessions = {}
    runner._session_lock = threading.RLock()

    with pytest.raises(PermissionError, match="attached client owner"):
        asyncio.run(runner_cls.detach_client(runner, "owner-b"))


def test_driver_client_close_detaches_and_kills_last_job_runtime(monkeypatch):
    from vane.runners.ray import driver as driver_module

    events: list[str] = []
    detach_ref = object()

    class DetachMethod:
        @staticmethod
        def remote(owner_id):
            assert owner_id == "owner-a"
            events.append("detach-rpc")
            return detach_ref

    actor = SimpleNamespace(detach_client=DetachMethod())
    client = object.__new__(driver_module.RayQueryDriverClient)
    client.runner = actor
    client._owner_id = "owner-a"
    client._ray_gcs_address = "gcs-a"
    client._opened_sessions = {}
    client._uncertain_sessions = {}
    client._opening_session_ids = set()
    client._closing_session_ids = set()
    client._closed_session_ids = driver_module.BoundedReplayMap(capacity=65_536)
    client._session_closes_in_progress = set()
    client._session_condition = threading.Condition()
    client._client_closing = False
    client._client_close_in_progress = False

    monkeypatch.setattr(driver_module.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        driver_module.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(gcs_address="gcs-a"),
    )

    def resolve(ref, **kwargs):
        assert ref is detach_ref
        assert kwargs == {"timeout": 300, "honor_query_deadline": False}
        events.append("detach-complete")
        return True

    def kill(target, *, no_restart):
        assert target is actor
        assert no_restart is True
        events.append("kill")

    monkeypatch.setattr(driver_module, "resolve_object_refs_blocking", resolve)
    monkeypatch.setattr(driver_module.ray, "kill", kill)

    client.close()

    assert client.runner is None
    assert events == ["detach-rpc", "detach-complete", "kill"]


def test_precreate_udf_actors_injects_driver_handle_for_ray_task(monkeypatch):
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class

    driver_handle = object()
    runner = SimpleNamespace(
        _driver_handle=driver_handle,
        _issue_query_task_admission_capability=lambda _query_id: _QUERY_GENERATION_CAPABILITY,
        _session_lock=threading.RLock(),
        _plan_session_ids={"test-plan": "session-1"},
        _query_udf_actor_nodes={},
        _query_udf_session_configs={},
        _active_udf_actor_by_unit={},
    )
    query_connection = object()

    plan = _FakePlan(
        [
            {
                "node_id": 7,
                "pool_name": "audio-transcriber",
                "actor_pool_size": 1,
                "gpus": 1.0,
                "payload": {
                    "execution_backend": "ray_task",
                    "gpus": 1.0,
                },
            }
        ]
    )

    created = runner_cls._precreate_udf_actors(
        runner,
        plan,
        SimpleNamespace(query_id="test-plan", units=()),
        query_connection=query_connection,
        session_config={"AWS_ACCESS_KEY_ID": "session-access-key"},
    )

    assert created == []
    assert plan.set_calls == [
        {
            "handles_map": {
                "7": {
                    "query_driver_handle": driver_handle,
                    "query_generation_capability": _QUERY_GENERATION_CAPABILITY,
                    "session_config": {
                        "AWS_ACCESS_KEY_ID": "session-access-key",
                    },
                }
            },
        }
    ]


def test_precreate_udf_actors_preserves_zero_physical_node_id():
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    driver_handle = object()
    runner = SimpleNamespace(
        _driver_handle=driver_handle,
        _issue_query_task_admission_capability=lambda _query_id: _QUERY_GENERATION_CAPABILITY,
        _session_lock=threading.RLock(),
        _plan_session_ids={"test-plan": "session-1"},
        _query_udf_actor_nodes={},
        _query_udf_session_configs={},
        _active_udf_actor_by_unit={},
    )
    plan = _FakePlan(
        [
            {
                "node_id": 0,
                "payload": {
                    "execution_backend": "ray_task",
                    "gpus": 0.0,
                },
            }
        ]
    )

    created = runner_cls._precreate_udf_actors(
        runner,
        plan,
        SimpleNamespace(query_id="test-plan", units=()),
        query_connection=object(),
        session_config={},
    )

    assert created == []
    assert plan.set_calls == [
        {
            "handles_map": {
                "0": {
                    "query_driver_handle": driver_handle,
                    "query_generation_capability": _QUERY_GENERATION_CAPABILITY,
                    "session_config": {},
                }
            }
        }
    ]


def test_precreate_udf_actors_skips_non_ray_nodes(monkeypatch):
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class

    runner = SimpleNamespace(
        _driver_handle=object(),
        _issue_query_task_admission_capability=lambda _query_id: _QUERY_GENERATION_CAPABILITY,
        _session_lock=threading.RLock(),
        _plan_session_ids={"test-plan": "session-1"},
        _query_udf_actor_nodes={},
        _query_udf_session_configs={},
        _active_udf_actor_by_unit={},
    )

    plan = _FakePlan(
        [
            {
                "node_id": 3,
                "pool_name": None,
                "actor_pool_size": 1,
                "gpus": 0.0,
                "payload": {"execution_backend": "subprocess_actor"},
            }
        ]
    )

    created = runner_cls._precreate_udf_actors(
        runner,
        plan,
        SimpleNamespace(query_id="test-plan", units=()),
        query_connection=object(),
        session_config={},
    )

    assert created == []
    assert plan.set_calls == [
        {
            "handles_map": {
                "3": {
                    "session_config": {},
                }
            }
        }
    ]


@pytest.mark.parametrize(
    ("module_name", "factory_name", "method_name", "active_attr", "by_plan_attr"),
    [
        (
            "vane.execution.vllm",
            "ensure_named_vllm_pools_for_plan",
            "_precreate_vllm_actors",
            "_active_vllm_actors",
            "_active_vllm_actors_by_plan",
        ),
    ],
)
def test_precreate_retains_partially_created_actor_pool_for_teardown(
    monkeypatch,
    module_name,
    factory_name,
    method_name,
    active_attr,
    by_plan_attr,
):
    from vane.runners.ray.driver import RayQueryDriverActor

    pool = object()
    creation_error = RuntimeError("partial actor cleanup failed")
    creation_error.owned_actor_pools = [pool]

    def _fail(*_args, **_kwargs):
        raise creation_error

    fake_mod = types.ModuleType(module_name)
    setattr(fake_mod, factory_name, _fail)
    monkeypatch.setitem(sys.modules, module_name, fake_mod)

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = SimpleNamespace(
        _driver_handle=object(),
        _issue_query_task_admission_capability=lambda _query_id: _QUERY_GENERATION_CAPABILITY,
        **{
            active_attr: [],
            by_plan_attr: {},
        },
    )
    plan = _FakePlan([])

    with pytest.raises(RuntimeError, match="partial actor cleanup failed"):
        method = getattr(runner_cls, method_name)
        if method_name == "_precreate_udf_actors":
            method(
                runner,
                plan,
                SimpleNamespace(query_id="test-plan", units=()),
                query_connection=object(),
                session_config={},
            )
        else:
            method(
                runner,
                plan,
                query_connection=object(),
                session_config={},
            )

    assert getattr(runner, active_attr) == [pool]
    assert getattr(runner, by_plan_attr) == {"test-plan": [pool]}


def test_driver_udf_actor_handle_hook_is_disabled_by_default(monkeypatch):
    from vane.runners.ray.driver import RayQueryDriverActor

    monkeypatch.delenv("VANE_ENABLE_UDF_TEST_HOOKS", raising=False)
    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = runner_cls.__new__(runner_cls)

    with pytest.raises(RuntimeError, match="VANE_ENABLE_UDF_TEST_HOOKS=1"):
        runner_cls.get_test_udf_actor_handle(runner, "owner-a", "plan-7", "actor_udf")


def test_precreate_udf_actors_enable_generic_async_for_distributed_pool(
    monkeypatch,
):
    import vane.execution.udf_ray as udf_ray

    calls = []

    class _FakeActorsObj:
        def __init__(self, actors):
            self.actors = actors
            self._init_refs = []
            self._confirmed_ready = set()

    class _FakeUDFActorPool:
        def __init__(
            self,
            *,
            payload,
            concurrency,
            gpus_per_actor,
            ray_options=None,
        ):
            calls.append(
                {
                    "payload": dict(payload),
                    "concurrency": concurrency,
                    "gpus_per_actor": gpus_per_actor,
                    "ray_options": ray_options,
                }
            )
            self.actors = [f"actor-{idx}" for idx in range(concurrency)]
            self._init_refs = []
            self.actor_node_ids = ["node-a"] * concurrency
            self._confirmed_ready = set(range(concurrency))

    fake_ray = types.ModuleType("ray")
    fake_ray.is_initialized = lambda: True
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(udf_ray, "_is_vane_worker_process", lambda: False)
    monkeypatch.setattr(udf_ray, "UDFActorPool", _FakeUDFActorPool)

    plan = _FakePlan(
        [
            {
                "node_id": 7,
                "pool_name": "audio-transcriber",
                "actor_pool_size": 2,
                "gpus": 0.0,
                "payload": {
                    "udf_name": "decode_images",
                    "execution_backend": "ray_actor",
                    "query_id": _TEST_QUERY_ID,
                    "resource_unit_id": "resource:test:actor",
                },
            }
        ]
    )

    created, handles_map = udf_ray.ensure_actor_pools_for_plan(
        plan,
        query_driver_handle=object(),
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
        session_config={},
        conn=object(),
    )

    assert len(created) == 1
    assert "7" in handles_map
    assert len(calls) == 1


def test_ensure_actor_pools_for_plan_creates_anonymous_handles_without_pool_name(monkeypatch):
    import vane.execution.udf_ray as udf_ray

    calls = []

    class _FakeActorsObj:
        def __init__(self, actors):
            self.actors = actors
            self._init_refs = []
            self._confirmed_ready = set()

    class _FakeUDFActorPool:
        def __init__(
            self,
            *,
            payload,
            concurrency,
            gpus_per_actor,
            ray_options=None,
        ):
            calls.append(
                {
                    "payload": dict(payload),
                    "concurrency": concurrency,
                    "gpus_per_actor": gpus_per_actor,
                    "ray_options": ray_options,
                }
            )
            self.actors = [f"actor-{idx}" for idx in range(concurrency)]
            self._init_refs = []
            self.actor_node_ids = ["node-a"] * concurrency
            self._confirmed_ready = set(range(concurrency))

    fake_ray = types.ModuleType("ray")
    fake_ray.is_initialized = lambda: True
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(udf_ray, "_is_vane_worker_process", lambda: False)
    monkeypatch.setattr(udf_ray, "UDFActorPool", _FakeUDFActorPool)

    plan = _FakePlan(
        [
            {
                "node_id": 7,
                "pool_name": "ignored-old-pool-name",
                "actor_pool_size": 2,
                "gpus": 0.0,
                "payload": {
                    "udf_name": "decode_images",
                    "execution_backend": "ray_actor",
                    "query_id": _TEST_QUERY_ID,
                    "resource_unit_id": "resource:test:actor",
                },
            }
        ]
    )

    query_driver_handle = object()
    created, handles_map = udf_ray.ensure_actor_pools_for_plan(
        plan,
        query_driver_handle=query_driver_handle,
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
        session_config={},
        conn=object(),
    )

    assert len(created) == 1
    assert len(calls) == 1
    assert calls[0]["payload"] == {
        "udf_name": "decode_images",
        "execution_backend": "ray_actor",
        "query_id": _TEST_QUERY_ID,
        "resource_unit_id": "resource:test:actor",
    }
    assert calls[0]["concurrency"] == 2
    assert calls[0]["gpus_per_actor"] == 0.0
    assert calls[0]["ray_options"]["num_cpus"] == 1.0
    _assert_actor_location_runtime_env(
        calls[0]["ray_options"]["runtime_env"]["env_vars"],
        "resource:test:actor",
    )
    assert handles_map["7"]["actor_handles"] == ["actor-0", "actor-1"]
    assert handles_map["7"]["query_driver_handle"] is query_driver_handle
    assert "ray_actor_pool_name" not in handles_map["7"]
    assert plan.set_calls == [
        {
            "handles_map": handles_map,
        }
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {
            "udf_name": "replicated_transform",
            "execution_backend": "ray_actor",
            "actor_number": 2,
            "query_id": _TEST_QUERY_ID,
            "resource_unit_id": "resource:test:retry-policy",
        },
        {
            "udf_name": "ai_prompt",
            "execution_backend": "ray_actor",
            "ai_operation": "prompt",
            "query_id": _TEST_QUERY_ID,
            "resource_unit_id": "resource:test:retry-policy",
        },
    ],
    ids=["class-udf", "ai"],
)
def test_ensure_actor_pools_for_plan_keeps_default_retry_policy(monkeypatch, payload):
    import vane.execution.udf_ray as udf_ray
    from vane.execution.udf_ray_config import MAX_ACTOR_RESTARTS, MAX_ACTOR_TASK_RETRIES

    calls = []

    class _FakeUDFActorPool:
        def __init__(
            self,
            *,
            payload,
            concurrency,
            gpus_per_actor,
            ray_options=None,
            max_restarts=MAX_ACTOR_RESTARTS,
            max_task_retries=MAX_ACTOR_TASK_RETRIES,
        ):
            calls.append((max_restarts, max_task_retries))
            self.actors = ["ordinary-actor"]
            self._init_refs = []
            self.actor_node_ids = ["node-a"] * concurrency
            self._confirmed_ready = {0}

    fake_ray = types.ModuleType("ray")
    fake_ray.is_initialized = lambda: True
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(udf_ray, "_is_vane_worker_process", lambda: False)
    monkeypatch.setattr(udf_ray, "UDFActorPool", _FakeUDFActorPool)

    plan = _FakePlan(
        [
            {
                "node_id": 8,
                "actor_pool_size": 1,
                "gpus": 0.0,
                "payload": payload,
            }
        ]
    )

    created, _ = udf_ray.ensure_actor_pools_for_plan(
        plan,
        query_driver_handle=object(),
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
        session_config={},
        conn=object(),
    )

    assert len(created) == 1
    assert calls == [(MAX_ACTOR_RESTARTS, MAX_ACTOR_TASK_RETRIES)]


def test_ensure_actor_pools_for_nodes_injects_with_callback(monkeypatch):
    import vane.execution.udf_ray as udf_ray

    calls = []
    injected = []

    class _FakeUDFActorPool:
        def __init__(
            self,
            *,
            payload,
            concurrency,
            gpus_per_actor,
            ray_options=None,
        ):
            calls.append(
                {
                    "payload": dict(payload),
                    "concurrency": concurrency,
                    "gpus_per_actor": gpus_per_actor,
                    "ray_options": ray_options,
                }
            )
            self.actors = [f"actor-{idx}" for idx in range(concurrency)]
            self._init_refs = []
            self.actor_node_ids = ["node-a"] * concurrency
            self._confirmed_ready = set(range(concurrency))

    fake_ray = types.ModuleType("ray")
    fake_ray.is_initialized = lambda: True
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(udf_ray, "_is_vane_worker_process", lambda: False)
    monkeypatch.setattr(udf_ray, "UDFActorPool", _FakeUDFActorPool)

    nodes = [
        {
            "node_id": 9,
            "actor_pool_size": 2,
            "gpus": 0.0,
            "payload": {
                "udf_name": "decode_images",
                "execution_backend": "ray_actor",
                "query_id": _TEST_QUERY_ID,
                "resource_unit_id": "resource:test:actor",
            },
        }
    ]

    def inject(handles_map):
        injected.append(handles_map)

    created, handles_map = udf_ray.ensure_actor_pools_for_nodes(
        nodes,
        query_driver_handle=object(),
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
        session_config={},
        set_handles=inject,
    )

    assert len(created) == 1
    assert len(calls) == 1
    assert calls[0]["payload"] == {
        "udf_name": "decode_images",
        "execution_backend": "ray_actor",
        "query_id": _TEST_QUERY_ID,
        "resource_unit_id": "resource:test:actor",
    }
    assert calls[0]["concurrency"] == 2
    assert calls[0]["gpus_per_actor"] == 0.0
    assert calls[0]["ray_options"]["num_cpus"] == 1.0
    _assert_actor_location_runtime_env(
        calls[0]["ray_options"]["runtime_env"]["env_vars"],
        "resource:test:actor",
    )
    assert handles_map["9"]["actor_handles"] == ["actor-0", "actor-1"]
    assert injected == [handles_map]


def test_ray_plan_injects_session_context_for_explicit_subprocess_backends(monkeypatch):
    import vane.execution.udf_ray as udf_ray

    fake_ray = types.ModuleType("ray")
    fake_ray.is_initialized = lambda: True
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(udf_ray, "_is_vane_worker_process", lambda: False)

    nodes = [
        {
            "node_id": 3,
            "payload": {
                "execution_backend": "subprocess_task",
            },
        },
        {
            "node_id": 4,
            "payload": {
                "execution_backend": "subprocess_actor",
            },
        },
    ]
    session_config = {
        "AWS_ACCESS_KEY_ID": "session-key",
        "VANE_AUTH_HEADER": "session-auth",
    }
    injected = []

    created, handles_map = udf_ray.ensure_actor_pools_for_nodes(
        nodes,
        query_driver_handle=object(),
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
        session_config=session_config,
        set_handles=injected.append,
    )

    assert created == []
    assert handles_map == {
        "3": {"session_config": session_config},
        "4": {"session_config": session_config},
    }
    assert injected == [handles_map]


def test_prepare_actor_pools_publishes_handles_before_waiting_for_init(monkeypatch):
    import vane.execution.udf_ray as udf_ray

    resolved = []

    class _InitRef:
        def __init__(self, actor_index):
            self.actor_index = actor_index

        def future(self):
            actor_index = self.actor_index

            class _Future:
                def result(self, timeout=None):
                    resolved.append((actor_index, timeout))
                    return ("node-a", "node-b")[actor_index]

            return _Future()

    class _FakeUDFActorPool:
        def __init__(
            self,
            *,
            payload,
            concurrency,
            gpus_per_actor,
            ray_options=None,
        ):
            self.actors = [f"actor-{idx}" for idx in range(concurrency)]
            self._init_refs = [_InitRef(idx) for idx in range(concurrency)]
            self.actor_node_ids = ["node-a"] * concurrency
            self._confirmed_ready = set()
            self._owns_actors = True

        def shutdown(self):
            self.actors = []

    fake_ray = types.ModuleType("ray")
    fake_ray.is_initialized = lambda: True
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(udf_ray, "_is_vane_worker_process", lambda: False)
    monkeypatch.setattr(udf_ray, "UDFActorPool", _FakeUDFActorPool)

    plan = _FakePlan(
        [
            {
                "node_id": 5,
                "actor_pool_size": 2,
                "payload": {
                    "udf_name": "embed",
                    "execution_backend": "ray_actor",
                    "query_id": _TEST_QUERY_ID,
                    "resource_unit_id": "resource:test:deferred-ready",
                },
            }
        ]
    )

    pools, handles = udf_ray.prepare_actor_pools_for_plan(
        plan,
        query_driver_handle=object(),
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
        session_config={},
    )

    assert resolved == []
    assert handles["5"]["actor_dispatch_indices"] == []
    assert plan.set_calls == [{"handles_map": handles}]

    udf_ray.wait_for_actor_pools_ready(pools)

    assert [actor_index for actor_index, _ in resolved] == [0, 1]
    assert pools[0]._confirmed_ready == {0, 1}


def test_actor_pool_opens_after_first_ray_core_actor_becomes_ready(monkeypatch):
    import vane.execution.udf_ray as udf_ray

    class _Ref:
        def __init__(self, node_id):
            self.node_id = node_id

        def future(self):
            node_id = self.node_id

            class _Future:
                def result(self, timeout=None):
                    return node_id

            return _Future()

    refs = [_Ref("node-a"), _Ref("node-b"), _Ref("node-c")]
    wait_calls = []

    def wait(pending, *, num_returns, timeout):
        wait_calls.append((list(pending), num_returns, timeout))
        if len(wait_calls) == 1:
            return [refs[1]], [refs[0], refs[2]]
        return [], list(pending)

    fake_ray = types.SimpleNamespace(wait=wait, kill=lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.delenv("VANE_RAY_ACTOR_INIT_TIMEOUT_S", raising=False)
    monkeypatch.delenv("VANE_RAY_OBJECT_GET_TIMEOUT_S", raising=False)
    monkeypatch.delenv("VANE_QUERY_DEADLINE_EPOCH_S", raising=False)
    pool = SimpleNamespace(
        actors=["actor-a", "actor-b", "actor-c"],
        actor_node_ids=["", "", ""],
        _init_refs=refs,
        _confirmed_ready=set(),
        _owns_actors=True,
    )

    ready = udf_ray.wait_for_first_actor_pool_ready(pool)

    assert ready == {1: "node-b"}
    assert pool.actor_node_ids == ["", "node-b", ""]
    assert pool._confirmed_ready == {1}
    assert wait_calls[0][1:] == (1, None)
    assert wait_calls[1][1:] == (2, 0)


def test_driver_publishes_later_actor_slots_as_ray_core_schedules_them():
    from concurrent.futures import Future

    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class

    class _Ref:
        def __init__(self, future):
            self._future = future

        def future(self):
            return self._future

    pending_future = Future()
    pool = SimpleNamespace(
        actor_node_ids=["node-a", ""],
        _confirmed_ready={0},
        _init_refs=[_Ref(Future()), _Ref(pending_future)],
        _vane_retired=False,
        _vane_init_errors={},
        _vane_readiness_futures=[],
    )
    published = []
    manager = SimpleNamespace(
        set_ready_actor_slots=lambda resource_unit_id, nodes: published.append((resource_unit_id, dict(nodes)))
    )
    runner = SimpleNamespace(
        _session_lock=threading.RLock(),
        _active_udf_actor_by_unit={"q1": {"resource:q1:actor": pool}},
    )

    runner_cls._watch_query_udf_actor_pool_readiness(
        runner,
        "q1",
        "resource:q1:actor",
        pool,
        manager,
    )
    pending_future.set_result("node-b")

    assert pool.actor_node_ids == ["node-a", "node-b"]
    assert pool._confirmed_ready == {0, 1}
    assert published == [
        (
            "resource:q1:actor",
            {0: "node-a", 1: "node-b"},
        )
    ]


def test_later_actor_init_failure_fences_query_without_releasing_live_owners():
    from concurrent.futures import Future

    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class

    class _Ref:
        def __init__(self, future):
            self._future = future

        def future(self):
            return self._future

    pending_future = Future()
    pool = SimpleNamespace(
        actor_node_ids=["node-a", ""],
        _confirmed_ready={0},
        _init_refs=[_Ref(Future()), _Ref(pending_future)],
        _vane_retired=False,
        _vane_init_errors={},
        _vane_readiness_futures=[],
    )
    failures = []
    manager = SimpleNamespace(
        set_ready_actor_slots=lambda *_args: pytest.fail("a failed actor must not join the ready set"),
        fail=failures.append,
    )
    runner = SimpleNamespace(
        _session_lock=threading.RLock(),
        _query_resource_lock=threading.RLock(),
        _query_terminal_errors={},
        _active_udf_actor_by_unit={"q1": {"resource:q1:actor": pool}},
    )

    runner_cls._watch_query_udf_actor_pool_readiness(
        runner,
        "q1",
        "resource:q1:actor",
        pool,
        manager,
    )
    pending_future.set_exception(RuntimeError("warmup failed"))

    assert isinstance(pool._vane_init_errors[1], RuntimeError)
    assert failures == [runner._query_terminal_errors["q1"]]
    assert "resource:q1:actor/actor-1" in failures[0]
    assert "warmup failed" in failures[0]


def test_retired_pool_ignores_late_actor_init_failure():
    from concurrent.futures import Future

    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class

    class _Ref:
        def __init__(self, future):
            self._future = future

        def future(self):
            return self._future

    pending_future = Future()
    pool = SimpleNamespace(
        actor_node_ids=[""],
        _confirmed_ready=set(),
        _init_refs=[_Ref(pending_future)],
        _vane_retired=False,
        _vane_init_errors={},
        _vane_readiness_futures=[],
    )
    failures = []
    manager = SimpleNamespace(
        set_ready_actor_slots=lambda *_args: pytest.fail("a retired actor must not join the ready set"),
        fail=failures.append,
    )
    runner = SimpleNamespace(
        _session_lock=threading.RLock(),
        _query_resource_lock=threading.RLock(),
        _query_terminal_errors={},
        _active_udf_actor_by_unit={"q1": {"resource:q1:actor": pool}},
    )

    runner_cls._watch_query_udf_actor_pool_readiness(
        runner,
        "q1",
        "resource:q1:actor",
        pool,
        manager,
    )
    with runner._session_lock:
        pool._vane_retired = True
    pending_future.set_exception(RuntimeError("killed during retirement"))

    assert pool._vane_init_errors == {}
    assert runner._query_terminal_errors == {}
    assert failures == []


def test_stale_execution_phase_callback_does_not_retire_current_actor_pool(
    monkeypatch,
):
    import vane.runners.ray.query_resource_runtime as resource_runtime
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    shutdowns = []
    pool = SimpleNamespace(
        _vane_retired=False,
        shutdown=lambda: shutdowns.append("shutdown"),
    )
    manager = SimpleNamespace(
        begin_actor_pool_retirement=lambda _resource_unit_id: False,
    )
    monkeypatch.setattr(
        resource_runtime,
        "get_query_resource_manager",
        lambda _query_id: manager,
    )
    runner = SimpleNamespace(
        _session_lock=threading.RLock(),
        _active_udf_actor_by_unit={"q1": {"resource:q1:actor": pool}},
        _active_udf_actors=[pool],
        _active_udf_actors_by_plan={"q1": [pool]},
    )

    runner_cls._retire_udf_actor_pools_outside_phase(
        runner,
        "q1",
        ("resource:q1:new-phase",),
    )

    assert shutdowns == []
    assert pool._vane_retired is False
    assert runner._active_udf_actor_by_unit["q1"]["resource:q1:actor"] is pool


def test_plan_actor_cleanup_reports_close_failure_without_retaining_terminated_pool():
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class

    class Pool:
        def __init__(self):
            self._vane_retired = False
            self.actors = ["actor-0"]

        def shutdown(self):
            self.actors = []
            raise RuntimeError("planned callable close failure")

    pool = Pool()
    runner = SimpleNamespace(
        _session_lock=threading.RLock(),
        _active_udf_actor_by_unit={"q1": {"resource:q1:actor": pool}},
        _active_udf_actors=[pool],
        _active_udf_actors_by_plan={"q1": [pool]},
    )

    diagnostics = runner_cls._cleanup_udf_actor_pools(runner, "q1")

    assert len(diagnostics) == 1
    assert "planned callable close failure" in str(diagnostics[0])
    assert runner._active_udf_actor_by_unit == {}
    assert runner._active_udf_actors == []
    assert "q1" not in runner._active_udf_actors_by_plan


def test_phase_actor_retirement_records_close_failure_after_releasing_terminated_pool(monkeypatch):
    import vane.runners.ray.query_resource_runtime as resource_runtime
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class

    class Pool:
        def __init__(self):
            self._vane_retired = False
            self.actors = ["actor-0"]

        def shutdown(self):
            self.actors = []
            raise RuntimeError("planned phase close failure")

    pool = Pool()
    retirements = []
    manager = SimpleNamespace(
        begin_actor_pool_retirement=lambda resource_unit_id: retirements.append(("begin", resource_unit_id)) or True,
        complete_actor_pool_retirement=lambda resource_unit_id: (
            retirements.append(("complete", resource_unit_id)) or True
        ),
    )
    monkeypatch.setattr(resource_runtime, "get_query_resource_manager", lambda _query_id: manager)
    runner = SimpleNamespace(
        _session_lock=threading.RLock(),
        _active_udf_actor_by_unit={"q1": {"resource:q1:actor": pool}},
        _active_udf_actors=[pool],
        _active_udf_actors_by_plan={"q1": [pool]},
        _udf_actor_cleanup_diagnostics_by_plan={},
    )

    runner_cls._retire_udf_actor_pools_outside_phase(
        runner,
        "q1",
        ("resource:q1:new-phase",),
    )

    assert retirements == [
        ("begin", "resource:q1:actor"),
        ("complete", "resource:q1:actor"),
    ]
    assert runner._active_udf_actor_by_unit["q1"] == {}
    assert runner._active_udf_actors == []
    assert "q1" not in runner._active_udf_actors_by_plan
    assert len(runner._udf_actor_cleanup_diagnostics_by_plan["q1"]) == 1
    assert "planned phase close failure" in str(runner._udf_actor_cleanup_diagnostics_by_plan["q1"][0])


def test_phase_actor_retirement_preserves_close_failure_when_completion_fails(monkeypatch):
    import vane.runners.ray.query_resource_runtime as resource_runtime
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class

    class Pool:
        def __init__(self):
            self._vane_retired = False
            self.actors = ["actor-0"]

        def shutdown(self):
            self.actors = []
            raise RuntimeError("planned phase close failure")

    pool = Pool()
    manager = SimpleNamespace(
        begin_actor_pool_retirement=lambda _resource_unit_id: True,
        complete_actor_pool_retirement=lambda _resource_unit_id: False,
    )
    monkeypatch.setattr(resource_runtime, "get_query_resource_manager", lambda _query_id: manager)
    runner = SimpleNamespace(
        _session_lock=threading.RLock(),
        _active_udf_actor_by_unit={"q1": {"resource:q1:actor": pool}},
        _active_udf_actors=[pool],
        _active_udf_actors_by_plan={"q1": [pool]},
        _udf_actor_cleanup_diagnostics_by_plan={},
    )

    with pytest.raises(RuntimeError, match="retirement state disappeared"):
        runner_cls._retire_udf_actor_pools_outside_phase(
            runner,
            "q1",
            ("resource:q1:new-phase",),
        )

    assert pool._vane_retired
    assert runner._active_udf_actor_by_unit["q1"]["resource:q1:actor"] is pool
    assert runner._active_udf_actors == [pool]
    assert runner._active_udf_actors_by_plan["q1"] == [pool]
    assert len(runner._udf_actor_cleanup_diagnostics_by_plan["q1"]) == 1
    assert "planned phase close failure" in str(runner._udf_actor_cleanup_diagnostics_by_plan["q1"][0])


def test_phase_actor_retirement_serializes_plan_cleanup(monkeypatch):
    import vane.runners.ray.query_resource_runtime as resource_runtime
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    shutdown_entered = threading.Event()
    release_shutdown = threading.Event()
    cleanup_finished = threading.Event()
    shutdowns = []
    errors = []

    def shutdown():
        shutdowns.append("shutdown")
        shutdown_entered.set()
        assert release_shutdown.wait(timeout=5)

    pool = SimpleNamespace(_vane_retired=False, shutdown=shutdown)
    retirements = []
    manager = SimpleNamespace(
        begin_actor_pool_retirement=lambda resource_unit_id: retirements.append(("begin", resource_unit_id)) or True,
        complete_actor_pool_retirement=lambda resource_unit_id: (
            retirements.append(("complete", resource_unit_id)) or True
        ),
    )
    monkeypatch.setattr(
        resource_runtime,
        "get_query_resource_manager",
        lambda _query_id: manager,
    )
    runner = SimpleNamespace(
        _session_lock=threading.RLock(),
        _active_udf_actor_by_unit={"q1": {"resource:q1:actor": pool}},
        _active_udf_actors=[pool],
        _active_udf_actors_by_plan={"q1": [pool]},
    )

    def retire_phase():
        try:
            runner_cls._retire_udf_actor_pools_outside_phase(
                runner,
                "q1",
                ("resource:q1:new-phase",),
            )
        except BaseException as exc:
            errors.append(exc)

    def cleanup_plan():
        try:
            runner_cls._cleanup_udf_actor_pools(runner, "q1")
        except BaseException as exc:
            errors.append(exc)
        finally:
            cleanup_finished.set()

    phase_thread = threading.Thread(target=retire_phase)
    cleanup_thread = threading.Thread(target=cleanup_plan)
    phase_thread.start()
    assert shutdown_entered.wait(timeout=5)
    cleanup_thread.start()
    assert cleanup_finished.wait(timeout=0.05) is False
    assert shutdowns == ["shutdown"]

    release_shutdown.set()
    phase_thread.join(timeout=5)
    cleanup_thread.join(timeout=5)

    assert phase_thread.is_alive() is False
    assert cleanup_thread.is_alive() is False
    assert errors == []
    assert shutdowns == ["shutdown"]
    assert retirements == [
        ("begin", "resource:q1:actor"),
        ("complete", "resource:q1:actor"),
    ]
    assert runner._active_udf_actor_by_unit["q1"] == {}
    assert runner._active_udf_actors == []
    assert "q1" not in runner._active_udf_actors_by_plan


def test_plan_cleanup_serializes_late_phase_actor_retirement(monkeypatch):
    import vane.runners.ray.query_resource_runtime as resource_runtime
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    shutdown_entered = threading.Event()
    release_shutdown = threading.Event()
    phase_finished = threading.Event()
    shutdowns = []
    errors = []

    def shutdown():
        shutdowns.append("shutdown")
        shutdown_entered.set()
        assert release_shutdown.wait(timeout=5)

    pool = SimpleNamespace(_vane_retired=False, shutdown=shutdown)
    manager_lookups = []
    monkeypatch.setattr(
        resource_runtime,
        "get_query_resource_manager",
        lambda query_id: manager_lookups.append(query_id),
    )
    runner = SimpleNamespace(
        _session_lock=threading.RLock(),
        _active_udf_actor_by_unit={"q1": {"resource:q1:actor": pool}},
        _active_udf_actors=[pool],
        _active_udf_actors_by_plan={"q1": [pool]},
    )

    def cleanup_plan():
        try:
            runner_cls._cleanup_udf_actor_pools(runner, "q1")
        except BaseException as exc:
            errors.append(exc)

    def retire_phase():
        try:
            runner_cls._retire_udf_actor_pools_outside_phase(
                runner,
                "q1",
                ("resource:q1:new-phase",),
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            phase_finished.set()

    cleanup_thread = threading.Thread(target=cleanup_plan)
    phase_thread = threading.Thread(target=retire_phase)
    cleanup_thread.start()
    assert shutdown_entered.wait(timeout=5)
    phase_thread.start()
    assert phase_finished.wait(timeout=0.05) is False
    assert shutdowns == ["shutdown"]

    release_shutdown.set()
    cleanup_thread.join(timeout=5)
    phase_thread.join(timeout=5)

    assert cleanup_thread.is_alive() is False
    assert phase_thread.is_alive() is False
    assert errors == []
    assert shutdowns == ["shutdown"]
    assert manager_lookups == []
    assert runner._active_udf_actor_by_unit == {}
    assert runner._active_udf_actors == []
    assert "q1" not in runner._active_udf_actors_by_plan


def test_actor_activation_rechecks_phase_after_slow_pool_creation(monkeypatch):
    import vane.execution.udf_ray as udf_ray
    import vane.runners.ray.query_resource_runtime as resource_runtime
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    resource_unit_id = "resource:q1:actor"
    eligibility_reads = 0
    submitted = []

    class _Manager:
        def current_eligible_resource_unit_ids(self):
            nonlocal eligibility_reads
            eligibility_reads += 1
            return (resource_unit_id,) if eligibility_reads == 1 else ()

        def set_submitted_actor_slots(self, unit_id, actor_indices):
            submitted.append((unit_id, set(actor_indices)))

    manager = _Manager()
    monkeypatch.setattr(
        resource_runtime,
        "get_query_resource_manager",
        lambda _query_id: manager,
    )
    shutdowns = []
    pool = SimpleNamespace(
        actors=["actor-0"],
        shutdown=lambda: shutdowns.append("shutdown"),
    )
    monkeypatch.setattr(
        udf_ray,
        "prepare_actor_pools_for_nodes",
        lambda *_args, **_kwargs: (
            [pool],
            {
                "node-1": {
                    "actor_handles": ["actor-0"],
                    "actor_node_ids": [""],
                    "actor_dispatch_indices": [],
                    "actor_init_refs": [],
                }
            },
        ),
    )
    runner = SimpleNamespace(
        _query_resource_lock=threading.RLock(),
        _session_lock=threading.RLock(),
        _query_resource_graphs={
            "q1": SimpleNamespace(unit_by_id=lambda _unit_id: SimpleNamespace(backend="ray_actor"))
        },
        _query_allocations={"q1": object()},
        _query_udf_actor_nodes={
            "q1": {
                resource_unit_id: {
                    "node_id": "node-1",
                }
            }
        },
        _query_udf_session_configs={"q1": {}},
        _plan_session_ids={"q1": "session-1"},
        _active_udf_actor_by_unit={"q1": {}},
        _active_udf_actors=[],
        _active_udf_actors_by_plan={},
        _driver_handle=object(),
    )

    with pytest.raises(RuntimeError, match="phase ended while activating"):
        runner_cls._activate_query_udf_actor_pool_sync(
            runner,
            {
                "query_id": "q1",
                "resource_unit_id": resource_unit_id,
                "physical_node_id": "node-1",
            },
            _QUERY_GENERATION_CAPABILITY,
        )

    assert eligibility_reads == 2
    assert submitted == []
    assert shutdowns == ["shutdown"]
    assert runner._active_udf_actor_by_unit["q1"] == {}


def test_actor_activation_cleans_up_when_atomic_slot_publication_is_fenced(monkeypatch):
    import vane.execution.udf_ray as udf_ray
    import vane.runners.ray.query_resource_runtime as resource_runtime
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    resource_unit_id = "resource:q1:actor"

    class _Manager:
        @staticmethod
        def current_eligible_resource_unit_ids():
            return (resource_unit_id,)

        @staticmethod
        def set_submitted_actor_slots(_unit_id, _actor_indices):
            raise RuntimeError("query was cancelled during actor creation")

    monkeypatch.setattr(
        resource_runtime,
        "get_query_resource_manager",
        lambda _query_id: _Manager(),
    )
    shutdowns = []
    pool = SimpleNamespace(
        actors=["actor-0"],
        shutdown=lambda: shutdowns.append("shutdown"),
    )
    monkeypatch.setattr(
        udf_ray,
        "prepare_actor_pools_for_nodes",
        lambda *_args, **_kwargs: (
            [pool],
            {
                "node-1": {
                    "actor_handles": ["actor-0"],
                    "actor_node_ids": [""],
                    "actor_dispatch_indices": [],
                    "actor_init_refs": [],
                }
            },
        ),
    )
    runner = SimpleNamespace(
        _query_resource_lock=threading.RLock(),
        _session_lock=threading.RLock(),
        _query_resource_graphs={
            "q1": SimpleNamespace(unit_by_id=lambda _unit_id: SimpleNamespace(backend="ray_actor"))
        },
        _query_allocations={"q1": object()},
        _query_udf_actor_nodes={
            "q1": {
                resource_unit_id: {
                    "node_id": "node-1",
                }
            }
        },
        _query_udf_session_configs={"q1": {}},
        _plan_session_ids={"q1": "session-1"},
        _active_udf_actor_by_unit={"q1": {}},
        _active_udf_actors=[],
        _active_udf_actors_by_plan={},
        _driver_handle=object(),
    )

    with pytest.raises(RuntimeError, match="cancelled during actor creation"):
        runner_cls._activate_query_udf_actor_pool_sync(
            runner,
            {
                "query_id": "q1",
                "resource_unit_id": resource_unit_id,
                "physical_node_id": "node-1",
            },
            _QUERY_GENERATION_CAPABILITY,
        )

    assert shutdowns == ["shutdown"]
    assert runner._active_udf_actor_by_unit["q1"] == {}
    assert runner._active_udf_actors == []
    assert runner._active_udf_actors_by_plan == {}


def test_actor_activation_recreates_retired_pool_after_phase_reentry(monkeypatch):
    import vane.runners.ray.query_resource_runtime as resource_runtime
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    resource_unit_id = "resource:q1:actor"
    manager = SimpleNamespace(current_eligible_resource_unit_ids=lambda: (resource_unit_id,))
    monkeypatch.setattr(resource_runtime, "get_query_resource_manager", lambda _query_id: manager)

    runner = SimpleNamespace(
        _session_lock=threading.RLock(),
        _query_udf_actor_lifecycle_locks={},
        _query_udf_actor_activation_tasks={},
        _active_udf_actor_by_unit={"q1": {}},
        _verify_query_task_admission_capability=lambda **_kwargs: "generation-1",
        _capability_targets_inactive_query_generation=lambda *_args: False,
    )
    native_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-driver-native")
    runner._get_driver_native_executor = lambda: native_executor
    activations = []

    def activate_sync(_locator, _capability):
        pool = SimpleNamespace(_vane_retired=False)
        with runner._session_lock:
            runner._active_udf_actor_by_unit["q1"][resource_unit_id] = pool
        activations.append(pool)
        return {"actor_handles": [f"actor-{len(activations)}"]}

    runner._activate_query_udf_actor_pool_sync = activate_sync
    locator = {
        "query_id": "q1",
        "resource_unit_id": resource_unit_id,
        "physical_node_id": "node-1",
    }

    async def scenario():
        stale = asyncio.create_task(asyncio.sleep(0, result={"actor_handles": ["retired-actor"]}))
        await stale
        runner._query_udf_actor_activation_tasks[("q1", resource_unit_id)] = stale
        return await runner_cls.activate_query_udf_actor_pool(
            runner,
            locator,
            _QUERY_GENERATION_CAPABILITY,
        )

    try:
        assert asyncio.run(scenario()) == {"actor_handles": ["actor-1"]}
    finally:
        native_executor.shutdown(wait=True)
    assert len(activations) == 1
    assert runner._active_udf_actor_by_unit["q1"][resource_unit_id] is activations[0]


def test_cancelled_actor_activation_waiter_keeps_shared_creation_cached(monkeypatch):
    import vane.runners.ray.query_resource_runtime as resource_runtime
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    resource_unit_id = "resource:q1:actor"
    manager = SimpleNamespace(current_eligible_resource_unit_ids=lambda: (resource_unit_id,))
    monkeypatch.setattr(resource_runtime, "get_query_resource_manager", lambda _query_id: manager)

    runner = SimpleNamespace(
        _session_lock=threading.RLock(),
        _query_udf_actor_lifecycle_locks={},
        _query_udf_actor_activation_tasks={},
        _active_udf_actor_by_unit={"q1": {}},
        _verify_query_task_admission_capability=lambda **_kwargs: "generation-1",
        _capability_targets_inactive_query_generation=lambda *_args: False,
    )
    native_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-driver-native")
    runner._get_driver_native_executor = lambda: native_executor
    activation_started = threading.Event()
    release_activation = threading.Event()
    activation_count = 0

    def activate_sync(_locator, _capability):
        nonlocal activation_count
        activation_count += 1
        activation_started.set()
        assert release_activation.wait(timeout=5)
        with runner._session_lock:
            runner._active_udf_actor_by_unit["q1"][resource_unit_id] = SimpleNamespace(_vane_retired=False)
        return {"actor_handles": ["actor-1"]}

    runner._activate_query_udf_actor_pool_sync = activate_sync
    locator = {
        "query_id": "q1",
        "resource_unit_id": resource_unit_id,
        "physical_node_id": "node-1",
    }

    async def scenario():
        waiter = asyncio.create_task(
            runner_cls.activate_query_udf_actor_pool(
                runner,
                locator,
                _QUERY_GENERATION_CAPABILITY,
            )
        )
        assert await asyncio.to_thread(activation_started.wait, 5)
        shared_task = runner._query_udf_actor_activation_tasks[("q1", resource_unit_id)]
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert runner._query_udf_actor_activation_tasks[("q1", resource_unit_id)] is shared_task
        assert not shared_task.done()
        release_activation.set()
        await shared_task
        return await runner_cls.activate_query_udf_actor_pool(
            runner,
            locator,
            _QUERY_GENERATION_CAPABILITY,
        )

    try:
        assert asyncio.run(scenario()) == {"actor_handles": ["actor-1"]}
    finally:
        release_activation.set()
        native_executor.shutdown(wait=True)
    assert activation_count == 1


def test_phase_retirement_cancels_pending_actor_readiness_without_deadlock(
    monkeypatch,
):
    import vane.execution.udf_ray as udf_ray
    import vane.runners.ray.query_resource_runtime as resource_runtime
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    resource_unit_id = "resource:q1:actor"
    readiness_entered = threading.Event()
    actor_killed = threading.Event()
    submitted = []
    retirements = []

    class _Manager:
        def current_eligible_resource_unit_ids(self):
            return (resource_unit_id,)

        def set_submitted_actor_slots(self, unit_id, actor_indices):
            submitted.append((unit_id, set(actor_indices)))

        def begin_actor_pool_retirement(self, unit_id):
            retirements.append(("begin", unit_id))
            return True

        def complete_actor_pool_retirement(self, unit_id):
            retirements.append(("complete", unit_id))
            return True

    manager = _Manager()
    monkeypatch.setattr(
        resource_runtime,
        "get_query_resource_manager",
        lambda _query_id: manager,
    )

    def shutdown():
        pool.actors = []
        actor_killed.set()

    pool = SimpleNamespace(
        actors=["actor-0"],
        actor_node_ids=[""],
        _init_refs=[object()],
        shutdown=shutdown,
    )
    monkeypatch.setattr(
        udf_ray,
        "prepare_actor_pools_for_nodes",
        lambda *_args, **_kwargs: (
            [pool],
            {
                "node-1": {
                    "actor_handles": ["actor-0"],
                    "actor_node_ids": [""],
                    "actor_dispatch_indices": [],
                    "actor_init_refs": list(pool._init_refs),
                }
            },
        ),
    )

    def wait_for_first_ready(_pool):
        readiness_entered.set()
        assert actor_killed.wait(timeout=5)
        raise RuntimeError("pending actor was killed during phase retirement")

    monkeypatch.setattr(
        udf_ray,
        "wait_for_first_actor_pool_ready",
        wait_for_first_ready,
    )
    runner = SimpleNamespace(
        _query_resource_lock=threading.RLock(),
        _session_lock=threading.RLock(),
        _query_resource_graphs={
            "q1": SimpleNamespace(unit_by_id=lambda _unit_id: SimpleNamespace(backend="ray_actor"))
        },
        _query_allocations={"q1": object()},
        _query_udf_actor_nodes={"q1": {resource_unit_id: {"node_id": "node-1"}}},
        _query_udf_session_configs={"q1": {}},
        _plan_session_ids={"q1": "session-1"},
        _active_udf_actor_by_unit={"q1": {}},
        _active_udf_actors=[],
        _active_udf_actors_by_plan={},
        _driver_handle=object(),
    )
    activation_errors = []

    def activate():
        try:
            runner_cls._activate_query_udf_actor_pool_sync(
                runner,
                {
                    "query_id": "q1",
                    "resource_unit_id": resource_unit_id,
                    "physical_node_id": "node-1",
                },
                _QUERY_GENERATION_CAPABILITY,
            )
        except BaseException as exc:
            activation_errors.append(exc)

    activation_thread = threading.Thread(target=activate)
    activation_thread.start()
    assert readiness_entered.wait(timeout=5)

    runner_cls._retire_udf_actor_pools_outside_phase(runner, "q1", ())
    activation_thread.join(timeout=5)

    assert activation_thread.is_alive() is False
    assert len(activation_errors) == 1
    assert "pending actor was killed" in str(activation_errors[0])
    assert submitted == [(resource_unit_id, {0})]
    assert retirements == [
        ("begin", resource_unit_id),
        ("complete", resource_unit_id),
    ]
    assert runner._active_udf_actor_by_unit["q1"] == {}
    assert runner._active_udf_actors == []
    assert "q1" not in runner._active_udf_actors_by_plan


def test_actor_activation_retains_unpublished_pool_when_shutdown_fails(monkeypatch):
    import vane.execution.udf_ray as udf_ray
    import vane.runners.ray.query_resource_runtime as resource_runtime
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    resource_unit_id = "resource:q1:actor"
    eligibility_reads = 0

    class _Manager:
        def current_eligible_resource_unit_ids(self):
            nonlocal eligibility_reads
            eligibility_reads += 1
            return (resource_unit_id,) if eligibility_reads == 1 else ()

    manager = _Manager()
    monkeypatch.setattr(
        resource_runtime,
        "get_query_resource_manager",
        lambda _query_id: manager,
    )

    def shutdown():
        raise RuntimeError("planned kill failure")

    pool = SimpleNamespace(actors=["actor-0"], shutdown=shutdown)
    monkeypatch.setattr(
        udf_ray,
        "prepare_actor_pools_for_nodes",
        lambda *_args, **_kwargs: (
            [pool],
            {
                "node-1": {
                    "actor_handles": ["actor-0"],
                    "actor_node_ids": [""],
                    "actor_dispatch_indices": [],
                    "actor_init_refs": [],
                }
            },
        ),
    )
    runner = SimpleNamespace(
        _query_resource_lock=threading.RLock(),
        _session_lock=threading.RLock(),
        _query_resource_graphs={
            "q1": SimpleNamespace(unit_by_id=lambda _unit_id: SimpleNamespace(backend="ray_actor"))
        },
        _query_allocations={"q1": object()},
        _query_udf_actor_nodes={"q1": {resource_unit_id: {"node_id": "node-1"}}},
        _query_udf_session_configs={"q1": {}},
        _plan_session_ids={"q1": "session-1"},
        _active_udf_actor_by_unit={"q1": {}},
        _active_udf_actors=[],
        _active_udf_actors_by_plan={},
        _driver_handle=object(),
    )

    with pytest.raises(
        RuntimeError,
        match="activation failed and actor cleanup also failed.*planned kill failure",
    ):
        runner_cls._activate_query_udf_actor_pool_sync(
            runner,
            {
                "query_id": "q1",
                "resource_unit_id": resource_unit_id,
                "physical_node_id": "node-1",
            },
            _QUERY_GENERATION_CAPABILITY,
        )

    assert pool._vane_retired is True
    assert runner._active_udf_actors == [pool]
    assert runner._active_udf_actors_by_plan == {"q1": [pool]}
    assert runner._active_udf_actor_by_unit["q1"] == {}


def test_actor_readiness_cleanup_failure_keeps_registered_pool_owned(monkeypatch):
    import vane.execution.udf_ray as udf_ray
    import vane.runners.ray.query_resource_runtime as resource_runtime
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    resource_unit_id = "resource:q1:actor"
    submitted = []

    class _Manager:
        def current_eligible_resource_unit_ids(self):
            return (resource_unit_id,)

        def set_submitted_actor_slots(self, unit_id, actor_indices):
            submitted.append((unit_id, set(actor_indices)))

    manager = _Manager()
    monkeypatch.setattr(
        resource_runtime,
        "get_query_resource_manager",
        lambda _query_id: manager,
    )

    def shutdown():
        raise RuntimeError("planned readiness kill failure")

    pool = SimpleNamespace(
        actors=["actor-0"],
        actor_node_ids=[""],
        _init_refs=[object()],
        shutdown=shutdown,
    )
    monkeypatch.setattr(
        udf_ray,
        "prepare_actor_pools_for_nodes",
        lambda *_args, **_kwargs: (
            [pool],
            {
                "node-1": {
                    "actor_handles": ["actor-0"],
                    "actor_node_ids": [""],
                    "actor_dispatch_indices": [],
                    "actor_init_refs": list(pool._init_refs),
                }
            },
        ),
    )
    monkeypatch.setattr(
        udf_ray,
        "wait_for_first_actor_pool_ready",
        lambda _pool: (_ for _ in ()).throw(RuntimeError("planned readiness failure")),
    )
    runner = SimpleNamespace(
        _query_resource_lock=threading.RLock(),
        _session_lock=threading.RLock(),
        _query_resource_graphs={
            "q1": SimpleNamespace(unit_by_id=lambda _unit_id: SimpleNamespace(backend="ray_actor"))
        },
        _query_allocations={"q1": object()},
        _query_udf_actor_nodes={"q1": {resource_unit_id: {"node_id": "node-1"}}},
        _query_udf_session_configs={"q1": {}},
        _plan_session_ids={"q1": "session-1"},
        _active_udf_actor_by_unit={"q1": {}},
        _active_udf_actors=[],
        _active_udf_actors_by_plan={},
        _driver_handle=object(),
    )

    with pytest.raises(
        RuntimeError,
        match="activation failed and actor cleanup also failed.*planned readiness kill failure",
    ):
        runner_cls._activate_query_udf_actor_pool_sync(
            runner,
            {
                "query_id": "q1",
                "resource_unit_id": resource_unit_id,
                "physical_node_id": "node-1",
            },
            _QUERY_GENERATION_CAPABILITY,
        )

    assert submitted == [(resource_unit_id, {0})]
    assert pool._vane_retired is True
    assert runner._active_udf_actors == [pool]
    assert runner._active_udf_actors_by_plan == {"q1": [pool]}
    assert "q1" not in runner._active_udf_actor_by_unit


def test_actor_readiness_close_failure_releases_terminated_pool(monkeypatch):
    import vane.execution.udf_ray as udf_ray
    import vane.runners.ray.query_resource_runtime as resource_runtime
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    resource_unit_id = "resource:q1:actor"
    events = []

    class _Manager:
        def current_eligible_resource_unit_ids(self):
            return (resource_unit_id,)

        def set_submitted_actor_slots(self, unit_id, actor_indices):
            events.append(("submitted", unit_id, set(actor_indices)))

        def complete_actor_pool_retirement(self, unit_id):
            events.append(("complete", unit_id))
            return False

        def set_ready_actor_slots(self, unit_id, actor_nodes):
            events.append(("ready", unit_id, dict(actor_nodes)))

    manager = _Manager()
    monkeypatch.setattr(resource_runtime, "get_query_resource_manager", lambda _query_id: manager)

    class _Pool:
        def __init__(self):
            self.actors = ["actor-0"]
            self.actor_node_ids = [""]
            self._init_refs = [object()]

        def shutdown(self):
            self.actors = []
            raise RuntimeError("planned readiness close failure")

    pool = _Pool()
    monkeypatch.setattr(
        udf_ray,
        "prepare_actor_pools_for_nodes",
        lambda *_args, **_kwargs: (
            [pool],
            {
                "node-1": {
                    "actor_handles": ["actor-0"],
                    "actor_node_ids": [""],
                    "actor_dispatch_indices": [],
                    "actor_init_refs": list(pool._init_refs),
                }
            },
        ),
    )
    monkeypatch.setattr(
        udf_ray,
        "wait_for_first_actor_pool_ready",
        lambda _pool: (_ for _ in ()).throw(RuntimeError("planned readiness failure")),
    )
    runner = SimpleNamespace(
        _query_resource_lock=threading.RLock(),
        _session_lock=threading.RLock(),
        _query_resource_graphs={
            "q1": SimpleNamespace(unit_by_id=lambda _unit_id: SimpleNamespace(backend="ray_actor"))
        },
        _query_allocations={"q1": object()},
        _query_udf_actor_nodes={"q1": {resource_unit_id: {"node_id": "node-1"}}},
        _query_udf_session_configs={"q1": {}},
        _plan_session_ids={"q1": "session-1"},
        _active_udf_actor_by_unit={"q1": {}},
        _active_udf_actors=[],
        _active_udf_actors_by_plan={},
        _driver_handle=object(),
    )

    with pytest.raises(
        RuntimeError,
        match="activation failed after graceful callable cleanup also failed.*planned readiness close failure",
    ):
        runner_cls._activate_query_udf_actor_pool_sync(
            runner,
            {
                "query_id": "q1",
                "resource_unit_id": resource_unit_id,
                "physical_node_id": "node-1",
            },
            _QUERY_GENERATION_CAPABILITY,
        )

    assert events == [
        ("submitted", resource_unit_id, {0}),
        ("complete", resource_unit_id),
        ("ready", resource_unit_id, {}),
        ("submitted", resource_unit_id, set()),
    ]
    assert pool._vane_retired is True
    assert runner._active_udf_actor_by_unit == {}
    assert runner._active_udf_actors == []
    assert runner._active_udf_actors_by_plan == {}


def test_udf_actor_pool_shutdown_accepts_query_owned_kill_flag(monkeypatch):
    import vane.execution.udf_ray_actor_pool as actor_pool_mod

    killed = []
    fake_ray = types.SimpleNamespace(
        kill=lambda actor, no_restart=True: killed.append(
            {
                "actor": actor,
                "no_restart": no_restart,
            }
        )
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    pool = actor_pool_mod.UDFActorPoolBase.__new__(actor_pool_mod.UDFActorPoolBase)
    pool._owns_actors = True
    pool.actors = ["actor-0"]

    pool.shutdown(kill=True)

    assert killed == [{"actor": "actor-0", "no_restart": True}]
    assert pool.actors == []


def test_udf_actor_cleanup_diagnostic_handles_unprintable_failure():
    import vane.execution.udf_ray_actor_pool as actor_pool_mod

    class _UnprintableError(RuntimeError):
        def __str__(self):
            raise RuntimeError("planned actor cleanup str failure")

    diagnostic = actor_pool_mod._actor_cleanup_failure("close", 7, _UnprintableError())

    assert diagnostic == "actor-7 close: _UnprintableError: <cleanup error message unavailable>"


def test_udf_actor_pool_constructor_cleans_partial_actor_creation(monkeypatch):
    import vane.execution.udf_ray_actor_pool as actor_pool_mod

    killed = []
    fake_ray = types.SimpleNamespace(
        put=lambda payload: ("payload-ref", payload),
        kill=lambda actor, no_restart=True: killed.append((actor, no_restart)),
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    class _ActorFactory:
        calls = 0

        @classmethod
        def options(cls, **_options):
            class _BoundActor:
                @staticmethod
                def remote():
                    actor_index = cls.calls
                    cls.calls += 1
                    if actor_index == 1:
                        raise RuntimeError("planned actor creation failure")
                    return f"actor-{actor_index}"

            return _BoundActor

    class _Pool(actor_pool_mod.UDFActorPoolBase):
        @staticmethod
        def _actor_class(_max_restarts, _max_task_retries):
            return _ActorFactory

        @staticmethod
        def _resolve_actor_num_cpus(_payload):
            return 1

        @staticmethod
        def _resolve_actor_memory_bytes(_payload):
            return 1

        @staticmethod
        def _build_actor_runtime_env(_ray_options):
            return {}

    with pytest.raises(RuntimeError, match="planned actor creation failure"):
        _Pool(
            payload={},
            concurrency=2,
            gpus_per_actor=0,
        )

    assert killed == [("actor-0", True)]


def test_udf_actor_pool_constructor_exposes_partial_actor_when_cleanup_fails(monkeypatch):
    import vane.execution.udf_ray_actor_pool as actor_pool_mod

    fail_kill = True

    def _kill(_actor, *, no_restart):
        assert no_restart is True
        if fail_kill:
            raise RuntimeError("planned actor cleanup failure")

    fake_ray = types.SimpleNamespace(
        put=lambda payload: ("payload-ref", payload),
        kill=_kill,
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    class _ActorFactory:
        calls = 0

        @classmethod
        def options(cls, **_options):
            class _BoundActor:
                @staticmethod
                def remote():
                    actor_index = cls.calls
                    cls.calls += 1
                    if actor_index == 1:
                        raise RuntimeError("planned actor creation failure")
                    return f"actor-{actor_index}"

            return _BoundActor

    class _Pool(actor_pool_mod.UDFActorPoolBase):
        @staticmethod
        def _actor_class(_max_restarts, _max_task_retries):
            return _ActorFactory

        @staticmethod
        def _resolve_actor_num_cpus(_payload):
            return 1

        @staticmethod
        def _resolve_actor_memory_bytes(_payload):
            return 1

        @staticmethod
        def _build_actor_runtime_env(_ray_options):
            return {}

    with pytest.raises(RuntimeError, match="partial actor cleanup also failed") as exc_info:
        _Pool(
            payload={},
            concurrency=2,
            gpus_per_actor=0,
        )

    owned_pools = exc_info.value.owned_actor_pools
    assert len(owned_pools) == 1
    assert owned_pools[0].actors == ["actor-0"]

    fail_kill = False
    owned_pools[0].shutdown(kill=True)
    assert owned_pools[0].actors == []


def test_ensure_actor_pools_for_plan_does_not_fail_fast_on_cluster_resource_snapshot(monkeypatch):
    import vane.execution.udf_ray as udf_ray

    calls = []

    class _FakeUDFActorPool:
        def __init__(
            self,
            *,
            payload,
            concurrency,
            gpus_per_actor,
            ray_options=None,
        ):
            calls.append(
                {
                    "payload": dict(payload),
                    "concurrency": concurrency,
                    "gpus_per_actor": gpus_per_actor,
                    "ray_options": ray_options,
                }
            )
            self.actors = [f"actor-{idx}" for idx in range(concurrency)]
            self._init_refs = []
            self.actor_node_ids = ["node-a"] * concurrency
            self._confirmed_ready = set(range(concurrency))

    fake_ray = types.ModuleType("ray")
    fake_ray.is_initialized = lambda: True
    fake_ray.cluster_resources = lambda: {"CPU": 0.0, "GPU": 0.0}
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(udf_ray, "_is_vane_worker_process", lambda: False)
    monkeypatch.setattr(udf_ray, "UDFActorPool", _FakeUDFActorPool)

    plan = _FakePlan(
        [
            {
                "node_id": 7,
                "actor_pool_size": 2,
                "payload": {
                    "udf_name": "decode_images",
                    "execution_backend": "ray_actor",
                    "query_id": _TEST_QUERY_ID,
                    "resource_unit_id": "resource:test:actor",
                    "cpus": 1.0,
                },
            }
        ]
    )

    created, handles_map = udf_ray.ensure_actor_pools_for_plan(
        plan,
        query_driver_handle=object(),
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
        session_config={},
        conn=object(),
    )

    assert len(created) == 1
    assert len(calls) == 1
    assert handles_map["7"]["actor_handles"] == ["actor-0", "actor-1"]


def test_ensure_actor_pools_for_plan_publishes_driver_handle_for_ray_task(monkeypatch):
    import vane.execution.udf_ray as udf_ray

    calls = []

    class _FakeUDFActorPool:
        def __init__(self, *, payload, concurrency, gpus_per_actor, ray_options=None):
            calls.append(
                {
                    "payload": dict(payload),
                    "concurrency": concurrency,
                    "gpus_per_actor": gpus_per_actor,
                    "ray_options": ray_options,
                }
            )
            raise AssertionError("python_udf should not create UDFActor pools")

    fake_ray = types.ModuleType("ray")
    fake_ray.is_initialized = lambda: True
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(udf_ray, "_is_vane_worker_process", lambda: False)
    monkeypatch.setattr(udf_ray, "UDFActorPool", _FakeUDFActorPool)

    plan = _FakePlan(
        [
            {
                "node_id": 9,
                "pool_name": "duckdb-udf-vane_document_chunk_paths",
                "actor_pool_size": 2,
                "gpus": 0.0,
                "payload": {
                    "udf_name": "vane_document_chunk_paths",
                    "call_mode": "map",
                    "execution_backend": "ray_task",
                    "scalar_udf_type": "arrow",
                    "return_type": "STRUCT(uploaded_pdf_path VARCHAR)[]",
                    "function_pickle": b"serialized",
                },
            }
        ]
    )

    query_driver_handle = object()
    created, handles_map = udf_ray.ensure_actor_pools_for_plan(
        plan,
        query_driver_handle=query_driver_handle,
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
        session_config={},
        conn=object(),
    )

    assert created == []
    assert handles_map == {
        "9": {
            "query_driver_handle": query_driver_handle,
            "query_generation_capability": _QUERY_GENERATION_CAPABILITY,
            "session_config": {},
        }
    }
    assert plan.set_calls == [{"handles_map": handles_map}]
    assert calls == []


def test_ensure_actor_pools_for_plan_propagates_collect_errors(monkeypatch):
    import vane.execution.udf_ray as udf_ray

    class _BadPlan:
        def collect_udf_nodes(self, conn=None):
            raise RuntimeError("collect failed")

    monkeypatch.setattr(udf_ray, "_is_vane_worker_process", lambda: False)

    with pytest.raises(RuntimeError, match="collect failed"):
        udf_ray.ensure_actor_pools_for_plan(
            _BadPlan(),
            query_driver_handle=object(),
            query_generation_capability=_QUERY_GENERATION_CAPABILITY,
            session_config={},
            conn=object(),
        )


def test_ensure_actor_pools_for_plan_propagates_actor_creation_errors(monkeypatch):
    import vane.execution.udf_ray as udf_ray

    class _FakeRay(types.ModuleType):
        def __init__(self) -> None:
            super().__init__("ray")
            self._initialized = True

        def is_initialized(self):
            return self._initialized

    class _FakeUDFActorPool:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("create failed")

    fake_ray = _FakeRay()
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(udf_ray, "_is_vane_worker_process", lambda: False)
    monkeypatch.setattr(udf_ray, "UDFActorPool", _FakeUDFActorPool)

    plan = _FakePlan(
        [
            {
                "node_id": 7,
                "pool_name": "audio-transcriber",
                "actor_pool_size": 2,
                "gpus": 0.0,
                "payload": {
                    "execution_backend": "ray_actor",
                    "query_id": _TEST_QUERY_ID,
                    "resource_unit_id": "resource:test:actor",
                },
            }
        ]
    )

    with pytest.raises(RuntimeError, match="create failed"):
        udf_ray.ensure_actor_pools_for_plan(
            plan,
            query_driver_handle=object(),
            query_generation_capability=_QUERY_GENERATION_CAPABILITY,
            session_config={},
            conn=object(),
        )
    assert plan.set_calls == []


def _table_from_native_result(result):
    pa = pytest.importorskip("pyarrow")

    payloads = list(result.partition_payloads)
    assert payloads
    if len(payloads) == 1:
        return payloads[0]
    return pa.concat_tables(payloads)


def _build_simple_ray_udf_plan(con):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    import vane

    class AddOne:
        def __call__(self, table):
            values = table.column(0).to_pylist()
            return pa.table({"y": [value + 1 for value in values]})

    relation = con.sql("SELECT 1 AS x UNION ALL SELECT 2 AS x").map_batches(
        AddOne,
        schema={"y": vane.sqltypes.BIGINT},
        execution_backend="ray_actor",
        actor_number=1,
        gpus=0.0,
    )
    plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        f"udf-executor-options-{uuid.uuid4().hex[:8]}",
    ).to_physical_plan(con)
    assert len(plan.collect_udf_nodes(conn=con)) == 1
    return plan


def test_physical_plan_structured_executor_options_reach_udf_builder(monkeypatch):
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    import vane
    import vane.execution.udf as udf_exec
    from vane.execution.ref_bundle import make_local_shm_ref_bundle_result

    build_calls = []

    class _FakeExecutor:
        def __init__(self):
            self._output = []
            self._finished = False
            self._wakeup = None
            self._admission_state = "idle"
            self._retained_input_bytes = 0

        def _notify(self):
            if self._wakeup is not None:
                self._wakeup()

        def submit(self, table):
            values = table.column(0).to_pylist()
            self._output.append(pa.table({"y": [value + 1 for value in values]}))
            self._notify()

        def submit_with_id(self, submit_id, table):
            self._admission_state = "idle"
            values = table.column(0).to_pylist()
            result = pa.table({"y": [value + 1 for value in values]})
            self._output.append(("__vane_submit_result__", int(submit_id), make_local_shm_ref_bundle_result(result)))
            self._notify()

        def take_ready_result(self):
            if not self._output:
                return None
            return self._output.pop(0)

        def finished_submitting(self):
            self._finished = True

        def all_tasks_finished(self):
            return self._finished and not self._output

        def supports_async_wakeup(self):
            return True

        def register_wakeup(self, callback):
            self._wakeup = callback

        def request_task_admission(self, retained_input_bytes):
            if self._admission_state != "idle":
                return False
            self._retained_input_bytes = int(retained_input_bytes)
            self._admission_state = "ready"
            return True

        def task_admission_state(self):
            return {
                "state": self._admission_state,
                "available": self._admission_state == "ready",
                "retained_input_bytes": self._retained_input_bytes,
            }

    def _build_executor(payload, options=None):
        build_calls.append(
            {
                "payload_execution_backend": payload.get("execution_backend"),
                "options": dict(options or {}),
            }
        )
        return _FakeExecutor()

    monkeypatch.setattr(udf_exec, "build_executor", _build_executor)

    con = vane.connect()
    try:
        plan = _build_simple_ray_udf_plan(con)
        query_driver_handle = object()
        plan.set_udf_actor_handles(
            {
                "0": {
                    "actor_handles": ["actor-0"],
                    "actor_node_ids": ["node-a"],
                    "query_driver_handle": query_driver_handle,
                    "query_generation_capability": _QUERY_GENERATION_CAPABILITY,
                    "session_config": {},
                }
            },
            conn=con,
        )

        result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(con.cursor(), plan, None, None)
        table = _table_from_native_result(result)
    finally:
        con.close()

    assert build_calls
    assert all(call["payload_execution_backend"] == "ray_actor" for call in build_calls)
    assert build_calls[0]["options"]["actor_handles"] == ["actor-0"]
    assert build_calls[0]["options"]["actor_node_ids"] == ["node-a"]
    assert build_calls[0]["options"]["query_driver_handle"] is query_driver_handle
    assert build_calls[0]["options"]["query_generation_capability"] == _QUERY_GENERATION_CAPABILITY
    assert build_calls[0]["options"]["session_config"] == {}
    assert sorted(table.column(0).to_pylist()) == [2, 3]


def test_execute_native_udf_cleanup_does_not_deadlock_with_gil_held():
    code = textwrap.dedent(
        """
        from __future__ import annotations

        import gc
        import uuid

        import vane
        import pyarrow as pa
        import vane.execution.udf as udf_exec
        from vane.execution.ref_bundle import make_local_shm_ref_bundle_result


        class _FakeExecutor:
            def __init__(self):
                self._output = []
                self._finished = False
                self._wakeup = None
                self._admission_state = "idle"
                self._retained_input_bytes = 0

            def _notify(self):
                if self._wakeup is not None:
                    self._wakeup()

            def submit(self, table):
                values = table.column(0).to_pylist()
                self._output.append(pa.table({"y": [value + 1 for value in values]}))
                self._notify()

            def submit_with_id(self, submit_id, table):
                self._admission_state = "idle"
                values = table.column(0).to_pylist()
                result = pa.table({"y": [value + 1 for value in values]})
                self._output.append(
                    ("__vane_submit_result__", int(submit_id), make_local_shm_ref_bundle_result(result))
                )
                self._notify()

            def take_ready_result(self):
                if not self._output:
                    return None
                return self._output.pop(0)

            def finished_submitting(self):
                self._finished = True

            def all_tasks_finished(self):
                return self._finished and not self._output

            def supports_async_wakeup(self):
                return True

            def register_wakeup(self, callback):
                self._wakeup = callback

            def request_task_admission(self, retained_input_bytes):
                if self._admission_state != "idle":
                    return False
                self._retained_input_bytes = int(retained_input_bytes)
                self._admission_state = "ready"
                return True

            def task_admission_state(self):
                return {
                    "state": self._admission_state,
                    "available": self._admission_state == "ready",
                    "retained_input_bytes": self._retained_input_bytes,
                }


        def _build_executor(payload, options=None):
            return _FakeExecutor()


        class AddOne:
            def __call__(self, table):
                values = table.column(0).to_pylist()
                return pa.table({"y": [value + 1 for value in values]})


        udf_exec.build_executor = _build_executor

        con = vane.connect()
        cursor = con.cursor()
        relation = con.sql("SELECT 1 AS x UNION ALL SELECT 2 AS x").map_batches(
            AddOne,
            schema={"y": vane.sqltypes.BIGINT},
            execution_backend="ray_actor",
            actor_number=1,
            gpus=0.0,
        )
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
            relation,
            f"udf-cleanup-gil-{uuid.uuid4().hex[:8]}",
        ).to_physical_plan(con)
        plan.set_udf_actor_handles(
            {
                "0": {
                    "actor_handles": ["actor-0"],
                            "query_driver_handle": object(),
                    "query_generation_capability": "test-query-generation-capability",
                }
            },
            conn=con,
        )

        result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(cursor, plan, None, None)
        payloads = list(result.partition_payloads)
        values = [
            value
            for payload in payloads
            for value in payload.column(0).to_pylist()
        ]
        assert sorted(values) == [2, 3], values

        cursor.close()
        con.close()
        del result, plan, relation, cursor, con
        gc.collect()
        print("ok", flush=True)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ok" in proc.stdout


def test_physical_plan_rejects_legacy_list_executor_options(monkeypatch):
    pytest.importorskip("pyarrow")
    import vane
    import vane.execution.udf as udf_exec

    build_call_count = 0

    def _unexpected_build_executor(*_args, **_kwargs):
        nonlocal build_call_count
        build_call_count += 1
        raise AssertionError("udf.build_executor should not run for legacy list executor options")

    monkeypatch.setattr(udf_exec, "build_executor", _unexpected_build_executor)

    con = vane.connect()
    try:
        plan = _build_simple_ray_udf_plan(con)
        plan.set_udf_actor_handles({"0": ["bad-handle"]}, conn=con)

        with pytest.raises(ValueError, match="udf executor options must be a dict"):
            vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(con.cursor(), plan, None, None)
    finally:
        con.close()

    assert build_call_count == 0


def test_ensure_actor_pools_for_plan_publishes_runtime_actor_nodes(monkeypatch):
    import vane.execution.udf_ray as udf_ray

    class _FakeActorsObj:
        def __init__(self, actors):
            self.actors = actors
            self._init_refs = []
            self._confirmed_ready = set()

    class _FakeRay(types.ModuleType):
        def __init__(self) -> None:
            super().__init__("ray")
            self._initialized = True

        def is_initialized(self):
            return self._initialized

    actors = ["actor-0", "actor-1"]
    fake_ray = _FakeRay()

    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(udf_ray, "_is_vane_worker_process", lambda: False)

    class _FakeUDFActorPool:
        def __init__(self, **kwargs):
            self.actors = actors
            self._init_refs = []
            self.actor_node_ids = ["node-a", "node-b"][: int(kwargs["concurrency"])]
            self._confirmed_ready = {0, 1}

    monkeypatch.setattr(udf_ray, "UDFActorPool", _FakeUDFActorPool)

    plan = _FakePlan(
        [
            {
                "node_id": 0,
                "pool_name": "pool-a",
                "actor_pool_size": 2,
                "gpus": 1.0,
                "payload": {
                    "execution_backend": "ray_actor",
                    "query_id": _TEST_QUERY_ID,
                    "resource_unit_id": "resource:test:actor",
                    "gpus": 1.0,
                },
            }
        ]
    )

    created, handles_map = udf_ray.ensure_actor_pools_for_plan(
        plan,
        query_driver_handle=object(),
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
        session_config={},
    )

    assert len(created) == 1
    assert handles_map["0"]["actor_node_ids"] == ["node-a", "node-b"]
    assert plan.set_calls
    assert plan.set_calls[0]["handles_map"]["0"]["actor_node_ids"] == ["node-a", "node-b"]


def test_ensure_actor_pools_waits_for_init_refs_before_ready_lookup(monkeypatch):
    import vane.execution.udf_ray as udf_ray

    class _FakeActorsObj:
        def __init__(self, actors, init_refs):
            self.actors = actors
            self._init_refs = init_refs
            self._confirmed_ready = set()

    class _FakeRay(types.ModuleType):
        def __init__(self) -> None:
            super().__init__("ray")
            self._initialized = True
            self.future_calls = []
            self.init_refs_resolved = False

        def is_initialized(self):
            return self._initialized

    actors = ["actor-0", "actor-1"]
    fake_ray = _FakeRay()

    class _FakeInitRef:
        def __init__(self, index):
            self.index = index

        def future(self):
            ref = self

            class _Future:
                def result(self, timeout=None):
                    fake_ray.future_calls.append((ref, timeout))
                    fake_ray.init_refs_resolved = len(fake_ray.future_calls) == 2
                    return ("node-a", "node-b")[ref.index]

            return _Future()

    init_refs = [_FakeInitRef(0), _FakeInitRef(1)]

    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(udf_ray, "_is_vane_worker_process", lambda: False)

    class _FakeUDFActorPool:
        def __init__(self, **kwargs):
            self.actors = actors
            self._init_refs = init_refs
            self.actor_node_ids = [""] * int(kwargs["concurrency"])
            self._confirmed_ready = set()

    monkeypatch.setattr(udf_ray, "UDFActorPool", _FakeUDFActorPool)

    plan = _FakePlan(
        [
            {
                "node_id": 0,
                "pool_name": "pool-a",
                "actor_pool_size": 2,
                "gpus": 0.0,
                "payload": {
                    "execution_backend": "ray_actor",
                    "query_id": _TEST_QUERY_ID,
                    "resource_unit_id": "resource:test:actor",
                    "gpus": 0.0,
                },
            }
        ]
    )

    _, handles_map = udf_ray.ensure_actor_pools_for_plan(
        plan,
        query_driver_handle=object(),
        query_generation_capability=_QUERY_GENERATION_CAPABILITY,
        session_config={},
    )

    assert [ref for ref, _timeout in fake_ray.future_calls] == init_refs
    assert all(timeout is None for _ref, timeout in fake_ray.future_calls)
    assert handles_map["0"]["actor_dispatch_indices"] == [0, 1]
