# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import queue
import subprocess
import sys
import textwrap
import threading
import types
import uuid
from types import SimpleNamespace

import pytest


class _FakePlan:
    def __init__(self, nodes):
        self._nodes = nodes
        self.set_calls = []

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
        "q-drop-owned",
        "q-drop-owned_orderby_range",
        "q-drop-owned_orderby_sample",
    ]


def test_precreate_udf_actors_skips_non_actor_backend(monkeypatch):
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class

    def _fake_prepare_actor_pools_for_plan(
        plan,
        *,
        actor_node_ids_by_stage,
        conn=None,
    ):
        assert actor_node_ids_by_stage == {}
        created = []
        handles_map = {}
        for node in plan.collect_udf_nodes(conn=conn):
            payload = node.get("payload") or {}
            if payload.get("execution_backend") != "ray_actor":
                continue
        if handles_map:
            plan.set_udf_actor_handles(handles_map, conn=conn)
        return created, handles_map

    fake_mod = types.ModuleType("vane.execution.udf_ray")
    fake_mod.prepare_actor_pools_for_plan = _fake_prepare_actor_pools_for_plan
    monkeypatch.setitem(sys.modules, "vane.execution.udf_ray", fake_mod)

    runner = SimpleNamespace(_duckdb_conn=object(), _active_udf_actors=[])

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
        SimpleNamespace(stages=()),
        SimpleNamespace(actor_node_ids_for_stage=lambda _stage_id: ()),
    )

    assert created == []
    assert plan.set_calls == []


def test_precreate_udf_actors_skips_non_ray_nodes(monkeypatch):
    from vane.runners.ray.driver import RayQueryDriverActor

    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class

    def _fake_prepare_actor_pools_for_plan(*_args, **_kwargs):
        return [], {}

    fake_mod = types.ModuleType("vane.execution.udf_ray")
    fake_mod.prepare_actor_pools_for_plan = _fake_prepare_actor_pools_for_plan
    monkeypatch.setitem(sys.modules, "vane.execution.udf_ray", fake_mod)

    runner = SimpleNamespace(_duckdb_conn=object(), _active_udf_actors=[])

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
        SimpleNamespace(stages=()),
        SimpleNamespace(actor_node_ids_for_stage=lambda _stage_id: ()),
    )

    assert created == []
    assert plan.set_calls == []


def test_driver_udf_actor_handle_hook_is_disabled_by_default(monkeypatch):
    from vane.runners.ray.driver import RayQueryDriverActor

    monkeypatch.delenv("VANE_ENABLE_UDF_TEST_HOOKS", raising=False)
    runner_cls = RayQueryDriverActor.__ray_metadata__.modified_class
    runner = runner_cls.__new__(runner_cls)

    with pytest.raises(RuntimeError, match="VANE_ENABLE_UDF_TEST_HOOKS=1"):
        runner_cls.get_test_udf_actor_handle(runner, "plan-7", "stateful_counter")


def test_stateful_actor_loss_error_includes_stable_query_context():
    from vane.execution.udf_ray_actor_state import format_stateful_actor_loss

    class RayActorError(RuntimeError):
        pass

    cause = RayActorError("actor process exited")
    error = format_stateful_actor_loss(
        {
            "stateful": True,
            "udf_name": "stateful_counter",
            "actor_id": "actor-deadbeef",
        },
        cause,
    )

    assert isinstance(error, RuntimeError)
    assert error is not cause
    assert error.__cause__ is cause
    assert "stateful_counter" in str(error)
    assert "actor-deadbeef" in str(error)
    assert "state was not recoverable" in str(error)


def test_non_stateful_udf_error_is_not_rewritten():
    from vane.execution.udf_ray_actor_state import format_stateful_actor_loss

    error = RuntimeError("ordinary UDF failure")

    assert format_stateful_actor_loss({"stateful": False}, error) is error


def test_stateful_actor_loss_during_readiness_keeps_recoverability_context():
    from vane.execution.udf_ray_remote_readiness import RemoteUDFActorReadinessMixin

    class RayActorError(RuntimeError):
        pass

    class ReadinessHarness(RemoteUDFActorReadinessMixin):
        def __init__(self):
            self._ready_actor_indices = []
            self._actor_init_errors = {0: RayActorError("actor died before ready")}
            self._pending_ready_refs = {}
            self.actors = [object()]

        def _refresh_actor_readiness(self):
            return None

        def error_context(self):
            return {
                "stateful": True,
                "udf_name": "readiness_counter",
                "actor_id": "actor-readiness",
            }

    with pytest.raises(RuntimeError, match="readiness_counter.*actor-readiness.*state was not recoverable"):
        ReadinessHarness()._wait_for_ready_actor()


def test_stateful_actor_loss_during_synchronous_submit_keeps_recoverability_context(monkeypatch):
    import vane.execution.udf_ray_remote_submit as remote_submit
    from vane.execution.udf_ray_remote_submit import RemoteUDFSubmitMixin

    class RayActorError(RuntimeError):
        pass

    class FailingRemoteMethod:
        def options(self, **_kwargs):
            return self

        def remote(self, *_args, **_kwargs):
            raise RayActorError("actor died during submit")

    class Actor:
        run_block_stream = FailingRemoteMethod()

    class SubmitHarness(RemoteUDFSubmitMixin):
        def __init__(self):
            self.actors = [Actor()]
            self._payload = {
                "query_id": "query-submit",
                "stage_id": "stage-submit",
                "execution_backend": "ray_actor",
                "udf_output_target_max_bytes": 1,
            }
            self.unavailable = []

        def _wait_for_ready_actor(self):
            return None

        def _pick_ready_actor_on_node(self, node_id, actor_index):
            assert node_id == "node-submit"
            assert actor_index == 0
            return 0, self.actors[0]

        def _take_task_admission(self):
            return SimpleNamespace(
                lease={
                    "query_id": "query-submit",
                    "stage_id": "stage-submit",
                    "lease_id": "lease-submit",
                    "attempt_id": "attempt-submit",
                    "node_id": "node-submit",
                    "execution_slot_id": "ray_actor:stage-submit:0",
                    "actor_index": 0,
                    "output_window_bytes": 2,
                }
            )

        def _mark_actor_unavailable(self, actor_idx, exc):
            self.unavailable.append((actor_idx, exc))

        def error_context(self):
            return {
                "stateful": True,
                "udf_name": "submit_counter",
                "actor_id": "actor-submit",
            }

    monkeypatch.setattr(
        remote_submit,
        "TaskLeaseObjectRefGenerator",
        lambda *, admission, submitter, **_kwargs: submitter(dict(admission.lease)),
    )

    executor = SubmitHarness()
    with pytest.raises(RuntimeError, match="submit_counter.*actor-submit.*state was not recoverable"):
        executor._submit_one(object(), submit_id=1)
    assert len(executor.unavailable) == 1


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
            actor_node_ids,
            ray_options=None,
        ):
            calls.append(
                {
                    "payload": dict(payload),
                    "concurrency": concurrency,
                    "gpus_per_actor": gpus_per_actor,
                    "actor_node_ids": list(actor_node_ids),
                    "ray_options": ray_options,
                }
            )
            self.actors = [f"actor-{idx}" for idx in range(concurrency)]
            self._init_refs = []
            self._confirmed_ready = set()

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
                    "stage_id": "stage:test:actor",
                },
            }
        ]
    )

    created, handles_map = udf_ray.ensure_actor_pools_for_plan(
        plan,
        actor_node_ids_by_stage={"stage:test:actor": ("node-a", "node-b")},
        conn=object(),
    )

    assert len(created) == 1
    assert "7" in handles_map
    assert len(calls) == 1
    assert calls[0]["actor_node_ids"] == ["node-a", "node-b"]


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
            actor_node_ids,
            ray_options=None,
        ):
            calls.append(
                {
                    "payload": dict(payload),
                    "concurrency": concurrency,
                    "gpus_per_actor": gpus_per_actor,
                    "actor_node_ids": list(actor_node_ids),
                    "ray_options": ray_options,
                }
            )
            self.actors = [f"actor-{idx}" for idx in range(concurrency)]
            self._init_refs = []
            self._confirmed_ready = set()

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
                    "stage_id": "stage:test:actor",
                },
            }
        ]
    )

    created, handles_map = udf_ray.ensure_actor_pools_for_plan(
        plan,
        actor_node_ids_by_stage={"stage:test:actor": ("node-a", "node-b")},
        conn=object(),
    )

    assert len(created) == 1
    assert calls == [
        {
            "payload": {
                "udf_name": "decode_images",
                "execution_backend": "ray_actor",
                "stage_id": "stage:test:actor",
            },
            "concurrency": 2,
            "gpus_per_actor": 0.0,
            "actor_node_ids": ["node-a", "node-b"],
            "ray_options": {"num_cpus": 1.0},
        }
    ]
    assert handles_map["7"]["actor_handles"] == ["actor-0", "actor-1"]
    assert "ray_actor_pool_name" not in handles_map["7"]
    assert plan.set_calls == [
        {
            "handles_map": handles_map,
        }
    ]


def test_ensure_actor_pools_for_plan_disables_restarts_and_retries_for_stateful_udf(monkeypatch):
    import vane.execution.udf_ray as udf_ray

    calls = []

    class _FakeUDFActorPool:
        def __init__(
            self,
            *,
            payload,
            concurrency,
            gpus_per_actor,
            actor_node_ids,
            ray_options=None,
            max_restarts=None,
            max_task_retries=None,
        ):
            calls.append(
                {
                    "payload": dict(payload),
                    "concurrency": concurrency,
                    "actor_node_ids": list(actor_node_ids),
                    "max_restarts": max_restarts,
                    "max_task_retries": max_task_retries,
                }
            )
            self.actors = ["stateful-actor"]
            self._init_refs = []
            self._confirmed_ready = set()

    fake_ray = types.ModuleType("ray")
    fake_ray.is_initialized = lambda: True
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(udf_ray, "_is_vane_worker_process", lambda: False)
    monkeypatch.setattr(udf_ray, "UDFActorPool", _FakeUDFActorPool)

    plan = _FakePlan(
        [
            {
                "node_id": 7,
                "actor_pool_size": 1,
                "gpus": 0.0,
                "payload": {
                    "udf_name": "stateful_counter",
                    "execution_backend": "ray_actor",
                    "stateful": True,
                    "side_effects": True,
                    "actor_number": 1,
                    "stage_id": "stage:test:stateful",
                },
            }
        ]
    )

    created, handles_map = udf_ray.ensure_actor_pools_for_plan(
        plan,
        actor_node_ids_by_stage={"stage:test:stateful": ("node-a",)},
        conn=object(),
    )

    assert len(created) == 1
    assert handles_map["7"]["actor_handles"] == ["stateful-actor"]
    assert calls == [
        {
            "payload": {
                "udf_name": "stateful_counter",
                "execution_backend": "ray_actor",
                "stateful": True,
                "side_effects": True,
                "actor_number": 1,
                "stage_id": "stage:test:stateful",
            },
            "concurrency": 1,
            "actor_node_ids": ["node-a"],
            "max_restarts": 0,
            "max_task_retries": 0,
        }
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {
            "udf_name": "stateless_transform",
            "execution_backend": "ray_actor",
            "stateful": False,
            "side_effects": False,
            "stage_id": "stage:test:retry-policy",
        },
        {
            "udf_name": "ai_prompt",
            "execution_backend": "ray_actor",
            "ai_operation": "prompt",
            "stage_id": "stage:test:retry-policy",
        },
    ],
    ids=["stateless", "ai"],
)
def test_ensure_actor_pools_for_plan_keeps_default_retry_policy_for_non_stateful_udf(monkeypatch, payload):
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
            actor_node_ids,
            ray_options=None,
            max_restarts=MAX_ACTOR_RESTARTS,
            max_task_retries=MAX_ACTOR_TASK_RETRIES,
        ):
            calls.append((max_restarts, max_task_retries))
            self.actors = ["ordinary-actor"]
            self._init_refs = []
            self._confirmed_ready = set()

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
        actor_node_ids_by_stage={"stage:test:retry-policy": ("node-a",)},
        conn=object(),
    )

    assert len(created) == 1
    assert calls == [(MAX_ACTOR_RESTARTS, MAX_ACTOR_TASK_RETRIES)]


def test_local_subprocess_actor_pool_rejects_multi_actor_stateful_payload():
    from vane.execution.udf_subprocess import ensure_local_subprocess_actor_pools_for_nodes

    payload = {
        "udf_name": "stateful_counter",
        "execution_backend": "subprocess_actor",
        "stateful": True,
        "actor_number": 2,
    }

    with pytest.raises(
        ValueError,
        match=r"actor_number must be exactly 1.*multi-actor state",
    ):
        ensure_local_subprocess_actor_pools_for_nodes(
            [{"node_id": 1, "actor_pool_size": 2, "payload": payload}],
            plan_identity="malformed-stateful-plan",
        )


def test_local_stateful_actor_loss_includes_udf_pid_and_recoverability_context():
    from vane.execution.udf_subprocess import LocalSubprocessActorPool

    pool = LocalSubprocessActorPool.__new__(LocalSubprocessActorPool)
    pool.payload = {"udf_name": "local_counter", "stateful": True}
    pool.pool_size = 1
    pool.name = "local-counter-pool"
    pool._lock = threading.Lock()
    pool._active = 0
    pool._idle_workers = queue.Queue()
    pool._idle_workers.put(0)
    pool._workers = [SimpleNamespace(_proc=SimpleNamespace(pid=4242), _actor_lost=True)]

    def fail_after_actor_loss(_worker):
        raise RuntimeError("UDF subprocess communication failed")

    with pytest.raises(RuntimeError, match="local_counter.*pid 4242.*state was not recoverable"):
        pool._run(fail_after_actor_loss)


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
            actor_node_ids,
            ray_options=None,
        ):
            calls.append(
                {
                    "payload": dict(payload),
                    "concurrency": concurrency,
                    "gpus_per_actor": gpus_per_actor,
                    "actor_node_ids": list(actor_node_ids),
                    "ray_options": ray_options,
                }
            )
            self.actors = [f"actor-{idx}" for idx in range(concurrency)]
            self._init_refs = []
            self._confirmed_ready = set()

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
                "stage_id": "stage:test:actor",
            },
        }
    ]

    def inject(handles_map):
        injected.append(handles_map)

    created, handles_map = udf_ray.ensure_actor_pools_for_nodes(
        nodes,
        actor_node_ids_by_stage={"stage:test:actor": ("node-a", "node-b")},
        set_handles=inject,
    )

    assert len(created) == 1
    assert calls == [
        {
            "payload": {
                "udf_name": "decode_images",
                "execution_backend": "ray_actor",
                "stage_id": "stage:test:actor",
            },
            "concurrency": 2,
            "gpus_per_actor": 0.0,
            "actor_node_ids": ["node-a", "node-b"],
            "ray_options": {"num_cpus": 1.0},
        }
    ]
    assert handles_map["9"]["actor_handles"] == ["actor-0", "actor-1"]
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
                    return None

            return _Future()

    class _FakeUDFActorPool:
        def __init__(
            self,
            *,
            payload,
            concurrency,
            gpus_per_actor,
            actor_node_ids,
            ray_options=None,
        ):
            self.actors = [f"actor-{idx}" for idx in range(concurrency)]
            self._init_refs = [_InitRef(idx) for idx in range(concurrency)]
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
                    "stage_id": "stage:test:deferred-ready",
                },
            }
        ]
    )

    pools, handles = udf_ray.prepare_actor_pools_for_plan(
        plan,
        actor_node_ids_by_stage={
            "stage:test:deferred-ready": ("node-a", "node-b"),
        },
    )

    assert resolved == []
    assert handles["5"]["actor_dispatch_indices"] == [0, 1]
    assert plan.set_calls == [{"handles_map": handles}]

    udf_ray.wait_for_actor_pools_ready(pools)

    assert [actor_index for actor_index, _ in resolved] == [0, 1]
    assert pools[0]._confirmed_ready == {0, 1}


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
            actor_node_ids,
            ray_options=None,
        ):
            calls.append(
                {
                    "payload": dict(payload),
                    "concurrency": concurrency,
                    "gpus_per_actor": gpus_per_actor,
                    "actor_node_ids": list(actor_node_ids),
                    "ray_options": ray_options,
                }
            )
            self.actors = [f"actor-{idx}" for idx in range(concurrency)]
            self._init_refs = []
            self._confirmed_ready = set()

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
                    "stage_id": "stage:test:actor",
                    "cpus": 1.0,
                },
            }
        ]
    )

    created, handles_map = udf_ray.ensure_actor_pools_for_plan(
        plan,
        actor_node_ids_by_stage={"stage:test:actor": ("node-a", "node-b")},
        conn=object(),
    )

    assert len(created) == 1
    assert len(calls) == 1
    assert handles_map["7"]["actor_handles"] == ["actor-0", "actor-1"]


def test_ensure_actor_pools_for_plan_skips_python_udf_payload(monkeypatch):
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

    created, handles_map = udf_ray.ensure_actor_pools_for_plan(
        plan,
        actor_node_ids_by_stage={},
        conn=object(),
    )

    assert created == []
    assert handles_map == {}
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
            actor_node_ids_by_stage={},
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
                    "stage_id": "stage:test:actor",
                },
            }
        ]
    )

    with pytest.raises(RuntimeError, match="create failed"):
        udf_ray.ensure_actor_pools_for_plan(
            plan,
            actor_node_ids_by_stage={"stage:test:actor": ("node-a", "node-b")},
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
        streaming_breaker=False,
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
        plan.set_udf_actor_handles(
            {
                "0": {
                    "actor_handles": ["actor-0"],
                    "actor_node_ids": ["node-a"],
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
    assert table.column(0).to_pylist() == [2, 3]


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
            streaming_breaker=False,
        )
        plan = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(
            relation,
            f"udf-cleanup-gil-{uuid.uuid4().hex[:8]}",
        ).to_physical_plan(con)
        plan.set_udf_actor_handles(
            {
                "0": {
                    "actor_handles": ["actor-0"],
                    "actor_node_ids": ["node-a"],
                }
            },
            conn=con,
        )

        result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(cursor, plan, None, None)
        payloads = list(result.partition_payloads)
        assert payloads[0].column(0).to_pylist() == [2, 3]

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


def test_ensure_actor_pools_for_plan_uses_coordinator_actor_nodes(monkeypatch):
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
        def __init__(self, **_kwargs):
            self.actors = actors
            self._init_refs = []
            self._confirmed_ready = set()

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
                    "stage_id": "stage:test:actor",
                    "gpus": 1.0,
                },
            }
        ]
    )

    created, handles_map = udf_ray.ensure_actor_pools_for_plan(
        plan,
        actor_node_ids_by_stage={"stage:test:actor": ("node-a", "node-b")},
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
                    return None

            return _Future()

    init_refs = [_FakeInitRef(0), _FakeInitRef(1)]

    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(udf_ray, "_is_vane_worker_process", lambda: False)

    class _FakeUDFActorPool:
        def __init__(self, **_kwargs):
            self.actors = actors
            self._init_refs = init_refs
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
                    "stage_id": "stage:test:actor",
                    "gpus": 0.0,
                },
            }
        ]
    )

    _, handles_map = udf_ray.ensure_actor_pools_for_plan(
        plan,
        actor_node_ids_by_stage={"stage:test:actor": ("node-a", "node-b")},
    )

    assert [ref for ref, _timeout in fake_ray.future_calls] == init_refs
    assert all(timeout > 0.0 for _ref, timeout in fake_ray.future_calls)
    assert handles_map["0"]["actor_dispatch_indices"] == [0, 1]
