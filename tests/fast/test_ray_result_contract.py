# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import duckdb
from duckdb.runners.common import PartitionMetadata
from duckdb.runners.ray import driver
from duckdb.runners.ray.partition_metadata import PartitionMetadataAccessor
from duckdb.runners.ray.safe_get import QueryDeadlineExceeded
from duckdb.runners.ray.worker import (
    _normalize_native_task_result,
    _validate_fte_output_publication,
)


@contextmanager
def _registered_low_level_plan(plan, con, *, node_id=None):
    """Exercise the internal C++ runner under the mandatory graph contract."""
    from duckdb.runners.ray.query_execution_graph import (
        NodeResourceAllocation,
        QueryAllocation,
        ResourceVector,
    )
    from duckdb.runners.ray.query_graph_builder import build_query_execution_graph
    from duckdb.runners.ray.query_resource_runtime import (
        register_query_graph,
        release_query_resource_manager,
    )

    if node_id is None:
        import ray

        if not ray.is_initialized():
            raise RuntimeError("low-level distributed plan registration requires an initialized Ray runtime")
        node_id = str(ray.get_runtime_context().get_node_id())
    node_id = str(node_id).strip()
    if not node_id:
        raise ValueError("low-level distributed plan registration requires a non-empty node_id")

    graph = build_query_execution_graph(
        plan.collect_execution_stages(conn=con),
        env={
            "VANE_FTE_TASK_HEAP_BYTES": "1",
            "VANE_TARGET_OUTPUT_BLOCK_BYTES": str(1024**2),
        },
    )
    allocation_resources = ResourceVector(
        cpu=128,
        gpu=8,
        heap_bytes=1 << 50,
        object_store_bytes=1 << 50,
    )
    manager = register_query_graph(
        graph,
        QueryAllocation(
            resources=allocation_resources,
            node_allocations=(
                NodeResourceAllocation(
                    node_id=node_id,
                    resources=allocation_resources,
                ),
            ),
            actor_placements=(),
            generation=1,
        ),
    )
    for stage in graph.stages:
        manager.update_stage_state(
            stage.stage_id,
            runnable=True,
            actor_ready=True,
        )
    try:
        yield graph
    finally:
        release_query_resource_manager(graph.query_id, reason="test_complete")


def _make_test_physical_plan(con=None):
    con = duckdb.connect() if con is None else con
    relation = con.sql("SELECT 1 AS i")
    return duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)


class _DummyStream:
    def __init__(self, items):
        self.items = list(items)

    def blocking_next(self):
        if not self.items:
            raise StopIteration
        return self.items.pop(0)


class _FakeOutputLeaseOwner:
    def __init__(self) -> None:
        self.states = ["stage_queue"]
        self.released = False

    def transition_to(self, state):
        self.states.append(str(state))
        return True

    def release(self):
        if self.released:
            return False
        self.released = True
        return True


class _FakePhysicalPlanWithoutPlanAttr:
    def __init__(self, plan_id: str = "fake-plan") -> None:
        self._plan_id = plan_id

    def idx(self) -> str:
        return self._plan_id


class _FakeLogicalPlan:
    def __init__(self, physical_plan: _FakePhysicalPlanWithoutPlanAttr) -> None:
        self.physical_plan = physical_plan

    def to_physical_plan(self, _conn):
        return self.physical_plan


def _make_local_query_driver_actor():
    cls = driver.RayQueryDriverActor.__ray_metadata__.modified_class
    runner = cls.__new__(cls)
    runner.curr_streams = {}
    runner.curr_plans = {}
    runner._plan_query_ids = {}
    runner._query_terminal_errors = {}
    runner._env_overrides = {}
    runner._duckdb_conn = object()
    runner.plan_runner = None
    runner._active_udf_actors = []
    runner._active_udf_actors_by_plan = {}
    runner._active_vllm_actors = []
    return cls, runner


def test_driver_env_override_applies_duckdb_execution_width(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    statements = []

    class _Connection:
        def execute(self, statement):
            statements.append(statement)

    runner._duckdb_conn = _Connection()
    monkeypatch.setenv("VANE_DUCKDB_THREADS", "7")

    cls.install_env_overrides(runner, {"VANE_DUCKDB_THREADS": "7"})

    assert statements == ["SET threads=7"]


def _bind_test_query_resource_owner(
    runner,
    plan_id: str,
    *,
    query_id: str | None = None,
):
    from duckdb.runners.ray.query_execution_graph import (
        NodeResourceAllocation,
        QueryAllocation,
        QueryExecutionGraph,
        ResourceVector,
        StageResourceSpec,
    )
    from duckdb.runners.ray.query_resource_runtime import register_query_graph

    query_id = str(plan_id if query_id is None else query_id)
    stage = StageResourceSpec(
        query_id=query_id,
        stage_id=f"stage:{query_id}:result",
        physical_node_id="result",
        stage_kind="fte",
        backend="ray_worker",
        input_stage_ids=(),
        per_task=ResourceVector(cpu=1, heap_bytes=1),
        target_output_block_bytes=1,
        generator_buffer_blocks=1,
        max_concurrency=1,
    )
    graph = QueryExecutionGraph(
        query_id=query_id,
        plan_digest=f"sha256:{query_id}",
        stages=(stage,),
        terminal_stage_ids=(stage.stage_id,),
    )
    resources = ResourceVector(cpu=1, heap_bytes=1, object_store_bytes=1)
    manager = register_query_graph(
        graph,
        QueryAllocation(
            resources=resources,
            node_allocations=(NodeResourceAllocation(node_id="node-a", resources=resources),),
            actor_placements=(),
            generation=1,
        ),
    )
    manager.update_stage_state(stage.stage_id, runnable=True)
    runner._plan_query_ids[str(plan_id)] = query_id
    return manager


def _fake_task_context_info(task_id):
    task_id = driver.FteTaskAttemptId.coerce(task_id)
    fragment_execution_id = int(task_id.fragment_execution_id)
    partition_id = int(task_id.partition_id)
    return {
        "query_idx": fragment_execution_id,
        "last_node_id": fragment_execution_id,
        "task_id": partition_id,
        "node_ids": [fragment_execution_id],
    }


def _fake_task_attempt_id(task_id):
    return driver.FteTaskAttemptId.coerce(task_id)


def test_ray_progress_snapshot_timeout_returns_none(monkeypatch):
    class _FakeRemoteMethod:
        def remote(self, *_args):
            return "snapshot-ref"

    class _FakeRunner:
        progress_snapshot = _FakeRemoteMethod()

    timeouts = []

    def _fake_get(_ref, *, timeout=None):
        timeouts.append(timeout)
        raise driver.ray.exceptions.GetTimeoutError("snapshot timed out")

    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _fake_get)

    assert driver._ray_progress_snapshot_or_none(_FakeRunner(), "plan-id", 123.0) is None
    assert timeouts == [0.1]


def test_query_driver_run_copy_plan_passes_distributed_physical_plan_wrapper(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    physical_plan = _FakePhysicalPlanWithoutPlanAttr("copy-plan")
    logical_plan = _FakeLogicalPlan(physical_plan)
    captured = {"lifecycle": []}

    class _PlanRunner:
        def run_copy_plan(self, plan, conn):
            captured["plan"] = plan
            captured["conn"] = conn
            return {"ok": True}

    def _precreate_udf_actors(_self, _plan, _graph, _allocation):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        captured["actor_init_thread"] = threading.current_thread().name
        return []

    monkeypatch.setattr(cls, "_precreate_udf_actors", _precreate_udf_actors)
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda _self, _plan: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(
        cls,
        "_register_query_resources",
        lambda _self, _plan: (SimpleNamespace(query_id="copy-plan", stages=()), object()),
    )
    monkeypatch.setattr(cls, "_mark_query_actor_stages_ready", lambda _self, _graph: None)
    monkeypatch.setattr(
        cls,
        "_drop_query_fragments_sync",
        lambda _self, _query_id: captured["lifecycle"].append("teardown"),
    )

    def _final_progress_snapshot(_self, query_id, _started_at):
        captured["lifecycle"].append("snapshot")
        return {"query_id": query_id, "state": "FINISHED"}

    monkeypatch.setattr(cls, "_build_local_progress_snapshot", _final_progress_snapshot)

    outcome = asyncio.run(runner.run_copy_plan(logical_plan))

    assert isinstance(outcome, driver.CopyPlanOutcome)
    assert outcome.result == {"ok": True}
    assert outcome.final_progress_snapshot == {"query_id": "copy-plan", "state": "FINISHED"}
    assert captured["plan"] is physical_plan
    assert captured["conn"] is runner._duckdb_conn
    assert captured["actor_init_thread"].startswith("asyncio_")
    assert captured["lifecycle"] == ["snapshot", "teardown"]


def test_query_driver_run_copy_plan_surfaces_terminal_actor_placement_loss(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    physical_plan = _FakePhysicalPlanWithoutPlanAttr("copy-plan-terminal")
    logical_plan = _FakeLogicalPlan(physical_plan)
    teardown_calls = []

    class _PlanRunner:
        def run_copy_plan(self, _plan, _conn):
            runner._query_terminal_errors["copy-query-terminal"] = "fixed Ray actor placement was lost"
            return {"ok": True}

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args: [])
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(
        cls,
        "_register_query_resources",
        lambda _self, _plan: (
            SimpleNamespace(query_id="copy-query-terminal", stages=()),
            object(),
        ),
    )
    monkeypatch.setattr(cls, "_mark_query_actor_stages_ready", lambda *_args: None)

    def _teardown(_self, plan_id, query_id, *, drop_fragments):
        teardown_calls.append((plan_id, query_id, drop_fragments))
        runner._query_terminal_errors.pop(query_id, None)

    monkeypatch.setattr(cls, "_teardown_plan_resources", _teardown)

    with pytest.raises(RuntimeError, match="fixed Ray actor placement was lost"):
        asyncio.run(runner.run_copy_plan(logical_plan))

    assert teardown_calls == [
        ("copy-plan-terminal", "copy-query-terminal", True),
    ]


def test_query_driver_copy_progress_contract_failure_still_tears_down(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    logical_plan = _FakeLogicalPlan(_FakePhysicalPlanWithoutPlanAttr("copy-progress-contract-failure"))
    teardown_calls = []

    class _PlanRunner:
        def run_copy_plan(self, _plan, _conn):
            return {"ok": True}

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args: [])
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(
        cls,
        "_register_query_resources",
        lambda _self, _plan: (
            SimpleNamespace(query_id="copy-progress-contract-failure", stages=()),
            object(),
        ),
    )
    monkeypatch.setattr(cls, "_mark_query_actor_stages_ready", lambda *_args: None)
    monkeypatch.setattr(
        cls,
        "_build_local_progress_snapshot",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("invalid progress topology")),
    )
    monkeypatch.setattr(
        cls,
        "_teardown_plan_resources",
        lambda _self, plan_id, query_id, *, drop_fragments: teardown_calls.append((plan_id, query_id, drop_fragments)),
    )

    with pytest.raises(RuntimeError, match="invalid progress topology"):
        asyncio.run(runner.run_copy_plan(logical_plan))

    assert teardown_calls == [
        (
            "copy-progress-contract-failure",
            "copy-progress-contract-failure",
            True,
        )
    ]


def test_query_driver_copy_opens_actor_stage_after_topology_and_actor_barriers(monkeypatch):
    import duckdb.runners.ray.fte_fragment_scheduler as scheduler_mod

    cls, runner = _make_local_query_driver_actor()
    physical_plan = _FakePhysicalPlanWithoutPlanAttr("copy-startup-order")
    logical_plan = _FakeLogicalPlan(physical_plan)
    plan_started = threading.Event()
    actor_stage_open = threading.Event()
    events: list[str] = []

    class _PlanRunner:
        def run_copy_plan(self, _plan, _conn):
            events.append("plan-started")
            plan_started.set()
            assert actor_stage_open.wait(timeout=2.0)
            events.append("plan-finished")
            return {"ok": True}

    def wait_for_topology(query_id, *, timeout_s):
        assert query_id == "copy-startup-order"
        assert timeout_s > 0
        assert plan_started.wait(timeout=2.0)
        events.append("topology-ready")

    def wait_for_actors(_self, actor_pools):
        assert actor_pools == ["actor-pool"]
        events.append("actors-ready")

    def mark_actor_stages(_self, _graph):
        events.append("actor-stage-open")
        actor_stage_open.set()

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args: ["actor-pool"])
    monkeypatch.setattr(cls, "_wait_for_udf_actors_ready", wait_for_actors)
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(
        cls,
        "_register_query_resources",
        lambda _self, _plan: (
            SimpleNamespace(query_id="copy-startup-order", stages=()),
            object(),
        ),
    )
    monkeypatch.setattr(cls, "_mark_query_actor_stages_ready", mark_actor_stages)
    monkeypatch.setattr(cls, "_teardown_plan_resources", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cls, "_build_local_progress_snapshot", lambda *_args: {})
    monkeypatch.setattr(
        scheduler_mod,
        "wait_for_fte_query_progress_topology",
        wait_for_topology,
    )

    outcome = asyncio.run(runner.run_copy_plan(logical_plan))

    assert outcome.result == {"ok": True}
    open_index = events.index("actor-stage-open")
    assert events.index("topology-ready") < open_index
    assert events.index("actors-ready") < open_index
    assert open_index < events.index("plan-finished")


def test_query_driver_copy_accepts_plan_success_before_startup_barriers(monkeypatch):
    import duckdb.runners.ray.fte_fragment_scheduler as scheduler_mod

    cls, runner = _make_local_query_driver_actor()
    logical_plan = _FakeLogicalPlan(_FakePhysicalPlanWithoutPlanAttr("copy-plan-before-startup-barriers"))
    plan_finished = threading.Event()
    release_barriers = threading.Event()
    events: list[str] = []

    class _PlanRunner:
        def run_copy_plan(self, _plan, _conn):
            events.append("plan-finished")
            plan_finished.set()
            return {"rows_copied": 0}

    def wait_for_topology(query_id, *, timeout_s):
        assert query_id == "copy-plan-before-startup-barriers"
        assert timeout_s > 0
        assert plan_finished.wait(timeout=2.0)
        assert release_barriers.wait(timeout=2.0)
        events.append("topology-ready")

    def wait_for_actors(_self, actor_pools):
        assert actor_pools == ["actor-pool"]
        assert plan_finished.wait(timeout=2.0)
        assert release_barriers.wait(timeout=2.0)
        events.append("actors-ready")

    def mark_actor_stages(_self, _graph):
        events.append("actor-stage-open")

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args: ["actor-pool"])
    monkeypatch.setattr(cls, "_wait_for_udf_actors_ready", wait_for_actors)
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(
        cls,
        "_register_query_resources",
        lambda _self, _plan: (
            SimpleNamespace(
                query_id="copy-plan-before-startup-barriers",
                stages=(),
            ),
            object(),
        ),
    )
    monkeypatch.setattr(cls, "_mark_query_actor_stages_ready", mark_actor_stages)
    monkeypatch.setattr(cls, "_teardown_plan_resources", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cls, "_build_local_progress_snapshot", lambda *_args: {})
    monkeypatch.setattr(
        scheduler_mod,
        "wait_for_fte_query_progress_topology",
        wait_for_topology,
    )

    release_thread = threading.Thread(
        target=lambda: (plan_finished.wait(timeout=2.0), release_barriers.set()),
        daemon=True,
    )
    release_thread.start()
    outcome = asyncio.run(runner.run_copy_plan(logical_plan))
    release_thread.join(timeout=2.0)

    assert outcome.result == {"rows_copied": 0}
    open_index = events.index("actor-stage-open")
    assert events.index("plan-finished") < events.index("topology-ready") < open_index
    assert events.index("plan-finished") < events.index("actors-ready") < open_index


def test_query_driver_copy_plan_failure_interrupts_startup_barriers(monkeypatch):
    import duckdb.runners.ray.fte_fragment_scheduler as scheduler_mod

    cls, runner = _make_local_query_driver_actor()
    logical_plan = _FakeLogicalPlan(_FakePhysicalPlanWithoutPlanAttr("copy-plan-startup-failure"))
    teardown_started = threading.Event()
    marked_ready = []

    class _PlanRunner:
        def run_copy_plan(self, _plan, _conn):
            raise ValueError("native plan startup failed")

    def wait_until_teardown(*_args, **_kwargs):
        assert teardown_started.wait(timeout=2.0)

    def teardown(*_args, **_kwargs):
        teardown_started.set()

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args: ["actor-pool"])
    monkeypatch.setattr(cls, "_wait_for_udf_actors_ready", wait_until_teardown)
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(
        cls,
        "_register_query_resources",
        lambda _self, _plan: (
            SimpleNamespace(query_id="copy-plan-startup-failure", stages=()),
            object(),
        ),
    )
    monkeypatch.setattr(
        cls,
        "_mark_query_actor_stages_ready",
        lambda *_args: marked_ready.append(True),
    )
    monkeypatch.setattr(cls, "_teardown_plan_resources", teardown)
    monkeypatch.setattr(
        scheduler_mod,
        "wait_for_fte_query_progress_topology",
        wait_until_teardown,
    )

    with pytest.raises(ValueError, match="native plan startup failed"):
        asyncio.run(runner.run_copy_plan(logical_plan))

    assert teardown_started.is_set()
    assert marked_ready == []


@pytest.mark.parametrize(
    ("failing_barrier", "error_type", "message"),
    [
        ("actor", RuntimeError, "actor init failed"),
        ("topology", TimeoutError, "topology init timed out"),
    ],
)
def test_query_driver_copy_startup_barrier_failure_tears_down_plan(
    monkeypatch,
    failing_barrier,
    error_type,
    message,
):
    import duckdb.runners.ray.fte_fragment_scheduler as scheduler_mod

    cls, runner = _make_local_query_driver_actor()
    logical_plan = _FakeLogicalPlan(_FakePhysicalPlanWithoutPlanAttr(f"copy-{failing_barrier}-barrier-failure"))
    teardown_started = threading.Event()
    marked_ready = []

    class _PlanRunner:
        def run_copy_plan(self, _plan, _conn):
            assert teardown_started.wait(timeout=2.0)
            return {"ok": True}

    def wait_for_actors(*_args):
        if failing_barrier == "actor":
            raise RuntimeError(message)

    def wait_for_topology(*_args, **_kwargs):
        if failing_barrier == "topology":
            raise TimeoutError(message)

    def teardown(*_args, **_kwargs):
        teardown_started.set()

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args: ["actor-pool"])
    monkeypatch.setattr(cls, "_wait_for_udf_actors_ready", wait_for_actors)
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(
        cls,
        "_register_query_resources",
        lambda _self, _plan: (
            SimpleNamespace(
                query_id=f"copy-{failing_barrier}-barrier-failure",
                stages=(),
            ),
            object(),
        ),
    )
    monkeypatch.setattr(
        cls,
        "_mark_query_actor_stages_ready",
        lambda *_args: marked_ready.append(True),
    )
    monkeypatch.setattr(cls, "_teardown_plan_resources", teardown)
    monkeypatch.setattr(
        scheduler_mod,
        "wait_for_fte_query_progress_topology",
        wait_for_topology,
    )

    with pytest.raises(error_type, match=message):
        asyncio.run(runner.run_copy_plan(logical_plan))

    assert teardown_started.is_set()
    assert marked_ready == []


def test_ray_query_driver_client_copy_refreshes_progress_and_uses_final_snapshot(monkeypatch):
    class _Future:
        def __init__(self, value, *, timeouts=0):
            self.value = value
            self.timeouts = timeouts

        def result(self, timeout=None):
            if self.timeouts:
                self.timeouts -= 1
                raise FutureTimeoutError
            return self.value

        def done(self):
            return False

    class _Ref:
        def __init__(self, value, *, timeouts=0):
            self._future = _Future(value, timeouts=timeouts)

        def future(self):
            return self._future

    class _RemoteMethod:
        def __init__(self, factory):
            self.factory = factory
            self.calls = []

        def remote(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return self.factory()

    running_snapshot = {
        "query_id": "copy-plan",
        "state": "RUNNING",
        "fragments": [{"id": "fragment-1"}],
    }
    final_snapshot = {
        "query_id": "copy-plan",
        "state": "FINISHED",
        "fragments": [{"id": "fragment-1"}],
    }
    outcome = driver.CopyPlanOutcome(
        result={"rows_copied": 7},
        final_progress_snapshot=final_snapshot,
    )

    class _Runner:
        install_env_overrides = _RemoteMethod(lambda: _Ref(None))
        run_copy_plan = _RemoteMethod(lambda: _Ref(outcome, timeouts=2))
        progress_snapshot = _RemoteMethod(lambda: _Ref(running_snapshot))

    class _Renderer:
        instances = []

        def __init__(self, snapshot_getter):
            self.snapshot_getter = snapshot_getter
            self.interval_s = 0.01
            self.snapshots = []
            self.finish_calls = []
            self.__class__.instances.append(self)

        def update(self):
            self.snapshots.append(self.snapshot_getter())

        def finish(self, **kwargs):
            self.finish_calls.append(kwargs)

    monkeypatch.setattr(driver, "ProgressRenderer", _Renderer)
    monkeypatch.setattr(driver, "progress_enabled", lambda: True)
    monkeypatch.setattr(driver, "_collect_vane_env_overrides", dict)
    client = object.__new__(driver.RayQueryDriverClient)
    client.runner = _Runner()

    result = client.run_copy_plan(_FakePhysicalPlanWithoutPlanAttr("copy-plan"))

    renderer = _Renderer.instances[0]
    assert result == {"rows_copied": 7}
    assert renderer.snapshots == [running_snapshot, running_snapshot]
    assert len(client.runner.progress_snapshot.calls) == 2
    assert renderer.finish_calls == [
        {
            "final_state": "FINISHED",
            "final_snapshot": final_snapshot,
        }
    ]


def test_ray_query_driver_client_stream_waits_through_progress_session(monkeypatch):
    class _Future:
        def __init__(self, value):
            self.value = value

        def result(self, timeout=None):
            return self.value

    class _Ref:
        def __init__(self, value):
            self._future = _Future(value)

        def future(self):
            return self._future

    class _RemoteMethod:
        def __init__(self, value):
            self.value = value

        def remote(self, *_args, **_kwargs):
            return _Ref(self.value)

    partition_ref = object()

    class _Runner:
        install_env_overrides = _RemoteMethod(None)
        run_plan = _RemoteMethod(None)
        get_next_partition = _RemoteMethod(partition_ref)

    class _ProgressSession:
        instances = []

        def __init__(self, runner, plan_id, started_at):
            self.resolved = []
            self.finish_calls = []
            self.__class__.instances.append(self)

        def resolve(self, ref):
            self.resolved.append(ref)
            return None

        def finish(self, **kwargs):
            self.finish_calls.append(kwargs)

    monkeypatch.setattr(driver, "_RayProgressSession", _ProgressSession)
    monkeypatch.setattr(driver, "_collect_vane_env_overrides", dict)
    client = object.__new__(driver.RayQueryDriverClient)
    client.runner = _Runner()

    assert list(client.stream_plan(_FakePhysicalPlanWithoutPlanAttr("stream-plan"))) == []

    progress = _ProgressSession.instances[0]
    assert len(progress.resolved) == 1
    assert progress.resolved[0].future().result() is partition_ref
    assert progress.finish_calls == [{"final_state": "FINISHED"}]


def test_query_driver_run_plan_passes_distributed_physical_plan_wrapper(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    physical_plan = _FakePhysicalPlanWithoutPlanAttr("stream-plan")
    logical_plan = _FakeLogicalPlan(physical_plan)
    stream = _DummyStream([])
    captured = {}

    class _PlanRunner:
        def run_plan(self, plan, conn):
            captured["plan"] = plan
            captured["conn"] = conn
            return stream

    def _precreate_udf_actors(_self, _plan, _graph, _allocation):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        captured["startup_thread"] = threading.current_thread().name
        return []

    monkeypatch.setattr(cls, "_precreate_udf_actors", _precreate_udf_actors)
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda _self, _plan: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(
        cls,
        "_register_query_resources",
        lambda _self, _plan: (SimpleNamespace(query_id="stream-plan", stages=()), object()),
    )
    monkeypatch.setattr(cls, "_mark_query_actor_stages_ready", lambda _self, _graph: None)
    monkeypatch.setattr(
        cls,
        "_release_query_resources",
        lambda _self, _query_id, reason, **_kwargs: None,
    )

    asyncio.run(runner.run_plan(logical_plan))

    assert captured["plan"] is physical_plan
    assert captured["conn"] is runner._duckdb_conn
    assert captured["startup_thread"].startswith("asyncio_")
    assert runner.curr_plans["stream-plan"] is physical_plan
    assert runner.curr_streams["stream-plan"] is stream


def test_query_driver_run_plan_start_failure_runs_complete_teardown(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    physical_plan = _FakePhysicalPlanWithoutPlanAttr("failed-plan")
    logical_plan = _FakeLogicalPlan(physical_plan)
    calls = []

    class _PlanRunner:
        def run_plan(self, _plan, _conn):
            raise ValueError("submission failed")

    def _cleanup_udf_actors(_self, plan_id):
        calls.append(("actors", plan_id))
        raise RuntimeError("actor cleanup failed")

    def _drop_fragments(_self, query_id):
        calls.append(("fragments", query_id))
        raise RuntimeError("fragment cleanup failed")

    monkeypatch.setattr(cls, "_precreate_udf_actors", lambda *_args: [])
    monkeypatch.setattr(cls, "_precreate_vllm_actors", lambda *_args: [])
    monkeypatch.setattr(cls, "_get_plan_runner", lambda _self: _PlanRunner())
    monkeypatch.setattr(
        cls,
        "_register_query_resources",
        lambda _self, _plan: (
            SimpleNamespace(query_id="failed-query", stages=()),
            object(),
        ),
    )
    monkeypatch.setattr(cls, "_mark_query_actor_stages_ready", lambda *_args: None)
    monkeypatch.setattr(cls, "_cleanup_udf_actor_pools", _cleanup_udf_actors)
    monkeypatch.setattr(cls, "_drop_query_fragments_sync", _drop_fragments)

    with pytest.raises(RuntimeError, match="failed to start and teardown also failed") as exc_info:
        asyncio.run(runner.run_plan(logical_plan))

    message = str(exc_info.value)
    assert "submission failed" in message
    assert "actor cleanup failed" in message
    assert "fragment cleanup failed" in message
    assert calls == [
        ("actors", "failed-plan"),
        ("fragments", "failed-query"),
    ]
    assert "failed-plan" not in runner.curr_plans
    assert "failed-plan" not in runner.curr_streams
    assert "failed-plan" not in runner._plan_query_ids


def test_teardown_plan_resources_attempts_every_owned_release(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    plan_id = "teardown-plan"
    calls = []

    class _OutputOwner:
        def release(self):
            calls.append("output")
            raise RuntimeError("output release failed")

    runner.curr_plans[plan_id] = object()
    runner.curr_streams[plan_id] = _DummyStream([])
    runner._plan_query_ids[plan_id] = "teardown-query"
    runner._leased_result_partition_refs = {plan_id: {"0": (object(), _OutputOwner())}}
    runner._result_partition_ref_counters = {plan_id: 1}

    def _cleanup_actors(_self, actual_plan_id):
        calls.append(f"actors:{actual_plan_id}")
        raise RuntimeError("actor release failed")

    def _drop_fragments(_self, query_id):
        calls.append(f"fragments:{query_id}")
        raise RuntimeError("fragment release failed")

    monkeypatch.setattr(cls, "_cleanup_udf_actor_pools", _cleanup_actors)
    monkeypatch.setattr(cls, "_drop_query_fragments_sync", _drop_fragments)

    with pytest.raises(RuntimeError, match="teardown failed") as exc_info:
        cls._teardown_plan_resources(
            runner,
            plan_id,
            "teardown-query",
            drop_fragments=True,
        )

    assert "output release failed" in str(exc_info.value)
    assert "actor release failed" in str(exc_info.value)
    assert "fragment release failed" in str(exc_info.value)
    assert calls == [
        "output",
        "actors:teardown-plan",
        "fragments:teardown-query",
    ]
    assert plan_id not in runner.curr_plans
    assert plan_id not in runner.curr_streams
    assert plan_id not in runner._plan_query_ids
    assert runner._leased_result_partition_refs == {}
    assert runner._result_partition_ref_counters == {}


def test_teardown_fence_failure_retains_retryable_query_ownership(monkeypatch):
    from duckdb.runners.ray.query_resource_runtime import (
        get_query_resource_manager,
        release_query_resource_manager,
    )

    cls, runner = _make_local_query_driver_actor()
    plan_id = "teardown-fence-owner"
    query_id = "teardown-fence-query"
    manager = _bind_test_query_resource_owner(runner, plan_id, query_id=query_id)
    runner.curr_plans[plan_id] = object()
    runner.curr_streams[plan_id] = _DummyStream([])

    def fail_before_owner_release(_self, actual_query_id):
        raise driver.QueryTeardownOwnershipError(f"planned admission fence failure for {actual_query_id}")

    monkeypatch.setattr(cls, "_drop_query_fragments_sync", fail_before_owner_release)

    try:
        with pytest.raises(RuntimeError, match="planned admission fence failure"):
            cls._teardown_plan_resources(
                runner,
                plan_id,
                query_id,
                drop_fragments=True,
            )

        assert runner._plan_query_ids[plan_id] == query_id
        assert get_query_resource_manager(query_id) is manager
    finally:
        release_query_resource_manager(query_id, reason="test_complete")


def test_normalize_native_task_result_preserves_schema_and_stats():
    m = duckdb.ray_cxx
    result = m.NativeDistributedTaskResult(
        ["payload"],
        [m.NativePartitionMetadata(3, 42)],
        {"names": ["x"], "types": ["INTEGER"]},
        [1, 2, 3],
        "ok",
        24601,
        {"attempt_id": 2},
        {"processed_input_rows": 3, "processed_input_bytes": 42},
    )

    (
        payloads,
        partition_metadatas,
        result_schema,
        stats,
        completion_status,
        flight_port,
        exchange_sink_instance,
        task_stats,
    ) = _normalize_native_task_result(result)

    assert payloads == ["payload"]
    assert partition_metadatas == [PartitionMetadata(3, 42)]
    assert result_schema == {"names": ["x"], "types": ["INTEGER"]}
    assert stats == [1, 2, 3]
    assert completion_status == "ok"
    assert flight_port == 24601
    assert exchange_sink_instance == {"attempt_id": 2}
    assert task_stats == {"processed_input_rows": 3, "processed_input_bytes": 42}


def test_normalize_native_task_result_rejects_legacy_shapes():
    with pytest.raises(TypeError, match="execute_native must return NativeDistributedTaskResult"):
        _normalize_native_task_result(([], [], None))


def test_fte_output_publication_is_bounded_by_block_target_and_task_window():
    lease = {
        "lease_id": "lease-1",
        "query_id": "q",
        "stage_id": "stage:q:node:1:fte",
        "attempt_id": "q.0.0.0",
        "target_output_block_bytes": 10,
        "output_window_bytes": 20,
    }

    assert _validate_fte_output_publication(
        [PartitionMetadata(1, 10), PartitionMetadata(0, 0)],
        lease,
    ) == (10, 1)

    with pytest.raises(RuntimeError, match="block 0.*11.*target 10"):
        _validate_fte_output_publication([PartitionMetadata(1, 11)], lease)
    with pytest.raises(RuntimeError, match="total output bytes 21.*window 20"):
        _validate_fte_output_publication(
            [PartitionMetadata(1, 10), PartitionMetadata(1, 10), PartitionMetadata(0, 0)],
            lease,
        )
    with pytest.raises(RuntimeError, match="missing positive size_bytes"):
        _validate_fte_output_publication([PartitionMetadata(1, 0)], lease)


class _RequiredFteWorkerCallbacks:
    def fte_cancel_task(self, _task_id):
        return {"state": "CANCELED"}

    def mark_fte_worker_failed(self, _worker_id, _error):
        return []

    def handle_fte_task_status(self, _status):
        return []

    def fte_attempt_is_selected(self, _task_id):
        return True

    def record_fte_task_terminal(self, _task_id):
        return None

    def finish_fte_task_with_outputs(self, _task_id, _query_task_lease, outputs):
        return [_FakeOutputLeaseOwner() for _ in outputs]

    def fte_ack_task_result(self, _task_id):
        return {"state": "FINISHED"}

    def fte_release_task_result(self, _task_id):
        return {"state": "FINISHED"}

    def enqueue_fte_ack_task_result(self, task_id):
        return self.fte_ack_task_result(task_id)

    def enqueue_fte_release_task_result(self, task_id):
        return self.fte_release_task_result(task_id)


class _FakeFteStatusWorker(_RequiredFteWorkerCallbacks):
    worker_id = "worker-a"

    def __init__(self):
        self.status = {"state": "RUNNING"}
        self.calls = []
        self.terminal_attempts = []
        self.ack_calls = []
        self.release_calls = []
        self.output_transfers = []

    def fte_get_task_status(self, _task_id):
        raise AssertionError("FTE task handles must use status wait, not status polling")

    def fte_wait_task_status(self, task_id, min_version, timeout_s):
        self.calls.append(("wait", task_id, min_version, timeout_s))
        status = dict(self.status)
        status.setdefault("task_id", task_id)
        return status

    def fte_cancel_task(self, task_id):
        self.calls.append(("cancel", task_id))
        self.status = {"state": "CANCELED", "task_id": task_id}
        return dict(self.status)

    def fte_ack_task_result(self, task_id):
        self.ack_calls.append(task_id)
        return {"state": "FINISHED", "task_id": task_id}

    def fte_release_task_result(self, task_id):
        self.release_calls.append(task_id)
        return {"state": "FINISHED", "task_id": task_id}

    def record_fte_task_terminal(self, task_id):
        self.terminal_attempts.append(str(task_id))

    def finish_fte_task_with_outputs(self, task_id, query_task_lease, outputs):
        self.output_transfers.append((str(driver.FteTaskAttemptId.coerce(task_id)), dict(query_task_lease), outputs))
        return [_FakeOutputLeaseOwner() for _ in outputs]


def _wait_batch_ready(handle, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ready = driver.batch_wait_ready([handle])
        if ready:
            return ready
        time.sleep(0.01)
    return driver.batch_wait_ready([handle])


def test_fte_worker_task_handle_finishes_via_status_wait():
    worker = _FakeFteStatusWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    handle = driver.FteWorkerTaskHandle(task_id, worker)

    assert handle.done() is False
    worker.status = {"state": "FINISHED", "stats": [1, 2, 3]}
    assert _wait_batch_ready(handle) == [0]

    result = handle.get_result_sync()
    assert result.ok
    assert result.result_schema is None
    assert worker.calls[0] == ("wait", task_id, -1, handle.status_wait_timeout_s)
    assert worker.terminal_attempts == ["q.1.2.0"]


def test_fte_worker_task_status_transition_runs_off_event_loop():
    worker = _FakeFteStatusWorker()
    worker.status = {"state": "FINISHED", "stats": [1]}
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    handle = driver.FteWorkerTaskHandle(task_id, worker)
    original_apply_status = handle._apply_status
    transition_threads = []

    def _apply_status(status):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        transition_threads.append(threading.current_thread().name)
        original_apply_status(status)

    handle._apply_status = _apply_status

    assert _wait_batch_ready(handle) == [0]
    assert handle.get_result_sync().ok
    assert transition_threads
    assert transition_threads[0].startswith("asyncio_")


def test_fte_worker_task_handle_starts_one_watcher_under_concurrent_polling(
    monkeypatch,
):
    class _SingleTransferWorker(_FakeFteStatusWorker):
        def __init__(self):
            super().__init__()
            self._transfer_lock = threading.Lock()
            self._transferred = False

        def finish_fte_task_with_outputs(
            self,
            task_id,
            query_task_lease,
            outputs,
        ):
            with self._transfer_lock:
                if self._transferred:
                    raise RuntimeError("FTE task lease is not active")
                self._transferred = True
            return super().finish_fte_task_with_outputs(
                task_id,
                query_task_lease,
                outputs,
            )

    worker = _SingleTransferWorker()
    task_id = {
        "query_id": "q",
        "fragment_execution_id": 1,
        "partition_id": 2,
        "attempt_id": 0,
    }
    worker.status = {
        "state": "FINISHED",
        "result": (["payload"], [{"num_rows": 5, "size_bytes": 64}], None, []),
    }
    handle = driver.FteWorkerTaskHandle(
        task_id,
        worker,
        query_task_lease={"lease_id": "lease-result"},
    )
    background_loop = driver._ensure_background_event_loop()
    first_lookup_entered = threading.Event()
    release_first_lookup = threading.Event()
    second_lookup_entered = threading.Event()
    lookup_lock = threading.Lock()
    lookup_count = 0

    def racing_get_event_loop():
        nonlocal lookup_count
        with lookup_lock:
            lookup_count += 1
            current = lookup_count
        if current == 1:
            first_lookup_entered.set()
            assert release_first_lookup.wait(timeout=2)
        else:
            second_lookup_entered.set()
        return background_loop

    monkeypatch.setattr(driver, "_get_global_event_loop", racing_get_event_loop)
    polls = [threading.Thread(target=handle.done) for _ in range(2)]
    polls[0].start()
    assert first_lookup_entered.wait(timeout=2)
    polls[1].start()
    second_lookup_entered.wait(timeout=0.2)
    release_first_lookup.set()
    for poll in polls:
        poll.join(timeout=2)
        assert not poll.is_alive()

    assert lookup_count == 1
    assert _wait_batch_ready(handle) == [0]
    assert handle.get_result_sync().ok
    assert len(worker.output_transfers) == 1
    assert len(worker.calls) == 1


def test_fte_finish_wins_atomically_over_concurrent_cancel():
    ack_entered = threading.Event()
    release_ack = threading.Event()

    class _BlockingAckWorker(_FakeFteStatusWorker):
        def enqueue_fte_ack_task_result(self, task_id):
            ack_entered.set()
            assert release_ack.wait(timeout=2)
            return super().enqueue_fte_ack_task_result(task_id)

    worker = _BlockingAckWorker()
    task_id = {
        "query_id": "q",
        "fragment_execution_id": 1,
        "partition_id": 2,
        "attempt_id": 0,
    }
    handle = driver.FteWorkerTaskHandle(
        task_id,
        worker,
        query_task_lease={"lease_id": "lease-finish-cancel"},
    )
    status = {
        "state": "FINISHED",
        "task_id": task_id,
        "result": (["payload"], [{"num_rows": 5, "size_bytes": 64}], None, []),
    }
    finishing = threading.Thread(target=handle._apply_status, args=(status,))
    finishing.start()
    assert ack_entered.wait(timeout=2)

    cancelling = threading.Thread(target=handle.cancel)
    cancelling.start()
    release_ack.set()
    finishing.join(timeout=2)
    cancelling.join(timeout=2)

    assert not finishing.is_alive()
    assert not cancelling.is_alive()
    result = handle.get_result_sync()
    assert result.has_output is True
    assert [call[0] for call in worker.calls if call[0] == "cancel"] == []
    assert worker.release_calls == []
    assert len(worker.output_transfers) == 1


def test_fte_terminal_record_failure_is_not_masked_by_adopted_result():
    class _TerminalRecordFailWorker(_FakeFteStatusWorker):
        def __init__(self):
            super().__init__()
            self.new_owners = []

        def finish_fte_task_with_outputs(self, task_id, query_task_lease, outputs):
            self.output_transfers.append(
                (
                    str(driver.FteTaskAttemptId.coerce(task_id)),
                    dict(query_task_lease),
                    outputs,
                )
            )
            self.new_owners = [_FakeOutputLeaseOwner() for _ in outputs]
            return list(self.new_owners)

        def record_fte_task_terminal(self, _task_id):
            raise RuntimeError("planned terminal record failure")

    worker = _TerminalRecordFailWorker()
    worker.status = {
        "state": "FINISHED",
        "result": (["payload"], [{"num_rows": 5, "size_bytes": 64}], None, []),
    }
    handle = driver.FteWorkerTaskHandle(
        {
            "query_id": "q",
            "fragment_execution_id": 1,
            "partition_id": 2,
            "attempt_id": 0,
        },
        worker,
        query_task_lease={"lease_id": "lease-terminal-record"},
    )

    assert _wait_batch_ready(handle) == [0]
    with pytest.raises(RuntimeError, match="planned terminal record failure"):
        handle.get_result_sync()
    assert len(worker.new_owners) == 1
    assert worker.new_owners[0].released is True
    assert worker.release_calls == [handle.task_id.to_dict()]


def test_fte_worker_task_handle_requires_status_wait_protocol():
    class _StatusOnlyWorker(_RequiredFteWorkerCallbacks):
        def fte_get_task_status(self, task_id):
            return {"state": "FINISHED", "task_id": task_id, "stats": [1]}

    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    with pytest.raises(RuntimeError, match="must provide fte_wait_task_status"):
        driver.FteWorkerTaskHandle(task_id, _StatusOnlyWorker())


def test_fte_worker_task_handle_requires_worker_id():
    class _NoWorkerIdWorker(_RequiredFteWorkerCallbacks):
        def fte_wait_task_status(self, task_id, _min_version, _timeout_s):
            return {"state": "FINISHED", "task_id": task_id, "stats": [1]}

    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    with pytest.raises(RuntimeError, match="non-empty worker_id"):
        driver.FteWorkerTaskHandle(task_id, _NoWorkerIdWorker())


def test_fte_worker_task_handle_finishes_with_result_payload():
    worker = _FakeFteStatusWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    schema = {"names": ["x"], "types": ["INTEGER"]}
    worker.status = {
        "state": "FINISHED",
        "result": (["payload"], [{"num_rows": 5, "size_bytes": 64}], schema, [9]),
    }
    handle = driver.FteWorkerTaskHandle(
        task_id,
        worker,
        query_task_lease={"lease_id": "lease-result"},
    )

    assert _wait_batch_ready(handle) == [0]

    result = handle.get_result_sync()
    assert result.ok
    assert result.result_schema == schema
    assert worker.output_transfers == [
        (
            "q.1.2.0",
            {"lease_id": "lease-result"},
            [{"block_id": "fte-block:lease-result:0", "size_bytes": 64}],
        )
    ]
    assert worker.terminal_attempts == ["q.1.2.0"]


def test_fte_worker_task_handle_rolls_back_new_output_owners_when_replacing_owner_fails():
    class _RaisingPreviousOwner:
        def transition_to(self, _state):
            return True

        def release(self):
            raise RuntimeError("previous owner release failed")

    class _TrackingWorker(_FakeFteStatusWorker):
        def __init__(self):
            super().__init__()
            self.new_owners = []

        def finish_fte_task_with_outputs(self, task_id, query_task_lease, outputs):
            self.output_transfers.append(
                (str(driver.FteTaskAttemptId.coerce(task_id)), dict(query_task_lease), outputs)
            )
            self.new_owners = [_FakeOutputLeaseOwner() for _ in outputs]
            return list(self.new_owners)

    worker = _TrackingWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    previous_ref = duckdb.ray_cxx.RayResultPartitionRef(
        "payload",
        5,
        64,
        _RaisingPreviousOwner(),
    )
    handle = driver.FteWorkerTaskHandle(
        task_id,
        worker,
        query_task_lease={"lease_id": "lease-result"},
    )

    with pytest.raises(RuntimeError, match="previous owner release failed"):
        handle._normalize_raw_result(([previous_ref], [{"num_rows": 5, "size_bytes": 64}], None, []))

    assert len(worker.new_owners) == 1
    assert worker.new_owners[0].released is True


def test_fte_worker_task_handle_releases_partial_invalid_ownership_transfer():
    class _InvalidTransferWorker(_FakeFteStatusWorker):
        def __init__(self):
            super().__init__()
            self.owner = _FakeOutputLeaseOwner()

        def finish_fte_task_with_outputs(self, _task_id, _query_task_lease, _outputs):
            return [self.owner]

    worker = _InvalidTransferWorker()
    handle = driver.FteWorkerTaskHandle(
        {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0},
        worker,
    )

    with pytest.raises(RuntimeError, match="one owner per output"):
        handle._finish_task_output_ownership(
            [
                {"block_id": "block-0", "size_bytes": 1},
                {"block_id": "block-1", "size_bytes": 1},
            ]
        )

    assert worker.owner.released is True


def test_fte_worker_task_handle_releases_adopted_and_remote_results_when_ack_fails():
    class _AckFailWorker(_FakeFteStatusWorker):
        def __init__(self):
            super().__init__()
            self.new_owners = []

        def finish_fte_task_with_outputs(self, task_id, query_task_lease, outputs):
            self.output_transfers.append(
                (str(driver.FteTaskAttemptId.coerce(task_id)), dict(query_task_lease), outputs)
            )
            self.new_owners = [_FakeOutputLeaseOwner() for _ in outputs]
            return list(self.new_owners)

        def fte_ack_task_result(self, task_id):
            self.ack_calls.append(task_id)
            raise RuntimeError("planned ack failure")

    worker = _AckFailWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    worker.status = {
        "state": "FINISHED",
        "result": (["payload"], [{"num_rows": 5, "size_bytes": 64}], None, []),
    }
    handle = driver.FteWorkerTaskHandle(
        task_id,
        worker,
        query_task_lease={"lease_id": "lease-result"},
    )

    assert _wait_batch_ready(handle) == [0]
    with pytest.raises(RuntimeError, match="planned ack failure"):
        handle.get_result_sync()

    assert worker.ack_calls == [task_id]
    assert worker.release_calls == [task_id]
    assert len(worker.new_owners) == 1
    assert worker.new_owners[0].released is True


def test_fte_worker_task_handle_defers_attempt_selection_to_query_commit():
    class _SelectionRacingWorker(_FakeFteStatusWorker):
        def fte_attempt_is_selected(self, _task_id):
            raise AssertionError("result adoption must not race the query-level selected-attempt decision")

    worker = _SelectionRacingWorker()
    task_id = {
        "query_id": "q",
        "fragment_execution_id": 1,
        "partition_id": 2,
        "attempt_id": 1,
    }
    worker.status = {
        "state": "FINISHED",
        "result": (["loser-ref"], [{"num_rows": 5, "size_bytes": 64}], None, []),
    }
    handle = driver.FteWorkerTaskHandle(
        task_id,
        worker,
        query_task_lease={"lease_id": "loser-lease"},
    )

    assert _wait_batch_ready(handle) == [0]
    result = handle.get_result_sync()

    assert result.ok
    assert worker.ack_calls == [task_id]
    assert worker.release_calls == []
    assert worker.output_transfers == [
        (
            "q.1.2.1",
            {"lease_id": "loser-lease"},
            [{"block_id": "fte-block:loser-lease:0", "size_bytes": 64}],
        )
    ]
    assert worker.terminal_attempts == ["q.1.2.1"]


def test_fte_worker_task_handle_acks_remote_result_once():
    worker = _FakeFteStatusWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    worker.status = {
        "state": "FINISHED",
        "result": (["payload"], [{"num_rows": 5, "size_bytes": 64}], None, []),
    }
    handle = driver.FteWorkerTaskHandle(task_id, worker)

    assert _wait_batch_ready(handle) == [0]
    assert handle.get_result_sync().ok
    assert handle.get_result_sync().ok

    assert worker.ack_calls == [task_id]


def test_fte_worker_task_handle_ack_does_not_release_remote_result():
    worker = _FakeFteStatusWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    worker.status = {
        "state": "FINISHED",
        "result": (["payload"], [{"num_rows": 5, "size_bytes": 64}], None, []),
    }
    handle = driver.FteWorkerTaskHandle(task_id, worker)

    assert _wait_batch_ready(handle) == [0]
    assert handle.get_result_sync().ok

    assert worker.ack_calls == [task_id]
    assert worker.release_calls == []


def test_fte_worker_task_handle_release_result_payload_calls_worker_once():
    worker = _FakeFteStatusWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    handle = driver.FteWorkerTaskHandle(task_id, worker)

    handle.release_result_payload()
    handle.release_result_payload()

    assert worker.release_calls == [task_id]


def test_fte_worker_task_handle_enqueues_result_controls_without_sync_rpc():
    class _QueuedControlWorker(_FakeFteStatusWorker):
        def __init__(self):
            super().__init__()
            self.queued_controls = []

        def fte_ack_task_result(self, _task_id):
            raise AssertionError("result ACK must not synchronously resolve a Ray control RPC")

        def fte_release_task_result(self, _task_id):
            raise AssertionError("result release must not synchronously resolve a Ray control RPC")

        def enqueue_fte_ack_task_result(self, task_id):
            self.queued_controls.append(("ack", task_id))

        def enqueue_fte_release_task_result(self, task_id):
            self.queued_controls.append(("release", task_id))

    worker = _QueuedControlWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    handle = driver.FteWorkerTaskHandle(task_id, worker)
    handle._result = driver.RayTaskResult.success([], [1], None)

    assert handle.get_result_sync().ok
    handle.release_result_payload()

    assert worker.queued_controls == [("ack", task_id), ("release", task_id)]


def test_fte_worker_task_handle_does_not_publish_finished_status_event():
    class _EventWorker(_FakeFteStatusWorker):
        def __init__(self):
            super().__init__()
            self.status_events = []

        def handle_fte_task_status(self, status):
            self.status_events.append(dict(status))
            return []

    worker = _EventWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    worker.status = {"state": "FINISHED", "stats": [1]}
    handle = driver.FteWorkerTaskHandle(task_id, worker)

    assert _wait_batch_ready(handle) == [0]

    assert worker.status_events == []


def test_fte_worker_task_handle_does_not_publish_failed_status_event():
    class _FailingEventWorker(_FakeFteStatusWorker):
        def handle_fte_task_status(self, _status):
            raise RuntimeError("publish exploded")

    worker = _FailingEventWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    worker.status = {
        "state": "FAILED",
        "failure": {"message": "remote failed"},
    }
    handle = driver.FteWorkerTaskHandle(task_id, worker)

    assert _wait_batch_ready(handle) == [0]

    with pytest.raises(RuntimeError, match="remote failed"):
        handle.get_result_sync()
    assert worker.terminal_attempts == ["q.1.2.0"]


def test_fte_worker_task_handle_does_not_adopt_retry_from_data_plane():
    class _EventWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-a"

        def fte_wait_task_status(self, task_id, _min_version, _timeout_s):
            return {
                "state": "FAILED",
                "task_id": task_id,
                "failure": {"message": "retryable"},
                "version": 1,
            }

        def handle_fte_task_status(self, _status):
            raise AssertionError("the authoritative status watcher owns retry scheduling")

    handle = driver.FteWorkerTaskHandle(
        {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0},
        _EventWorker(),
    )

    with pytest.raises(RuntimeError, match="retryable"):
        asyncio.run(handle.get_result())
    assert handle.task_id.attempt_id == 0


def test_fte_worker_task_handle_malformed_status_fails_worker_and_records_terminal_once():
    class _MalformedStatusWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-malformed-status"

        def __init__(self):
            self.worker_failures = []
            self.terminal_attempts = []

        def fte_wait_task_status(self, task_id, _min_version, _timeout_s):
            return {
                "state": "MALFORMED",
                "task_id": task_id,
                "version": 1,
            }

        def mark_fte_worker_failed(self, worker_id, error):
            self.worker_failures.append((worker_id, error))
            return []

        def record_fte_task_terminal(self, task_id):
            self.terminal_attempts.append(str(driver.FteTaskAttemptId.coerce(task_id)))

    worker = _MalformedStatusWorker()
    handle = driver.FteWorkerTaskHandle(
        {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0},
        worker,
    )

    with pytest.raises(RuntimeError, match="failed to apply FTE task status.*MALFORMED"):
        asyncio.run(handle.get_result())

    assert handle.done() is True
    assert len(worker.worker_failures) == 1
    assert worker.worker_failures[0][0] == "worker-malformed-status"
    assert "status protocol failed" in worker.worker_failures[0][1]
    assert worker.terminal_attempts == ["q.1.2.0"]


def test_fte_worker_task_handle_rejects_mismatched_status_identity():
    class _MismatchedStatusWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-mismatched-status"

        def __init__(self):
            self.worker_failures = []
            self.terminal_attempts = []

        def fte_wait_task_status(self, _task_id, _min_version, _timeout_s):
            return {
                "state": "FINISHED",
                "task_id": {
                    "query_id": "q",
                    "fragment_execution_id": 1,
                    "partition_id": 99,
                    "attempt_id": 0,
                },
                "version": 1,
            }

        def mark_fte_worker_failed(self, worker_id, error):
            self.worker_failures.append((worker_id, error))
            return []

        def record_fte_task_terminal(self, task_id):
            self.terminal_attempts.append(str(driver.FteTaskAttemptId.coerce(task_id)))

    worker = _MismatchedStatusWorker()
    handle = driver.FteWorkerTaskHandle(
        {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0},
        worker,
    )

    with pytest.raises(RuntimeError, match="status identity mismatch"):
        asyncio.run(handle.get_result())

    assert len(worker.worker_failures) == 1
    assert worker.worker_failures[0][0] == "worker-mismatched-status"
    assert worker.terminal_attempts == ["q.1.2.0"]


def test_fte_worker_task_handle_treats_query_deadline_as_hard_failure():
    class _DeadlineWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-query-deadline"

        def __init__(self):
            self.worker_failures = []
            self.terminal_attempts = []

        def fte_wait_task_status(self, _task_id, _min_version, _timeout_s):
            raise QueryDeadlineExceeded("query deadline expired before Ray ObjectRef get")

        def mark_fte_worker_failed(self, worker_id, error):
            self.worker_failures.append((worker_id, error))
            return []

        def record_fte_task_terminal(self, task_id):
            self.terminal_attempts.append(str(driver.FteTaskAttemptId.coerce(task_id)))

    worker = _DeadlineWorker()
    handle = driver.FteWorkerTaskHandle(
        {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0},
        worker,
    )

    with pytest.raises(QueryDeadlineExceeded, match="query deadline expired"):
        asyncio.run(handle.get_result())

    assert len(worker.worker_failures) == 1
    assert worker.terminal_attempts == ["q.1.2.0"]


def test_fte_worker_task_handle_rejects_exchange_finish_without_final_info():
    worker = _FakeFteStatusWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    worker.status = {"state": "FINISHED"}
    handle = driver.FteWorkerTaskHandle(
        task_id,
        worker,
        task_context_info={"exchange_sink_instance": {"attempt_id": 0}},
    )

    assert _wait_batch_ready(handle) == [0]
    with pytest.raises(RuntimeError, match="FINISHED without final task info"):
        handle.get_result_sync()
    assert worker.terminal_attempts == ["q.1.2.0"]


def test_fte_worker_task_handle_accepts_exchange_finish_with_spooling_stats():
    worker = _FakeFteStatusWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    worker.status = {
        "state": "FINISHED",
        "spooling_output_stats": {"rows": 3},
    }
    handle = driver.FteWorkerTaskHandle(
        task_id,
        worker,
        task_context_info={"exchange_sink_instance": {"attempt_id": 0}},
    )

    assert _wait_batch_ready(handle) == [0]
    result = handle.get_result_sync()

    assert result.ok
    assert result.exchange_sink_instance == {"attempt_id": 0}
    assert worker.terminal_attempts == ["q.1.2.0"]


def test_fte_worker_task_handle_uses_status_long_poll_when_available():
    class _LongPollWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-a"

        def __init__(self):
            self.calls = []

        def fte_wait_task_status(self, task_id, min_version, timeout_s):
            self.calls.append(("wait", task_id, min_version, timeout_s))
            return {
                "state": "FINISHED",
                "task_id": task_id,
                "version": 7,
                "stats": [4, 5, 6],
            }

        def fte_get_task_status(self, _task_id):
            raise AssertionError("long-poll path should not use synchronous status polling")

    worker = _LongPollWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    handle = driver.FteWorkerTaskHandle(task_id, worker)

    result = asyncio.run(handle.get_result())

    assert result.ok
    assert worker.calls == [("wait", task_id, -1, handle.status_wait_timeout_s)]


def test_fte_worker_task_handle_get_result_sync_accepts_prepopulated_result():
    class _SlowLongPollWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-a"

        def fte_wait_task_status(self, task_id, _min_version, _timeout_s):
            time.sleep(60)
            return {"state": "RUNNING", "task_id": task_id, "version": 1}

    handle = driver.FteWorkerTaskHandle(
        {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0},
        _SlowLongPollWorker(),
    )
    handle._result = driver.RayTaskResult.success([], [1, 2, 3], None)

    assert _wait_batch_ready(handle) == [0]
    result = handle.get_result_sync()

    assert result.ok
    assert handle.done() is True


def test_fte_worker_task_handle_publishes_worker_loss_without_adopting_retry():
    class _RetryWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-b"

        def fte_wait_task_status(self, task_id, _min_version, _timeout_s):
            return {
                "state": "FINISHED",
                "task_id": task_id,
                "version": 1,
                "stats": [8],
            }

    class _LostWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-a"

        def __init__(self):
            self.calls = []

        def fte_wait_task_status(self, task_id, min_version, timeout_s):
            self.calls.append(("wait", task_id, min_version, timeout_s))
            raise RuntimeError("actor lost")

        def mark_fte_worker_failed(self, worker_id, error):
            self.calls.append(("mark_failed", worker_id, error))
            return [
                driver.FteWorkerTaskHandle(
                    {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 1},
                    _RetryWorker(),
                )
            ]

    worker = _LostWorker()
    handle = driver.FteWorkerTaskHandle(
        {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0},
        worker,
    )

    with pytest.raises(RuntimeError, match="actor lost"):
        asyncio.run(handle.get_result())
    assert handle.task_id.attempt_id == 0
    assert worker.calls[0][0] == "wait"
    assert worker.calls[1][0] == "mark_failed"


def test_fte_worker_task_handle_does_not_pop_scheduler_result_registry_after_worker_lost():
    class _RetryWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-b"

        def fte_wait_task_status(self, task_id, _min_version, _timeout_s):
            return {
                "state": "FINISHED",
                "task_id": task_id,
                "version": 1,
                "stats": [13],
            }

        def record_fte_task_terminal(self, _task_id):
            return None

    class _Coordinator:
        worker_id = "coordinator"

        def __init__(self):
            self.retry = driver.FteWorkerTaskHandle(
                {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 1},
                _RetryWorker(),
            )
            self.pop_calls = 0

        def pop_fte_result_handle_for_task(self, _task_id):
            self.pop_calls += 1
            retry = self.retry
            self.retry = None
            return retry

    class _LostWorker(_RequiredFteWorkerCallbacks):
        worker_id = "worker-a"

        def fte_wait_task_status(self, _task_id, _min_version, _timeout_s):
            raise RuntimeError("actor lost")

        def mark_fte_worker_failed(self, _worker_id, _error):
            return []

        def pop_fte_result_handle_for_task(self, task_id):
            return coordinator.pop_fte_result_handle_for_task(task_id)

    coordinator = _Coordinator()
    handle = driver.FteWorkerTaskHandle(
        {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0},
        _LostWorker(),
    )

    with pytest.raises(RuntimeError, match="actor lost"):
        asyncio.run(handle.get_result())

    assert handle.task_id.attempt_id == 0
    assert handle.worker_id == "worker-a"
    assert coordinator.pop_calls == 0


def test_fte_worker_task_handle_failed_status_raises_result_error():
    worker = _FakeFteStatusWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    handle = driver.FteWorkerTaskHandle(task_id, worker)
    worker.status = {"state": "FAILED", "failure": {"message": "boom"}}

    assert _wait_batch_ready(handle) == [0]
    with pytest.raises(RuntimeError, match="boom"):
        handle.get_result_sync()
    assert worker.terminal_attempts == ["q.1.2.0"]


def test_fte_worker_task_handle_cancel_calls_worker():
    worker = _FakeFteStatusWorker()
    task_id = {"query_id": "q", "fragment_execution_id": 1, "partition_id": 2, "attempt_id": 0}
    handle = driver.FteWorkerTaskHandle(task_id, worker)

    handle.cancel()

    assert handle.done() is True
    assert worker.calls == [("cancel", task_id)]
    assert worker.release_calls == [task_id]


def test_get_next_partition_wraps_metadata_aware_fragment(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    plan_id = "plan-ok"
    manager = _bind_test_query_resource_owner(runner, plan_id)
    payload = object()
    fragment = duckdb.ray_cxx.RayResultPartitionRef(payload, 7, 99, _FakeOutputLeaseOwner())
    runner.curr_streams[plan_id] = _DummyStream([fragment])
    runner.curr_plans[plan_id] = object()

    class _LocalMetadataAccessor:
        def __init__(self, metadatas):
            self._metadatas = list(metadatas)

        def get_index(self, key: int):
            return self._metadatas[key]

    monkeypatch.setattr(
        PartitionMetadataAccessor,
        "from_metadata_list",
        classmethod(lambda _cls, meta: _LocalMetadataAccessor(meta)),
    )

    result = asyncio.run(cls.get_next_partition(runner, plan_id))

    assert result is not None
    assert result.partition_ref() is payload
    assert result.partition() is payload
    assert result.metadata() == PartitionMetadata(7, 99)
    assert manager.snapshot()["external_consumer_waiting"] is False


def test_get_next_partition_leases_and_releases_metadata_aware_fragment(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    plan_id = "plan-lease"
    _bind_test_query_resource_owner(runner, plan_id)
    payload = object()
    output_owner = _FakeOutputLeaseOwner()
    fragment = duckdb.ray_cxx.RayResultPartitionRef(payload, 7, 99, output_owner)
    runner.curr_streams[plan_id] = _DummyStream([fragment])
    runner.curr_plans[plan_id] = object()

    released = []

    class _FakeRemoteMethod:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *args):
            return self._fn(*args)

    class _FakeReleaseOwner:
        def __init__(self):
            self.release_result_partition_ref = _FakeRemoteMethod(self._release)

        def _release(self, owner_plan_id, release_token):
            released.append((owner_plan_id, release_token))
            return cls.release_result_partition_ref(runner, owner_plan_id, release_token)

    class _LocalMetadataAccessor:
        def __init__(self, metadatas):
            self._metadatas = list(metadatas)

        def get_index(self, key: int):
            return self._metadatas[key]

    monkeypatch.setattr(
        PartitionMetadataAccessor,
        "from_metadata_list",
        classmethod(lambda _cls, meta: _LocalMetadataAccessor(meta)),
    )
    monkeypatch.setattr(driver, "resolve_object_refs_blocking", lambda value, **_kwargs: value)

    result = asyncio.run(
        cls.get_next_partition(
            runner,
            plan_id,
            release_owner=_FakeReleaseOwner(),
        )
    )

    assert result is not None
    assert result.partition() is payload
    assert released == [(plan_id, "0")]
    assert output_owner.released is True
    assert runner._leased_result_partition_refs == {}


def test_get_next_partition_releases_by_lease_id_not_object_ref(monkeypatch):
    cls, runner = _make_local_query_driver_actor()
    plan_id = "plan-lease-id"
    _bind_test_query_resource_owner(runner, plan_id)
    payload = object()
    output_owner = _FakeOutputLeaseOwner()
    fragment = duckdb.ray_cxx.RayResultPartitionRef(payload, 7, 99, output_owner)
    runner.curr_streams[plan_id] = _DummyStream([fragment])
    runner.curr_plans[plan_id] = object()

    released = []

    class _FakeRemoteMethod:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *args):
            return self._fn(*args)

    class _FakeReleaseOwner:
        def __init__(self):
            self.release_result_partition_ref = _FakeRemoteMethod(self._release)

        def _release(self, owner_plan_id, release_token):
            assert owner_plan_id == plan_id
            assert release_token is not payload
            assert isinstance(release_token, str)
            released.append(release_token)
            return cls.release_result_partition_ref(runner, owner_plan_id, release_token)

    class _LocalMetadataAccessor:
        def __init__(self, metadatas):
            self._metadatas = list(metadatas)

        def get_index(self, key: int):
            return self._metadatas[key]

    monkeypatch.setattr(
        PartitionMetadataAccessor,
        "from_metadata_list",
        classmethod(lambda _cls, meta: _LocalMetadataAccessor(meta)),
    )
    monkeypatch.setattr(driver, "resolve_object_refs_blocking", lambda value, **_kwargs: value)

    result = asyncio.run(
        cls.get_next_partition(
            runner,
            plan_id,
            release_owner=_FakeReleaseOwner(),
        )
    )

    assert result is not None
    assert result.partition() is payload
    assert released == ["0"]
    assert output_owner.released is True
    assert runner._leased_result_partition_refs == {}


def test_get_next_partition_rejects_unleased_arrow_payload():
    pa = pytest.importorskip("pyarrow")
    cls, runner = _make_local_query_driver_actor()
    plan_id = "plan-arrow"
    _bind_test_query_resource_owner(runner, plan_id)
    table = pa.table({"x": [1, 2, 3]})
    runner.curr_streams[plan_id] = _DummyStream([table])
    runner.curr_plans[plan_id] = object()

    with pytest.raises(TypeError, match="expected metadata-aware fragment"):
        asyncio.run(cls.get_next_partition(runner, plan_id))


def test_get_next_partition_rejects_non_metadata_aware_fragment():
    cls, runner = _make_local_query_driver_actor()
    plan_id = "plan-bad"
    _bind_test_query_resource_owner(runner, plan_id)
    runner.curr_streams[plan_id] = _DummyStream([{"rows": 1}])
    runner.curr_plans[plan_id] = object()

    with pytest.raises(TypeError, match="expected metadata-aware fragment"):
        asyncio.run(cls.get_next_partition(runner, plan_id))


def test_get_next_partition_surfaces_terminal_actor_placement_loss_before_delivery():
    cls, runner = _make_local_query_driver_actor()
    plan_id = "plan-placement-lost"
    query_id = "query-placement-lost"
    _bind_test_query_resource_owner(runner, plan_id, query_id=query_id)
    undelivered = object()
    runner.curr_streams[plan_id] = _DummyStream([undelivered])
    runner.curr_plans[plan_id] = object()
    runner._query_terminal_errors[query_id] = "fixed Ray actor placement was lost"
    teardown_calls = []

    def _teardown(actual_plan_id, actual_query_id, *, drop_fragments):
        teardown_calls.append((actual_plan_id, actual_query_id, drop_fragments))
        runner._query_terminal_errors.pop(actual_query_id, None)

    runner._teardown_plan_resources = _teardown

    with pytest.raises(RuntimeError, match="fixed Ray actor placement was lost"):
        asyncio.run(cls.get_next_partition(runner, plan_id))

    assert teardown_calls == [(plan_id, query_id, True)]
    assert runner.curr_streams[plan_id].items == [undelivered]
    from duckdb.runners.ray.query_resource_runtime import release_query_resource_manager

    release_query_resource_manager(query_id, reason="test_complete")


def test_get_next_partition_waits_for_teardown_without_blocking_event_loop():
    cls, runner = _make_local_query_driver_actor()
    plan_id = "plan-end"
    _bind_test_query_resource_owner(runner, plan_id, query_id="query-end")
    runner.curr_streams[plan_id] = _DummyStream([])
    runner.curr_plans[plan_id] = object()

    cleanup_started = threading.Event()
    cleanup_release = threading.Event()
    cleanup_calls = []

    def _slow_drop_query_fragments(query_id: str) -> None:
        cleanup_calls.append(query_id)
        cleanup_started.set()
        cleanup_release.wait(timeout=1.0)

    runner._drop_query_fragments_sync = _slow_drop_query_fragments

    async def _consume_to_completion():
        consume_task = asyncio.create_task(cls.get_next_partition(runner, plan_id))
        assert await asyncio.to_thread(cleanup_started.wait, 1.0)
        assert consume_task.done() is False

        # Cleanup runs on a worker thread: the driver event loop stays responsive,
        # while query completion remains fenced on deterministic teardown.
        await asyncio.sleep(0)
        assert consume_task.done() is False
        cleanup_release.set()
        return await asyncio.wait_for(consume_task, timeout=1.0)

    try:
        result = asyncio.run(_consume_to_completion())

        assert result is None
        assert cleanup_calls == ["query-end"]
        assert plan_id not in runner.curr_streams
        assert plan_id not in runner.curr_plans
        assert plan_id not in runner._plan_query_ids
    finally:
        cleanup_release.set()


def test_close_plan_runs_blocking_teardown_off_actor_event_loop():
    cls, runner = _make_local_query_driver_actor()
    cleanup_threads = []

    def _cleanup(plan_id):
        assert plan_id == "plan-close"
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        cleanup_threads.append(threading.current_thread().name)

    runner._cleanup_finished_plan = _cleanup

    async def _close():
        await cls.close_plan(runner, "plan-close")

    asyncio.run(_close())

    assert len(cleanup_threads) == 1
    assert cleanup_threads[0].startswith("asyncio_")


def test_fragment_stats_runs_worker_observation_off_actor_event_loop():
    cls, runner = _make_local_query_driver_actor()
    stats_started = threading.Event()
    stats_release = threading.Event()

    class _PlanRunner:
        def fragment_stats(self):
            with pytest.raises(RuntimeError, match="no running event loop"):
                asyncio.get_running_loop()
            stats_started.set()
            stats_release.wait(timeout=1.0)
            return {"fragment_count": 3}

    runner._get_plan_runner = lambda: _PlanRunner()

    async def _observe_without_blocking():
        heartbeat = asyncio.Event()
        asyncio.get_running_loop().call_soon(heartbeat.set)
        task = asyncio.create_task(cls.fragment_stats(runner))
        assert await asyncio.to_thread(stats_started.wait, 1.0)
        await asyncio.wait_for(heartbeat.wait(), timeout=0.1)
        assert task.done() is False
        stats_release.set()
        return await asyncio.wait_for(task, timeout=1.0)

    try:
        stats = asyncio.run(_observe_without_blocking())
    finally:
        stats_release.set()

    assert stats == {"fragment_count": 3}


def test_progress_snapshot_build_runs_off_actor_event_loop(monkeypatch):
    from duckdb.runners.ray import fte_fragment_scheduler

    cls, runner = _make_local_query_driver_actor()
    build_started = threading.Event()
    build_release = threading.Event()

    def _slow_registry_snapshot(query_id):
        assert query_id == "query-progress"
        build_started.set()
        build_release.wait(timeout=1.0)
        return {
            "queries": {
                query_id: {
                    "query_id": query_id,
                    "fragment_executions": {},
                }
            }
        }

    monkeypatch.setattr(
        fte_fragment_scheduler,
        "fte_progress_registry_snapshot",
        _slow_registry_snapshot,
    )
    monkeypatch.setattr(
        fte_fragment_scheduler,
        "fte_registry_stats",
        lambda: (_ for _ in ()).throw(AssertionError("hot progress path must not use the diagnostic registry dump")),
    )

    async def _snapshot_without_blocking():
        heartbeat = asyncio.Event()
        asyncio.get_running_loop().call_soon(heartbeat.set)
        watchdog = threading.Timer(0.5, build_release.set)
        watchdog.start()
        try:
            started_at = time.monotonic()
            progress_call = cls.progress_snapshot(runner, "query-progress", 0.0)
            call_elapsed = time.monotonic() - started_at

            assert call_elapsed < 0.05
            assert hasattr(progress_call, "__await__")
            task = asyncio.create_task(progress_call)
            assert await asyncio.to_thread(build_started.wait, 1.0)
            await asyncio.wait_for(heartbeat.wait(), timeout=0.1)
            assert task.done() is False
            build_release.set()
            return await asyncio.wait_for(task, timeout=1.0)
        finally:
            build_release.set()
            watchdog.cancel()

    snapshot = asyncio.run(_snapshot_without_blocking())

    assert snapshot["query_id"] == "query-progress"


def test_progress_snapshot_returns_cached_value_while_refresh_runs():
    cls, runner = _make_local_query_driver_actor()
    refresh_started = threading.Event()
    refresh_release = threading.Event()
    build_count = 0

    def _snapshot(query_id, _started_at):
        nonlocal build_count
        build_count += 1
        if build_count == 1:
            return {"query_id": query_id, "version": 1}
        refresh_started.set()
        refresh_release.wait(timeout=1.0)
        return {"query_id": query_id, "version": 2}

    runner._build_local_progress_snapshot = _snapshot

    async def _read_cached_during_refresh():
        first = await cls.progress_snapshot(runner, "query-cache-progress", 0.0)
        await asyncio.sleep(0)
        second_call = asyncio.create_task(cls.progress_snapshot(runner, "query-cache-progress", 0.0))
        assert await asyncio.to_thread(refresh_started.wait, 1.0)
        second = await asyncio.wait_for(second_call, timeout=0.1)
        assert runner._progress_snapshot_builds
        refresh_release.set()
        await asyncio.sleep(0)
        return first, second

    try:
        first, second = asyncio.run(_read_cached_during_refresh())
    finally:
        refresh_release.set()

    assert first == {"query_id": "query-cache-progress", "version": 1}
    assert second == first


def test_progress_snapshot_state_is_cancelled_and_dropped_with_query():
    cls, runner = _make_local_query_driver_actor()
    build_started = threading.Event()
    build_release = threading.Event()

    def _slow_snapshot(query_id, _started_at):
        build_started.set()
        build_release.wait(timeout=1.0)
        return {"query_id": query_id}

    runner._build_local_progress_snapshot = _slow_snapshot

    async def _drop_active_snapshot():
        progress = asyncio.create_task(cls.progress_snapshot(runner, "query-drop-progress", 0.0))
        assert await asyncio.to_thread(build_started.wait, 1.0)
        cls._drop_progress_snapshot_state(runner, "query-drop-progress")
        with pytest.raises(asyncio.CancelledError):
            await progress
        build_release.set()
        await asyncio.sleep(0)

    try:
        asyncio.run(_drop_active_snapshot())
    finally:
        build_release.set()

    assert runner._progress_snapshot_builds == {}
    assert runner._progress_snapshot_cache == {}


def test_execute_native_empty_result_returns_typed_contract():
    con = duckdb.connect()
    con.execute("CREATE TABLE a AS SELECT i FROM range(10) tbl(i)")
    relation = con.sql("SELECT * FROM a WHERE 1=0")
    plan = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)

    runner = duckdb.ray_cxx.DistributedPhysicalPlanRunner()
    cursor = con.cursor()
    result = runner.execute_native(cursor, plan, None, None)

    assert isinstance(result, duckdb.ray_cxx.NativeDistributedTaskResult)
    assert result.completion_status == "empty"
    assert list(result.partition_payloads) == []
    assert list(result.partition_metadatas) == []
    assert result.result_schema["types"] == ["BIGINT"]


def test_describe_native_progress_materializes_deferred_clone_without_execution(tmp_path):
    import ray

    con = duckdb.connect()
    src = tmp_path / "progress_topology_input.parquet"
    con.execute(f"COPY (SELECT i::INTEGER AS i FROM range(10) tbl(i)) TO '{src}' (FORMAT PARQUET)")
    plan = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        con.sql(f"SELECT i * 2 AS value FROM read_parquet('{src}')"),
        str(uuid.uuid4()),
    ).to_physical_plan(con)
    deferred = ray.cloudpickle.loads(ray.cloudpickle.dumps(plan))
    assert deferred.has_root() is False

    runner = duckdb.ray_cxx.DistributedPhysicalPlanRunner()
    topology = duckdb.ray_cxx.describe_native_progress(con.cursor(), deferred)

    assert deferred.has_root() is False
    assert set(topology) == {"schema", "pipelines"}
    assert topology["schema"] == "pipeline_topology"
    assert topology["pipelines"]
    assert any("TABLE_SCAN" in pipeline["operators"] for pipeline in topology["pipelines"])
    assert all(
        set(pipeline) == {"pipeline_id", "operators", "operator_details", "stage_ids"}
        for pipeline in topology["pipelines"]
    )
    result_collector_roles = {
        pipeline["operator_details"][index].get("pipeline_role")
        for pipeline in topology["pipelines"]
        for index, operator in enumerate(pipeline["operators"])
        if operator == "RESULT_COLLECTOR"
    }
    assert result_collector_roles == {"source", "sink"}

    result = runner.execute_native(con.cursor(), deferred, None, None)
    assert result.completion_status == "ok"
    assert sum(metadata.num_rows for metadata in result.partition_metadatas) == 10
    final_pipelines = result.task_stats["pipelines"]
    assert [(pipeline["pipeline_id"], pipeline["operators"]) for pipeline in final_pipelines] == [
        (pipeline["pipeline_id"], pipeline["operators"]) for pipeline in topology["pipelines"]
    ]
    assert all(
        pipeline["total_pipeline_tasks"] > 0
        and pipeline["completed_pipeline_tasks"] == pipeline["total_pipeline_tasks"]
        and pipeline["queued_pipeline_tasks"] == 0
        and pipeline["running_pipeline_tasks"] == 0
        for pipeline in final_pipelines
    )
    scan_pipeline = next(pipeline for pipeline in final_pipelines if "TABLE_SCAN" in pipeline["operators"])
    assert scan_pipeline["input_rows"] == 10


def test_remote_exchange_sink_progress_does_not_add_result_collector(tmp_path, monkeypatch):
    import duckdb.runners.ray.worker_handle as ray_worker_handle

    class _CapturingWorker:
        def __init__(self):
            self.tasks = []

        def submit_tasks(self, tasks):
            self.tasks.extend(tasks)
            return []

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def task_input_stream_exhausted_for_query(self, _query_id, _source_node_ids):
            return []

        def shutdown(self):
            return None

    worker = _CapturingWorker()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids: [duckdb.ray_cxx.RayWorkerRuntime("node-a", worker, 4.0, 0.0, 8 << 30)],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    con = duckdb.connect()
    src = tmp_path / "remote_exchange_progress.parquet"
    con.sql("SELECT i::INTEGER AS i FROM range(32) tbl(i)").write_parquet(str(src))
    relation = con.read_parquet(str(src)).repartition(2)
    plan = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)
    runner = duckdb.ray_cxx.DistributedPhysicalPlanRunner()

    sink_topologies = []
    sink_results = []
    with _registered_low_level_plan(plan, con, node_id="node-a"):
        stream = runner.run_plan(plan, con)
        with pytest.raises(StopIteration):
            stream.blocking_next()

        for task in worker.tasks:
            task_plan = task.plan()
            topology = duckdb.ray_cxx.describe_native_progress(con.cursor(), task_plan)
            operators = [operator for pipeline in topology["pipelines"] for operator in pipeline["operators"]]
            if "EXCHANGE_SINK" in operators:
                sink_topologies.append(topology)
                sink_results.append(
                    runner.execute_native(
                        con.cursor(),
                        task_plan,
                        exchange_sink_instance=task.exchange_sink_instance(),
                    )
                )

    assert sink_topologies
    assert len(sink_results) == len(sink_topologies)
    assert all(
        "RESULT_COLLECTOR" not in pipeline["operators"]
        for topology in sink_topologies
        for pipeline in topology["pipelines"]
    )
    for topology, result in zip(sink_topologies, sink_results, strict=True):
        final_pipelines = result.task_stats["pipelines"]
        assert [(pipeline["pipeline_id"], pipeline["operators"]) for pipeline in final_pipelines] == [
            (pipeline["pipeline_id"], pipeline["operators"]) for pipeline in topology["pipelines"]
        ]
        assert sum(pipeline["input_rows"] for pipeline in final_pipelines) == 32
        assert all(
            pipeline["total_pipeline_tasks"] > 0
            and pipeline["completed_pipeline_tasks"] == pipeline["total_pipeline_tasks"]
            and pipeline["queued_pipeline_tasks"] == 0
            and pipeline["running_pipeline_tasks"] == 0
            for pipeline in final_pipelines
        )
    assert all(
        pipeline["operators"] != ["EXCHANGE_SINK"] for topology in sink_topologies for pipeline in topology["pipelines"]
    )


def test_distributed_physical_plan_clone_executes_on_worker_connection():
    driver_con = duckdb.connect()
    worker_con = duckdb.connect()
    relation = driver_con.sql("SELECT 42::INTEGER AS i")
    plan = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(driver_con)

    clone = plan.clone(worker_con)

    assert clone is not plan
    assert clone.idx() == plan.idx()
    assert clone.has_root() is True
    assert plan.has_root() is True

    runner = duckdb.ray_cxx.DistributedPhysicalPlanRunner()
    clone_result = runner.execute_native(worker_con.cursor(), clone, None, None)
    original_result = runner.execute_native(driver_con.cursor(), plan, None, None)

    assert clone_result.completion_status == "ok"
    assert original_result.completion_status == "ok"
    assert list(clone_result.partition_payloads)[0].column(0).to_pylist() == [42]
    assert list(original_result.partition_payloads)[0].column(0).to_pylist() == [42]


def test_execute_native_repartition_uses_local_exchange_not_passthrough():
    con = duckdb.connect()
    con.execute("SET threads=4")
    relation = con.sql("SELECT i::INTEGER AS i FROM range(32) tbl(i)").repartition(4)
    plan = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)

    runner = duckdb.ray_cxx.DistributedPhysicalPlanRunner()
    result = runner.execute_native(con.cursor(), plan, None, None)

    assert result.completion_status == "ok"
    payloads = list(result.partition_payloads)
    assert len(payloads) == 1
    values = payloads[0].column(0).to_pylist()
    assert sorted(values) == list(range(32))
    assert values != list(range(32))
    pipelines = list(result.task_stats.get("pipelines") or [])
    assert any("REPARTITION" in list(pipeline.get("operators") or []) for pipeline in pipelines)


def test_execute_native_hash_repartition_uses_resolved_partition_expressions():
    con = duckdb.connect()
    con.execute("SET threads=4")
    relation = con.sql("SELECT i::INTEGER AS i, (i % 2)::INTEGER AS k FROM range(1000) tbl(i)").repartition(4, "k")
    plan = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)

    runner = duckdb.ray_cxx.DistributedPhysicalPlanRunner()
    result = runner.execute_native(con.cursor(), plan, None, None)

    assert result.completion_status == "ok"
    metadatas = list(result.partition_metadatas)
    assert sum(metadata.num_rows for metadata in metadatas) == 1000
    pipelines = list(result.task_stats.get("pipelines") or [])
    assert any("REPARTITION" in list(pipeline.get("operators") or []) for pipeline in pipelines)


def test_execute_native_applies_dynamic_filter_domains_to_table_scan(tmp_path):
    pa = pytest.importorskip("pyarrow")

    con = duckdb.connect()
    src = tmp_path / "dynamic_filter_input.parquet"
    con.sql(
        """
        select i::integer as id, (i * 10)::integer as value
        from range(0, 8) tbl(i)
        """
    ).write_parquet(str(src))
    relation = con.sql(f"select id, value from read_parquet('{src}') order by id")
    plan = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)

    runner = duckdb.ray_cxx.DistributedPhysicalPlanRunner()
    result = runner.execute_native(
        con.cursor(),
        plan,
        dynamic_filter_domains={"df0": {"column": "id", "range": [2, 4]}},
    )

    assert isinstance(result, duckdb.ray_cxx.NativeDistributedTaskResult)
    payloads = list(result.partition_payloads)
    assert len(payloads) == 1
    table = payloads[0]
    assert isinstance(table, pa.Table)
    assert table.column(0).to_pylist() == [2, 3, 4]
    assert table.column(1).to_pylist() == [20, 30, 40]


def test_execute_native_rejects_invalid_positional_exchange_sink_instance():
    con = duckdb.connect()
    cursor = con.cursor()
    plan = _make_test_physical_plan(con)
    runner = duckdb.ray_cxx.DistributedPhysicalPlanRunner()

    with pytest.raises(ValueError, match="exchange_sink_instance must be bytes or dict"):
        runner.execute_native(cursor, plan, None, None, None, [object()])


def test_execute_native_rejects_legacy_copy_output_string():
    con = duckdb.connect()
    cursor = con.cursor()
    plan = _make_test_physical_plan(con)
    runner = duckdb.ray_cxx.DistributedPhysicalPlanRunner()

    with pytest.raises(ValueError, match="copy_output_info must be a dict"):
        runner.execute_native(cursor, plan, None, None, "/tmp/out")


@pytest.mark.usefixtures("ray_local")
def test_run_plan_uses_distributed_worker_path(tmp_path):
    pa = pytest.importorskip("pyarrow")
    ray = pytest.importorskip("ray")

    con = duckdb.connect()
    src = tmp_path / "scan_typed_input.parquet"
    con.sql("select 1 as x union all select 2 as x union all select 3 as x").write_parquet(str(src))

    relation = con.sql(f"select * from read_parquet('{src}')")
    plan = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        relation,
        str(uuid.uuid4()),
    ).to_physical_plan(con)

    scan_task_descriptors = dict(plan.scan_task_descriptor_map())
    assert scan_task_descriptors

    runner = duckdb.ray_cxx.DistributedPhysicalPlanRunner()
    with _registered_low_level_plan(plan, con):
        parts = list(iter(runner.run_plan(plan, con)))

        assert len(parts) == 1
        assert isinstance(parts[0], duckdb.ray_cxx.RayResultPartitionRef)
        payload = ray.get(parts[0].object_ref)
    assert isinstance(payload, pa.Table)
    assert payload.to_pylist() == [{"c0": 1}, {"c0": 2}, {"c0": 3}]
    con.close()


@pytest.mark.usefixtures("ray_local")
def test_run_copy_plan_uses_distributed_worker_path(tmp_path, monkeypatch):
    captured = []
    monkeypatch.delenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", raising=False)

    class _DummyRunner:
        def run_write(self, relation):
            captured.append(relation)
            return {"ok": True}

    import duckdb.runners as runners_mod

    monkeypatch.setattr(runners_mod, "set_runner_ray", lambda *_args, **_kwargs: _DummyRunner())

    con = duckdb.connect()
    src = tmp_path / "copy_scan_typed_input.parquet"
    dst = tmp_path / "copy_scan_typed_output.parquet"
    con.sql("select 1 as x union all select 2 as x union all select 3 as x").write_parquet(str(src))

    monkeypatch.setenv("VANE_RUNNER", "ray")
    con.sql(f"select * from read_parquet('{src}')").write_parquet(str(dst))

    assert captured, "expected write relation to be captured"

    plan = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        captured[0],
        str(uuid.uuid4()),
    ).to_physical_plan(con)

    scan_task_descriptors = dict(plan.scan_task_descriptor_map())
    assert scan_task_descriptors

    runner = duckdb.ray_cxx.DistributedPhysicalPlanRunner()
    with _registered_low_level_plan(plan, con):
        result = runner.run_copy_plan(plan, con)

    assert result["rows_copied"] == 3
    assert result["copy_output_base_path"] == str(dst)
    assert result["copy_output_run_id"]
    assert result["copy_output_direct_write"] is True
    assert result["copy_output_committed"] is True
    assert Path(result["copy_output_commit_dir"]).is_dir()
    assert Path(result["copy_output_lifecycle_path"]).is_file()
    assert Path(result["copy_output_manifest_path"]).is_file()
    assert Path(result["copy_output_committed_marker_path"]).is_file()
    assert result["copy_output_manifest_path"].endswith(
        f"{dst.name}.duckdb_commit/{result['copy_output_run_id']}/manifest.txt"
    )
    committed = duckdb.ray_cxx.read_committed_copy_direct_write_result(
        result["copy_output_base_path"],
        result["copy_output_run_id"],
    )
    committed_paths = [entry["final_path"] for entry in committed["files"]]
    assert committed["rows_copied"] == 3
    assert committed_paths
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    assert sorted(row[0] for row in con.read_parquet(committed_paths).fetchall()) == [1, 2, 3]
    assert not Path(str(dst) + ".duckdb_staging").exists()
    con.close()


@pytest.mark.usefixtures("ray_local")
def test_run_copy_plan_local_staging_env_preserves_rename_path(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", "1")

    class _DummyRunner:
        def run_write(self, relation):
            captured.append(relation)
            return {"ok": True}

    import duckdb.runners as runners_mod

    monkeypatch.setattr(runners_mod, "set_runner_ray", lambda *_args, **_kwargs: _DummyRunner())

    con = duckdb.connect()
    src = tmp_path / "copy_staging_input.parquet"
    dst = tmp_path / "copy_staging_output.parquet"
    con.sql("select 10 as x union all select 20 as x").write_parquet(str(src))

    monkeypatch.setenv("VANE_RUNNER", "ray")
    con.sql(f"select * from read_parquet('{src}')").write_parquet(str(dst))

    assert captured, "expected write relation to be captured"
    plan = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        captured[0],
        str(uuid.uuid4()),
    ).to_physical_plan(con)

    runner = duckdb.ray_cxx.DistributedPhysicalPlanRunner()
    with _registered_low_level_plan(plan, con):
        result = runner.run_copy_plan(plan, con)

    assert result["rows_copied"] == 2
    assert result["copy_output_base_path"] == str(dst)
    assert result["copy_output_direct_write"] is False
    assert result["copy_output_committed"] is True
    assert dst.exists()
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    assert sorted(row[0] for row in con.sql(f"select * from read_parquet('{dst}')").fetchall()) == [10, 20]
    assert not Path(str(dst) + ".duckdb_staging").exists()
    con.close()


@pytest.mark.usefixtures("ray_local")
def test_run_copy_plan_with_fte_preserves_copy_sink_output_for_existing_dir(tmp_path, monkeypatch):
    captured = []
    monkeypatch.delenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", raising=False)

    class _DummyRunner:
        def run_write(self, relation):
            captured.append(relation)
            return {"ok": True}

    import duckdb.runners as runners_mod

    monkeypatch.setattr(runners_mod, "set_runner_ray", lambda *_args, **_kwargs: _DummyRunner())

    con = duckdb.connect()
    src = tmp_path / "copy_fte_input.parquet"
    dst = tmp_path / "copy_fte_output"
    dst.mkdir()
    con.sql("select 1 as x union all select 2 as x union all select 3 as x").write_parquet(str(src))

    monkeypatch.setenv("VANE_RUNNER", "ray")
    con.sql(f"select * from read_parquet('{src}')").write_parquet(str(dst))

    assert captured, "expected write relation to be captured"
    plan = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        captured[0],
        str(uuid.uuid4()),
    ).to_physical_plan(con)

    scan_task_descriptors = dict(plan.scan_task_descriptor_map())
    assert scan_task_descriptors

    runner = duckdb.ray_cxx.DistributedPhysicalPlanRunner()
    with _registered_low_level_plan(plan, con):
        result = runner.run_copy_plan(plan, con)

    files = sorted(path for path in dst.rglob("*") if path.is_file())
    assert result["rows_copied"] == 3
    assert files
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    assert sum(con.sql(f"select count(*) from read_parquet('{path}')").fetchone()[0] for path in files) == 3
    assert not Path(str(dst) + ".duckdb_staging").exists()
    con.close()


@pytest.mark.usefixtures("ray_local")
def test_run_copy_plan_local_direct_write_committed_reader(tmp_path, monkeypatch):
    captured = []
    monkeypatch.delenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", raising=False)

    class _DummyRunner:
        def run_write(self, relation):
            captured.append(relation)
            return {"ok": True}

    import duckdb.runners as runners_mod

    monkeypatch.setattr(runners_mod, "set_runner_ray", lambda *_args, **_kwargs: _DummyRunner())

    con = duckdb.connect()
    src = tmp_path / "copy_direct_success_input.parquet"
    dst = tmp_path / "copy_direct_success_output"
    con.sql("select 1 as x union all select 2 as x union all select 3 as x").write_parquet(str(src))

    monkeypatch.setenv("VANE_RUNNER", "ray")
    con.sql(f"select * from read_parquet('{src}')").write_parquet(str(dst))

    assert captured, "expected write relation to be captured"
    plan = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        captured[0],
        str(uuid.uuid4()),
    ).to_physical_plan(con)

    scan_task_descriptors = dict(plan.scan_task_descriptor_map())
    assert scan_task_descriptors

    runner = duckdb.ray_cxx.DistributedPhysicalPlanRunner()
    with _registered_low_level_plan(plan, con):
        result = runner.run_copy_plan(plan, con)

    assert result["rows_copied"] == 3
    assert result["copy_output_base_path"] == str(dst)
    assert result["copy_output_run_id"]
    assert result["copy_output_direct_write"] is True
    assert result["copy_output_committed"] is True
    assert Path(result["copy_output_lifecycle_path"]).is_file()
    assert Path(result["copy_output_manifest_path"]).is_file()
    assert Path(result["copy_output_committed_marker_path"]).is_file()
    assert not Path(str(dst) + ".duckdb_staging").exists()

    committed = duckdb.ray_cxx.read_committed_copy_direct_write_result(
        result["copy_output_base_path"],
        result["copy_output_run_id"],
    )
    committed_paths = [entry["final_path"] for entry in committed["files"]]
    assert committed["rows_copied"] == 3
    assert committed["copy_output_direct_write"] is True
    assert committed_paths
    assert all("_vane_direct_write_" not in path for path in committed_paths)
    assert all(Path(path).parent == dst for path in committed_paths)
    assert all(Path(path).name.startswith(f"{result['copy_output_run_id']}_") for path in committed_paths)
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    assert sum(con.sql(f"select count(*) from read_parquet('{path}')").fetchone()[0] for path in committed_paths) == 3

    loser_file = dst / f"{result['copy_output_run_id']}_w_loser_part.parquet"
    loser_file.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY (SELECT 999 AS x) TO '{loser_file}' (FORMAT PARQUET)")
    all_run_files = [*committed_paths, str(loser_file)]
    assert con.read_parquet(all_run_files).aggregate("count(*)").fetchone()[0] == 4

    from duckdb.runners.ray import read_committed_copy_direct_write_parquet

    committed_rel = read_committed_copy_direct_write_parquet(
        result["copy_output_base_path"],
        result["copy_output_run_id"],
        conn=con,
    )
    assert sorted(row[0] for row in committed_rel.fetchall()) == [1, 2, 3]
    con.close()


def test_run_copy_plan_propagates_worker_task_failure_before_finalize(tmp_path, monkeypatch):
    class _FailingTaskHandle:
        _is_done = True
        _result = None
        _future = None
        task = None
        worker_id = "worker-fail"

        def __init__(self, message, task_id):
            self.task_id = _fake_task_attempt_id(task_id)
            self.task_context_info = _fake_task_context_info(self.task_id)
            self._error = RuntimeError(message)

        def _ensure_started(self):
            return None

        def done(self):
            return True

        def get_result_sync(self):
            raise self._error

        def cancel(self):
            self._is_done = True

        def release_result_payload(self):
            return None

    class _FailingWorkerHandle:
        def __init__(self):
            self.submit_count = 0
            self.staging_roots = []
            self.handles_by_query = {}

        def submit_tasks(self, tasks):
            handles = []
            for task in tasks:
                self.submit_count += 1
                context = task.context()
                query_id = context["query_id"]
                staging_base = context["copy_output_base"]
                run_id = context["copy_output_run_id"]
                assert staging_base
                staging_root = Path(staging_base) / run_id
                output_file = staging_root / "w_fake" / "part.parquet"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_bytes(b"partial-copy-output")
                self.staging_roots.append(staging_root)
                task_id = {
                    "query_id": query_id,
                    "fragment_execution_id": 0,
                    "partition_id": self.submit_count - 1,
                    "attempt_id": 0,
                }
                handle = _FailingTaskHandle("planned worker failure", task_id)
                handles.append(handle)
                self.handles_by_query.setdefault(query_id, []).append(handle)
            return handles

        def fte_query_status(self, query_id):
            return {
                "failed": False,
                "finished": bool(self.handles_by_query.get(query_id)),
                "selected_attempt_task_ids": [],
            }

        def pop_fte_result_handles(self, query_id):
            return list(self.handles_by_query.pop(query_id, []))

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def task_input_stream_exhausted_for_query(self, _query_id, _source_node_ids):
            return []

        def shutdown(self):
            return None

    import duckdb.runners as runners_mod
    import duckdb.runners.ray.worker_handle as ray_worker_handle

    captured = []

    class _CapturingRunner:
        def run_write(self, relation):
            captured.append(relation)
            return {"ok": True}

    monkeypatch.setenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", "1")
    failing_worker = _FailingWorkerHandle()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids: [duckdb.ray_cxx.RayWorkerRuntime("worker-fail", failing_worker, 1.0, 0.0, 1024)],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    con = duckdb.connect()
    src = tmp_path / "copy_failure_input.parquet"
    dst = tmp_path / "copy_failure_output.parquet"
    con.sql("select 1 as x union all select 2 as x").write_parquet(str(src))

    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setattr(runners_mod, "set_runner_ray", lambda *_args, **_kwargs: _CapturingRunner())
    con.sql(f"select * from read_parquet('{src}')").write_parquet(str(dst))
    assert captured, "expected write relation to be captured"

    plan = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        captured[0],
        str(uuid.uuid4()),
    ).to_physical_plan(con)
    scan_task_descriptors = dict(plan.scan_task_descriptor_map())
    assert scan_task_descriptors

    runner = duckdb.ray_cxx.DistributedPhysicalPlanRunner()
    with _registered_low_level_plan(plan, con, node_id="node-a"):
        with pytest.raises(ValueError, match="planned worker failure"):
            runner.run_copy_plan(plan, con)
    assert failing_worker.submit_count >= 1
    assert not dst.exists()
    assert failing_worker.staging_roots
    for staging_root in failing_worker.staging_roots:
        assert not staging_root.exists()
    assert not Path(str(dst) + ".duckdb_staging").exists()


def test_run_copy_plan_direct_write_failure_cleans_uncommitted_run(tmp_path, monkeypatch):
    class _FailingTaskHandle:
        _is_done = True
        _result = None
        _future = None
        task = None
        worker_id = "worker-direct-fail"

        def __init__(self, message, task_id):
            self.task_id = _fake_task_attempt_id(task_id)
            self.task_context_info = _fake_task_context_info(self.task_id)
            self._error = RuntimeError(message)

        def _ensure_started(self):
            return None

        def done(self):
            return True

        def get_result_sync(self):
            raise self._error

        def cancel(self):
            self._is_done = True

        def release_result_payload(self):
            return None

    class _FailingDirectWriteWorkerHandle:
        def __init__(self):
            self.submit_count = 0
            self.output_files = []
            self.handles_by_query = {}

        def submit_tasks(self, tasks):
            handles = []
            for task in tasks:
                self.submit_count += 1
                context = task.context()
                query_id = context["query_id"]
                assert context["copy_output_base"] == ""
                run_id = context["copy_output_run_id"]
                remote_base = context["copy_output_remote_base"]
                output_file = Path(remote_base) / f"{run_id}_w_fake_part.parquet"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_bytes(b"partial-direct-copy-output")
                self.output_files.append(output_file)
                task_id = {
                    "query_id": query_id,
                    "fragment_execution_id": 0,
                    "partition_id": self.submit_count - 1,
                    "attempt_id": 0,
                }
                handle = _FailingTaskHandle("planned direct worker failure", task_id)
                handles.append(handle)
                self.handles_by_query.setdefault(query_id, []).append(handle)
            return handles

        def fte_query_status(self, query_id):
            return {
                "failed": False,
                "finished": bool(self.handles_by_query.get(query_id)),
                "selected_attempt_task_ids": [],
            }

        def pop_fte_result_handles(self, query_id):
            return list(self.handles_by_query.pop(query_id, []))

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def task_input_stream_exhausted_for_query(self, _query_id, _source_node_ids):
            return []

        def shutdown(self):
            return None

    import duckdb.runners as runners_mod
    import duckdb.runners.ray.worker_handle as ray_worker_handle

    captured = []

    class _CapturingRunner:
        def run_write(self, relation):
            captured.append(relation)
            return {"ok": True}

    monkeypatch.delenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", raising=False)
    failing_worker = _FailingDirectWriteWorkerHandle()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids: [duckdb.ray_cxx.RayWorkerRuntime("worker-direct-fail", failing_worker, 1.0, 0.0, 1024)],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    con = duckdb.connect()
    src = tmp_path / "copy_direct_failure_input.parquet"
    dst = tmp_path / "copy_direct_failure_output.parquet"
    con.sql("select 1 as x union all select 2 as x").write_parquet(str(src))

    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setattr(runners_mod, "set_runner_ray", lambda *_args, **_kwargs: _CapturingRunner())
    con.sql(f"select * from read_parquet('{src}')").write_parquet(str(dst))
    assert captured, "expected write relation to be captured"

    plan = duckdb.ray_cxx.PyLogicalPlan.from_duckdb_relation(
        captured[0],
        str(uuid.uuid4()),
    ).to_physical_plan(con)
    scan_task_descriptors = dict(plan.scan_task_descriptor_map())
    assert scan_task_descriptors

    runner = duckdb.ray_cxx.DistributedPhysicalPlanRunner()
    with _registered_low_level_plan(plan, con, node_id="node-a"):
        with pytest.raises(ValueError, match="planned direct worker failure"):
            runner.run_copy_plan(plan, con)
    assert failing_worker.submit_count >= 1
    assert failing_worker.output_files
    for output_file in failing_worker.output_files:
        assert not output_file.exists()
    assert not Path(str(dst) + ".duckdb_commit").exists()


def test_wait_fte_query_propagates_status_errors(monkeypatch):
    class _StatusFailingWorkerHandle:
        def __init__(self):
            self.status_calls = 0

        def fte_query_status(self, _query_id):
            self.status_calls += 1
            raise RuntimeError("status exploded")

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def shutdown(self):
            return None

    import duckdb.runners.ray.worker_handle as ray_worker_handle

    failing_worker = _StatusFailingWorkerHandle()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids: [duckdb.ray_cxx.RayWorkerRuntime("worker-status-fail", failing_worker, 1.0, 0.0, 1024)],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    manager = duckdb.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        with pytest.raises(Exception, match="status exploded"):
            manager.wait_fte_query("query-status-error", 0.01)
        assert failing_worker.status_calls == 1
    finally:
        manager.shutdown()


def test_wait_fte_query_releases_gil_while_waiting(monkeypatch):
    class _ThreadProgressWorkerHandle:
        def __init__(self):
            self.finished = False
            self.status_calls = 0
            self.status_polled = threading.Event()
            self.finished_event = threading.Event()

        def fte_query_status(self, _query_id):
            self.status_calls += 1
            self.status_polled.set()
            return {
                "failed": False,
                "finished": self.finished,
                "selected_attempt_task_ids": [],
            }

        def pop_fte_result_handles(self, _query_id):
            return []

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def shutdown(self):
            return None

    import duckdb.runners.ray.worker_handle as ray_worker_handle

    worker = _ThreadProgressWorkerHandle()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids: [duckdb.ray_cxx.RayWorkerRuntime("worker-gil-wait", worker, 1.0, 0.0, 1024)],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    def finish_after_first_status_poll():
        assert worker.status_polled.wait(1.0)
        time.sleep(0.02)
        worker.finished = True
        worker.finished_event.set()

    thread = threading.Thread(target=finish_after_first_status_poll, daemon=True)
    thread.start()

    manager = duckdb.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        manager.wait_fte_query("query-gil-wait", 1.0)
        assert worker.finished_event.is_set()
        assert worker.status_calls >= 2
    finally:
        manager.shutdown()
        thread.join(1.0)


def test_wait_fte_query_rejects_malformed_query_status(monkeypatch):
    class _MalformedStatusWorkerHandle:
        def __init__(self):
            self.status_calls = 0

        def fte_query_status(self, _query_id):
            self.status_calls += 1
            return {"finished": True}

        def pop_fte_result_handles(self, _query_id):
            return []

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def shutdown(self):
            return None

    import duckdb.runners.ray.worker_handle as ray_worker_handle

    worker = _MalformedStatusWorkerHandle()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids: [duckdb.ray_cxx.RayWorkerRuntime("worker-status-malformed", worker, 1.0, 0.0, 1024)],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    manager = duckdb.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        with pytest.raises(Exception, match="FTE query status must include boolean 'failed'"):
            manager.wait_fte_query("query-status-malformed", 1.0)
        assert worker.status_calls == 1
    finally:
        manager.shutdown()


def test_wait_fte_query_rejects_result_handles_without_task_id(monkeypatch):
    class _MalformedHandle:
        worker_id = "worker-handle-malformed"
        task_context_info = _fake_task_context_info(
            {
                "query_id": "query-handle-malformed",
                "fragment_execution_id": 0,
                "partition_id": 0,
                "attempt_id": 0,
            }
        )

        def done(self):
            return True

        def get_result_sync(self):
            return duckdb.ray_cxx.RayTaskResult.no_output()

    class _MalformedHandleWorker:
        def __init__(self):
            self.pop_calls = 0

        def fte_query_status(self, _query_id):
            return {
                "failed": False,
                "finished": True,
                "selected_attempt_task_ids": [],
            }

        def pop_fte_result_handles(self, _query_id):
            self.pop_calls += 1
            return [_MalformedHandle()]

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def shutdown(self):
            return None

    import duckdb.runners.ray.worker_handle as ray_worker_handle

    worker = _MalformedHandleWorker()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids: [duckdb.ray_cxx.RayWorkerRuntime("worker-handle-malformed", worker, 1.0, 0.0, 1024)],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    manager = duckdb.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        with pytest.raises(Exception, match="FTE result handle must provide task_id"):
            manager.wait_fte_query("query-handle-malformed", 1.0)
        assert worker.pop_calls == 1
    finally:
        manager.shutdown()


def test_wait_fte_query_rejects_result_handles_without_worker_id(monkeypatch):
    class _MalformedHandle:
        def __init__(self):
            self.task_id = _fake_task_attempt_id(
                {
                    "query_id": "query-handle-missing-worker",
                    "fragment_execution_id": 0,
                    "partition_id": 0,
                    "attempt_id": 0,
                }
            )
            self.task_context_info = _fake_task_context_info(self.task_id)

        def done(self):
            return True

        def get_result_sync(self):
            return duckdb.ray_cxx.RayTaskResult.no_output()

    class _MalformedHandleWorker:
        def __init__(self):
            self.pop_calls = 0

        def fte_query_status(self, query_id):
            return {
                "failed": False,
                "finished": True,
                "selected_attempt_task_ids": [f"{query_id}.0.0.0"],
            }

        def pop_fte_result_handles(self, _query_id):
            self.pop_calls += 1
            return [_MalformedHandle()]

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def shutdown(self):
            return None

    import duckdb.runners.ray.worker_handle as ray_worker_handle

    worker = _MalformedHandleWorker()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids: [duckdb.ray_cxx.RayWorkerRuntime("worker-coordinator", worker, 1.0, 0.0, 1024)],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    manager = duckdb.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        with pytest.raises(Exception, match="worker_id"):
            manager.wait_fte_query("query-handle-missing-worker", 1.0)
        assert worker.pop_calls == 1
    finally:
        manager.shutdown()


def test_wait_fte_query_propagates_selected_attempt_handle_errors(monkeypatch):
    class _FailedSelectedAttemptHandle:
        worker_id = "worker-selected"

        def __init__(self, task_id):
            self.task_id = _fake_task_attempt_id(task_id)
            self.task_context_info = _fake_task_context_info(self.task_id)
            self._is_done = True
            self._result = None
            self._error = RuntimeError("selected attempt failed")
            self._future = None
            self.task = None
            self.release_calls = 0

        def _ensure_started(self):
            return None

        def done(self):
            return True

        def get_result_sync(self):
            raise RuntimeError("selected attempt failed")

        def release_result_payload(self):
            self.release_calls += 1

    class _StatusSupportedWorkerHandle:
        def __init__(self):
            self.status_calls = 0
            self.pop_calls = 0
            self.handle = None

        def fte_query_status(self, query_id):
            self.status_calls += 1
            return {
                "failed": False,
                "finished": self.status_calls >= 2,
                "selected_attempt_task_ids": [f"{query_id}.0.0.0"],
            }

        def pop_fte_result_handles(self, query_id):
            self.pop_calls += 1
            if self.pop_calls != 1:
                return []
            task_id = {
                "query_id": query_id,
                "fragment_execution_id": 0,
                "partition_id": 0,
                "attempt_id": 0,
            }
            self.handle = _FailedSelectedAttemptHandle(task_id)
            return [self.handle]

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def shutdown(self):
            return None

    import duckdb.runners.ray.worker_handle as ray_worker_handle

    worker = _StatusSupportedWorkerHandle()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids: [duckdb.ray_cxx.RayWorkerRuntime("worker-selected", worker, 1.0, 0.0, 1024)],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    manager = duckdb.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        with pytest.raises(Exception, match="selected attempt failed"):
            manager.wait_fte_query("query-selected-error", 1.0)
        assert worker.pop_calls >= 1
        assert worker.status_calls >= 2
        assert worker.handle is not None
        assert worker.handle.release_calls == 0
        manager.drop_query_fragments("query-selected-error")
        assert worker.handle.release_calls == 1
    finally:
        manager.shutdown()


def test_wait_fte_query_ignores_retry_loser_attempt_errors(monkeypatch):
    class _FailedAttemptHandle:
        worker_id = "worker-retry"

        def __init__(self, task_id):
            self.task_id = _fake_task_attempt_id(task_id)
            self.task_context_info = _fake_task_context_info(self.task_id)
            self._is_done = True
            self._result = None
            self._error = RuntimeError("loser attempt failed")
            self._future = None
            self.task = None
            self.release_calls = 0

        def _ensure_started(self):
            return None

        def done(self):
            return True

        def get_result_sync(self):
            raise RuntimeError("loser attempt failed")

        def release_result_payload(self):
            self.release_calls += 1

    class _NoOutputAttemptHandle:
        worker_id = "worker-retry"

        def __init__(self, task_id):
            self.task_id = _fake_task_attempt_id(task_id)
            self.task_context_info = _fake_task_context_info(self.task_id)
            self._is_done = True
            self._result = duckdb.ray_cxx.RayTaskResult.no_output()
            self._error = None
            self._future = None
            self.task = None

        def _ensure_started(self):
            return None

        def done(self):
            return True

        def get_result_sync(self):
            return duckdb.ray_cxx.RayTaskResult.no_output()

        def release_result_payload(self):
            return None

    class _StatusSupportedWorkerHandle:
        def __init__(self):
            self.status_calls = 0
            self.pop_calls = 0
            self.loser_handle = None

        def fte_query_status(self, query_id):
            self.status_calls += 1
            return {
                "failed": False,
                "finished": self.status_calls >= 2,
                "selected_attempt_task_ids": [f"{query_id}.0.0.1"],
            }

        def pop_fte_result_handles(self, query_id):
            self.pop_calls += 1
            if self.pop_calls != 1:
                return []
            failed_task_id = {
                "query_id": query_id,
                "fragment_execution_id": 0,
                "partition_id": 0,
                "attempt_id": 0,
            }
            retry_task_id = {
                "query_id": query_id,
                "fragment_execution_id": 0,
                "partition_id": 0,
                "attempt_id": 1,
            }
            self.loser_handle = _FailedAttemptHandle(failed_task_id)
            return [self.loser_handle, _NoOutputAttemptHandle(retry_task_id)]

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def shutdown(self):
            return None

    import duckdb.runners.ray.worker_handle as ray_worker_handle

    worker = _StatusSupportedWorkerHandle()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids: [duckdb.ray_cxx.RayWorkerRuntime("worker-retry", worker, 1.0, 0.0, 1024)],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    manager = duckdb.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        manager.wait_fte_query("query-retry-loser", 1.0)
        assert worker.pop_calls >= 1
        assert worker.status_calls >= 2
        assert worker.loser_handle is not None
        assert worker.loser_handle.release_calls == 1
    finally:
        manager.shutdown()


def test_wait_fte_query_release_failure_preserves_failed_handle_and_releases_rest(
    monkeypatch,
):
    class _NoOutputHandle:
        worker_id = "worker-release-failure"

        def __init__(self, task_id, *, fail_release_once=False):
            self.task_id = _fake_task_attempt_id(task_id)
            self.task_context_info = _fake_task_context_info(self.task_id)
            self._result = duckdb.ray_cxx.RayTaskResult.no_output()
            self.fail_release_once = fail_release_once
            self.release_calls = 0

        def done(self):
            return True

        def get_result_sync(self):
            return self._result

        def release_result_payload(self):
            self.release_calls += 1
            if self.fail_release_once:
                self.fail_release_once = False
                raise RuntimeError("planned result payload release failure")

    class _Worker:
        def __init__(self):
            self.pop_calls = 0
            self.handles = []

        def fte_query_status(self, query_id):
            return {
                "failed": False,
                "finished": True,
                "selected_attempt_task_ids": [
                    f"{query_id}.0.0.0",
                    f"{query_id}.0.1.0",
                ],
            }

        def pop_fte_result_handles(self, query_id):
            self.pop_calls += 1
            if self.pop_calls != 1:
                return []
            self.handles = [
                _NoOutputHandle(
                    {
                        "query_id": query_id,
                        "fragment_execution_id": 0,
                        "partition_id": 0,
                        "attempt_id": 0,
                    },
                    fail_release_once=True,
                ),
                _NoOutputHandle(
                    {
                        "query_id": query_id,
                        "fragment_execution_id": 0,
                        "partition_id": 1,
                        "attempt_id": 0,
                    }
                ),
            ]
            return list(self.handles)

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def shutdown(self):
            return None

    import duckdb.runners.ray.worker_handle as ray_worker_handle

    worker = _Worker()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids: [
            duckdb.ray_cxx.RayWorkerRuntime(
                "worker-release-failure",
                worker,
                1.0,
                0.0,
                1024,
            )
        ],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    manager = duckdb.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        with pytest.raises(Exception, match="planned result payload release failure"):
            manager.wait_fte_query("query-release-failure", 1.0)

        assert worker.handles[0].release_calls == 1
        assert worker.handles[1].release_calls == 1
        manager.drop_query_fragments("query-release-failure")
        assert worker.handles[0].release_calls == 2
        assert worker.handles[1].release_calls == 1
    finally:
        manager.shutdown()


def test_wait_fte_query_does_not_drain_pending_retry_loser_attempt(monkeypatch):
    class _PendingAttemptHandle:
        worker_id = "worker-retry-pending"

        def __init__(self, task_id):
            self.task_id = _fake_task_attempt_id(task_id)
            self.task_context_info = _fake_task_context_info(self.task_id)
            self._is_done = False
            self._result = None
            self._error = None
            self._future = None
            self.task = None
            self.done_calls = 0
            self.get_result_sync_calls = 0

        def _ensure_started(self):
            return None

        def done(self):
            self.done_calls += 1
            return False

        def get_result_sync(self):
            self.get_result_sync_calls += 1
            raise AssertionError("pending loser attempt should not be drained")

        def release_result_payload(self):
            return None

    class _NoOutputAttemptHandle:
        worker_id = "worker-retry-pending"

        def __init__(self, task_id):
            self.task_id = _fake_task_attempt_id(task_id)
            self.task_context_info = _fake_task_context_info(self.task_id)
            self._is_done = True
            self._result = duckdb.ray_cxx.RayTaskResult.no_output()
            self._error = None
            self._future = None
            self.task = None

        def _ensure_started(self):
            return None

        def done(self):
            return True

        def get_result_sync(self):
            return duckdb.ray_cxx.RayTaskResult.no_output()

        def release_result_payload(self):
            return None

    class _StatusSupportedWorkerHandle:
        def __init__(self):
            self.status_calls = 0
            self.pop_calls = 0
            self.pending_handle = None

        def fte_query_status(self, query_id):
            self.status_calls += 1
            return {
                "failed": False,
                "finished": self.status_calls >= 2,
                "selected_attempt_task_ids": [f"{query_id}.0.0.1"],
            }

        def pop_fte_result_handles(self, query_id):
            self.pop_calls += 1
            if self.pop_calls != 1:
                return []
            loser_task_id = {
                "query_id": query_id,
                "fragment_execution_id": 0,
                "partition_id": 0,
                "attempt_id": 0,
            }
            selected_task_id = {
                "query_id": query_id,
                "fragment_execution_id": 0,
                "partition_id": 0,
                "attempt_id": 1,
            }
            self.pending_handle = _PendingAttemptHandle(loser_task_id)
            return [self.pending_handle, _NoOutputAttemptHandle(selected_task_id)]

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def shutdown(self):
            return None

    import duckdb.runners.ray.worker_handle as ray_worker_handle

    worker = _StatusSupportedWorkerHandle()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids: [duckdb.ray_cxx.RayWorkerRuntime("worker-retry-pending", worker, 1.0, 0.0, 1024)],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    manager = duckdb.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        manager.wait_fte_query("query-retry-pending", 0.1)
        assert worker.pending_handle is not None
        assert worker.pending_handle.get_result_sync_calls == 0
    finally:
        manager.shutdown()


def test_wait_fte_query_clears_cached_handles_after_failed_status(monkeypatch):
    class _PendingAttemptHandle:
        worker_id = "worker-stale-failed"

        def __init__(self, task_id):
            self.task_id = _fake_task_attempt_id(task_id)
            self.task_context_info = _fake_task_context_info(self.task_id)
            self._is_done = False
            self._result = None
            self._error = None
            self._future = None
            self.task = None
            self.done_calls = 0

        def _ensure_started(self):
            return None

        def done(self):
            self.done_calls += 1
            return False

        def get_result_sync(self):
            raise AssertionError("stale cached handle should have been cleared")

        def release_result_payload(self):
            return None

    class _StatusFailsAfterCollectWorkerHandle:
        def __init__(self):
            self.status_calls = 0
            self.pop_calls = 0
            self.pending_handle = None

        def fte_query_status(self, query_id):
            self.status_calls += 1
            if self.status_calls == 1:
                return {
                    "failed": False,
                    "finished": False,
                    "selected_attempt_task_ids": [],
                }
            if self.status_calls == 2:
                return {
                    "failed": True,
                    "finished": False,
                    "selected_attempt_task_ids": [],
                }
            return {
                "failed": False,
                "finished": True,
                "selected_attempt_task_ids": [f"{query_id}.0.0.0"],
            }

        def pop_fte_result_handles(self, query_id):
            self.pop_calls += 1
            if self.pop_calls != 1:
                return []
            task_id = {
                "query_id": query_id,
                "fragment_execution_id": 0,
                "partition_id": 0,
                "attempt_id": 0,
            }
            self.pending_handle = _PendingAttemptHandle(task_id)
            return [self.pending_handle]

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def shutdown(self):
            return None

    import duckdb.runners.ray.worker_handle as ray_worker_handle

    worker = _StatusFailsAfterCollectWorkerHandle()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids: [duckdb.ray_cxx.RayWorkerRuntime("worker-stale-failed", worker, 1.0, 0.0, 1024)],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    manager = duckdb.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        with pytest.raises(Exception, match="FTE query failed"):
            manager.wait_fte_query("query-stale-failed", 0.1)
        assert worker.pending_handle is not None
        done_calls_after_failure = worker.pending_handle.done_calls

        manager.wait_fte_query("query-stale-failed", 0.001)

        assert worker.pending_handle.done_calls == done_calls_after_failure
    finally:
        manager.shutdown()


def test_wait_fte_query_timeout_preserves_collected_handles(monkeypatch):
    class _ReadyAfterQueryTimeoutHandle:
        worker_id = "worker-timeout-preserve"
        _result = None
        _error = None
        _future = None
        _is_done = False
        task = None

        def __init__(self, task_id, worker):
            self.task_id = _fake_task_attempt_id(task_id)
            self.task_context_info = _fake_task_context_info(self.task_id)
            self.worker = worker
            self.get_result_sync_calls = 0

        def _ensure_started(self):
            return None

        def done(self):
            return self.worker.finish_query

        def get_result_sync(self):
            self.get_result_sync_calls += 1
            return duckdb.ray_cxx.RayTaskResult.no_output()

        def release_result_payload(self):
            return None

    class _TimeoutThenFinishedWorkerHandle:
        def __init__(self):
            self.finish_query = False
            self.pop_calls = 0
            self.handle = None

        def fte_query_status(self, query_id):
            return {
                "failed": False,
                "finished": self.finish_query,
                "selected_attempt_task_ids": [f"{query_id}.0.0.0"] if self.finish_query else [],
            }

        def pop_fte_result_handles(self, query_id):
            self.pop_calls += 1
            if self.pop_calls != 1:
                return []
            task_id = {
                "query_id": query_id,
                "fragment_execution_id": 0,
                "partition_id": 0,
                "attempt_id": 0,
            }
            self.handle = _ReadyAfterQueryTimeoutHandle(task_id, self)
            return [self.handle]

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def shutdown(self):
            return None

    import duckdb.runners.ray.worker_handle as ray_worker_handle

    worker = _TimeoutThenFinishedWorkerHandle()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids: [duckdb.ray_cxx.RayWorkerRuntime("worker-timeout-preserve", worker, 1.0, 0.0, 1024)],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    manager = duckdb.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        with pytest.raises(Exception, match="timed out waiting for FTE query"):
            manager.wait_fte_query("query-timeout-preserve", 0.001)
        assert worker.handle is not None
        assert worker.handle.get_result_sync_calls == 0

        worker.finish_query = True
        manager.wait_fte_query("query-timeout-preserve", 1.0)

        assert worker.handle.get_result_sync_calls == 1
        assert worker.pop_calls >= 2
    finally:
        manager.shutdown()


def test_wait_fte_query_respects_timeout_after_finished_status_during_drain(monkeypatch):
    class _EventuallyReadyAttemptHandle:
        worker_id = "worker-drain-timeout"
        _result = None
        _error = None
        _future = None
        _is_done = False
        task = None

        def __init__(self, task_id):
            self.task_id = _fake_task_attempt_id(task_id)
            self.task_context_info = _fake_task_context_info(self.task_id)
            self.ready_at = time.monotonic() + 0.2
            self.get_result_sync_calls = 0

        def _ensure_started(self):
            return None

        def done(self):
            return time.monotonic() >= self.ready_at

        def get_result_sync(self):
            self.get_result_sync_calls += 1
            return duckdb.ray_cxx.RayTaskResult.no_output()

        def release_result_payload(self):
            return None

    class _SlowResultWorkerHandle:
        def __init__(self):
            self.pop_calls = 0
            self.handle = None

        def fte_query_status(self, query_id):
            return {
                "failed": False,
                "finished": True,
                "selected_attempt_task_ids": [f"{query_id}.0.0.0"],
            }

        def pop_fte_result_handles(self, query_id):
            self.pop_calls += 1
            if self.pop_calls != 1:
                return []
            time.sleep(0.02)
            task_id = {
                "query_id": query_id,
                "fragment_execution_id": 0,
                "partition_id": 0,
                "attempt_id": 0,
            }
            self.handle = _EventuallyReadyAttemptHandle(task_id)
            return [self.handle]

        def stats_fragments(self):
            return {"registered_total": 0, "existing_total": 0, "lookup_hits": 0}

        def fte_drop_query(self, _query_id):
            return {"tasks_removed": 0, "tasks_canceled": 0, "fragments_removed": 0}

        def shutdown(self):
            return None

    import duckdb.runners.ray.worker_handle as ray_worker_handle

    worker = _SlowResultWorkerHandle()
    monkeypatch.setattr(
        ray_worker_handle,
        "start_ray_workers",
        lambda _existing_ids: [duckdb.ray_cxx.RayWorkerRuntime("worker-drain-timeout", worker, 1.0, 0.0, 1024)],
    )
    monkeypatch.setattr(ray_worker_handle, "try_autoscale", lambda _bundles: None)

    manager = duckdb.ray_cxx.RayWorkerManager()
    try:
        manager.worker_snapshots()
        with pytest.raises(Exception, match="timed out draining FTE result handles"):
            manager.wait_fte_query("query-drain-timeout", 0.001)
        assert worker.handle is not None
        assert worker.handle.get_result_sync_calls == 0

        time.sleep(0.25)
        manager.wait_fte_query("query-drain-timeout", 1.0)

        assert worker.handle.get_result_sync_calls == 1
        assert worker.pop_calls >= 1
    finally:
        manager.shutdown()


def test_ray_query_driver_client_retries_stale_named_actor_with_get_if_exists(monkeypatch):
    class _FakeMethod:
        def __init__(self, label):
            self.label = label

        def remote(self, *args, **kwargs):
            return (self.label, args, kwargs)

    class _FakeHandle:
        def __init__(self, name):
            self.name = name
            self.ping = _FakeMethod(("ping", name))
            self.install_env_overrides = _FakeMethod(("install_env_overrides", name))

    handles = [_FakeHandle("stale"), _FakeHandle("fresh")]
    option_calls = []
    kill_calls = []

    def _fake_options(**kwargs):
        option_calls.append(kwargs)

        class _Factory:
            def remote(self, env_overrides, duckdb_memory_bytes):
                assert env_overrides == {}
                assert duckdb_memory_bytes == 50
                return handles.pop(0)

        return _Factory()

    def _fake_ray_get(token, **_kwargs):
        if token[0] == ("ping", "stale"):
            raise RuntimeError("stale actor")

    def _fake_ray_kill(handle, no_restart=False):
        kill_calls.append((handle.name, no_restart))

    monkeypatch.setattr(driver, "_maybe_set_distributed_cluster_capacity", lambda: None)
    monkeypatch.setattr(
        driver,
        "get_head_node",
        lambda: {"NodeID": "a" * 56, "Resources": {"memory": 1_000}},
    )
    monkeypatch.setattr(driver, "_collect_vane_env_overrides", dict)
    monkeypatch.setattr(driver.RayQueryDriverActor, "options", _fake_options)
    monkeypatch.setattr(driver, "resolve_object_refs_blocking", _fake_ray_get)
    monkeypatch.setattr(driver.ray, "kill", _fake_ray_kill)

    runner = driver.RayQueryDriverClient()

    assert runner.runner.name == "fresh"
    assert len(option_calls) == 2
    assert option_calls[0]["name"] == "ray-query-driver-actor"
    assert option_calls[0]["namespace"] == "vane"
    assert option_calls[0]["memory"] == 50
    assert option_calls[0]["get_if_exists"] is True
    assert option_calls[1]["get_if_exists"] is False
    assert kill_calls == [("stale", False)]
